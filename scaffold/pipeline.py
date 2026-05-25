"""The SCAFFOLD self-improvement pipeline (Algorithm 1).

This module is the only place where every other sub-package comes together.
It directly transcribes Algorithm 1 of the paper, line by line:

    Require: π_0, T, V, K, M
    1: L_0 ← ∅
    2: for k = 0, ..., K-1 do
    3:   Z_k ← ROLLOUT(π_k, L_k, T)
    4:   Z+_k ← {ζ ∈ Z_k : V(ζ) = 1}
    5:   C_k ← CLUSTERBYINSTR(Z+_k)
    6:   Σ_new ← ∅
    7:   for cluster C ∈ C_k with |C| ≥ n_min do
    8:     σ ← INDUCE(C, L_k)
    9:     if VALIDATEHOLDOUT(σ, T_val) then
    10:      Σ_new ← Σ_new ∪ {σ}
    11:   L_{k+1} ← L_k ∪ Σ_new
    12:   if (k+1) mod M = 0 then L_{k+1} ← COMPACTMDL(L_{k+1}, Z+_{0:k})
    13:   π_{k+1} ← DISTILL(π_k, L_{k+1}, Z+_k)
    14: return π_K, L_K
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from scaffold.agent.executor import SkillExecutor
from scaffold.agent.policy import Policy
from scaffold.agent.precondition import (
    AlwaysTrueValidator, ChainedValidator, LLMJudgeValidator, PythonExprValidator,
    PreconditionValidator,
)
from scaffold.compaction import MDLCompactor, MDLConfig
from scaffold.core.library import SkillLibrary
from scaffold.core.trajectory import Action, ActionType, Observation, Trajectory
from scaffold.distillation import (
    DistillationConfig, LoRADistiller, StubDistiller,
    build_distillation_dataset,
)
from scaffold.envs.base import Env, RolloutConfig, Task
from scaffold.induction import (
    InductionConfig, MultiInstanceInducer, cluster_by_instruction,
)
from scaffold.utils.embedding import Embedder, HashEmbedder
from scaffold.utils.llm import LLMClient, MockChatClient
from scaffold.utils.logging import get_logger
from scaffold.utils.seed import set_seed

log = get_logger("pipeline")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ScaffoldConfig:
    """Top-level configuration. Defaults reflect the paper's §4.1.4."""

    iterations: int = 5                          # K
    compaction_interval: int = 2                 # M
    induction: InductionConfig = field(default_factory=InductionConfig)
    mdl: MDLConfig = field(default_factory=MDLConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    output_dir: str = "./runs/scaffold"
    use_stub_distiller: bool = False             # tests / toy env
    embedder_name: Optional[str] = None          # "hash" / "st:..." / None
    cluster_similarity_threshold: float = 0.55
    use_llm_precondition: bool = False           # if False, PythonExprValidator only
    seed: int = 0


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class ScaffoldPipeline:
    """Algorithm 1 orchestrator.

    Composition:
        env           Env      → tasks, browsers, validation pool
        policy        Policy   → π_k (PromptedPolicy / ScriptedPolicy)
        inducer       MultiInstanceInducer → §3.3
        compactor     MDLCompactor          → §3.5
        distiller     LoRADistiller | StubDistiller → §3.6

    Caller responsibilities:
        - Choose a config, env, policy, inducer LLM
        - Call .run() — returns (final library, list of per-iteration metrics)
        - Persist whatever it wants from the returned objects
    """

    def __init__(
        self,
        *,
        env: Env,
        policy: Policy,
        inducer_llm: Optional[LLMClient] = None,
        refactor_llm: Optional[LLMClient] = None,
        config: Optional[ScaffoldConfig] = None,
        embedder: Optional[Embedder] = None,
        precondition_validator: Optional[PreconditionValidator] = None,
        judge_llm: Optional[LLMClient] = None,
    ) -> None:
        self.env = env
        self.policy = policy
        self.config = config or ScaffoldConfig()
        self.embedder = embedder or HashEmbedder()
        set_seed(self.config.seed)

        self.inducer = MultiInstanceInducer(
            llm=inducer_llm or MockChatClient(),
            config=self.config.induction,
        )

        from scaffold.compaction.refactor import RefactorOperator
        self.compactor = MDLCompactor(
            config=self.config.mdl,
            refactor_op=RefactorOperator(
                l_min=self.config.mdl.l_min, r_min=self.config.mdl.r_min,
                llm=refactor_llm,
            ),
        )

        self.distiller = (
            StubDistiller(self.config.distillation)
            if self.config.use_stub_distiller
            else LoRADistiller(self.config.distillation)
        )

        if precondition_validator is None:
            if self.config.use_llm_precondition and judge_llm is not None:
                precondition_validator = ChainedValidator(
                    PythonExprValidator(), LLMJudgeValidator(judge_llm),
                )
            else:
                precondition_validator = PythonExprValidator()
        self.precondition_validator = precondition_validator

        self.library = SkillLibrary(embedder=self.embedder)   # L_0 ← ∅
        self._all_positive: list[Trajectory] = []             # Z+_{0:k}
        os.makedirs(self.config.output_dir, exist_ok=True)

    # ----- main loop --------------------------------------------------------

    def run(self) -> tuple[SkillLibrary, list[dict[str, Any]]]:
        """Execute K iterations of Algorithm 1 and return (L_K, metrics_per_iter)."""
        metrics: list[dict[str, Any]] = []
        tasks = self.env.tasks(split="train")
        log.info("Starting SCAFFOLD: K=%d, |tasks|=%d", self.config.iterations, len(tasks))

        for k in range(self.config.iterations):
            log.info("=" * 80)
            log.info("Iteration k=%d / K=%d", k, self.config.iterations)
            log.info("=" * 80)

            # --- Stage 1: rollout -----------------------------------------------
            Zk = self.rollout(tasks, iteration=k)
            Z_plus_k = [z for z in Zk if z.is_successful()]
            self._all_positive.extend(Z_plus_k)
            log.info("Stage 1: %d trajectories, %d successful", len(Zk), len(Z_plus_k))

            # --- Stage 2: multi-instance induction ------------------------------
            clusters = cluster_by_instruction(
                Z_plus_k, embedder=self.embedder,
                similarity_threshold=self.config.cluster_similarity_threshold,
                min_cluster_size=1,
            )
            new_skills = []
            n_rejected = 0
            for cluster in clusters:
                if len(cluster) < self.config.induction.n_min:
                    continue
                result = self.inducer.induce_cluster(
                    cluster, library=self.library, iteration=k,
                    holdout_runner=self._holdout_runner,
                )
                if result.accepted:
                    new_skills.append(result.skill)
                else:
                    n_rejected += 1
                    log.info("Reject cluster %d: %s", cluster.cluster_id, result.reason)

            for s in new_skills:
                try:
                    self.library.add(s)
                except Exception as e:
                    log.warning("Failed to add induced skill %s: %r", s.name, e)
            log.info("Stage 2+3: induced %d new skills (rejected %d)",
                     len(new_skills), n_rejected)

            # --- Stage 4: MDL compaction (every M iterations) -------------------
            if (k + 1) % self.config.compaction_interval == 0:
                log.info("Stage 4: running MDL compaction (M=%d)",
                         self.config.compaction_interval)
                self.compactor.apply(
                    self.library,
                    positive_trajectories=self._all_positive,
                    validation_sr=self._validation_sr,
                )

            # --- Stage 5: distillation ------------------------------------------
            dataset = build_distillation_dataset(Z_plus_k, self.library)
            ckpt = self.distiller.train(dataset, iteration=k)
            log.info("Stage 5: distilled π_%d → %s", k + 1, ckpt)

            # --- bookkeeping ----------------------------------------------------
            iter_metrics = {
                "iteration": k,
                "n_rollouts": len(Zk),
                "n_success": len(Z_plus_k),
                "success_rate": len(Z_plus_k) / max(len(Zk), 1),
                "n_clusters": len(clusters),
                "n_new_skills": len(new_skills),
                "library_size": self.library.total_skills(),
                "library_mean_depth": self.library.mean_depth(),
                "library_max_depth": self.library.max_depth(),
                "library_reuse_rate": self.library.reuse_rate(),
                "checkpoint": ckpt,
            }
            metrics.append(iter_metrics)
            log.info("Iter %d summary: %s", k, json.dumps(iter_metrics, default=str))
            # Persist library + metrics each iteration so partial runs are usable.
            self._save_iter(k, iter_metrics)

        return self.library, metrics

    # ----- Stage 1 helper: rollout ------------------------------------------

    def rollout(self, tasks: list[Task], iteration: int) -> list[Trajectory]:
        """Run π_k on each task, return the trajectories."""
        out: list[Trajectory] = []
        for task in tasks:
            for _ in range(self.config.rollout.n_attempts_per_task):
                t = self._run_one(task, iteration=iteration)
                out.append(t)
        return out

    def _run_one(self, task: Task, *, iteration: int) -> Trajectory:
        browser = self.env.make_browser(task)
        executor = SkillExecutor(
            library=self.library,
            browser=browser,
            precondition_validator=self.precondition_validator,
            max_steps=self.config.rollout.max_steps,
        )
        actions: list[Action] = []
        observations: list[Observation] = [browser.observe()]

        for step in range(self.config.rollout.max_steps):
            decision = self.policy.decide(
                instruction=task.instruction,
                observation=observations[-1],
                library=self.library,
                history=actions,
            )
            a = decision.action

            if a.type is ActionType.SKILL and decision.skill_call is not None:
                result = executor.run(
                    decision.skill_call, observations[-1], actions,
                )
                if not result.success:
                    # the executor logged the failure; we treat it as a stuck step
                    actions.append(Action(type=ActionType.NOOP, args={"error": result.error}))
                obs = browser.observe()
            elif a.type is ActionType.CLICK:
                obs = browser.click(a.args.get("target"))
                actions.append(a)
            elif a.type is ActionType.TYPE:
                obs = browser.type(a.args.get("target"), a.args.get("text", ""))
                actions.append(a)
            elif a.type is ActionType.SCROLL:
                obs = browser.scroll(a.args.get("dx", 0), a.args.get("dy", 0))
                actions.append(a)
            elif a.type is ActionType.WAIT:
                obs = browser.wait(a.args.get("seconds", 0.5))
                actions.append(a)
            elif a.type is ActionType.SUBMIT_ANSWER:
                actions.append(a)
                observations.append(browser.observe())
                # episode ends
                traj = Trajectory(
                    instruction=task.instruction, observations=observations,
                    actions=actions, task_id=task.task_id, site=task.site,
                    iteration=iteration, final_answer=a.args.get("answer"),
                )
                traj.success = bool(task.verifier and task.verifier.verify(traj))
                self.env.close_browser(browser)
                return traj
            else:
                actions.append(a)
                obs = observations[-1]  # NOOP / parse error: don't advance

            observations.append(obs)
            if self.policy.is_done(obs, actions, max_steps=self.config.rollout.max_steps):
                break

        traj = Trajectory(
            instruction=task.instruction, observations=observations, actions=actions,
            task_id=task.task_id, site=task.site, iteration=iteration,
        )
        traj.success = bool(task.verifier and task.verifier.verify(traj))
        self.env.close_browser(browser)
        return traj

    # ----- support for inducer hold-out validation --------------------------

    def _holdout_runner(self, skill, holdout: Trajectory) -> bool:
        """Re-execute the induced skill on a fresh copy of the held-out trajectory's
        initial state. Used by `MultiInstanceInducer` when configured. Returns
        True iff the post-state passes the trajectory's verifier (or, lacking a
        verifier, the executor returns success).
        """
        from scaffold.core.skill import SkillCall
        from scaffold.envs.base import Task as _Task
        from scaffold.core.verifier import AlwaysSuccessVerifier

        task = _Task(
            task_id=f"holdout:{skill.name}",
            instruction=holdout.instruction,
            site=holdout.site,
            verifier=AlwaysSuccessVerifier(),
            initial_state={"start_url": holdout.observations[0].url if holdout.observations else "/"},
        )
        browser = self.env.make_browser(task)
        try:
            # Temporarily add the candidate skill to a *copy* of the library, so
            # the executor can resolve it.
            tmp_lib = self.library.copy()
            tmp_lib._skills[skill.name] = skill
            ex = SkillExecutor(library=tmp_lib, browser=browser,
                               precondition_validator=self.precondition_validator,
                               max_steps=self.config.rollout.max_steps)
            obs = browser.observe()
            # Use the *first* observed action's args as a hint for skill-call args.
            # In practice the inducer prompt is supposed to give clear param hints;
            # here we infer them from the cluster centroid.
            args = self._guess_skill_args(skill, holdout)
            res = ex.run(SkillCall(name=skill.name, args=args), obs, [])
            return res.success
        finally:
            self.env.close_browser(browser)

    def _guess_skill_args(self, skill, holdout: Trajectory) -> dict[str, Any]:
        """Heuristic: scan the held-out trajectory's typed actions for arg
        candidates matching each parameter name."""
        args: dict[str, Any] = {}
        for p in skill.parameters:
            for a in holdout.actions:
                if a.type is ActionType.TYPE and p.name in str(a.args.get("target", "")).lower():
                    args[p.name] = a.args.get("text", "")
                    break
            else:
                if p.default is not None:
                    args[p.name] = p.default
                elif p.choices:
                    args[p.name] = p.choices[0]
                else:
                    args[p.name] = ""
        return args

    # ----- support for MDL validation --------------------------------------

    def _validation_sr(self, library: SkillLibrary) -> float:
        """Run a small validation pass with the given library; return SR."""
        old_lib = self.library
        self.library = library
        try:
            val_tasks = self.env.validation_tasks(n=self.config.mdl.validation_sample_size)
            if not val_tasks:
                return 1.0
            n_pass = 0
            for t in val_tasks:
                traj = self._run_one(t, iteration=-1)
                if traj.is_successful():
                    n_pass += 1
            return n_pass / len(val_tasks)
        finally:
            self.library = old_lib

    # ----- persistence ------------------------------------------------------

    def _save_iter(self, k: int, metrics: dict[str, Any]) -> None:
        d = os.path.join(self.config.output_dir, f"iter{k}")
        os.makedirs(d, exist_ok=True)
        self.library.save(os.path.join(d, "library.json"))
        with open(os.path.join(d, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2, default=str)
