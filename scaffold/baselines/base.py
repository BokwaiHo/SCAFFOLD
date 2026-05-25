"""Shared scaffolding for all baselines.

Each baseline inherits from `BaselineBase` and overrides one or more of the four
methods that the SCAFFOLD pipeline exposes:

  - `induce_skills(Z+_k, L_k) -> list[Skill]`     Stage 2+3 replacement
  - `compact_library(L_k, Z+_{0:k}) -> L_{k+1}`   Stage 4 replacement
  - `distill(pi_k, L_k+1, Z+_k) -> pi_k+1`        Stage 5 replacement

`BaselineBase.run` mirrors `ScaffoldPipeline.run` but consults these hooks
instead of the SCAFFOLD-specific algorithms. This makes every baseline a
single-page subclass.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

from scaffold.agent.policy import Policy
from scaffold.core.library import SkillLibrary
from scaffold.core.skill import Skill
from scaffold.core.trajectory import Trajectory
from scaffold.distillation import DistillationConfig, StubDistiller
from scaffold.envs.base import Env, RolloutConfig
from scaffold.pipeline import ScaffoldPipeline, ScaffoldConfig
from scaffold.utils.embedding import HashEmbedder
from scaffold.utils.llm import LLMClient, MockChatClient
from scaffold.utils.logging import get_logger

log = get_logger("baselines")


@dataclass
class BaselineConfig:
    """Common knobs shared by baselines. Per-baseline specializations add fields."""

    iterations: int = 5
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    output_dir: str = "./runs/baseline"
    use_stub_distiller: bool = True


class BaselineBase:
    """Inherit + override hooks; `run()` orchestrates."""

    name: str = "BaselineBase"

    def __init__(
        self,
        *,
        env: Env,
        policy: Policy,
        llm: Optional[LLMClient] = None,
        config: Optional[BaselineConfig] = None,
    ) -> None:
        self.env = env
        self.policy = policy
        self.llm = llm or MockChatClient()
        self.config = config or BaselineConfig()
        os.makedirs(self.config.output_dir, exist_ok=True)
        # Reuse the SCAFFOLD pipeline's rollout / executor machinery — only the
        # induction / compaction / distillation hooks differ.
        # We construct a ScaffoldPipeline configured to skip induction (handled by
        # this baseline's `induce_skills` hook) and reuse its `_run_one` helper.
        self._scaffold = ScaffoldPipeline(
            env=env,
            policy=policy,
            inducer_llm=MockChatClient(),  # induction is replaced by this baseline
            config=ScaffoldConfig(
                iterations=self.config.iterations,
                rollout=self.config.rollout,
                distillation=self.config.distillation,
                use_stub_distiller=self.config.use_stub_distiller,
                output_dir=self.config.output_dir,
            ),
            embedder=HashEmbedder(),
        )
        self.library = self._scaffold.library
        self._all_positive: list[Trajectory] = []

    # ---- Hook overrides (defaults: do nothing) ---------------------------

    def induce_skills(
        self, positive_traj: list[Trajectory], library: SkillLibrary, iteration: int
    ) -> list[Skill]:
        return []

    def compact_library(
        self, library: SkillLibrary, all_positive: list[Trajectory], iteration: int
    ) -> SkillLibrary:
        return library

    def distill(
        self, library: SkillLibrary, positive_traj: list[Trajectory], iteration: int
    ) -> str:
        ds = self._scaffold.distiller
        from scaffold.distillation import build_distillation_dataset
        return ds.train(build_distillation_dataset(positive_traj, library), iteration=iteration)

    # ---- Orchestration ----------------------------------------------------

    def run(self) -> tuple[SkillLibrary, list[dict[str, Any]]]:
        metrics: list[dict[str, Any]] = []
        tasks = self.env.tasks(split="train")
        log.info("[%s] starting K=%d", self.name, self.config.iterations)

        for k in range(self.config.iterations):
            Zk = self._scaffold.rollout(tasks, iteration=k)
            Zk_plus = [z for z in Zk if z.is_successful()]
            self._all_positive.extend(Zk_plus)

            new_skills = self.induce_skills(Zk_plus, self.library, k)
            for s in new_skills:
                try:
                    self.library.add(s)
                except Exception as e:
                    log.warning("[%s] could not add %s: %r", self.name, s.name, e)
            self.library = self.compact_library(self.library, self._all_positive, k)
            ckpt = self.distill(self.library, Zk_plus, k)

            metrics.append({
                "iteration": k, "n_rollouts": len(Zk),
                "n_success": len(Zk_plus),
                "success_rate": len(Zk_plus) / max(len(Zk), 1),
                "library_size": self.library.total_skills(),
                "library_mean_depth": self.library.mean_depth(),
                "checkpoint": ckpt,
            })
            log.info("[%s] iter %d: %s", self.name, k, metrics[-1])
        return self.library, metrics
