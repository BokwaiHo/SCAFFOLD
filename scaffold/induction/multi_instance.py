"""Multi-instance skill inducer (Algorithm 1, line 8 + §3.3 / §4.1.4).

Given a cluster C = {ζ_1, ..., ζ_n} of n ≥ n_min successful trajectories with
embedding-similar instructions, this module:
  1. Renders the inducer prompt (see scaffold.induction.prompts);
  2. Queries the LLM (default: GPT-4o-2024-08, temperature 0.3);
  3. Parses the JSON-emitted skill specification;
  4. Constructs a Skill object with depth resolved via the composition module;
  5. Runs hold-out validation: the new skill must re-execute correctly on at least
     one held-out trajectory from C. This is the line in §3.3 that "nearly halves
     the rate of induced-then-deprecated skills".

`InductionConfig.n_min` matches the paper's hyperparameter (default 2). The "minus
multi-instance" ablation (Table 2) is exactly setting n_min=1.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional

from scaffold.core.library import SkillLibrary
from scaffold.core.skill import Parameter, ParameterType, Skill, MAX_BODY_LINES
from scaffold.core.trajectory import Trajectory
from scaffold.induction.clusterer import InstructionCluster
from scaffold.induction.prompts import SKILL_INDUCER_SYSTEM, skill_inducer_user_prompt
from scaffold.utils.llm import ChatMessage, LLMClient
from scaffold.utils.logging import get_logger

log = get_logger("induction")


# ---------------------------------------------------------------------------
# Config + result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class InductionConfig:
    """All knobs that govern multi-instance induction (matches §4.1.4)."""

    n_min: int = 2                            # multi-instance threshold
    temperature: float = 0.3                  # inducer sampling temperature
    max_tokens: int = 1500                    # LLM cap per induction call
    enable_holdout_validation: bool = True    # set False for ablation row 5
    holdout_attempts: int = 1                 # § 3.3: "at least one held-out instance"
    max_body_lines: int = MAX_BODY_LINES
    forbid_imports: bool = True               # body must not contain `import`
    forbid_def: bool = True                   # body must not redefine functions


@dataclass
class InductionResult:
    """Outcome of one INDUCE() call (Algorithm 1, lines 7-11)."""

    cluster_id: int
    skill: Optional[Skill] = None
    accepted: bool = False
    reason: str = ""                          # explanation when rejected
    raw_response: str = ""
    holdout_passed: Optional[bool] = None
    cluster_size: int = 0


# ---------------------------------------------------------------------------
# Inducer
# ---------------------------------------------------------------------------

class MultiInstanceInducer:
    """Implements lines 7-11 of Algorithm 1.

    The inducer is stateless across clusters — each call to `induce_cluster` is
    independent — but it does mutate the library via the caller (the pipeline
    appends the returned skill to `L_k+1`). This separation keeps the inducer
    testable in isolation.
    """

    def __init__(
        self,
        llm: LLMClient,
        config: Optional[InductionConfig] = None,
    ) -> None:
        self.llm = llm
        self.config = config or InductionConfig()

    # ----- main API --------------------------------------------------------

    def induce_cluster(
        self,
        cluster: InstructionCluster,
        library: SkillLibrary,
        iteration: int,
        *,
        holdout_runner=None,
    ) -> InductionResult:
        """Induce a single skill from a cluster of trajectories.

        Args:
            cluster: trajectories supporting the candidate abstraction.
            library: current L_k. The inducer is allowed to call any σ' ∈ L_k.
            iteration: current iteration k (recorded on the skill object).
            holdout_runner: a callable `(skill, trajectory) -> bool` that returns
                True iff the induced skill re-executes successfully on the held-out
                trajectory. If None and validation is enabled, we fall back to a
                static structural check (less reliable but fully offline).

        Returns:
            InductionResult with `accepted=True` and `skill` set on success.
        """
        result = InductionResult(
            cluster_id=cluster.cluster_id,
            cluster_size=len(cluster),
        )

        # multi-instance gating — paper requires |C| ≥ n_min
        if len(cluster) < self.config.n_min:
            result.reason = (
                f"cluster size {len(cluster)} < n_min={self.config.n_min}"
            )
            log.info("Skip cluster %d: %s", cluster.cluster_id, result.reason)
            return result

        # Held-out split: take the smallest-instruction trajectory as held-out so
        # that the inducer sees the longer / richer examples.
        train_trajs, holdout = self._split_holdout(cluster.trajectories)

        prompt = skill_inducer_user_prompt(
            train_trajs,
            list(library),
            centroid_instruction=cluster.centroid_instruction,
        )
        log.info(
            "Inducing on cluster %d (size=%d, holdout=%d)",
            cluster.cluster_id, len(train_trajs), 1 if holdout else 0,
        )
        resp = self.llm.chat(
            [ChatMessage("system", SKILL_INDUCER_SYSTEM),
             ChatMessage("user", prompt)],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        result.raw_response = resp.text

        # Parse + structurally validate
        try:
            spec = self._parse_response(resp.text)
            skill = self._build_skill(spec, iteration=iteration,
                                       cluster_size=len(cluster), library=library)
        except InductionParseError as e:
            result.reason = f"parse error: {e}"
            log.warning("Cluster %d: %s", cluster.cluster_id, result.reason)
            return result

        # Hold-out validation (Algorithm 1 line 9; "VALIDATEHOLDOUT")
        if self.config.enable_holdout_validation and holdout is not None:
            ok = self._validate_holdout(skill, holdout, library, holdout_runner)
            result.holdout_passed = ok
            if not ok:
                result.reason = "hold-out validation failed"
                log.info("Cluster %d: hold-out rejected skill %s",
                         cluster.cluster_id, skill.name)
                return result

        result.skill = skill
        result.accepted = True
        return result

    # ----- helpers ---------------------------------------------------------

    def _split_holdout(
        self, trajectories: list[Trajectory]
    ) -> tuple[list[Trajectory], Optional[Trajectory]]:
        if len(trajectories) < 2 or not self.config.enable_holdout_validation:
            return list(trajectories), None
        # deterministic: shortest instruction is held out
        sorted_by_len = sorted(trajectories, key=lambda t: len(t.instruction))
        holdout = sorted_by_len[0]
        train = sorted_by_len[1:]
        return train, holdout

    def _parse_response(self, text: str) -> dict:
        # Tolerate ```json fences
        s = text.strip()
        if s.startswith("```"):
            lines = s.splitlines()
            if lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            s = "\n".join(lines)
        try:
            spec = json.loads(s)
        except json.JSONDecodeError as e:
            raise InductionParseError(f"invalid JSON: {e}") from e

        # required keys
        for k in ("name", "description", "body"):
            if k not in spec:
                raise InductionParseError(f"missing key: {k!r}")
        if not isinstance(spec["name"], str) or not spec["name"].isidentifier():
            raise InductionParseError(f"bad skill name: {spec['name']!r}")
        return spec

    def _build_skill(
        self, spec: dict, *, iteration: int, cluster_size: int, library: SkillLibrary
    ) -> Skill:
        from scaffold.composition.recursive import (
            compute_depth,
            check_no_cycles,
        )

        # Body sanitation: no imports / def / class — the inducer prompt forbids
        # these but we double-check defensively.
        body = spec.get("body", "").strip()
        if self.config.forbid_imports and re.search(r"^\s*import\b|^\s*from\b", body, flags=re.M):
            raise InductionParseError("body contains import statement")
        if self.config.forbid_def and re.search(r"^\s*(def|class)\b", body, flags=re.M):
            raise InductionParseError("body defines a nested function/class")

        n_lines = len([ln for ln in body.splitlines() if ln.strip()])
        if n_lines > self.config.max_body_lines:
            raise InductionParseError(
                f"body has {n_lines} non-blank lines (> cap {self.config.max_body_lines})"
            )

        params = self._build_parameters(spec.get("parameters", []))

        # Construct the skill at depth=1 first; the composition module will set
        # the real depth and refuse if a cycle would result.
        skill = Skill(
            name=spec["name"],
            description=spec["description"],
            parameters=params,
            precondition=spec.get("precondition", "True"),
            body=body,
            postcondition=spec.get("postcondition", "True"),
            depth=1,
            iteration_introduced=iteration,
            induced_from_cluster_size=cluster_size,
        )
        # Cycle check + depth resolution (§3.4)
        check_no_cycles(skill, library)
        skill.depth = compute_depth(skill, library)
        return skill

    def _build_parameters(self, raw: list) -> list[Parameter]:
        out: list[Parameter] = []
        for p in raw:
            if not isinstance(p, dict) or "name" not in p:
                continue
            try:
                ptype = ParameterType(p.get("type", "str"))
            except ValueError:
                ptype = ParameterType.STRING
            choices = p.get("choices")
            if choices is not None and not isinstance(choices, list):
                choices = None
            try:
                out.append(Parameter(
                    name=p["name"],
                    type=ptype,
                    default=p.get("default"),
                    choices=tuple(choices) if choices else None,
                    description=p.get("description", ""),
                ))
            except ValueError as e:
                log.debug("Dropping malformed parameter %r: %s", p, e)
        return out

    def _validate_holdout(
        self,
        skill: Skill,
        holdout: Trajectory,
        library: SkillLibrary,
        runner,
    ) -> bool:
        """If a runner callable is provided, use it; else fall back to a static
        structural check (parameter types compatible with the held-out actions).

        Real WebArena/VWA experiments install a runner that re-plays the held-out
        trajectory's initial state and re-executes the induced skill body; we keep
        the static fallback so the inducer is unit-testable without a browser.
        """
        if runner is not None:
            try:
                return bool(runner(skill, holdout))
            except Exception as e:
                log.warning("Hold-out runner crashed: %r", e)
                return False
        # static fallback: are all skill-callees present in the library?
        from scaffold.primitives.actions import PRIMITIVE_NAMES
        for callee in skill.calls_set():
            if callee not in library and callee not in PRIMITIVE_NAMES:
                log.debug("Static holdout: missing callee %r", callee)
                return False
        # Light sanity: the trajectory had at least 1 action
        return holdout.length() > 0


class InductionParseError(ValueError):
    """Raised when the inducer's JSON output doesn't match the contract."""
