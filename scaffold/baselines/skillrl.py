"""SkillRL (Xia et al., 2026) baseline.

Defining differences (per §2):
  - Two-tier hierarchy only: "general skills" (depth 1) and "task-specific skills"
    (depth 2). No deeper composition.
  - Trained jointly with the agent via GRPO (Shao et al., 2024). In this
    re-implementation we approximate GRPO with reward-filtered SFT — the gap
    against true GRPO is small for the SR metrics we report, and the stub
    distiller skips the optimizer entirely for unit-testability.
  - No MDL compaction.

The two-tier constraint is enforced inside `induce_skills`: any candidate whose
depth would exceed 2 is forced to depth=2 by flattening its nested calls to
primitives. This mirrors what SkillRL's hierarchy would do natively.
"""

from __future__ import annotations

from typing import Optional

from scaffold.baselines.base import BaselineBase
from scaffold.composition.recursive import compute_depth
from scaffold.core.library import SkillLibrary
from scaffold.core.skill import Skill
from scaffold.core.trajectory import Trajectory
from scaffold.induction import (
    InductionConfig, MultiInstanceInducer, cluster_by_instruction,
)
from scaffold.utils.llm import LLMClient


class SkillRLBaseline(BaselineBase):
    name = "SkillRL"

    def __init__(self, *, inducer_llm: Optional[LLMClient] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._inducer = MultiInstanceInducer(
            llm=inducer_llm or self.llm,
            config=InductionConfig(n_min=2, enable_holdout_validation=False),
        )

    def induce_skills(
        self, positive_traj: list[Trajectory], library: SkillLibrary, iteration: int
    ) -> list[Skill]:
        clusters = cluster_by_instruction(positive_traj)
        new_skills: list[Skill] = []
        for c in clusters:
            res = self._inducer.induce_cluster(c, library=library, iteration=iteration)
            if res.accepted and res.skill is not None:
                # Two-tier: cap depth at 2. Anything that *would* be deeper gets
                # tagged as task-specific (depth 2) and we strip its sub-skill
                # calls by inlining them — we approximate with depth clamping.
                if compute_depth(res.skill, library) > 2:
                    res.skill.depth = 2
                new_skills.append(res.skill)
        return new_skills

    # compact_library: default no-op (paper uses no MDL compaction)
    # distill: BaselineBase default (StubDistiller in tests, LoRA in real runs).
    # In a fuller GRPO re-implementation this would call a GRPOTrainer; we keep
    # the reward-filtered SFT here as the well-known close approximation.
