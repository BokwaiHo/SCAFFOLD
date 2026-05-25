"""SkillWeaver (Zheng et al., 2025) baseline.

Defining differences from SCAFFOLD (per §1, §2, and Table 2 last row):
  - Flat library: skills are stored in a single pool with no recursive composition
    (depth is effectively bounded by 1).
  - Single-instance induction: a skill can be proposed from a single trajectory
    (no n_min ≥ 2 requirement).
  - No principled compression: skills accumulate; duplicates are not merged.
  - No weight-level distillation: experience lives only in prompts.

Running SkillWeaver in our framework is therefore simply BaselineBase with
`induce_skills` doing single-instance induction.
"""

from __future__ import annotations

from typing import Optional

from scaffold.baselines.base import BaselineBase, BaselineConfig
from scaffold.core.library import SkillLibrary
from scaffold.core.skill import Skill
from scaffold.core.trajectory import Trajectory
from scaffold.induction import (
    InductionConfig, MultiInstanceInducer, cluster_by_instruction,
)
from scaffold.utils.llm import LLMClient


class SkillWeaverBaseline(BaselineBase):
    name = "SkillWeaver"

    def __init__(
        self,
        *,
        inducer_llm: Optional[LLMClient] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        # Single-instance induction: n_min = 1, no hold-out validation.
        self._inducer = MultiInstanceInducer(
            llm=inducer_llm or self.llm,
            config=InductionConfig(n_min=1, enable_holdout_validation=False),
        )

    def induce_skills(
        self, positive_traj: list[Trajectory], library: SkillLibrary, iteration: int
    ) -> list[Skill]:
        clusters = cluster_by_instruction(positive_traj)
        new_skills: list[Skill] = []
        for c in clusters:
            res = self._inducer.induce_cluster(c, library=library, iteration=iteration)
            if res.accepted and res.skill is not None:
                # SkillWeaver: force depth=1 (flat library)
                res.skill.depth = 1
                new_skills.append(res.skill)
        return new_skills

    # compact_library: default no-op (flat library grows monotonically)
    # distill: default no-op (in-context only)
