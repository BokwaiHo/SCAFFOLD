"""Synapse (Zheng et al., 2024b) baseline.

Defining difference (§2): "trajectory-as-exemplar prompting with memory". We
store a small bank of successful trajectories indexed by instruction embedding;
at rollout time the policy is shown the top-k most-similar exemplars instead of
synthesized skills.

This is implemented as zero new skills + a side-channel exemplar bank — the
exemplar bank is conceptually part of the policy's prompt, not the library, so
we just keep it on the baseline instance.
"""

from __future__ import annotations

from typing import Optional

from scaffold.baselines.base import BaselineBase
from scaffold.core.library import SkillLibrary
from scaffold.core.skill import Skill
from scaffold.core.trajectory import Trajectory


class SynapseBaseline(BaselineBase):
    name = "Synapse"

    def __init__(self, *, max_exemplars: int = 64, **kwargs) -> None:
        super().__init__(**kwargs)
        self.max_exemplars = max_exemplars
        self._exemplars: list[Trajectory] = []

    def induce_skills(
        self, positive_traj: list[Trajectory], library: SkillLibrary, iteration: int
    ) -> list[Skill]:
        # Side-channel: extend the exemplar bank, no library changes.
        self._exemplars.extend(positive_traj)
        self._exemplars = self._exemplars[-self.max_exemplars :]
        return []
