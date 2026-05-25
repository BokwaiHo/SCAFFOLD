"""Reflexion (Shinn et al., 2023) baseline.

Defining difference (§2): "verbal in-context self-correction without weight
updates". We model this by storing a per-task reflection string in the skill
library (as a desc-only zero-arg "skill" whose body is `pass`) that the
retrieval index surfaces on similar tasks at the next iteration.

This keeps the implementation as a one-page subclass while preserving the
fundamental property of Reflexion that distinguishes it: it produces verbal
hints, not executable skills.
"""

from __future__ import annotations

from typing import Optional

from scaffold.baselines.base import BaselineBase
from scaffold.core.library import SkillLibrary
from scaffold.core.skill import Skill
from scaffold.core.trajectory import Trajectory
from scaffold.utils.llm import ChatMessage, LLMClient


class ReflexionBaseline(BaselineBase):
    name = "Reflexion"

    def __init__(self, *, reflection_llm: Optional[LLMClient] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._llm = reflection_llm or self.llm
        self._n_reflections = 0

    def induce_skills(
        self, positive_traj: list[Trajectory], library: SkillLibrary, iteration: int
    ) -> list[Skill]:
        # Reflect on the BAD trajectories from this iteration, not the good ones.
        # We don't have a Z- list directly; we approximate as "any task whose
        # plan was empty" — these are the ones where the agent struggled most.
        empty_plan_tasks = [t for t in positive_traj if not t.skill_plan()]
        out: list[Skill] = []
        for t in empty_plan_tasks[:5]:
            resp = self._llm.chat([
                ChatMessage("system", "Reflect on the trajectory and write ONE short hint "
                                       "(<=20 words) the agent should remember for similar tasks."),
                ChatMessage("user", f"Instruction: {t.instruction}\n"
                                     f"Final URL: {t.observations[-1].url if t.observations else '?'}\n"
                                     f"Hint:"),
            ], temperature=0.3, max_tokens=40)
            self._n_reflections += 1
            out.append(Skill(
                name=f"reflection_{iteration}_{self._n_reflections}",
                description=f"Hint for tasks like '{t.instruction[:60]}': {resp.text.strip()}",
                parameters=[],
                precondition="True", body="pass", postcondition="True",
                depth=1, iteration_introduced=iteration,
            ))
        return out
