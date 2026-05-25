"""Agent Workflow Memory (Wang et al., 2025c) baseline.

Defining difference (per §2):
  - "Induces natural-language workflows from past trajectories and retrieves
    them at test time".  Workflows are stored as desc-only "skills" with empty
    bodies (i.e. the agent reads the workflow text but the framework does NOT
    execute it as a program).

We model this by storing skills whose `body` is "pass" and whose `description`
encodes the workflow steps in natural language. The policy is responsible for
following the workflow; the executor short-circuits "pass"-body skills to a
no-op that simply attributes any subsequent actions to the skill (for plan-σ
construction during distillation).

In practice this means AWM benefits from retrieval but not from compositionality.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from scaffold.baselines.base import BaselineBase
from scaffold.core.library import SkillLibrary
from scaffold.core.skill import Parameter, ParameterType, Skill
from scaffold.core.trajectory import Trajectory
from scaffold.induction import cluster_by_instruction
from scaffold.utils.llm import ChatMessage, LLMClient


_AWM_SYSTEM = (
    "You are a helpful workflow author. Given several successful web-agent "
    "trajectories that achieve similar goals, write a short numbered workflow "
    "in plain English that a future agent could follow to accomplish the same "
    "kind of task. Reply with STRICT JSON only:\n"
    '{"name": "snake_case", "description": "one-line description", '
    '"steps": ["1. ...", "2. ...", "3. ..."]}'
)


class AWMBaseline(BaselineBase):
    name = "AWM"

    def __init__(self, *, inducer_llm: Optional[LLMClient] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._llm = inducer_llm or self.llm

    def induce_skills(
        self, positive_traj: list[Trajectory], library: SkillLibrary, iteration: int
    ) -> list[Skill]:
        clusters = cluster_by_instruction(positive_traj, min_cluster_size=2)
        out: list[Skill] = []
        for c in clusters:
            instr_lines = "\n".join(
                f"- {t.instruction} ({len(t.actions)} steps)"
                for t in c.trajectories
            )
            resp = self._llm.chat(
                [ChatMessage("system", _AWM_SYSTEM),
                 ChatMessage("user", f"Trajectories:\n{instr_lines}")],
                temperature=0.3, max_tokens=512,
            )
            try:
                spec = self._parse(resp.text)
            except Exception:
                continue
            name = self._safe_name(spec["name"], existing=library.names())
            workflow_text = "\n".join(spec.get("steps", [])) or spec.get("description", "")
            out.append(Skill(
                name=name,
                description=spec.get("description", "") + " " + workflow_text,
                parameters=[],
                precondition="True",
                body="pass",                # AWM workflows are NL only
                postcondition="True",
                depth=1,
                iteration_introduced=iteration,
            ))
        return out

    @staticmethod
    def _parse(text: str) -> dict:
        s = text.strip()
        if s.startswith("```"):
            s = "\n".join(s.splitlines()[1:-1])
        return json.loads(s)

    @staticmethod
    def _safe_name(s: str, *, existing: list[str]) -> str:
        s = re.sub(r"[^A-Za-z0-9_]", "_", s).strip("_") or "workflow"
        if not s[0].isalpha():
            s = "wf_" + s
        base, i = s, 1
        while s in existing:
            s = f"{base}_{i}"
            i += 1
        return s
