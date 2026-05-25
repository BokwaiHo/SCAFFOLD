"""Trajectory data structure: ζ = (o_0, a_0, o_1, ..., o_T) per §3.1.

A trajectory records the sequence of observations and actions for a single task
attempt, plus a binary success flag set by the verifier. Each action carries an
optional `skill_call` field referencing the skill invocation (if any) that emitted
it, which is what enables Stage 5 distillation to recover the `plan_σ` for each
trajectory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ActionType(str, Enum):
    CLICK = "click"
    TYPE = "type"
    SCROLL = "scroll"
    WAIT = "wait"
    SKILL = "skill"          # invocation of a skill from L_k
    SUBMIT_ANSWER = "submit"  # task-completion answer (for QA-style tasks)
    NOOP = "noop"


@dataclass
class Observation:
    """o_t = (s_t, d_t, u_t) from §3.1, plus a step index for convenience."""

    step: int
    screenshot: Optional[bytes] = None      # s_t: raw PNG bytes
    dom: str = ""                           # d_t: accessibility tree or DOM snippet
    url: str = ""                           # u_t
    extra: dict[str, Any] = field(default_factory=dict)  # site-specific fields

    def summary(self, max_chars: int = 400) -> str:
        """Compact rendering for prompts and logs (never serializes the screenshot)."""
        d = self.dom.replace("\n", " ")
        if len(d) > max_chars:
            d = d[:max_chars] + "..."
        return f"[step {self.step}] url={self.url} dom='{d}'"


@dataclass
class Action:
    """A single action a_t emitted by the policy.

    `skill_call` is set whenever the action originated from a skill invocation; this
    is required for distillation (§3.6) to recover the per-trajectory skill plan
    `plan_σ`. For primitive actions emitted directly by the base policy it is None.
    """

    type: ActionType
    args: dict[str, Any] = field(default_factory=dict)
    skill_call: Optional[str] = None        # name of the skill, if invoked from one
    skill_depth: int = 0                    # d_σ of the calling skill, 0 if primitive

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "args": self.args,
            "skill_call": self.skill_call,
            "skill_depth": self.skill_depth,
        }


@dataclass
class Trajectory:
    """ζ = (o_0, a_0, o_1, ..., o_T) plus an instruction I and binary success."""

    instruction: str
    observations: list[Observation] = field(default_factory=list)
    actions: list[Action] = field(default_factory=list)
    success: Optional[bool] = None
    task_id: Optional[str] = None
    site: Optional[str] = None
    iteration: int = 0
    final_answer: Optional[str] = None

    # ---------- validation ----------

    def __post_init__(self) -> None:
        # T actions, T+1 observations after a completed rollout
        n_obs, n_act = len(self.observations), len(self.actions)
        if n_obs > 0 and n_obs != n_act + 1:
            raise ValueError(
                f"Trajectory length mismatch: {n_obs} obs vs {n_act} actions "
                f"(expected obs = actions + 1)"
            )

    # ---------- query ----------

    def length(self) -> int:
        return len(self.actions)

    def is_successful(self) -> bool:
        return bool(self.success)

    def primitive_subsequence(self) -> list[str]:
        """Flat sequence of action *types* (or skill name when invoked).

        Used by the Refactor operator (§3.5) to detect recurring patterns across
        skills.
        """
        out: list[str] = []
        for a in self.actions:
            if a.type is ActionType.SKILL and a.skill_call:
                out.append(a.skill_call)
            else:
                out.append(a.type.value)
        return out

    def skill_plan(self) -> list[str]:
        """plan_σ from §3.6: ordered list of distinct skill invocations in the trajectory.

        Primitive-only actions are omitted; for distillation we want the
        compositional structure, not the flat token stream.
        """
        plan: list[str] = []
        for a in self.actions:
            if a.type is ActionType.SKILL and a.skill_call:
                plan.append(a.skill_call)
        return plan

    # ---------- serialisation ----------

    def to_dict(self) -> dict[str, Any]:
        return {
            "instruction": self.instruction,
            "task_id": self.task_id,
            "site": self.site,
            "success": self.success,
            "iteration": self.iteration,
            "final_answer": self.final_answer,
            "observations": [
                {"step": o.step, "url": o.url, "dom": o.dom, "extra": o.extra}
                for o in self.observations
            ],
            "actions": [a.to_dict() for a in self.actions],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Trajectory":
        obs = [
            Observation(step=o["step"], url=o["url"], dom=o.get("dom", ""),
                        extra=o.get("extra", {}))
            for o in d.get("observations", [])
        ]
        acts = [
            Action(
                type=ActionType(a["type"]),
                args=a.get("args", {}),
                skill_call=a.get("skill_call"),
                skill_depth=a.get("skill_depth", 0),
            )
            for a in d.get("actions", [])
        ]
        return cls(
            instruction=d["instruction"],
            observations=obs,
            actions=acts,
            success=d.get("success"),
            task_id=d.get("task_id"),
            site=d.get("site"),
            iteration=d.get("iteration", 0),
            final_answer=d.get("final_answer"),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
