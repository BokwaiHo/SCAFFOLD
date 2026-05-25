"""Policy abstraction π_k.

A policy receives an observation and the currently available skill library, and
emits either a primitive action or a skill invocation. The paper's base policy is
Qwen2.5-VL-7B (or UI-TARS-7B); we abstract that behind `PromptedPolicy` so the
pipeline can be exercised with a `ScriptedPolicy` in tests.

The retrieval step (which skills to show the policy this turn) is keyed on the
*observation* — concretely on (instruction, last_dom_summary). We use the library's
embedding index for the top-k filtering so the prompt stays compact.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from scaffold.core.library import SkillLibrary
from scaffold.core.skill import Skill, SkillCall
from scaffold.core.trajectory import Action, ActionType, Observation


@dataclass
class PolicyDecision:
    """What the policy chose to do at step t."""

    action: Action
    rationale: str = ""
    # if the action is a skill invocation, the resolved SkillCall + Skill object
    skill_call: Optional[SkillCall] = None
    skill_obj: Optional[Skill] = None


class Policy(ABC):
    @abstractmethod
    def decide(
        self,
        instruction: str,
        observation: Observation,
        library: SkillLibrary,
        history: list[Action],
    ) -> PolicyDecision: ...

    # The pipeline asks the policy whether the task is done after each action.
    # Default is a fixed step budget; subclasses may add task-specific exits.
    def is_done(
        self,
        observation: Observation,
        history: list[Action],
        max_steps: int = 30,
    ) -> bool:
        if len(history) >= max_steps:
            return True
        # if the policy issued a SUBMIT_ANSWER, we treat the episode as done
        return bool(history) and history[-1].type is ActionType.SUBMIT_ANSWER


# ---------- Concrete: ScriptedPolicy --------------------------------------------

class ScriptedPolicy(Policy):
    """A deterministic policy driven by a pre-recorded action sequence.

    Useful for: unit tests, building demonstration trajectories, and the toy env
    that ships with the repo. NOT a baseline — it has no learning capacity.
    """

    def __init__(self, script: list[Action]) -> None:
        self._script = list(script)
        self._cursor = 0

    def decide(
        self,
        instruction: str,
        observation: Observation,
        library: SkillLibrary,
        history: list[Action],
    ) -> PolicyDecision:
        if self._cursor >= len(self._script):
            return PolicyDecision(action=Action(type=ActionType.NOOP))
        a = self._script[self._cursor]
        self._cursor += 1
        skill_obj = library[a.skill_call] if a.skill_call and a.skill_call in library else None
        return PolicyDecision(action=a, skill_obj=skill_obj)


# ---------- Concrete: PromptedPolicy --------------------------------------------

class PromptedPolicy(Policy):
    """LLM-prompted policy: composes a prompt from (instruction, obs, library_excerpt)
    and asks the underlying LLM for the next action in a tightly constrained JSON
    schema. This is the production setup matching §4.1.4.
    """

    JSON_SCHEMA_HINT = (
        '{"type": "<click|type|scroll|wait|skill|submit>", '
        '"args": {<action-specific>}, '
        '"skill_call": "<skill_name_or_null>", '
        '"rationale": "<short reason>"}'
    )

    def __init__(
        self,
        llm,
        *,
        temperature: float = 0.7,           # §4.1.4: temperature 0.7 for rollouts
        top_k_skills: int = 8,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.llm = llm
        self.temperature = temperature
        self.top_k_skills = top_k_skills
        self.system_prompt = system_prompt or self._default_system_prompt()

    def _default_system_prompt(self) -> str:
        return (
            "You are a web agent. At each step, you receive an instruction, the "
            "current observation (URL + DOM snippet), the action history, and a "
            "list of available SKILLS you may invoke as a single action. Choose ONE "
            "action. Prefer invoking a skill over emitting primitives directly when "
            "a skill matches the goal. Reply with EXACTLY one JSON object matching:\n"
            f"{self.JSON_SCHEMA_HINT}"
        )

    def decide(
        self,
        instruction: str,
        observation: Observation,
        library: SkillLibrary,
        history: list[Action],
    ) -> PolicyDecision:
        from scaffold.utils.llm import ChatMessage

        skills = library.retrieve(instruction + " " + observation.url, k=self.top_k_skills)
        skills_excerpt = "\n".join(self._format_skill(s) for s in skills) or "(none)"
        history_excerpt = self._format_history(history[-5:])

        user = (
            f"Instruction: {instruction}\n"
            f"Observation: {observation.summary()}\n"
            f"History (last 5):\n{history_excerpt}\n"
            f"Available skills (top-{self.top_k_skills}):\n{skills_excerpt}\n"
            f"Reply with one JSON object."
        )
        resp = self.llm.chat(
            [ChatMessage("system", self.system_prompt), ChatMessage("user", user)],
            temperature=self.temperature,
            max_tokens=512,
        )
        action, skill_call = self._parse(resp.text, library)
        skill_obj = library[skill_call.name] if skill_call and skill_call.name in library else None
        return PolicyDecision(
            action=action,
            rationale="",
            skill_call=skill_call,
            skill_obj=skill_obj,
        )

    @staticmethod
    def _format_skill(s: Skill) -> str:
        params = ", ".join(p.signature_repr() for p in s.parameters)
        return f"- {s.name}({params})  # d={s.depth}: {s.description}"

    @staticmethod
    def _format_history(hist: list[Action]) -> str:
        return "\n".join(f"  {i}: {a.type.value} {a.args} (skill={a.skill_call})"
                         for i, a in enumerate(hist))

    @staticmethod
    def _parse(
        text: str, library: SkillLibrary
    ) -> tuple[Action, Optional[SkillCall]]:
        import json
        # tolerate code fences
        s = text.strip()
        if s.startswith("```"):
            s = "\n".join(s.splitlines()[1:-1])
        try:
            d = json.loads(s)
        except json.JSONDecodeError:
            # fall back to noop: keeps the rollout alive, marks the step as broken
            return Action(type=ActionType.NOOP, args={"raw": text[:100]}), None

        t = d.get("type", "noop")
        args = d.get("args", {}) or {}
        skill_call_name = d.get("skill_call")

        if t == "skill" and skill_call_name and skill_call_name in library:
            sk = library[skill_call_name]
            return (
                Action(
                    type=ActionType.SKILL,
                    args=args,
                    skill_call=skill_call_name,
                    skill_depth=sk.depth,
                ),
                SkillCall(name=skill_call_name, args=args),
            )

        try:
            atype = ActionType(t)
        except ValueError:
            atype = ActionType.NOOP
        return Action(type=atype, args=args), None
