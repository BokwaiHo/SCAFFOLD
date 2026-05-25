"""SkillExecutor.

When a policy chooses a SKILL action, the executor expands it: it sets up a
PrimitiveContext, dynamically `exec`s the skill body, lets the body call primitives
(which append to the trajectory action list), and validates the pre/postcondition.

Why exec-from-string? Induced skills are produced by an LLM in textual Python; we
want to call them in the same process the policy lives in, without round-tripping
through file I/O. The execution namespace is hand-curated (no builtins like `open`)
and the context-var pattern ensures primitives can find the current browser.

Cycle safety: depth is computed at induction time; here we just refuse to descend
into a skill we're already inside (defense-in-depth).
"""

from __future__ import annotations

import contextlib
import contextvars
from dataclasses import dataclass, field
from typing import Any, Optional

from scaffold.agent.precondition import PreconditionValidator, AlwaysTrueValidator
from scaffold.core.library import SkillLibrary
from scaffold.core.skill import Skill, SkillCall
from scaffold.core.trajectory import Action, Observation
from scaffold.primitives.actions import (
    Browser,
    PrimitiveContext,
    StepBudgetExceeded,
    find,
    find_input,
    primitive_click,
    primitive_scroll,
    primitive_type,
    primitive_wait,
)
from scaffold.utils.logging import get_logger

log = get_logger("executor")

# Context-var: each `run` call sets this so primitives can find the active context.
_CURRENT_CONTEXT: contextvars.ContextVar[Optional[PrimitiveContext]] = contextvars.ContextVar(
    "_scaffold_primitive_context", default=None
)


def current_context() -> Optional[PrimitiveContext]:
    return _CURRENT_CONTEXT.get()


@dataclass
class ExecutionResult:
    """Outcome of executing a skill body once."""

    skill_name: str
    success: bool                                # postcondition + no exceptions
    actions: list[Action] = field(default_factory=list)
    error: Optional[str] = None
    pre_ok: bool = True
    post_ok: bool = True


class SkillExecutor:
    """Runs skill bodies. One instance per env attempt is the typical pattern.

    The executor keeps a *call stack* of skill names so that:
      - cycles are caught even if the static `calls_set` analysis missed something;
      - the `Action.skill_depth` we record is always the depth of the *outermost*
        active skill, matching what we want for distillation supervision.
    """

    def __init__(
        self,
        library: SkillLibrary,
        browser: Browser,
        *,
        precondition_validator: Optional[PreconditionValidator] = None,
        max_steps: int = 30,
    ) -> None:
        self.library = library
        self.browser = browser
        self.pre = precondition_validator or AlwaysTrueValidator()
        self.max_steps = max_steps
        self._stack: list[str] = []          # active skill names, top of stack last

    def run(
        self,
        call: SkillCall,
        observation: Observation,
        actions: list[Action],
    ) -> ExecutionResult:
        skill = self.library[call.name]

        if skill.name in self._stack:
            return ExecutionResult(
                skill_name=skill.name, success=False,
                error=f"cycle detected: {self._stack} -> {skill.name}",
            )

        # Precondition check (§3.3 / App. A): refuse to execute if the world doesn't
        # match what the skill expects.
        pre_ok = self.pre.check(skill.precondition, observation)
        if not pre_ok:
            log.info("Skill %s: precondition failed; skipping", skill.name)
            return ExecutionResult(
                skill_name=skill.name, success=False, pre_ok=False,
                error="precondition failed",
            )

        ctx = PrimitiveContext(
            browser=self.browser,
            actions=actions,
            skill_call_name=skill.name,
            skill_depth=skill.depth,
            max_steps=self.max_steps - len(actions),
        )
        ns = self._build_namespace(skill, call.args, observation)

        self._stack.append(skill.name)
        token = _CURRENT_CONTEXT.set(ctx)
        try:
            exec(skill._body_as_function_source(), ns)
            fn = ns[skill.name]
            fn(observation, **call.args)
        except StepBudgetExceeded as e:
            log.warning("Step budget exceeded inside %s: %s", skill.name, e)
            return ExecutionResult(
                skill_name=skill.name, success=False,
                actions=ctx.actions, error=str(e),
            )
        except Exception as e:
            log.warning("Skill %s raised: %r", skill.name, e)
            return ExecutionResult(
                skill_name=skill.name, success=False,
                actions=ctx.actions, error=repr(e),
            )
        finally:
            _CURRENT_CONTEXT.reset(token)
            self._stack.pop()

        # Postcondition check — fetch the latest observation from the browser.
        latest_obs = self.browser.observe()
        post_ok = self.pre.check(skill.postcondition, latest_obs)
        if not post_ok:
            log.info("Skill %s: postcondition failed", skill.name)

        # Mark usage for §3.5 prune accounting and §4.5 reuse-rate metric.
        self.library.increment_usage([skill.name])

        return ExecutionResult(
            skill_name=skill.name,
            success=post_ok,
            actions=ctx.actions,
            post_ok=post_ok,
        )

    def _build_namespace(
        self, skill: Skill, args: dict[str, Any], observation: Observation
    ) -> dict[str, Any]:
        ns: dict[str, Any] = {
            "obs": observation,
            "primitive_click": primitive_click,
            "primitive_type": primitive_type,
            "primitive_scroll": primitive_scroll,
            "primitive_wait": primitive_wait,
            "click": primitive_click,
            "type": primitive_type,
            "scroll": primitive_scroll,
            "wait": primitive_wait,
            "find": find,
            "find_input": find_input,
        }
        # let the body call any other skill in the library
        for s in self.library:
            if s.is_alias() or s.name == skill.name:
                continue
            ns[s.name] = self._make_skill_proxy(s)
        # restricted builtins
        ns["__builtins__"] = {
            "len": len, "range": range, "isinstance": isinstance,
            "any": any, "all": all, "str": str, "int": int, "float": float,
            "bool": bool, "list": list, "dict": dict, "set": set, "tuple": tuple,
            "min": min, "max": max, "enumerate": enumerate, "zip": zip,
            "print": print, "True": True, "False": False, "None": None,
        }
        return ns

    def _make_skill_proxy(self, sub_skill: Skill):
        """Return a callable that re-enters `run` for a sub-skill.

        This is how nested calls inside an induced skill body actually take effect:
        the body literally calls `search_and_filter(obs, query=q, ...)` and our proxy
        translates it into another `executor.run(SkillCall, ...)` invocation.
        """

        executor = self

        def proxy(obs, **kwargs):
            sub_call = SkillCall(name=sub_skill.name, args=kwargs)
            result = executor.run(sub_call, obs, executor._current_action_list())
            if not result.success:
                raise RuntimeError(
                    f"sub-skill {sub_skill.name} failed: {result.error or 'postcondition'}"
                )
            return result

        proxy.__name__ = sub_skill.name
        return proxy

    def _current_action_list(self) -> list[Action]:
        ctx = current_context()
        return ctx.actions if ctx is not None else []
