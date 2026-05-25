"""Primitive action implementations.

The skill executor (see scaffold.agent.executor) binds these functions into the
namespace it uses to `exec` skill bodies. Each primitive returns the next observation
or a sentinel; the actual side-effect routes through a `PrimitiveContext` which the
executor sets up before invoking a skill.

Why `PrimitiveContext` instead of globals: skill bodies are dynamically exec'd in a
namespace we control, but they must not have a direct handle on the live browser. The
context is the only authorized bridge; that keeps induced skill bodies pure & loggable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol

from scaffold.core.trajectory import Action, ActionType, Observation


PRIMITIVE_NAMES: frozenset[str] = frozenset({
    "primitive_click",
    "primitive_type",
    "primitive_scroll",
    "primitive_wait",
    # convenience helpers exposed in induced skill bodies; also "primitive-like"
    "click",
    "type",
    "scroll",
    "wait",
    "find",
    "find_input",
})


class Browser(Protocol):
    """Minimal interface that any env adapter must implement.

    Real WebArena uses Playwright; the toy env uses a deterministic stub. Both
    satisfy this protocol so the executor doesn't care which one is underneath.
    """

    def click(self, target: Any) -> Observation: ...
    def type(self, target: Any, text: str) -> Observation: ...
    def scroll(self, dx: int = 0, dy: int = 0) -> Observation: ...
    def wait(self, seconds: float) -> Observation: ...
    def observe(self) -> Observation: ...


@dataclass
class PrimitiveContext:
    """Per-skill-invocation context handed to primitives by the executor."""

    browser: Browser
    actions: list[Action]                # appended to as primitives fire
    skill_call_name: Optional[str] = None  # set by executor to attribute actions
    skill_depth: int = 0
    max_steps: int = 30                   # safety cap; paper uses 30-step horizon
    steps_taken: int = 0

    def record(self, type_: ActionType, args: dict[str, Any]) -> None:
        self.actions.append(Action(
            type=type_,
            args=args,
            skill_call=self.skill_call_name,
            skill_depth=self.skill_depth,
        ))
        self.steps_taken += 1
        if self.steps_taken >= self.max_steps:
            raise StepBudgetExceeded(
                f"Step budget {self.max_steps} exceeded in skill "
                f"{self.skill_call_name or '<root>'}"
            )


class StepBudgetExceeded(RuntimeError):
    """Raised when a skill consumes more than `max_steps` primitives."""


# --- bound primitives the executor injects into skill-body namespaces ------------

def _ctx() -> PrimitiveContext:
    from scaffold.agent.executor import current_context

    ctx = current_context()
    if ctx is None:
        raise RuntimeError(
            "Primitives invoked outside of a SkillExecutor.run() scope. "
            "This usually means a skill body was exec'd in the wrong namespace."
        )
    return ctx


def primitive_click(target: Any) -> Observation:
    """Click `target`, which may be a CSS selector, semantic role, or bbox tuple."""
    ctx = _ctx()
    obs = ctx.browser.click(target)
    ctx.record(ActionType.CLICK, {"target": target})
    return obs


def primitive_type(target: Any, text: str) -> Observation:
    """Type `text` into element `target`."""
    ctx = _ctx()
    obs = ctx.browser.type(target, text)
    ctx.record(ActionType.TYPE, {"target": target, "text": text})
    return obs


def primitive_scroll(dx: int = 0, dy: int = 0) -> Observation:
    ctx = _ctx()
    obs = ctx.browser.scroll(dx=dx, dy=dy)
    ctx.record(ActionType.SCROLL, {"dx": dx, "dy": dy})
    return obs


def primitive_wait(seconds: float = 0.5) -> Observation:
    ctx = _ctx()
    obs = ctx.browser.wait(seconds)
    ctx.record(ActionType.WAIT, {"seconds": seconds})
    return obs


# --- short aliases used by induced skill bodies (App. B style) ------------------

click = primitive_click
type = primitive_type  # noqa: A001 (we deliberately shadow builtin here in the exec namespace)
scroll = primitive_scroll
wait = primitive_wait


# --- DOM helpers also exposed in induced bodies (App. B uses `find`, `find_input`) -

def find(dom: Any, *, text: Optional[str] = None,
         role: Optional[str] = None, **kwargs: Any) -> Any:
    """Locate an element semantically.

    Mirrors the App. B skill examples: `find(obs.dom, text=...)`. Falls back to
    role + attribute filters when text is None.
    """
    if hasattr(dom, "find"):
        return dom.find(text=text, role=role, **kwargs)
    # default impl: pure-text search through a list of element-like dicts
    if isinstance(dom, list):
        for el in dom:
            if text is not None and getattr(el, "text", None) == text:
                return el
            if role is not None and getattr(el, "role", None) == role:
                return el
    return None


def find_input(dom: Any, label: str) -> Any:
    """Locate an input field by visible label or aria-label."""
    if hasattr(dom, "find_input"):
        return dom.find_input(label)
    return find(dom, role="input", label=label)
