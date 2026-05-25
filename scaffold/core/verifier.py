"""Verifier protocol: V(ζ) → {0, 1}.

Per §3.1 a task τ = (I, init, V), where V returns binary success on the final state.
WebArena and VisualWebArena ship deterministic verifiers; for the toy env we use a
trivial pattern-matching verifier defined alongside the env. See Limitation (2) in
§Limitations: SCAFFOLD assumes a per-task verifier exists.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from scaffold.core.trajectory import Trajectory


class Verifier(ABC):
    """Abstract base for any verifier handed to the pipeline."""

    @abstractmethod
    def verify(self, trajectory: Trajectory) -> bool: ...

    def __call__(self, trajectory: Trajectory) -> bool:
        return self.verify(trajectory)


class BinaryVerifier(Verifier):
    """Convenience wrapper around a Callable[[Trajectory], bool]."""

    def __init__(self, fn):
        self._fn = fn

    def verify(self, trajectory: Trajectory) -> bool:
        return bool(self._fn(trajectory))


class AlwaysSuccessVerifier(Verifier):
    """Useful only for development and the toy env smoke tests."""

    def verify(self, trajectory: Trajectory) -> bool:  # noqa: ARG002
        return True


class FinalAnswerVerifier(Verifier):
    """Marks a trajectory successful iff `final_answer` matches `expected` (case- and
    whitespace-insensitive). Mirrors the WebArena exact-string reward family.
    """

    def __init__(self, expected: str) -> None:
        self.expected = expected.strip().lower()

    def verify(self, trajectory: Trajectory) -> bool:
        got: Optional[str] = trajectory.final_answer
        if got is None:
            return False
        return got.strip().lower() == self.expected
