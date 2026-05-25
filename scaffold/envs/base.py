"""Base environment interfaces.

The pipeline only depends on this module — concrete envs (toy/WebArena/VWA/OM2W)
plug in here. Splitting it out keeps the dependency on real browsers strictly
optional.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from scaffold.core.trajectory import Trajectory
from scaffold.core.verifier import Verifier
from scaffold.primitives.actions import Browser


@dataclass
class Task:
    """τ = (I, init, V) from §3.1."""

    task_id: str
    instruction: str
    site: Optional[str] = None
    verifier: Optional[Verifier] = None
    initial_state: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RolloutConfig:
    """Per-iteration rollout knobs (matches §4.1.4)."""

    max_steps: int = 30
    temperature: float = 0.7
    n_attempts_per_task: int = 1
    seed: int = 0


class Env(ABC):
    """Web environment adapter."""

    @abstractmethod
    def tasks(self, *, split: str = "train") -> list[Task]: ...

    @abstractmethod
    def make_browser(self, task: Task) -> Browser:
        """Open a fresh browser at the task's initial state. The pipeline
        calls this once per task attempt and disposes after the trajectory
        is logged."""

    @abstractmethod
    def close_browser(self, browser: Browser) -> None: ...

    # The default rollout loop lives in `scaffold.pipeline.ScaffoldPipeline.rollout`,
    # which mediates between Env + Policy + SkillExecutor. Envs only need to
    # supply the tasks and the per-task browser.

    # Validation set sampling for the MDL compactor's no-SR-drop check.
    def validation_tasks(self, *, n: int = 32) -> list[Task]:
        all_tasks = self.tasks(split="val")
        return all_tasks[:n]
