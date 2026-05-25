"""VisualWebArena environment adapter (Koh et al., 2024).

VWA shares WebArena's infrastructure but adds tasks that explicitly require
visual reasoning. Most of the adapter logic is identical; the main differences
are:
  - the tasks JSON has 910 entries (vs 812 in WebArena);
  - tasks may carry a `requires_screenshot` flag that the rollout uses to
    decide whether to attach the rendered page image to the policy prompt;
  - some verifiers are image-based (template matching) rather than text-based.

The image-verifier implementations are out of scope for the public stub —
when present they are exposed as opaque callables in `Task.verifier`. Users
running the real benchmark should install the upstream `visualwebarena`
package which provides these verifiers.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from scaffold.core.verifier import BinaryVerifier
from scaffold.envs.base import Task
from scaffold.envs.webarena_env import WebArenaEnv
from scaffold.utils.logging import get_logger

log = get_logger("env.vwa")


class VisualWebArenaEnv(WebArenaEnv):
    def __init__(
        self,
        tasks_path: Optional[str] = None,
        base_urls: Optional[dict[str, str]] = None,
        *,
        sites: Optional[list[str]] = None,
        headless: bool = True,
    ) -> None:
        super().__init__(
            tasks_path=tasks_path or os.environ.get("VWA_TASKS"),
            base_urls=base_urls,
            sites=sites or ["classifieds", "reddit", "shopping"],
            headless=headless,
        )

    def tasks(self, *, split: str = "train") -> list[Task]:
        # delegate to WebArenaEnv loader (same JSON schema); annotate each task
        # with whether visual reasoning is required.
        tasks = super().tasks(split=split)
        for t in tasks:
            t.metadata["benchmark"] = "vwa"
            t.metadata["requires_screenshot"] = t.metadata.get(
                "requires_screenshot", True
            )
        return tasks
