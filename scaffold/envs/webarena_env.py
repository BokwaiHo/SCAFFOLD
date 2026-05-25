"""WebArena environment adapter (Zhou et al., 2024).

This is a thin wrapper around the official WebArena setup. To use it:

  1. Start the WebArena docker stack as described in
     https://github.com/web-arena-x/webarena (the four self-hosted sites:
     E-commerce / Social-forum / GitLab / CMS).
  2. Install `playwright` and `webarena` (the upstream PyPI / git package).
  3. Set environment variables:
       WEBARENA_TASKS  = path to the official tasks JSON
       WEBARENA_BASE_URLS = JSON map of site names → URLs
  4. Instantiate `WebArenaEnv(...)` from your config.

We do NOT vendor the upstream code; the import is lazy so that running the
toy-env / unit-tests doesn't require any of these installs.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from scaffold.core.trajectory import Observation
from scaffold.core.verifier import BinaryVerifier
from scaffold.envs.base import Env, Task
from scaffold.primitives.actions import Browser
from scaffold.utils.logging import get_logger

log = get_logger("env.webarena")


class WebArenaEnv(Env):
    """Wraps WebArena tasks + Playwright browser."""

    def __init__(
        self,
        tasks_path: Optional[str] = None,
        base_urls: Optional[dict[str, str]] = None,
        *,
        sites: Optional[list[str]] = None,
        headless: bool = True,
    ) -> None:
        self.tasks_path = tasks_path or os.environ.get("WEBARENA_TASKS")
        env_base = os.environ.get("WEBARENA_BASE_URLS")
        if env_base and not base_urls:
            base_urls = json.loads(env_base)
        self.base_urls = base_urls or {}
        # WebArena's canonical four sites; users can restrict via `sites`.
        self.sites = sites or ["onestopshop", "reddit", "gitlab", "cms"]
        self.headless = headless
        self._tasks_cache: dict[str, list[Task]] = {}

    def tasks(self, *, split: str = "train") -> list[Task]:
        if split in self._tasks_cache:
            return self._tasks_cache[split]
        if not self.tasks_path or not os.path.exists(self.tasks_path):
            log.warning(
                "WebArena tasks file not found (%s); returning empty list. "
                "Set WEBARENA_TASKS env var to the official tasks JSON.",
                self.tasks_path,
            )
            self._tasks_cache[split] = []
            return []
        with open(self.tasks_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        tasks: list[Task] = []
        for entry in raw:
            if entry.get("split", "train") != split:
                continue
            if entry.get("site") not in self.sites:
                continue
            answer = entry.get("expected_answer", "")
            tasks.append(Task(
                task_id=str(entry["task_id"]),
                instruction=entry["instruction"],
                site=entry["site"],
                verifier=BinaryVerifier(
                    lambda t, a=answer:
                        (t.final_answer or "").strip().lower() == a.strip().lower()
                ),
                initial_state={"start_url": entry.get("start_url",
                                                       self.base_urls.get(entry["site"]))},
                metadata=entry,
            ))
        self._tasks_cache[split] = tasks
        log.info("Loaded %d WebArena tasks (split=%s)", len(tasks), split)
        return tasks

    def make_browser(self, task: Task) -> Browser:
        try:
            from scaffold.envs._playwright_browser import PlaywrightBrowser
        except ImportError as e:
            raise ImportError(
                "Playwright is required for WebArenaEnv; "
                "`pip install playwright && playwright install`"
            ) from e
        return PlaywrightBrowser(
            start_url=task.initial_state["start_url"],
            headless=self.headless,
        )

    def close_browser(self, browser: Browser) -> None:
        if hasattr(browser, "close"):
            browser.close()
