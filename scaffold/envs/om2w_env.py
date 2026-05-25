"""Online-Mind2Web environment adapter (Xue et al., 2025).

OM2W has 300 tasks on 136 websites across 12 domains. The paper partitions by
SITES (not by tasks) into:

  - training pool: 9 domains
  - held-out test: 3 domains  ←  "Jobs & Careers", "Travel & Transportation",
                                  "Government & Services"  (§4.1.1)

We expose this partition as the `split` argument: "train" returns tasks from
the 9 training domains, "test" returns held-out test tasks. The cross-site
transfer matrix in §4.6 is computed by training the agent on just one site
(filtering via `train_sites=`) and evaluating on the others.

Like WebArenaEnv, the live-web Playwright dependency is lazy.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from scaffold.core.verifier import BinaryVerifier
from scaffold.envs.base import Env, Task
from scaffold.primitives.actions import Browser
from scaffold.utils.logging import get_logger

log = get_logger("env.om2w")


HELD_OUT_DOMAINS: tuple[str, ...] = (
    "jobs_and_careers",
    "travel_and_transportation",
    "government_and_services",
)


class OnlineMind2WebEnv(Env):
    def __init__(
        self,
        tasks_path: Optional[str] = None,
        *,
        train_sites: Optional[list[str]] = None,
        held_out_domains: tuple[str, ...] = HELD_OUT_DOMAINS,
        headless: bool = True,
    ) -> None:
        self.tasks_path = tasks_path or os.environ.get("OM2W_TASKS")
        self.train_sites = train_sites
        self.held_out_domains = held_out_domains
        self.headless = headless
        self._tasks_cache: dict[str, list[Task]] = {}

    def tasks(self, *, split: str = "train") -> list[Task]:
        if split in self._tasks_cache:
            return self._tasks_cache[split]
        if not self.tasks_path or not os.path.exists(self.tasks_path):
            log.warning("OM2W tasks file not found (%s)", self.tasks_path)
            self._tasks_cache[split] = []
            return []
        with open(self.tasks_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        in_held = lambda d: d in self.held_out_domains
        out: list[Task] = []
        for entry in raw:
            domain = entry.get("domain", "")
            site = entry.get("site", "")
            if split == "train":
                if in_held(domain):
                    continue
                if self.train_sites and site not in self.train_sites:
                    continue
            elif split == "test":
                if not in_held(domain):
                    continue
            elif split == "val":
                # carve a tiny within-train val slice: 5% per domain
                if in_held(domain):
                    continue
                if hash(entry.get("task_id", "")) % 20 != 0:
                    continue
            else:
                continue

            answer = entry.get("expected_answer", "")
            out.append(Task(
                task_id=str(entry["task_id"]),
                instruction=entry["instruction"],
                site=site,
                verifier=BinaryVerifier(
                    lambda t, a=answer:
                        (t.final_answer or "").strip().lower() == a.strip().lower()
                ),
                initial_state={"start_url": entry.get("start_url", "")},
                metadata=entry,
            ))

        self._tasks_cache[split] = out
        log.info("Loaded %d OM2W tasks (split=%s)", len(out), split)
        return out

    def make_browser(self, task: Task) -> Browser:
        try:
            from scaffold.envs._playwright_browser import PlaywrightBrowser
        except ImportError as e:
            raise ImportError("Playwright is required for OnlineMind2WebEnv") from e
        return PlaywrightBrowser(
            start_url=task.initial_state["start_url"], headless=self.headless,
        )

    def close_browser(self, browser: Browser) -> None:
        if hasattr(browser, "close"):
            browser.close()
