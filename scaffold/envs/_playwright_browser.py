"""Playwright-backed Browser implementation.

Optional dependency: only imported by WebArenaEnv / VisualWebArenaEnv /
OnlineMind2WebEnv when an actual browser is needed. Keeping it in its own
module means the rest of the codebase has no hard dependency on Playwright.
"""

from __future__ import annotations

from typing import Any

from scaffold.core.trajectory import Observation


class PlaywrightBrowser:
    """Thin Playwright wrapper exposing the Browser protocol from primitives."""

    def __init__(self, start_url: str, *, headless: bool = True) -> None:
        from playwright.sync_api import sync_playwright  # type: ignore

        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=headless)
        self._context = self._browser.new_context()
        self._page = self._context.new_page()
        self._page.goto(start_url)
        self._step = 0

    def _resolve(self, target: Any):
        if isinstance(target, str) and target.startswith(("css=", "//", "/", "#", ".")):
            return self._page.locator(target).first
        # semantic role lookup: target like {"role": "button", "name": "..."}
        if isinstance(target, dict) and "role" in target:
            return self._page.get_by_role(target["role"], name=target.get("name", ""))
        # fall back to text match (matches App. B `click_text` style)
        return self._page.get_by_text(str(target)).first

    def observe(self) -> Observation:
        self._step += 1
        dom = self._page.content()
        return Observation(
            step=self._step, dom=dom[:8000], url=self._page.url,
        )

    def click(self, target: Any) -> Observation:
        self._resolve(target).click()
        return self.observe()

    def type(self, target: Any, text: str) -> Observation:
        loc = self._resolve(target)
        loc.fill(text)
        return self.observe()

    def scroll(self, dx: int = 0, dy: int = 0) -> Observation:
        self._page.mouse.wheel(dx, dy)
        return self.observe()

    def wait(self, seconds: float) -> Observation:
        self._page.wait_for_timeout(int(seconds * 1000))
        return self.observe()

    def close(self) -> None:
        try:
            self._page.close()
            self._context.close()
            self._browser.close()
            self._pw.stop()
        except Exception:
            pass
