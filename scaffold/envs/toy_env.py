"""Synthetic toy environment.

A handful of "pages" with deterministic transitions, designed so the pipeline
can run end-to-end in a few seconds without any browser, GPU, or API keys.

The toy env supports four task families that *intentionally* share procedural
structure so multi-instance induction has something to cluster:

  - search_and_view_product(query)        depth-1 patterns
  - login_then_search(user, query)        composes login + search_and_view
  - filter_and_add_to_cart(query, max_price)  depth-2 expected
  - bulk_purchase(items)                  depth-3 expected after k=3

Pipeline tests rely on these clustering nicely under HashEmbedder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from scaffold.core.trajectory import Observation
from scaffold.core.verifier import BinaryVerifier, Verifier
from scaffold.envs.base import Env, Task
from scaffold.primitives.actions import Browser


@dataclass
class ToyPage:
    url: str
    elements: list[dict[str, Any]] = field(default_factory=list)

    def dom(self) -> str:
        return "\n".join(
            f"<{el.get('tag','a')} text={el.get('text','')!r} role={el.get('role','')!r}>"
            for el in self.elements
        )


class ToyBrowser(Browser):
    """Deterministic in-memory browser; simulates page transitions."""

    def __init__(self, pages: dict[str, ToyPage], start_url: str) -> None:
        self.pages = pages
        self.current_url = start_url
        self.cart: list[str] = []
        self.logged_in = False
        self.purchased: bool = False
        self._step = 0
        self._typed: dict[str, str] = {}

    def observe(self) -> Observation:
        self._step += 1
        page = self.pages[self.current_url]
        return Observation(
            step=self._step,
            url=self.current_url,
            dom=page.dom(),
            extra={"cart": list(self.cart), "logged_in": self.logged_in,
                   "purchased": self.purchased, "typed": dict(self._typed)},
        )

    def click(self, target: Any) -> Observation:
        # Toy transition rules.
        page = self.pages[self.current_url]
        text = target if isinstance(target, str) else ""
        for el in page.elements:
            if el.get("text") == text:
                if "goto" in el:
                    self.current_url = el["goto"]
                if el.get("action") == "add_to_cart":
                    self.cart.append(self._typed.get("Search", "item"))
                if el.get("action") == "place_order" and self.cart:
                    self.purchased = True
                if el.get("action") == "login":
                    self.logged_in = True
                break
        return self.observe()

    def type(self, target: Any, text: str) -> Observation:
        self._typed[str(target)] = str(text)
        return self.observe()

    def scroll(self, dx: int = 0, dy: int = 0) -> Observation:
        return self.observe()

    def wait(self, seconds: float) -> Observation:
        return self.observe()


def _shop_pages() -> dict[str, ToyPage]:
    return {
        "/login": ToyPage("/login", [
            {"tag": "input", "role": "input", "label": "username", "text": "Username"},
            {"tag": "input", "role": "input", "label": "password", "text": "Password"},
            {"tag": "button", "text": "Sign In", "action": "login", "goto": "/shop"},
        ]),
        "/shop": ToyPage("/shop", [
            {"tag": "input", "role": "input", "label": "Search", "text": "Search"},
            {"tag": "button", "text": "Search", "goto": "/results"},
            {"tag": "a", "text": "Electronics"},
            {"tag": "a", "text": "Books"},
        ]),
        "/results": ToyPage("/results", [
            {"tag": "a", "text": "first_result", "goto": "/product"},
            {"tag": "button", "text": "Sort: price_asc"},
            {"tag": "input", "role": "input", "label": "Max Price",
             "text": "Max Price"},
        ]),
        "/product": ToyPage("/product", [
            {"tag": "button", "text": "Add to Cart", "action": "add_to_cart"},
            {"tag": "button", "text": "Checkout", "goto": "/checkout"},
        ]),
        "/checkout": ToyPage("/checkout", [
            {"tag": "input", "role": "input", "label": "PaymentToken",
             "text": "PaymentToken"},
            {"tag": "button", "text": "Place Order", "action": "place_order"},
        ]),
    }


class ToyEnv(Env):
    """Deterministic toy env. All tasks land on /shop after login."""

    def __init__(self, *, n_train: int = 60, n_val: int = 16, n_test: int = 32) -> None:
        self.n_train = n_train
        self.n_val = n_val
        self.n_test = n_test
        self._pages = _shop_pages()

    def tasks(self, *, split: str = "train") -> list[Task]:
        n = {"train": self.n_train, "val": self.n_val, "test": self.n_test}.get(split, 0)
        queries = ["laptop", "phone", "book", "watch", "tablet", "headphones",
                   "camera", "monitor", "keyboard", "mouse"]
        cats = ["Electronics", "Books"]
        out: list[Task] = []
        for i in range(n):
            q = queries[i % len(queries)]
            c = cats[i % len(cats)]
            instruction = f"Search for {q} in {c} and add the cheapest to cart"
            out.append(Task(
                task_id=f"toy:{split}:{i:04d}",
                instruction=instruction,
                site="toyshop",
                verifier=BinaryVerifier(lambda t, q=q: q in (t.observations[-1].extra.get("cart") or [])),
                initial_state={"query": q, "category": c, "start_url": "/login"},
                metadata={"split": split},
            ))
        return out

    def make_browser(self, task: Task) -> Browser:
        start_url = task.initial_state.get("start_url", "/login")
        return ToyBrowser(pages=self._pages, start_url=start_url)

    def close_browser(self, browser: Browser) -> None:
        # nothing to clean up for the in-memory toy
        return None


@dataclass
class ToyTask:
    """Re-export for backward compat with examples that import ToyTask directly."""
    inner: Task
