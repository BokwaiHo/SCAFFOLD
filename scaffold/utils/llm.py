"""LLM client abstraction.

The paper uses GPT-4o-2024-08 (temperature 0.3) for the inducer and the MDL refactor
proposer (§4.1.4). We hide that behind an interface so:
  - tests can use `MockChatClient` for deterministic, offline runs;
  - users without OpenAI access can swap in Anthropic / vLLM / Ollama;
  - the rate-limited retry / cost-accounting wrapping is in one place.

Cost note (§Limitations, point 1): the inducer accounts for ~70% of total $ cost.
The `usage` field on every Response object lets callers monitor this in real time.
"""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


# ---------- data classes ----------------------------------------------------


@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant"
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class Response:
    text: str
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    finish_reason: Optional[str] = None

    def json(self) -> Any:
        """Convenience: parse the model output as JSON, raising if invalid."""
        s = self.text.strip()
        # strip markdown code fences if the model wrapped its output in them
        if s.startswith("```"):
            lines = s.splitlines()
            s = "\n".join(lines[1:-1]) if len(lines) >= 2 else s
        return json.loads(s)


# ---------- abstract client -------------------------------------------------


class LLMClient(ABC):
    """Abstract sync chat client."""

    @abstractmethod
    def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        stop: Optional[list[str]] = None,
    ) -> Response: ...

    # public running totals for cost tracking
    cumulative_usage: Usage

    def reset_usage(self) -> None:
        self.cumulative_usage = Usage()


# ---------- OpenAI backend --------------------------------------------------


class OpenAIChatClient(LLMClient):
    """Default backend matching the paper's setup (GPT-4o-2024-08, t=0.3)."""

    def __init__(
        self,
        model: str = "gpt-4o-2024-08-06",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai package is required for OpenAIChatClient; "
                "`pip install openai` or switch to MockChatClient"
            ) from e

        self.model = model
        self._client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url,
        )
        self.cumulative_usage = Usage()

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        reraise=True,
    )
    def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        stop: Optional[list[str]] = None,
    ) -> Response:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[m.to_dict() for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
        )
        choice = resp.choices[0]
        usage = Usage(
            prompt_tokens=getattr(resp.usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(resp.usage, "completion_tokens", 0) or 0,
            total_tokens=getattr(resp.usage, "total_tokens", 0) or 0,
        )
        self.cumulative_usage.prompt_tokens += usage.prompt_tokens
        self.cumulative_usage.completion_tokens += usage.completion_tokens
        self.cumulative_usage.total_tokens += usage.total_tokens
        return Response(
            text=choice.message.content or "",
            usage=usage,
            model=self.model,
            finish_reason=choice.finish_reason,
        )


# ---------- Mock backend (tests, toy env) -----------------------------------


class MockChatClient(LLMClient):
    """Deterministic client driven by a list of canned replies.

    Used by the unit tests so the pipeline can exercise every stage without network
    access. Also handy for reproducing a buggy run from a logged transcript.
    """

    def __init__(self, replies: Optional[list[str]] = None) -> None:
        self.replies: list[str] = list(replies or [])
        self.calls: list[list[ChatMessage]] = []
        self.cumulative_usage = Usage()
        self._default = "{}"

    def push(self, *replies: str) -> None:
        self.replies.extend(replies)

    def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.3,  # noqa: ARG002
        max_tokens: int = 1024,    # noqa: ARG002
        stop: Optional[list[str]] = None,  # noqa: ARG002
    ) -> Response:
        self.calls.append(list(messages))
        if self.replies:
            text = self.replies.pop(0)
        else:
            text = self._default
        usage = Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20)
        self.cumulative_usage.total_tokens += usage.total_tokens
        return Response(text=text, usage=usage, model="mock")


def make_llm_client(
    backend: str = "openai", **kwargs: Any
) -> LLMClient:
    if backend == "openai":
        return OpenAIChatClient(**kwargs)
    if backend == "mock":
        return MockChatClient(**kwargs)
    raise ValueError(f"Unknown LLM backend: {backend}")
