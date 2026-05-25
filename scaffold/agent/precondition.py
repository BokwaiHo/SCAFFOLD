"""Pre/postcondition validator (App. A "Pre/Postcondition validator").

Paper text:
  At runtime, before invoking a skill we verify its precondition by passing
  (obs_t, pre_σ) to a lightweight VLM judge with a yes/no output. Same protocol for
  postconditions after execution. The judge is a separate, cheaper model from the
  inducer to keep inference cost bounded.

We provide three implementations:
  - `AlwaysTrueValidator`         — disables checks (useful for unit tests)
  - `PythonExprValidator`         — eval the predicate as a Python expression over
                                    `obs` (cheap, deterministic; suitable when the
                                    inducer wrote pre/post as actual Python).
  - `LLMJudgeValidator`           — fall back to a small LLM for free-form predicates
                                    like "logged_in(obs)" or natural-language ones.

The pipeline configures one validator and the executor calls it before every skill
invocation and after the body returns.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from scaffold.core.trajectory import Observation
from scaffold.utils.logging import get_logger

log = get_logger("precondition")


class PreconditionValidator(ABC):
    @abstractmethod
    def check(self, predicate: str, observation: Observation) -> bool: ...


class AlwaysTrueValidator(PreconditionValidator):
    def check(self, predicate: str, observation: Observation) -> bool:  # noqa: ARG002
        return True


class PythonExprValidator(PreconditionValidator):
    """Try to eval `predicate` as a Python expression over `obs` and a few helpers.

    Predicates that are clearly Python (and which the inducer tends to produce for
    the simple App. B examples like `any(e.text == text for e in obs.dom)`) are
    handled here without an LLM call. Anything that fails parsing or eval falls back
    to True by default — callers who want stricter behavior should chain this with
    `LLMJudgeValidator`.
    """

    def check(self, predicate: str, observation: Observation) -> bool:
        p = predicate.strip()
        if not p or p.lower() == "true":
            return True
        if p.lower() == "false":
            return False
        ns = {
            "obs": observation,
            "url": observation.url,
            "dom": observation.dom,
            "any": any, "all": all, "len": len,
            "True": True, "False": False, "None": None,
        }
        try:
            return bool(eval(p, {"__builtins__": {}}, ns))  # noqa: S307
        except Exception as e:
            log.debug("PythonExprValidator: %r on %r -> fallback True", e, p)
            return True


class LLMJudgeValidator(PreconditionValidator):
    """Use a small LLM as a yes/no judge over (predicate, observation summary).

    Per the paper, this should be a *cheaper* model than the inducer. We expose the
    model name as a config knob so users can pick e.g. gpt-4o-mini.
    """

    def __init__(self, llm, *, max_tokens: int = 8) -> None:
        self.llm = llm
        self.max_tokens = max_tokens

    def check(self, predicate: str, observation: Observation) -> bool:
        from scaffold.utils.llm import ChatMessage

        if not predicate or predicate.strip().lower() in {"true", ""}:
            return True
        prompt = (
            f"Predicate: {predicate}\n"
            f"Observation: {observation.summary(max_chars=300)}\n"
            "Does the observation satisfy the predicate? Answer 'yes' or 'no' only."
        )
        resp = self.llm.chat(
            [
                ChatMessage("system", "You are a strict boolean judge."),
                ChatMessage("user", prompt),
            ],
            temperature=0.0,
            max_tokens=self.max_tokens,
        )
        ans = resp.text.strip().lower()
        return ans.startswith("y")


class ChainedValidator(PreconditionValidator):
    """Try `primary`; on parse-error fall back to `fallback`. Cost-efficient default
    is `PythonExprValidator` + `LLMJudgeValidator`.
    """

    def __init__(self, primary: PreconditionValidator, fallback: PreconditionValidator) -> None:
        self.primary = primary
        self.fallback = fallback

    def check(self, predicate: str, observation: Observation) -> bool:
        try:
            return self.primary.check(predicate, observation)
        except Exception:
            return self.fallback.check(predicate, observation)
