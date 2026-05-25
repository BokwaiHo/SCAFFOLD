"""Merge operator (§3.5).

Paper:
    Merge: two skills σ_a, σ_b that are behaviorally equivalent on a sampled
    held-out set (i.e., produce identical post-states with probability ≥ ρ) are
    merged into the shorter of the two, with the longer being replaced by an
    alias.

We implement two equivalence checks:
  - Static: parameter list, return type-shape, and primitive call sequence
    after argument-blanking. A pure-static check that's cheap and catches the
    `login_v1 / login_v2 / login_with_email` family (Fig. 3) the paper calls out.
  - Behavioral (optional): if a `behavioral_check(σ_a, σ_b) -> float` is
    injected, use its returned probability against the threshold ρ.

Static check is used by default since the data cost in F(L) makes it a sound
proxy for behavioral equivalence on the trajectories the library was inferred
from. Behavioral checks are used during the full WebArena runs reported in the
main results (driven by env replay).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from scaffold.core.library import SkillLibrary
from scaffold.core.skill import Skill
from scaffold.compaction.mdl import _extract_primitive_calls_from_body
from scaffold.utils.logging import get_logger

log = get_logger("mdl.merge")

# Type alias for behavioral checks. Returns equivalence probability in [0,1].
BehavioralCheck = Callable[[Skill, Skill], float]


@dataclass
class MergeCandidate:
    """A proposed merge: alias `loser` to `winner` (the shorter, simpler one)."""

    winner: str
    loser: str
    reason: str = ""

    def apply(self, library: SkillLibrary) -> None:
        # The library's alias mechanism preserves the name in the namespace but
        # forwards every lookup to the winner. This is what the paper means by
        # "the longer being replaced by an alias".
        library.mark_alias(self.loser, self.winner)

    def __repr__(self) -> str:
        return f"MergeCandidate({self.loser} -> {self.winner})"


class MergeOperator:
    """Yields MergeCandidate proposals until called code rejects or accepts each.

    Determinism: candidates are emitted in (sorted-name pair) order so two
    identical runs yield identical merge sequences.
    """

    def __init__(
        self,
        rho: float = 0.9,
        *,
        behavioral_check: Optional[BehavioralCheck] = None,
    ) -> None:
        self.rho = rho
        self.behavioral_check = behavioral_check

    def propose(self, library: SkillLibrary) -> Iterable[MergeCandidate]:
        skills = [s for s in library if not s.is_alias()]
        skills.sort(key=lambda s: s.name)
        for i, a in enumerate(skills):
            for b in skills[i + 1 :]:
                if a.name == b.name:
                    continue
                # parameter list must be compatible
                if not self._parameter_lists_compatible(a, b):
                    continue
                if self.behavioral_check is not None:
                    p = self.behavioral_check(a, b)
                    if p < self.rho:
                        continue
                    cand = self._build_candidate(a, b, reason=f"behavioral p={p:.2f}")
                else:
                    # static equivalence: same primitive-call sequence ignoring args
                    if not self._static_equivalent(a, b):
                        continue
                    cand = self._build_candidate(a, b, reason="static-equivalent body")
                yield cand

    # ---- helpers -----------------------------------------------------------

    def _parameter_lists_compatible(self, a: Skill, b: Skill) -> bool:
        if len(a.parameters) != len(b.parameters):
            return False
        for pa, pb in zip(a.parameters, b.parameters):
            if pa.type != pb.type:
                return False
        return True

    def _static_equivalent(self, a: Skill, b: Skill) -> bool:
        if a.calls_set() != b.calls_set():
            return False
        seq_a = _extract_primitive_calls_from_body(a.body)
        seq_b = _extract_primitive_calls_from_body(b.body)
        return seq_a == seq_b and self._normalized_body(a) == self._normalized_body(b)

    @staticmethod
    def _normalized_body(s: Skill) -> str:
        """Strip whitespace, comments, and string literals so token-level
        differences in selectors / labels don't block a merge."""
        body = s.body
        body = re.sub(r"#.*", "", body)
        body = re.sub(r'"[^"]*"', '"_"', body)
        body = re.sub(r"'[^']*'", "'_'", body)
        body = re.sub(r"\s+", " ", body).strip()
        return body

    @staticmethod
    def _build_candidate(a: Skill, b: Skill, *, reason: str) -> MergeCandidate:
        # winner = shorter body; ties broken alphabetically for determinism
        if a.token_length() < b.token_length():
            winner, loser = a, b
        elif a.token_length() > b.token_length():
            winner, loser = b, a
        elif a.name < b.name:
            winner, loser = a, b
        else:
            winner, loser = b, a
        return MergeCandidate(winner=winner.name, loser=loser.name, reason=reason)
