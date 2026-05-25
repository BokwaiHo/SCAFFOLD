"""Prune operator (§3.5).

Paper:
    Prune: skills used by zero trajectories in Z+_{0:k} over the most recent M
    iterations are removed (unless invoked transitively by another skill).

The compactor injects a `recent_used_skill_names` set computed from Z+; any skill
not in that set and not transitively invoked by another active skill is a Prune
candidate. The library refuses removal of a transitively-invoked skill, which
gives us defense in depth: even if the call-site bookkeeping in `recent_used`
were wrong, we'd never break composition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from scaffold.core.library import SkillLibrary
from scaffold.utils.logging import get_logger

log = get_logger("mdl.prune")


@dataclass
class PruneCandidate:
    """Remove `skill_name` from the library (if still safe to do so)."""

    skill_name: str

    def apply(self, library: SkillLibrary) -> None:
        # library.remove already refuses if the skill is transitively invoked
        library.remove(self.skill_name)

    def __repr__(self) -> str:
        return f"PruneCandidate({self.skill_name})"


class PruneOperator:
    """Yields PruneCandidates for skills that look unused and removable."""

    def propose(
        self,
        library: SkillLibrary,
        *,
        protected: Optional[set[str]] = None,
    ) -> Iterable[PruneCandidate]:
        protected = protected or set()

        # Compute who is transitively invoked, so we don't propose those.
        invoked: set[str] = set()
        for s in list(library):
            if s.is_alias():
                continue
            for callee in s.calls_set():
                if callee in library:
                    invoked.add(callee)

        # Snapshot library iteration before yielding so callers can mutate
        # during the loop (the compactor's _try_accept does exactly this).
        snapshot = list(library)
        for s in snapshot:
            if s.is_alias():
                continue
            if s.name in protected:
                continue
            if s.name in invoked:
                continue
            if s.times_used > 0:
                continue
            yield PruneCandidate(skill_name=s.name)
