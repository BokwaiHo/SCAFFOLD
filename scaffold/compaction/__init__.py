"""Stage 4: MDL-driven library compaction (§3.5).

Public API:
  - `MDLCompactor`         the orchestrator that wraps the three operators;
  - `MDLConfig`             knobs (rho, l_min, r_min, ...);
  - `mdl_objective(L, Z+)`  Eq. (2) functional;
  - operator classes:       MergeOperator, RefactorOperator, PruneOperator.

The paper's accept/reject rule is in `MDLCompactor.apply_candidate`: any
proposed modification is taken only if (a) it reduces F AND (b) it does not
decrease success rate on a held-out validation set T_val.
"""

from scaffold.compaction.mdl import (
    MDLCompactor,
    MDLConfig,
    mdl_objective,
    ParseResult,
)
from scaffold.compaction.merge import MergeOperator, MergeCandidate
from scaffold.compaction.refactor import RefactorOperator, RefactorCandidate
from scaffold.compaction.prune import PruneOperator, PruneCandidate

__all__ = [
    "MDLCompactor",
    "MDLConfig",
    "mdl_objective",
    "ParseResult",
    "MergeOperator",
    "MergeCandidate",
    "RefactorOperator",
    "RefactorCandidate",
    "PruneOperator",
    "PruneCandidate",
]
