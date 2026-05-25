"""MDL functional (Eq. 2) and compaction orchestration (§3.5).

  F(L) = Σ_σ |σ|                                  ← library cost
       + Σ_ζ min_{parse} |parse(ζ | L)|          ← data cost

Implementation notes:
  - |σ| uses the cheap word/symbol tokenizer in `Skill.token_length`. The MDL
    operators only care about *relative* deltas, so any monotone proxy works.
  - The data cost is approximated by the trajectory-parse heuristic in
    `parse_trajectory_under_library`: a greedy longest-match decoder over the
    library + primitive vocabulary. This matches the "shortest program that
    reproduces ζ" notion well enough that operator gains stay aligned with full
    re-rollouts in our ablations.
  - Both costs are integers; F is non-negative and bounded above by Σ_σ |σ| +
    Σ_ζ |ζ| (the all-primitives parse).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Protocol

from scaffold.core.library import SkillLibrary
from scaffold.core.skill import Skill
from scaffold.core.trajectory import ActionType, Trajectory
from scaffold.primitives.actions import PRIMITIVE_NAMES
from scaffold.utils.logging import get_logger

log = get_logger("mdl")


# ---------------------------------------------------------------------------
# Trajectory parsing under a library (the data-cost piece)
# ---------------------------------------------------------------------------

@dataclass
class ParseResult:
    tokens: list[str] = field(default_factory=list)   # sequence of skill / primitive names
    length: int = 0


def parse_trajectory_under_library(
    trajectory: Trajectory, library: SkillLibrary
) -> ParseResult:
    """Greedy longest-match parse of a trajectory under library + primitives.

    The trajectory's action sequence is already a flat list of (primitive | skill)
    invocations because the executor records every step. So the parse is one of:
      - if action `a` is a skill invocation, keep it as one token;
      - if it's a primitive, see whether the *next k* primitives match the body
        of some skill σ ∈ library (after parameter-blanking). If so, count σ as
        one token and advance k positions.

    This is a useful, cheap proxy for the "shortest program that reproduces ζ"
    quantity in Eq. (2). It is monotone in library quality: any new skill that
    covers a recurring primitive sequence strictly reduces the parse length.
    """
    seq = trajectory.primitive_subsequence()
    primitive_bodies = _library_primitive_bodies(library)

    tokens: list[str] = []
    i = 0
    while i < len(seq):
        tok = seq[i]
        # if the next k tokens equal one of the library skill bodies, fold.
        best_len, best_name = 1, tok
        for name, body_seq in primitive_bodies.items():
            n = len(body_seq)
            if n > 1 and seq[i : i + n] == body_seq and n > best_len:
                best_len, best_name = n, name
        tokens.append(best_name)
        i += best_len
    return ParseResult(tokens=tokens, length=len(tokens))


def _library_primitive_bodies(library: SkillLibrary) -> dict[str, list[str]]:
    """Best-effort mapping skill_name -> sequence of primitive action types in its
    body. Used only by the parser; we extract from the static body source.
    """
    out: dict[str, list[str]] = {}
    for s in library:
        if s.is_alias():
            continue
        seq = _extract_primitive_calls_from_body(s.body)
        # only useful entries (those reducing >1 primitive)
        if len(seq) > 1:
            out[s.name] = seq
    return out


def _extract_primitive_calls_from_body(body: str) -> list[str]:
    import re

    out: list[str] = []
    for ln in body.splitlines():
        ln = ln.split("#", 1)[0].strip()  # strip comments
        if not ln:
            continue
        m = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", ln)
        if not m:
            continue
        name = m.group(1)
        # map primitive aliases back to their canonical action type
        if name in ("primitive_click", "click"):
            out.append(ActionType.CLICK.value)
        elif name in ("primitive_type", "type"):
            out.append(ActionType.TYPE.value)
        elif name in ("primitive_scroll", "scroll"):
            out.append(ActionType.SCROLL.value)
        elif name in ("primitive_wait", "wait"):
            out.append(ActionType.WAIT.value)
        elif name in PRIMITIVE_NAMES:
            # other primitive helpers (find, find_input) don't generate actions
            continue
        else:
            # nested skill call; we track that as a skill token
            out.append(name)
    return out


# ---------------------------------------------------------------------------
# MDL objective
# ---------------------------------------------------------------------------

def mdl_objective(
    library: SkillLibrary, trajectories: Iterable[Trajectory]
) -> tuple[int, int, int]:
    """Compute F(L). Returns (F_total, library_cost, data_cost)."""
    lib_cost = sum(s.token_length() for s in library if not s.is_alias())
    data_cost = 0
    n_traj = 0
    for t in trajectories:
        if not t.is_successful():
            continue
        data_cost += parse_trajectory_under_library(t, library).length
        n_traj += 1
    return lib_cost + data_cost, lib_cost, data_cost


# ---------------------------------------------------------------------------
# Compactor orchestrator
# ---------------------------------------------------------------------------

@dataclass
class MDLConfig:
    """Hyperparameters governing the three operators (matches §4.1.4)."""

    rho: float = 0.9              # behavioral-equivalence threshold for Merge
    l_min: int = 3                # min subseq length for Refactor
    r_min: int = 3                # min #skills sharing a subseq for Refactor
    max_iterations: int = 8       # outer-loop safety cap
    require_validation_no_drop: bool = True   # accept iff SR(T_val) doesn't drop
    validation_sample_size: int = 32          # # holdout tasks sampled to check SR


class ValidationCallback(Protocol):
    """SR estimator on a held-out validation set T_val. Returns a float in [0,1].

    The pipeline injects a closure over the env adapter so the compactor can call
    it cheaply; in tests we pass an always-1.0 stub.
    """

    def __call__(self, library: SkillLibrary) -> float: ...


class MDLCompactor:
    """Implements Algorithm 1, lines 14-16 + §3.5.

    `apply` runs the three operators in rotation. Each operator emits CANDIDATE
    modifications; the compactor accepts a candidate iff
        (a) F(L) strictly decreases  AND
        (b) val_sr(L_new) >= val_sr(L_old) (no drop on T_val).
    Both checks are performed in `apply_candidate`.
    """

    def __init__(
        self,
        config: Optional[MDLConfig] = None,
        *,
        merge_op: Optional["MergeOperator"] = None,
        refactor_op: Optional["RefactorOperator"] = None,
        prune_op: Optional["PruneOperator"] = None,
    ) -> None:
        from scaffold.compaction.merge import MergeOperator
        from scaffold.compaction.prune import PruneOperator
        from scaffold.compaction.refactor import RefactorOperator

        self.config = config or MDLConfig()
        self.merge = merge_op or MergeOperator(rho=self.config.rho)
        self.refactor = refactor_op or RefactorOperator(
            l_min=self.config.l_min, r_min=self.config.r_min,
        )
        self.prune = prune_op or PruneOperator()

    def apply(
        self,
        library: SkillLibrary,
        positive_trajectories: list[Trajectory],
        *,
        validation_sr: Optional[ValidationCallback] = None,
        recent_used_skill_names: Optional[set[str]] = None,
    ) -> SkillLibrary:
        """Run the compactor until F stops decreasing or we hit max_iterations.

        Args:
            library: L_{k+1} (post-induction).
            positive_trajectories: Z+_{0:k}, used by the parser-based data cost
                and by the Prune operator's usage accounting.
            validation_sr: callable returning success rate on T_val for a given
                library. If None, the no-SR-drop check is skipped (use only for
                tests/ablations).
            recent_used_skill_names: names of skills invoked in the most recent
                M iterations — these are protected from Prune. Defaults to "all
                names that appear in any positive trajectory's plan".

        Returns:
            The (possibly modified) library. The original is mutated in-place
            and also returned for ergonomic chaining.
        """
        log = get_logger("mdl.compactor")
        baseline_sr = validation_sr(library) if validation_sr else 1.0
        if recent_used_skill_names is None:
            recent_used_skill_names = set()
            for t in positive_trajectories:
                recent_used_skill_names.update(t.skill_plan())

        for it in range(self.config.max_iterations):
            f_old, _, _ = mdl_objective(library, positive_trajectories)
            log.info("MDL iter %d: F=%d, |L|=%d", it, f_old, library.total_skills())

            # Try operators in order: Merge → Refactor → Prune.
            # The order is empirically what gives the fastest convergence:
            # merging behavioral dups first reduces the refactor pattern search.
            applied_any = False

            for cand in self.merge.propose(library):
                ok = self._try_accept(
                    library, cand, positive_trajectories,
                    validation_sr, baseline_sr,
                )
                if ok:
                    applied_any = True

            for cand in self.refactor.propose(library):
                ok = self._try_accept(
                    library, cand, positive_trajectories,
                    validation_sr, baseline_sr,
                )
                if ok:
                    applied_any = True

            for cand in self.prune.propose(
                library, protected=recent_used_skill_names,
            ):
                ok = self._try_accept(
                    library, cand, positive_trajectories,
                    validation_sr, baseline_sr,
                )
                if ok:
                    applied_any = True

            if not applied_any:
                log.info("MDL converged after %d iterations", it)
                break

        return library

    # ----- private --------------------------------------------------------

    def _try_accept(
        self,
        library: SkillLibrary,
        candidate,
        trajectories: list[Trajectory],
        validation_sr: Optional[ValidationCallback],
        baseline_sr: float,
    ) -> bool:
        f_before, _, _ = mdl_objective(library, trajectories)
        snapshot = library.copy()
        try:
            candidate.apply(library)
        except Exception as e:
            log.debug("Candidate %r failed to apply: %r", candidate, e)
            # restore
            library._skills.clear()
            library._embeddings.clear()
            library._skills.update(snapshot._skills)
            library._embeddings.update(snapshot._embeddings)
            return False

        f_after, _, _ = mdl_objective(library, trajectories)
        if f_after >= f_before:
            # roll back
            library._skills.clear()
            library._embeddings.clear()
            library._skills.update(snapshot._skills)
            library._embeddings.update(snapshot._embeddings)
            return False

        if validation_sr is not None and self.config.require_validation_no_drop:
            new_sr = validation_sr(library)
            if new_sr < baseline_sr - 1e-9:
                library._skills.clear()
                library._embeddings.clear()
                library._skills.update(snapshot._skills)
                library._embeddings.update(snapshot._embeddings)
                log.info(
                    "Reject candidate %s: F %d→%d but val SR %.3f→%.3f",
                    type(candidate).__name__, f_before, f_after, baseline_sr, new_sr,
                )
                return False

        log.info(
            "Accept candidate %s: F %d→%d  (|L|: %d)",
            type(candidate).__name__, f_before, f_after, library.total_skills(),
        )
        return True
