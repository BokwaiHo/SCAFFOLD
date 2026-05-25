"""Recursive composition (§3.4).

This module enforces three invariants on every skill that enters the library:

  1. **Depth Eq. (1)**:  d_σ = 1 + max{ d_σ' : σ' ∈ calls(σ) },  with d_σ = 1 if
     calls(σ) ⊆ A_prim.

  2. **No cycles**: a topological check rejects any skill whose static call set
     transitively includes itself. The paper's wording: "we forbid cycles via a
     simple topological check at induction time".

  3. **Body cap** (already enforced by `Skill.__post_init__`): per-skill body
     length ≤ 30 lines.

We also expose two reporting helpers used by §4.5:
  - `depth_distribution`: counts of skills at each depth across iterations;
  - `composition_graph`: a name→{callees} adjacency map for visualization.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Optional

from scaffold.core.library import SkillLibrary
from scaffold.core.skill import Skill
from scaffold.primitives.actions import PRIMITIVE_NAMES


class CycleError(ValueError):
    """Raised when adding `skill` would create a cycle in the call graph."""


def compute_depth(skill: Skill, library: SkillLibrary) -> int:
    """Implements Eq. (1) using the skill's static `calls_set` and the depths
    already recorded for skills in `library`.

    If `calls(σ)` contains only primitives, depth=1. Otherwise depth = 1 + max
    of callee depths. Unknown callees (i.e. neither primitive nor in library)
    are treated as depth 0 for this computation; the library's add() will reject
    such a skill separately.
    """
    callees = skill.calls_set()
    skill_callees = [c for c in callees if c not in PRIMITIVE_NAMES]
    if not skill_callees:
        return 1
    max_callee_depth = 0
    for name in skill_callees:
        if name in library:
            max_callee_depth = max(max_callee_depth, library[name].depth)
    return 1 + max_callee_depth


def check_no_cycles(skill: Skill, library: SkillLibrary) -> None:
    """Verify that adding `skill` would not create a cycle in the call graph.

    DFS from `skill`'s callees through the existing library. If we ever reach
    `skill.name`, that's a cycle.
    """
    target = skill.name
    seen: set[str] = set()
    stack: list[str] = [
        c for c in skill.calls_set() if c not in PRIMITIVE_NAMES
    ]
    while stack:
        cur = stack.pop()
        if cur == target:
            raise CycleError(
                f"Adding skill {skill.name!r} would create a cycle "
                f"(reaches itself through {sorted(seen | {cur})})"
            )
        if cur in seen:
            continue
        seen.add(cur)
        if cur in library:
            for c in library[cur].calls_set():
                if c not in PRIMITIVE_NAMES and c not in seen:
                    stack.append(c)


def topological_sort(library: SkillLibrary) -> list[str]:
    """Return the active library skills in dependency order: leaves first,
    composites last. Useful for serialization and bulk-rebuild.
    """
    indeg: dict[str, int] = {}
    adj: dict[str, list[str]] = defaultdict(list)
    for s in library:
        if s.is_alias():
            continue
        indeg.setdefault(s.name, 0)
        for callee in s.calls_set():
            if callee in PRIMITIVE_NAMES or callee not in library:
                continue
            adj[callee].append(s.name)
            indeg[s.name] = indeg.get(s.name, 0) + 1
    queue = sorted(n for n, d in indeg.items() if d == 0)
    order: list[str] = []
    while queue:
        n = queue.pop(0)
        order.append(n)
        for succ in sorted(adj[n]):
            indeg[succ] -= 1
            if indeg[succ] == 0:
                queue.append(succ)
    if len(order) != sum(1 for _ in library if not _.is_alias()):
        # cycle remained (shouldn't happen if check_no_cycles fired on every add)
        raise CycleError("library still has a cycle at topological-sort time")
    return order


def composition_graph(library: SkillLibrary) -> dict[str, set[str]]:
    g: dict[str, set[str]] = {}
    for s in library:
        if s.is_alias():
            continue
        g[s.name] = {
            c for c in s.calls_set() if c not in PRIMITIVE_NAMES and c in library
        }
    return g


def depth_distribution(library: SkillLibrary) -> dict[int, int]:
    """Count of skills at each depth — drives the bottom panel of Figure 3."""
    cnt: Counter = Counter()
    for s in library:
        if s.is_alias():
            continue
        cnt[s.depth] += 1
    return dict(cnt)
