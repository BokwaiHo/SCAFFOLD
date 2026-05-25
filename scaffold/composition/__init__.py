"""Stage 3: recursive composition & depth tracking (§3.4)."""

from scaffold.composition.recursive import (
    compute_depth,
    check_no_cycles,
    CycleError,
    topological_sort,
    composition_graph,
    depth_distribution,
)

__all__ = [
    "compute_depth",
    "check_no_cycles",
    "CycleError",
    "topological_sort",
    "composition_graph",
    "depth_distribution",
]
