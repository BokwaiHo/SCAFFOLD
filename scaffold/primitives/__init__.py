"""Primitive actions A_prim = {click, type, scroll, wait}, per §3.1.

These are the leaf nodes of every skill body: any skill at depth d=1 directly emits
only primitive calls; depth > 1 skills mix primitives with calls to lower-level
skills. The set is small *by design* — keeping A_prim small is what lets the inducer
produce compact, transferable skills.
"""

from scaffold.primitives.actions import (
    PRIMITIVE_NAMES,
    primitive_click,
    primitive_type,
    primitive_scroll,
    primitive_wait,
    PrimitiveContext,
)

__all__ = [
    "PRIMITIVE_NAMES",
    "primitive_click",
    "primitive_type",
    "primitive_scroll",
    "primitive_wait",
    "PrimitiveContext",
]
