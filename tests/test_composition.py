"""Unit tests for the composition module (Eq. 1 + cycle check)."""

from __future__ import annotations

import pytest

from scaffold.composition.recursive import (
    CycleError,
    check_no_cycles,
    compute_depth,
    depth_distribution,
    topological_sort,
)
from scaffold.core.library import SkillLibrary
from scaffold.core.skill import Parameter, ParameterType, Skill
from scaffold.utils.embedding import HashEmbedder


def _skill(name: str, body: str = "pass", depth: int = 1) -> Skill:
    return Skill(
        name=name, description="", parameters=[],
        precondition="True", body=body, postcondition="True",
        depth=depth,
    )


def test_compute_depth_primitives_only_is_one():
    lib = SkillLibrary(embedder=HashEmbedder())
    s = _skill("click_text", body="primitive_click('x')")
    # don't add to lib yet — compute_depth doesn't need it
    assert compute_depth(s, lib) == 1


def test_compute_depth_recursive():
    """Eq. (1): d = 1 + max{d_callee}."""
    lib = SkillLibrary(embedder=HashEmbedder())
    lib.add(_skill("click_text", body="primitive_click('x')", depth=1))
    lib.add(_skill("type_in_field",
                    body="primitive_click('f')\nprimitive_type('val')", depth=1))

    mid = _skill(
        "search_and_filter",
        body="type_in_field(obs, 'Search', q)\nclick_text(obs, 'Search')",
        depth=1,  # placeholder; compute_depth resolves
    )
    d = compute_depth(mid, lib)
    assert d == 2  # 1 + max(1, 1)


def test_compute_depth_three_levels():
    """Worked App. B example: depth-3 skill calls a depth-2 and depth-1 skill."""
    lib = SkillLibrary(embedder=HashEmbedder())
    lib.add(_skill("click_text", body="primitive_click('x')", depth=1))
    lib.add(_skill("type_in_field",
                    body="primitive_click('f')\nprimitive_type('val')", depth=1))
    saf = _skill(
        "search_and_filter",
        body="type_in_field(obs, 'Search', q)\nclick_text(obs, 'Search')",
        depth=2,
    )
    lib.add(saf)
    composite = _skill(
        "checkout_cheapest",
        body="search_and_filter(obs, q, c)\nclick_text(obs, 'Add to Cart')",
        depth=1,
    )
    assert compute_depth(composite, lib) == 3


def test_check_no_cycles_passes_dag():
    lib = SkillLibrary(embedder=HashEmbedder())
    lib.add(_skill("leaf", body="primitive_click('x')"))
    parent = _skill("parent", body="leaf(obs)")
    check_no_cycles(parent, lib)  # no raise


def test_check_no_cycles_detects_direct_cycle():
    # We bypass library.add()'s strict known-callee check via direct injection
    # so we can stage a graph foo → bar and then propose bar → foo (cycle).
    lib = SkillLibrary(embedder=HashEmbedder())
    lib._skills["foo"] = _skill("foo", body="bar(obs)")
    bar_cycle = _skill("bar", body="foo(obs)")
    with pytest.raises(CycleError):
        check_no_cycles(bar_cycle, lib)


def test_check_no_cycles_detects_transitive_cycle():
    lib = SkillLibrary(embedder=HashEmbedder())
    lib._skills["a"] = _skill("a", body="b(obs)")
    lib._skills["b"] = _skill("b", body="c(obs)")
    c = _skill("c", body="a(obs)")
    with pytest.raises(CycleError):
        check_no_cycles(c, lib)


def test_topological_sort_orders_leaves_first():
    lib = SkillLibrary(embedder=HashEmbedder())
    lib.add(_skill("leaf", body="primitive_click('x')", depth=1))
    lib.add(_skill("mid", body="leaf(obs)", depth=2))
    lib.add(_skill("top", body="mid(obs)\nleaf(obs)", depth=3))
    order = topological_sort(lib)
    assert order.index("leaf") < order.index("mid") < order.index("top")


def test_depth_distribution_counts_correctly():
    lib = SkillLibrary(embedder=HashEmbedder())
    lib.add(_skill("a", depth=1))
    lib.add(_skill("b", depth=1))
    lib.add(_skill("c", depth=2))
    dist = depth_distribution(lib)
    assert dist == {1: 2, 2: 1}
