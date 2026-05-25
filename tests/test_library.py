"""Unit tests for SkillLibrary."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from scaffold.core.library import SkillLibrary
from scaffold.core.skill import Parameter, ParameterType, Skill
from scaffold.utils.embedding import HashEmbedder


def _make_skill(name: str, desc: str, depth: int = 1, body: str = "pass") -> Skill:
    return Skill(
        name=name, description=desc,
        parameters=[],
        precondition="True", body=body, postcondition="True",
        depth=depth,
    )


def test_library_add_and_lookup():
    lib = SkillLibrary(embedder=HashEmbedder())
    s = _make_skill("click_text", "click a text element")
    lib.add(s)
    assert "click_text" in lib
    assert lib["click_text"].name == "click_text"
    assert lib.total_skills() == 1


def test_library_rejects_duplicate():
    lib = SkillLibrary(embedder=HashEmbedder())
    lib.add(_make_skill("a", "a"))
    with pytest.raises(ValueError):
        lib.add(_make_skill("a", "duplicate"))


def test_library_alias_forwards_lookup():
    lib = SkillLibrary(embedder=HashEmbedder())
    lib.add(_make_skill("login_v1", "login v1"))
    lib.add(_make_skill("login_v2", "login v2"))
    lib.mark_alias("login_v2", "login_v1")
    # lookup of alias resolves to winner
    assert lib["login_v2"].name == "login_v1"
    # but the alias still appears in iteration via is_alias() filter
    aliases = [s for s in lib if s.is_alias()]
    assert any(s.name == "login_v2" for s in aliases)


def test_library_retrieve_orders_by_similarity():
    lib = SkillLibrary(embedder=HashEmbedder())
    lib.add(_make_skill("login", "log in to the website"))
    lib.add(_make_skill("search", "search for products"))
    lib.add(_make_skill("checkout", "complete payment"))
    results = lib.retrieve("user wants to log in", k=2)
    assert len(results) <= 2
    assert any(s.name == "login" for s in results)


def test_library_remove_refuses_invoked_skills():
    lib = SkillLibrary(embedder=HashEmbedder())
    lib.add(_make_skill("click_text", "click", body="pass"))
    parent = Skill(
        name="parent", description="",
        parameters=[],
        precondition="True", body="click_text(obs, 'x')", postcondition="True",
        depth=2,
    )
    lib.add(parent)
    with pytest.raises(ValueError):
        lib.remove("click_text")  # invoked by parent


def test_library_save_load_roundtrip():
    lib = SkillLibrary(embedder=HashEmbedder())
    lib.add(_make_skill("a", "first"))
    lib.add(_make_skill("b", "second"))
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "lib.json")
        lib.save(p)
        lib2 = SkillLibrary.load(p, embedder=HashEmbedder())
    assert lib2.total_skills() == 2
    assert "a" in lib2 and "b" in lib2


def test_library_copy_is_independent():
    lib = SkillLibrary(embedder=HashEmbedder())
    lib.add(_make_skill("a", "a"))
    copy = lib.copy()
    copy.add(_make_skill("b", "b"))
    assert "b" in copy and "b" not in lib


def test_library_stats():
    lib = SkillLibrary(embedder=HashEmbedder())
    lib.add(_make_skill("s1", "x", depth=1))
    lib.add(_make_skill("s2", "x", depth=2))
    lib.add(_make_skill("s3", "x", depth=3))
    assert lib.mean_depth() == pytest.approx(2.0)
    assert lib.max_depth() == 3
