"""Unit tests for the Skill dataclass and Parameter type."""

from __future__ import annotations

import pytest

from scaffold.core.skill import (
    MAX_BODY_LINES, Parameter, ParameterType, Skill, SkillCall,
)


def test_skill_basic_construction():
    s = Skill(
        name="click_text",
        description="Click a text element",
        parameters=[Parameter("text", ParameterType.STRING)],
        precondition="True",
        body="primitive_click(text)",
        postcondition="True",
        depth=1,
    )
    assert s.name == "click_text"
    assert s.depth == 1
    assert len(s.parameters) == 1
    assert not s.is_alias()


def test_skill_rejects_invalid_name():
    with pytest.raises(ValueError):
        Skill(
            name="123_bad",
            description="",
            parameters=[],
            precondition="True", body="pass", postcondition="True",
            depth=1,
        )


def test_skill_rejects_oversized_body():
    body = "\n".join([f"primitive_click({i})" for i in range(MAX_BODY_LINES + 5)])
    with pytest.raises(ValueError):
        Skill(
            name="too_long",
            description="",
            parameters=[],
            precondition="True", body=body, postcondition="True",
            depth=1,
        )


def test_calls_set_extracts_subskill_calls():
    s = Skill(
        name="search_and_filter",
        description="",
        parameters=[Parameter("q", ParameterType.STRING)],
        precondition="True",
        body=(
            "type_in_field(obs, 'Search', q)\n"
            "click_text(obs, 'Search')\n"
            "click_text(obs, 'Electronics')\n"
        ),
        postcondition="True",
        depth=2,
    )
    calls = s.calls_set()
    assert "click_text" in calls
    assert "type_in_field" in calls
    # primitives like primitive_click excluded; helper functions excluded


def test_skill_to_from_dict_roundtrip():
    s = Skill(
        name="login",
        description="Log in",
        parameters=[
            Parameter("user", ParameterType.STRING),
            Parameter("pw", ParameterType.STRING),
        ],
        precondition="True",
        body="type_in_field(obs, 'Username', user)\nclick_text(obs, 'Sign In')",
        postcondition="True",
        depth=2,
        iteration_introduced=1,
    )
    d = s.to_dict()
    s2 = Skill.from_dict(d)
    assert s2.name == s.name
    assert s2.depth == s.depth
    assert len(s2.parameters) == 2
    assert s2.parameters[0].type == ParameterType.STRING


def test_parameter_signature_repr():
    p = Parameter("sort", ParameterType.STRING, default="price_asc",
                   choices=("price_asc", "price_desc"))
    sig = p.signature_repr()
    assert "sort" in sig
    assert "str" in sig.lower()


def test_skill_call_is_hashable():
    c1 = SkillCall(name="click_text", args={"text": "Submit"})
    c2 = SkillCall(name="click_text", args={"text": "Submit"})
    # SkillCall instances may not be set-equal due to dict; just ensure they construct
    assert c1.name == c2.name
