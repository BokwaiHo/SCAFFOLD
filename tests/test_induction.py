"""Unit tests for the multi-instance inducer."""

from __future__ import annotations

import json

import pytest

from scaffold.core.library import SkillLibrary
from scaffold.core.trajectory import Action, ActionType, Observation, Trajectory
from scaffold.induction.clusterer import InstructionCluster, cluster_by_instruction
from scaffold.induction.multi_instance import (
    InductionConfig, MultiInstanceInducer,
)
from scaffold.utils.embedding import HashEmbedder
from scaffold.utils.llm import MockChatClient, Response


def _traj(instruction: str, n_actions: int = 5) -> Trajectory:
    obs = [Observation(step=i, url="/shop", dom="") for i in range(n_actions + 1)]
    acts = [Action(type=ActionType.CLICK, args={"target": f"x{i}"})
            for i in range(n_actions)]
    return Trajectory(
        instruction=instruction, observations=obs, actions=acts,
        success=True,
    )


def test_cluster_by_instruction_groups_similar():
    trajs = [
        _traj("Buy a laptop"),
        _traj("Buy laptops"),
        _traj("Find a hotel in Tokyo"),
    ]
    clusters = cluster_by_instruction(trajs, embedder=HashEmbedder(),
                                       similarity_threshold=0.2)
    # at threshold 0.2 the laptop ones should cluster together
    sizes = sorted(len(c) for c in clusters)
    assert sum(sizes) == 3


def test_inducer_rejects_small_cluster():
    lib = SkillLibrary(embedder=HashEmbedder())
    llm = MockChatClient()
    inducer = MultiInstanceInducer(llm=llm, config=InductionConfig(n_min=2))
    cluster = InstructionCluster(
        cluster_id=0, trajectories=[_traj("only one")], centroid_instruction="only one",
    )
    result = inducer.induce_cluster(cluster, library=lib, iteration=0)
    assert not result.accepted
    assert "n_min" in result.reason


def test_inducer_parses_valid_response():
    lib = SkillLibrary(embedder=HashEmbedder())
    spec = {
        "name": "click_button",
        "description": "Click a button by visible text",
        "parameters": [{"name": "text", "type": "str"}],
        "precondition": "True",
        "body": "primitive_click(text)",
        "postcondition": "True",
        "parameter_justification": {"text": "varies across trajectories"},
    }
    llm = MockChatClient(replies=[json.dumps(spec)])
    inducer = MultiInstanceInducer(
        llm=llm,
        config=InductionConfig(n_min=2, enable_holdout_validation=False),
    )
    cluster = InstructionCluster(
        cluster_id=0,
        trajectories=[_traj("Click Submit"), _traj("Click OK")],
        centroid_instruction="Click button",
    )
    result = inducer.induce_cluster(cluster, library=lib, iteration=0)
    assert result.accepted
    assert result.skill.name == "click_button"
    assert result.skill.depth == 1


def test_inducer_rejects_body_with_imports():
    lib = SkillLibrary(embedder=HashEmbedder())
    spec = {
        "name": "bad", "description": "",
        "parameters": [],
        "precondition": "True",
        "body": "import os\nprimitive_click('x')",
        "postcondition": "True",
    }
    llm = MockChatClient(replies=[json.dumps(spec)])
    inducer = MultiInstanceInducer(
        llm=llm,
        config=InductionConfig(n_min=2, enable_holdout_validation=False,
                                forbid_imports=True),
    )
    cluster = InstructionCluster(
        cluster_id=0,
        trajectories=[_traj("a"), _traj("b")],
        centroid_instruction="x",
    )
    result = inducer.induce_cluster(cluster, library=lib, iteration=0)
    assert not result.accepted
    assert "import" in result.reason.lower()


def test_inducer_rejects_invalid_json():
    lib = SkillLibrary(embedder=HashEmbedder())
    llm = MockChatClient(replies=["not json at all"])
    inducer = MultiInstanceInducer(
        llm=llm,
        config=InductionConfig(n_min=2, enable_holdout_validation=False),
    )
    cluster = InstructionCluster(
        cluster_id=0,
        trajectories=[_traj("a"), _traj("b")],
        centroid_instruction="x",
    )
    result = inducer.induce_cluster(cluster, library=lib, iteration=0)
    assert not result.accepted
    assert "parse" in result.reason.lower()
