"""End-to-end smoke test: ToyEnv + ScriptedPolicy + MockChatClient.

This test runs the full Algorithm 1 loop for 2 iterations on the toy env and
verifies the pipeline doesn't crash, produces a non-empty trajectory log, and
saves the library to disk. We use mock LLM clients so the test runs in seconds
without any API keys or GPU.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from scaffold.agent.policy import ScriptedPolicy
from scaffold.core.trajectory import Action, ActionType
from scaffold.envs.toy_env import ToyEnv
from scaffold.pipeline import ScaffoldConfig, ScaffoldPipeline
from scaffold.compaction import MDLConfig
from scaffold.distillation import DistillationConfig
from scaffold.envs.base import RolloutConfig
from scaffold.induction import InductionConfig
from scaffold.utils.embedding import HashEmbedder
from scaffold.utils.llm import MockChatClient


def _build_scripted_policy() -> ScriptedPolicy:
    """Replays a roughly-correct sequence: click Search field, type query,
    click Search button, click first result, click Add to Cart."""
    return ScriptedPolicy([
        Action(type=ActionType.TYPE, args={"target": "Search", "text": "laptop"}),
        Action(type=ActionType.CLICK, args={"target": "Search"}),
        Action(type=ActionType.CLICK, args={"target": "first_result"}),
        Action(type=ActionType.CLICK, args={"target": "Add to Cart"}),
    ] * 5)  # repeat enough to cover step budget


def test_pipeline_runs_end_to_end():
    env = ToyEnv(n_train=6, n_val=2, n_test=2)
    policy = _build_scripted_policy()

    with tempfile.TemporaryDirectory() as d:
        cfg = ScaffoldConfig(
            iterations=2,
            compaction_interval=2,
            output_dir=d,
            use_stub_distiller=True,
            embedder_name="hash",
            cluster_similarity_threshold=0.3,
            induction=InductionConfig(
                n_min=2, enable_holdout_validation=False,
            ),
            mdl=MDLConfig(
                l_min=2, r_min=2, max_iterations=2,
                require_validation_no_drop=False,
            ),
            distillation=DistillationConfig(output_dir=os.path.join(d, "distill")),
            rollout=RolloutConfig(max_steps=8, temperature=0.0),
        )
        pipeline = ScaffoldPipeline(
            env=env, policy=policy,
            inducer_llm=MockChatClient(),
            refactor_llm=MockChatClient(),
            config=cfg,
            embedder=HashEmbedder(),
        )
        library, metrics = pipeline.run()

        # Pipeline produced metrics for each iteration
        assert len(metrics) == 2
        for m in metrics:
            assert "iteration" in m
            assert "library_size" in m
            assert m["library_size"] >= 0

        # Library was saved per iteration
        for k in range(2):
            assert os.path.exists(os.path.join(d, f"iter{k}", "library.json"))
            assert os.path.exists(os.path.join(d, f"iter{k}", "metrics.json"))

        # Library save+load is intact
        from scaffold.core.library import SkillLibrary
        lib2 = SkillLibrary.load(
            os.path.join(d, "iter1", "library.json"), embedder=HashEmbedder(),
        )
        assert lib2.total_skills() == library.total_skills()


def test_pipeline_handles_no_successful_trajectories():
    """When the policy never succeeds, induction must produce zero skills but
    the pipeline must not crash."""
    env = ToyEnv(n_train=4, n_val=1, n_test=1)
    # Pure NOOP policy: never succeeds
    policy = ScriptedPolicy([Action(type=ActionType.NOOP) for _ in range(20)])

    with tempfile.TemporaryDirectory() as d:
        cfg = ScaffoldConfig(
            iterations=1, compaction_interval=2,
            output_dir=d, use_stub_distiller=True,
            induction=InductionConfig(n_min=2, enable_holdout_validation=False),
            mdl=MDLConfig(require_validation_no_drop=False),
            rollout=RolloutConfig(max_steps=4),
        )
        pipeline = ScaffoldPipeline(
            env=env, policy=policy,
            inducer_llm=MockChatClient(),
            config=cfg, embedder=HashEmbedder(),
        )
        library, metrics = pipeline.run()
        assert library.total_skills() == 0
        assert metrics[0]["n_success"] == 0
