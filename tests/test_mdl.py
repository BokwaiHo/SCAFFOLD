"""Unit tests for the MDL compactor (§3.5).

Two properties are critical:
  1. F(L) computed by mdl_objective is non-negative and consistent
     (library cost + data cost).
  2. Each operator, when accepted, strictly reduces F.
  3. The validation-SR guard rolls back any accept that drops SR below baseline.
"""

from __future__ import annotations

import pytest

from scaffold.compaction.merge import MergeOperator
from scaffold.compaction.mdl import (
    MDLCompactor, MDLConfig, mdl_objective, parse_trajectory_under_library,
)
from scaffold.compaction.prune import PruneOperator
from scaffold.core.library import SkillLibrary
from scaffold.core.skill import Skill
from scaffold.core.trajectory import Action, ActionType, Observation, Trajectory
from scaffold.utils.embedding import HashEmbedder


def _skill(name: str, body: str, depth: int = 1) -> Skill:
    return Skill(
        name=name, description="", parameters=[],
        precondition="True", body=body, postcondition="True",
        depth=depth,
    )


def _traj(action_types: list[ActionType]) -> Trajectory:
    obs = [Observation(step=i, url="/", dom="") for i in range(len(action_types) + 1)]
    acts = [Action(type=t, args={}) for t in action_types]
    return Trajectory(instruction="test", observations=obs, actions=acts,
                       success=True)


def test_mdl_objective_non_negative():
    lib = SkillLibrary(embedder=HashEmbedder())
    lib.add(_skill("a", body="primitive_click('x')"))
    t = _traj([ActionType.CLICK, ActionType.TYPE])
    F, lc, dc = mdl_objective(lib, [t])
    assert F == lc + dc
    assert F >= 0


def test_mdl_objective_decreases_when_skill_covers_subseq():
    """A skill whose body matches a recurring subsequence should reduce data cost."""
    lib_empty = SkillLibrary(embedder=HashEmbedder())
    lib_full = SkillLibrary(embedder=HashEmbedder())
    lib_full.add(_skill(
        "click_then_type",
        body="primitive_click('x')\nprimitive_type('val')",
    ))
    t = _traj([ActionType.CLICK, ActionType.TYPE,
               ActionType.CLICK, ActionType.TYPE])
    _, _, dc_empty = mdl_objective(lib_empty, [t])
    _, _, dc_full = mdl_objective(lib_full, [t])
    assert dc_full < dc_empty


def test_prune_removes_unused_skills():
    lib = SkillLibrary(embedder=HashEmbedder())
    lib.add(_skill("unused", body="primitive_click('x')"))
    op = PruneOperator()
    candidates = list(op.propose(lib, protected=set()))
    assert len(candidates) == 1
    candidates[0].apply(lib)
    assert "unused" not in lib


def test_prune_protects_recently_used_skills():
    lib = SkillLibrary(embedder=HashEmbedder())
    lib.add(_skill("recent", body="primitive_click('x')"))
    op = PruneOperator()
    cands = list(op.propose(lib, protected={"recent"}))
    assert len(cands) == 0


def test_merge_proposes_static_equivalent_skills():
    lib = SkillLibrary(embedder=HashEmbedder())
    lib.add(_skill("login_v1", body="primitive_type('user')\nprimitive_click('Sign In')"))
    lib.add(_skill("login_v2", body="primitive_type('user2')\nprimitive_click('Sign In')"))
    # bodies differ only in string literals — _normalized_body should equate them
    op = MergeOperator(rho=0.9)
    cands = list(op.propose(lib))
    assert len(cands) >= 1
    cands[0].apply(lib)
    # one should now be an alias
    aliased = [s for s in lib if s.is_alias()]
    assert len(aliased) == 1


def test_compactor_accepts_only_F_decreasing_changes():
    lib = SkillLibrary(embedder=HashEmbedder())
    lib.add(_skill("dup_a", body="primitive_click('x')"))
    lib.add(_skill("dup_b", body="primitive_click('x')"))
    # provide a trajectory so data cost is nonzero
    trajs = [_traj([ActionType.CLICK])]
    compactor = MDLCompactor(MDLConfig(max_iterations=3,
                                        require_validation_no_drop=False))
    F_before, _, _ = mdl_objective(lib, trajs)
    compactor.apply(lib, positive_trajectories=trajs)
    F_after, _, _ = mdl_objective(lib, trajs)
    assert F_after <= F_before


def test_compactor_rolls_back_when_validation_drops():
    lib = SkillLibrary(embedder=HashEmbedder())
    lib.add(_skill("dup_a", body="primitive_click('x')"))
    lib.add(_skill("dup_b", body="primitive_click('x')"))
    trajs = [_traj([ActionType.CLICK])]
    # validation_sr always returns 0.0; any change drops below baseline = first call
    calls = {"n": 0}
    def fake_sr(lib):
        calls["n"] += 1
        # first call (baseline) returns 1.0; later calls return 0.0
        return 1.0 if calls["n"] == 1 else 0.0
    compactor = MDLCompactor(MDLConfig(max_iterations=2,
                                        require_validation_no_drop=True))
    n_before = lib.total_skills()
    compactor.apply(lib, positive_trajectories=trajs, validation_sr=fake_sr)
    # nothing should have been accepted
    assert lib.total_skills() == n_before


def test_parse_trajectory_under_library_is_monotone():
    """Adding a skill that covers the trajectory shouldn't increase the parse length."""
    lib_empty = SkillLibrary(embedder=HashEmbedder())
    lib_with = SkillLibrary(embedder=HashEmbedder())
    lib_with.add(_skill("cl_ty", body="primitive_click('x')\nprimitive_type('y')"))
    t = _traj([ActionType.CLICK, ActionType.TYPE])
    p_empty = parse_trajectory_under_library(t, lib_empty)
    p_with = parse_trajectory_under_library(t, lib_with)
    assert p_with.length <= p_empty.length
