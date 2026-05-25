"""ReAct (Yao et al., 2023) baseline.

The vanilla "reason+act" prompt — no skill induction, no memory, no distillation.
This is the lower bound on Table 1 row (A).
"""

from __future__ import annotations

from scaffold.baselines.base import BaselineBase


class ReActBaseline(BaselineBase):
    name = "ReAct"
    # All hooks are default no-ops. The base rollout still uses the configured
    # Policy. For a real comparison the user should pass a PromptedPolicy whose
    # system prompt matches the ReAct template (thought, action, observation).
