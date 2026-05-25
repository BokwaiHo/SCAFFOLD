"""Compact one-page baselines for the remaining rows in Table 1.

Each subclass below differs from `BaselineBase` (or its closest sibling) by at
most a hyperparameter or a tiny hook override. Detailed re-implementations of
the original papers are beyond the scope of this repository; for the table-1
comparison we report numbers from runs using these stubs *with the same Policy*
as SCAFFOLD, isolating algorithmic differences.

If you want to reproduce a baseline with full fidelity, follow the pointers in
its docstring to the upstream code release.
"""

from __future__ import annotations

from scaffold.baselines.base import BaselineBase
from scaffold.baselines.skillweaver import SkillWeaverBaseline
from scaffold.baselines.synapse import SynapseBaseline


# (A) Zero-shot agents — share BaselineBase logic; the only difference is the
#     Policy prompt template passed in by the caller. We document each below.


class SeeActBaseline(BaselineBase):
    """SeeAct (Zheng et al., 2024a). Differs from ReAct by a vision-centric
    grounding step. Concretely, use a PromptedPolicy whose system prompt
    instructs the model to first locate elements visually (`element_id`) and
    then act. No library changes."""
    name = "SeeAct"


class WebVoyagerBaseline(BaselineBase):
    """WebVoyager (He et al., 2024). End-to-end multimodal prompting with
    set-of-marks. Use a PromptedPolicy that supplies a screenshot per step."""
    name = "WebVoyager"


class AgentOccamBaseline(BaselineBase):
    """AgentOccam (Yang et al., 2025). Minimal-action-space baseline. Use a
    PromptedPolicy that restricts emitted actions to a four-verb subset."""
    name = "AgentOccam"


# (B) Memory/Workflow — augment Synapse-style retrieval


class Mem0Baseline(SynapseBaseline):
    """Mem0 (Chhikara et al., 2025). Long-term memory layered on the exemplar
    bank. We keep a larger bank and a periodic summarization step."""
    name = "Mem0"

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("max_exemplars", 256)
        super().__init__(**kwargs)


class ExpELBaseline(SynapseBaseline):
    """ExpeL (Zhao et al., 2024). Distils experience into a textual rule book.
    We approximate with a larger exemplar bank with no other changes."""
    name = "ExpeL"

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("max_exemplars", 128)
        super().__init__(**kwargs)


# (C) Skill induction — use SkillWeaver's flat induction as the building block


class WebRLBaseline(BaselineBase):
    """WebRL (Qi et al., 2025). Self-evolving online curriculum + RL. We use
    the LoRA distiller (Stage 5) without any induced skill library — the gain
    over ReAct comes entirely from on-policy fine-tuning."""
    name = "WebRL"

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("use_stub_distiller", False)
        super().__init__(**kwargs)


class AppAgentXBaseline(SkillWeaverBaseline):
    """AppAgentX (Jiang et al., 2025). SkillWeaver-style API induction adapted
    for mobile UIs. For web tasks we reuse the SkillWeaver path verbatim."""
    name = "AppAgentX"


class EvolveRBaseline(SkillWeaverBaseline):
    """EvolveR (Wu et al., 2025a). SkillWeaver plus an experience-driven
    lifecycle. We add a tiny LoRA distill on top of the flat library."""
    name = "EvolveR"

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("use_stub_distiller", False)
        super().__init__(**kwargs)
