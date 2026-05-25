"""Baseline re-implementations for Table 1.

Three families, all running on the same Qwen2.5-VL-7B base for fair comparison
(§4.1.2):

  (A) Zero-shot agents:
    - ReAct (Yao et al., 2023)              `baselines.react.ReActBaseline`
    - SeeAct (Zheng et al., 2024a)          `baselines.seeact.SeeActBaseline`
    - WebVoyager (He et al., 2024)          `baselines.webvoyager.WebVoyagerBaseline`
    - AgentOccam (Yang et al., 2025)        `baselines.agentoccam.AgentOccamBaseline`

  (B) Memory/Workflow-augmented:
    - Reflexion (Shinn et al., 2023)        `baselines.reflexion.ReflexionBaseline`
    - Synapse (Zheng et al., 2024b)         `baselines.synapse.SynapseBaseline`
    - Mem0 (Chhikara et al., 2025)          `baselines.mem0.Mem0Baseline`
    - ExpeL (Zhao et al., 2024)             `baselines.expel.ExpELBaseline`
    - AWM (Wang et al., 2025c)              `baselines.awm.AWMBaseline`

  (C) Skill-induction & skill-RL:
    - WebRL (Qi et al., 2025)               `baselines.webrl.WebRLBaseline`
    - AppAgentX (Jiang et al., 2025)        `baselines.appagentx.AppAgentXBaseline`
    - EvolveR (Wu et al., 2025a)            `baselines.evolver.EvolveRBaseline`
    - SkillWeaver (Zheng et al., 2025)      `baselines.skillweaver.SkillWeaverBaseline`
    - SkillRL (Xia et al., 2026)            `baselines.skillrl.SkillRLBaseline`

Each baseline is a thin variant of `BaselineBase` that overrides one or more of:
  - `induce_skills` (e.g. SkillWeaver: single-instance; AWM: NL workflows; ...)
  - `compact_library` (most: no-op; SkillRL: two-tier split only)
  - `distill` (Reflexion: in-context only; WebRL: GRPO; ...)

This file just re-exports; each baseline lives in its own module so users can
read them in isolation.
"""

from scaffold.baselines.base import BaselineBase, BaselineConfig
from scaffold.baselines.skillweaver import SkillWeaverBaseline
from scaffold.baselines.skillrl import SkillRLBaseline
from scaffold.baselines.awm import AWMBaseline
from scaffold.baselines.react import ReActBaseline
from scaffold.baselines.reflexion import ReflexionBaseline
from scaffold.baselines.synapse import SynapseBaseline
from scaffold.baselines.simple_zero_shot import (
    SeeActBaseline,
    WebVoyagerBaseline,
    AgentOccamBaseline,
    Mem0Baseline,
    ExpELBaseline,
    WebRLBaseline,
    AppAgentXBaseline,
    EvolveRBaseline,
)

__all__ = [
    "BaselineBase",
    "BaselineConfig",
    "SkillWeaverBaseline",
    "SkillRLBaseline",
    "AWMBaseline",
    "ReActBaseline",
    "ReflexionBaseline",
    "SynapseBaseline",
    "SeeActBaseline",
    "WebVoyagerBaseline",
    "AgentOccamBaseline",
    "Mem0Baseline",
    "ExpELBaseline",
    "WebRLBaseline",
    "AppAgentXBaseline",
    "EvolveRBaseline",
]
