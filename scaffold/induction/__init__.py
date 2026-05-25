"""Stage 2: multi-instance skill induction (§3.3 of the paper)."""

from scaffold.induction.multi_instance import (
    MultiInstanceInducer,
    InductionConfig,
    InductionResult,
)
from scaffold.induction.clusterer import (
    cluster_by_instruction,
    InstructionCluster,
)
from scaffold.induction.prompts import (
    SKILL_INDUCER_SYSTEM,
    skill_inducer_user_prompt,
)

__all__ = [
    "MultiInstanceInducer",
    "InductionConfig",
    "InductionResult",
    "cluster_by_instruction",
    "InstructionCluster",
    "SKILL_INDUCER_SYSTEM",
    "skill_inducer_user_prompt",
]
