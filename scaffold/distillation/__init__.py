"""Stage 5: distill skill-augmented behavior into π_{k+1} (§3.6).

LoRA fine-tunes the base policy on (instruction, plan_σ, ζ) triples with token-level
cross-entropy and an auxiliary next-skill-name prediction loss.
"""

from scaffold.distillation.dataset import (
    PlanAugmentedExample,
    PlanAugmentedDataset,
    build_distillation_dataset,
)
from scaffold.distillation.trainer import (
    DistillationConfig,
    LoRADistiller,
    StubDistiller,
)

__all__ = [
    "PlanAugmentedExample",
    "PlanAugmentedDataset",
    "build_distillation_dataset",
    "DistillationConfig",
    "LoRADistiller",
    "StubDistiller",
]
