"""Plan-augmented training examples (§3.6).

Per §3.6:
    We convert each such trajectory into a plan-augmented training example
    (I, plan_σ, ζ), where plan_σ is the sequence of skill invocations the agent
    used.

So each example carries three things:
  - the instruction I (becomes the prompt),
  - the skill plan plan_σ (becomes the training target for the auxiliary loss),
  - the full action sequence ζ (becomes the training target for the main loss).

The dataset object renders each example into a (prompt, target) string pair that
the trainer's tokenizer turns into token ids. Keeping the rendering in the dataset
(rather than the trainer) keeps the trainer model-agnostic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Iterator, Optional

from scaffold.core.library import SkillLibrary
from scaffold.core.skill import Skill
from scaffold.core.trajectory import Action, ActionType, Trajectory


@dataclass
class PlanAugmentedExample:
    """One training instance for the distillation loss."""

    instruction: str
    plan: list[str]                      # plan_σ
    actions: list[Action]                # ζ (action sequence only; obs omitted)
    final_url: str = ""
    site: Optional[str] = None

    def to_prompt(self, library: SkillLibrary, *, top_k: int = 8) -> str:
        """Render (I, top-k retrieved skill signatures, history-so-far) as prompt."""
        retrieved = library.retrieve(self.instruction, k=top_k)
        skill_lines = "\n".join(
            f"- {s.name}({', '.join(p.signature_repr() for p in s.parameters)}): "
            f"{s.description}"
            for s in retrieved if not s.is_alias()
        ) or "(none)"
        return (
            f"# Instruction\n{self.instruction}\n"
            f"# Available skills\n{skill_lines}\n"
            f"# Plan and actions\n"
        )

    def to_main_target(self) -> str:
        """Cross-entropy target: the textual plan + action sequence."""
        plan_str = " -> ".join(self.plan) if self.plan else "(no skills)"
        action_strs = [
            f"{a.type.value}({json.dumps(a.args, ensure_ascii=False)})"
            + (f" [from {a.skill_call}]" if a.skill_call else "")
            for a in self.actions
        ]
        return f"plan: {plan_str}\nactions:\n" + "\n".join(action_strs)

    def to_aux_target(self) -> list[tuple[str, str]]:
        """Auxiliary next-skill-prediction examples.

        §3.6: "an auxiliary loss that predicts the next skill name conditioned
        only on (I, o_0:t)". We approximate this with a list of
        (history-prefix-as-text, next-skill-name) pairs, one per skill invocation.
        The trainer feeds each pair as a separate mini-example.
        """
        examples: list[tuple[str, str]] = []
        running_actions: list[Action] = []
        for a in self.actions:
            if a.type is ActionType.SKILL and a.skill_call:
                prefix = (
                    f"Instruction: {self.instruction}\n"
                    f"History: {self._render_history(running_actions)}\n"
                    f"Next skill:"
                )
                examples.append((prefix, a.skill_call))
            running_actions.append(a)
        return examples

    @staticmethod
    def _render_history(actions: list[Action]) -> str:
        if not actions:
            return "(none)"
        return "; ".join(
            f"{a.type.value}" + (f"({a.skill_call})" if a.skill_call else "")
            for a in actions[-10:]   # cap to keep prompt short
        )


def build_distillation_dataset(
    successful_trajectories: list[Trajectory],
    library: SkillLibrary,
) -> "PlanAugmentedDataset":
    """Construct the Z+_k dataset from §3.6."""
    examples: list[PlanAugmentedExample] = []
    for t in successful_trajectories:
        if not t.is_successful():
            continue
        examples.append(
            PlanAugmentedExample(
                instruction=t.instruction,
                plan=t.skill_plan(),
                actions=t.actions,
                final_url=t.observations[-1].url if t.observations else "",
                site=t.site,
            )
        )
    return PlanAugmentedDataset(examples=examples, library=library)


@dataclass
class PlanAugmentedDataset:
    """Iterable wrapper. Trainers consume this via __iter__."""

    examples: list[PlanAugmentedExample] = field(default_factory=list)
    library: Optional[SkillLibrary] = None

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self) -> Iterator[PlanAugmentedExample]:
        return iter(self.examples)

    def render_main(self) -> list[tuple[str, str]]:
        if self.library is None:
            return []
        return [(ex.to_prompt(self.library), ex.to_main_target()) for ex in self.examples]

    def render_aux(self) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        for ex in self.examples:
            out.extend(ex.to_aux_target())
        return out
