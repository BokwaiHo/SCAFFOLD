"""Prompt templates for the skill inducer (App. A of the paper).

Reproduced almost verbatim from the paper, with explicit JSON-output constraints
added so we can parse the response reliably. The "justify parameters" diagnostic
that App. A mentions ("the model is then asked to justify its choice of parameters
by pointing to specific token positions in each trajectory that vary") is folded
into the same JSON envelope as a `parameter_justification` field.
"""

from __future__ import annotations

import json
from typing import Iterable

from scaffold.core.skill import Skill
from scaffold.core.trajectory import Trajectory


SKILL_INDUCER_SYSTEM = (
    "You are a senior software engineer who specializes in writing concise, "
    "reusable agent skills. You will be given N successful trajectories that "
    "accomplish similar goals on a website. Your job is to abstract them into a "
    "single PARAMETRIC, EXECUTABLE skill by identifying which positions vary "
    "across trajectories and replacing them with typed parameters.\n\n"
    "Output STRICT JSON only, matching this schema (no markdown fences):\n"
    "{\n"
    '  "name": "snake_case_identifier",\n'
    '  "description": "one-line description used for embedding-keyed retrieval",\n'
    '  "parameters": [{"name": "...", "type": "str|int|float|bool|url|selector|enum", '
    '"default": null, "choices": null, "description": "..."}],\n'
    '  "precondition": "Python expression over obs, e.g. \\"\'shop\' in obs.url\\"",\n'
    '  "body": "Python statements; one per line; may call primitives '
    "{click, type, scroll, wait, find, find_input} or any AVAILABLE SKILL.\",\n"
    '  "postcondition": "Python expression over obs",\n'
    '  "parameter_justification": {"<param>": "tokens that varied across trajectories"}\n'
    "}\n\n"
    "Hard constraints:\n"
    "- body must be a valid Python suite with at most 30 non-blank lines\n"
    "- body MUST NOT define new functions or classes\n"
    "- body MUST NOT import anything\n"
    "- prefer semantic locators ('the Search input') over hard-coded ids\n"
    "- if a sub-procedure matches an AVAILABLE SKILL exactly, call it instead of re-emitting primitives\n"
)


def _format_trajectory(t: Trajectory, idx: int, max_actions: int = 20) -> str:
    lines = [f"--- Trajectory {idx} ---", f"Instruction: {t.instruction}"]
    if t.site:
        lines.append(f"Site: {t.site}")
    lines.append("Actions:")
    for i, a in enumerate(t.actions[:max_actions]):
        skill_tag = f" [skill={a.skill_call}]" if a.skill_call else ""
        lines.append(f"  {i}: {a.type.value} {json.dumps(a.args, ensure_ascii=False)}{skill_tag}")
    if len(t.actions) > max_actions:
        lines.append(f"  ... ({len(t.actions) - max_actions} more)")
    # one observation hint at the start and end is plenty without screenshots
    if t.observations:
        lines.append(f"Initial URL: {t.observations[0].url}")
        lines.append(f"Final URL: {t.observations[-1].url}")
    return "\n".join(lines)


def _format_available_skills(skills: Iterable[Skill]) -> str:
    rows: list[str] = []
    for s in skills:
        if s.is_alias():
            continue
        params = ", ".join(p.signature_repr() for p in s.parameters)
        rows.append(f"- {s.name}({params})  # d={s.depth}: {s.description}")
    return "\n".join(rows) if rows else "(none)"


def skill_inducer_user_prompt(
    trajectories: list[Trajectory],
    available_skills: Iterable[Skill],
    *,
    centroid_instruction: str = "",
) -> str:
    """Render the full user prompt for the inducer."""
    parts = []
    if centroid_instruction:
        parts.append(f"Centroid instruction: {centroid_instruction}")
    parts.append(f"You are given {len(trajectories)} successful trajectories.")
    parts.extend(_format_trajectory(t, i + 1) for i, t in enumerate(trajectories))
    parts.append("Available existing skills (you may call them as sub-routines):")
    parts.append(_format_available_skills(available_skills))
    parts.append(
        "Now output the JSON. Choose parameters that are TRULY variable across "
        "the trajectories shown — do not parameterize constants."
    )
    return "\n\n".join(parts)
