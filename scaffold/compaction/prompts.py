"""Refactor proposer prompt (App. A).

Paper text:
    The following action subsequence appears in m existing skills (shown below).
    Decide whether to refactor it into a new mid-level skill. If yes, name it,
    parametrize it, and rewrite all m existing skills to call the new one. If
    no, briefly explain why (e.g., the subsequence is too short or too
    site-specific to be reusable).
"""

from __future__ import annotations

from scaffold.core.skill import Skill


REFACTOR_PROPOSER_SYSTEM = (
    "You are a careful library architect. You will be shown a recurring "
    "primitive-action subsequence that appears across several existing skills. "
    "Decide whether the subsequence deserves promotion to its own mid-level "
    "skill. Promotion is appropriate when the subsequence is: (a) at least 3 "
    "primitives long, (b) re-used by at least 3 skills, and (c) parametrically "
    "uniform across uses.\n\n"
    "Reply with STRICT JSON only (no fences):\n"
    "{\n"
    '  "decision": "refactor" | "skip",\n'
    '  "reason": "...",\n'
    '  "new_skill": {  // only if decision=="refactor"\n'
    '    "name": "snake_case",\n'
    '    "description": "...",\n'
    '    "parameters": [{"name": "...", "type": "str|int|float|bool|url|selector|enum"}],\n'
    '    "precondition": "Python expression over obs",\n'
    '    "body": "Python statements; primitives or earlier skills only; <= 30 lines",\n'
    '    "postcondition": "Python expression over obs"\n'
    "  },\n"
    '  "rewrites": {  // skill_name -> new_body\n'
    '    "<existing_skill_name>": "...",\n'
    "    ...\n"
    "  }\n"
    "}\n\n"
    "Every `rewrites` body must call the new skill exactly once in place of the "
    "old subsequence and otherwise leave each parent's body unchanged."
)


def refactor_proposer_user_prompt(
    *,
    subsequence: tuple[str, ...],
    skills: list[Skill],
) -> str:
    lines = [
        f"Recurring subsequence (length {len(subsequence)}): "
        f"{' -> '.join(subsequence)}",
        f"\nThe subsequence appears in {len(skills)} existing skills:",
    ]
    for s in skills:
        params = ", ".join(p.signature_repr() for p in s.parameters)
        lines.append(f"\n--- skill {s.name}({params})  (depth {s.depth}) ---")
        lines.append(f"description: {s.description}")
        lines.append("body:")
        for ln in s.body.splitlines():
            lines.append(f"  {ln}")
    lines.append(
        "\nNow decide whether to refactor and, if so, emit the JSON. "
        "Be conservative: skip if the subsequence varies meaningfully in args "
        "across the skills, or if a one-line helper would be clearer than a new skill."
    )
    return "\n".join(lines)
