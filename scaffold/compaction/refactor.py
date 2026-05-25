"""Refactor operator (§3.5).

Paper:
    Refactor: if the same primitive-action subsequence of length ≥ ℓ_min appears
    in ≥ r_min existing skill bodies, the inducer is asked to propose a new
    mid-level skill that captures it, and the existing skills are rewritten to
    call it.

Implementation steps:
  1. For every active skill, extract its primitive-action subsequence (using
     `_extract_primitive_calls_from_body` from the MDL module).
  2. Enumerate every contiguous sub-window of length ≥ ℓ_min and count how
     many distinct skill bodies contain it (as a contiguous slice).
  3. For every subsequence with count ≥ r_min, build a RefactorCandidate that
     proposes a new skill σ_new and substitutes σ_new(<args>) into each parent
     skill's body.

The actual LLM call (App. A refactor-proposer prompt) is performed by the
candidate's `apply` method when an LLM client has been provided. When no LLM is
configured (e.g. in offline tests), we fall back to a deterministic name and a
synthetic body that just re-emits the primitive subsequence.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional

from scaffold.compaction.mdl import _extract_primitive_calls_from_body
from scaffold.compaction.prompts import (
    REFACTOR_PROPOSER_SYSTEM,
    refactor_proposer_user_prompt,
)
from scaffold.core.library import SkillLibrary
from scaffold.core.skill import Skill, Parameter, ParameterType
from scaffold.utils.llm import ChatMessage, LLMClient
from scaffold.utils.logging import get_logger

log = get_logger("mdl.refactor")


@dataclass
class RefactorCandidate:
    """A proposal to refactor a recurring subsequence into a new mid-level skill."""

    subsequence: tuple[str, ...]
    parent_skill_names: list[str]
    new_skill_name: str = ""
    # populated by `apply()` after the LLM (or fallback) runs
    new_skill: Optional[Skill] = None
    rewrites: dict[str, str] = field(default_factory=dict)  # parent -> new body

    llm: Optional[LLMClient] = None

    def apply(self, library: SkillLibrary) -> None:
        if self.new_skill is None:
            self._populate(library)
        # 1. Add the new mid-level skill (may raise CycleError if malformed).
        library.add(self.new_skill)
        # 2. Rewrite each parent skill's body to call the new skill.
        for parent_name, new_body in self.rewrites.items():
            old = library[parent_name]
            updated = Skill(
                name=old.name,
                description=old.description,
                parameters=old.parameters,
                precondition=old.precondition,
                body=new_body,
                postcondition=old.postcondition,
                depth=old.depth,            # recomputed below
                iteration_introduced=old.iteration_introduced,
                times_used=old.times_used,
                induced_from_cluster_size=old.induced_from_cluster_size,
                is_alias_of=None,
                parent_skills=tuple(list(old.parent_skills) + [self.new_skill.name]),
            )
            library._skills[parent_name] = updated
        # 3. Recompute depths for every affected parent (and transitive ancestors).
        from scaffold.composition.recursive import compute_depth, topological_sort

        for n in topological_sort(library):
            s = library[n]
            new_d = compute_depth(s, library)
            if new_d != s.depth:
                library._skills[n] = Skill(
                    **{**s.to_dict(), "depth": new_d}
                ) if False else _with_depth(s, new_d)

    # ---- internals --------------------------------------------------------

    def _populate(self, library: SkillLibrary) -> None:
        """Either call the LLM refactor proposer (App. A) or build a deterministic
        skill from the subsequence as a fallback."""
        if self.llm is None:
            self._populate_offline(library)
            return
        try:
            self._populate_llm(library)
        except Exception as e:
            log.warning("LLM refactor proposer failed: %r; falling back", e)
            self._populate_offline(library)

    def _populate_llm(self, library: SkillLibrary) -> None:
        parents = [library[n] for n in self.parent_skill_names]
        msgs = [
            ChatMessage("system", REFACTOR_PROPOSER_SYSTEM),
            ChatMessage("user", refactor_proposer_user_prompt(
                subsequence=self.subsequence, skills=parents,
            )),
        ]
        resp = self.llm.chat(msgs, temperature=0.3, max_tokens=1500)
        text = resp.text.strip()
        if text.startswith("```"):
            text = "\n".join(text.splitlines()[1:-1])
        spec = json.loads(text)
        if spec.get("decision") != "refactor":
            raise ValueError(f"proposer declined: {spec.get('reason', '')}")
        new_spec = spec["new_skill"]
        self.new_skill_name = new_spec["name"]
        self.new_skill = Skill(
            name=new_spec["name"],
            description=new_spec.get("description", "refactored mid-level skill"),
            parameters=[
                Parameter(
                    name=p["name"],
                    type=ParameterType(p.get("type", "str")),
                    default=p.get("default"),
                )
                for p in new_spec.get("parameters", [])
            ],
            precondition=new_spec.get("precondition", "True"),
            body=new_spec.get("body", "pass"),
            postcondition=new_spec.get("postcondition", "True"),
            depth=1,
            iteration_introduced=max(p.iteration_introduced for p in parents) + 0,
        )
        # depth set lazily by composition module on add
        self.rewrites = dict(spec["rewrites"])

    def _populate_offline(self, library: SkillLibrary) -> None:
        # synthesize a deterministic name + body from the subsequence
        if not self.new_skill_name:
            self.new_skill_name = "refactored_" + "_".join(self.subsequence)
            # ensure uniqueness
            i = 1
            base = self.new_skill_name
            while self.new_skill_name in library:
                self.new_skill_name = f"{base}_{i}"
                i += 1
        body_lines = [f"primitive_{p}()" for p in self.subsequence
                      if p in ("click", "type", "scroll", "wait")]
        body = "\n".join(body_lines) or "pass"
        self.new_skill = Skill(
            name=self.new_skill_name,
            description=f"Refactored subsequence: {' -> '.join(self.subsequence)}",
            parameters=[],
            precondition="True",
            body=body,
            postcondition="True",
            depth=1,
        )
        # Each parent: replace first occurrence of the subsequence with one call.
        # We do a per-line splice that approximates what the LLM proposer would emit.
        for parent_name in self.parent_skill_names:
            parent = library[parent_name]
            new_body = _splice_subsequence(parent.body, self.subsequence,
                                            self.new_skill_name)
            self.rewrites[parent_name] = new_body

    def __repr__(self) -> str:
        return (
            f"RefactorCandidate(subseq={'/'.join(self.subsequence)}, "
            f"parents={len(self.parent_skill_names)})"
        )


def _with_depth(s: Skill, d: int) -> Skill:
    return Skill(
        name=s.name, description=s.description, parameters=s.parameters,
        precondition=s.precondition, body=s.body, postcondition=s.postcondition,
        depth=d, iteration_introduced=s.iteration_introduced,
        times_used=s.times_used, induced_from_cluster_size=s.induced_from_cluster_size,
        is_alias_of=s.is_alias_of, parent_skills=s.parent_skills,
    )


def _splice_subsequence(body: str, subseq: tuple[str, ...], new_call: str) -> str:
    """Replace the first occurrence of the primitive subsequence in `body` with
    a single call to `new_call`. Best-effort line-based splice."""
    lines = body.splitlines()
    seq = _extract_primitive_calls_from_body(body)
    # find a starting line such that seq[start:start+len(subseq)] matches
    for start in range(len(seq) - len(subseq) + 1):
        if seq[start : start + len(subseq)] == list(subseq):
            # Map seq-index -> line-index. Re-derive line indices for primitives only.
            line_idxs: list[int] = []
            for li, ln in enumerate(lines):
                ln_clean = ln.split("#", 1)[0].strip()
                if re.match(r"[A-Za-z_][A-Za-z0-9_]*\s*\(", ln_clean):
                    line_idxs.append(li)
            if start + len(subseq) > len(line_idxs):
                continue
            first_li = line_idxs[start]
            last_li = line_idxs[start + len(subseq) - 1]
            new_lines = lines[:first_li] + [f"{new_call}()"] + lines[last_li + 1 :]
            return "\n".join(new_lines)
    return body  # subseq not present (rare; pattern mining edge case)


# ---------------------------------------------------------------------------
# Operator (the proposer)
# ---------------------------------------------------------------------------

class RefactorOperator:
    """Mines recurring primitive subsequences and emits RefactorCandidates."""

    def __init__(
        self, l_min: int = 3, r_min: int = 3, *, llm: Optional[LLMClient] = None,
        max_window: int = 6,
    ) -> None:
        self.l_min = l_min
        self.r_min = r_min
        self.llm = llm
        self.max_window = max_window

    def propose(self, library: SkillLibrary) -> Iterable[RefactorCandidate]:
        # Map each skill name to its primitive-call subsequence.
        body_seqs: dict[str, list[str]] = {}
        for s in library:
            if s.is_alias():
                continue
            seq = _extract_primitive_calls_from_body(s.body)
            if len(seq) >= self.l_min:
                body_seqs[s.name] = seq

        # Count subsequences across skills (each skill contributes each unique
        # subseq it contains exactly once).
        subseq_to_skills: dict[tuple[str, ...], set[str]] = defaultdict(set)
        for name, seq in body_seqs.items():
            seen_in_this_skill: set[tuple[str, ...]] = set()
            for L in range(self.l_min, min(self.max_window, len(seq)) + 1):
                for i in range(len(seq) - L + 1):
                    sub = tuple(seq[i : i + L])
                    if sub in seen_in_this_skill:
                        continue
                    seen_in_this_skill.add(sub)
                    subseq_to_skills[sub].add(name)

        # Emit one candidate per qualifying subseq, longer subseqs first so we
        # try the most-compressive proposals before shorter ones.
        candidates = [
            (sub, sorted(parents))
            for sub, parents in subseq_to_skills.items()
            if len(parents) >= self.r_min
        ]
        candidates.sort(key=lambda x: (-len(x[0]), -len(x[1]), x[0]))
        for sub, parents in candidates:
            yield RefactorCandidate(
                subsequence=sub,
                parent_skill_names=list(parents),
                llm=self.llm,
            )
