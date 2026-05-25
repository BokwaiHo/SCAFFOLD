"""SkillLibrary: a set of skills + embedding-keyed retrieval index, per §3.1.

This module is intentionally storage-agnostic: the embedder is injected, so for tests
we can use a deterministic hash-based embedder, while production runs use a real
SentenceTransformer.

Two indices are maintained:
  - `_by_name`: O(1) name → Skill lookup (immutable IDs across iterations)
  - `_embeddings`: parallel array used for top-k semantic retrieval keyed on
                   skill.description (the paper says "keyed on desc")
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Optional, Protocol

import numpy as np

from scaffold.core.skill import Skill


class Embedder(Protocol):
    """Anything that produces fixed-dim float vectors for a list of strings."""

    def encode(self, texts: list[str]) -> np.ndarray: ...


class SkillLibrary:
    """The library L_k from the paper.

    The library is mutable across iterations: skills are added at the end of
    induction, possibly mutated (alias-merged) or removed (pruned) by the MDL
    compactor. We track all of this so that the compactor's accept/reject decisions
    can be audited.
    """

    def __init__(self, embedder: Optional[Embedder] = None) -> None:
        self._skills: dict[str, Skill] = {}
        self._embeddings: dict[str, np.ndarray] = {}
        self._embedder = embedder

    # ---------- container interface ----------

    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, name: str) -> bool:
        return name in self._skills

    def __iter__(self) -> Iterator[Skill]:
        return iter(self._skills.values())

    def __getitem__(self, name: str) -> Skill:
        skill = self._skills[name]
        # follow alias chain (created by Merge operator)
        while skill.is_alias():
            skill = self._skills[skill.is_alias_of]  # type: ignore[index]
        return skill

    def names(self) -> list[str]:
        return list(self._skills.keys())

    def active_names(self) -> list[str]:
        """Names of non-alias skills (i.e. callable from a fresh body)."""
        return [n for n, s in self._skills.items() if not s.is_alias()]

    # ---------- mutation ----------

    def add(self, skill: Skill, *, replace: bool = False) -> None:
        if skill.name in self._skills and not replace:
            raise ValueError(f"Skill {skill.name!r} already exists in library")
        # cycle-prevention check is done by composition module before calling here;
        # here we only refuse to add a skill that calls something we don't know.
        unknown = self._unknown_callees(skill)
        if unknown:
            raise ValueError(
                f"Skill {skill.name!r} calls undefined names: {sorted(unknown)}"
            )
        self._skills[skill.name] = skill
        if self._embedder is not None:
            v = self._embedder.encode([skill.description])[0]
            self._embeddings[skill.name] = v / (np.linalg.norm(v) + 1e-12)

    def remove(self, name: str) -> None:
        if name not in self._skills:
            return
        # safety: refuse to remove a skill that is still invoked transitively (§3.5)
        if self._is_transitively_invoked(name):
            raise ValueError(f"Cannot prune {name!r}: still invoked by another skill")
        self._skills.pop(name, None)
        self._embeddings.pop(name, None)

    def mark_alias(self, old_name: str, new_name: str) -> None:
        """Used by Merge: keep `old_name` as an alias that forwards to `new_name`."""
        if new_name not in self._skills:
            raise ValueError(f"Alias target {new_name!r} not in library")
        if old_name == new_name:
            return
        old = self._skills[old_name]
        # rebuild with the alias flag set
        self._skills[old_name] = Skill(
            name=old.name,
            description=old.description,
            parameters=old.parameters,
            precondition=old.precondition,
            body=old.body,
            postcondition=old.postcondition,
            depth=old.depth,
            iteration_introduced=old.iteration_introduced,
            times_used=old.times_used,
            induced_from_cluster_size=old.induced_from_cluster_size,
            is_alias_of=new_name,
            parent_skills=old.parent_skills,
        )

    # ---------- retrieval ----------

    def retrieve(self, query: str, k: int = 5) -> list[Skill]:
        """Top-k cosine-similar skills to `query` (keyed on each skill's `desc`)."""
        if not self._skills or self._embedder is None or not self._embeddings:
            return list(self._skills.values())[:k]
        q = self._embedder.encode([query])[0]
        q = q / (np.linalg.norm(q) + 1e-12)
        names = list(self._embeddings.keys())
        mat = np.stack([self._embeddings[n] for n in names], axis=0)
        scores = mat @ q
        order = np.argsort(-scores)
        out: list[Skill] = []
        for idx in order[:k]:
            s = self._skills[names[idx]]
            if not s.is_alias():
                out.append(s)
        return out

    # ---------- statistics (Table 5 / Figure 3 inputs) ----------

    def total_skills(self) -> int:
        return sum(1 for s in self._skills.values() if not s.is_alias())

    def mean_depth(self) -> float:
        ds = [s.depth for s in self._skills.values() if not s.is_alias()]
        return float(np.mean(ds)) if ds else 0.0

    def max_depth(self) -> int:
        ds = [s.depth for s in self._skills.values() if not s.is_alias()]
        return int(max(ds)) if ds else 0

    def reuse_rate(self) -> float:
        """Fraction of skills called by >= 2 trajectories, see §4.5."""
        active = [s for s in self._skills.values() if not s.is_alias()]
        if not active:
            return 0.0
        return float(np.mean([s.times_used >= 2 for s in active]))

    # ---------- serialisation ----------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(
                {"skills": [s.to_dict() for s in self._skills.values()]},
                f,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def load(cls, path: str | Path, embedder: Optional[Embedder] = None) -> "SkillLibrary":
        lib = cls(embedder=embedder)
        with Path(path).open("r", encoding="utf-8") as f:
            d = json.load(f)
        # add in dependency order so callee validation succeeds
        skills = [Skill.from_dict(x) for x in d["skills"]]
        skills.sort(key=lambda s: s.depth)
        for s in skills:
            lib._skills[s.name] = s  # bypass `add` validation during bulk-load
            if embedder is not None:
                v = embedder.encode([s.description])[0]
                lib._embeddings[s.name] = v / (np.linalg.norm(v) + 1e-12)
        return lib

    def copy(self) -> "SkillLibrary":
        """Shallow copy that shares the embedder. Used by MDL accept/reject trials."""
        new = SkillLibrary(embedder=self._embedder)
        new._skills = dict(self._skills)
        new._embeddings = dict(self._embeddings)
        return new

    # ---------- internals ----------

    def _unknown_callees(self, skill: Skill) -> set[str]:
        from scaffold.primitives.actions import PRIMITIVE_NAMES

        unknown: set[str] = set()
        for callee in skill.calls_set():
            if callee in self._skills or callee in PRIMITIVE_NAMES:
                continue
            unknown.add(callee)
        return unknown

    def _is_transitively_invoked(self, name: str) -> bool:
        for other in self._skills.values():
            if other.name == name or other.is_alias():
                continue
            if name in other.calls_set():
                return True
        return False

    # ---------- iteration update ----------

    def increment_usage(self, names: Iterable[str]) -> None:
        for n in names:
            if n in self._skills:
                self._skills[n].times_used += 1
