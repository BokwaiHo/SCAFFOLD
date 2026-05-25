"""Skill data structure.

Per §3.1 of the paper, a skill σ is the tuple

    σ = (name, desc, θ, pre, body, post, d_σ)

where
    name   : str          identifier
    desc   : str          natural-language description (used for embedding-keyed retrieval)
    θ      : Parameter[]  typed parameter list
    pre    : str          LLM-checkable precondition predicate over obs_t (Python expr)
    body   : str          executable Python program over A_prim ∪ A_skill
    post   : str          LLM-checkable postcondition predicate over obs_t (Python expr)
    d_σ    : int          abstraction depth, defined recursively in §3.4 / Eq. (1)
"""

from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class ParameterType(str, Enum):
    """Typed parameter tags. Kept small on purpose: web tasks rarely need more."""

    STRING = "str"
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    URL = "url"
    SELECTOR = "selector"  # CSS / XPath selector or semantic role
    ENUM = "enum"          # one of a fixed value set, see Parameter.choices


@dataclass(frozen=True)
class Parameter:
    """A single typed slot in a skill's parameter list θ."""

    name: str
    type: ParameterType = ParameterType.STRING
    default: Optional[Any] = None
    choices: Optional[tuple[str, ...]] = None  # for ENUM
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name.isidentifier():
            raise ValueError(f"Parameter name must be a Python identifier: {self.name!r}")
        if self.type is ParameterType.ENUM and not self.choices:
            raise ValueError(f"ENUM parameter {self.name} requires choices=")

    def signature_repr(self) -> str:
        """Used by MDL token-length proxy and prompt rendering."""
        s = f"{self.name}: {self.type.value}"
        if self.default is not None:
            s += f" = {self.default!r}"
        return s


@dataclass
class SkillCall:
    """A single invocation of a skill (or primitive) within a skill body or plan."""

    name: str
    args: dict[str, Any] = field(default_factory=dict)

    def is_primitive(self) -> bool:
        from scaffold.primitives.actions import PRIMITIVE_NAMES

        return self.name in PRIMITIVE_NAMES

    def to_source(self) -> str:
        kw = ", ".join(f"{k}={v!r}" for k, v in self.args.items())
        return f"{self.name}({kw})"


# Per §3.4: "we cap the per-skill body length at 30 lines"
MAX_BODY_LINES = 30


@dataclass
class Skill:
    """The fundamental unit of SCAFFOLD's library.

    `depth` is set by the composition module (Eq. 1) and validated against
    `calls_set` whenever the skill is added to / mutated within a library.
    """

    name: str
    description: str
    parameters: list[Parameter] = field(default_factory=list)
    precondition: str = "True"
    body: str = "pass"
    postcondition: str = "True"
    depth: int = 1                          # d_σ, see §3.4
    iteration_introduced: int = 0           # which iter k this was created at
    times_used: int = 0                     # for prune operator (§3.5)
    induced_from_cluster_size: int = 0      # n_min provenance (§3.3)
    is_alias_of: Optional[str] = None       # set by Merge operator (§3.5)
    parent_skills: tuple[str, ...] = ()     # used to track refactor lineage

    # ---------- validation ----------

    def __post_init__(self) -> None:
        if not self.name.isidentifier():
            raise ValueError(f"Skill name must be a Python identifier: {self.name!r}")
        # body length cap (§3.4)
        n_lines = len([ln for ln in self.body.splitlines() if ln.strip()])
        if n_lines > MAX_BODY_LINES:
            raise ValueError(
                f"Skill {self.name}: body has {n_lines} non-blank lines, "
                f"exceeds cap of {MAX_BODY_LINES} (see §3.4)"
            )
        # syntactic validity check; the body must be a valid Python suite
        try:
            ast.parse(self._body_as_function_source())
        except SyntaxError as e:
            raise ValueError(f"Skill {self.name}: body has SyntaxError: {e}") from e

    # ---------- programmatic surface ----------

    def signature(self) -> str:
        """Render the Python signature used when the skill is exposed to the policy."""
        params = ", ".join(["obs"] + [p.signature_repr() for p in self.parameters])
        return f"def {self.name}({params}):"

    def calls_set(self) -> set[str]:
        """Static set of skill / primitive names called in `body`.

        Approximate: scans the body AST for `Call` nodes whose callable is a `Name`.
        We deliberately do not chase dynamic calls; the inducer prompt forbids them.
        """
        try:
            tree = ast.parse(self._body_as_function_source())
        except SyntaxError:
            return set()
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                names.add(node.func.id)
        # exclude self-references (cycles checked separately) and obvious builtins
        return names - {self.name, "len", "range", "any", "all", "isinstance",
                        "str", "int", "float", "bool", "list", "dict", "set", "print"}

    def token_length(self) -> int:
        """|σ| from Eq. (2), used by the MDL functional.

        We use a cheap whitespace+symbol tokenization here; production code substitutes
        ``tiktoken`` for fidelity with the inducer LLM. The constants used by the MDL
        functional are calibrated against this proxy in tests.
        """
        src = self._render_full()
        return len(re.findall(r"\w+|[^\s\w]", src))

    def is_alias(self) -> bool:
        return self.is_alias_of is not None

    # ---------- serialisation ----------

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type.value,
                    "default": p.default,
                    "choices": list(p.choices) if p.choices else None,
                    "description": p.description,
                }
                for p in self.parameters
            ],
            "precondition": self.precondition,
            "body": self.body,
            "postcondition": self.postcondition,
            "depth": self.depth,
            "iteration_introduced": self.iteration_introduced,
            "times_used": self.times_used,
            "induced_from_cluster_size": self.induced_from_cluster_size,
            "is_alias_of": self.is_alias_of,
            "parent_skills": list(self.parent_skills),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Skill":
        params = [
            Parameter(
                name=p["name"],
                type=ParameterType(p["type"]),
                default=p.get("default"),
                choices=tuple(p["choices"]) if p.get("choices") else None,
                description=p.get("description", ""),
            )
            for p in d.get("parameters", [])
        ]
        return cls(
            name=d["name"],
            description=d["description"],
            parameters=params,
            precondition=d.get("precondition", "True"),
            body=d.get("body", "pass"),
            postcondition=d.get("postcondition", "True"),
            depth=d.get("depth", 1),
            iteration_introduced=d.get("iteration_introduced", 0),
            times_used=d.get("times_used", 0),
            induced_from_cluster_size=d.get("induced_from_cluster_size", 0),
            is_alias_of=d.get("is_alias_of"),
            parent_skills=tuple(d.get("parent_skills", [])),
        )

    def content_hash(self) -> str:
        """Hash of (parameters, body, pre, post). Useful to detect literal duplicates
        prior to running the more expensive behavioral equivalence check."""
        h = hashlib.sha256()
        h.update(self.body.encode())
        h.update(self.precondition.encode())
        h.update(self.postcondition.encode())
        for p in self.parameters:
            h.update(p.signature_repr().encode())
        return h.hexdigest()[:16]

    # ---------- internal ----------

    def _body_as_function_source(self) -> str:
        params = ", ".join(["obs"] + [p.signature_repr() for p in self.parameters])
        indented = "\n".join("    " + ln for ln in self.body.splitlines()) or "    pass"
        return f"def {self.name}({params}):\n{indented}\n"

    def _render_full(self) -> str:
        """Full source-form rendering used by the MDL token counter and the prompt."""
        out = [f'"""{self.description}"""']
        out.append(self.signature())
        if self.precondition and self.precondition != "True":
            out.append(f"    # pre: {self.precondition}")
        for ln in self.body.splitlines() or ["pass"]:
            out.append("    " + ln)
        if self.postcondition and self.postcondition != "True":
            out.append(f"    # post: {self.postcondition}")
        return "\n".join(out)

    def __repr__(self) -> str:
        return f"Skill({self.name}, d={self.depth}, |θ|={len(self.parameters)})"
