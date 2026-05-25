"""Command-line entry points (referenced by pyproject.toml [project.scripts]).

Two commands:
  scaffold-train  → scripts/run_training.py (full self-improvement loop)
  scaffold-eval   → scripts/evaluate.py (load a saved library and report SR)

These are thin dispatchers that forward to the actual script logic so the user
can either run `python scripts/run_training.py ...` or `scaffold-train ...`.
"""

from __future__ import annotations

import argparse
import sys


def train_main(argv: list[str] | None = None) -> int:
    from scripts.run_training import main as _main
    return _main(argv)


def eval_main(argv: list[str] | None = None) -> int:
    from scripts.evaluate import main as _main
    return _main(argv)


def main(argv: list[str] | None = None) -> int:
    """Single dispatcher used when entry-point installation isn't available."""
    parser = argparse.ArgumentParser(prog="scaffold", description="SCAFFOLD CLI")
    parser.add_argument("command", choices=["train", "eval", "ablation", "cross-site"])
    args, rest = parser.parse_known_args(argv)
    if args.command == "train":
        return train_main(rest)
    if args.command == "eval":
        return eval_main(rest)
    if args.command == "ablation":
        from scripts.ablation import main as _m
        return _m(rest)
    if args.command == "cross-site":
        from scripts.cross_site import main as _m
        return _m(rest)
    return 1


if __name__ == "__main__":
    sys.exit(main())
