"""Evaluation driver.

Loads a saved library + checkpoint and runs rollouts on the test split,
reporting the metrics from §4.1.3:
  (1) Success rate
  (2) Step efficiency (avg steps per successful task)
  (3) Library statistics (|L|, mean depth, reuse rate)

Usage:
    python scripts/evaluate.py --config configs/webarena.yaml \\
        --library runs/webarena/iter4/library.json --split test
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from typing import Any

import yaml

from scaffold.core.library import SkillLibrary
from scaffold.envs import (
    OnlineMind2WebEnv, ToyEnv, VisualWebArenaEnv, WebArenaEnv,
)
from scaffold.pipeline import ScaffoldConfig, ScaffoldPipeline
from scaffold.utils.embedding import make_embedder
from scaffold.utils.llm import MockChatClient, make_llm_client
from scaffold.utils.logging import configure_logging, get_logger
from scripts.run_training import _build_env, _build_policy, _build_config


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser("scaffold-eval")
    ap.add_argument("--config", required=True)
    ap.add_argument("--library", required=True, help="path to library.json")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--max-tasks", type=int, default=None)
    ap.add_argument("--mock-llm", action="store_true")
    args = ap.parse_args(argv)

    configure_logging(level="INFO")
    log = get_logger("eval")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f) or {}
    cfg = _build_config(cfg_dict)
    cfg.use_stub_distiller = True  # never train during eval
    env = _build_env(cfg_dict.get("env") or {})

    llm = MockChatClient() if args.mock_llm else make_llm_client("openai")
    policy = _build_policy(cfg_dict.get("policy") or {"kind": "prompted"}, llm=llm)
    embedder = make_embedder(cfg.embedder_name)

    pipeline = ScaffoldPipeline(
        env=env, policy=policy,
        inducer_llm=MockChatClient(), refactor_llm=MockChatClient(),
        config=cfg, embedder=embedder,
    )
    pipeline.library = SkillLibrary.load(args.library, embedder=embedder)
    log.info("Loaded library with %d skills", pipeline.library.total_skills())

    tasks = env.tasks(split=args.split)
    if args.max_tasks:
        tasks = tasks[: args.max_tasks]
    log.info("Evaluating on %d tasks (split=%s)", len(tasks), args.split)

    rollouts = pipeline.rollout(tasks, iteration=-1)
    n_success = sum(1 for t in rollouts if t.is_successful())
    step_counts = [len(t.actions) for t in rollouts if t.is_successful()]

    metrics = {
        "split": args.split,
        "n_tasks": len(rollouts),
        "n_success": n_success,
        "success_rate": n_success / max(len(rollouts), 1),
        "mean_steps_success": statistics.mean(step_counts) if step_counts else 0.0,
        "median_steps_success": statistics.median(step_counts) if step_counts else 0.0,
        "library_size": pipeline.library.total_skills(),
        "library_mean_depth": pipeline.library.mean_depth(),
        "library_max_depth": pipeline.library.max_depth(),
        "library_reuse_rate": pipeline.library.reuse_rate(),
    }
    print(json.dumps(metrics, indent=2))

    out_path = os.path.join(os.path.dirname(args.library) or ".",
                             f"eval_{args.split}.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Wrote %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
