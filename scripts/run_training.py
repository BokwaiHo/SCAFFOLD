"""SCAFFOLD training driver.

Usage:
    python scripts/run_training.py --config configs/toy.yaml
    python scripts/run_training.py --config configs/webarena.yaml --seed 0

This is the one place where YAML → typed config → env+policy+pipeline →
.run() happens. Everything else in the repo is exercised through ScaffoldPipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

try:
    import yaml
except ImportError:
    print("PyYAML is required: pip install pyyaml", file=sys.stderr)
    raise

from scaffold.agent.policy import PromptedPolicy, ScriptedPolicy
from scaffold.compaction import MDLConfig
from scaffold.distillation import DistillationConfig
from scaffold.envs import (
    Env, OnlineMind2WebEnv, ToyEnv, VisualWebArenaEnv, WebArenaEnv,
)
from scaffold.envs.base import RolloutConfig
from scaffold.induction import InductionConfig
from scaffold.pipeline import ScaffoldConfig, ScaffoldPipeline
from scaffold.utils.embedding import make_embedder
from scaffold.utils.llm import MockChatClient, OpenAIChatClient, make_llm_client
from scaffold.utils.logging import configure_logging, get_logger


def _build_env(spec: dict[str, Any]) -> Env:
    name = (spec or {}).get("name", "toy")
    if name == "toy":
        return ToyEnv(
            n_train=spec.get("n_train", 60),
            n_val=spec.get("n_val", 16),
            n_test=spec.get("n_test", 32),
        )
    if name == "webarena":
        return WebArenaEnv(
            tasks_path=spec.get("tasks_path"),
            sites=spec.get("sites"),
            headless=spec.get("headless", True),
        )
    if name == "vwa":
        return VisualWebArenaEnv(
            tasks_path=spec.get("tasks_path"),
            sites=spec.get("sites"),
            headless=spec.get("headless", True),
        )
    if name == "om2w":
        return OnlineMind2WebEnv(
            tasks_path=spec.get("tasks_path"),
            train_sites=spec.get("train_sites"),
            held_out_domains=tuple(spec.get("held_out_domains", ())) or None,
            headless=spec.get("headless", True),
        )
    raise ValueError(f"Unknown env name: {name!r}")


def _build_policy(spec: dict[str, Any], llm):
    kind = (spec or {}).get("kind", "prompted")
    if kind == "prompted":
        return PromptedPolicy(
            llm=llm,
            temperature=spec.get("temperature", 0.7),
            top_k_skills=spec.get("top_k_skills", 8),
        )
    if kind == "scripted":
        from scaffold.core.trajectory import Action, ActionType
        actions = [
            Action(type=ActionType(a["type"]), args=a.get("args", {}))
            for a in spec.get("script", [])
        ]
        return ScriptedPolicy(actions)
    raise ValueError(f"Unknown policy kind: {kind!r}")


def _build_config(cfg_dict: dict[str, Any]) -> ScaffoldConfig:
    """Map a YAML dict into the nested ScaffoldConfig dataclass tree."""
    ind = InductionConfig(**(cfg_dict.get("induction") or {}))
    mdl = MDLConfig(**(cfg_dict.get("mdl") or {}))
    dist = DistillationConfig(**(cfg_dict.get("distillation") or {}))
    roll = RolloutConfig(**(cfg_dict.get("rollout") or {}))
    return ScaffoldConfig(
        iterations=cfg_dict.get("iterations", 5),
        compaction_interval=cfg_dict.get("compaction_interval", 2),
        induction=ind,
        mdl=mdl,
        distillation=dist,
        rollout=roll,
        output_dir=cfg_dict.get("output_dir", "./runs/scaffold"),
        use_stub_distiller=cfg_dict.get("use_stub_distiller", False),
        embedder_name=cfg_dict.get("embedder_name"),
        cluster_similarity_threshold=cfg_dict.get("cluster_similarity_threshold", 0.55),
        use_llm_precondition=cfg_dict.get("use_llm_precondition", False),
        seed=cfg_dict.get("seed", 0),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser("scaffold-train")
    ap.add_argument("--config", required=True, help="path to YAML config")
    ap.add_argument("--seed", type=int, default=None, help="override seed")
    ap.add_argument("--output-dir", type=str, default=None, help="override output_dir")
    ap.add_argument("--mock-llm", action="store_true",
                    help="use MockChatClient instead of OpenAI (smoke tests)")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)

    configure_logging(level=args.log_level)
    log = get_logger("train")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f) or {}

    if args.seed is not None:
        cfg_dict["seed"] = args.seed
    if args.output_dir:
        cfg_dict["output_dir"] = args.output_dir

    cfg = _build_config(cfg_dict)
    env = _build_env(cfg_dict.get("env") or {})
    log.info("Env: %s | tasks(train)=%d", type(env).__name__, len(env.tasks(split="train")))

    # LLM clients: a single one drives inducer + refactor + judge by default.
    if args.mock_llm or cfg.use_stub_distiller:
        inducer_llm = MockChatClient()
        refactor_llm = MockChatClient()
        judge_llm = MockChatClient()
    else:
        inducer_llm = make_llm_client((cfg_dict.get("inducer") or {}).get("provider", "openai"))
        refactor_llm = inducer_llm
        judge_llm = make_llm_client((cfg_dict.get("judge") or {}).get("provider", "openai"))

    policy = _build_policy(cfg_dict.get("policy") or {"kind": "prompted"}, llm=inducer_llm)
    embedder = make_embedder(cfg.embedder_name)

    pipeline = ScaffoldPipeline(
        env=env,
        policy=policy,
        inducer_llm=inducer_llm,
        refactor_llm=refactor_llm,
        config=cfg,
        embedder=embedder,
        judge_llm=judge_llm,
    )
    library, metrics = pipeline.run()

    # Final dump
    final_dir = os.path.join(cfg.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    library.save(os.path.join(final_dir, "library.json"))
    with open(os.path.join(final_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    log.info("Saved final library + metrics to %s", final_dir)
    log.info("Final library: |L|=%d, mean_depth=%.2f, max_depth=%d, reuse=%.2f",
             library.total_skills(), library.mean_depth(),
             library.max_depth(), library.reuse_rate())
    return 0


if __name__ == "__main__":
    sys.exit(main())
