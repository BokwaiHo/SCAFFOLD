"""Cross-site transfer reproducer (Figure 4 / §4.6).

For each training site in {amazon, landwatch, github, ziprecruiter, ticketmaster,
justice}, train SCAFFOLD with `train_sites=[site]` and then evaluate on each of
the six sites. Output a 6×6 SR matrix as CSV.

This is a long run — six full self-improvement loops — so users typically run
it overnight on a cluster. For smoke tests, --max-iterations 1 + --mock-llm
finishes in minutes.
"""

from __future__ import annotations

import argparse
import csv
import sys
from copy import deepcopy

import yaml

from scaffold.envs import OnlineMind2WebEnv
from scaffold.pipeline import ScaffoldPipeline
from scaffold.utils.embedding import make_embedder
from scaffold.utils.llm import MockChatClient, make_llm_client
from scaffold.utils.logging import configure_logging, get_logger
from scripts.run_training import _build_config, _build_policy


SITES = ["amazon", "landwatch", "github", "ziprecruiter", "ticketmaster", "justice"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser("scaffold-cross-site")
    ap.add_argument("--config", required=True)
    ap.add_argument("--sites", nargs="+", default=SITES)
    ap.add_argument("--max-iterations", type=int, default=None)
    ap.add_argument("--mock-llm", action="store_true")
    ap.add_argument("--out", default="cross_site_matrix.csv")
    args = ap.parse_args(argv)

    configure_logging(level="INFO")
    log = get_logger("cross_site")

    with open(args.config, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f) or {}
    if args.max_iterations is not None:
        base_cfg["iterations"] = args.max_iterations

    matrix: dict[str, dict[str, float]] = {}

    for train_site in args.sites:
        log.info("=== Training on %s ===", train_site)
        cfg_dict = deepcopy(base_cfg)
        cfg_dict.setdefault("env", {})["train_sites"] = [train_site]

        cfg = _build_config(cfg_dict)
        env = OnlineMind2WebEnv(
            tasks_path=cfg_dict.get("env", {}).get("tasks_path"),
            train_sites=[train_site],
        )
        llm = MockChatClient() if args.mock_llm else make_llm_client("openai")
        policy = _build_policy(cfg_dict.get("policy") or {"kind": "prompted"}, llm=llm)
        embedder = make_embedder(cfg.embedder_name)

        pipeline = ScaffoldPipeline(
            env=env, policy=policy,
            inducer_llm=llm, refactor_llm=llm,
            config=cfg, embedder=embedder,
        )
        library, _ = pipeline.run()

        # Evaluate on every site
        matrix[train_site] = {}
        for eval_site in args.sites:
            tasks = [
                t for t in env.tasks(split="train") + env.tasks(split="test")
                if t.site == eval_site
            ]
            if not tasks:
                matrix[train_site][eval_site] = 0.0
                continue
            rollouts = pipeline.rollout(tasks, iteration=-1)
            sr = sum(1 for t in rollouts if t.is_successful()) / max(len(rollouts), 1)
            matrix[train_site][eval_site] = sr
            log.info("  %s → %s : SR=%.3f (n=%d)", train_site, eval_site, sr, len(tasks))

    # write CSV
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["training\\evaluation"] + args.sites)
        for train_site in args.sites:
            row = [train_site] + [
                f"{matrix.get(train_site, {}).get(s, 0.0):.3f}" for s in args.sites
            ]
            w.writerow(row)
    log.info("Wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
