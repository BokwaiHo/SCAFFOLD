"""Ablation driver — reproduces Table 2.

Six variants by toggling exactly one component each:
    full              the SCAFFOLD baseline
    - multi_instance  n_min=1
    - recursive       cap depth at 1
    - mdl             skip Stage 4
    - distillation    skip Stage 5
    - holdout         disable holdout validation in induction
    + skillweaver     all four innovations off (sanity row)

Each row is a short training run; the output is a CSV with one column per benchmark.
For real benchmarks the user needs the matching env config; for smoke tests, the
toy env (configs/toy.yaml) gives a result in seconds.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import os
import sys
from copy import deepcopy
from typing import Any

import yaml

from scaffold.pipeline import ScaffoldPipeline
from scaffold.utils.embedding import make_embedder
from scaffold.utils.llm import MockChatClient, make_llm_client
from scaffold.utils.logging import configure_logging, get_logger
from scripts.run_training import _build_env, _build_policy, _build_config


VARIANTS: list[tuple[str, dict[str, Any]]] = [
    ("full",                {}),
    ("no_multi_instance",   {"induction": {"n_min": 1}}),
    ("no_recursive",        {"_clamp_depth_to_1": True}),
    ("no_mdl",              {"_skip_mdl": True}),
    ("no_distillation",     {"use_stub_distiller": True}),
    ("no_holdout",          {"induction": {"enable_holdout_validation": False}}),
    ("skillweaver",         {
        "induction": {"n_min": 1, "enable_holdout_validation": False},
        "_clamp_depth_to_1": True,
        "_skip_mdl": True,
        "use_stub_distiller": True,
    }),
]


def _deep_merge(base: dict, overlay: dict) -> dict:
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def run_one_variant(base_cfg: dict, variant: dict, *, mock_llm: bool, seed: int) -> dict:
    """Run a single ablation row and return its final metrics."""
    cfg_dict = deepcopy(base_cfg)
    cfg_dict["seed"] = seed
    clamp_depth = bool(variant.pop("_clamp_depth_to_1", False))
    skip_mdl = bool(variant.pop("_skip_mdl", False))
    cfg_dict = _deep_merge(cfg_dict, variant)

    cfg = _build_config(cfg_dict)
    env = _build_env(cfg_dict.get("env") or {})
    llm = MockChatClient() if mock_llm else make_llm_client("openai")
    policy = _build_policy(cfg_dict.get("policy") or {"kind": "prompted"}, llm=llm)
    embedder = make_embedder(cfg.embedder_name)

    pipeline = ScaffoldPipeline(
        env=env, policy=policy,
        inducer_llm=llm, refactor_llm=llm,
        config=cfg, embedder=embedder,
    )

    # Apply the two special toggles in-place on the pipeline.
    if skip_mdl:
        from scaffold.core.library import SkillLibrary
        pipeline.compactor.apply = lambda lib, **kw: lib   # type: ignore[assignment]
    if clamp_depth:
        original_add = pipeline.library.add
        def _clamping_add(skill):
            skill.depth = 1
            return original_add(skill)
        pipeline.library.add = _clamping_add  # type: ignore[assignment]

    library, metrics = pipeline.run()
    final_metrics = metrics[-1] if metrics else {}
    final_metrics["variant_library_size"] = library.total_skills()
    return final_metrics


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser("scaffold-ablation")
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--out", default="ablation_table.csv")
    ap.add_argument("--mock-llm", action="store_true")
    args = ap.parse_args(argv)

    configure_logging(level="INFO")
    log = get_logger("ablation")

    with open(args.config, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f) or {}

    rows: list[dict[str, Any]] = []
    for name, variant in VARIANTS:
        for seed in args.seeds:
            log.info("=== Ablation row: %s (seed=%d) ===", name, seed)
            try:
                m = run_one_variant(base_cfg, deepcopy(variant),
                                    mock_llm=args.mock_llm, seed=seed)
            except Exception as e:
                log.error("Variant %s failed: %r", name, e)
                m = {"error": repr(e)}
            m["variant"] = name
            m["seed"] = seed
            rows.append(m)

    # Write CSV
    if rows:
        keys = sorted({k for r in rows for k in r})
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        log.info("Wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
