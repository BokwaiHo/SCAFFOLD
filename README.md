# SCAFFOLD: Self-Improving Web Agents via Recursive Parametric Skill Abstraction

Implementation of **SCAFFOLD**, a self-improving framework for visual web agents that (i) induces parametric, executable skills from successful trajectories under a multi-instance abstraction constraint, (ii) maintains a recursively composed hierarchy in which higher-level skills invoke lower-level ones, (iii) compacts the library via a minimum-description-length (MDL) criterion and behavioral-equivalence checking, and (iv) periodically distills skill-augmented trajectories back into model weights.

> Anonymous ARR submission. This repository contains the algorithmic implementation; environment back-ends (WebArena, VisualWebArena, Online-Mind2Web) must be installed, separately following their official instructions. See `docs/SETUP.md`.

## Quick start

```bash
git clone <this-repo>
cd scaffold
pip install -e .

# Sanity-check the algorithmic core on a synthetic toy env (no WebArena needed)
python scripts/run_training.py --config configs/toy.yaml --iters 3

# Full WebArena run (requires WebArena docker stack; see docs/SETUP.md)
python scripts/run_training.py --config configs/webarena.yaml --iters 5

# Evaluate a trained library
python scripts/evaluate.py --library runs/webarena/iter5/library.json \
                           --benchmark webarena --split test
```

## Repository layout

```
scaffold/
├── scaffold/                 ← Python package
│   ├── core/                 ← Skill, Library, Trajectory dataclasses + verifier API
│   ├── induction/            ← Stage 2: multi-instance skill induction
│   ├── composition/          ← Stage 3: recursive composition + depth & cycle bookkeeping
│   ├── compaction/           ← Stage 4: MDL functional, merge / refactor / prune
│   ├── distillation/         ← Stage 5: LoRA fine-tuning on plan-augmented trajectories
│   ├── agent/                ← Policy interface, skill executor, observation pre-proc
│   ├── envs/                 ← Environment adapters (WebArena, VWA, OM2W, toy)
│   ├── baselines/            ← Re-implementations of SkillWeaver / SkillRL / AWM / etc.
│   ├── primitives/           ← {click, type, scroll, wait} + helpers
│   ├── utils/                ← LLM wrapper, embeddings, logging, seeding
│   └── pipeline.py           ← Overall algorithm
├── configs/                  ← YAML configs per benchmark
├── scripts/                  ← run_training.py, evaluate.py, ablation.py, cross_site.py
├── tests/                    ← Unit tests for each component
├── examples/                 ← Worked depth-3 skill example (Appendix B)
└── docs/                     ← Setup, design notes, reproducibility checklist
```

## Hyperparameters (defaults)

| Symbol | Value | Description |
| --- | --- | --- |
| `n_min` | 2 | Min trajectories per cluster before induction |
| `M` | 2 | Compaction interval (iters between MDL pass) |
| `ρ` | 0.9 | Behavioral-equivalence threshold for merge |
| `ℓ_min` | 3 | Min subsequence length for refactor |
| `r_min` | 3 | Min #skills sharing subsequence for refactor |
| LoRA rank / α | 64 / 128 | Distillation adapter shape |
| LR | 1e-5 | Constant LR across iterations |
| `K` | 5 | Self-improvement iterations |
| Body cap | 30 lines | Per-skill body length limit |
| Rollout temp | 0.7 | Trajectory collection sampling |
| Inducer temp | 0.3 | Skill induction / refactor proposer |

Override any of these via the YAML config or CLI flag.

## Reproducing main results

| Benchmark | Expected SR (Qwen2.5-VL-7B) | Config |
| --- | --- | --- |
| WebArena | 42.7 ± 0.8 | `configs/webarena.yaml` |
| VisualWebArena | 36.3 ± 0.9 | `configs/vwa.yaml` |
| Online-Mind2Web (held-out) | 43.5 ± 1.2 | `configs/om2w.yaml` |

Each is `K=5` self-improvement iterations on `8×A100-80GB` (~36h per benchmark for the
WA run). See `docs/REPRODUCING.md` for end-to-end instructions and `docs/CHECKPOINTS.md`
for releasing the induced libraries.

## Tests

```bash
pytest tests/ -v
```

Covers: depth recursion, cycle detection, MDL functional monotone decrease per operator, multi-instance threshold logic, library equivalence under merge, prompt template roundtripping, and the toy-env end-to-end loop.


## License

Apache 2.0 (see `LICENSE`). The induced skill libraries we release inherit the same license; environment-specific skills must respect each benchmark's terms.
