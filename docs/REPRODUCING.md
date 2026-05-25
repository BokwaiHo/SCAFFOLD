# Reproducing the paper

Concrete commands to reproduce each table and figure. All commands assume the
working directory is the repo root.

## Table 1 (main results)

Reproduces SCAFFOLD's row on each benchmark. Each command runs the full
5-iteration self-improvement loop; expect ~36 A100-hours per command per seed.

```bash
# WebArena → target SR 42.7 ± 0.8
python scripts/run_training.py --config configs/webarena.yaml --seed 0
python scripts/run_training.py --config configs/webarena.yaml --seed 1
python scripts/run_training.py --config configs/webarena.yaml --seed 2

# VisualWebArena → target SR 36.3 ± 0.9
python scripts/run_training.py --config configs/vwa.yaml --seed 0
python scripts/run_training.py --config configs/vwa.yaml --seed 1
python scripts/run_training.py --config configs/vwa.yaml --seed 2

# Online-Mind2Web held-out (OM2W-X) → target SR 43.5 ± 1.2
python scripts/run_training.py --config configs/om2w.yaml --seed 0
python scripts/run_training.py --config configs/om2w.yaml --seed 1
python scripts/run_training.py --config configs/om2w.yaml --seed 2
```

After each finishes, evaluate the final library on the test split:

```bash
python scripts/evaluate.py \
    --config configs/webarena.yaml \
    --library runs/webarena/iter4/library.json \
    --split test
```

The evaluator prints SR, step efficiency, and library statistics in the §4.1.3
format. Aggregate across three seeds for the mean ± std reported in Table 1.

## Table 2 (ablation)

```bash
python scripts/ablation.py --config configs/webarena.yaml --seeds 0 1 2 \
    --out tables/table2_webarena.csv
```

This runs six variants:
  - full SCAFFOLD
  - − multi-instance (n_min=1)
  - − recursive composition (depth ≤ 1)
  - − MDL compaction
  - − distillation (in-context only)
  - − holdout validation in induction
  - SkillWeaver-equivalent (all four innovations off)

Repeat with `configs/vwa.yaml` and `configs/om2w.yaml` for the other columns.

## Figure 2 (iteration scaling)

Already produced as a side-effect of Table 1 — each run writes per-iteration
metrics. Plot from `runs/<bench>/iter*/metrics.json`:

```bash
python - <<'PY'
import json, glob, matplotlib.pyplot as plt
xs, ys = [], []
for p in sorted(glob.glob("runs/webarena/iter*/metrics.json")):
    m = json.load(open(p))
    xs.append(m["iteration"]); ys.append(m["success_rate"] * 100)
plt.plot(xs, ys, marker="o", label="SCAFFOLD")
plt.xlabel("iteration k"); plt.ylabel("SR (%)")
plt.savefig("figures/figure2.pdf"); print("wrote figures/figure2.pdf")
PY
```

## Figure 3 (library growth dynamics)

The `library_size`, `library_mean_depth`, and `library_max_depth` fields of
`metrics.json` give you the data for the top and bottom panels. For the dashed
"without MDL compaction" line, re-run with the `no_mdl` ablation variant:

```bash
python scripts/ablation.py --config configs/webarena.yaml \
    --seeds 0 --out tables/no_mdl_webarena.csv
```

## Figure 4 (cross-site transfer)

```bash
python scripts/cross_site.py --config configs/om2w.yaml \
    --sites amazon landwatch github ziprecruiter ticketmaster justice \
    --out tables/figure4_matrix.csv
```

Each cell `(i, j)` of the resulting 6×6 CSV is the SR on test site `j` after
training only on site `i`. Plot as a heatmap to reproduce the figure.

## Tables 4 + 5 (per-site, hyperparameter sensitivity)

Table 4: re-run `evaluate.py` with `--split test` and aggregate by `site` from
the per-task JSON.

Table 5: a sweep over `n_min ∈ {1,2,3,4}`, `M ∈ {1,2,3,5}`, `ρ ∈ {0.7,0.8,0.9,0.95}`,
and LoRA rank `∈ {16,32,64,128}`. Each cell is a separate training run with
the corresponding override:

```bash
python scripts/run_training.py --config configs/webarena.yaml \
    --seed 0 \
    --output-dir runs/sweep/n_min_3 \
    --override induction.n_min=3      # via shell-substitution into the YAML
```

(The CLI does not yet support `--override`; for now copy the YAML and modify
the field by hand. PRs welcome.)
