# Checkpoints

Per-iteration artifacts produced by `scripts/run_training.py`.

## Directory layout

```
runs/<benchmark>/
├── iter0/
│   ├── library.json       SkillLibrary at end of iteration 0
│   └── metrics.json       per-iteration metrics
├── iter1/
│   ├── library.json
│   ├── metrics.json
├── ...
├── iter4/
│   ├── library.json       L_K (final library)
│   └── metrics.json
├── distill/
│   ├── iter0/             LoRA adapter for π_1
│   ├── iter1/             LoRA adapter for π_2
│   └── ...
└── final/
    ├── library.json       symlink/copy of iter4/library.json for convenience
    └── metrics.json       all-iteration metrics aggregated
```

## `library.json` schema

```json
{
  "version": 1,
  "skills": [
    {
      "name": "click_text",
      "description": "...",
      "parameters": [{"name": "text", "type": "str", ...}],
      "precondition": "any(...)",
      "body": "el = find(obs.dom, ...)\\nprimitive_click(el)",
      "postcondition": "True",
      "depth": 1,
      "iteration_introduced": 1,
      "times_used": 17,
      "induced_from_cluster_size": 4,
      "is_alias_of": null,
      "parent_skills": ["search_and_filter"]
    }
  ]
}
```

Load with:

```python
from scaffold.core.library import SkillLibrary
from scaffold.utils.embedding import make_embedder
lib = SkillLibrary.load("runs/webarena/iter4/library.json",
                         embedder=make_embedder("st:all-MiniLM-L6-v2"))
print(f"|L| = {lib.total_skills()}, mean depth = {lib.mean_depth():.2f}")
```

## `metrics.json` schema (per-iteration)

```json
{
  "iteration": 4,
  "n_rollouts": 812,
  "n_success": 347,
  "success_rate": 0.4273,
  "n_clusters": 84,
  "n_new_skills": 19,
  "library_size": 181,
  "library_mean_depth": 2.71,
  "library_max_depth": 5,
  "library_reuse_rate": 0.64,
  "checkpoint": "runs/webarena/distill/iter4"
}
```

## LoRA adapter directories

Each `distill/iterK/` contains a PEFT-format adapter:

```
distill/iter4/
├── adapter_config.json
├── adapter_model.safetensors
└── tokenizer.json
```

Load with:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
model = PeftModel.from_pretrained(base, "runs/webarena/distill/iter4")
tok = AutoTokenizer.from_pretrained("runs/webarena/distill/iter4")
```

## When using StubDistiller

Tests and the toy env use `StubDistiller`, which writes only a `manifest.json` under `distill/stub_iterK_v*/`. No model weights are saved; the pipeline treats the path as opaque.
