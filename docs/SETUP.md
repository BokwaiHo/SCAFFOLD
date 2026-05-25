# Setup

This document covers a clean install of SCAFFOLD for the three supported
scenarios.

## 1. Minimal install (offline, toy env only)

For unit tests, the App. B worked example, and any offline development that
doesn't touch a real browser or train LoRA adapters.

```bash
git clone <repo-url> scaffold
cd scaffold
python -m venv .venv && source .venv/bin/activate
pip install -e .              # installs core deps; skips torch/peft
```

Run the unit tests to confirm:

```bash
pytest tests/ -v
```

Run the toy-env smoke training:

```bash
python scripts/run_training.py --config configs/toy.yaml --mock-llm
```

This should finish in well under a minute and write `runs/toy/iter*/library.json`.

## 2. Full install (GPU + real LoRA distillation)

For reproducing Table 1 on a single benchmark.

Requirements:
  - Linux with 1–8× A100-80GB (or any GPU with ≥ 24GB VRAM for Qwen2.5-VL-7B)
  - CUDA 12.1+ and a matching torch wheel

```bash
pip install -e ".[train]"     # adds torch, transformers, peft, accelerate
pip install playwright && playwright install chromium
```

Environment variables:

| variable                  | meaning                                          |
| ------------------------- | ------------------------------------------------ |
| `OPENAI_API_KEY`          | required for GPT-4o inducer / refactor proposer  |
| `WEBARENA_TASKS`          | path to the official WebArena tasks JSON         |
| `WEBARENA_BASE_URLS`      | JSON map `{"onestopshop": "http://localhost:7770", ...}` |
| `VWA_TASKS`               | path to the VisualWebArena tasks JSON            |
| `OM2W_TASKS`              | path to the Online-Mind2Web tasks JSON           |

## 3. WebArena docker stack

To reproduce the WebArena row of Table 1, install the upstream docker stack
following the instructions in
https://github.com/web-arena-x/webarena, then start the four self-hosted sites:

```bash
docker compose -f webarena/docker-compose.yml up -d
```

Verify the four base URLs come up:

```bash
curl -s http://localhost:7770/   # OneStopShop
curl -s http://localhost:9999/   # Reddit clone
curl -s http://localhost:8023/   # GitLab
curl -s http://localhost:8083/   # CMS
```

VisualWebArena uses its own docker stack — see the upstream README.

## 4. GPU memory tips

The default config uses `bf16 = true` and LoRA rank 64, fitting easily on a
single 80GB A100. For 24–40GB GPUs, drop `lora_rank` to 16 or 32 and lower
`max_seq_len` to 2048. The smoke test (`configs/toy.yaml`) sets
`use_stub_distiller: true` so it requires no GPU.
