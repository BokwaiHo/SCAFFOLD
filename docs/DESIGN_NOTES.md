# Design Notes

This document explains what's fully implemented in the repository, what's interface-stubbed for portability, and where the boundaries are between the two. Read this before attempting to extend the framework.

## Fully implemented (algorithmic core)

These components implement the paper's algorithms in full and are directly testable without any external dependencies:

| component                  | paper section | file(s)                            |
| -------------------------- | ------------- | ---------------------------------- |
| Algorithm 1 main loop      | §3.2          | `scaffold/pipeline.py`             |
| Multi-instance induction   | §3.3          | `scaffold/induction/multi_instance.py` |
| Instruction clustering     | §3.3          | `scaffold/induction/clusterer.py`  |
| Recursive composition      | §3.4 (Eq. 1)  | `scaffold/composition/recursive.py`|
| Cycle prevention           | §3.4          | `scaffold/composition/recursive.py`|
| MDL functional F(L)        | §3.5 (Eq. 2)  | `scaffold/compaction/mdl.py`       |
| Merge operator             | §3.5          | `scaffold/compaction/merge.py`     |
| Refactor operator          | §3.5 + App.A  | `scaffold/compaction/refactor.py`  |
| Prune operator             | §3.5          | `scaffold/compaction/prune.py`     |
| Plan-augmented dataset     | §3.6          | `scaffold/distillation/dataset.py` |
| Skill library + retrieval  | §3.1          | `scaffold/core/library.py`         |
| Skill executor             | §3.1 / App.B  | `scaffold/agent/executor.py`       |
| Pre/postcondition validator| App. A        | `scaffold/agent/precondition.py`   |

These are the parts you should read first if you want to understand or modify SCAFFOLD's algorithmic behavior.

## Interface-stubbed (production behind a façade)

These components have a real production path and a lightweight stub that preserves the interface contract:

| component         | real path                        | stub path              |
| ----------------- | -------------------------------- | ---------------------- |
| LoRA distillation | `LoRADistiller` (torch + peft)   | `StubDistiller`        |
| LLM inducer       | `OpenAIChatClient`               | `MockChatClient`       |
| Embedder          | `SentenceTransformerEmbedder`    | `HashEmbedder`         |
| Browser           | `PlaywrightBrowser`              | `ToyBrowser`           |
| WebArena tasks    | upstream JSON (`WEBARENA_TASKS`) | `ToyEnv`               |

The stubs are deliberately *behaviorally* simple so that tests run in seconds and don't require network or GPUs. They are sufficient to exercise every code path in the production system except for the actual ML training kernel and the real browser DOM. They are NOT meant to reproduce the paper's numbers.

## Algorithmic faithfulness vs. shortcuts

Three places where the implementation takes a justified shortcut from the paper's full description:

1. **Behavioral-equivalence in Merge (§3.5)**. The paper specifies a probabilistic check ("post-states identical with probability ≥ ρ" on a sampled hold-out set). Our default `MergeOperator` uses *static* body equivalence (after string-literal normalization), which is a sound but slightly stricter approximation. Users running the WebArena docker stack can plug in a real behavioral check via `MergeOperator(behavioral_check=...)`.

2. **MDL parse length (Eq. 2)**. The data cost requires `min_{parse} |parse(ζ | L)|`, the length of the *shortest* program. We approximate with a *greedy longest-match decoder* (`parse_trajectory_under_library`). This is the standard approximation used in DreamCoder-style library learning; it matches the optimal parse on the trajectories we observed but is not guaranteed to in general.

3. **GRPO baseline (§4.1.2, SkillRL)**. The SkillRL baseline in the paper is trained with GRPO. Our re-implementation uses reward-filtered SFT, which is the closest no-RL-trainer-dependency approximation. This is documented in the baseline's docstring.

## Adding a new env

Subclass `scaffold.envs.base.Env` and provide:

- `tasks(split: str) -> list[Task]`
- `make_browser(task: Task) -> Browser` — `Browser` is the protocol in
  `scaffold/primitives/actions.py`. You need `observe()`, `click()`, `type()`,
  `scroll()`, `wait()`.
- `close_browser(browser: Browser) -> None`

That's it — the pipeline takes care of policy invocation, executor wiring, and trajectory recording. See `scaffold/envs/toy_env.py` for a complete reference implementation.

## Adding a new operator

Subclass `MDLOperator` (informally — the protocol is just `propose(library) -> Iterable[Candidate]` and `Candidate.apply(library)`). Add it to the rotation in
`MDLCompactor.apply()`. The `_try_accept` path handles F-decrease + SR-no-drop gating for you.

## What's NOT in this repo

- The proprietary WebArena docker stack (use upstream).
- Pre-trained Qwen2.5-VL-7B / UI-TARS-7B weights (use HuggingFace).
- The OM2W-X tasks JSON (use upstream).

For all three, the repo's adapters expect external paths via env vars; see `docs/SETUP.md`.
