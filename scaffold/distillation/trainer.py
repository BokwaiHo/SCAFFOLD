"""LoRA-based distillation (§3.6).

Paper:
    The base policy π_{k+1} is then obtained by supervised fine-tuning of π_k on
    these triples with a standard token-level cross-entropy loss, plus an
    auxiliary loss that predicts the next skill name conditioned only on
    (I, o_0:t)... We use LoRA (Hu et al., 2022) adapters and a constant learning
    rate of [1e-5] for stability across iterations.

§4.1.4 hyperparameters: rank 64, α=128, 1 epoch per iteration.

We provide two implementations:
  - `LoRADistiller`: a real HuggingFace + PEFT-backed trainer. Used in actual
    runs; requires a GPU and a transformers-compatible base model.
  - `StubDistiller`: a no-op that just bumps the saved adapter version. Used
    by unit tests and the toy env; preserves all interface invariants
    (load/save/forward) so the pipeline runs end-to-end without torch+CUDA.

We deliberately don't import torch at module top-level: the stub path must work
on a torch-less environment. The real trainer's heavy imports are inside the
LoRADistiller methods.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

from scaffold.distillation.dataset import PlanAugmentedDataset
from scaffold.utils.logging import get_logger

log = get_logger("distill")


@dataclass
class DistillationConfig:
    """LoRA + training-loop hyperparameters (defaults match §4.1.4)."""

    base_model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.0
    learning_rate: float = 1e-5
    epochs: int = 1
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_seq_len: int = 4096
    aux_loss_weight: float = 0.2     # auxiliary next-skill loss multiplier
    output_dir: str = "./runs/distill"
    save_adapter_only: bool = True   # only keep the LoRA delta, not full weights
    bf16: bool = True


class StubDistiller:
    """No-op distiller. Increments an internal version on each call.

    The pipeline talks to distillers through `train(dataset) -> str` returning a
    checkpoint path. The stub returns a synthetic path; this is enough to
    exercise the pipeline end-to-end in tests.
    """

    def __init__(self, config: Optional[DistillationConfig] = None) -> None:
        self.config = config or DistillationConfig()
        self._version = 0

    def train(self, dataset: PlanAugmentedDataset, iteration: int = 0) -> str:
        self._version += 1
        out_dir = os.path.join(self.config.output_dir, f"stub_iter{iteration}_v{self._version}")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "manifest.json"), "w") as f:
            json.dump({
                "iteration": iteration,
                "version": self._version,
                "n_examples": len(dataset),
                "type": "stub",
            }, f, indent=2)
        log.info(
            "[StubDistiller] iteration=%d examples=%d → %s",
            iteration, len(dataset), out_dir,
        )
        return out_dir


class LoRADistiller:
    """Production distiller. Uses HuggingFace transformers + PEFT.

    We intentionally keep the implementation compact and reproducible: SFT only,
    no DPO/RLHF, no multi-GPU subtleties beyond what `accelerate` provides
    out of the box.
    """

    def __init__(self, config: Optional[DistillationConfig] = None) -> None:
        self.config = config or DistillationConfig()
        self._loaded = False
        self._tokenizer = None
        self._model = None
        self._iteration = 0

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
            )
            from peft import LoraConfig, get_peft_model
        except ImportError as e:
            raise ImportError(
                "LoRADistiller requires torch + transformers + peft; "
                "switch to StubDistiller for environments without GPUs"
            ) from e

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model_name, trust_remote_code=True,
        )
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        dtype = torch.bfloat16 if self.config.bf16 else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto",
        )
        peft_cfg = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        self._model = get_peft_model(self._model, peft_cfg)
        self._loaded = True

    def train(self, dataset: PlanAugmentedDataset, iteration: int = 0) -> str:
        """SFT for one epoch over (main_targets + aux_targets) with constant LR.

        Returns the adapter checkpoint directory.
        """
        self._ensure_loaded()
        import torch
        from torch.utils.data import DataLoader

        main_pairs = dataset.render_main()
        aux_pairs = dataset.render_aux()
        log.info(
            "Distill iter %d: main_pairs=%d aux_pairs=%d",
            iteration, len(main_pairs), len(aux_pairs),
        )
        if not main_pairs and not aux_pairs:
            log.warning("Empty distillation dataset; skipping")
            return self._save_adapter(iteration)

        # Tokenize lazily
        def tokenize_pair(prompt: str, target: str):
            full = prompt + target
            tok = self._tokenizer(
                full, truncation=True, max_length=self.config.max_seq_len,
                return_tensors="pt",
            )
            prompt_len = len(self._tokenizer(prompt, truncation=True,
                                              max_length=self.config.max_seq_len).input_ids)
            labels = tok.input_ids.clone()
            labels[0, :prompt_len] = -100  # mask the prompt
            return tok.input_ids, tok.attention_mask, labels

        optimizer = torch.optim.AdamW(
            (p for p in self._model.parameters() if p.requires_grad),
            lr=self.config.learning_rate,
        )
        self._model.train()
        step = 0
        accum_loss = 0.0
        for epoch in range(self.config.epochs):
            # interleave aux (smaller weight) and main batches
            for prompt, target in main_pairs:
                input_ids, attn_mask, labels = tokenize_pair(prompt, target)
                out = self._model(
                    input_ids=input_ids.to(self._model.device),
                    attention_mask=attn_mask.to(self._model.device),
                    labels=labels.to(self._model.device),
                )
                (out.loss / self.config.gradient_accumulation_steps).backward()
                accum_loss += float(out.loss)
                step += 1
                if step % self.config.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    log.info("step %d main_loss=%.4f",
                             step, accum_loss / self.config.gradient_accumulation_steps)
                    accum_loss = 0.0
            for prompt, target in aux_pairs:
                input_ids, attn_mask, labels = tokenize_pair(prompt, " " + target)
                out = self._model(
                    input_ids=input_ids.to(self._model.device),
                    attention_mask=attn_mask.to(self._model.device),
                    labels=labels.to(self._model.device),
                )
                loss = out.loss * self.config.aux_loss_weight
                (loss / self.config.gradient_accumulation_steps).backward()
                step += 1
                if step % self.config.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return self._save_adapter(iteration)

    def _save_adapter(self, iteration: int) -> str:
        out_dir = os.path.join(self.config.output_dir, f"iter{iteration}")
        os.makedirs(out_dir, exist_ok=True)
        if self._model is not None and self.config.save_adapter_only:
            self._model.save_pretrained(out_dir)
        if self._tokenizer is not None:
            self._tokenizer.save_pretrained(out_dir)
        return out_dir
