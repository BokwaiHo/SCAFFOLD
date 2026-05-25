"""Embedding utilities.

The library retrieval index (§3.1) and the instruction-clustering step (§3.3) both
require fixed-dim vector encodings of text. Production runs use a SentenceTransformer
(e.g. all-MiniLM-L6-v2); tests use a deterministic hash embedder so the cluster
assignments are reproducible without downloading a model.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class Embedder(ABC):
    """Abstract embedder: list[str] -> (n, d) float array."""

    dim: int

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder(Embedder):
    """Wraps any sentence-transformers model. The default mirrors common practice
    in the web-agent literature, where MiniLM strikes a good speed/quality balance.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers required; "
                "`pip install sentence-transformers` or switch to HashEmbedder"
            ) from e
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.asarray(
            self._model.encode(texts, normalize_embeddings=False, convert_to_numpy=True),
            dtype=np.float32,
        )


class HashEmbedder(Embedder):
    """Deterministic hashed-trigram embedder for tests and offline development.

    Surprisingly competitive for short instruction strings; we use it as the default
    when no model is configured, so toy-env experiments don't require any download.
    """

    def __init__(self, dim: int = 64, ngram: int = 3) -> None:
        self.dim = dim
        self.ngram = ngram

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            t = t.lower()
            for j in range(max(0, len(t) - self.ngram + 1)):
                tok = t[j : j + self.ngram]
                h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:4], "little")
                out[i, h % self.dim] += 1.0
            n = np.linalg.norm(out[i])
            if n > 0:
                out[i] /= n
        return out


def make_embedder(name: Optional[str] = None) -> Embedder:
    """Factory: 'hash' / 'st:<model>' / None -> SentenceTransformer default."""
    if name is None or name == "hash":
        return HashEmbedder()
    if name.startswith("st:"):
        return SentenceTransformerEmbedder(name[len("st:") :])
    raise ValueError(f"Unknown embedder spec: {name}")
