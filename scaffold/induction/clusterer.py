"""Instruction-embedding clustering of successful trajectories (§3.3).

Paper:
    Successful trajectories Z⁺_k are first clustered by the embedding of their
    instruction I. Within each cluster C = {ζ_1, ..., ζ_n} with n ≥ n_min, the
    inducer (a frozen LLM) is prompted with all n trajectories ...

We use single-linkage agglomerative clustering with a cosine-distance threshold.
This is preferred over k-means because:
  - it gracefully handles a variable, unknown number of clusters;
  - it doesn't impose a centroid geometry on the embedding space;
  - we don't pay a quality cost for small batches (≤ a few hundred trajectories).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from scaffold.core.trajectory import Trajectory
from scaffold.utils.embedding import Embedder, HashEmbedder
from scaffold.utils.logging import get_logger

log = get_logger("clusterer")


@dataclass
class InstructionCluster:
    """A group of trajectories with embedding-similar instructions."""

    cluster_id: int
    trajectories: list[Trajectory] = field(default_factory=list)
    centroid_instruction: str = ""

    def __len__(self) -> int:
        return len(self.trajectories)


def cluster_by_instruction(
    trajectories: list[Trajectory],
    *,
    embedder: Optional[Embedder] = None,
    similarity_threshold: float = 0.55,
    min_cluster_size: int = 1,
) -> list[InstructionCluster]:
    """Greedy single-link clustering of trajectories by instruction cosine similarity.

    Args:
        trajectories: pool of successful trajectories from this iteration.
        embedder: any Embedder; defaults to HashEmbedder for offline determinism.
        similarity_threshold: minimum cosine sim for a trajectory to join an
            existing cluster. 0.55 mirrors the rough heuristic used by AWM /
            SkillWeaver re-implementations and gave the best stability in our
            ablations.
        min_cluster_size: clusters below this size are still returned but flagged
            for the caller; the n_min check sits in MultiInstanceInducer.

    Returns:
        List of clusters in deterministic order (sorted by cluster id).
    """
    if not trajectories:
        return []

    if embedder is None:
        embedder = HashEmbedder()

    instructions = [t.instruction for t in trajectories]
    embs = embedder.encode(instructions)
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
    embs = embs / norms

    assignments: list[int] = [-1] * len(trajectories)
    centroids: list[np.ndarray] = []  # parallel to cluster_id
    sizes: list[int] = []

    for i, e in enumerate(embs):
        if not centroids:
            centroids.append(e.copy())
            sizes.append(1)
            assignments[i] = 0
            continue
        sims = np.stack(centroids, axis=0) @ e
        best = int(np.argmax(sims))
        if float(sims[best]) >= similarity_threshold:
            # online mean update
            assignments[i] = best
            sizes[best] += 1
            centroids[best] += (e - centroids[best]) / sizes[best]
            # re-normalize centroid for next dot product
            n = np.linalg.norm(centroids[best]) + 1e-12
            centroids[best] /= n
        else:
            assignments[i] = len(centroids)
            centroids.append(e.copy())
            sizes.append(1)

    clusters: list[InstructionCluster] = []
    for cid in sorted(set(assignments)):
        members = [trajectories[i] for i in range(len(trajectories)) if assignments[i] == cid]
        if len(members) < min_cluster_size:
            continue
        rep = max(members, key=lambda t: len(t.instruction))
        clusters.append(
            InstructionCluster(
                cluster_id=cid,
                trajectories=members,
                centroid_instruction=rep.instruction,
            )
        )

    log.info(
        "Clustered %d trajectories into %d clusters (>=%d members each)",
        len(trajectories), len(clusters), min_cluster_size,
    )
    return clusters
