"""OutlierScorer — the two-axis novelty signal.

Design decision (docs/spec §4): we deliberately do **not** blend YOLO confidence
and DINOv2 distance into one opaque score. Keeping them as two independent axes
(a) avoids an arbitrary weighting and (b) lets every routed sample carry a
human-readable reason — directly serving the "explainability" goal.

All distances are cosine, computed in the **original** DINOv2 embedding space
(never UMAP).
"""

from __future__ import annotations

import numpy as np

from ..types import Detection, Scores


def _l2norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def cosine_knn_distance(query: np.ndarray, neighbors: np.ndarray, k: int) -> float:
    """Mean cosine distance from ``query`` to its ``k`` nearest ``neighbors``."""
    neighbors = np.asarray(neighbors, dtype=float)
    if neighbors.ndim != 2 or neighbors.shape[0] == 0:
        return float("inf")
    q = _l2norm(np.asarray(query, dtype=float).reshape(-1))
    N = _l2norm(neighbors)
    sims = N @ q
    dists = 1.0 - sims
    kk = min(k, dists.shape[0])
    nearest = np.partition(dists, kk - 1)[:kk]
    return float(nearest.mean())


def intra_class_knn_distances(embeddings: np.ndarray, k: int) -> np.ndarray:
    """Leave-one-out kNN distance for each sample within a single class.

    Used to calibrate the per-class distance ceiling. Self is excluded.
    """
    E = np.asarray(embeddings, dtype=float)
    n = E.shape[0]
    if n < 2:
        return np.zeros((n,), dtype=float)
    N = _l2norm(E)
    sims = N @ N.T
    np.fill_diagonal(sims, -np.inf)  # exclude self -> distance becomes +inf
    dists = 1.0 - sims
    kk = min(k, n - 1)
    out = np.empty(n, dtype=float)
    for i in range(n):
        nearest = np.partition(dists[i], kk - 1)[:kk]
        out[i] = nearest.mean()
    return out


class OutlierScorer:
    """Scores detections against per-class golden embeddings."""

    def __init__(self, class_embeddings: dict[str, np.ndarray], k: int = 10):
        self.class_embeddings = {
            c: np.asarray(e, dtype=float) for c, e in class_embeddings.items()
        }
        self.k = k

    def score_detection(self, embedding: np.ndarray, predicted_class: str) -> tuple[float, bool]:
        """Return (knn_distance, low_support) for one detection.

        Unknown / unsupported class -> (inf, True): treated as maximally novel.
        """
        neigh = self.class_embeddings.get(predicted_class)
        if neigh is None or neigh.shape[0] == 0:
            return float("inf"), True
        low_support = neigh.shape[0] < self.k
        return cosine_knn_distance(embedding, neigh, self.k), low_support

    def score_image(self, detections: list[Detection]) -> Scores:
        """Aggregate detections to image-level routing signals.

        Mutates each detection's ``knn_dist`` / ``low_support`` in place so the
        caller can keep per-detection reasons. ``conf_max`` = max confidence;
        ``knn_dist`` = max kNN distance (the most-outlier detection drives review).
        """
        if not detections:
            return Scores(conf_max=0.0, knn_dist=float("inf"))
        conf_max = max(d.confidence for d in detections)
        dists: list[float] = []
        for det in detections:
            if det.embedding is not None:
                dist, low = self.score_detection(det.embedding, det.label)
                det.knn_dist = dist
                det.low_support = low
                dists.append(dist)
        knn = max(dists) if dists else float("inf")
        return Scores(conf_max=conf_max, knn_dist=knn)
