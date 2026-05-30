"""Random-hyperplane LSH for sub-quadratic candidate generation (T10 scale).

Brute-force O(n²) pairwise cosine is fine for thousands but not for ~500k
images. LSH buckets near-parallel vectors together so dedup / label-error only
compare within buckets — near-linear for well-separated data — while staying
pure numpy (no FAISS/FiftyOne needed for the core path).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .scorer import _l2norm


def lsh_signatures(M: np.ndarray, n_bits: int = 16, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    planes = rng.randn(M.shape[1], n_bits)
    bits = (M @ planes) > 0
    powers = (1 << np.arange(n_bits)).astype(np.int64)
    return bits.astype(np.int64) @ powers


def lsh_buckets(M: np.ndarray, n_bits: int = 16, seed: int = 0) -> dict[int, list[int]]:
    sig = lsh_signatures(_l2norm(M), n_bits, seed)
    buckets: dict[int, list[int]] = defaultdict(list)
    for i, s in enumerate(sig):
        buckets[int(s)].append(i)
    return buckets


def candidate_pairs(
    M: np.ndarray, n_bits: int = 16, n_tables: int = 4, seed: int = 0
) -> set[tuple[int, int]]:
    """Index pairs that landed in the same bucket in at least one table."""
    Mn = _l2norm(M)
    pairs: set[tuple[int, int]] = set()
    for t in range(n_tables):
        for idxs in lsh_buckets(Mn, n_bits, seed + t).values():
            if len(idxs) > 1:
                for a in range(len(idxs)):
                    for b in range(a + 1, len(idxs)):
                        pairs.add((idxs[a], idxs[b]))
    return pairs


def neighbor_map(
    M: np.ndarray, n_bits: int = 16, n_tables: int = 4, seed: int = 0
) -> dict[int, set[int]]:
    """For each index, the set of bucket-mate candidate neighbours."""
    Mn = _l2norm(M)
    nbrs: dict[int, set[int]] = defaultdict(set)
    for t in range(n_tables):
        for idxs in lsh_buckets(Mn, n_bits, seed + t).values():
            s = set(idxs)
            for i in idxs:
                nbrs[i] |= s - {i}
    return nbrs
