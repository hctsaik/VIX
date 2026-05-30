"""FrozenReference — the un-poisoned anchor that guards the closed loop.

v0.1 uses a frozen *pretrained* DINOv2 embedding space (zero training). Two
signals detect that newly merged data is dragging a class definition around:

    1. centroid_shift   — cosine shift of each class centroid vs the frozen anchor
    2. label_consistency — fraction of samples whose kNN-majority label matches

If a batch exceeds the drift/consistency thresholds, ``guard`` reports
``triggered=True`` so the CLI can require a written self-acknowledgement
(single-person self-gate). The full frozen-reference-YOLO KL gate is v0.2.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .scorer import _l2norm


def class_centroids(per_class_emb: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for c, e in per_class_emb.items():
        e = np.asarray(e, dtype=float)
        if e.size == 0:
            continue
        out[c] = _l2norm(e.mean(axis=0, keepdims=True)).ravel()
    return out


def label_consistency(
    query_per_class: dict[str, np.ndarray],
    ref_per_class: dict[str, np.ndarray],
    k: int = 10,
) -> float:
    """Fraction of query samples whose kNN-majority label (in ref) matches.

    Self-matches (cosine ~1) are excluded so a sample never votes for itself,
    which makes this valid even when query is a subset of ref.
    """
    ref_vecs, ref_labels = [], []
    for c, e in ref_per_class.items():
        e = np.asarray(e, dtype=float)
        if e.size == 0:
            continue
        ref_vecs.append(_l2norm(e))
        ref_labels.extend([c] * e.shape[0])
    if not ref_vecs:
        return 1.0
    R = np.vstack(ref_vecs)
    labels = np.array(ref_labels)

    total = correct = 0
    for c, e in query_per_class.items():
        e = np.asarray(e, dtype=float)
        if e.size == 0:
            continue
        Q = _l2norm(e)
        sims = Q @ R.T
        for i in range(Q.shape[0]):
            order = np.argsort(-sims[i])
            sel = order[1 : k + 1] if sims[i, order[0]] > 0.99999 else order[:k]
            if sel.size == 0:
                continue
            vals, counts = np.unique(labels[sel], return_counts=True)
            maj = vals[int(np.argmax(counts))]
            total += 1
            correct += int(maj == c)
    return correct / total if total else 1.0


@dataclass
class GuardReport:
    max_shift: float
    shifts: dict[str, float]
    consistency: float
    baseline_consistency: float
    consistency_drop: float
    triggered: bool
    reasons: list[str] = field(default_factory=list)


class FrozenReference:
    def __init__(self, anchor_per_class: dict[str, np.ndarray], baseline_consistency: float):
        self.anchor_per_class = {c: np.asarray(e, dtype=float) for c, e in anchor_per_class.items()}
        self.centroids = class_centroids(self.anchor_per_class)
        self.baseline_consistency = float(baseline_consistency)

    @classmethod
    def build(
        cls,
        anchor_per_class: dict[str, np.ndarray],
        golden_per_class: dict[str, np.ndarray] | None = None,
        k: int = 10,
    ) -> "FrozenReference":
        ref = golden_per_class or anchor_per_class
        baseline = label_consistency(anchor_per_class, ref, k)
        return cls(anchor_per_class, baseline)

    def centroid_shift(self, new_per_class: dict[str, np.ndarray]) -> dict[str, float]:
        nc = class_centroids(new_per_class)
        shifts: dict[str, float] = {}
        for c, cent in self.centroids.items():
            if c in nc:
                shifts[c] = float(1.0 - float(np.dot(cent, nc[c])))
        return shifts

    def guard(
        self,
        new_per_class: dict[str, np.ndarray],
        shift_threshold: float = 0.15,
        consistency_drop_threshold: float = 0.05,
        k: int = 10,
    ) -> GuardReport:
        shifts = self.centroid_shift(new_per_class)
        max_shift = max(shifts.values()) if shifts else 0.0
        cons = (
            label_consistency(new_per_class, self.anchor_per_class, k)
            if new_per_class
            else self.baseline_consistency
        )
        drop = self.baseline_consistency - cons
        reasons: list[str] = []
        if max_shift > shift_threshold:
            reasons.append("centroid_shift")
        if drop > consistency_drop_threshold:
            reasons.append("consistency_drop")
        return GuardReport(
            max_shift=max_shift,
            shifts=shifts,
            consistency=cons,
            baseline_consistency=self.baseline_consistency,
            consistency_drop=drop,
            triggered=bool(reasons),
            reasons=reasons,
        )

    # --- persistence (npz, no pickle) ---
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f"cls::{c}": e for c, e in self.anchor_per_class.items()}
        arrays["__baseline__"] = np.array([self.baseline_consistency], dtype=float)
        arrays["__classes__meta"] = np.frombuffer(
            json.dumps(list(self.anchor_per_class.keys())).encode("utf-8"), dtype=np.uint8
        )
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "FrozenReference":
        path = Path(path)
        # np.savez appends .npz if missing
        if not path.exists() and path.with_suffix(".npz").exists():
            path = path.with_suffix(".npz")
        data = np.load(path, allow_pickle=False)
        anchor = {
            key[len("cls::") :]: data[key] for key in data.files if key.startswith("cls::")
        }
        baseline = float(data["__baseline__"][0]) if "__baseline__" in data.files else 1.0
        return cls(anchor, baseline)
