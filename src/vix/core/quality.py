"""Time/reviewer quality analytics.

reviewer_consistency  (U2)  same reviewer, similar samples, opposite decisions
class_quality_trend   (U10) per-class per-batch confidence trend + significant drop
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .analytics import EmbItem
from .scorer import _l2norm


def reviewer_consistency(
    decisions: list[dict], items: list[EmbItem], sim_threshold: float = 0.9,
    label_filter: str | None = None,
) -> dict:
    """Per-reviewer intra-consistency: do similar samples get the same decision?

    ``decisions``: list of {"reviewer_id", "id", "decision"}; ``items`` supply
    embeddings keyed by the same id. ``label_filter`` restricts to one class.
    """
    emb = {
        it.id: _l2norm(np.asarray(it.embedding, float).reshape(1, -1)).ravel()
        for it in items
        if label_filter is None or it.label == label_filter
    }
    by_rev: dict[str, list[dict]] = defaultdict(list)
    for d in decisions:
        if d.get("id") in emb:
            by_rev[d.get("reviewer_id", "")].append(d)

    out: dict[str, dict] = {}
    for rev, ds in by_rev.items():
        conflicts, pairs, consistent = [], 0, 0
        for a in range(len(ds)):
            for b in range(a + 1, len(ds)):
                ia, ib = ds[a]["id"], ds[b]["id"]
                if float(emb[ia] @ emb[ib]) >= sim_threshold:
                    pairs += 1
                    if ds[a]["decision"] == ds[b]["decision"]:
                        consistent += 1
                    else:
                        conflicts.append((ia, ib))
        out[rev] = {
            "intra_consistency": consistent / pairs if pairs else 1.0,
            "similar_pairs": pairs,
            "conflicts": conflicts,
        }
    return out


def class_quality_trend(
    items: list[EmbItem], batch_order: list[str] | None = None, drop_threshold: float = 0.15
) -> dict:
    """Per-class mean-confidence trend across batches + significant-drop alerts."""
    by_batch: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for it in items:
        by_batch[it.batch][it.label].append(it.confidence)
    batches = batch_order or sorted(b for b in by_batch if b != "")
    classes = sorted({it.label for it in items})

    trend: dict[str, list[dict]] = {}
    alerts: list[dict] = []
    for c in classes:
        series = []
        for idx, b in enumerate(batches):
            confs = by_batch[b].get(c, [])
            mean = float(np.mean(confs)) if confs else None
            series.append({"batch": b, "mean_conf": mean, "n": len(confs)})
            if idx > 0:
                prev = series[idx - 1]["mean_conf"]
                if prev is not None and mean is not None and (prev - mean) > drop_threshold:
                    alerts.append({"class": c, "batch": b, "drop": round(prev - mean, 4)})
        trend[c] = series
    return {"trend": trend, "alerts": alerts}
