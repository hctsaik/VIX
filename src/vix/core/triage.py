"""Unified review queue (T3) — one risk ranking from all signals.

Combines uncertainty (1-confidence), novelty (kNN distance to the golden set),
and suspected-label-error into a single risk score so a time-boxed reviewer sees
the highest-value items first. Each item carries the reasons that drove its rank
(which feeds the plain-language explainer, T7).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .analytics import EmbItem
from .scorer import cosine_knn_distance
from ..types import Flag


@dataclass
class RiskItem:
    id: str
    risk: float
    reasons: list[str] = field(default_factory=list)
    scores: dict = field(default_factory=dict)


def review_queue(
    candidates: list[EmbItem],
    reference: list[EmbItem],
    k: int = 10,
    dist_norm: float = 0.5,
    label_issue_ids: set[str] | None = None,
    weights: tuple[float, float, float] = (0.4, 0.4, 0.2),
) -> list[RiskItem]:
    """Rank ``candidates`` by combined risk, highest first."""
    label_issue_ids = label_issue_ids or set()
    w_unc, w_nov, w_lbl = weights
    ref_raw = (
        np.vstack([np.asarray(r.embedding, float) for r in reference]) if reference else None
    )

    out: list[RiskItem] = []
    for c in candidates:
        knn = cosine_knn_distance(c.embedding, ref_raw, k) if ref_raw is not None else float("inf")
        unc = 1.0 - c.confidence
        nov = min(1.0, knn / dist_norm) if np.isfinite(knn) else 1.0
        is_lbl = c.id in label_issue_ids
        risk = w_unc * unc + w_nov * nov + w_lbl * (1.0 if is_lbl else 0.0)

        reasons: list[str] = []
        if unc >= 0.5:
            reasons.append(Flag.LOW_CONF)
        if nov >= 0.5:
            reasons.append(Flag.FAR_FROM_KNOWN)
        if is_lbl:
            reasons.append("suspected_label_error")
        out.append(
            RiskItem(c.id, float(risk), reasons, {"conf": c.confidence, "knn_dist": float(knn)})
        )

    out.sort(key=lambda r: -r.risk)
    return out
