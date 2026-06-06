"""Multi-bank Top-K embedding audit (the bank-audit keystone, design of record:
docs/discussion/bank-audit-design.md).

Generalises the single-bank ``OutlierScorer`` to a labelled MULTI-bank Top-K vote:
audit a low-confidence proposal crop's DINOv2 embedding against several reference
banks (e.g. Defect / Reflection / Normal) and return a verdict
(defect_like / reflection_like / normal_like / unknown) plus the Top-K evidence a
human can overturn. Pure / numpy-only / FiftyOne-free — the voter takes pre-stacked
banks + PRE-COMPUTED per-bank scales + a query vector; the pipeline does all I/O.

Key design points (multi-agent consensus):
- cosine distance in raw DINOv2 space (consistent with the rest of VIX).
- per-bank distance calibration: ``s_b = exp(-d_b / scale_b)`` where ``scale_b`` is
  that bank's own leave-one-out median kNN distance (with an eps floor), so a small
  tight bank and a large diffuse bank are comparable — never a single pooled Top-K.
- two abstain gates: novelty radius (far from every bank -> unknown) and a single
  margin knob ``tau`` (top vs runner-up calibrated score too close -> unknown).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..types import Detection
from .eval_ingest import iou
from .scorer import _l2norm, intra_class_knn_distances


@dataclass
class BankVerdict:
    verdict: str                       # defect_like | reflection_like | normal_like | unknown
    winning_bank: str | None
    margin: float                      # top calibrated score minus runner-up
    min_raw_dist: float | None         # min across banks of the mean cosine dist to that bank's k-NN (novelty gate)
    per_bank: dict = field(default_factory=dict)       # {bank: {cal_score, raw_dist, scale}}
    topk_evidence: list = field(default_factory=list)  # winning bank's Top-K [{bank, member_idx, raw_dist}]


def build_bank_scales(banks: dict[str, np.ndarray], k: int = 10, eps_floor: float = 1e-3) -> dict[str, float]:
    """Per-bank calibration scale = max(median(intra-bank LOO kNN distance @k), eps_floor).

    Computed ONCE at build time (not per query). eps_floor guards a near-duplicated
    bank whose LOO scale -> 0 (which would blow up the calibrated score)."""
    scales: dict[str, float] = {}
    for label, emb in banks.items():
        d = intra_class_knn_distances(np.asarray(emb, dtype=float), k)
        med = float(np.median(d)) if d.size else 0.0
        if not math.isfinite(med):  # a NaN/inf embedding must not poison a bank's calibration
            med = 0.0
        scales[label] = max(med, eps_floor)
    return scales


def _topk_to_bank(query: np.ndarray, bank: np.ndarray, k: int):
    """(mean cosine dist of q to its k nearest in bank, sorted member idxs, their dists)."""
    bank = np.asarray(bank, dtype=float)
    if bank.ndim != 2 or bank.shape[0] == 0:
        return float("inf"), [], []
    qv = np.asarray(query, dtype=float).reshape(-1)
    if not np.any(qv) or not np.all(np.isfinite(qv)):  # degenerate/empty embedding -> max distance
        return float("inf"), [], []
    q = _l2norm(qv)
    dists = 1.0 - (_l2norm(bank) @ q)
    kk = min(k, dists.shape[0])
    idx = np.argpartition(dists, kk - 1)[:kk]
    idx = idx[np.argsort(dists[idx])]
    return float(dists[idx].mean()), idx.tolist(), dists[idx].tolist()


def bank_vote(
    query: np.ndarray,
    banks: dict[str, np.ndarray],
    scales: dict[str, float],
    bank_label_map: dict[str, str] | None = None,
    k: int = 10,
    tau: float = 0.10,
    novelty_radius: float = 0.30,
) -> BankVerdict:
    """Vote one query embedding across the banks (see module docstring)."""
    bank_label_map = bank_label_map or {b: b for b in banks}
    per_bank: dict[str, dict] = {}
    for b, emb in banks.items():
        mean_d, idx, dlist = _topk_to_bank(query, emb, k)
        s = math.exp(-mean_d / scales.get(b, 1.0)) if mean_d != float("inf") else 0.0
        per_bank[b] = {
            "cal_score": round(s, 4),
            "raw_dist": (round(mean_d, 4) if mean_d != float("inf") else None),
            "scale": round(scales.get(b, 1.0), 4),
            "_idx": idx, "_dlist": dlist,
        }
    ranked = sorted(per_bank.items(), key=lambda kv: -kv[1]["cal_score"])
    raw = [pb["raw_dist"] for pb in per_bank.values() if pb["raw_dist"] is not None]
    min_raw = min(raw) if raw else float("inf")

    winning_bank, margin, verdict = None, 0.0, "unknown"
    if ranked:
        top_b, top = ranked[0]
        runner = ranked[1][1]["cal_score"] if len(ranked) > 1 else 0.0
        margin = round(top["cal_score"] - runner, 4)
        if min_raw > novelty_radius:       # far from every bank -> novel/unknown
            verdict = "unknown"
        elif margin < tau:                 # too close to call -> abstain
            verdict = "unknown"
        else:
            winning_bank = top_b
            verdict = bank_label_map.get(top_b, top_b)

    topk_evidence = []
    if winning_bank is not None:
        pb = per_bank[winning_bank]
        topk_evidence = [
            {"bank": winning_bank, "member_idx": int(i), "raw_dist": round(float(d), 4)}
            for i, d in zip(pb["_idx"], pb["_dlist"])
        ]
    clean = {b: {kk: vv for kk, vv in pb.items() if not kk.startswith("_")} for b, pb in per_bank.items()}
    return BankVerdict(
        verdict=verdict, winning_bank=winning_bank, margin=margin,
        min_raw_dist=(round(min_raw, 4) if min_raw != float("inf") else None),
        per_bank=clean, topk_evidence=topk_evidence,
    )


def audit_batch(queries, banks, scales, bank_label_map=None, k=10, tau=0.10, novelty_radius=0.30):
    return [bank_vote(q, banks, scales, bank_label_map, k, tau, novelty_radius) for q in queries]


def loose_nms(dets: list[Detection], iou_thr: float = 0.7) -> list[Detection]:
    """Class-agnostic greedy NMS (by descending confidence) to merge near-duplicate
    overlapping low-conf proposals before the audit (NMS-off would explode 1 region
    into many crops). One proposal = one audit unit."""
    order = sorted(range(len(dets)), key=lambda i: -dets[i].confidence)
    suppressed = [False] * len(dets)
    keep: list[Detection] = []
    for pos, i in enumerate(order):
        if suppressed[i]:
            continue
        keep.append(dets[i])
        for j in order[pos + 1:]:
            if not suppressed[j] and iou(dets[i].bbox.as_tuple(), dets[j].bbox.as_tuple()) >= iou_thr:
                suppressed[j] = True
    return keep
