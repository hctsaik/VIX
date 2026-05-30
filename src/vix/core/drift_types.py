"""Drift type diagnosis (concept #3) — covariate vs concept/label drift.

VIX detects *that* something drifted; this says *what kind*, because the fix
differs: input-distribution (covariate) shift → collect/augment data; output/
label-distribution (concept) shift → retrain or audit annotation. Separating
them is what lets you answer "why did adding this batch help (or not)?".
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from .scorer import _l2norm


def _label_dist(labels, classes) -> np.ndarray:
    c = Counter(labels)
    total = sum(c.values()) or 1
    return np.array([c.get(k, 0) / total for k in classes], dtype=float)


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p, q = p + 1e-12, q + 1e-12
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(a * np.log(a / b)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def covariate_shift(ref_emb: np.ndarray, new_emb: np.ndarray) -> float:
    """Cosine shift between the mean (normalised) embeddings of the two batches."""
    if ref_emb.size == 0 or new_emb.size == 0:
        return 0.0
    a = _l2norm(np.asarray(ref_emb, float).mean(axis=0, keepdims=True)).ravel()
    b = _l2norm(np.asarray(new_emb, float).mean(axis=0, keepdims=True)).ravel()
    return float(1.0 - float(np.dot(a, b)))


def diagnose_drift_type(
    ref_emb,
    new_emb,
    ref_labels,
    new_labels,
    cov_threshold: float = 0.15,
    pred_threshold: float = 0.05,
) -> dict:
    classes = sorted(set(ref_labels) | set(new_labels))
    cov = covariate_shift(np.asarray(ref_emb, float), np.asarray(new_emb, float))
    pred = _js_divergence(_label_dist(ref_labels, classes), _label_dist(new_labels, classes))
    is_cov, is_pred = cov > cov_threshold, pred > pred_threshold

    if is_cov and is_pred:
        verdict, action = "both", "輸入與輸出皆漂移:先查輸入(光源/批號),再評估重訓"
    elif is_cov:
        verdict, action = "covariate", "輸入分布漂移(covariate):補/換代表性資料,通常不必重訓標準"
    elif is_pred:
        verdict, action = "concept", "輸出/標籤分布漂移(concept):查標註定義一致性,必要時重訓"
    else:
        verdict, action = "none", "無顯著漂移"
    return {
        "covariate_shift": round(cov, 4),
        "prediction_shift": round(pred, 4),
        "verdict": verdict,
        "recommended_action": action,
    }
