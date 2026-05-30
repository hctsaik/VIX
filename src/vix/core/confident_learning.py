"""Confident Learning (concept #2) — cleanlab-style label-error detection.

Complements the kNN heuristic in analytics: this estimates the **confident
joint** C[given][pred] using per-(predicted-)class confidence thresholds, so it
surfaces *systematic* class-pair mislabeling ("class A is 8% labeled as B") —
directly diagnosing label-definition drift. Pure numpy; needs only top-1
predictions + confidences (no retraining).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

_TOL = 1e-9  # so a sample sitting exactly at the mean-confidence threshold still counts


@dataclass
class CLIssue:
    id: str
    given_label: str
    pred_label: str
    confidence: float


def _class_thresholds(pred, conf, classes):
    """Per predicted-class self-confidence threshold = mean conf of that class."""
    thr = {}
    for c in classes:
        vals = [conf[i] for i in range(len(pred)) if pred[i] == c]
        thr[c] = float(np.mean(vals)) if vals else 1.0
    return thr


def confident_joint(given, pred, conf, classes=None):
    """Return (C, classes, thresholds): C[i][j] = # samples given=i, confidently pred=j."""
    given, pred = list(given), list(pred)
    conf = np.asarray(conf, dtype=float)
    classes = classes or sorted(set(given) | set(pred))
    idx = {c: i for i, c in enumerate(classes)}
    thr = _class_thresholds(pred, conf, classes)
    C = np.zeros((len(classes), len(classes)), dtype=int)
    for g, p, cf in zip(given, pred, conf):
        if cf >= thr[p] - _TOL:
            C[idx[g]][idx[p]] += 1
    return C, classes, thr


def noise_rates(C, classes) -> dict[str, float]:
    """Off-diagonal class-pair noise rate: P(confidently pred=j | given=i)."""
    out = {}
    row_sums = C.sum(axis=1)
    for i, gi in enumerate(classes):
        for j, pj in enumerate(classes):
            if i != j and C[i][j] > 0 and row_sums[i] > 0:
                out[f"{gi}->{pj}"] = round(C[i][j] / row_sums[i], 4)
    return out


def find_label_issues(ids, given, pred, conf, classes=None) -> list[CLIssue]:
    """Samples whose given label differs from a confident prediction, ranked by confidence."""
    _C, classes, thr = confident_joint(given, pred, conf, classes)
    conf = np.asarray(conf, dtype=float)
    issues = [
        CLIssue(ids[i], str(given[i]), str(pred[i]), float(conf[i]))
        for i in range(len(ids))
        if given[i] != pred[i] and conf[i] >= thr[pred[i]] - _TOL
    ]
    issues.sort(key=lambda x: -x.confidence)
    return issues
