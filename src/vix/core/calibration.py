"""Confidence calibration (concept #1) — Temperature Scaling + ECE.

YOLO confidence is usually over-confident, yet VIX's thresholds, quality score,
harmful ranking and gate all read it. Calibrating once makes "confidence 0.8"
actually mean ~80% correct, so every downstream module stands on firmer ground.
Pure numpy (no scipy): T is fit by ternary search on the NLL.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-6


def _clip(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def apply_temperature(conf, temperature: float):
    """Re-scale probabilities by temperature T (T>1 softens over-confidence)."""
    p = _clip(conf)
    z = np.log(p / (1.0 - p)) / max(temperature, _EPS)
    out = _sigmoid(z)
    return float(out) if np.isscalar(conf) or np.ndim(conf) == 0 else out


def _nll(conf: np.ndarray, correct: np.ndarray, t: float) -> float:
    p = _clip(apply_temperature(conf, t))
    return float(-np.mean(correct * np.log(p) + (1 - correct) * np.log(1.0 - p)))


def fit_temperature(conf, correct, lo: float = 0.05, hi: float = 10.0, iters: int = 60) -> float:
    """Fit the temperature that minimises NLL on (confidence, correct) pairs."""
    conf = _clip(conf)
    correct = np.asarray(correct, dtype=float)
    if conf.size == 0:
        return 1.0
    for _ in range(iters):  # ternary search (NLL is unimodal in T)
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if _nll(conf, correct, m1) < _nll(conf, correct, m2):
            hi = m2
        else:
            lo = m1
    return round((lo + hi) / 2, 4)


def expected_calibration_error(conf, correct, n_bins: int = 10) -> float:
    """ECE: average gap between confidence and accuracy across bins."""
    conf = _clip(conf)
    correct = np.asarray(correct, dtype=float)
    if conf.size == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (conf > bins[i]) & (conf <= bins[i + 1])
        if mask.any():
            ece += mask.mean() * abs(conf[mask].mean() - correct[mask].mean())
    return float(ece)
