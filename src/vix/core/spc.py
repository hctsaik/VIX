"""SPC control charts (concept #4) — EWMA / CUSUM as a leading FA indicator.

A fixed threshold on FA only fires after FA is already bad. EWMA and CUSUM catch
small, gradual drifts several batches earlier, and have a long regulatory track
record in semiconductor SPC. Run them on VIX's per-batch FA / review-rate series.
Pure numpy.
"""

from __future__ import annotations

import numpy as np


def ewma(series, lam: float = 0.3) -> np.ndarray:
    series = np.asarray(series, dtype=float)
    out = np.empty_like(series)
    prev = series[0] if series.size else 0.0
    for i, v in enumerate(series):
        prev = lam * v + (1 - lam) * prev
        out[i] = prev
    return out


def ewma_alarm(series, target: float, sigma: float, lam: float = 0.3, L: float = 3.0) -> dict:
    """First index where the EWMA leaves target ± L·sigma·sqrt(lam/(2-lam))."""
    z = ewma(series, lam)
    half = L * sigma * np.sqrt(lam / (2 - lam))
    upper, lower = target + half, target - half
    breaches = np.where((z > upper) | (z < lower))[0]
    return {
        "ewma": z.tolist(),
        "upper": upper,
        "lower": lower,
        "alarm": bool(breaches.size),
        "alarm_index": int(breaches[0]) if breaches.size else None,
    }


def cusum_alarm(series, target: float, k: float, h: float) -> dict:
    """Two-sided CUSUM; alarms when the cumulative deviation exceeds h."""
    series = np.asarray(series, dtype=float)
    sp = sn = 0.0
    for i, v in enumerate(series):
        sp = max(0.0, sp + (v - target) - k)
        sn = max(0.0, sn + (target - v) - k)
        if sp > h or sn > h:
            return {"alarm": True, "alarm_index": i, "s_pos": sp, "s_neg": sn}
    return {"alarm": False, "alarm_index": None, "s_pos": sp, "s_neg": sn}
