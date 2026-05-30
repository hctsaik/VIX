"""BBox geometry stats + drift (W3/W4 complement to embedding-space drift).

Catches annotation-behaviour changes that embeddings miss — e.g. a guideline
tweak that suddenly makes boxes much smaller or changes aspect ratios.
"""

from __future__ import annotations

import numpy as np

from ..types import Detection


def bbox_geometry_stats(dets: list[Detection]) -> dict:
    if not dets:
        return {}
    w = np.array([d.bbox.w for d in dets], dtype=float)
    h = np.array([d.bbox.h for d in dets], dtype=float)
    aspect = w / np.clip(h, 1e-9, None)

    def summary(a: np.ndarray) -> dict:
        return {
            "mean": float(a.mean()),
            "p05": float(np.percentile(a, 5)),
            "p95": float(np.percentile(a, 95)),
        }

    return {"n": len(dets), "w": summary(w), "h": summary(h), "aspect": summary(aspect)}


def geometry_drift(
    dets_a: list[Detection], dets_b: list[Detection], shift_threshold: float = 0.2
) -> dict:
    sa, sb = bbox_geometry_stats(dets_a), bbox_geometry_stats(dets_b)
    if not sa or not sb:
        return {"alert": False, "shifts": {}}
    shifts = {k: abs(sa[k]["mean"] - sb[k]["mean"]) for k in ("w", "h", "aspect")}
    # also compare tail of the distribution, not just the mean
    shifts.update({f"{k}_p95": abs(sa[k]["p95"] - sb[k]["p95"]) for k in ("w", "h")})
    return {"alert": any(v > shift_threshold for v in shifts.values()), "shifts": shifts, "a": sa, "b": sb}
