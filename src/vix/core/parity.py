"""Cross-group performance parity (concept #5).

Same model across 20 fabs: a healthy fleet average can hide a few fabs doing
badly. This flags groups whose metric deviates from the fleet median beyond a
relative threshold — the cheap, high-value guard against "mean hides the worst".
Pure numpy; group = any tag namespace (e.g. ``fab:F3``).
"""

from __future__ import annotations

import numpy as np


def performance_parity(
    group_values: dict[str, float], rel_threshold: float = 0.1, lower_is_worse: bool = True
) -> dict:
    """group_values: {group -> metric (e.g. mean confidence / CR)}.

    Returns the fleet median, per-group deviation, and the groups flagged as
    significantly worse than the median.
    """
    if not group_values:
        return {"median": 0.0, "groups": {}, "flagged": []}
    median = float(np.median(list(group_values.values())))
    groups, flagged = {}, []
    for g, v in group_values.items():
        rel = (v - median) / median if median else 0.0
        worse = (rel < -rel_threshold) if lower_is_worse else (rel > rel_threshold)
        groups[g] = {"value": float(v), "rel_to_median": round(rel, 4), "worse": bool(worse)}
        if worse:
            flagged.append(g)
    flagged.sort(key=lambda g: groups[g]["rel_to_median"], reverse=not lower_is_worse)
    return {"median": median, "groups": groups, "flagged": flagged}
