"""Per-box static quality QA (box-qa, model-loop-v2 T1c).

Pure geometry over normalised BBoxes. Catches what aggregate geometry-drift and embedding
kNN cannot: an individual malformed annotation box. Box tightness is the ceiling on
mAP@0.75:0.95; a degenerate (w·h≈0) box injects NaN/zero-area targets; an edge-truncated
box should usually be `ignore` rather than a positive. Read-only — returns a ranked issue
list, never mutates. Per-class area/aspect envelopes are only built where there is enough
support to be trustworthy (else a 4-box class would generate noise).
"""

from __future__ import annotations

import numpy as np

_SEV = {"degenerate": 0, "truncated": 1, "area_outlier": 2, "aspect_outlier": 3}


def audit_boxes(
    records: list[dict],
    *,
    area_eps: float = 1e-4,
    lin_eps: float = 1e-3,
    edge_eps: float = 1e-3,
    min_support: int = 8,
    lo_pct: float = 1.0,
    hi_pct: float = 99.0,
) -> list[dict]:
    """records: [{"id", "label", "bbox": (cx, cy, w, h)}]. Returns ranked issues
    [{"id","label","issue","why"}] (most severe first). One issue per box (degenerate
    dominates; a truncated box isn't also flagged as an outlier)."""
    by_class: dict[str, list[dict]] = {}
    for r in records:
        by_class.setdefault(r["label"], []).append(r)

    env: dict[str, dict] = {}  # per-class [p_lo, p_hi] envelopes, only where support suffices
    for label, rs in by_class.items():
        if len(rs) < min_support:
            continue
        areas = np.array([r["bbox"][2] * r["bbox"][3] for r in rs], float)
        asp = np.array([r["bbox"][2] / r["bbox"][3] for r in rs if r["bbox"][3] > 0], float)
        env[label] = {
            "area": (float(np.percentile(areas, lo_pct)), float(np.percentile(areas, hi_pct))),
            "aspect": (float(np.percentile(asp, lo_pct)), float(np.percentile(asp, hi_pct))) if asp.size else None,
        }

    issues: list[dict] = []
    for r in records:
        cx, cy, w, h = r["bbox"]
        rid, label = r["id"], r["label"]
        area = w * h
        if area < area_eps or w < lin_eps or h < lin_eps:
            issues.append({"id": rid, "label": label, "issue": "degenerate",
                           "why": f"框退化 (w={w:.4f}, h={h:.4f}, area={area:.5f});會注入 NaN/零面積目標"})
            continue
        if cx - w / 2 <= edge_eps or cy - h / 2 <= edge_eps or cx + w / 2 >= 1 - edge_eps or cy + h / 2 >= 1 - edge_eps:
            issues.append({"id": rid, "label": label, "issue": "truncated",
                           "why": "框貼邊/越界;通常應設 ignore,而非當作完整正樣本"})
            continue
        e = env.get(label)
        if not e:
            continue
        lo, hi = e["area"]
        if area < lo or area > hi:
            issues.append({"id": rid, "label": label, "issue": "area_outlier",
                           "why": f"面積 {area:.5f} 超出 {label} 類 [p{lo_pct:g},p{hi_pct:g}]=[{lo:.5f},{hi:.5f}]"})
            continue
        if e["aspect"] and h > 0:
            alo, ahi = e["aspect"]
            asp = w / h
            if asp < alo or asp > ahi:
                issues.append({"id": rid, "label": label, "issue": "aspect_outlier",
                               "why": f"長寬比 {asp:.2f} 超出 {label} 類 [p{lo_pct:g},p{hi_pct:g}]=[{alo:.2f},{ahi:.2f}]"})

    issues.sort(key=lambda i: _SEV.get(i["issue"], 9))
    return issues
