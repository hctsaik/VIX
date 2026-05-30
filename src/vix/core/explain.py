"""Plain-language explanations (T7) — turn reason codes into a sentence a
non-ML stakeholder understands.
"""

from __future__ import annotations

_ZH = {
    "low_conf": "模型信心偏低(低於同類別第 5 百分位)",
    "far_from_known": "外觀與已知該類別樣本差異大(嵌入距離超過第 95 百分位)",
    "low_support": "此類別的已知樣本太少,判斷不夠可靠",
    "suspected_label_error": "鄰近樣本多數屬於其他類別,疑似標錯",
    "no_detection": "模型沒有偵測到任何物件",
}


def explain_image(
    label: str,
    confidence: float,
    knn_dist: float,
    conf_thr: float | None = None,
    dist_thr: float | None = None,
    label_issue: bool = False,
) -> dict:
    """Per-image drill-down (U9): each axis's value vs threshold + sensitivity
    ("how much would need to change to flip the decision")."""
    axes: list[dict] = []
    if conf_thr is not None:
        fail = confidence < conf_thr
        axes.append({
            "axis": "confidence",
            "value": confidence,
            "threshold": conf_thr,
            "fail": fail,
            "sensitivity": (f"信心再 +{conf_thr - confidence:.2f} 即通過"
                            if fail else f"高於門檻 {confidence - conf_thr:.2f}"),
        })
    if dist_thr is not None:
        fail = knn_dist > dist_thr
        axes.append({
            "axis": "knn_dist",
            "value": knn_dist,
            "threshold": dist_thr,
            "fail": fail,
            "sensitivity": (f"距離再 -{knn_dist - dist_thr:.3f} 即通過"
                            if fail else f"低於門檻 {dist_thr - knn_dist:.3f}"),
        })
    axes.append({"axis": "label_consistency", "value": label_issue, "fail": bool(label_issue)})
    failing = [a["axis"] for a in axes if a.get("fail")]
    calibrated = conf_thr is not None or dist_thr is not None
    summary = "通過" if not failing else "被攔,主要因:" + ", ".join(failing)
    if not calibrated:
        summary += "(尚未 calibrate:門檻未知,無法計算敏感度,請先執行 vix calibrate)"
    return {
        "label": label,
        "axes": axes,
        "failing_axes": failing,
        "calibrated": calibrated,
        "summary": summary,
    }


def explain(reasons: list[str], scores: dict | None = None) -> str:
    """Return a one-sentence Chinese explanation of why an image was flagged."""
    if not reasons:
        return "通過:信心足夠且與已知樣本相似,可直接納入。"
    parts = "; ".join(_ZH.get(r, r) for r in reasons)
    sentence = f"此影像被攔下覆核,原因:{parts}。"
    if scores:
        conf = scores.get("conf", scores.get("conf_max"))
        dist = scores.get("knn_dist")
        bits = []
        if conf is not None:
            bits.append(f"信心 {conf:.2f}")
        if dist is not None and dist == dist:  # not NaN
            bits.append(f"距離 {dist:.3f}")
        if bits:
            sentence += "(" + ", ".join(bits) + ")"
    return sentence
