"""Confidently-wrong (hard-negative) mining — the "YOLO is most confident yet wrong" weakness lens.

Ported as the one genuinely-new idea from the SAFE project (its `hardneg.py`): rank the boxes
where the detector is MOST confident and yet WRONG. That is the highest-value weakness signal —
a confident mistake is both the most damaging error and the cheapest to fix with a few labels.

Two modes, both fully OFFLINE (no training, no YOLO/SAM inference — VIX's locked constraint):

  GT-mode    (`rank_eval_fps`): rank confirmed eval false-positives (from eval_ingest's fp_detail,
             which now carries each FP's confidence) by confidence. Ground-truth wrong; the most
             reliable signal, available when a labelled val set exists.
  GT-free    (`rank_overturns`): rank detections the detector is confident about (conf >= conf_thr)
             but the DINOv2 embedding REJECTS (knn_dist > dist_thr) — "confidently wrong" by the
             embedding prior, needing NO labels. Works on unlabelled incoming data.

`wrongness` is a PROXY weakness / label-priority signal — never proof. Without a retrain it cannot
be claimed that labelling these raises mAP; it ranks the most suspicious confident detections.
Pure / stdlib-only / unit-testable.
"""

from __future__ import annotations

_MAX_COS_DIST = 2.0  # cosine distance upper bound (1 - (-1)); caps an inf knn_dist for a finite margin


def rank_eval_fps(fp_detail: dict, top: int = 50) -> list[dict]:
    """GT-mode: confirmed false positives ranked by confidence (most-confident mistakes first).

    fp_detail: {vix_hash: [{label, bbox, type, conf}]} from eval_ingest. Entries without `conf`
    (e.g. produced before conf was recorded) are skipped rather than guessed."""
    rows: list[dict] = []
    for h, boxes in (fp_detail or {}).items():
        for b in boxes:
            conf = b.get("conf")
            if conf is None:
                continue
            conf = float(conf)
            ftype = b.get("type", "background")
            rows.append({
                "id": h, "pred_class": b.get("label"), "conf": round(conf, 4),
                "fp_type": ftype, "wrongness": round(conf, 4),
                "why": f"YOLO {conf:.2f} 自信卻是 {ftype} 誤報(驗證集 GT 證實錯)",
            })
    rows.sort(key=lambda r: -r["wrongness"])
    return rows[:top]


def rank_overturns(detections, top: int = 50, eps: float = 1e-9) -> list[dict]:
    """GT-free: detections the detector is confident about but the embedding prior overturns.

    detections: iterable of dicts {id, pred_class, conf, knn_dist, conf_thr, dist_thr}.
    An overturn requires BOTH conf >= conf_thr (YOLO passes its own confidence bar) AND
    knn_dist > dist_thr (the embedding says this is far from that class -> outlier). Uncalibrated
    classes (dist_thr inf/<=0) are skipped — we can't judge an overturn without a distance bar.
    wrongness = conf * relative_overshoot, relative_overshoot = (min(knn_dist, 2.0) - dist_thr)/dist_thr."""
    rows: list[dict] = []
    for d in detections:
        conf = float(d.get("conf", 0.0))
        kd = float(d.get("knn_dist", float("inf")))
        ct, dt = d.get("conf_thr"), d.get("dist_thr")
        if ct is None or dt is None:
            continue
        ct, dt = float(ct), float(dt)
        if conf < ct:                       # not YOLO-confident -> not "confidently" wrong
            continue
        if dt == float("inf") or dt <= 0:   # uncalibrated class -> no distance bar to overturn
            continue
        if not (kd > dt):                   # embedding agrees with YOLO -> not an overturn
            continue
        margin = (min(kd, _MAX_COS_DIST) - dt) / (dt + eps)  # relative overshoot past the class ceiling
        wrongness = conf * margin
        kd_str = "inf" if kd == float("inf") else f"{kd:.3f}"
        rows.append({
            "id": d.get("id"), "pred_class": d.get("pred_class"),
            "conf": round(conf, 4), "knn_dist": (None if kd == float("inf") else round(kd, 4)),
            "dist_thr": round(dt, 4), "wrongness": round(float(wrongness), 4),
            "why": f"YOLO {conf:.2f} 自信但嵌入離 {d.get('pred_class')} 太遠(dist {kd_str}>{dt:.3f});疑似自信誤報",
        })
    rows.sort(key=lambda r: -r["wrongness"])
    return rows[:top]
