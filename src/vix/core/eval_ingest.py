"""Eval ingestion — close the data <-> model loop (roadmap #1, keystone; deepened in
model-loop v2, design of record: docs/discussion/model-loop-v2-design.md).

VIX is otherwise model-blind: every signal (routing, coverage, harmful, active-
learn) is computed from YOLO confidence + DINOv2 distance, never from what the
*trained* model actually got wrong on a held-out validation set. This module
ingests a val evaluation (ground truth + predictions, matched at IoU) and
produces the model-validated quantities VIX needs:

  * per-class AP@IoU and mAP  (which classes actually under-perform)
  * an IoU sweep (mAP@0.5 / @0.75 + loc_gap) so the localization tail is visible
  * a confusion matrix         (which class pairs the model conflates)
  * per-image FP / FN counts   (which exact images the model fails on)
  * TYPED per-error detail      (fp_detail / fn_detail): each error is classified
    classification / localization / missed / background and reported ONCE (a
    same-class loose box is an FN-localization, not also a background FP), so the
    detail drives error-mining and diagnosis without double counting.

Pure / numpy-free (stdlib only) and unit-testable. Boxes are YOLO normalised
(cx, cy, w, h). No FiftyOne, no torch.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict


def eval_set_hash(images: list[dict]) -> str:
    """Content hash over (vix_hash + sorted GT label/box) only. Changes iff the eval SET
    itself changes (samples added/removed, or GT relabeled/moved) — so a swapped or quietly
    relabeled eval set invalidates a baseline mAP comparison (model-loop-v2 R6). Predictions
    are deliberately excluded (they're what we're measuring)."""
    canon = sorted(
        [
            img.get("vix_hash", ""),
            sorted(
                [g["label"], [round(float(x), 6) for x in g["bbox"]]]
                for g in (img.get("gt", []) or [])
            ),
        ]
        for img in images
    )
    return hashlib.sha256(json.dumps(canon, sort_keys=True).encode()).hexdigest()[:16]


def iou(a: tuple, b: tuple) -> float:
    """IoU of two normalised (cx, cy, w, h) boxes."""
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def _match_image(gts: list[dict], preds: list[dict], thr: float, loc_band: float = 0.1):
    """Match one image's preds to its GTs (COCO-style, class-specific) and TYPE the errors.

    A pred is a TP only if it matches an unmatched GT of the SAME class at IoU>=thr.
    Returns (per_pred [(conf,label,is_tp)], fn_labels, confusion_pairs, n_fp, fp_detail, fn_detail):

      * fn_detail: one entry per unmatched GT, typed (see below). Covers every FN.
      * fp_detail: ONLY standalone `background` FPs (hallucination / duplicate). An FP that
        is the partner of an FN (a wrong-class box over an unmatched GT = classification, or
        a same-class loose box in [loc_band,thr) = localization) is suppressed here and
        reported on the FN side, so one physical error is never counted twice.
      * n_fp stays the count of ALL false-positive preds (back-compat with the eval_fp field).

    FN type:  classification (a different-class pred covers it at >=thr)
            | localization (a same-class pred covers it at IoU in [loc_band,thr))
            | missed (no pred overlaps it meaningfully).
    """
    preds = sorted(preds, key=lambda p: -float(p.get("conf", 0.0)))
    used = [False] * len(gts)
    per_pred, fp_idx = [], []
    for pidx, p in enumerate(preds):
        pb, pl = tuple(p["bbox"]), p["label"]
        best_j, best_iou = -1, thr
        for j, g in enumerate(gts):
            if used[j] or g["label"] != pl:  # class-specific
                continue
            ov = iou(pb, tuple(g["bbox"]))
            if ov >= best_iou:
                best_iou, best_j = ov, j
        if best_j >= 0:
            used[best_j] = True
            per_pred.append((float(p.get("conf", 0.0)), pl, True))
        else:
            per_pred.append((float(p.get("conf", 0.0)), pl, False))  # FP
            fp_idx.append(pidx)

    # FN typing (one verdict per unmatched GT)
    fn_labels, fn_detail = [], []
    for j, g in enumerate(gts):
        if used[j]:
            continue
        gl, gb = g["label"], tuple(g["bbox"])
        fn_labels.append(gl)
        diff_ge_thr = same_band = False
        for p in preds:
            ov = iou(tuple(p["bbox"]), gb)
            if p["label"] != gl and ov >= thr:
                diff_ge_thr = True
            elif p["label"] == gl and loc_band <= ov < thr:
                same_band = True
        t = "classification" if diff_ge_thr else "localization" if same_band else "missed"
        fn_detail.append({"label": gl, "bbox": list(gb), "type": t})

    # confusion: an FP pred overlapping an unmatched GT of a DIFFERENT class (GT still an FN)
    confusion = []
    for pidx in fp_idx:
        pl, pb = preds[pidx]["label"], tuple(preds[pidx]["bbox"])
        for j, g in enumerate(gts):
            if not used[j] and g["label"] != pl and iou(pb, tuple(g["bbox"])) >= thr:
                confusion.append((pl, g["label"]))
                break

    # FP typing: keep only standalone background FPs (paired loc/cls FPs live on the FN side)
    fp_detail = []
    for pidx in fp_idx:
        pl, pb = preds[pidx]["label"], tuple(preds[pidx]["bbox"])
        paired = False
        for j, g in enumerate(gts):
            if used[j]:
                continue
            ov = iou(pb, tuple(g["bbox"]))
            if (g["label"] != pl and ov >= thr) or (g["label"] == pl and loc_band <= ov < thr):
                paired = True  # this FP is an FN's partner -> reported there, not here
                break
        if not paired:
            fp_detail.append({"label": pl, "bbox": list(pb), "type": "background",
                              "conf": round(float(preds[pidx].get("conf", 0.0)), 4)})  # for confidently-wrong mining

    return per_pred, fn_labels, confusion, len(fp_idx), fp_detail, fn_detail


def _ap(pred_flags: list[tuple], n_gt: int) -> float:
    """All-point AP from [(conf, is_tp)] given n_gt positives."""
    if n_gt == 0:
        return 0.0
    order = sorted(pred_flags, key=lambda x: -x[0])
    tp = fp = 0
    points = []  # (recall, precision)
    for _conf, is_tp in order:
        if is_tp:
            tp += 1
        else:
            fp += 1
        points.append((tp / n_gt, tp / (tp + fp)))
    # all-point interpolation: monotone-decreasing-from-right precision envelope, integrated over recall
    envelope, run = [], 0.0
    for recall, precision in reversed(points):
        run = max(run, precision)
        envelope.append((recall, run))
    envelope.reverse()
    ap, prev_recall = 0.0, 0.0
    for recall, precision in envelope:
        ap += precision * (recall - prev_recall)
        prev_recall = recall
    return round(ap, 4)


def _map_at(images: list[dict], thr: float, n_gt: dict, loc_band: float) -> float:
    """mAP at a single IoU threshold (re-matches: a TP@0.5 may be an FP@0.75)."""
    by_class: dict[str, list[tuple]] = defaultdict(list)
    for img in images:
        per_pred, *_ = _match_image(img.get("gt", []) or [], img.get("pred", []) or [], thr, loc_band)
        for conf, label, is_tp in per_pred:
            by_class[label].append((conf, is_tp))
    classes = sorted(set(n_gt) | set(by_class))
    if not classes:
        return 0.0
    return round(sum(_ap(by_class.get(c, []), n_gt.get(c, 0)) for c in classes) / len(classes), 4)


def evaluate(
    images: list[dict], iou_thr: float = 0.5, loc_band: float = 0.1, sweep: tuple = (0.5, 0.75)
) -> dict:
    """images: [{vix_hash, gt:[{label,bbox}], pred:[{label,bbox,conf}]}].

    Returns per_class_ap, mAP (@iou_thr), an IoU sweep (map_by_iou + loc_gap), confusion,
    per-image fp/fn counts, and TYPED fp_detail/fn_detail so the caller can attach fields,
    drive error mining, and see the localization tail. All keys are additive over v1.
    """
    by_class_preds: dict[str, list[tuple]] = defaultdict(list)
    n_gt: dict[str, int] = defaultdict(int)
    confusion: dict[str, int] = defaultdict(int)
    per_image: dict[str, dict] = {}
    fn_hashes: list[str] = []
    fp_hashes: list[str] = []
    fp_detail: dict[str, list] = {}
    fn_detail: dict[str, list] = {}

    for img in images:
        gts = img.get("gt", []) or []
        preds = img.get("pred", []) or []
        for g in gts:
            n_gt[g["label"]] += 1
        per_pred, fn_labels, conf_pairs, n_fp, fpd, fnd = _match_image(gts, preds, iou_thr, loc_band)
        for conf, label, is_tp in per_pred:
            by_class_preds[label].append((conf, is_tp))
        for pl, gl in conf_pairs:
            confusion[f"{gl}->{pl}"] += 1  # truth gl mis-detected as pl
        h = img.get("vix_hash", "")
        per_image[h] = {"n_fp": n_fp, "n_fn": len(fn_labels)}
        if fn_labels:
            fn_hashes.append(h)
        if n_fp:
            fp_hashes.append(h)
        if fpd:
            fp_detail[h] = fpd
        if fnd:
            fn_detail[h] = fnd

    classes = sorted(set(n_gt) | set(by_class_preds))
    per_class_ap = {c: _ap(by_class_preds.get(c, []), n_gt.get(c, 0)) for c in classes}
    m_ap = round(sum(per_class_ap.values()) / len(per_class_ap), 4) if per_class_ap else 0.0

    # IoU sweep (fleet-level localization signal; not per-class causal)
    map_by_iou = {t: _map_at(images, t, n_gt, loc_band) for t in sorted(set(sweep) | {iou_thr})}
    loc_gap = (
        round(map_by_iou[0.5] - map_by_iou[0.75], 4) if (0.5 in map_by_iou and 0.75 in map_by_iou) else None
    )
    return {
        "iou_thr": iou_thr,
        "mAP": m_ap,
        "per_class_ap": per_class_ap,
        "map_by_iou": map_by_iou,
        "loc_gap": loc_gap,
        "n_gt": dict(n_gt),
        "confusion": dict(sorted(confusion.items(), key=lambda kv: -kv[1])),
        "per_image": per_image,
        "fn_hashes": fn_hashes,
        "fp_hashes": fp_hashes,
        "fp_detail": fp_detail,
        "fn_detail": fn_detail,
    }
