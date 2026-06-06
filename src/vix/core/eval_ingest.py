"""Eval ingestion — close the data <-> model loop (roadmap #1, keystone).

VIX is otherwise model-blind: every signal (routing, coverage, harmful, active-
learn) is computed from YOLO confidence + DINOv2 distance, never from what the
*trained* model actually got wrong on a held-out validation set. This module
ingests a val evaluation (ground truth + predictions, matched at IoU) and
produces the model-validated quantities VIX needs:

  * per-class AP@IoU and mAP  (which classes actually under-perform)
  * a confusion matrix         (which class pairs the model conflates)
  * per-image FP / FN / TP     (which exact images the model fails on)

Pure / numpy-free at the core (stdlib only) and unit-testable. Boxes are YOLO
normalised (cx, cy, w, h). No FiftyOne, no torch.
"""

from __future__ import annotations

from collections import defaultdict


def iou(a: tuple, b: tuple) -> float:
    """IoU of two normalised (cx, cy, w, h) boxes."""
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def _match_image(gts: list[dict], preds: list[dict], thr: float):
    """Match one image's preds to its GTs (COCO-style, class-specific).

    Returns (per_pred [(conf, label, is_tp)], fn_labels, confusion_pairs, n_fp, n_fn).
    A pred is a TP only if it matches an unmatched GT of the SAME class at IoU>=thr
    (class-specific). Unmatched preds are FPs; a wrong-class overlap with an unmatched
    GT is recorded as a confusion (pred_label, gt_label) but the GT still counts as a
    FN for its true class (the model failed to detect it correctly).
    """
    preds = sorted(preds, key=lambda p: -float(p.get("conf", 0.0)))
    used = [False] * len(gts)
    per_pred, fp_boxes = [], []
    for p in preds:
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
            fp_boxes.append((pl, pb))
    # confusion: an FP pred overlapping an unmatched GT of a DIFFERENT class (GT still an FN)
    confusion = []
    for pl, pb in fp_boxes:
        for j, g in enumerate(gts):
            if not used[j] and g["label"] != pl and iou(pb, tuple(g["bbox"])) >= thr:
                confusion.append((pl, g["label"]))
                break
    fn_labels = [gts[j]["label"] for j in range(len(gts)) if not used[j]]
    return per_pred, fn_labels, confusion, len(fp_boxes), len(fn_labels)


def _ap(pred_flags: list[tuple], n_gt: int) -> float:
    """All-point AP from [(conf, is_tp)] sorted by conf desc, given n_gt positives."""
    if n_gt == 0:
        return 0.0
    order = sorted(pred_flags, key=lambda x: -x[0])
    tp = fp = 0
    prev_recall = 0.0
    ap = 0.0
    max_prec_at = []  # (recall, precision) points
    for _conf, is_tp in order:
        if is_tp:
            tp += 1
        else:
            fp += 1
        recall = tp / n_gt
        precision = tp / (tp + fp)
        max_prec_at.append((recall, precision))
    # all-point interpolation: integrate the precision envelope over recall
    best_prec = 0.0
    for recall, precision in reversed(max_prec_at):
        best_prec = max(best_prec, precision)
        # accumulate later; we redo forward for clarity below
    # forward pass with monotone-decreasing-from-right envelope
    envelope, run = [], 0.0
    for recall, precision in reversed(max_prec_at):
        run = max(run, precision)
        envelope.append((recall, run))
    envelope.reverse()
    prev_recall = 0.0
    for recall, precision in envelope:
        ap += precision * (recall - prev_recall)
        prev_recall = recall
    return round(ap, 4)


def evaluate(images: list[dict], iou_thr: float = 0.5) -> dict:
    """images: [{vix_hash, gt:[{label,bbox}], pred:[{label,bbox,conf}]}].

    Returns per_class_ap, mAP, confusion (pair->count), and per-image fp/fn so the
    caller can attach fields + drive error mining.
    """
    by_class_preds: dict[str, list[tuple]] = defaultdict(list)
    n_gt: dict[str, int] = defaultdict(int)
    confusion: dict[str, int] = defaultdict(int)
    per_image: dict[str, dict] = {}
    fn_hashes: list[str] = []
    fp_hashes: list[str] = []

    for img in images:
        gts = img.get("gt", []) or []
        preds = img.get("pred", []) or []
        for g in gts:
            n_gt[g["label"]] += 1
        per_pred, fn_labels, conf_pairs, n_fp, n_fn = _match_image(gts, preds, iou_thr)
        for conf, label, is_tp in per_pred:
            by_class_preds[label].append((conf, is_tp))
        for pl, gl in conf_pairs:
            confusion[f"{gl}->{pl}"] += 1  # truth gl mis-detected as pl
        h = img.get("vix_hash", "")
        per_image[h] = {"n_fp": n_fp, "n_fn": n_fn}
        if n_fn:
            fn_hashes.append(h)
        if n_fp:
            fp_hashes.append(h)

    classes = sorted(set(n_gt) | set(by_class_preds))
    per_class_ap = {c: _ap(by_class_preds.get(c, []), n_gt.get(c, 0)) for c in classes}
    m_ap = round(sum(per_class_ap.values()) / len(per_class_ap), 4) if per_class_ap else 0.0
    return {
        "iou_thr": iou_thr,
        "mAP": m_ap,
        "per_class_ap": per_class_ap,
        "n_gt": dict(n_gt),
        "confusion": dict(sorted(confusion.items(), key=lambda kv: -kv[1])),
        "per_image": per_image,
        "fn_hashes": fn_hashes,
        "fp_hashes": fp_hashes,
    }
