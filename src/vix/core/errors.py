"""Annotation error diagnosis (T5): classification vs localization.

Compares predictions against ground truth by IoU to split quality problems into
"wrong class, box fine" (classification error — cheap to fix) vs "right class,
box off" (localization error — needs re-drawing). Pure geometry, testable.
"""

from __future__ import annotations

from ..types import BBox, Detection
from .eval_ingest import iou as _iou  # single shared IoU primitive (no duplicate math)


def bbox_iou(a: BBox, b: BBox) -> float:
    return _iou(a.as_tuple(), b.as_tuple())


def diagnose_image(
    preds: list[Detection], gts: list[Detection], loc_iou: float = 0.5, min_overlap: float = 0.1
) -> list[str]:
    """Per-GT verdict: ok | classification_error | localization_error | missed."""
    verdicts: list[str] = []
    for gt in gts:
        best, best_iou = None, 0.0
        for p in preds:
            i = bbox_iou(p.bbox, gt.bbox)
            if i > best_iou:
                best, best_iou = p, i
        if best is None or best_iou < min_overlap:
            verdicts.append("missed")
        elif best_iou >= loc_iou:
            verdicts.append("ok" if best.label == gt.label else "classification_error")
        else:
            verdicts.append("localization_error")
    return verdicts


def diagnose_errors(
    samples: list[tuple[str, list[Detection], list[Detection]]],
    loc_iou: float = 0.5,
) -> dict:
    """Aggregate across (id, preds, gts) samples into actionable lists."""
    classification, localization, missed = [], [], []
    n_ok = 0
    for sid, preds, gts in samples:
        verdicts = diagnose_image(preds, gts, loc_iou)
        if "classification_error" in verdicts:
            classification.append(sid)
        if "localization_error" in verdicts:
            localization.append(sid)
        if "missed" in verdicts:
            missed.append(sid)
        n_ok += verdicts.count("ok")
    return {
        "classification_errors": classification,
        "localization_errors": localization,
        "missed": missed,
        "n_ok": n_ok,
    }
