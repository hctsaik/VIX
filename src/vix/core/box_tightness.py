"""Pixel-level box-tightness scoring (opt-in; the SAM mask is supplied by the pipeline).

`box_qa` audits box GEOMETRY (degenerate / truncated / area·aspect outliers) but cannot tell whether a
well-shaped GT box is actually TIGHT around the object — that needs pixels. Given a segmentation mask
(from an external SAM), this scores how much a GT box disagrees with the object's true extent: a low
IoU between the GT box and the mask's own tight box means a loose / misaligned annotation.

Pure (stdlib only): no torch / no SAM here — the pipeline does the SAM inference and hands the mask's
tight box in. HONEST: the mask is itself a model's guess, so this is a PROXY suspicion to review, never
an auto-edit.
"""

from __future__ import annotations


def iou_cxcywh(a, b) -> float:
    """IoU of two normalised (cx, cy, w, h) boxes."""
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def tightness(gt_box, mask_box, iou_thr: float = 0.6) -> dict:
    """gt_box / mask_box = normalised (cx, cy, w, h). Returns the IoU of the GT box vs the object mask's
    tight box, both areas, a `loose` flag (IoU < iou_thr → the annotation doesn't hug the object), and a
    plain-language reason. The mask box being much smaller than the GT box ⇒ the GT box is too loose."""
    iou = iou_cxcywh(gt_box, mask_box)
    ga, ma = gt_box[2] * gt_box[3], mask_box[2] * mask_box[3]
    loose = iou < iou_thr
    why = (f"GT 框與物件遮罩貼合度低(IoU={iou:.2f}<{iou_thr};GT 面積 {ga:.4f} vs 遮罩 {ma:.4f})→ 框可能太鬆/沒對齊"
           if loose else f"框貼合物件(IoU={iou:.2f})")
    return {"iou": round(iou, 4), "gt_area": round(ga, 4), "mask_area": round(ma, 4),
            "loose": bool(loose), "why": why}
