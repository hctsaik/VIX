"""T1a: error-typed eval matcher + IoU sweep (model-loop-v2 design §6 R2/R3/R5).

Asserts the typed fp_detail/fn_detail classify each error ONCE (no double-counting of a
same-class loose box as both background-FP and localization-FN), and that the IoU sweep
surfaces a localization gap on systematically loose boxes. errors.py keeps its legacy API.
"""

from vix.core.eval_ingest import evaluate, precision_recall_f1
from vix.core.errors import bbox_iou, diagnose_image
from vix.types import BBox


def _types(detail_for_image):
    return sorted(d["type"] for d in detail_for_image)


def test_four_distinct_error_types_no_double_count():
    # one image, four GTs, deliberately one of each failure mode:
    images = [{
        "vix_hash": "img",
        "gt": [
            {"label": "cat", "bbox": [0.20, 0.20, 0.10, 0.10]},  # A: localization (same-class loose pred)
            {"label": "cat", "bbox": [0.50, 0.50, 0.20, 0.20]},  # B: classification (covered by a dog at >=thr)
            {"label": "cat", "bbox": [0.80, 0.80, 0.10, 0.10]},  # C: missed (no pred near it)
        ],
        "pred": [
            {"label": "cat", "bbox": [0.235, 0.235, 0.10, 0.10], "conf": 0.9},  # ~IoU .39 w/ A -> localization band
            {"label": "dog", "bbox": [0.50, 0.50, 0.20, 0.20], "conf": 0.9},    # IoU 1.0 w/ B, wrong class -> classification
            {"label": "dog", "bbox": [0.05, 0.95, 0.05, 0.05], "conf": 0.9},    # nowhere near a GT -> background FP
        ],
    }]
    r = evaluate(images, iou_thr=0.5, loc_band=0.1)
    assert _types(r["fn_detail"]["img"]) == ["classification", "localization", "missed"]
    assert _types(r["fp_detail"]["img"]) == ["background"]   # the loose-box & wrong-class FPs are NOT re-counted here
    assert r["per_image"]["img"]["n_fn"] == 3                # counts stay totals (back-compat meaning)


def test_confusion_hashes_and_per_class_prf1():
    """The Model-Eval panel needs (a) which images back each confusion cell, and (b) per-class TP/FP/FN
    for precision/recall/F1. One image: a cat mis-detected as dog (confusion) + a correctly-found cat (TP)."""
    images = [{
        "vix_hash": "im1",
        "gt": [{"label": "cat", "bbox": [0.5, 0.5, 0.2, 0.2]},     # B: covered by a dog -> confusion cat->dog (FN cat)
               {"label": "cat", "bbox": [0.2, 0.2, 0.1, 0.1]}],    # A: matched by a cat pred -> TP cat
        "pred": [{"label": "dog", "bbox": [0.5, 0.5, 0.2, 0.2], "conf": 0.9},   # wrong class over B -> FP dog + confusion
                 {"label": "cat", "bbox": [0.2, 0.2, 0.1, 0.1], "conf": 0.9}],  # matches A -> TP cat
    }]
    r = evaluate(images, iou_thr=0.5, loc_band=0.1)
    assert r["confusion"].get("cat->dog") == 1                     # truth cat mis-detected as dog
    assert r["confusion_hashes"].get("cat->dog") == ["im1"]        # clickable cell -> which image to jump to
    assert r["per_class"]["cat"] == {"tp": 1, "fp": 0, "fn": 1}
    assert r["per_class"]["dog"]["fp"] == 1
    prf = precision_recall_f1(r["per_class"])
    assert prf["cat"]["precision"] == 1.0 and prf["cat"]["recall"] == 0.5   # P=1/1, R=1/2
    assert abs(prf["cat"]["f1"] - 0.6667) < 1e-3


def test_confusion_hashes_dedup_and_fp_only_class():
    """Same (truth->pred) pair twice in ONE image -> the image hash is deduped (not listed twice); a class
    that only ever appears as a false positive still shows up in per_class."""
    images = [{
        "vix_hash": "im2",
        "gt": [{"label": "cat", "bbox": [0.2, 0.2, 0.1, 0.1]}, {"label": "cat", "bbox": [0.6, 0.6, 0.1, 0.1]}],
        "pred": [{"label": "dog", "bbox": [0.2, 0.2, 0.1, 0.1], "conf": 0.9},
                 {"label": "dog", "bbox": [0.6, 0.6, 0.1, 0.1], "conf": 0.9}],
    }]
    r = evaluate(images, iou_thr=0.5, loc_band=0.1)
    assert r["confusion"]["cat->dog"] == 2                       # two confusions in the image
    assert r["confusion_hashes"]["cat->dog"] == ["im2"]          # but the hash is deduped, not ["im2","im2"]
    assert "dog" in r["per_class"] and r["per_class"]["dog"] == {"tp": 0, "fp": 2, "fn": 0}  # FP-only class present


def test_precision_recall_f1_pure():
    out = precision_recall_f1({"a": {"tp": 3, "fp": 1, "fn": 1}})
    assert out["a"] == {"precision": 0.75, "recall": 0.75, "f1": 0.75, "tp": 3, "fp": 1, "fn": 1}
    assert precision_recall_f1({"b": {"tp": 0, "fp": 0, "fn": 0}})["b"]["f1"] == 0.0   # zero-division safe


def test_systematic_loose_boxes_show_localization_gap():
    # every pred is the right class but a bit loose -> matches @0.5, fails @0.75 -> loc_gap > 0
    box_gt = [0.5, 0.5, 0.40, 0.40]
    box_loose = [0.545, 0.545, 0.40, 0.40]  # IoU ~0.66: TP@0.5, FP@0.75
    images = [
        {"vix_hash": f"i{i}", "gt": [{"label": "a", "bbox": box_gt}],
         "pred": [{"label": "a", "bbox": box_loose, "conf": 0.9}]}
        for i in range(4)
    ]
    r = evaluate(images, iou_thr=0.5)
    assert r["map_by_iou"][0.5] > r["map_by_iou"][0.75]      # localization tail is now visible
    assert r["loc_gap"] is not None and r["loc_gap"] > 0


def test_errors_legacy_api_preserved():
    # errors.py shares iou() but keeps its GT-centric legacy strings + BBox signature
    assert abs(bbox_iou(BBox(0.5, 0.5, 0.2, 0.2), BBox(0.5, 0.5, 0.2, 0.2)) - 1.0) < 1e-9
    from vix.types import Detection
    preds = [Detection("cat", 0.9, BBox(0.5, 0.5, 0.2, 0.2))]
    gts = [Detection("cat", 1.0, BBox(0.6, 0.6, 0.2, 0.2))]
    assert diagnose_image(preds, gts, loc_iou=0.5) == ["localization_error"]
