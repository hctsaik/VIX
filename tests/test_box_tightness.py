"""Pixel-level box-tightness scoring (the SAM-backed check box_qa can't do). Pure core only — the SAM
inference is opt-in/Tier-2 (exercised by docs/examples + manual runs), kept out of CI."""

from vix.core.box_tightness import iou_cxcywh, tightness


def test_iou_cxcywh():
    assert iou_cxcywh((0.5, 0.5, 0.2, 0.2), (0.5, 0.5, 0.2, 0.2)) > 0.999  # identical (float)
    assert iou_cxcywh((0.2, 0.2, 0.1, 0.1), (0.8, 0.8, 0.1, 0.1)) == 0.0   # disjoint
    assert 0.0 < iou_cxcywh((0.5, 0.5, 0.2, 0.2), (0.56, 0.5, 0.2, 0.2)) < 1.0  # partial


def test_tightness_flags_loose_box():
    # GT box much bigger than the object mask's tight box -> loose / misaligned annotation
    loose = tightness((0.5, 0.5, 0.6, 0.6), (0.5, 0.5, 0.2, 0.2), iou_thr=0.6)
    assert loose["loose"] and loose["iou"] < 0.6 and "太鬆" in loose["why"]


def test_tightness_passes_a_tight_box():
    tight = tightness((0.5, 0.5, 0.20, 0.20), (0.5, 0.5, 0.20, 0.21), iou_thr=0.6)
    assert not tight["loose"] and tight["iou"] >= 0.6
