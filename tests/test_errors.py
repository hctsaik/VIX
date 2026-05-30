from vix.core.errors import bbox_iou, diagnose_errors, diagnose_image
from vix.types import BBox, Detection


def test_iou_identical_and_disjoint():
    b = BBox(0.5, 0.5, 0.2, 0.2)
    assert abs(bbox_iou(b, b) - 1.0) < 1e-9
    assert bbox_iou(BBox(0.1, 0.1, 0.1, 0.1), BBox(0.9, 0.9, 0.1, 0.1)) == 0.0


def test_classification_error_well_localized_wrong_class():
    preds = [Detection("dog", 0.9, BBox(0.5, 0.5, 0.4, 0.4))]
    gts = [Detection("cat", 1.0, BBox(0.5, 0.5, 0.4, 0.4))]
    assert diagnose_image(preds, gts) == ["classification_error"]


def test_localization_error_right_class_poor_box():
    preds = [Detection("cat", 0.9, BBox(0.5, 0.5, 0.2, 0.2))]
    gts = [Detection("cat", 1.0, BBox(0.6, 0.6, 0.2, 0.2))]
    assert diagnose_image(preds, gts, loc_iou=0.5) == ["localization_error"]


def test_diagnose_errors_aggregate():
    samples = [
        ("img1", [Detection("dog", 0.9, BBox(0.5, 0.5, 0.4, 0.4))], [Detection("cat", 1, BBox(0.5, 0.5, 0.4, 0.4))]),
        ("img2", [Detection("cat", 0.9, BBox(0.5, 0.5, 0.2, 0.2))], [Detection("cat", 1, BBox(0.6, 0.6, 0.2, 0.2))]),
    ]
    res = diagnose_errors(samples, loc_iou=0.5)
    assert "img1" in res["classification_errors"]
    assert "img2" in res["localization_errors"]
