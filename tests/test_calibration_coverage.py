"""Unit tests for the calibration↔dataset coverage honesty guard (core.calibration_coverage)."""

from vix.core.calibration_coverage import assess_coverage
from vix.core.threshold import ClassThreshold


def _thr(classes, n=5):
    return {c: ClassThreshold(conf_thr=0.5, dist_thr=0.2, n_support=n) for c in classes}


def test_ok_when_calibration_matches():
    v = assess_coverage(_thr(["a", "b"]), {"embedding_backend": "dino"}, {"a", "b"}, "dino")
    assert v.ok and not v.uncovered_classes and v.reason == "" and not v.backend_mismatch


def test_missing_class_is_uncovered():
    # the patHole shape: thresholds belong to another dataset (horiz/vert), data is pothole
    v = assess_coverage(_thr(["horiz", "vert"]), {"embedding_backend": "pixel_fallback"},
                        {"pothole"}, "pixel_fallback")
    assert not v.ok and v.uncovered_classes == ["pothole"] and "未涵蓋" in v.reason


def test_backend_mismatch():
    v = assess_coverage(_thr(["a"]), {"embedding_backend": "pixel_fallback"}, {"a"}, "dinov2-vitb14-torch")
    assert not v.ok and v.backend_mismatch and "後端" in v.reason


def test_fp_mismatch():
    v = assess_coverage(_thr(["a"]), {"embedding_backend": "dino", "encoder_fp": "AAA"}, {"a"}, "dino", "BBB")
    assert not v.ok and v.fp_mismatch


def test_fp_absent_is_lenient():
    # calibration predates fingerprinting -> fail OPEN (don't cry wolf), matches the gate's policy
    v = assess_coverage(_thr(["a"]), {"embedding_backend": "dino"}, {"a"}, "dino", "BBB")
    assert v.ok and not v.fp_mismatch


def test_no_thresholds_at_all():
    v = assess_coverage(None, None, {"a"}, "dino")
    assert not v.ok and v.no_thresholds and "calibrate" in v.reason


def test_empty_detection_classes_is_ok():
    v = assess_coverage(_thr(["a"]), {}, set(), "dino")  # nothing to cover
    assert v.ok


def test_n_support_zero_counts_as_uncovered():
    v = assess_coverage({"a": ClassThreshold(0.5, 0.2, 0)}, {}, {"a"}, "dino")  # n_support 0 == uncalibrated
    assert not v.ok and v.uncovered_classes == ["a"]
