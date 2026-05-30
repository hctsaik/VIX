import numpy as np

from vix.core.threshold import ThresholdPolicy
from vix.types import Flag, Routing


def _policy():
    confs = {"a": np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.4, 0.85, 0.88])}
    dists = {"a": np.array([0.0, 0.05, 0.1, 0.02, 0.03, 0.04, 0.06, 0.07, 0.08, 0.2])}
    return ThresholdPolicy.calibrate(confs, dists, conf_pct=5, dist_pct=95)


def test_low_conf_routes_review():
    pol = _policy()
    ct = pol.thresholds["a"]
    r = pol.route("a", conf=ct.conf_thr - 0.01, knn_dist=0.0)
    assert r.decision == Routing.REVIEW
    assert Flag.LOW_CONF in r.reasons


def test_far_from_known_routes_review():
    pol = _policy()
    ct = pol.thresholds["a"]
    r = pol.route("a", conf=0.99, knn_dist=ct.dist_thr + 0.5)
    assert Flag.FAR_FROM_KNOWN in r.reasons


def test_in_distribution_passes():
    pol = _policy()
    r = pol.route("a", conf=0.99, knn_dist=0.0)
    assert r.decision == Routing.PASS
    assert not r.reasons


def test_unknown_class_routes_review():
    pol = _policy()
    r = pol.route("zzz", 0.99, 0.0)
    assert r.decision == Routing.REVIEW
    assert Flag.FAR_FROM_KNOWN in r.reasons


def test_low_support_flag():
    pol = _policy()
    r = pol.route("a", conf=0.99, knn_dist=0.0, low_support=True)
    assert Flag.LOW_SUPPORT in r.reasons
    assert r.decision == Routing.REVIEW


def test_save_load_roundtrip(tmp_path):
    pol = _policy()
    p = tmp_path / "thr.json"
    pol.save(p)
    pol2 = ThresholdPolicy.load(p)
    assert set(pol2.thresholds) == set(pol.thresholds)
    assert abs(pol2.thresholds["a"].conf_thr - pol.thresholds["a"].conf_thr) < 1e-9
    assert abs(pol2.thresholds["a"].dist_thr - pol.thresholds["a"].dist_thr) < 1e-9
