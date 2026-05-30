import numpy as np

from vix.core.analytics import (
    EmbItem,
    active_learning_ranking,
    class_distribution,
    coverage_delta,
    coverage_gaps,
    cross_period_drift,
    near_duplicate_groups,
    suspected_label_errors,
)


def _it(id, label, vec, conf=1.0):
    return EmbItem(id, label, np.array(vec, dtype=float), conf)


def test_suspected_label_errors_finds_mislabel():
    items = [_it(f"a{i}", "a", [1, 0]) for i in range(8)]
    items += [_it(f"b{i}", "b", [0, 1]) for i in range(8)]
    items.append(_it("bad", "a", [0, 1]))  # labeled 'a' but sits in 'b' cluster
    issues = suspected_label_errors(items, k=5)
    bad = [x for x in issues if x.id == "bad"]
    assert bad and bad[0].suggested_label == "b"


def test_no_false_positive_when_consistent():
    items = [_it(f"a{i}", "a", [1, 0]) for i in range(8)]
    items += [_it(f"b{i}", "b", [0, 1]) for i in range(8)]
    assert suspected_label_errors(items, k=5) == []


def test_near_duplicate_groups():
    items = [_it("x1", "a", [1, 0]), _it("x2", "a", [1, 0.0001]), _it("y", "a", [0, 1])]
    groups = near_duplicate_groups(items, max_distance=0.01)
    assert any(set(g) == {"x1", "x2"} for g in groups)
    assert all("y" not in g for g in groups)


def test_class_distribution():
    items = [_it("1", "a", [1, 0]), _it("2", "a", [1, 0]), _it("3", "b", [0, 1])]
    assert class_distribution(items) == {"a": 2, "b": 1}


def test_coverage_gaps_under_represented():
    items = [_it(f"a{i}", "a", [1, 0]) for i in range(10)] + [_it("b0", "b", [0, 1])]
    gaps = coverage_gaps(items, k=3)
    assert gaps["b"]["under_represented"] is True
    assert gaps["a"]["under_represented"] is False


def test_coverage_delta():
    existing = [_it(f"a{i}", "a", [1, 0]) for i in range(5)]
    new = [_it("inside", "a", [1, 0]), _it("novel", "a", [0, 1])]
    res = coverage_delta(new, existing, radius=0.2)
    assert "novel" in res["novel_ids"]
    assert "inside" not in res["novel_ids"]
    assert 0 < res["novel_fraction"] < 1


def test_active_learning_prefers_uncertain_and_novel():
    existing = [_it(f"a{i}", "a", [1, 0]) for i in range(5)]
    cands = [
        _it("certain_dup", "a", [1, 0], conf=0.99),
        _it("uncertain_novel", "a", [0, 1], conf=0.1),
        _it("dup2", "a", [1, 0], conf=0.98),
    ]
    ranked = active_learning_ranking(cands, existing, budget=2)
    assert ranked[0] == "uncertain_novel"


def test_cross_period_drift_alert():
    a = [_it(f"a{i}", "helmet", [1, 0]) for i in range(6)]
    b = [_it(f"b{i}", "helmet", [0, 1]) for i in range(6)]  # definition moved
    out = cross_period_drift(a, b, shift_threshold=0.15)
    assert out["helmet"]["alert"] is True
    assert out["helmet"]["shift"] > 0.5
    assert len(out["helmet"]["representatives"]) >= 1
