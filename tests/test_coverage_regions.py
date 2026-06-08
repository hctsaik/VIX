"""Core unit tests for the coverage manager primitives (pure, FiftyOne-free).

coverage_regions  — per-class within-class density groups: scarce/enough/over + the two honesty guards
                    (support floor withholds; single-linkage chaining withholds).
coverage_gapfill  — per new sample: fills_gap / redundant / duplicate, with the nearest-neighbour to show.
"""

import numpy as np

from vix.core.analytics import (
    _MIN_SUPPORT,
    EmbItem,
    coverage_delta,
    coverage_gapfill,
    coverage_regions,
)


def _e(id, label, vec, conf=0.9):
    return EmbItem(id, label, np.array(vec, dtype=float), confidence=conf)


def _onehot(dim, i, scale=1.0):
    v = np.zeros(dim)
    v[i] = scale
    return v


def _cluster(label, base_idx, n, dim, start, conf=0.9):
    """n items near the one-hot direction base_idx, with tiny noise (same region, distinct points)."""
    out = []
    for j in range(n):
        v = _onehot(dim, base_idx)
        v[(base_idx + 1) % dim] += 0.001 * (j + 1)  # tiny perturbation -> cosine dist ~0 within cluster
        out.append(_e(f"{label}{start + j}", label, v, conf))
    return out


def test_regions_split_scarce_enough_over():
    """A class with separated sub-clusters (no single region dominating) yields OVER (>=3),
    ENOUGH (2) and SCARCE (singleton) verdicts."""
    dim = 8
    items = []
    items += _cluster("car", 0, 13, dim, 0)   # OVER
    items += _cluster("car", 1, 10, dim, 0)   # OVER
    items += _cluster("car", 2, 3, dim, 0)    # OVER
    items += _cluster("car", 3, 2, dim, 0)    # ENOUGH (pair)
    items += [_e("car_s1", "car", _onehot(dim, 4))]  # SCARCE singleton
    items += [_e("car_s2", "car", _onehot(dim, 5))]  # SCARCE singleton
    # n = 30, largest region 13/30 = 0.43 < chain_frac -> not chained; >= support floor -> not withheld
    out = coverage_regions(items, region_distance=0.25)
    car = out["car"]
    assert car["count"] == 30
    assert car["chained"] is False
    assert car["verdict_withheld"] is False
    statuses = sorted(r["status"] for r in car["regions"])
    assert statuses.count("OVER") == 3
    assert statuses.count("ENOUGH") == 1
    assert statuses.count("SCARCE") == 2
    assert len(car["over_regions"]) == 3
    assert len(car["scarce_regions"]) == 2
    # every region names a representative that is one of its own members
    for r in car["regions"]:
        assert r["representative_id"] in r["ids"]


def test_chaining_guard_withholds_verdict():
    """Single-linkage at a loose distance chains a near-collinear class into one blob; the guard must
    withhold per-region verdicts (chained=True, statuses empty) instead of lying 'this class is covered'."""
    dim = 6
    # a smooth continuum: each point a small step from the previous -> union-find chains them all
    items = [_e(f"p{i}", "ped", _onehot(dim, 0) + np.array([0, 0.02 * i, 0, 0, 0, 0])) for i in range(25)]
    out = coverage_regions(items, region_distance=0.25)
    ped = out["ped"]
    assert ped["count"] == 25
    assert ped["chained"] is True
    assert ped["max_region_frac"] > 0.6
    assert all(r["status"] == "" for r in ped["regions"])      # verdicts withheld
    assert ped["scarce_regions"] == [] and ped["over_regions"] == []


def test_support_floor_withholds_verdict():
    """Below the support floor a class verdict is unstable -> withheld (only the count is trustworthy)."""
    dim = 6
    items = _cluster("rare", 0, 5, dim, 0) + _cluster("rare", 3, 4, dim, 100)  # n = 9 < _MIN_SUPPORT
    out = coverage_regions(items, region_distance=0.25)
    rare = out["rare"]
    assert rare["count"] == 9 < _MIN_SUPPORT
    assert rare["verdict_withheld"] is True
    assert all(r["status"] == "" for r in rare["regions"])


def test_weakness_proxy_is_mean_one_minus_conf():
    """proxy = mean(1 - confidence): an eval-free uncertainty proxy, higher = model less sure."""
    items = [_e("a", "x", _onehot(4, 0), conf=0.2), _e("b", "x", _onehot(4, 0), conf=0.4)]
    out = coverage_regions(items, region_distance=0.25)
    assert abs(out["x"]["weakness_proxy"] - 0.7) < 1e-6  # mean(0.8, 0.6)


def test_gapfill_verdicts_and_nearest_neighbor():
    dim = 6
    existing = _cluster("car", 0, 4, dim, 0)
    new = [
        _e("dup", "car", _onehot(dim, 0)),                 # ~identical -> duplicate
        _e("novel", "car", _onehot(dim, 3)),               # orthogonal -> fills_gap
    ]
    rows = {r["id"]: r for r in coverage_gapfill(new, existing, radius=0.2, dup_distance=0.05)}
    assert rows["dup"]["verdict"] == "duplicate"
    assert rows["novel"]["verdict"] == "fills_gap"
    # every row carries a nearest existing id to SHOW for human veto
    assert rows["dup"]["nearest_id"] in {e.id for e in existing}
    assert rows["novel"]["nearest_distance"] > 0.2


def test_gapfill_fills_gap_matches_coverage_delta():
    """The fills_gap set must equal coverage_delta's novel_ids (same radius) — gap-fill is the
    per-sample, neighbour-annotated view of the existing coverage_delta primitive."""
    dim = 6
    existing = _cluster("c", 0, 5, dim, 0)
    new = [_e("n0", "c", _onehot(dim, 0)), _e("n1", "c", _onehot(dim, 2)), _e("n2", "c", _onehot(dim, 4))]
    rows = coverage_gapfill(new, existing, radius=0.2)
    fills = {r["id"] for r in rows if r["verdict"] == "fills_gap"}
    novel = set(coverage_delta(new, existing, radius=0.2)["novel_ids"])
    assert fills == novel


def test_gapfill_no_existing_all_novel():
    new = [_e("a", "x", _onehot(4, 0)), _e("b", "x", _onehot(4, 1))]
    rows = coverage_gapfill(new, [], radius=0.2)
    assert all(r["verdict"] == "fills_gap" and r["nearest_id"] is None for r in rows)
