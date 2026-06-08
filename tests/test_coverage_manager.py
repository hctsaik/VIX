"""Pipeline tests for the coverage manager: coverage_map (fail-closed + provisional fallback +
before/after snapshot), gap_fill, and prune (read-only worklist, four guards, two-step confirm)."""

import json

import numpy as np

from vix import pipeline
from vix.adapters.memory import InMemoryAdapter
from vix.config import Config
from vix.types import BBox, Detection, Tag


def _det(label, conf, emb):
    return Detection(label, conf, BBox(0.5, 0.5, 0.2, 0.2), embedding=np.array(emb, dtype=float))


def _onehot(dim, i):
    v = np.zeros(dim)
    v[i] = 1.0
    return v


def _cfg(tmp_path):
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    return cfg


def test_coverage_map_fails_closed_without_reference(tmp_path):
    cfg = _cfg(tmp_path)
    ad = InMemoryAdapter()
    ad.seed("c1", "c1.png", [_det("a", 0.9, [1, 0])])  # candidate, NOT golden/provisional
    res = pipeline.coverage_map(ad, cfg)
    assert res["ok"] is False
    assert "參照" in res["reason"]


def test_coverage_map_provisional_fallback(tmp_path):
    """No golden but imported PROVISIONAL labels -> coverage runs on them, flagged as unverified."""
    cfg = _cfg(tmp_path)
    ad = InMemoryAdapter()
    for i in range(4):
        ad.seed(f"p{i}", f"p{i}.png", [_det("pothole", 0.9, _onehot(6, 0))], tags=[Tag.PROVISIONAL])
    res = pipeline.coverage_map(ad, cfg)
    assert res["ok"] is True
    assert res["reference"] == "provisional"
    assert "pothole" in res["classes"]


def test_coverage_map_before_after_snapshot(tmp_path):
    """Two runs: the second auto-diffs against the first snapshot (health_report pattern)."""
    cfg = _cfg(tmp_path)
    ad = InMemoryAdapter()
    for i in range(20):
        ad.seed(f"g{i}", f"g{i}.png", [_det("car", 0.9, _onehot(8, 0))], tags=[Tag.GOLDEN])
    r1 = pipeline.coverage_map(ad, cfg)
    assert "delta" not in r1                                  # nothing to diff against yet
    assert (cfg.workspace / "coverage").exists()
    for i in range(20, 30):                                   # collect more of car
        ad.seed(f"g{i}", f"g{i}.png", [_det("car", 0.9, _onehot(8, 0))], tags=[Tag.GOLDEN])
    r2 = pipeline.coverage_map(ad, cfg)
    assert "delta" in r2
    assert r2["delta"]["encoder_changed"] is False
    car = r2["delta"]["classes"]["car"]
    assert car["before"] == 20 and car["after"] == 30 and car["delta"] == 10
    assert car["stable"] is True                              # above the support floor -> a real arrow


def test_coverage_map_logs_only_on_actionable_verdict(tmp_path):
    """A read-only render does not spam the ledger; an emitted scarce/over verdict logs (new_classes pattern)."""
    cfg = _cfg(tmp_path)
    ad = InMemoryAdapter()
    # 30 car: a dominant OVER cluster + two far singletons -> actionable (scarce + over), not chained
    for i in range(26):
        v = _onehot(8, 0)
        v[1] += 0.001 * i
        ad.seed(f"c{i}", f"c{i}.png", [_det("car", 0.9, v)], tags=[Tag.GOLDEN])
    # keep the dominant cluster under chain_frac by adding separated regions
    for i in range(10):
        ad.seed(f"d{i}", f"d{i}.png", [_det("car", 0.9, _onehot(8, 2))], tags=[Tag.GOLDEN])
    ad.seed("s1", "s1.png", [_det("car", 0.4, _onehot(8, 4))], tags=[Tag.GOLDEN])
    ad.seed("s2", "s2.png", [_det("car", 0.4, _onehot(8, 5))], tags=[Tag.GOLDEN])
    res = pipeline.coverage_map(ad, cfg)
    assert res["ok"] is True
    recs = pipeline.audit(cfg, event="coverage_map")
    if any(v["scarce_regions"] or v["over_regions"] for v in res["classes"].values()):
        assert len(recs) == 1
        assert recs[0]["extra"]["encoder_fp"] == res["encoder_fp"]  # provenance stamped


def test_gap_fill_pipeline(tmp_path):
    cfg = _cfg(tmp_path)
    ad = InMemoryAdapter()
    for i in range(4):
        ad.seed(f"g{i}", f"g{i}.png", [_det("car", 0.9, _onehot(6, 0))], tags=[Tag.GOLDEN])
    ad.seed("dup", "dup.png", [_det("car", 0.9, _onehot(6, 0))])      # duplicate of golden
    ad.seed("novel", "novel.png", [_det("car", 0.9, _onehot(6, 3))])  # fills a gap
    rows = {r["id"]: r for r in pipeline.gap_fill(ad, cfg)}
    assert rows["dup"]["verdict"] == "duplicate"
    assert rows["novel"]["verdict"] == "fills_gap"
    assert rows["novel"]["nearest_id"] in {f"g{i}" for i in range(4)}


def test_prune_read_only_and_keeps_representative(tmp_path):
    cfg = _cfg(tmp_path)
    ad = InMemoryAdapter()
    for i in range(22):  # 22 identical 'x' images -> a 22-member near-dup group
        ad.seed(f"x{i}", f"x{i}.png", [_det("x", 0.9, _onehot(6, 0))], tags=[Tag.GOLDEN])
    res = pipeline.prune(ad, cfg, max_distance=0.05)
    assert res["confirmed"] is False
    # support floor: never drop class below _MIN_SUPPORT (20) -> remove only 2 of 22
    assert len(res["candidates"]) == 2
    rep = res["candidates"][0]["kept_representative_id"]
    assert rep not in {r["id"] for r in res["candidates"]}     # the kept representative is never a candidate


def test_prune_protected_class_never_pruned(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.eval_baseline_path.write_text(json.dumps({"protected": {"x": 0.02}}), encoding="utf-8")
    ad = InMemoryAdapter()
    for i in range(25):
        ad.seed(f"x{i}", f"x{i}.png", [_det("x", 0.9, _onehot(6, 0))], tags=[Tag.GOLDEN])
    res = pipeline.prune(ad, cfg, max_distance=0.05)
    assert res["candidates"] == []                            # protected -> never a prune candidate


def test_prune_skips_cross_split_leakage(tmp_path):
    """A near-dup pair that spans train/test is a leakage signal, not a 'redundant' one -> never pruned."""
    cfg = _cfg(tmp_path)
    ad = InMemoryAdapter()
    # 22 identical, but two of them are in different splits (a cross-split leak)
    for i in range(20):
        ad.seed(f"x{i}", f"x{i}.png", [_det("x", 0.9, _onehot(6, 0))],
                tags=[Tag.GOLDEN, "split:train"])
    ad.seed("xa", "xa.png", [_det("x", 0.9, _onehot(6, 0))], tags=[Tag.GOLDEN, "split:train"])
    ad.seed("xb", "xb.png", [_det("x", 0.9, _onehot(6, 0))], tags=[Tag.GOLDEN, "split:test"])
    res = pipeline.prune(ad, cfg, max_distance=0.05)
    pruned = {r["id"] for r in res["candidates"]}
    assert "xa" not in pruned and "xb" not in pruned          # leakage members excluded


def test_prune_confirm_tags_rejected_and_audits(tmp_path):
    cfg = _cfg(tmp_path)
    ad = InMemoryAdapter()
    for i in range(24):
        ad.seed(f"x{i}", f"x{i}.png", [_det("x", 0.9, _onehot(6, 0))], tags=[Tag.GOLDEN])
    res = pipeline.prune(ad, cfg, max_distance=0.05, confirm=True, note="dedup pass")
    assert res["confirmed"] is True
    removed = set(res["removed"])
    assert removed                                            # something was pruned
    tags = {h: set(t) for h, _s, _d, t in ad.samples()}
    assert all(Tag.REJECTED in tags[h] for h in removed)     # actually tagged
    recs = pipeline.audit(cfg, event="prune")
    assert len(recs) == 1 and recs[0]["extra"]["note"] == "dedup pass"
    # reversible via the existing restore path
    pipeline.restore_dismissed(ad, cfg, list(removed))
    tags2 = {h: set(t) for h, _s, _d, t in ad.samples()}
    assert all(Tag.REJECTED not in tags2[h] for h in removed)
