import numpy as np

from vix import pipeline
from vix.adapters.memory import InMemoryAdapter
from vix.config import Config
from vix.core.analytics import EmbItem
from vix.core.explain import explain_image
from vix.core.geometry import geometry_drift
from vix.core.quality import reviewer_consistency
from vix.types import BBox, Detection, Tag


def _det(label, conf, emb):
    return Detection(label, conf, BBox(0.5, 0.5, 0.2, 0.2), embedding=np.array(emb, dtype=float))


def _it(id, label, emb):
    return EmbItem(id, label, np.array(emb, dtype=float))


def test_reviewer_class_filter():
    items = [_it("x1", "needle", [1, 0]), _it("x2", "needle", [1, 0]), _it("y", "tube", [0, 1])]
    decisions = [
        {"reviewer_id": "A", "id": "x1", "decision": "pass"},
        {"reviewer_id": "A", "id": "x2", "decision": "reject"},
        {"reviewer_id": "A", "id": "y", "decision": "pass"},
    ]
    res = reviewer_consistency(decisions, items, sim_threshold=0.9, label_filter="needle")
    assert res["A"]["intra_consistency"] == 0.0 and len(res["A"]["conflicts"]) == 1


def test_geometry_p95_tail_shift():
    a = [Detection("a", 0.9, BBox(0.5, 0.5, 0.1, 0.1)) for _ in range(10)]
    b = [Detection("a", 0.9, BBox(0.5, 0.5, 0.1, 0.1)) for _ in range(9)]
    b.append(Detection("a", 0.9, BBox(0.5, 0.5, 0.9, 0.9)))  # tail outlier
    d = geometry_drift(a, b, shift_threshold=0.2)
    assert "w_p95" in d["shifts"]
    assert d["alert"] is True  # mean barely moves but the p95 tail does


def test_merge_preview_from_tags(tmp_path):
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    ad = InMemoryAdapter()
    ad.seed("a1", "a1.png", [_det("car", 0.9, [1, 0])], tags=["teamA"])
    ad.seed("b1", "b1.png", [_det("vehicle_car", 0.9, [1, 0])], tags=["teamB"])
    merged = pipeline.merge_preview_tags(ad, cfg, "teamA", "teamB", {"vehicle_car": "car"})
    assert merged == {"car": 2}


def test_new_classes_size_differentiated_suggestion(tmp_path):
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    ad = InMemoryAdapter()
    for i in range(5):
        ad.seed(f"a{i}", f"a{i}.png", [_det("a", 0.9, [1, 0, 0])], tags=[Tag.GOLDEN])
        ad.seed(f"b{i}", f"b{i}.png", [_det("b", 0.9, [0, 1, 0])], tags=[Tag.GOLDEN])
    for i in range(4):
        ad.seed(f"n{i}", f"n{i}.png", [_det("a", 0.5, [0, 0, 1])])  # 4 novel -> large cluster
    clusters = pipeline.new_classes(ad, cfg)
    big = [c for c in clusters if c["size"] >= 3]
    assert big and "新類別" in big[0]["suggestion"]


def test_active_learn_has_plain_why(tmp_path):
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    ad = InMemoryAdapter()
    for i in range(5):
        ad.seed(f"g{i}", f"g{i}.png", [_det("a", 0.95, [1, 0])], tags=[Tag.GOLDEN])
    ad.seed("u", "u.png", [_det("a", 0.1, [0, 1])])
    r = pipeline.active_learn(ad, cfg, budget=1)
    assert "why" in r[0] and "效益" in r[0]["why"]


def test_explain_image_uncalibrated_hint():
    r = explain_image("a", 0.5, 0.5)  # no thresholds passed
    assert r["calibrated"] is False
    assert "calibrate" in r["summary"]


def test_snapshot_freezes_anchor_fingerprint(tmp_path):
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    ad = InMemoryAdapter()
    for i in range(4):
        ad.seed(f"a{i}", f"a{i}.png", [_det("a", 0.9, [1, 0])], tags=[Tag.GOLDEN, Tag.ANCHOR])
        ad.seed(f"b{i}", f"b{i}.png", [_det("b", 0.9, [0, 1])], tags=[Tag.GOLDEN, Tag.ANCHOR])
    pipeline.build_reference(ad, cfg)
    snap, _out = pipeline.snapshot(ad, cfg, "v1")
    assert "anchor_ref_sha256" in snap["thresholds_meta"]
