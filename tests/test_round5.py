import numpy as np

from vix import pipeline
from vix.adapters.memory import InMemoryAdapter
from vix.config import Config
from vix.core.geometry import geometry_drift
from vix.core.labelmap import preview_merged_distribution
from vix.types import BBox, Detection, Tag


def _det(label, conf, bbox, emb=None):
    return Detection(label, conf, bbox, embedding=np.array(emb, dtype=float) if emb is not None else None)


def test_geometry_drift_detects_size_shift():
    small = [Detection("a", 0.9, BBox(0.5, 0.5, 0.05, 0.05)) for _ in range(10)]
    large = [Detection("a", 0.9, BBox(0.5, 0.5, 0.5, 0.5)) for _ in range(10)]
    assert geometry_drift(small, large, shift_threshold=0.2)["alert"] is True
    assert geometry_drift(small, small, shift_threshold=0.2)["alert"] is False


def test_merge_preview_distribution():
    merged = preview_merged_distribution({"car": 10, "truck": 3}, {"vehicle_car": 5}, {"vehicle_car": "car"})
    assert merged == {"car": 15, "truck": 3}


def test_report_auto_prev_and_backend(tmp_path):
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    cfg.embedding_backend = "pixel_fallback"
    ad = InMemoryAdapter()
    for i in range(4):
        ad.seed(f"a{i}", f"a{i}.png", [_det("a", 0.9, BBox(0.5, 0.5, 0.2, 0.2), [1, 0])], tags=[Tag.GOLDEN])
    rep1, _ = pipeline.health_report(ad, cfg, tmp_path / "r1")
    assert rep1["embedding_backend"] == "pixel_fallback"
    assert "quality_score" in rep1 and "gate_verdict" in rep1

    ad.seed("a9", "a9.png", [_det("b", 0.9, BBox(0.5, 0.5, 0.2, 0.2), [0, 1])], tags=[Tag.GOLDEN])
    rep2, _ = pipeline.health_report(ad, cfg, tmp_path / "r2")
    assert "diff" in rep2  # auto-picked the prior report as baseline


def test_gate_auto_detects_drift(tmp_path):
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    ad = InMemoryAdapter()
    for i in range(6):
        ad.seed(f"a{i}", f"a{i}.png", [_det("a", 0.9, BBox(0.5, 0.5, 0.2, 0.2), [1, 0])], tags=[Tag.GOLDEN, Tag.ANCHOR])
        ad.seed(f"b{i}", f"b{i}.png", [_det("b", 0.9, BBox(0.5, 0.5, 0.2, 0.2), [0, 1])], tags=[Tag.GOLDEN, Tag.ANCHOR])
    pipeline.build_reference(ad, cfg)
    ad.seed("drift", "d.png", [_det("a", 0.9, BBox(0.5, 0.5, 0.2, 0.2), [0, 1])], tags=[Tag.REVIEW])
    r = pipeline.pre_train_gate_stage(ad, cfg)  # drift auto-detected (no manual flag)
    assert r.verdict == "NO-GO"
    assert any("漂移" in reason for reason in r.reasons)
