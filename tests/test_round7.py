import numpy as np

from vix import pipeline
from vix.adapters.memory import InMemoryAdapter
from vix.config import Config
from vix.types import BBox, Detection, Tag


def _det(label, conf, emb):
    return Detection(label, conf, BBox(0.5, 0.5, 0.2, 0.2), embedding=np.array(emb, dtype=float))


def test_run_summary_has_leakage_and_verify(tmp_path):
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    ad = InMemoryAdapter()
    for i in range(6):
        ad.seed(f"a{i}", f"a{i}.png", [_det("a", 0.9, [1, 0])], tags=[Tag.GOLDEN])
        ad.seed(f"b{i}", f"b{i}.png", [_det("b", 0.9, [0, 1])], tags=[Tag.GOLDEN])
    ad.seed("c", "c.png", [_det("a", 0.9, [1, 0])])
    s = pipeline.run_pipeline(ad, cfg)
    assert "n_leakage" in s
    assert s["audit_verified"] is True


def test_active_learn_returns_reasons(tmp_path):
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    ad = InMemoryAdapter()
    for i in range(5):
        ad.seed(f"g{i}", f"g{i}.png", [_det("a", 0.95, [1, 0])], tags=[Tag.GOLDEN])
    ad.seed("u", "u.png", [_det("a", 0.1, [0, 1])])
    r = pipeline.active_learn(ad, cfg, budget=1)
    assert r[0]["id"] == "u"
    assert "uncertainty" in r[0] and "novelty" in r[0]


def test_new_classes_has_disposition_suggestion(tmp_path):
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    ad = InMemoryAdapter()
    for i in range(5):
        ad.seed(f"a{i}", f"a{i}.png", [_det("a", 0.9, [1, 0, 0])], tags=[Tag.GOLDEN])
        ad.seed(f"b{i}", f"b{i}.png", [_det("b", 0.9, [0, 1, 0])], tags=[Tag.GOLDEN])
    ad.seed("n", "n.png", [_det("a", 0.5, [0, 0, 1])])
    clusters = pipeline.new_classes(ad, cfg)
    assert clusters and "suggestion" in clusters[0]
