import numpy as np

from vix import pipeline
from vix.adapters.memory import InMemoryAdapter
from vix.config import Config
from vix.core.analytics import EmbItem, coverage_gaps
from vix.types import BBox, Detection


def _det(label, conf, emb):
    return Detection(label, conf, BBox(0.5, 0.5, 0.2, 0.2), embedding=np.array(emb, dtype=float))


def _it(id, label, emb):
    return EmbItem(id, label, np.array(emb, dtype=float))


def test_merge_datasets_one_command_conflicts_and_preview(tmp_path):
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    ad = InMemoryAdapter()
    ad.seed("a1", "a1.png", [_det("car", 0.9, [1, 0])], tags=["teamA"])
    ad.seed("a2", "a2.png", [_det("truck", 0.9, [1, 0])], tags=["teamA"])
    ad.seed("b1", "b1.png", [_det("vehicle_car", 0.9, [1, 0])], tags=["teamB"])
    res = pipeline.merge_datasets(ad, cfg, "teamA", "teamB", {"vehicle_car": "car"})
    assert res["preview_distribution"] == {"car": 2, "truck": 1}
    assert "truck" in res["only_in_a"]  # conflict surfaced in the same call


def test_coverage_absolute_target():
    items = [_it(f"a{i}", "a", [1, 0]) for i in range(20)] + [_it(f"b{i}", "b", [0, 1]) for i in range(3)]
    gaps = coverage_gaps(items, k=2, target=30)
    assert gaps["b"]["under_represented"] is True
    assert gaps["b"]["need"] == 27  # 30 target - 3 current
