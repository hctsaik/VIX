import numpy as np

from vix import pipeline
from vix.adapters.memory import InMemoryAdapter
from vix.config import Config
from vix.types import BBox, Detection, Tag


def _det(label, conf, emb):
    return Detection(label, conf, BBox(0.5, 0.5, 0.2, 0.2), embedding=np.array(emb, dtype=float))


def test_round3_pipeline_stages(tmp_path):
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    ad = InMemoryAdapter()
    for i in range(6):
        ad.seed(f"a{i}", f"a{i}.png", [_det("a", 0.9, [1, 0, 0])], tags=[Tag.GOLDEN, "split:train", "batch:w1"])
        ad.seed(f"b{i}", f"b{i}.png", [_det("b", 0.9, [0, 1, 0])], tags=[Tag.GOLDEN, "split:train", "batch:w1"])
    ad.seed("leak_val", "a0.png", [_det("a", 0.9, [1, 0, 0])], tags=["split:val"])  # dup of a0 in val
    ad.seed("novel", "novel.png", [_det("a", 0.5, [0, 0, 1])])  # candidate novel class

    clusters = pipeline.new_classes(ad, cfg)
    novel_ids = {x for c in clusters for x in c["ids"]}
    assert any(x.startswith("novel") for x in novel_ids)  # detection id is "novel:0"

    leaks = pipeline.leakage(ad, cfg, max_distance=0.001)
    assert any(set(lk["splits"]) == {"train", "val"} for lk in leaks)

    assert isinstance(pipeline.harmful(ad, cfg, top=5), list)
    assert "trend" in pipeline.quality_trend(ad, cfg)
    assert pipeline.pre_train_gate_stage(ad, cfg).verdict in ("GO", "NO-GO")

    pipeline.calibrate(ad, cfg)
    ex = pipeline.explain_one(ad, cfg, "novel")
    assert "axes" in ex and ex["failing_axes"]
