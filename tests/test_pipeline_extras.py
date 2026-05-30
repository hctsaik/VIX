import numpy as np

from vix import pipeline
from vix.adapters.memory import InMemoryAdapter
from vix.config import Config
from vix.types import BBox, Detection, Tag


def _det(label, conf, emb):
    return Detection(label, conf, BBox(0.5, 0.5, 0.2, 0.2), embedding=np.array(emb, dtype=float))


def test_review_queue_and_audit(tmp_path):
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    ad = InMemoryAdapter()
    for i in range(6):
        ad.seed(f"g{i}", f"g{i}.png", [_det("a", 0.9, [1, 0])], tags=[Tag.GOLDEN])
    ad.seed("safe", "safe.png", [_det("a", 0.95, [1, 0])])
    ad.seed("novel", "novel.png", [_det("a", 0.9, [0, 1])])

    q = pipeline.review_queue(ad, cfg, top=10)
    assert q[0]["id"] == "novel"      # farthest from golden -> highest risk
    assert q[-1]["id"] == "safe"
    assert q[0]["reasons"]

    pipeline.health_report(ad, cfg, tmp_path / "rep")  # emits a 'report' event
    recs = pipeline.audit(cfg, event="report")
    assert len(recs) >= 1 and all(r["event"] == "report" for r in recs)


def test_relabel_merges_classes_with_log(tmp_path):
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    ad = InMemoryAdapter()
    ad.seed("s1", "s1.png", [_det("sedan", 0.9, [1, 0])], tags=[Tag.GOLDEN])
    ad.seed("s2", "s2.png", [_det("hatchback", 0.9, [1, 0])], tags=[Tag.GOLDEN])

    diff = pipeline.relabel_dataset(ad, cfg, {"sedan": "passenger_car", "hatchback": "passenger_car"})
    assert diff["total_changed"] == 2
    labels = {d.label for _h, _s, dets, _t in ad.samples() for d in dets}
    assert labels == {"passenger_car"}
    assert (cfg.workspace / "relabel_changes.jsonl").exists()


def test_merge_maps_reconciles_names():
    res = pipeline.merge_maps({0: "car"}, {0: "vehicle_car"}, {"vehicle_car": "car"})
    assert res["unified_names"] == ["car"]
    assert res["orphans"] == []
