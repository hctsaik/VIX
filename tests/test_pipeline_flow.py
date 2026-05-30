import numpy as np

from vix import pipeline
from vix.adapters.memory import InMemoryAdapter
from vix.config import Config
from vix.core.decision_log import DecisionLog
from vix.types import BBox, Detection, Tag


def _det(label, conf, emb):
    return Detection(label, conf, BBox(0.5, 0.5, 0.2, 0.2), embedding=np.asarray(emb, dtype=float))


def _seed_golden(ad):
    # Tight, deterministic clusters: class a == [1,0], class b == [0,1].
    for i in range(12):
        ad.seed(f"a{i}", f"imgs/a{i}.png", [_det("a", 0.9, [1.0, 0.0])], tags=[Tag.GOLDEN])
        ad.seed(f"b{i}", f"imgs/b{i}.png", [_det("b", 0.9, [0.0, 1.0])], tags=[Tag.GOLDEN])
    ad.seed("anc_a", "imgs/anc_a.png", [_det("a", 0.95, [1.0, 0.0])], tags=[Tag.ANCHOR])
    ad.seed("anc_b", "imgs/anc_b.png", [_det("b", 0.95, [0.0, 1.0])], tags=[Tag.ANCHOR])


def test_calibrate_route_export(tmp_path):
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    ad = InMemoryAdapter()
    _seed_golden(ad)
    # New candidates
    ad.seed("new_ok", "imgs/new_ok.png", [_det("a", 0.92, [1.0, 0.0])])   # in-distribution -> pass
    ad.seed("new_low", "imgs/new_low.png", [_det("a", 0.05, [1.0, 0.0])])  # low confidence
    ad.seed("new_far", "imgs/new_far.png", [_det("a", 0.92, [0.0, 1.0])])  # looks like b -> far

    policy = pipeline.calibrate(ad, cfg)
    assert cfg.thresholds_path.exists()
    assert {"a", "b"} <= set(policy.thresholds)

    counts = pipeline.route(ad, cfg, policy)
    assert counts["pass"] == 1 and counts["review"] == 2

    assert ad.fields("new_ok")["routing_decision"] == "pass"
    assert ad.fields("new_low")["routing_decision"] == "review"
    assert "low_conf" in ad.fields("new_low")["flag_reason"]
    assert ad.fields("new_far")["routing_decision"] == "review"
    assert "far_from_known" in ad.fields("new_far")["flag_reason"]

    # Audit log intact (append-only hash-chain)
    assert DecisionLog(cfg.decision_log_path).verify_chain() is True

    # One-way export of golden set
    res = pipeline.export(ad, cfg, ["a", "b"], tmp_path / "out")
    assert res["n_images"] == 24
    assert (tmp_path / "out" / "data.yaml").exists()


def test_guard_detects_definition_drift(tmp_path):
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    ad = InMemoryAdapter()
    _seed_golden(ad)
    pipeline.build_reference(ad, cfg)
    assert cfg.anchor_ref_path.exists()

    # A drifting batch: class 'a' data that actually looks like 'b'
    ad.seed("drift1", "imgs/d1.png", [_det("a", 0.9, [0.0, 1.0])], tags=[Tag.REVIEW])
    ad.seed("drift2", "imgs/d2.png", [_det("a", 0.9, [0.0, 1.0])], tags=[Tag.REVIEW])

    report = pipeline.guard(ad, cfg)
    assert report.triggered is True
    assert report.max_shift > 0.5

    # Self-gate logged as HOLD until acknowledged
    events = [r["event"] for r in DecisionLog(cfg.decision_log_path).read_all()]
    assert "guard_alert" in events
