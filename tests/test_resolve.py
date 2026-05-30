import numpy as np

from vix import pipeline
from vix.adapters.memory import InMemoryAdapter
from vix.config import Config
from vix.core.decision_log import DecisionLog
from vix.types import BBox, Detection, Tag


def _det(label, conf, emb):
    return Detection(label, conf, BBox(0.5, 0.5, 0.2, 0.2), embedding=np.array(emb, dtype=float))


def _setup(tmp_path):
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    ad = InMemoryAdapter()
    for i in range(5):
        ad.seed(f"g{i}", f"g{i}.png", [_det("a", 0.9, [1, 0])], tags=[Tag.GOLDEN])
    ad.seed("r1", "r1.png", [_det("a", 0.3, [1, 0])], tags=[Tag.REVIEW])  # awaiting review
    return cfg, ad


def _tags(ad, h):
    return next(t for hh, _s, _d, t in ad.samples() if hh == h)


def test_resolve_confirm_moves_to_golden_and_leaves_queue(tmp_path):
    cfg, ad = _setup(tmp_path)
    assert pipeline.resolve_review(ad, cfg, "r1", "confirm") == "confirmed"
    assert Tag.GOLDEN in _tags(ad, "r1")
    assert all(r["id"] != "r1" for r in pipeline.review_queue(ad, cfg))  # excluded now


def test_resolve_relabel_then_confirm(tmp_path):
    cfg, ad = _setup(tmp_path)
    assert pipeline.resolve_review(ad, cfg, "r1", "confirm", label="b") == "b"
    label = next(d.label for h, _s, dets, _t in ad.samples() if h == "r1" for d in dets)
    assert label == "b"
    assert Tag.GOLDEN in _tags(ad, "r1")


def test_resolve_false_alarm_rejects(tmp_path):
    cfg, ad = _setup(tmp_path)
    assert pipeline.resolve_review(ad, cfg, "r1", "false_alarm") == "false_alarm"
    assert Tag.REJECTED in _tags(ad, "r1")


def test_resolve_batch_audited(tmp_path):
    cfg, ad = _setup(tmp_path)
    n = pipeline.resolve_batch(ad, cfg, [{"vix_hash": "r1", "decision": "confirm", "label": "a"}], reviewer_id="u1")
    assert n == 1
    recs = DecisionLog(cfg.decision_log_path).read_all()
    assert any(r["event"] == "review" and r["reviewer_id"] == "u1" for r in recs)
