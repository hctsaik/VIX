"""T1c box-qa: per-box geometry QA (model-loop-v2 T1c). Pure-core checks + the read-only
pipeline stage (golden only, no tag/ledger writes)."""

import numpy as np

from vix import pipeline
from vix.adapters.memory import InMemoryAdapter
from vix.config import Config
from vix.core.box_qa import audit_boxes
from vix.core.decision_log import DecisionLog
from vix.types import BBox, Detection, Tag


def _rec(rid, label, bbox):
    return {"id": rid, "label": label, "bbox": bbox}


def test_flags_degenerate_and_truncated():
    recs = [
        _rec("d", "a", (0.5, 0.5, 0.0008, 0.2)),   # w ~ 0 -> degenerate
        _rec("t", "a", (0.02, 0.5, 0.1, 0.2)),     # left edge at -0.03 -> truncated
        _rec("ok", "a", (0.5, 0.5, 0.1, 0.2)),     # fine
    ]
    issues = {i["id"]: i["issue"] for i in audit_boxes(recs, min_support=99)}  # no envelopes
    assert issues["d"] == "degenerate" and issues["t"] == "truncated" and "ok" not in issues


def test_area_outlier_only_with_enough_support():
    recs = [_rec(f"n{i}", "a", (0.5, 0.5, 0.10, 0.10)) for i in range(12)]  # tight cluster
    recs.append(_rec("big", "a", (0.5, 0.5, 0.90, 0.90)))                   # huge -> area outlier
    issues = {i["id"]: i["issue"] for i in audit_boxes(recs, min_support=8)}
    assert issues.get("big") == "area_outlier"
    # same single outlier with insufficient support -> no envelope -> not flagged
    assert not audit_boxes(recs[:3] + [recs[-1]], min_support=8)


def test_pipeline_box_qa_is_golden_only_and_readonly(tmp_path):
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    ad = InMemoryAdapter()
    ad.seed("g", "g.png", [Detection("a", 0.9, BBox(0.5, 0.5, 0.0005, 0.2),
                                      embedding=np.array([1.0, 0.0]))], tags=[Tag.GOLDEN])
    ad.seed("r", "r.png", [Detection("a", 0.9, BBox(0.5, 0.5, 0.0005, 0.2),
                                      embedding=np.array([1.0, 0.0]))], tags=[Tag.REVIEW])  # ignored
    before = len(DecisionLog(cfg.decision_log_path).read_all())
    issues = pipeline.box_qa(ad, cfg)
    assert [i["id"] for i in issues] == ["g"] and issues[0]["issue"] == "degenerate"  # only golden
    assert len(DecisionLog(cfg.decision_log_path).read_all()) == before  # read-only: no ledger write
