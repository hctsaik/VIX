"""Round 17 補強的回歸測試:
  - calibrate 記後端入 policy.meta;route 偵測後端不符;gate 對混用後端 NO-GO
  - routing-diff 納入新增/消失的 id
  - capacity 由歷史 + 預期量估算人力
"""

import json
from datetime import datetime, timezone

import numpy as np

from vix import pipeline
from vix.adapters.memory import InMemoryAdapter
from vix.config import Config
from vix.core.decision_log import DecisionLog
from vix.core.gate import pre_train_gate
from vix.types import BBox, Detection, Tag


def _det(label, conf, emb):
    return Detection(label, conf, BBox(0.5, 0.5, 0.2, 0.2), embedding=np.array(emb, dtype=float))


def test_calibrate_stamps_backend_and_route_detects_mismatch(tmp_path):  # AI6
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    cfg.embedding_backend = "dinov2"
    ad = InMemoryAdapter()
    for i in range(6):
        ad.seed(f"g{i}", "g.png", [_det("a", 0.9, [1, 0])], tags=[Tag.GOLDEN])
    pol = pipeline.calibrate(ad, cfg)
    assert pol.meta["embedding_backend"] == "dinov2"

    cfg.embedding_backend = "pixel_fallback"  # now route under a DIFFERENT backend
    ad.seed("c0", "c.png", [_det("a", 0.05, [9, 9])], tags=[])
    counts = pipeline.route(ad, cfg)  # reloads the dinov2-stamped policy from disk
    assert counts["backend_mismatch"] is True


def test_gate_blocks_on_mixed_backend():  # AI6
    assert pre_train_gate(backend_mixed=True, n_golden=5).verdict == "NO-GO"
    assert pre_train_gate(backend_mixed=False, n_golden=5).verdict == "GO"


def test_routing_diff_added_removed(tmp_path):  # AI1
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    (cfg.workspace / "routing_prev.json").write_text(json.dumps({"a": "pass", "b": "review"}), encoding="utf-8")
    (cfg.workspace / "routing_current.json").write_text(json.dumps({"a": "review", "c": "pass"}), encoding="utf-8")
    d = pipeline.routing_diff(cfg)
    assert any(x["id"] == "a" for x in d["changed"])  # a flipped pass->review
    assert "c" in d["added"] and "b" in d["removed"]


def test_capacity_estimate(tmp_path):  # AI9
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    dl = DecisionLog(cfg.decision_log_path)
    t0 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 1, 1, 1, 0, 0, tzinfo=timezone.utc)
    dl.append("route", vix_hash="x1", decision="review", ts=t0)
    dl.append("review", vix_hash="x1", decision="confirmed", ts=t1)
    dl.append("route", vix_hash="x2", decision="pass", ts=t0)

    r = pipeline.capacity(cfg, volume=100)
    assert r["flag_rate"] == 0.5            # 1 review / 2 routes
    assert r["projected_review"] == 50      # 100 x 0.5
    assert r["median_hours"] == 1.0
    assert r["total_hours"] == 50.0         # backlog 0 + 50 incoming x 1h
