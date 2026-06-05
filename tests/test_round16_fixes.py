"""Round 16 補強的回歸測試:
  - Tag.EVAL 一級隔離(不被 route)+ gate 對 eval∩golden 重疊 NO-GO + ingest 互斥
  - route 稽核紀錄寫入 embedding_backend
  - throughput 由帳本配對 route->resolve 算週轉
  - Windows BOM JSON 不再讓 merge-preview 失敗
"""

import codecs
import json
from datetime import datetime, timezone

import numpy as np
import pytest

from vix import pipeline
from vix.adapters.memory import InMemoryAdapter
from vix.cli import main
from vix.config import Config
from vix.core.decision_log import DecisionLog
from vix.core.gate import pre_train_gate
from vix.types import BBox, Detection, Tag


def _det(label, conf, emb):
    return Detection(label, conf, BBox(0.5, 0.5, 0.2, 0.2), embedding=np.array(emb, dtype=float))


def _golden(ad, n=6):
    for i in range(n):
        ad.seed(f"g{i}", "g.png", [_det("a", 0.9, [1, 0])], tags=[Tag.GOLDEN])


def test_gate_blocks_on_eval_golden_overlap():  # AH2
    assert pre_train_gate(eval_golden_overlap=3).verdict == "NO-GO"
    assert pre_train_gate(eval_golden_overlap=0, n_golden=5).verdict == "GO"


def test_ingest_rejects_golden_and_eval(tmp_path):  # AH2
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    with pytest.raises(ValueError):
        pipeline.ingest(InMemoryAdapter(), cfg, str(tmp_path), "b", tags=[Tag.GOLDEN, Tag.EVAL])


def test_eval_sample_is_not_routed(tmp_path):  # AH2
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    ad = InMemoryAdapter()
    _golden(ad)
    ad.seed("e0", "e.png", [_det("a", 0.05, [9, 9])], tags=[Tag.EVAL])  # would be REVIEW if routed
    pipeline.route(ad, cfg, pipeline.calibrate(ad, cfg))
    e_tags = next(t for h, _s, _d, t in ad.samples() if h == "e0")
    assert Tag.REVIEW not in e_tags and Tag.PASS not in e_tags  # held-out, untouched


def test_route_audit_records_backend(tmp_path):  # AH9
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    cfg.embedding_backend = "pixel_fallback"
    ad = InMemoryAdapter()
    _golden(ad)
    ad.seed("c0", "c.png", [_det("a", 0.05, [9, 9])], tags=[])
    pipeline.route(ad, cfg, pipeline.calibrate(ad, cfg))
    routes = [r for r in DecisionLog(cfg.decision_log_path).read_all() if r["event"] == "route"]
    assert routes and all(r["extra"].get("embedding_backend") == "pixel_fallback" for r in routes)


def test_throughput_pairs_route_resolve(tmp_path):  # AH8
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    dl = DecisionLog(cfg.decision_log_path)
    t0 = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    t2 = datetime(2026, 1, 1, 2, 0, 0, tzinfo=timezone.utc)
    dl.append("route", vix_hash="x1", decision="review", ts=t0)
    dl.append("review", vix_hash="x1", decision="confirmed", ts=t2)
    dl.append("route", vix_hash="x2", decision="review", ts=t0)  # still open

    r = pipeline.throughput(cfg)
    assert r["n_resolved"] == 1 and abs(r["median_hours"] - 2.0) < 0.01 and r["n_open"] == 1


def test_merge_preview_reads_bom_json(tmp_path):  # Windows BOM bug
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_bytes(codecs.BOM_UTF8 + json.dumps({"cat": 10, "dog": 5}).encode("utf-8"))
    b.write_bytes(codecs.BOM_UTF8 + json.dumps({"cat": 8}).encode("utf-8"))
    rc = main(["--workspace", str(tmp_path / "ws"), "--adapter", "memory",
               "merge-preview", "--counts-a", str(a), "--counts-b", str(b)])
    assert rc == 0  # BOM no longer breaks the JSON read
