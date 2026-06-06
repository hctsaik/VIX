"""batch-gate: the weekly 'can THIS batch go in / what to clean' verdict. Pure verdict logic +
the pipeline orchestrator (NEW batch->frozen leakage block, degenerate-box block, advisory clean-list,
PARTIAL when there's no frozen eval/golden to check leakage against). Hygiene, not a mAP promise."""

import numpy as np

from vix import pipeline
from vix.adapters.memory import InMemoryAdapter
from vix.config import Config
from vix.core.decision_log import DecisionLog
from vix.core.gate import batch_gate_verdict
from vix.types import BBox, Detection, Tag


def _det(label, emb, bbox=None):
    return Detection(label, 0.9, bbox or BBox(0.5, 0.5, 0.2, 0.2), embedding=np.asarray(emb, float))


# --- pure verdict ---

def test_verdict_block_on_causal_harm():
    v, reasons = batch_gate_verdict({"eval_leakage": ["b1"], "degenerate_boxes": []}, {"open_review": 0}, eval_available=True)
    assert v == "BLOCK" and any("eval_leakage" in r for r in reasons)


def test_verdict_partial_when_no_frozen_eval():
    v, reasons = batch_gate_verdict({"eval_leakage": [], "degenerate_boxes": []}, {"open_review": 0}, eval_available=False)
    assert v == "PARTIAL" and any("洩漏檢查跳過" in r for r in reasons)  # never a silent PASS


def test_verdict_clean_vs_pass():
    assert batch_gate_verdict({"eval_leakage": []}, {"label_noise": ["x", "y"]}, eval_available=True)[0] == "CLEAN"
    assert batch_gate_verdict({"eval_leakage": []}, {"open_review": 0, "label_noise": []}, eval_available=True)[0] == "PASS"


# --- pipeline orchestrator ---

def _cfg(tmp_path):
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    return cfg


def test_batch_gate_blocks_leakage_into_frozen(tmp_path):
    cfg = _cfg(tmp_path)
    ad = InMemoryAdapter()
    ad.seed("g0", "g.png", [_det("a", [1, 0])], tags=[Tag.GOLDEN])               # frozen golden
    ad.seed("b_leak", "b.png", [_det("a", [1, 0])], tags=["batch:w23"])          # near-dup of golden -> leakage
    ad.seed("b_ok", "b.png", [_det("a", [0, 1])], tags=["batch:w23"])            # far -> fine
    r = pipeline.batch_gate(ad, cfg, "w23")
    assert r["verdict"] == "BLOCK" and r["block"]["eval_leakage"] == ["b_leak"]


def test_batch_gate_blocks_degenerate_box(tmp_path):
    cfg = _cfg(tmp_path)
    ad = InMemoryAdapter()
    ad.seed("g0", "g.png", [_det("a", [1, 0])], tags=[Tag.GOLDEN])
    ad.seed("b_bad", "b.png", [_det("a", [0, 1], BBox(0.5, 0.5, 0.0005, 0.2))], tags=["batch:w23"])  # degenerate
    r = pipeline.batch_gate(ad, cfg, "w23")
    assert r["verdict"] == "BLOCK" and r["block"]["degenerate_boxes"] == ["b_bad"]


def test_batch_gate_partial_without_frozen(tmp_path):
    cfg = _cfg(tmp_path)
    ad = InMemoryAdapter()
    ad.seed("b0", "b.png", [_det("a", [0, 1])], tags=["batch:w23"])              # no golden/eval at all
    r = pipeline.batch_gate(ad, cfg, "w23")
    assert r["verdict"] == "PARTIAL" and not r["eval_available"]


def test_batch_gate_pass_clean_batch_and_audited(tmp_path):
    cfg = _cfg(tmp_path)
    ad = InMemoryAdapter()
    ad.seed("g0", "g.png", [_det("a", [1, 0])], tags=[Tag.GOLDEN])
    ad.seed("b0", "b.png", [_det("a", [0, 1])], tags=["batch:w23"])              # far from golden, valid box
    r = pipeline.batch_gate(ad, cfg, "w23")
    assert r["verdict"] == "PASS" and not r["block"]["eval_leakage"]
    rec = [e for e in DecisionLog(cfg.decision_log_path).read_all() if e["event"] == "batch_gate"]
    assert rec and rec[0]["decision"] == "PASS" and rec[0]["batch_id"] == "w23"


def test_batch_gate_unknown_batch_raises(tmp_path):
    import pytest
    with pytest.raises(ValueError, match="找不到 batch"):
        pipeline.batch_gate(InMemoryAdapter(), _cfg(tmp_path), "nope")
