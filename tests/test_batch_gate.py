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


# --- batch-admit / un-admit / ledger (governance keystone: defensible + reversible + queryable) ---

def _clean_batch(tmp_path):
    cfg = _cfg(tmp_path)
    ad = InMemoryAdapter()
    ad.seed("g0", "g.png", [_det("a", [1, 0])], tags=[Tag.GOLDEN])
    ad.seed("b0", "b.png", [_det("a", [0, 1])], tags=["batch:w23"])   # far from golden, distinct -> PASS
    ad.seed("b1", "b.png", [_det("a", [1, 1])], tags=["batch:w23"])   # not a within-batch dup of b0
    return cfg, ad


def test_admit_clean_batch_records_and_changes_pool_hash(tmp_path):
    cfg, ad = _clean_batch(tmp_path)
    r = pipeline.batch_admit(ad, cfg, "w23")
    assert r["admitted"] and r["verdict"] == "PASS" and r["n_admitted"] == 2
    assert r["pre_hash"] != r["post_hash"]                              # admitting changed the training pool
    tags = next(t for h, _s, _d, t in ad.samples() if h == "b0")
    assert Tag.ADMITTED in tags
    led = pipeline.batch_ledger(cfg)
    assert led["admitted_batches"] == ["w23"]


def test_admit_refused_on_block_unless_forced(tmp_path):
    cfg = _cfg(tmp_path)
    ad = InMemoryAdapter()
    ad.seed("g0", "g.png", [_det("a", [1, 0])], tags=[Tag.GOLDEN])
    ad.seed("b_leak", "b.png", [_det("a", [1, 0])], tags=["batch:w23"])  # near-dup of golden -> BLOCK
    r = pipeline.batch_admit(ad, cfg, "w23")
    assert not r["admitted"] and r["verdict"] == "BLOCK"
    assert Tag.ADMITTED not in next(t for h, _s, _d, t in ad.samples() if h == "b_leak")
    rf = pipeline.batch_admit(ad, cfg, "w23", force=True)               # override is allowed + logged
    assert rf["admitted"] and rf["forced"]
    hist = pipeline.batch_ledger(cfg)["history"]
    assert any(h["decision"] == "REFUSED" for h in hist) and any(h["decision"] == "FORCED" for h in hist)


def test_unadmit_reverses_and_is_logged(tmp_path):
    cfg, ad = _clean_batch(tmp_path)
    admit = pipeline.batch_admit(ad, cfg, "w23")
    r = pipeline.batch_unadmit(ad, cfg, "w23")
    assert r["unadmitted"] == 2
    assert all(Tag.ADMITTED not in t for h, _s, _d, t in ad.samples() if "batch:w23" in t)
    assert r["post_hash"] == admit["pre_hash"]                          # pool reverted to pre-admit state
    assert pipeline.batch_ledger(cfg)["admitted_batches"] == []          # no longer admitted


def test_unadmit_without_admit_raises(tmp_path):
    cfg, ad = _clean_batch(tmp_path)
    import pytest
    with pytest.raises(ValueError, match="未被 admit"):
        pipeline.batch_unadmit(ad, cfg, "w23")
