"""Encoder fingerprint (audit-truth fix) + near-dup annotation-consistency (causal-certain label errors).

R3 consensus: the encoder behind every PROXY number was the one input NOT in the audit truth (computed
then discarded) — a swapped/drifted encoder stamped the identical identity while thresholds drifted. Now
it's stamped into calibration (-> snapshot content_hash) and the gate NO-GOs on an encoder change.
"""

import numpy as np

from vix import pipeline
from vix.adapters.memory import InMemoryAdapter
from vix.config import Config
from vix.core.analytics import EmbItem, near_dup_label_conflicts
from vix.core.encoder_fingerprint import encoder_fingerprint, probe_digest
from vix.core.threshold import ThresholdPolicy
from vix.types import BBox, Detection, Tag


# ---- pure fingerprint ----

def test_encoder_fingerprint_is_behaviour_anchored_and_omits_none():
    a = encoder_fingerprint({"backend": "dinov2-vitb14-torch", "probe_digest": "abc", "torch_version": "2.1"})
    b = encoder_fingerprint({"backend": "dinov2-vitb14-torch", "probe_digest": "abc", "torch_version": "2.1"})
    c = encoder_fingerprint({"backend": "dinov2-vitb14-torch", "probe_digest": "XYZ", "torch_version": "2.1"})
    assert a["fp"] == b["fp"] and a["fp"] != c["fp"]                 # behaviour change -> identity change
    assert encoder_fingerprint({"backend": "x", "device": None})["components"] == {"backend": "x"}  # None omitted


def test_probe_digest_reflects_behaviour():
    f1 = probe_digest(lambda im: np.array([1.0, 0.0, 0.0]))
    f2 = probe_digest(lambda im: np.array([1.0, 0.0, 0.0]))
    f3 = probe_digest(lambda im: np.array([0.0, 1.0, 0.0]))
    assert f1 == f2 and f1 != f3

    def _boom(im):
        raise RuntimeError("no model")
    assert probe_digest(_boom) is None


# ---- calibrate stamps it + gate detects an encoder change ----

def _seed_golden(ad, n=6):
    for i in range(n):
        ad.seed(f"g{i}", "g.png", [Detection("a", 0.9, BBox(0.5, 0.5, 0.2, 0.2),
                                              embedding=np.array([1.0, 0.0, 0.0]) + 0.01 * i)], tags=[Tag.GOLDEN])


def test_calibrate_binds_encoder_fp_and_gate_blocks_on_change(tmp_path, monkeypatch):
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    ad = InMemoryAdapter()
    _seed_golden(ad)
    pipeline.calibrate(ad, cfg)
    fp = ThresholdPolicy.load(cfg.thresholds_path).meta.get("encoder_fp")
    assert fp                                                        # encoder identity is now in the audit truth
    assert "encoder_fp_mismatch" not in pipeline.pre_train_gate_stage(ad, cfg).checks  # same encoder -> no flag
    # simulate a swapped/drifted encoder after calibrate -> NO-GO
    monkeypatch.setattr(ad, "encoder_fingerprint", lambda: {"fp": "DIFFERENT", "components": {}})
    res = pipeline.pre_train_gate_stage(ad, cfg)
    assert res.checks.get("encoder_fp_mismatch") and res.verdict == "NO-GO"


def test_encoder_fp_flows_into_snapshot_content_hash(tmp_path, monkeypatch):
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    ad = InMemoryAdapter()
    _seed_golden(ad)
    monkeypatch.setattr(ad, "encoder_fingerprint", lambda: {"fp": "ENC_A", "components": {}})
    pipeline.calibrate(ad, cfg)
    h_a = pipeline._training_pool_hash(ad, cfg)
    monkeypatch.setattr(ad, "encoder_fingerprint", lambda: {"fp": "ENC_B", "components": {}})
    pipeline.calibrate(ad, cfg)  # re-calibrate under a different encoder
    h_b = pipeline._training_pool_hash(ad, cfg)
    assert h_a != h_b  # the encoder is bound into the training-pool / snapshot content hash


# ---- near-dup annotation-consistency ----

def test_near_dup_label_conflicts_flags_only_real_conflicts():
    v = np.array([1.0, 0.0, 0.0])
    items = [
        EmbItem("a1", "crack", v), EmbItem("a2", "scratch", v + 1e-6),          # near-identical, conflicting -> flag
        EmbItem("b1", "crack", np.array([0.0, 1.0, 0.0])),
        EmbItem("b2", "crack", np.array([0.0, 1.0, 0.0]) + 1e-6),               # near-identical, SAME label -> ok
    ]
    conf = near_dup_label_conflicts(items, max_distance=0.03)
    flagged = {i for c in conf for i in c["ids"]}
    assert flagged == {"a1", "a2"}                                              # only the contradictory pair
    assert conf[0]["labels"] == {"crack": 1, "scratch": 1}
