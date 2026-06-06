"""Multi-agent R2 fixes: the box-level audit hole (BOX) + the ap-trend honesty bug (TRENDFIX).

BOX: vix_hash hashes IMAGE BYTES only and content_hash bound image-id + thresholds, so a native-editor
box edit (tighten/add/delete/relabel) flowed into `export` while the snapshot identity + DecisionLog
stayed byte-identical — violating "DecisionLog is audit truth" + "snapshot<->mAP registry". Fix: fold a
canonical golden-box digest into content_hash (snapshot + training-pool) and stamp boxes_hash on export.

TRENDFIX: ap-trend printed per-class ↑進步/↓退步 deltas even when the eval set changed, while the gate
withholds its verdict in exactly that case. Fix: withhold the directional arrow when eval_set_changed.
"""

import numpy as np

from vix import pipeline
from vix.adapters.memory import InMemoryAdapter
from vix.cli import main
from vix.config import Config
from vix.core.decision_log import DecisionLog
from vix.core.snapshot import _content_hash
from vix.types import BBox, Detection, Tag


def _det(label, box, emb=(1.0, 0.0, 0.0)):
    return Detection(label, 0.9, BBox(*box), embedding=np.asarray(emb, float))


# ---- BOX: content_hash binds box geometry ----------------------------------------------------

def test_content_hash_binds_box_geometry():
    base = ["g1"]
    h_a = _content_hash(base, {}, {"g1": [["pothole", 0.5, 0.5, 0.2, 0.2]]})
    h_same = _content_hash(base, {}, {"g1": [["pothole", 0.5, 0.5, 0.2, 0.2]]})
    h_moved = _content_hash(base, {}, {"g1": [["pothole", 0.5, 0.5, 0.30, 0.2]]})  # box widened
    h_relabel = _content_hash(base, {}, {"g1": [["crack", 0.5, 0.5, 0.2, 0.2]]})    # label changed
    h_none = _content_hash(base, {})                                                # legacy (boxes unbound)
    assert h_a == h_same          # deterministic
    assert h_a != h_moved         # geometry change -> identity change (hole closed)
    assert h_a != h_relabel       # label change -> identity change
    assert h_a != h_none          # binding boxes differs from the legacy image-id-only hash


def test_training_pool_hash_changes_on_box_edit(tmp_path):
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    ad = InMemoryAdapter()
    ad.seed("g1", "g1.png", [_det("pothole", (0.5, 0.5, 0.2, 0.2))], tags=[Tag.GOLDEN])
    h_before = pipeline._training_pool_hash(ad, cfg)
    ad.set_detections("g1", [_det("pothole", (0.5, 0.5, 0.35, 0.25))])  # tighten/resize the box
    h_after = pipeline._training_pool_hash(ad, cfg)
    assert h_before != h_after  # a native box edit now changes the training-pool identity


def test_export_stamps_boxes_hash_that_changes_on_box_edit(tmp_path):
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    ad = InMemoryAdapter()
    ad.seed("g1", "g1.png", [_det("pothole", (0.5, 0.5, 0.2, 0.2))], tags=[Tag.GOLDEN])
    res1 = pipeline.export(ad, cfg, ["pothole"], tmp_path / "out1")
    assert "boxes_hash" in res1
    # the export DecisionLog event carries the fingerprint (audit log records WHAT boxes trained)
    ev = [e for e in DecisionLog(cfg.decision_log_path).read_all() if e["event"] == "export"][-1]
    assert ev["extra"]["boxes_hash"] == res1["boxes_hash"]
    ad.set_detections("g1", [_det("pothole", (0.5, 0.5, 0.4, 0.3))])  # edit a golden box
    res2 = pipeline.export(ad, cfg, ["pothole"], tmp_path / "out2")
    assert res2["boxes_hash"] != res1["boxes_hash"]  # the edit is no longer invisible to the audit log


# ---- TRENDFIX: ap-trend withholds the arrow when the eval set changed -------------------------

def _seed_evals(cfg, hashes_and_ap):
    log = DecisionLog(cfg.decision_log_path)
    for esh, ap in hashes_and_ap:
        log.append("eval_ingest", decision="eval", extra={"eval_set_hash": esh, "mAP": ap,
                                                          "per_class_ap": {"pothole": ap}})


def test_ap_trend_suppresses_arrow_when_eval_set_changed(tmp_path, capsys):
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    _seed_evals(cfg, [("hashA", 0.5), ("hashB", 0.6)])  # eval set changed between the two ingests
    rc = main(["--workspace", str(cfg.workspace), "--adapter", "memory", "ap-trend"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "0.5 → 0.6" in out               # the raw series is still shown
    assert "不可比較" in out                 # but the directional verdict is withheld
    assert "↑進步" not in out                # no false "improved" arrow across a changed eval set


def test_ap_trend_shows_arrow_when_comparable(tmp_path, capsys):
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    _seed_evals(cfg, [("hashA", 0.5), ("hashA", 0.6)])  # SAME eval set -> comparable
    main(["--workspace", str(cfg.workspace), "--adapter", "memory", "ap-trend"])
    out = capsys.readouterr().out
    assert "↑進步" in out and "不可比較" not in out
