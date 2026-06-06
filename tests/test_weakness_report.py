"""weakness-report: the two-mode "where YOLO is weak / go label these" rollup + the class-aware
error-mine queue (model-loop-v2). Pure renderer + pipeline end-to-end in GT mode."""

import json

import numpy as np

from vix import pipeline
from vix.adapters.memory import InMemoryAdapter
from vix.config import Config
from vix.core.decision_log import DecisionLog
from vix.core.weakness_report import render_weakness_report
from vix.types import BBox, Detection, Tag


def _det(label, conf, emb, bbox=None):
    return Detection(label, conf, bbox or BBox(0.5, 0.5, 0.2, 0.2), embedding=np.array(emb, float))


def test_render_is_proxy_stamped_and_complete():
    md = render_weakness_report({
        "mode": "gt", "mAP": 0.5, "loc_gap": 0.1,
        "per_class": [{"cls": "b", "ap": 0.0, "n_gt": 1, "dom_fn_type": "missed", "top_confusion": None}],
        "confusion": [("b->a", 2)],
        "confident_wrong": [{"id": "i3", "pred_class": "a", "conf": 0.9, "fp_type": "background"}],
        "overturns": [], "queue": {"b": [{"id": "c0", "closeness": 0.99}]}})
    assert "YOLO 弱點報告" in md and "PROXY" in md and "b->a" in md and "c0" in md


def test_pipeline_weakness_report_gt_with_queue(tmp_path):
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    ad = InMemoryAdapter()
    box = [0.5, 0.5, 0.4, 0.4]
    # eval images tagged EVAL (excluded as label candidates); i1 carries the 'b' error-region embedding
    ad.seed("i1", "i1.png", [_det("b", 0.9, [0, 1, 0], BBox(0.5, 0.5, 0.2, 0.2))], tags=[Tag.EVAL])
    ad.seed("i2", "i2.png", [_det("a", 0.95, [1, 0, 0])], tags=[Tag.EVAL])
    ad.seed("i3", "i3.png", [_det("a", 0.9, [1, 0, 0])], tags=[Tag.EVAL])
    ad.seed("c0", "c0.png", [_det("b", 0.9, [0, 1, 0.02])], tags=[])  # near 'b' failure -> should head b's queue
    ad.seed("c1", "c1.png", [_det("a", 0.9, [1, 0, 0])], tags=[])     # near 'a' -> not for b
    res = tmp_path / "res.jsonl"
    res.write_text("\n".join(json.dumps(x) for x in [
        {"vix_hash": "i1", "gt": [{"label": "b", "bbox": box}], "pred": []},                                  # b missed -> b weakest
        {"vix_hash": "i2", "gt": [{"label": "a", "bbox": box}], "pred": [{"label": "a", "bbox": box, "conf": 0.95}]},  # a TP
        {"vix_hash": "i3", "gt": [], "pred": [{"label": "a", "bbox": box, "conf": 0.9}]},                      # a background FP, conf 0.9
    ]), encoding="utf-8")
    pipeline.eval_ingest(ad, cfg, str(res))

    r = pipeline.weakness_report(ad, cfg)
    d = r["data"]
    assert d["mode"] == "gt" and d["mAP"] == 0.5
    assert d["per_class"][0]["cls"] == "b"                                   # weakest (AP 0) first
    assert d["per_class"][0]["dom_fn_type"] == "missed"
    assert any(row["pred_class"] == "a" and row["conf"] == 0.9 for row in d["confident_wrong"])  # confidently-wrong FP
    assert "b" in d["queue"] and d["queue"]["b"][0]["id"] == "c0"           # class-aware queue points at b's neighbor
    assert (cfg.workspace / "weakness_report.md").exists()
    rec = [e for e in DecisionLog(cfg.decision_log_path).read_all() if e["event"] == "weakness_report"]
    assert rec and rec[0]["extra"]["weak_classes"][0] == "b"
