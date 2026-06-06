"""Keystone (eval-ingest 閉環) + 殘餘修補的測試:
  - core IoU 配對 / per-class AP / 混淆 / FP-FN
  - pipeline.eval_ingest 端對端 + error_mine 反查最接近誤差的候選
  - infer --synthetic 離線種子偵測
  - drift/compare 等接受裸 batch id(自動解析 batch:<id>)
"""

import json

import numpy as np

from vix import pipeline
from vix.adapters.memory import InMemoryAdapter
from vix.config import Config
from vix.core.eval_ingest import evaluate, iou
from vix.types import BBox, Detection


def _det(label, conf, emb):
    return Detection(label, conf, BBox(0.5, 0.5, 0.2, 0.2), embedding=np.array(emb, dtype=float))


def test_iou_basic():
    assert iou((0.5, 0.5, 1, 1), (0.5, 0.5, 1, 1)) == 1.0
    assert iou((0.25, 0.5, 0.5, 1), (0.75, 0.5, 0.5, 1)) == 0.0  # adjacent, no overlap


def test_evaluate_ap_confusion_fn_fp():
    box = [0.5, 0.5, 0.4, 0.4]
    images = [
        {"vix_hash": "i1", "gt": [{"label": "a", "bbox": box}], "pred": [{"label": "a", "bbox": box, "conf": 0.9}]},
        {"vix_hash": "i2", "gt": [{"label": "b", "bbox": box}], "pred": [{"label": "a", "bbox": box, "conf": 0.8}]},
        {"vix_hash": "i3", "gt": [{"label": "a", "bbox": box}], "pred": []},
    ]
    r = evaluate(images, iou_thr=0.5)
    assert r["confusion"].get("b->a") == 1                 # truth b mis-detected as a
    assert set(r["fn_hashes"]) == {"i2", "i3"}             # i2 missed b, i3 missed a
    assert r["per_image"]["i2"] == {"n_fp": 1, "n_fn": 1}
    assert r["per_class_ap"]["a"] == 0.5 and r["per_class_ap"]["b"] == 0.0
    assert r["mAP"] == 0.25


def test_pipeline_eval_ingest_and_error_mine(tmp_path):
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    ad = InMemoryAdapter()
    ad.seed("i2", "i2.png", [_det("a", 0.9, [1, 0])], tags=[])        # the error image (FN)
    ad.seed("c0", "c0.png", [_det("a", 0.9, [1, 0.01])], tags=[])     # near the error
    ad.seed("c1", "c1.png", [_det("a", 0.9, [0, 1])], tags=[])        # far from the error
    res = tmp_path / "res.jsonl"
    res.write_text(
        json.dumps({"vix_hash": "i2", "gt": [{"label": "a", "bbox": [0.5, 0.5, 0.4, 0.4]}], "pred": []}) + "\n",
        encoding="utf-8",
    )
    r = pipeline.eval_ingest(ad, cfg, str(res))
    assert "i2" in r["fn_hashes"] and (cfg.workspace / "eval_results.json").exists()

    mined = pipeline.error_mine(ad, cfg, top=5)
    ids = [m["id"].split(":")[0] for m in mined]
    assert ids and ids.index("c0") < ids.index("c1")  # candidate near the model's error ranks first


def test_infer_synthetic_seeds_detections(tmp_path):
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    ad = InMemoryAdapter()
    ad.seed("h0", str(tmp_path / "cat" / "x.png"), [], tags=[])  # no detections yet
    n = pipeline.infer_synthetic(ad, cfg)
    assert n == 1
    dets = next(d for h, _s, d, _t in ad.samples() if h == "h0")
    assert len(dets) == 1 and dets[0].label == "cat"  # label from source parent folder


def test_resolve_bare_batch_tag(tmp_path):
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    ad = InMemoryAdapter()
    ad.seed("a0", "a.png", [_det("x", 0.9, [1, 0])], tags=["batch:w20"])
    ad.seed("b0", "b.png", [_det("x", 0.9, [0, 1])], tags=["batch:w23"])
    r = pipeline.drift_type(ad, cfg, "w20", "w23")  # bare ids resolve to batch:w20 / batch:w23
    assert r["verdict"] in ("covariate", "concept", "both", "none")  # compared non-empty, no crash
