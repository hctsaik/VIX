"""hardneg: confidently-wrong mining (ported from SAFE). GT-mode (confirmed eval-FPs by conf) +
GT-free (high-conf detections the embedding overturns). Both offline, no training/inference."""

import json

import numpy as np

from vix import pipeline
from vix.adapters.memory import InMemoryAdapter
from vix.config import Config
from vix.core.hardneg import rank_eval_fps, rank_overturns
from vix.core.threshold import ClassThreshold, ThresholdPolicy
from vix.types import BBox, Detection, Tag


def _det(label, conf, emb):
    return Detection(label, conf, BBox(0.5, 0.5, 0.2, 0.2), embedding=np.array(emb, float))


def test_rank_eval_fps_by_conf_skips_no_conf():
    fp = {
        "i1": [{"label": "a", "bbox": [.5, .5, .2, .2], "type": "background", "conf": 0.9}],
        "i2": [{"label": "a", "bbox": [.5, .5, .2, .2], "type": "classification", "conf": 0.4},
               {"label": "b", "bbox": [.1, .1, .1, .1], "type": "background"}],  # no conf -> skipped
    }
    rows = rank_eval_fps(fp, top=10)
    assert [r["conf"] for r in rows] == [0.9, 0.4]  # most-confident mistake first; no-conf dropped
    assert rows[0]["id"] == "i1" and rows[0]["fp_type"] == "background"


def test_rank_overturns_requires_confident_AND_far():
    dets = [
        {"id": "over", "pred_class": "a", "conf": 0.9, "knn_dist": 0.8, "conf_thr": 0.3, "dist_thr": 0.1},   # confident + far -> overturn
        {"id": "near", "pred_class": "a", "conf": 0.9, "knn_dist": 0.05, "conf_thr": 0.3, "dist_thr": 0.1},  # embedding agrees -> no
        {"id": "timid", "pred_class": "a", "conf": 0.1, "knn_dist": 0.8, "conf_thr": 0.3, "dist_thr": 0.1},  # not confident -> no
        {"id": "uncal", "pred_class": "a", "conf": 0.9, "knn_dist": 0.8, "conf_thr": 0.3, "dist_thr": float("inf")},  # no bar -> skip
    ]
    rows = rank_overturns(dets, top=10)
    assert [r["id"] for r in rows] == ["over"] and rows[0]["wrongness"] > 0


def test_pipeline_hardneg_gt_mode(tmp_path):
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    cfg.eval_results_path.write_text(json.dumps(
        {"fp_detail": {"i1": [{"label": "a", "bbox": [.5, .5, .2, .2], "type": "background", "conf": 0.88}]}}),
        encoding="utf-8")
    r = pipeline.hardneg(InMemoryAdapter(), cfg, mode="auto")  # auto -> GT (fp_detail has conf)
    assert r["mode"] == "gt" and r["rows"][0]["id"] == "i1" and r["rows"][0]["conf"] == 0.88


def test_pipeline_hardneg_gt_free_overturn(tmp_path):
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    ad = InMemoryAdapter()
    for i in range(5):
        ad.seed(f"g{i}", "g.png", [_det("a", 0.9, [1, 0])], tags=[Tag.GOLDEN])
    ad.seed("over", "o.png", [_det("a", 0.9, [0, 1])], tags=[])      # confident but far from 'a' golden -> overturn
    ad.seed("near", "k.png", [_det("a", 0.9, [1, 0.02])], tags=[])   # confident + near -> not an overturn
    ThresholdPolicy({"a": ClassThreshold(conf_thr=0.3, dist_thr=0.1, n_support=5)}).save(cfg.thresholds_path)
    r = pipeline.hardneg(ad, cfg, mode="gt_free")
    ids = [row["id"] for row in r["rows"]]
    assert r["mode"] == "gt_free" and "over" in ids and "near" not in ids
