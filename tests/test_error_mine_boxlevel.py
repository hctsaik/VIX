"""T1b: box-level error-mine (model-loop-v2 R1). The typed fp/fn boxes are IoU-matched back
to stored detection embeddings, so mining keys on the actual error REGION rather than the
whole-image detection mean — decisive when a small defect (FN) sits next to a reflection (FP)
in the same image. Falls back to the image mean (no crash) when no box matches."""

import json

import numpy as np

from vix import pipeline
from vix.adapters.memory import InMemoryAdapter
from vix.config import Config
from vix.types import BBox, Detection, Tag


def _write(cfg, image):
    p = cfg.workspace / "res.jsonl"
    p.write_text(json.dumps(image) + "\n", encoding="utf-8")
    return str(p)


def test_error_mine_keys_on_region_not_image_mean(tmp_path):
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    ad = InMemoryAdapter()
    box_def, box_ref = [0.25, 0.25, 0.2, 0.2], [0.75, 0.75, 0.2, 0.2]
    # error image: a defect-region detection [1,0] at box_def + a reflection detection [0,1] at box_ref
    ad.seed("e", "e.png", [
        Detection("a", 0.9, BBox(*box_def), embedding=np.array([1.0, 0.0])),
        Detection("a", 0.9, BBox(*box_ref), embedding=np.array([0.0, 1.0])),
    ], tags=[])
    ad.seed("c_def", "c.png", [Detection("a", 0.9, BBox(0.5, 0.5, 0.2, 0.2), embedding=np.array([1.0, 0.02]))], tags=[])
    ad.seed("c_mid", "c.png", [Detection("a", 0.9, BBox(0.5, 0.5, 0.2, 0.2), embedding=np.array([0.707, 0.707]))], tags=[])
    # GT defect at box_def is MISSED (the only pred is a reflection at box_ref = background FP)
    pipeline.eval_ingest(ad, cfg, _write(cfg, {
        "vix_hash": "e", "gt": [{"label": "a", "bbox": box_def}],
        "pred": [{"label": "a", "bbox": box_ref, "conf": 0.9}]}))
    mined = {m["id"].split(":")[0]: m["closeness"] for m in pipeline.error_mine(ad, cfg, top=5)}
    # box-level: max cosine to {defect [1,0], reflection [0,1]} -> defect-near candidate wins (~1.0);
    # the image-mean candidate [.707,.707] only reaches ~.707. (image-mean mining would invert this.)
    assert mined["c_def"] > 0.99 and mined["c_def"] > mined["c_mid"]


def test_error_mine_falls_back_when_no_box_matches(tmp_path):
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    ad = InMemoryAdapter()
    # stored detection nowhere near the GT box -> no IoU matchback -> image-mean fallback, still ranks
    ad.seed("e", "e.png", [Detection("a", 0.9, BBox(0.1, 0.1, 0.05, 0.05), embedding=np.array([1.0, 0.0]))], tags=[])
    ad.seed("c0", "c.png", [Detection("a", 0.9, BBox(0.5, 0.5, 0.2, 0.2), embedding=np.array([1.0, 0.0]))], tags=[])
    pipeline.eval_ingest(ad, cfg, _write(cfg, {
        "vix_hash": "e", "gt": [{"label": "a", "bbox": [0.8, 0.8, 0.2, 0.2]}], "pred": []}))
    mined = pipeline.error_mine(ad, cfg, top=5)
    assert any(m["id"].split(":")[0] == "c0" for m in mined)  # degrades cleanly, no crash


def test_error_mine_batch_scope(tmp_path):
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    ad = InMemoryAdapter()
    ad.seed("e", "e.png", [Detection("a", 0.9, BBox(0.5, 0.5, 0.2, 0.2), embedding=np.array([1.0, 0.0]))], tags=[Tag.EVAL])
    ad.seed("c_in", "c.png", [Detection("a", 0.9, BBox(0.5, 0.5, 0.2, 0.2), embedding=np.array([1.0, 0.0]))], tags=["batch:w23"])
    ad.seed("c_out", "c.png", [Detection("a", 0.9, BBox(0.5, 0.5, 0.2, 0.2), embedding=np.array([1.0, 0.0]))], tags=[])
    pipeline.eval_ingest(ad, cfg, _write(cfg, {
        "vix_hash": "e", "gt": [{"label": "a", "bbox": [0.5, 0.5, 0.4, 0.4]}], "pred": []}))
    ids_all = {m["id"].split(":")[0] for m in pipeline.error_mine(ad, cfg, top=10)}
    ids_b = {m["id"].split(":")[0] for m in pipeline.error_mine(ad, cfg, top=10, batch="w23")}
    assert {"c_in", "c_out"} <= ids_all                       # unscoped: both candidates
    assert "c_in" in ids_b and "c_out" not in ids_b           # batch scope: only this batch's candidates
