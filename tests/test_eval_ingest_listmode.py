"""eval_ingest accepts a list OR a path (polymorphic, additive) — and the strict_join
guard rejects a stem-keyed eval against a content-hash manifest (kill-shot #3)."""

import json

import pytest

from vix.adapters.memory import InMemoryAdapter
from vix.config import Config
from vix.pipeline import eval_ingest
from vix.types import BBox, Detection, Tag


def _seed(ad):
    for h in ("h1", "h2"):
        ad.seed(h, f"{h}.png", [Detection("a", 0.9, BBox(0.5, 0.5, 0.2, 0.2))], tags=[Tag.EVAL])


def _images():
    return [
        {"vix_hash": "h1", "gt": [{"label": "a", "bbox": [0.5, 0.5, 0.2, 0.2]}],
         "pred": [{"label": "a", "bbox": [0.5, 0.5, 0.2, 0.2], "conf": 0.9}]},
        {"vix_hash": "h2", "gt": [{"label": "a", "bbox": [0.3, 0.3, 0.2, 0.2]}],
         "pred": [{"label": "a", "bbox": [0.8, 0.8, 0.1, 0.1], "conf": 0.8}]},
    ]


def test_eval_ingest_accepts_list_equals_path(tmp_path):
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    imgs = _images()
    jl = tmp_path / "eval.jsonl"
    jl.write_text("\n".join(json.dumps(x) for x in imgs), encoding="utf-8")

    ad1 = InMemoryAdapter(); _seed(ad1)
    res_path = eval_ingest(ad1, cfg, str(jl))
    file_path = cfg.eval_results_path.read_text(encoding="utf-8")

    ad2 = InMemoryAdapter(); _seed(ad2)
    res_list = eval_ingest(ad2, cfg, imgs)  # list form
    file_list = cfg.eval_results_path.read_text(encoding="utf-8")

    assert res_path == res_list          # identical result dict
    assert file_path == file_list        # identical eval_results.json bytes
    # and per-image FP/FN actually attached
    assert ad2.fields("h2")["eval_fp"] == 1 and ad2.fields("h2")["eval_fn"] == 1


def test_strict_join_rejects_stem_keyed_results(tmp_path):
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    ad = InMemoryAdapter(); _seed(ad)  # manifest keys: h1, h2
    stem_keyed = [{"vix_hash": "image_0001",  # a filename stem, NOT a content hash
                   "gt": [{"label": "a", "bbox": [0.5, 0.5, 0.2, 0.2]}], "pred": []}]
    with pytest.raises(ValueError, match="不在資料集裡"):
        eval_ingest(ad, cfg, stem_keyed, strict_join=True)


def test_nonstrict_tolerates_external_hashes(tmp_path):
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    ad = InMemoryAdapter(); _seed(ad)
    external = [{"vix_hash": "not_seeded",
                 "gt": [{"label": "a", "bbox": [0.5, 0.5, 0.2, 0.2]}], "pred": []}]
    res = eval_ingest(ad, cfg, external)  # legacy best-effort: no raise
    assert res["mAP"] == 0.0
