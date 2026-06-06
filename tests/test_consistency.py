"""GT-powered consistency attribution: LOO-kNN separability + embedding-overlap + 2x2 verdict
(taxonomy / model / label_noise / insufficient_support), and the pipeline integration + HTML write."""

import json

import numpy as np

from vix import pipeline
from vix.adapters.memory import InMemoryAdapter
from vix.config import Config
from vix.core.consistency import consistency_findings, k_rule, pair_stats, wilson
from vix.types import BBox, Detection, Tag


def _cluster(center, n, jit, seed):
    rng = np.random.RandomState(seed)
    return np.asarray(center, float) + jit * rng.randn(n, len(center))


_SEP_A = _cluster([1, 0, 0, 0], 25, 0.05, 1)   # tight, far from B
_SEP_B = _cluster([0, 1, 0, 0], 25, 0.05, 2)
_OVL_A = _cluster([1, 0, 0, 0], 25, 0.6, 1)     # same centre as B -> inseparable
_OVL_B = _cluster([1, 0, 0, 0], 25, 0.6, 2)


def test_wilson_and_k_rule():
    lo, hi = wilson(5, 10)
    assert 0.0 <= lo < 0.5 < hi <= 1.0
    assert k_rule(25) == 5 and k_rule(4) == 3 and k_rule(1) == 3  # floor 3, odd


def test_pair_stats_separable_vs_overlapping():
    sep = pair_stats(_SEP_A, _SEP_B)
    ovl = pair_stats(_OVL_A, _OVL_B)
    assert sep["sep_err"] < 0.10 and sep["O_ij"] < 0.15      # cleanly separable in embedding
    assert ovl["sep_err"] > 0.35 and ovl["O_ij"] > 0.30      # inseparable / overlapping


def test_verdict_taxonomy_overlap_and_confusion():
    # overlapping embeddings + matching model confusion (C~O) -> taxonomy
    f = consistency_findings({"a": _OVL_A, "b": _OVL_B}, confusion={"a->b": 10}, n_gt={"a": 20, "b": 20})
    pair = next(x for x in f if set(x["pair"]) == {"a", "b"})
    assert pair["verdict"] == "taxonomy" and pair["separable_in_embedding"] == "no"


def test_verdict_model_separable_but_confused():
    # separable embeddings + high model confusion -> model defect (not labels, not taxonomy)
    f = consistency_findings({"a": _SEP_A, "b": _SEP_B}, confusion={"a->b": 10}, n_gt={"a": 20, "b": 20})
    pair = next(x for x in f if set(x["pair"]) == {"a", "b"})
    assert pair["verdict"] == "model" and pair["separable_in_embedding"] == "yes"


def test_verdict_label_noise_overlap_without_confusion():
    # overlapping embeddings but model does NOT confuse them -> label-noise (re-adjudicate)
    f = consistency_findings({"a": _OVL_A, "b": _OVL_B}, confusion={}, n_gt={"a": 20, "b": 20})
    pair = next(x for x in f if set(x["pair"]) == {"a", "b"})
    assert pair["verdict"] == "label_noise"


def test_insufficient_support_small_gt():
    f = consistency_findings({"a": _OVL_A[:5], "b": _OVL_B[:5]}, confusion={"a->b": 3}, n_gt={"a": 5, "b": 5})
    pair = next(x for x in f if set(x["pair"]) == {"a", "b"})
    assert pair["verdict"] == "insufficient_support"  # never a confident verdict on thin GT


def _det(label, emb):
    return Detection(label, 0.9, BBox(0.5, 0.5, 0.2, 0.2), embedding=np.asarray(emb, float))


def test_pipeline_consistency_and_html(tmp_path):
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    ad = InMemoryAdapter()
    for i, v in enumerate(_OVL_A):
        ad.seed(f"a{i}", "a.png", [_det("a", v)], tags=[Tag.GOLDEN])
    for i, v in enumerate(_OVL_B):
        ad.seed(f"b{i}", "b.png", [_det("b", v)], tags=[Tag.GOLDEN])
    box = [0.5, 0.5, 0.4, 0.4]
    imgs = [{"vix_hash": f"e{i}", "gt": [{"label": "a", "bbox": box}],
             "pred": ([{"label": "b", "bbox": box, "conf": 0.9}] if i < 10 else [])} for i in range(20)]
    (tmp_path / "res.jsonl").write_text("\n".join(json.dumps(x) for x in imgs), encoding="utf-8")
    pipeline.eval_ingest(ad, cfg, str(tmp_path / "res.jsonl"))  # confusion a->b = 10 / n_gt a = 20 -> C ~ 0.5

    r = pipeline.consistency(ad, cfg)
    assert r["has_eval"] and any(f["verdict"] == "taxonomy" for f in r["findings"])

    wr = pipeline.weakness_report(ad, cfg)
    assert wr.get("html") and wr["data"]["consistency"]
    html = (cfg.workspace / "weakness_report.html").read_text(encoding="utf-8")
    assert "id='consistency'" in html and "taxonomy" in html  # the headline surface renders
