"""T2 challenge-guard: regression_check (pure) + set-eval-baseline + gate wiring.
Design model-loop-v2 §6 R6/R7: eval_set_hash binds the comparison; protected classes are
fail-closed (small support -> block, not advisory); no baseline -> gate behaves as before.
"""

import numpy as np

from vix import pipeline
from vix.adapters.memory import InMemoryAdapter
from vix.config import Config
from vix.core.gate import regression_check
from vix.types import BBox, Detection, Tag

_BIG = {f"c{i}": 50 for i in range(3)}  # ample support


def test_regression_overall_map_drop_blocks():
    blk, adv = regression_check({"a": 0.8}, {"a": 0.8}, current_map=0.70, baseline_map=0.80,
                                map_drop_thr=0.02, eval_support={"a": 50})
    assert any("整體 mAP 退步" in r for r in blk) and not adv


def test_regression_no_drop_passes():
    blk, adv = regression_check({"a": 0.81}, {"a": 0.80}, 0.81, 0.80, eval_support={"a": 50})
    assert not blk  # improvement -> nothing blocks


def test_protected_class_is_fail_closed_on_low_support():
    # rare protected class with too little eval data -> BLOCK (not a silent advisory pass)
    blk, adv = regression_check({"rare": 0.9}, {"rare": 0.9}, 0.9, 0.9,
                                protected={"rare": 0.05}, eval_support={"rare": 3}, min_support=20)
    assert any("受保護類別 rare" in r and "覆蓋不足" in r for r in blk)


def test_protected_class_ap_drop_blocks_even_when_overall_ok():
    blk, adv = regression_check({"p": 0.60, "x": 0.99}, {"p": 0.90, "x": 0.50},
                                current_map=0.795, baseline_map=0.70,  # overall IMPROVED
                                protected={"p": 0.05}, eval_support={"p": 50, "x": 50})
    assert any("受保護類別 p AP 退步" in r for r in blk)  # protected class still caught


def test_nonprotected_low_support_is_advisory_only():
    blk, adv = regression_check({"a": 0.5}, {"a": 0.9}, 0.5, 0.9,
                                eval_support={"a": 3}, min_support=20)
    # 'a' is not protected and under-supported -> the per-class drop is NOT blocking; only overall mAP is
    assert all("受保護" not in r for r in blk)
    assert any("樣本少" in a for a in adv)


def test_changed_eval_set_is_advisory_not_block():
    blk, adv = regression_check({"a": 0.1}, {"a": 0.9}, 0.1, 0.9,
                                eval_support={"a": 50}, eval_set_changed=True)
    assert not blk and any("eval set 已變" in a for a in adv)


# --- end-to-end through the gate ---------------------------------------------

def _det(label, emb):
    return Detection(label, 0.9, BBox(0.5, 0.5, 0.2, 0.2), embedding=np.array(emb, float))


def _seeded(tmp_path):
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    ad = InMemoryAdapter()
    for i in range(3):
        ad.seed(f"g{i}", "g.png", [_det("a", [1, 0])], tags=[Tag.GOLDEN])  # gate needs golden + clean state
    return cfg, ad


def _write_eval(cfg, images, iou=0.5):
    import json
    p = cfg.workspace / "res.jsonl"
    p.write_text("\n".join(json.dumps(i) for i in images), encoding="utf-8")
    return pipeline.eval_ingest(InMemoryAdapter(), cfg, str(p), iou_thr=iou)


def test_gate_blocks_after_baseline_then_regression(tmp_path):
    cfg, ad = _seeded(tmp_path)
    box = [0.5, 0.5, 0.4, 0.4]
    good = [{"vix_hash": f"e{i}", "gt": [{"label": "a", "bbox": box}],
             "pred": [{"label": "a", "bbox": box, "conf": 0.9}]} for i in range(30)]
    _write_eval(cfg, good)
    pipeline.set_eval_baseline(ad, cfg, protected={"a": 0.05}, map_drop_thr=0.02)
    assert pipeline.pre_train_gate_stage(ad, cfg).verdict == "GO"  # same eval -> no regression

    # same eval SET (identical GT -> same hash), but predictions now miss -> AP craters
    bad = [{"vix_hash": f"e{i}", "gt": [{"label": "a", "bbox": box}], "pred": []} for i in range(30)]
    _write_eval(cfg, bad)
    res = pipeline.pre_train_gate_stage(ad, cfg)
    assert res.verdict == "NO-GO" and any("mAP" in r or "受保護" in r for r in res.reasons)


def test_gate_unchanged_without_baseline(tmp_path):
    cfg, ad = _seeded(tmp_path)
    # no eval_baseline.json -> challenge-guard is fully opt-in, gate behaves exactly as before
    assert pipeline.pre_train_gate_stage(ad, cfg).verdict == "GO"
