"""Round 12 補強的回歸測試:
  - export 排除 rejected/dismissed(PII 移除真的不會匯出)
  - relabel / rollback 在持久化記憶體 adapter 下跨指令生效
  - 稽核鏈損毀 -> gate NO-GO
  - fp-rate 納入 resolve 的 false_alarm
"""

import numpy as np

from vix import pipeline
from vix.adapters.memory import InMemoryAdapter
from vix.config import Config
from vix.core.decision_log import DecisionLog
from vix.core.gate import pre_train_gate
from vix.types import BBox, Detection, Tag


def _det(label, conf, emb):
    return Detection(label, conf, BBox(0.5, 0.5, 0.2, 0.2), embedding=np.array(emb, dtype=float))


def test_export_excludes_rejected(tmp_path):  # AD8
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    ad = InMemoryAdapter()
    ad.seed("keep0", "keep0.png", [_det("a", 0.9, [1, 0])], tags=[Tag.GOLDEN])
    ad.seed("keep1", "keep1.png", [_det("a", 0.9, [1, 0])], tags=[Tag.GOLDEN])
    ad.seed("pii", "pii.png", [_det("a", 0.9, [1, 0])], tags=[Tag.GOLDEN])
    pipeline.dismiss(ad, cfg, ["pii"])  # PII/removal request -> rejected

    dst = tmp_path / "out"
    res = pipeline.export(ad, cfg, ["a"], dst)
    assert res["n_images"] == 2  # only the two non-rejected golden samples
    assert (dst / "labels" / "train" / "keep0.txt").exists()
    assert not (dst / "labels" / "train" / "pii.txt").exists()  # removed sample is absent
    assert "pii" not in (dst / "export_manifest.jsonl").read_text(encoding="utf-8")


def test_relabel_persists_and_rolls_back_across_instances(tmp_path):  # AD7
    ws = tmp_path / "ws"
    cfg = Config(workspace=ws)
    cfg.ensure_dirs()
    sp = ws / "memory_state.pkl"

    a = InMemoryAdapter(state_path=sp)
    a.seed("g0", "g0.png", [_det("a", 0.9, [1, 0])], tags=[Tag.GOLDEN])
    pipeline.relabel_dataset(a, cfg, {"a": "b"})

    b = InMemoryAdapter(state_path=sp)  # fresh process
    assert next(d.label for _h, _s, dets, _t in b.samples() for d in dets) == "b"

    pipeline.relabel_rollback(b, cfg)
    c = InMemoryAdapter(state_path=sp)  # fresh process again
    assert next(d.label for _h, _s, dets, _t in c.samples() for d in dets) == "a"


def test_gate_blocks_on_broken_audit_chain():  # AD9
    assert pre_train_gate(audit_chain_intact=True).verdict == "GO"
    r = pre_train_gate(audit_chain_intact=False)
    assert r.verdict == "NO-GO"
    assert any("稽核鏈" in reason for reason in r.reasons)


def test_fp_rate_counts_resolve_false_alarm(tmp_path):  # AD3/AD10
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    dl = DecisionLog(cfg.decision_log_path)
    dl.append("route", vix_hash="x1", decision="review")
    dl.append("review", vix_hash="x1", decision="false_alarm")  # resolved as false alarm (not `dismiss`)
    r = pipeline.false_positive_rate(cfg)
    assert r["reviewed"] == 1
    assert r["dismissed_false_alarms"] == 1  # the resolve-path false alarm is counted
    assert r["fp_rate"] == 1.0
