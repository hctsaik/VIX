import numpy as np

from vix.core.analytics import EmbItem
from vix.core.gate import pre_train_gate
from vix.core.quality import class_quality_trend, reviewer_consistency
from vix.core.verify import verify_export, write_export_manifest


def _it(id, label, emb, conf=1.0, batch=""):
    return EmbItem(id, label, np.array(emb, dtype=float), conf, batch=batch)


def test_reviewer_consistency_flags_contradiction():
    items = [_it("x1", "a", [1, 0]), _it("x2", "a", [1, 0])]  # near-identical
    decisions = [
        {"reviewer_id": "u1", "id": "x1", "decision": "pass"},
        {"reviewer_id": "u1", "id": "x2", "decision": "reject"},
    ]
    res = reviewer_consistency(decisions, items, sim_threshold=0.9)
    assert res["u1"]["intra_consistency"] == 0.0
    assert len(res["u1"]["conflicts"]) == 1


def test_class_quality_trend_drop_alert():
    items = [_it(f"a{i}", "a", [1, 0], conf=0.9, batch="w1") for i in range(3)]
    items += [_it(f"b{i}", "a", [1, 0], conf=0.5, batch="w2") for i in range(3)]
    res = class_quality_trend(items, drop_threshold=0.15)
    assert any(al["class"] == "a" and al["batch"] == "w2" for al in res["alerts"])


def test_pre_train_gate_go_and_nogo():
    assert pre_train_gate().verdict == "GO"
    r = pre_train_gate(n_review_open=3, golden_train_overlap=1, under_represented=["x"], drift_triggered=True)
    assert r.verdict == "NO-GO" and len(r.reasons) == 4


def test_verify_export_detects_tamper(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.png").write_bytes(b"AAAA")
    (src / "b.png").write_bytes(b"BBBB")
    records = [(str(src / "a.png"), []), (str(src / "b.png"), [])]
    man = write_export_manifest(records, tmp_path / "export")

    recv = tmp_path / "recv" / "images"
    recv.mkdir(parents=True)
    (recv / "a.png").write_bytes(b"AAAA")
    (recv / "b.png").write_bytes(b"BBBB")
    assert verify_export(man, tmp_path / "recv")["ok"] is True

    (recv / "a.png").write_bytes(b"XXXX")  # tamper
    res = verify_export(man, tmp_path / "recv")
    assert res["ok"] is False and "a.png" in res["mismatched"]
