"""Round 13 補強的回歸測試:
  - DecisionLog 容忍當機半行 + 鎖內鏈接(verify 不崩潰、優雅降級)
  - vix set-threshold 逐類門檻覆寫(改 JSON + 記稽核)
  - vix reasons 依 flag_reason 彙總
  - parity 小樣本不誤判(low_confidence)
"""

from vix import pipeline
from vix.config import Config
from vix.core.decision_log import DecisionLog
from vix.core.parity import performance_parity
from vix.core.threshold import ClassThreshold, ThresholdPolicy


def test_decision_log_tolerates_torn_line(tmp_path):  # AE4/AE5
    dl = DecisionLog(tmp_path / "log.jsonl")
    dl.append("route", vix_hash="a", decision="review")
    with open(dl.path, "a", encoding="utf-8") as f:  # simulate a crash mid-write
        f.write('{"event":"route","vix_hash":"b"')  # half a line, no newline
    recs = dl.read_all()
    assert len(recs) == 1 and recs[0]["vix_hash"] == "a"  # torn line skipped, no crash
    assert dl.verify_chain() is True  # graceful

    dl.append("route", vix_hash="c", decision="review")  # next append chains from last good record
    assert [r["vix_hash"] for r in dl.read_all()] == ["a", "c"]
    assert dl.verify_chain() is True


def test_set_threshold_override_audited(tmp_path):  # AE6
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    ThresholdPolicy(
        {"crack": ClassThreshold(0.3, 1.0, 50), "scuff": ClassThreshold(0.3, 1.0, 50)}
    ).save(cfg.thresholds_path)

    r = pipeline.set_threshold(None, cfg, "crack", conf_thr=0.8)
    assert r["conf_thr"] == 0.8
    reloaded = ThresholdPolicy.load(cfg.thresholds_path)
    assert reloaded.thresholds["crack"].conf_thr == 0.8  # safety-critical class tightened
    assert reloaded.thresholds["scuff"].conf_thr == 0.3  # other class untouched
    assert "crack" in reloaded.meta.get("overrides", {})
    assert any(e["event"] == "set_threshold" for e in DecisionLog(cfg.decision_log_path).read_all())


def test_reasons_breakdown(tmp_path):  # AE7
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    dl = DecisionLog(cfg.decision_log_path)
    dl.append("route", vix_hash="a", decision="review", extra={"reasons": ["low_conf"]})
    dl.append("route", vix_hash="b", decision="review", extra={"reasons": ["low_conf", "far_from_known"]})
    dl.append("route", vix_hash="c", decision="pass")
    dl.append("dismiss", decision="1", extra={"ids": ["a"]})

    r = pipeline.reasons_breakdown(cfg)
    assert r["n_review"] == 2
    assert r["by_reason"]["low_conf"] == 2 and r["by_reason"]["far_from_known"] == 1
    assert r["rejected"] == 1


def test_parity_low_sample_not_flagged():  # AE9
    r = performance_parity(
        {"big": 0.95, "tiny": 0.50}, group_counts={"big": 100, "tiny": 1}, min_samples=5
    )
    assert r["groups"]["tiny"]["low_confidence"] is True
    assert "tiny" not in r["flagged"]  # a 1-sample group is not a trustworthy "worse" verdict
