"""Report trend over time: per-class AP / mAP / health read from the decision log, with the
eval-set-changed honesty flag. Pure series-builder + the pipeline log read."""

from vix import pipeline
from vix.config import Config
from vix.core.decision_log import DecisionLog
from vix.core.trend import eval_trend


def _rec(event, extra, ts="t"):
    return {"event": event, "extra": extra, "ts_utc": ts}


def test_per_class_delta_same_eval_set():
    recs = [
        _rec("eval_ingest", {"mAP": 0.5, "per_class_ap": {"bubble": 0.4, "scratch": 0.9}, "eval_set_hash": "h1"}, "t1"),
        _rec("eval_ingest", {"mAP": 0.65, "per_class_ap": {"bubble": 0.6, "scratch": 0.9}, "eval_set_hash": "h1"}, "t2"),
    ]
    t = eval_trend(recs)
    assert t["n_evals"] == 2 and not t["eval_set_changed"]
    assert t["per_class_delta"]["bubble"] == 0.2 and t["per_class_delta"]["scratch"] == 0.0
    assert [v for _ts, v in t["mAP_series"]] == [0.5, 0.65]


def test_flags_changed_eval_set():
    recs = [
        _rec("eval_ingest", {"mAP": 0.5, "per_class_ap": {"a": 0.4}, "eval_set_hash": "h1"}, "t1"),
        _rec("eval_ingest", {"mAP": 0.9, "per_class_ap": {"a": 0.9}, "eval_set_hash": "h2"}, "t2"),  # different val set
    ]
    t = eval_trend(recs)
    assert t["eval_set_changed"] and "不可直接比較" in t["note"]


def test_health_series_from_weakness_report_events():
    recs = [_rec("weakness_report", {"health": "RED"}, "t1"), _rec("weakness_report", {"health": "AMBER"}, "t2")]
    t = eval_trend(recs)
    assert [v for _ts, v in t["health_series"]] == ["RED", "AMBER"]


def test_pipeline_report_trend_reads_log(tmp_path):
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    dl = DecisionLog(cfg.decision_log_path)
    dl.append("eval_ingest", decision="0.4", extra={"mAP": 0.4, "per_class_ap": {"bubble": 0.3}, "eval_set_hash": "h1"})
    dl.append("eval_ingest", decision="0.55", extra={"mAP": 0.55, "per_class_ap": {"bubble": 0.5}, "eval_set_hash": "h1"})
    t = pipeline.report_trend(cfg)
    assert t["n_evals"] == 2 and t["per_class_delta"]["bubble"] == 0.2 and not t["eval_set_changed"]
