"""Queue hit-rate: did VIX's suggestion queues turn out right? Pure metric (precision/coverage/
trend/insufficient, predict-aware, only resolved-after-emission counts) + the pipeline log join."""

from vix import pipeline
from vix.config import Config
from vix.core.decision_log import DecisionLog
from vix.core.queue_metrics import hit_rate


def _q(rows):
    return {r["queue"]: r for r in rows}


def test_wrong_queue_precision_and_coverage():
    em = [{"queue": "hardneg", "predict": "wrong", "ids": ["a", "b", "c"], "seq": 0}]
    res = [{"id": "a", "outcome": "rejected", "seq": 1},     # predicted wrong, rejected -> hit
           {"id": "b", "outcome": "confirmed", "seq": 2}]    # predicted wrong, kept -> miss; c unresolved
    r = _q(hit_rate(em, res, min_resolved=1))["hardneg"]
    assert r["resolved"] == 2 and r["hits"] == 1 and r["precision"] == 0.5 and r["coverage"] == round(2 / 3, 3)


def test_defect_queue_hit_on_confirm():
    em = [{"queue": "bank", "predict": "defect", "ids": ["a", "b"], "seq": 0}]
    res = [{"id": "a", "outcome": "confirmed", "seq": 1}, {"id": "b", "outcome": "rejected", "seq": 2}]
    r = _q(hit_rate(em, res, min_resolved=1))["bank"]
    assert r["hits"] == 1 and r["precision"] == 0.5


def test_label_queue_precision_is_acted_on_rate():
    em = [{"queue": "wq", "predict": "label", "ids": ["a", "b"], "seq": 0}]
    res = [{"id": "a", "outcome": "confirmed", "seq": 1}, {"id": "b", "outcome": "rejected", "seq": 2}]
    r = _q(hit_rate(em, res, min_resolved=1))["wq"]
    assert r["precision"] == 1.0 and r["coverage"] == 1.0   # any resolution = acted on


def test_resolution_before_emission_is_ignored():
    em = [{"queue": "q", "predict": "wrong", "ids": ["a"], "seq": 5}]
    res = [{"id": "a", "outcome": "rejected", "seq": 2}]     # happened BEFORE the suggestion
    r = hit_rate(em, res, min_resolved=1)[0]
    assert r["resolved"] == 0 and r["precision"] is None


def test_insufficient_flag_and_trend():
    em = [{"queue": "q", "predict": "wrong", "ids": ["a", "b"], "seq": 0},
          {"queue": "q", "predict": "wrong", "ids": ["c", "d"], "seq": 3}]
    res = [{"id": "a", "outcome": "rejected", "seq": 1}, {"id": "b", "outcome": "rejected", "seq": 1},
           {"id": "c", "outcome": "rejected", "seq": 4}, {"id": "d", "outcome": "confirmed", "seq": 4}]
    r = hit_rate(em, res, min_resolved=5)[0]
    assert r["trend"] == [1.0, 0.5]                          # per-emission precision over cycles
    assert r["insufficient"] and r["resolved"] == 4


def test_pipeline_queue_hit_rate_joins_log(tmp_path):
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    pipeline._log_queue(cfg, "hardneg", ["h1", "h2"], "wrong")           # emission first
    dl = DecisionLog(cfg.decision_log_path)
    dl.append("review", vix_hash="h1", decision="false_alarm")          # hit (wrong -> rejected)
    dl.append("review", vix_hash="h2", decision="confirmed")            # miss
    r = pipeline.queue_hit_rate(cfg, min_resolved=1)
    q = _q(r["queues"])["hardneg"]
    assert r["n_emissions"] == 1 and q["resolved"] == 2 and q["precision"] == 0.5


def test_pipeline_queue_hit_rate_dismiss_is_rejected(tmp_path):
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    pipeline._log_queue(cfg, "hardneg", ["x1"], "wrong")
    DecisionLog(cfg.decision_log_path).append("dismiss", decision="1", extra={"ids": ["x1"]})
    q = _q(pipeline.queue_hit_rate(cfg, min_resolved=1)["queues"])["hardneg"]
    assert q["precision"] == 1.0                                        # dismiss == rejected == hit for 'wrong'
