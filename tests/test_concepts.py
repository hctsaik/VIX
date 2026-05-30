import numpy as np

from vix import pipeline
from vix.adapters.memory import InMemoryAdapter
from vix.config import Config
from vix.core import spc
from vix.core.calibration import apply_temperature, expected_calibration_error, fit_temperature
from vix.core.confident_learning import confident_joint, find_label_issues, noise_rates
from vix.core.drift_types import diagnose_drift_type
from vix.core.gate import cost_gate
from vix.core.parity import performance_parity
from vix.types import BBox, Decision, Detection, Tag


def _det(label, conf, emb):
    return Detection(label, conf, BBox(0.5, 0.5, 0.2, 0.2), embedding=np.array(emb, dtype=float))


# (#1) Temperature scaling
def test_temperature_scaling_reduces_ece():
    rng = np.random.RandomState(0)
    n = 400
    correct = (rng.rand(n) < 0.7).astype(float)        # true accuracy ~70%
    conf = np.clip(0.92 + 0.05 * rng.rand(n), 0, 0.999)  # but reported ~95% (over-confident)
    t = fit_temperature(conf, correct)
    before = expected_calibration_error(conf, correct)
    after = expected_calibration_error(apply_temperature(conf, t), correct)
    assert t > 1.0          # softens over-confidence
    assert after <= before  # better calibrated


# (#2) Confident learning
def test_confident_learning_finds_class_pair_noise():
    ids = [f"a{i}" for i in range(20)] + [f"b{i}" for i in range(20)] + [f"x{i}" for i in range(5)]
    given = ["a"] * 20 + ["b"] * 20 + ["a"] * 5     # x are *given* 'a'...
    pred = ["a"] * 20 + ["b"] * 20 + ["b"] * 5      # ...but confidently predicted 'b'
    conf = [0.9] * 40 + [0.97] * 5
    issues = find_label_issues(ids, given, pred, conf)
    assert any(i.id.startswith("x") for i in issues)
    C, classes, _ = confident_joint(given, pred, conf)
    assert noise_rates(C, classes).get("a->b", 0) > 0


# (#3) Drift type
def test_drift_type_covariate_vs_concept():
    rng = np.random.RandomState(1)
    a = np.tile([1, 0, 0], (20, 1)) + 0.01 * rng.randn(20, 3)
    b = np.tile([0, 1, 0], (20, 1)) + 0.01 * rng.randn(20, 3)
    assert diagnose_drift_type(a, b, ["a"] * 20, ["a"] * 20)["verdict"] in ("covariate", "both")
    assert diagnose_drift_type(a, a, ["a"] * 20, ["b"] * 20)["verdict"] in ("concept", "both")


# (#4) SPC
def test_spc_ewma_alarms_on_drift_quiet_when_flat():
    assert spc.ewma_alarm([0.05] * 20, target=0.05, sigma=0.01)["alarm"] is False
    a = spc.ewma_alarm([0.05] * 10 + [0.2] * 10, target=0.05, sigma=0.01)
    assert a["alarm"] and a["alarm_index"] >= 10


# (#5) Parity
def test_parity_flags_worse_group():
    r = performance_parity({"F1": 0.95, "F2": 0.94, "F3": 0.70}, rel_threshold=0.1)
    assert "F3" in r["flagged"] and "F1" not in r["flagged"]


# (#6) Asymmetric cost gate
def test_cost_gate_asymmetric():
    r = cost_gate(miss_rate=0.02, fa_rate=0.08, miss_cost=500, fa_cost=1, budget_per_unit=5)
    assert r["miss_component"] == 10.0 and r["verdict"] == "NO-GO"
    assert cost_gate(0.005, 0.08, 500, 1, 5)["verdict"] == "GO"


# (a) Review writeback loop
def test_sync_reviews_writeback(tmp_path):
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    ad = InMemoryAdapter()
    for i in range(5):
        ad.seed(f"g{i}", f"g{i}.png", [_det("a", 0.9, [1, 0])], tags=[Tag.GOLDEN])
    ad.seed("r1", "r1.png", [_det("a", 0.3, [1, 0])], tags=[Tag.REVIEW])
    ad.seed("r2", "r2.png", [_det("a", 0.3, [1, 0])], tags=[Tag.REVIEW])
    ad.stage_decision(Decision(vix_hash="r1", decision="bubble", reviewer_id="u1"))
    ad.stage_decision(Decision(vix_hash="r2", decision="false_alarm", reviewer_id="u1"))
    assert pipeline.sync_reviews(ad, cfg) == 2
    assert Tag.GOLDEN in next(t for h, _s, _d, t in ad.samples() if h == "r1")
    assert next(d.label for h, _s, dets, _t in ad.samples() if h == "r1" for d in dets) == "bubble"
    assert Tag.REJECTED in next(t for h, _s, _d, t in ad.samples() if h == "r2")


# pipeline wiring
def test_pipeline_label_noise(tmp_path):
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    ad = InMemoryAdapter()
    for i in range(10):
        ad.seed(f"a{i}", "a.png", [_det("a", 0.9, [1, 0])], tags=[Tag.GOLDEN])
        ad.seed(f"b{i}", "b.png", [_det("b", 0.9, [0, 1])], tags=[Tag.GOLDEN])
    for i in range(3):
        ad.seed(f"x{i}", "x.png", [_det("a", 0.9, [0, 1])], tags=[Tag.GOLDEN])  # given a, looks like b
    r = pipeline.label_noise(ad, cfg)
    assert any(iss.id.startswith("x") for iss in r["issues"])
    assert "a->b" in r["noise_rates"]


def test_pipeline_parity_by_fab(tmp_path):
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    ad = InMemoryAdapter()
    for i in range(4):
        ad.seed(f"f1_{i}", "x.png", [_det("a", 0.95, [1, 0])], tags=["fab:F1"])
        ad.seed(f"f2_{i}", "x.png", [_det("a", 0.6, [1, 0])], tags=["fab:F2"])
    assert "F2" in pipeline.parity(ad, cfg, by="fab")["flagged"]
