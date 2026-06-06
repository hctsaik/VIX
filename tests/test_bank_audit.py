"""Bank-audit(多銀行 Top-K embedding 審查)測試:
  - 純核心:defect/reflection 投票、novelty/margin abstain、scale eps_floor、loose NMS
  - pipeline 端對端:從 tag 建銀行、審低信心 proposal、attach 諮詢欄位 + hard_positive、指紋入稽核
"""

import numpy as np

from vix import pipeline
from vix.adapters.memory import InMemoryAdapter
from vix.config import Config
from vix.core.bank_audit import bank_vote, build_bank_scales, loose_nms
from vix.core.decision_log import DecisionLog
from vix.types import BBox, Detection, Tag

_MAP = {"golden": "defect_like", "rejected": "reflection_like"}


def _bank(vec, n=8, jitter=0.01, seed=0):
    rng = np.random.RandomState(seed)
    return np.asarray(vec, float) + jitter * rng.randn(n, len(vec))


def _det(label, conf, emb, bbox=None):
    return Detection(label, conf, bbox or BBox(0.5, 0.5, 0.2, 0.2), embedding=np.array(emb, dtype=float))


def test_bank_vote_defect_and_reflection():
    banks = {"golden": _bank([1, 0, 0]), "rejected": _bank([0, 1, 0], seed=1)}
    scales = build_bank_scales(banks, k=5)
    vd = bank_vote(np.array([1, 0, 0.02]), banks, scales, _MAP, k=5, tau=0.02)
    assert vd.verdict == "defect_like" and vd.winning_bank == "golden" and vd.topk_evidence
    vr = bank_vote(np.array([0, 1, 0.02]), banks, scales, _MAP, k=5, tau=0.02)
    assert vr.verdict == "reflection_like"


def test_bank_vote_unknown_when_far():
    banks = {"golden": _bank([1, 0, 0]), "rejected": _bank([0, 1, 0], seed=1)}
    scales = build_bank_scales(banks, k=5)
    v = bank_vote(np.array([0, 0, 1]), banks, scales, _MAP, k=5, tau=0.02, novelty_radius=0.3)
    assert v.verdict == "unknown"  # orthogonal to both banks -> novelty gate


def test_build_bank_scales_eps_floor():
    dup = np.tile([1.0, 0, 0], (10, 1))  # identical -> LOO distance 0
    assert build_bank_scales({"d": dup}, k=5, eps_floor=1e-3)["d"] == 1e-3  # floored, not 0


def test_loose_nms_merges_overlap():
    a = Detection("x", 0.9, BBox(0.5, 0.5, 0.4, 0.4))
    b = Detection("x", 0.5, BBox(0.51, 0.5, 0.4, 0.4))  # ~identical to a
    c = Detection("x", 0.8, BBox(0.1, 0.1, 0.1, 0.1))   # far away
    kept = {id(d) for d in loose_nms([a, b, c], iou_thr=0.7)}
    assert id(a) in kept and id(c) in kept and id(b) not in kept  # b suppressed by higher-conf a


def test_pipeline_bank_audit_end_to_end(tmp_path):
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    ad = InMemoryAdapter()
    for i in range(8):
        ad.seed(f"g{i}", "g.png", [_det("a", 0.9, [1, 0])], tags=[Tag.GOLDEN])     # defect bank
        ad.seed(f"r{i}", "r.png", [_det("a", 0.9, [0, 1])], tags=[Tag.REJECTED])   # reflection bank
    ad.seed("p_def", "p.png", [_det("a", 0.10, [1, 0.02])], tags=[])  # low-conf, near defect
    ad.seed("p_ref", "p.png", [_det("a", 0.10, [0.02, 1])], tags=[])  # low-conf, near reflection

    r = pipeline.bank_audit(ad, cfg, tau=0.02)
    verdicts = {row["id"]: row["verdict"] for row in r["results"]}
    assert verdicts.get("p_def") == "defect_like" and verdicts.get("p_ref") == "reflection_like"
    assert r["fingerprint"] and r["banks"] == {"golden": 8, "rejected": 8}

    p_def_tags = next(t for h, _s, _d, t in ad.samples() if h == "p_def")
    assert Tag.PROPOSAL in p_def_tags and Tag.HARD_POSITIVE in p_def_tags  # defect_like -> staged
    assert ad.fields("p_def")["bank_verdict"] == "defect_like"             # advisory field attached
    p_ref_tags = next(t for h, _s, _d, t in ad.samples() if h == "p_ref")
    assert Tag.HARD_POSITIVE not in p_ref_tags                              # reflection_like NOT staged

    rec = [e for e in DecisionLog(cfg.decision_log_path).read_all() if e["event"] == "bank_audit"]
    assert rec and "bank_fingerprint" in rec[0]["extra"]                    # audit + fingerprint logged


def test_bank_audit_stages_hard_positive_from_non_representative(tmp_path):
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    ad = InMemoryAdapter()
    for i in range(8):
        ad.seed(f"g{i}", "g.png", [_det("a", 0.9, [1, 0])], tags=[Tag.GOLDEN])
        ad.seed(f"r{i}", "r.png", [_det("a", 0.9, [0, 1])], tags=[Tag.REJECTED])
    # one image, two NON-overlapping proposals: higher-conf reflection-like + lower-conf defect-like
    ad.seed("p", "p.png", [
        _det("a", 0.20, [0.02, 1], BBox(0.2, 0.2, 0.1, 0.1)),  # representative (higher conf) -> reflection-like
        _det("a", 0.10, [1, 0.02], BBox(0.8, 0.8, 0.1, 0.1)),  # non-representative -> defect-like
    ], tags=[])

    pipeline.bank_audit(ad, cfg, tau=0.02)
    tags = next(t for h, _s, _d, t in ad.samples() if h == "p")
    assert Tag.HARD_POSITIVE in tags                            # staged because a defect-like proposal exists
    assert ad.fields("p")["bank_verdict"] == "reflection_like"  # advisory field = highest-conf representative
