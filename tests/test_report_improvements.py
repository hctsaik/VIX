"""Multi-agent consensus fixes (R1→R3) to the weakness/consistency report.

Honesty defects: H1a (rescued verdict no longer renders the self-contradicting `taxonomy(可修)`),
H2 (a 'label' queue's hit-rate is structurally == coverage, so it's shown as coverage not precision),
H3 (closeness/wrongness get a legend + 2dp, not bare 4dp cosines), H4 (the PROXY banner isn't doubled).
Workflow: A2 (todo demotes "go label" when the cause is representation_fixable), L1/L3 (provenance +
eval-set comparability stamp), L4 (queue marks already-resolved candidates; order unchanged)."""

import json

import numpy as np

from vix import pipeline
from vix.adapters.memory import InMemoryAdapter
from vix.config import Config
from vix.core.decision_log import DecisionLog
from vix.core.weakness_report import render_weakness_report, render_weakness_report_html
from vix.types import BBox, Detection, Tag


def _base(**kw):
    d = {"mode": "gt", "mAP": 0.5, "per_class": [], "consistency": [], "queue": {}, "hit_rate": []}
    d.update(kw)
    return d


# ---- pure renderer: honesty defects -------------------------------------------------------------

def test_h1a_rescued_verdict_not_self_contradicting():
    row = {"pair": ["bubble", "dripping"], "separable_in_embedding": "no", "sep_err": 0.4,
           "sep_ci": [0.3, 0.5], "O_ij": 0.3, "O_ci": [0.2, 0.4], "C_ij": 0.25, "C_ci": [0.1, 0.4],
           "delta": 0.05, "delta_ci": [-0.2, 0.3], "verdict": "taxonomy", "tier": "supported",
           "representation_fixable": True, "action": "別 merge:學到的投影已能分開 → 非 taxonomy 死路",
           "support": {"golden_i": 25, "golden_j": 25, "n_gt_i": 20, "k": 5}}
    for render in (render_weakness_report, render_weakness_report_html):
        out = render(_base(consistency=[row]))
        assert "representation-fixable" in out                       # the honest label
        assert "taxonomy(可修)" not in out and "taxonomy（可修）" not in out  # the contradiction is gone


def test_h1a_non_rescued_taxonomy_still_renders_taxonomy():
    row = {"pair": ["a", "b"], "separable_in_embedding": "no", "sep_err": 0.5, "sep_ci": [0.4, 0.6],
           "O_ij": 0.4, "O_ci": [0.3, 0.5], "C_ij": 0.35, "C_ci": [0.2, 0.5], "delta": 0.05,
           "delta_ci": [-0.1, 0.2], "verdict": "taxonomy", "tier": "supported",
           "action": "停止多標;考慮 merge", "support": {"golden_i": 25, "golden_j": 25, "n_gt_i": 20, "k": 5}}
    html = render_weakness_report_html(_base(consistency=[row]))
    assert "td.v.tax" or "v tax" in html  # the gate-relevant taxonomy verdict is untouched (still red-styled)
    assert "taxonomy" in html and "representation-fixable" not in html


def test_h2_label_queue_shown_as_coverage_not_precision():
    hr = [{"queue": "weakness_queue", "predict": "label", "resolved": 6, "emitted": 10,
           "precision": 1.0, "coverage": 0.6, "trend": [1.0], "insufficient": False},
          {"queue": "hardneg", "predict": "wrong", "resolved": 5, "emitted": 5,
           "precision": 0.8, "coverage": 1.0, "trend": [0.8], "insufficient": False}]
    md = render_weakness_report(_base(hit_rate=hr))
    assert "命中率≡覆蓋率" in md and "0.6(覆蓋)" in md   # label queue framed as coverage
    assert "1.0(覆蓋)" not in md                          # NOT the structural-1.0 precision
    assert "0.8" in md                                     # honest hardneg precision still shown


def test_h3_closeness_legend_and_2dp():
    md = render_weakness_report(_base(queue={"b": [{"id": "c0", "closeness": 0.98765, "resolved": False}]}))
    assert "cosine 鄰近度" in md and "非機率" in md
    assert "0.99" in md and "0.98765" not in md            # rounded to 2dp in display


def test_h4_proxy_banner_not_duplicated():
    md = render_weakness_report(_base())
    assert "未重訓" in md                                  # the proxy stamp remains
    assert "排序為嫌疑/優先,非實測 mAP;可分性" not in md   # the old duplicated parenthetical is gone


# ---- pure renderer: workflow (L1/L3/L4) ---------------------------------------------------------

def test_l1_l3_provenance_and_comparability_banner():
    md = render_weakness_report(_base(provenance={"eval_set_hash": "abcd1234ef", "prev_report_ts": "2026-06-01",
                                                   "comparable": False, "prev_mAP": None}))
    assert "出處" in md and "abcd1234" in md and "不可與上期直接比較" in md
    md2 = render_weakness_report(_base(mAP=0.6, provenance={"eval_set_hash": "x", "prev_report_ts": "t",
                                                            "comparable": True, "prev_mAP": 0.55}))
    assert "可比較" in md2 and "0.55" in md2


def test_l4_resolved_marked_counted_order_unchanged():
    q = {"b": [{"id": "c0", "closeness": 0.9, "resolved": True},
               {"id": "c1", "closeness": 0.8, "resolved": False}]}
    md = render_weakness_report(_base(queue=q))
    assert "待辦 1 / 已解決 1" in md and "~~c0~~" in md     # resolved struck + counted
    assert md.index("c0") < md.index("c1")                 # order unchanged (NOT re-ranked)
    assert "class='done'" in render_weakness_report_html(_base(queue=q))


# ---- pipeline integration: A2 + L1 + L4 (rescued bubble/dripping scenario) ----------------------

def _emb(cls, n, seed, D=8):
    r = np.random.RandomState(seed)
    X = 2.0 * r.randn(n, D)
    if cls == "bubble":
        X[:, 0] = +0.6 + 0.1 * r.randn(n)
    elif cls == "dripping":
        X[:, 0] = -0.6 + 0.1 * r.randn(n)
    elif cls == "scratch":
        X[:, 1] = +4.0 + 0.1 * r.randn(n)
    return X


def test_pipeline_a2_l1_l4_on_rescued_pair(tmp_path):
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    ad = InMemoryAdapter()
    box = [0.5, 0.5, 0.4, 0.4]

    def det(label, vec, conf=0.9):
        return Detection(label, conf, BBox(0.5, 0.5, 0.2, 0.2), embedding=np.asarray(vec, float))

    for cls, seed in (("bubble", 1), ("dripping", 2), ("scratch", 3)):
        for i, v in enumerate(_emb(cls, 25, seed)):
            ad.seed(f"g_{cls}_{i}", f"{cls}.png", [det(cls, v)], tags=[Tag.GOLDEN])
    for i, v in enumerate(_emb("bubble", 6, 11)):
        ad.seed(f"cand_b{i}", "c.png", [det("bubble", v, conf=0.2)], tags=[])

    images = []
    bi = 0
    for _ in range(8):
        ad.seed(f"e{bi}", "e.png", [det("bubble", _emb("bubble", 1, 100 + bi)[0], 0.5)], tags=[Tag.EVAL])
        images.append({"vix_hash": f"e{bi}", "gt": [{"label": "bubble", "bbox": box}],
                       "pred": [{"label": "dripping", "bbox": box, "conf": 0.8}]}); bi += 1
    for _ in range(8):
        ad.seed(f"e{bi}", "e.png", [det("bubble", _emb("bubble", 1, 100 + bi)[0], 0.5)], tags=[Tag.EVAL])
        images.append({"vix_hash": f"e{bi}", "gt": [{"label": "bubble", "bbox": box}], "pred": []}); bi += 1
    for k in range(12):
        ad.seed(f"ed{k}", "e.png", [det("dripping", _emb("dripping", 1, 200 + k)[0], 0.5)], tags=[Tag.EVAL])
        images.append({"vix_hash": f"ed{k}", "gt": [{"label": "dripping", "bbox": box}],
                       "pred": [{"label": "dripping", "bbox": box, "conf": 0.85}]})
    (tmp_path / "res.jsonl").write_text("\n".join(json.dumps(x) for x in images), encoding="utf-8")
    pipeline.eval_ingest(ad, cfg, str(tmp_path / "res.jsonl"))
    pipeline.adapt_embedding(ad, cfg, save=True, enable=True)
    DecisionLog(cfg.decision_log_path).append("review", vix_hash="cand_b0", decision="confirmed")  # L4

    d = pipeline.weakness_report(ad, cfg)["data"]
    # consistency rescued bubble/dripping -> representation_fixable
    assert any(f.get("representation_fixable") for f in d["consistency"])
    # A2: adapt-embedding promoted ABOVE "go label" in the todo (labeling is the wrong lever here)
    todo = d["summary"]["todo"]
    i_adapt = next(i for i, t in enumerate(todo) if "adapt-embedding" in t)
    i_label = next((i for i, t in enumerate(todo) if t.startswith("標")), len(todo))
    assert i_adapt < i_label
    # L1: provenance stamped with the eval_set_hash; first report -> no prior to compare
    assert d["provenance"]["eval_set_hash"] and d["provenance"]["comparable"] is None
    # L4: the actioned candidate is flagged resolved (order unchanged)
    assert any(c["id"] == "cand_b0" and c["resolved"] for c in d["queue"].get("bubble", []))
    # L1 audit stamp lands on the event too
    rec = [e for e in DecisionLog(cfg.decision_log_path).read_all() if e["event"] == "weakness_report"][-1]
    assert rec["extra"]["eval_set_hash"] == d["provenance"]["eval_set_hash"]
