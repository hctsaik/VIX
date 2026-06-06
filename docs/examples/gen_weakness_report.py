"""Generate a sample weakness_report.html with synthetic defect data that exercises EVERY section:
per-class AP / confusion / FP-FN typing / loc_gap, confidently-wrong (hardneg), the per-weak-class
label queue, the GT x embedding consistency attribution (incl. a representation_fixable rescue after
adapt-embedding), and the queue hit-rate. Reproducible: `python docs/examples/gen_weakness_report.py`.

Scenario: bubble / dripping / scratch defects. bubble & dripping are separable ONLY in a low-variance
dim swamped by noise (so frozen DINO can't split them, but the LDA projection rescues them); scratch
is cleanly separable. The model is weak on bubble (confuses it as dripping, misses some) and throws a
few confident background false alarms.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vix import pipeline
from vix.adapters.memory import InMemoryAdapter
from vix.config import Config
from vix.core.decision_log import DecisionLog
from vix.types import BBox, Detection, Tag

ROOT = Path(__file__).resolve().parent
WS = ROOT / "_ws"
D = 8


def emb(cls, n, seed):
    r = np.random.RandomState(seed)
    X = 2.0 * r.randn(n, D)                       # heavy noise: swamps the bubble/dripping signal frozen
    if cls == "bubble":
        X[:, 0] = +0.6 + 0.1 * r.randn(n)         # bubble vs dripping differ only in dim0 (low variance)
    elif cls == "dripping":
        X[:, 0] = -0.6 + 0.1 * r.randn(n)
    elif cls == "scratch":
        X[:, 1] = +4.0 + 0.1 * r.randn(n)         # scratch cleanly separable (high-variance dim)
    return X


def det(label, vec, conf=0.9, bbox=(0.5, 0.5, 0.2, 0.2)):
    return Detection(label, conf, BBox(*bbox), embedding=np.asarray(vec, float))


def main():
    cfg = Config(workspace=WS)
    cfg.ensure_dirs()
    ad = InMemoryAdapter()
    box = [0.5, 0.5, 0.4, 0.4]

    # 1) golden (human-confirmed true instances) -> consistency + adapt-embedding
    for cls, seed in (("bubble", 1), ("dripping", 2), ("scratch", 3)):
        for i, v in enumerate(emb(cls, 25, seed)):
            ad.seed(f"g_{cls}_{i}", f"{cls}.png", [det(cls, v)], tags=[Tag.GOLDEN])
    # 2) unlabeled candidates near bubble failures -> the per-weak-class label queue
    for i, v in enumerate(emb("bubble", 6, 11)):
        ad.seed(f"cand_b{i}", "c.png", [det("bubble", v, conf=0.2)], tags=[])
    for i, v in enumerate(emb("scratch", 4, 12)):
        ad.seed(f"cand_s{i}", "c.png", [det("scratch", v, conf=0.2)], tags=[])

    # 3) eval set (GT + model predictions). Eval images are tagged EVAL (excluded as candidates) but
    #    carry the defect-region embedding so error-mine can match FN boxes.
    images = []

    def add(h, gt, pred, vec):
        ad.seed(h, f"{h}.png", [det(gt or "bubble", vec, conf=0.5)], tags=[Tag.EVAL])
        images.append({"vix_hash": h, "gt": ([{"label": gt, "bbox": box}] if gt else []), "pred": pred})

    bi = 0
    for _ in range(8):   # bubble confused as dripping (confusion bubble->dripping)
        add(f"e_b{bi}", "bubble", [{"label": "dripping", "bbox": box, "conf": 0.8}], emb("bubble", 1, 100 + bi)[0]); bi += 1
    for _ in range(6):   # bubble correct
        add(f"e_b{bi}", "bubble", [{"label": "bubble", "bbox": box, "conf": 0.85}], emb("bubble", 1, 100 + bi)[0]); bi += 1
    for _ in range(6):   # bubble missed
        add(f"e_b{bi}", "bubble", [], emb("bubble", 1, 100 + bi)[0]); bi += 1
    for k in range(12):  # dripping mostly correct
        add(f"e_d{k}", "dripping", [{"label": "dripping", "bbox": box, "conf": 0.85}], emb("dripping", 1, 200 + k)[0])
    for k in range(12):  # scratch strong
        add(f"e_s{k}", "scratch", [{"label": "scratch", "bbox": box, "conf": 0.9}], emb("scratch", 1, 300 + k)[0])
    for k in range(4):   # confident background false alarms (bubble pred, no GT) -> confidently-wrong
        add(f"e_fp{k}", None, [{"label": "bubble", "bbox": box, "conf": 0.92}], emb("bubble", 1, 400 + k)[0])

    (WS / "eval.jsonl").write_text("\n".join(json.dumps(x) for x in images), encoding="utf-8")
    ev = pipeline.eval_ingest(ad, cfg, str(WS / "eval.jsonl"))

    # 4) domain-adapted embedding: rescue bubble/dripping, gate-validated enable
    ar = pipeline.adapt_embedding(ad, cfg, save=True, enable=True)

    # 5) simulate a prior cycle's queue + human resolutions -> queue hit-rate has data
    pipeline._log_queue(cfg, "hardneg", ["e_fp0", "e_fp1", "e_fp2", "e_b0", "e_b1"], "wrong")
    dl = DecisionLog(cfg.decision_log_path)
    for h, outcome in (("e_fp0", "false_alarm"), ("e_fp1", "false_alarm"), ("e_fp2", "false_alarm"),
                       ("e_b0", "false_alarm"), ("e_b1", "confirmed")):
        dl.append("review", vix_hash=h, decision=outcome)

    r = pipeline.weakness_report(ad, cfg)
    out = ROOT / "weakness_report.html"
    out.write_text(Path(r["html"]).read_text(encoding="utf-8"), encoding="utf-8")
    print(f"mAP={ev['mAP']} loc_gap={ev['loc_gap']} | adapt gate={'GO' if ar['gate']['go'] else 'NO-GO'} "
          f"rescued={ar['n_rescued']} enabled={ar['enabled']}")
    print(f"consistency findings: {[(f['pair'], f['verdict'], f.get('representation_fixable')) for f in r['data']['consistency']]}")
    print(f"hit_rate: {[(q['queue'], q['precision'], q['trend']) for q in r['data']['hit_rate']]}")
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
