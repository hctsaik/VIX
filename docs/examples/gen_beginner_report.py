"""Generate a realistic MULTI-CLASS weakness report through the REAL VIX renderer, for the
beginner docs site. Deterministic + offline (no GPU/model/FiftyOne): we craft a representative
eval (two COMPARABLE cycles on a frozen GT set) and run the actual pipeline.eval_ingest +
pipeline.weakness_report(reference_unverified=True) — so the screenshots show the true output:
per-class AP weakest-first with a before/after Δ column, confusion, typed FP/FN, confidently-wrong,
and the "imported labels are unverified" honesty banner.

Run:  python docs/examples/gen_beginner_report.py
Out:  docs/guide/site/_artifacts/weakness_report.html  (+ .md)
"""
from __future__ import annotations

import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from vix import pipeline  # noqa: E402
from vix.adapters.memory import InMemoryAdapter  # noqa: E402
from vix.config import Config  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "guide" / "site" / "_artifacts"

# A frozen 5-class held-out eval set. Each entry: (class, n_gt). traffic_light is deliberately tiny
# (<min_support=20) to demonstrate the low-support "n少不穩" Δ hedge.
CLASSES = {"car": 60, "person": 50, "bicycle": 30, "motorcycle": 28, "traffic_light": 6}


def _box(i, j=0):
    # deterministic, well-separated boxes within an image
    return [round(0.15 + 0.12 * (i % 6), 4), round(0.15 + 0.12 * (j % 6), 4), 0.1, 0.1]


def build_eval(true_positives: dict, confuse=("bicycle", "motorcycle"), n_confuse=6, n_bg_fp=4):
    """Build [{vix_hash, gt, pred}] with controlled per-class recall + a confusion pair + background FPs.
    GT is IDENTICAL across cycles (only `true_positives` / preds change) so eval_set_hash stays stable."""
    images, k = [], 0
    ci, cj = confuse
    for cls, n_gt in CLASSES.items():
        tp = true_positives.get(cls, 0)
        for i in range(n_gt):
            h = f"img_{cls}_{i}"
            gt = [{"label": cls, "bbox": _box(i)}]
            pred = []
            if i < tp:  # a true positive: pred matches the GT box, same class, high conf
                pred = [{"label": cls, "bbox": _box(i), "conf": 0.92}]
            elif cls == ci and i < tp + n_confuse:  # confusion: predict cj over a ci GT box
                pred = [{"label": cj, "bbox": _box(i), "conf": 0.55}]
            images.append({"vix_hash": h, "gt": gt, "pred": pred})
            k += 1
    # background false positives (no GT here) — high confidence -> "confidently wrong"
    for i in range(n_bg_fp):
        images.append({"vix_hash": f"img_bg_{i}", "gt": [],
                       "pred": [{"label": "car", "bbox": _box(i, 3), "conf": round(0.97 - 0.02 * i, 3)}]})
    return images


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = Config(workspace=OUT / "_ws")
    cfg.ensure_dirs()
    ad = InMemoryAdapter()

    # Cycle 1 (a weaker model) — logged so cycle 2 has a comparable baseline on the SAME eval set.
    weak = {"car": 50, "person": 38, "bicycle": 9, "motorcycle": 18, "traffic_light": 2}
    pipeline.eval_ingest(ad, cfg, build_eval(weak))
    pipeline.weakness_report(ad, cfg, reference_unverified=True, out_path=OUT / "weakness_report.md")

    # Cycle 2 (after the engineer fixed labels + retrained externally) — SAME GT, better recall on the
    # weak classes -> per-class Δ appears; traffic_light swings on tiny support -> rendered "n少不穩".
    better = {"car": 54, "person": 46, "bicycle": 22, "motorcycle": 20, "traffic_light": 5}
    pipeline.eval_ingest(ad, cfg, build_eval(better))
    res = pipeline.weakness_report(ad, cfg, reference_unverified=True, out_path=OUT / "weakness_report.md")

    d = res["data"]
    print("health:", d["summary"]["health"], "| weakest:", d["summary"]["weakest"])
    print("per-class (weakest first):", [(r["cls"], r["ap"], r.get("delta_ap")) for r in d["per_class"]])
    print("confusion:", d["confusion"][:5])
    print("confident_wrong:", len(d["confident_wrong"]))
    print("report ->", res["html"])


if __name__ == "__main__":
    main()
