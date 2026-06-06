"""DOGFOOD: run VIX's offline curation/audit loop on the REAL patHole_Dataset (665 images, Pascal-VOC
boxes) and VALIDATE the two R2 fixes (BOX audit hole, TRENDFIX) on real data.

Honest scope: there are no YOLO weights / GPU here, and VIX never trains anyway — so we use the VOC
ground-truth boxes as the detections and pixel-fallback embeddings (offline). That exercises the
curation/audit machinery (snapshot identity, export fingerprint, label-consistency, dedup, coverage)
and proves the box-level audit hole is closed on real boxes. It does NOT measure mAP (out of scope for
a no-retrain tool). Reproducible: `python docs/examples/dogfood_pathole.py`.
"""
from __future__ import annotations

import sys
import tempfile
import time

for _s in (sys.stdout, sys.stderr):  # the default Windows cp950 console can't encode 中文/⚠
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
import xml.etree.ElementTree as ET
from pathlib import Path

from vix import pipeline
from vix.adapters.memory import InMemoryAdapter
from vix.config import Config
from vix.embedding.simple import pixel_embedding
from vix.types import BBox, Detection, Tag

DS = Path(r"C:\code\claude\patHole_Dataset")


def _parse_voc(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    sz = root.find("size")
    W, H = float(sz.find("width").text), float(sz.find("height").text)
    out = []
    if W <= 0 or H <= 0:
        return out
    for o in root.findall("object"):
        name = (o.findtext("name") or "pothole").strip()
        b = o.find("bndbox")
        x1, y1, x2, y2 = (float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        out.append((name, ((x1 + x2) / 2) / W, ((y1 + y2) / 2) / H, abs(x2 - x1) / W, abs(y2 - y1) / H))
    return out


def main():
    cfg = Config(workspace=Path(tempfile.mkdtemp(prefix="vix_dogfood_")))
    cfg.ensure_dirs()
    cfg.embedding_backend = "pixel_fallback"
    ad = InMemoryAdapter()

    imgs = sorted((DS / "images").glob("*.png"))
    t0, n, nb = time.time(), 0, 0
    for p in imgs:
        xml = DS / "annotations" / (p.stem + ".xml")
        if not xml.exists():
            continue
        boxes = _parse_voc(xml)
        if not boxes:
            continue
        emb = pixel_embedding(str(p), size=8)
        ad.seed(p.stem, str(p), [Detection(nm, 1.0, BBox(cx, cy, w, h), embedding=emb)
                                 for (nm, cx, cy, w, h) in boxes], tags=[Tag.GOLDEN])
        n += 1
        nb += len(boxes)
    load_s = time.time() - t0
    classes = sorted({d.label for _h, _s, dets, _t in ad.samples() for d in dets})
    print("# DOGFOOD — VIX on patHole_Dataset (real data, offline)")
    print(f"loaded {n} images / {nb} GT boxes as golden in {load_s:.1f}s | embeddings=pixel_fallback | classes={classes}")

    # ---- BOX fix on REAL boxes ----
    h1 = pipeline._training_pool_hash(ad, cfg)
    fh, _s, fdets, _t = next(iter(ad.samples()))
    d0 = fdets[0]
    widened = Detection(d0.label, d0.confidence, BBox(d0.bbox.cx, d0.bbox.cy, min(0.99, d0.bbox.w + 0.05), d0.bbox.h),
                        embedding=d0.embedding)
    ad.set_detections(fh, [widened] + fdets[1:])
    h2 = pipeline._training_pool_hash(ad, cfg)
    r_edit = pipeline.export(ad, cfg, classes, Path(tempfile.mkdtemp()))
    ad.set_detections(fh, fdets)  # restore the original box
    h3 = pipeline._training_pool_hash(ad, cfg)
    r_orig = pipeline.export(ad, cfg, classes, Path(tempfile.mkdtemp()))
    print("\n[BOX] editing one real pothole box ->")
    print(f"  training-pool content_hash changes: {h1 != h2}   restore returns identical: {h1 == h3}")
    print(f"  export boxes_hash changes:          {r_edit['boxes_hash'] != r_orig['boxes_hash']}")
    print(f"  hashes: pool {h1[:12]}->{h2[:12]}->{h3[:12]} | export {r_orig['boxes_hash'][:12]} vs {r_edit['boxes_hash'][:12]}")

    # ---- what VIX flags on the real set (offline analytics) ----
    def _count(x):
        try:
            return len(x)
        except TypeError:
            return x
    t = time.time()
    flags = {}
    for name, fn in (("suspected_label_issues", lambda: pipeline.audit_labels(ad, cfg)),
                     ("near_duplicate_groups", lambda: pipeline.dedup(ad, cfg)),
                     ("coverage", lambda: pipeline.coverage(ad, cfg))):
        try:
            flags[name] = _count(fn())
        except Exception as e:  # noqa: BLE001
            flags[name] = f"(n/a: {type(e).__name__})"
    print(f"\n[FLAGS] {flags}  (analytics {time.time() - t:.1f}s)")

    # ---- TRENDFIX on a changed eval set ----
    from vix.core.decision_log import DecisionLog
    log = DecisionLog(cfg.decision_log_path)
    for esh, ap in (("evalA", 0.50), ("evalB", 0.61)):  # different eval_set_hash between cycles
        log.append("eval_ingest", decision="eval", extra={"eval_set_hash": esh, "mAP": ap, "per_class_ap": {"pothole": ap}})
    tr = pipeline.report_trend(cfg)
    print(f"\n[TRENDFIX] eval_set_changed={tr['eval_set_changed']} -> per-class Δ arrow withheld; note='{tr['note'][:40]}...'")
    print(f"\nworkspace: {cfg.workspace}")


if __name__ == "__main__":
    main()
