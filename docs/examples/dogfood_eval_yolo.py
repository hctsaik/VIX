"""DOGFOOD step 2/2: run the trained YOLO on the val split, feed its predictions + GT into VIX's
eval-ingest + weakness-report, and print the REAL model-weakness signals (per-class AP, mAP, typed
FP/FN, confidently-wrong). This is the half the earlier dogfood couldn't reach (it had no model).

Run after dogfood_train_yolo.py: python docs/examples/dogfood_eval_yolo.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from vix import pipeline  # noqa: E402
from vix.adapters.memory import InMemoryAdapter  # noqa: E402
from vix.config import Config  # noqa: E402
from vix.embedding.simple import pixel_embedding  # noqa: E402
from vix.types import BBox, Detection, Tag  # noqa: E402

DS = Path(r"C:\code\claude\patHole_Dataset")
WORK = Path(__file__).resolve().parent.parent.parent / "_dogfood_yolo"
BEST = WORK / "runs" / "pothole" / "weights" / "best.pt"


def _gt(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    sz = root.find("size")
    W, H = float(sz.find("width").text), float(sz.find("height").text)
    out = []
    if W > 0 and H > 0:
        for o in root.findall("object"):
            b = o.find("bndbox")
            x1, y1, x2, y2 = (float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
            out.append([((x1 + x2) / 2) / W, ((y1 + y2) / 2) / H, abs(x2 - x1) / W, abs(y2 - y1) / H])
    return out


def main():
    if not BEST.exists():
        print(f"no trained model at {BEST} — run dogfood_train_yolo.py first")
        return
    from ultralytics import YOLO
    model = YOLO(str(BEST))
    val_imgs = sorted((WORK / "images" / "val").glob("*.png"))

    cfg = Config(workspace=Path(tempfile.mkdtemp(prefix="vix_dogfood_eval_")))
    cfg.ensure_dirs()
    cfg.embedding_backend = "pixel_fallback"
    ad = InMemoryAdapter()
    images = []
    for p in val_imgs:
        gt = _gt(DS / "annotations" / (p.stem + ".xml"))
        r = model.predict(str(p), imgsz=320, conf=0.05, verbose=False)[0]  # low conf -> capture FPs too
        preds = [{"label": "pothole", "bbox": [round(v, 6) for v in r.boxes.xywhn[i].tolist()],
                  "conf": round(float(r.boxes.conf[i]), 4)} for i in range(len(r.boxes))]
        images.append({"vix_hash": p.stem,
                       "gt": [{"label": "pothole", "bbox": [round(v, 6) for v in g]} for g in gt],
                       "pred": preds})
        # seed the val sample with its YOLO predictions (EVAL-tagged) so hardneg can surface confident FPs
        emb = pixel_embedding(str(p), size=8)
        dets = [Detection("pothole", d["conf"], BBox(*d["bbox"]), embedding=emb) for d in preds] or \
               [Detection("pothole", 0.0, BBox(0.5, 0.5, 0.01, 0.01), embedding=emb)]
        ad.seed(p.stem, str(p), dets, tags=[Tag.EVAL])

    jl = cfg.workspace / "eval.jsonl"
    jl.write_text("\n".join(json.dumps(x) for x in images), encoding="utf-8")
    ev = pipeline.eval_ingest(ad, cfg, str(jl))

    n_pred = sum(len(x["pred"]) for x in images)
    n_gt = sum(len(x["gt"]) for x in images)
    print("# DOGFOOD step 2 — REAL trained YOLOv8n evaluated through VIX")
    print(f"val images={len(images)}  GT boxes={n_gt}  YOLO preds(conf>=0.05)={n_pred}")
    print(f"\n[eval-ingest]  mAP@0.5 = {ev.get('mAP')}  loc_gap = {ev.get('loc_gap')}  map_by_iou = {ev.get('map_by_iou')}")
    print(f"  per-class AP: {ev.get('per_class_ap')}   n_gt: {ev.get('n_gt')}")
    fp_types = {}
    for boxes in (ev.get("fp_detail") or {}).values():
        for b in boxes:
            fp_types[b["type"]] = fp_types.get(b["type"], 0) + 1
    fn_types = {}
    for boxes in (ev.get("fn_detail") or {}).values():
        for b in boxes:
            fn_types[b["type"]] = fn_types.get(b["type"], 0) + 1
    print(f"  FP types: {fp_types}   FN types: {fn_types}")

    wr = pipeline.weakness_report(ad, cfg)["data"]
    print(f"\n[weakness-report]  health={wr['summary']['health']}  weakest={wr['summary']['weakest']}")
    cw = wr.get("confident_wrong") or []
    print(f"  confidently-wrong (GT-confirmed high-conf FPs): {len(cw)}")
    for row in cw[:5]:
        print(f"    {row['id']}  conf={row['conf']}  type={row.get('fp_type')}")
    print(f"\n  -> VIX consumed a REAL trained model's eval and surfaced its weaknesses (mAP, typed FP/FN, "
          f"confident errors). report: {cfg.workspace / 'weakness_report.md'}")


if __name__ == "__main__":
    main()
