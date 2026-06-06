"""DOGFOOD step 1/2: train a quick YOLOv8n on patHole_Dataset so VIX has a REAL model to evaluate.

VIX itself never trains — this is the EXTERNAL trainer a real user runs to produce the predictions VIX
ingests. Deterministic 80/20 split, single class 'pothole', small/short (CPU) — the goal is a real (and
deliberately imperfect) model whose FP/FN VIX's eval-ingest + weakness-report can surface, NOT SOTA mAP.
Writes the YOLO dataset + run under _dogfood_yolo/ (gitignored). Run: python docs/examples/dogfood_train_yolo.py
"""
from __future__ import annotations

import random
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

DS = Path(r"C:\code\claude\patHole_Dataset")
WORK = Path(__file__).resolve().parent.parent.parent / "_dogfood_yolo"


def _parse(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    sz = root.find("size")
    W, H = float(sz.find("width").text), float(sz.find("height").text)
    out = []
    if W > 0 and H > 0:
        for o in root.findall("object"):
            b = o.find("bndbox")
            x1, y1, x2, y2 = (float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
            out.append((((x1 + x2) / 2) / W, ((y1 + y2) / 2) / H, abs(x2 - x1) / W, abs(y2 - y1) / H))
    return out


def prep():
    pairs = [(p, DS / "annotations" / (p.stem + ".xml")) for p in sorted((DS / "images").glob("*.png"))]
    pairs = [(p, x) for p, x in pairs if x.exists() and _parse(x)]
    random.Random(0).shuffle(pairs)
    nval = max(1, int(len(pairs) * 0.2))
    splits = {"val": pairs[:nval], "train": pairs[nval:]}
    if WORK.exists():
        shutil.rmtree(WORK)
    for split, items in splits.items():
        (WORK / "images" / split).mkdir(parents=True, exist_ok=True)
        (WORK / "labels" / split).mkdir(parents=True, exist_ok=True)
        for p, x in items:
            shutil.copy(p, WORK / "images" / split / p.name)
            (WORK / "labels" / split / (p.stem + ".txt")).write_text(
                "\n".join(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for (cx, cy, w, h) in _parse(x)), encoding="utf-8")
    (WORK / "data.yaml").write_text(
        yaml.safe_dump({"path": str(WORK), "train": "images/train", "val": "images/val", "names": ["pothole"]}),
        encoding="utf-8")
    return len(splits["train"]), len(splits["val"])


def main():
    nt, nv = prep()
    print(f"[prep] train={nt} val={nv} -> {WORK}")
    from ultralytics import YOLO
    try:
        model = YOLO("yolov8n.pt")  # pretrained (auto-download)
    except Exception as e:  # noqa: BLE001
        print("pretrained unavailable, training from scratch:", str(e)[:80])
        model = YOLO("yolov8n.yaml")
    model.train(data=str(WORK / "data.yaml"), epochs=12, imgsz=320, batch=16, device="cpu",
                project=str(WORK / "runs"), name="pothole", exist_ok=True, verbose=False, plots=False)
    best = WORK / "runs" / "pothole" / "weights" / "best.pt"
    print(f"[done] BEST={best} exists={best.exists()}")


if __name__ == "__main__":
    main()
