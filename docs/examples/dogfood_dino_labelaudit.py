"""Use REAL DINOv2 embeddings to revisit LABELING problems on the patHole data.

VIX's model-weakness numbers (mAP/FP/FN) need no embeddings — they're pure GT-vs-pred IoU. DINO is for
the OTHER half: relationships between labelled instances. Here we embed each labelled pothole CROP with
real DINOv2 (offline, via the SAFE torchhub cache) and flag instances whose appearance is an OUTLIER vs
the pothole cluster — i.e. "this box is labelled pothole but doesn't look like one" (a likely mislabel /
box-on-wrong-thing). That's the single-class form of what `vix audit-labels` does cross-class.

To prove it bites, we PLANT 3 obvious mislabels (a box on a non-pothole top strip) — DINO should rank
them as the top outliers. NOTE: pixel-fallback embeddings (the offline default when no DINO) could NOT do
this — the signal is the learned DINO representation. Saves the top suspect crops + an HTML contact sheet.

Run: python docs/examples/dogfood_dino_labelaudit.py
"""
from __future__ import annotations

import glob
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from vix.embedding.dinov2 import crop_detection  # noqa: E402
from vix.types import BBox  # noqa: E402

DS = Path(r"C:\code\claude\patHole_Dataset")
SAFE_HUB = r"C:\code\claude\SAFE\.cache\torchhub"
ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "docs" / "guide" / "dino_labelaudit"
N_IMAGES = 120  # CPU subset
K = 8


def _gt(stem):
    r = ET.parse(DS / "annotations" / (stem + ".xml")).getroot()
    z = r.find("size")
    W, H = float(z.find("width").text), float(z.find("height").text)
    out = []
    for o in r.findall("object"):
        b = o.find("bndbox")
        x1, y1, x2, y2 = (float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        out.append(BBox((x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H))
    return out


def main():
    torch.hub.set_dir(SAFE_HUB)
    t = time.time()
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", source="github",
                          verbose=False, trust_repo=True).eval()
    tf = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    print(f"loaded real DINOv2 vits14 (offline) in {time.time() - t:.1f}s")

    @torch.no_grad()
    def embed(pil):
        return dino(tf(pil.convert("RGB")).unsqueeze(0))[0].numpy()

    imgs = sorted(glob.glob(str(DS / "images" / "*.png")))[:N_IMAGES]
    recs = []  # (img_path, BBox, planted)
    for p in imgs:
        for bb in _gt(Path(p).stem):
            recs.append((p, bb, False))
    for p in imgs[:3]:  # PLANT 3 obvious mislabels: a box on the top strip (sky/road, not a pothole)
        recs.append((p, BBox(0.5, 0.06, 0.26, 0.10), True))

    t = time.time()
    X = np.stack([embed(crop_detection(Image.open(p).convert("RGB"), bb)) for p, bb, _ in recs])
    print(f"embedded {len(recs)} labelled crops in {time.time() - t:.1f}s (incl. 3 planted mislabels)")

    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    S = Xn @ Xn.T
    np.fill_diagonal(S, -np.inf)
    knn = np.sort(S, axis=1)[:, -K:]                # k nearest cosine sims
    outlier = 1.0 - knn.mean(axis=1)                # high = far from the pothole cluster = suspect label
    order = np.argsort(-outlier)

    planted = [i for i, (_p, _b, pl) in enumerate(recs) if pl]
    ranks = {i: int(np.where(order == i)[0][0]) for i in planted}
    print(f"\nDINO outlier ranking (of {len(recs)} crops):")
    for rank, i in enumerate(order[:15]):
        p, bb, pl = recs[i]
        print(f"  #{rank:<2} score={outlier[i]:.3f}  {Path(p).stem}  bbox=({bb.cx:.2f},{bb.cy:.2f},{bb.w:.2f},{bb.h:.2f}){'  <-- PLANTED MISLABEL' if pl else ''}")
    print(f"\nplanted mislabels' DINO ranks: {sorted(ranks.values())}  (0 = most-outlier; all in top-{max(ranks.values())+1} of {len(recs)})")

    OUT.mkdir(parents=True, exist_ok=True)
    cards = []
    for rank, i in enumerate(order[:12]):
        p, bb, pl = recs[i]
        crop = crop_detection(Image.open(p).convert("RGB"), bb)
        crop.thumbnail((180, 180))
        fn = f"{rank:02d}.png"
        crop.save(OUT / fn)
        cards.append(f"<div class='c'><img src='dino_labelaudit/{fn}'><div>#{rank} score {outlier[i]:.2f}"
                     f"{' · PLANTED' if pl else ''}<br><small>{Path(p).stem}</small></div></div>")
    (ROOT / "docs" / "guide" / "DINO_LABELAUDIT.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>DINO label audit</title>"
        "<style>body{font-family:system-ui,'Microsoft JhengHei';background:#0f1115;color:#e6e6e6;max-width:1000px;margin:0 auto;padding:24px}"
        ".c{display:inline-block;width:190px;margin:8px;vertical-align:top;font-size:12px;text-align:center}"
        "img{width:180px;height:auto;border:2px solid #2a3a50;border-radius:6px}.c:nth-child(-n+3) img{border-color:#d9544f}</style>"
        "<h1>用 DINOv2 embedding 回頭看 labeling 問題</h1>"
        "<p>每個「被標成 pothole」的框,用真 DINOv2 取嵌入,標出外觀最不像 pothole 群的(=疑似標錯/框到別的東西)。"
        "下面是 DINO 判定的前 12 名離群標註(分數越高越可疑);我們<b>故意種了 3 個明顯標錯</b>(畫在天空/路面上)驗證 DINO 抓得到。"
        "<br>PROXY:DINO 也是模型的判斷,需人工覆核、勿自動改標。pixel_fallback 嵌入做不到這件事 —— 這就是為什麼這半需要 DINO。</p>"
        + "".join(cards), encoding="utf-8")
    print(f"\nsaved top-12 suspect crops + docs/guide/DINO_LABELAUDIT.html")


if __name__ == "__main__":
    main()
