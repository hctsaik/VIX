"""自我檢查 vix_animals 的分群:
  1) 客觀:用 feat_umap 的 2D 座標 + ground_truth 算 silhouette 分數(>0 代表有分群)。
  2) 自畫:把 UMAP 點依類別上色畫成 PNG(docs/spec/img/animals_umap.png)。
  3) Playwright:連到正在跑的 App 截圖(格狀 + 嘗試開 Embeddings 面板)。
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import fiftyone as fo  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402
from sklearn.preprocessing import LabelEncoder  # noqa: E402

IMG = ROOT / "docs" / "spec" / "img"
IMG.mkdir(parents=True, exist_ok=True)
URL = "http://localhost:5151"
PALETTE = {
    "cat": (220, 50, 50), "dog": (50, 120, 220), "bird": (40, 170, 70),
    "horse": (160, 90, 200), "automobile": (230, 150, 30), "ship": (40, 190, 200),
}


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ds = fo.load_dataset("vix_animals")
    labels = ds.values("ground_truth.label")
    pts = np.array(ds.load_brain_results("feat_umap").points)
    print(f"dataset={len(ds)}  classes={dict(Counter(labels))}")

    # (1) 客觀分群指標
    y = LabelEncoder().fit_transform(labels)
    sil = silhouette_score(pts, y)
    print(f"silhouette score (2D UMAP, by class) = {sil:.3f}  (>0.1 已可見分群, >0.3 明顯)")

    # (2) 自畫 UMAP 散佈圖(依類別上色)
    W = H = 760
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    x, yv = pts[:, 0], pts[:, 1]
    xn = (x - x.min()) / (np.ptp(x) + 1e-9) * (W - 60) + 30
    yn = (yv - yv.min()) / (np.ptp(yv) + 1e-9) * (H - 90) + 30
    for i, lab in enumerate(labels):
        c = PALETTE.get(lab, (120, 120, 120))
        d.ellipse([xn[i] - 4, yn[i] - 4, xn[i] + 4, yn[i] + 4], fill=c)
    for j, (lab, c) in enumerate(PALETTE.items()):
        d.rectangle([30 + j * 125, H - 28, 44 + j * 125, H - 14], fill=c)
        d.text((48 + j * 125, H - 28), lab, fill=(0, 0, 0))
    img.save(IMG / "animals_umap.png")
    print(f"[OK] 自畫 UMAP -> {IMG / 'animals_umap.png'}")

    # (3) Playwright 連到正在跑的 App 截圖
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            br = p.chromium.launch()
            page = br.new_page(viewport={"width": 1600, "height": 1000})
            page.goto(URL, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(12000)  # App 有常駐 websocket,不能等 networkidle
            page.screenshot(path=str(IMG / "animals_app_grid.png"), full_page=True)
            print(f"[OK] App 格狀截圖 -> {IMG / 'animals_app_grid.png'}")
            # 嘗試開 Embeddings 面板
            try:
                page.get_by_text("Samples", exact=True).first.wait_for(timeout=5000)
                for sel in ['[title*="panel" i]', '[aria-label*="panel" i]', 'button:has-text("+")']:
                    loc = page.locator(sel)
                    if loc.count():
                        loc.first.click()
                        break
                page.wait_for_timeout(1500)
                emb = page.get_by_text("Embeddings", exact=False)
                if emb.count():
                    emb.first.click()
                    page.wait_for_timeout(9000)
                page.screenshot(path=str(IMG / "animals_app_embeddings.png"))
                print(f"[OK] App Embeddings 嘗試截圖 -> {IMG / 'animals_app_embeddings.png'}")
            except Exception as exc:  # noqa: BLE001
                print(f"[??] Embeddings 面板互動未完成: {exc}")
            br.close()
    except Exception as exc:  # noqa: BLE001
        print(f"[??] Playwright 截圖略過: {exc}")


if __name__ == "__main__":
    main()
