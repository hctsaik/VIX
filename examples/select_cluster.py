"""重現「在 Embeddings 面板框選一群 -> 反查這群是什麼影像」:
  1) 在 UMAP 2D 座標上用 KMeans 找出一個自然群集(等同人工套索框一團)。
  2) 挑「純度最高」的那群(最像一個清楚的聚落),取出它的 sample ids。
  3) 用 ds.select(ids) 建立只含這群的 View -> 等同 App 裡的「Only show selected」。
  4) 另開一個 App(port 5153,不干擾使用者的 5151)截圖格狀。
  5) 同時印出這群的 ground_truth 類別組成(回答「這群是什麼」)。
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import fiftyone as fo  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402

IMG = ROOT / "docs" / "spec" / "img"
PORT = 5153


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ds = fo.load_dataset("vix_animals")
    ids = np.array(ds.values("id"))
    labels = np.array(ds.values("ground_truth.label"))
    pts = np.array(ds.load_brain_results("feat_umap").points)

    # (1) KMeans 找 6 群(等同在散佈圖上各框一團)
    km = KMeans(n_clusters=6, n_init=10, random_state=51).fit(pts)

    # (2) 挑純度最高的一群
    best = None  # (cluster_id, dominant_label, mask, purity, counter)
    for c in range(6):
        mask = km.labels_ == c
        cnt = Counter(labels[mask])
        dom, domn = cnt.most_common(1)[0]
        purity = domn / mask.sum()
        if best is None or purity > best[3]:
            best = (c, dom, mask, purity, cnt)

    _, dom, mask, purity, cnt = best
    sel_ids = ids[mask].tolist()
    print(f"框選到的群:{mask.sum()} 張  主類別={dom}  純度={purity:.0%}")
    print(f"類別組成:{dict(cnt)}")

    # (3) 只含這群的 View(= App 的 Only show selected)
    view = ds.select(sel_ids)

    # (4) 另開 App 截圖
    session = fo.launch_app(view=view, remote=True, port=PORT)
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            br = p.chromium.launch()
            page = br.new_page(viewport={"width": 1400, "height": 900})
            page.goto(f"http://localhost:{PORT}", wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(12000)
            out = IMG / "animals_selected_cluster.png"
            page.screenshot(path=str(out), full_page=True)
            print(f"[OK] 只顯示選取的格狀截圖 -> {out}")
            br.close()
    finally:
        session.close()


if __name__ == "__main__":
    main()
