"""啟 App(含 Embeddings 面板),擷取與 raw_app.png 完全對齊的截圖,
並抓出關鍵 UI 元件的像素邊界框 -> docs/guide/img/boxes.json,供後續 PIL 畫紅框。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import fiftyone as fo  # noqa: E402

IMG = ROOT / "docs" / "guide" / "img"
IMG.mkdir(parents=True, exist_ok=True)
PORT = 5155
VIEWPORT = {"width": 1600, "height": 980}

# label -> 多個候選 selector(取第一個抓得到的)
TARGETS = {
    "dataset_name": ['[data-cy="selector-dataset"]', 'text=vix_animals'],
    "new_panel": ['[data-cy="new-panel-btn"]', '[title="New panel"]'],
    "umap_tab": ['[data-cy="selector-feat_umap"]', 'text=feat_umap'],
    "colorby": ['[data-cy="selector-ground_truth"]'],
    "grid_toolbar": ['[data-cy="action-bar"]', '[data-cy="fo-grid-actions"]'],
    "view_bar": ['[data-cy="view-bar"]', 'text=add stage'],
}


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ds = fo.load_dataset("vix_animals")
    try:
        samples = fo.Panel(type="Samples", pinned=True)
        emb = fo.Panel(type="Embeddings", state=dict(brainResult="feat_umap", colorByField="ground_truth"))
        spaces = fo.Space(children=[samples, emb], orientation="horizontal")
    except Exception:
        spaces = None

    session = fo.launch_app(ds, spaces=spaces, remote=True, port=PORT)
    boxes: dict = {}
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            br = p.chromium.launch()
            page = br.new_page(viewport=VIEWPORT)
            page.goto(f"http://localhost:{PORT}", wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(15000)
            # 關掉 enterprise popover
            for txt in ("Dismiss", "No thanks", "Close"):
                try:
                    b = page.get_by_role("button", name=txt)
                    if b.count():
                        b.first.click(timeout=2000)
                        break
                except Exception:
                    pass
            page.wait_for_timeout(1500)
            page.screenshot(path=str(IMG / "raw_app.png"), full_page=True)

            for label, sels in TARGETS.items():
                for sel in sels:
                    try:
                        loc = page.locator(sel).first
                        if loc.count():
                            bb = loc.bounding_box()
                            if bb:
                                boxes[label] = {k: round(v) for k, v in bb.items()}
                                break
                    except Exception:
                        continue
            br.close()
    finally:
        session.close()

    (IMG / "boxes.json").write_text(json.dumps(boxes, indent=2, ensure_ascii=False), encoding="utf-8")
    print("boxes:", json.dumps(boxes, ensure_ascii=False))


if __name__ == "__main__":
    main()
