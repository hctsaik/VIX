"""為逐步教學擷取原始截圖:
  - 用 FiftyOne spaces API 直接把 Embeddings 面板開好(brain key=feat_umap、color by=ground_truth),
    避開手動點 '+' 面板被 popover 擋住的問題。
  - 擷取:(a) 純格狀 (b) 已開 Embeddings 面板的雙欄畫面。
存到 docs/guide/img/raw_*.png,稍後再用 PIL 加上紅圈/箭頭/步驟編號。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import fiftyone as fo  # noqa: E402

IMG = ROOT / "docs" / "guide" / "img"
IMG.mkdir(parents=True, exist_ok=True)
PORT = 5154


def _dismiss_popover(page) -> None:
    for txt in ("Dismiss", "No thanks", "Close"):
        try:
            loc = page.get_by_role("button", name=txt)
            if loc.count():
                loc.first.click(timeout=2000)
                page.wait_for_timeout(500)
                return
        except Exception:  # noqa: BLE001
            pass


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ds = fo.load_dataset("vix_animals")

    # 直接把 Embeddings 面板放進 workspace
    try:
        samples = fo.Panel(type="Samples", pinned=True)
        emb = fo.Panel(
            type="Embeddings",
            state=dict(brainResult="feat_umap", colorByField="ground_truth"),
        )
        spaces = fo.Space(children=[samples, emb], orientation="horizontal")
    except Exception as exc:  # noqa: BLE001
        print(f"[??] spaces API 不可用,改用預設版面: {exc}")
        spaces = None

    session = fo.launch_app(ds, spaces=spaces, remote=True, port=PORT)
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            br = p.chromium.launch()
            page = br.new_page(viewport={"width": 1600, "height": 980})
            page.goto(f"http://localhost:{PORT}", wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(15000)  # 等 UMAP 點雲算好渲染
            _dismiss_popover(page)
            page.wait_for_timeout(1500)
            out = IMG / "raw_embeddings.png"
            page.screenshot(path=str(out), full_page=True)
            print(f"[OK] Embeddings 面板畫面 -> {out}")
            br.close()
    finally:
        session.close()


if __name__ == "__main__":
    main()
