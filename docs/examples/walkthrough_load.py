"""Playwright walkthrough: FROM the App's initial screen, THROUGH the VIX load-dataset button,
to the FIRST dataset loaded (with the user's model run) — capturing a screenshot at each step.

Drives the live FiftyOne App at :5151:
  01 initial screen  -> 02 click the folder (load) button -> 03 form open
  04 form filled (folder + model .pt) -> 05 Execute clicked -> 06 first dataset loaded + report

Run (Tier-2 .venv311, App already serving): python docs/examples/walkthrough_load.py
Out: docs/guide/walkthrough_load/NN_*.png
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

FOLDER = r"C:\code\claude\patHole_Dataset"
WEIGHTS = r"C:\code\claude\VIX\_dogfood_yolo\runs\pothole\weights\best.pt"
EXPECT_DATASET = "patHole_Dataset"   # operator derives the dataset name from the folder basename
ROOT = Path(__file__).resolve().parent.parent.parent
DONE = ROOT / ".venv311" / "appws" / "weakness_report.html"  # operator writes this LAST = real completion signal
OUT = Path(__file__).resolve().parent.parent / "guide" / "walkthrough_load"
OUT.mkdir(parents=True, exist_ok=True)


def shot(pg, n, label):
    p = OUT / f"{n:02d}_{label}.png"
    pg.screenshot(path=str(p))
    print(f"  [{n:02d}] {label} -> {p.name}", flush=True)


def main():
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1500, "height": 950}, device_scale_factor=2)
        pg.goto("http://localhost:5151", wait_until="domcontentloaded")
        pg.wait_for_timeout(12000)

        # dismiss the FiftyOne Enterprise popup (covers the right toolbar)
        d = pg.query_selector("text=Dismiss")
        if d:
            try:
                d.click(); pg.wait_for_timeout(500)
            except Exception:
                pass
        shot(pg, 1, "initial-screen")

        # 2: click the VIX load button (folder icon) in the grid toolbar
        fimg = pg.query_selector("img[src*='folder.svg']")
        if fimg is None:
            print("ERROR: load (folder) button not found in toolbar", flush=True)
            b.close(); sys.exit(1)
        fimg.click()
        pg.wait_for_timeout(2500)
        shot(pg, 2, "load-form-open")

        # 3: fill the two operator fields. The modal's inputs are portal-appended LAST in the DOM,
        # so the last two text inputs are (folder, model) — and being on top, fill won't hit background.
        execute = pg.get_by_role("button", name="Execute")
        execute.wait_for(timeout=15000)
        boxes = pg.get_by_role("textbox")  # MUI inputs expose as textbox role (no type=text attr)
        n = boxes.count()
        if n < 2:
            print(f"ERROR: operator form inputs not found (textboxes={n})", flush=True); b.close(); sys.exit(1)
        boxes.nth(n - 2).fill(FOLDER)      # 影像資料夾路徑
        boxes.nth(n - 1).fill(WEIGHTS)     # (選填)模型 .pt
        pg.wait_for_timeout(800)
        shot(pg, 3, "form-filled")

        # 4: Execute -> kicks off import + the model run (~90s on 665 imgs)
        if DONE.exists():
            DONE.unlink()  # so we wait for the FRESH report this run writes
        execute.click()
        pg.wait_for_timeout(2500)
        shot(pg, 4, "executing")

        # 5: wait for the REAL completion signal — the operator writes weakness_report.html LAST
        ok = False
        t0 = time.time()
        while time.time() - t0 < 300:
            if DONE.exists():
                ok = True; break
            pg.wait_for_timeout(3000)
        elapsed = time.time() - t0
        print(f"  diagnose done={ok} after {elapsed:.0f}s; url={pg.url}", flush=True)
        pg.wait_for_timeout(9000)  # let the App switch dataset + paint
        shot(pg, 5, "first-data-loaded")          # the result summary dialog (now clean: no error line)
        for t in ["Close", "關閉", "Done"]:        # dismiss the dialog -> clean grid with boxes
            el = pg.query_selector(f"text={t}")
            if el:
                try:
                    el.click(); pg.wait_for_timeout(900); break
                except Exception:
                    pass
        pg.wait_for_timeout(2500)
        shot(pg, 6, "loaded-clean")
        b.close()
        print("WALKTHROUGH_OK" if ok else "WALKTHROUGH_INCOMPLETE", flush=True)


if __name__ == "__main__":
    main()
