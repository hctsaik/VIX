"""Playwright verification of the beginner docs site: every page loads, every <img> resolves
(naturalWidth>0), and every internal nav/link target exists on disk. Screenshots index + diagnose
for a visual sanity check. Run (Tier-2 .venv311): python docs/examples/verify_site.py
"""
from __future__ import annotations

from pathlib import Path
from urllib.parse import urldefrag

from playwright.sync_api import sync_playwright

SITE = Path(__file__).resolve().parent.parent / "guide" / "site"
PAGES = sorted(p.name for p in SITE.glob("*.html"))


def main():
    problems = []
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1280, "height": 1000}, device_scale_factor=1)
        for name in PAGES:
            pg.goto((SITE / name).as_uri(), wait_until="domcontentloaded")
            pg.wait_for_timeout(150)
            broken = pg.eval_on_selector_all(
                "img", "els => els.filter(e => !e.complete || e.naturalWidth === 0).map(e => e.getAttribute('src'))")
            for src in broken:
                problems.append(f"{name}: broken img {src}")
            hrefs = pg.eval_on_selector_all(
                "a", "els => els.map(e => e.getAttribute('href'))")
            for href in hrefs:
                if not href or href.startswith(("http", "#", "mailto:")):
                    continue
                target = urldefrag(href)[0]
                if not target:
                    continue
                if not (SITE / target).resolve().exists():
                    problems.append(f"{name}: dead link {href}")
        pg.goto((SITE / "index.html").as_uri(), wait_until="domcontentloaded")
        pg.wait_for_timeout(200)
        pg.screenshot(path=str(SITE / "img" / "_verify_index.png"), full_page=True)
        pg.goto((SITE / "diagnose.html").as_uri(), wait_until="domcontentloaded")
        pg.wait_for_timeout(200)
        pg.screenshot(path=str(SITE / "img" / "_verify_diagnose.png"), full_page=True)
        b.close()

    print(f"checked {len(PAGES)} pages:", ", ".join(PAGES))
    if problems:
        print(f"\n{len(problems)} PROBLEM(S):")
        for x in problems:
            print("  -", x)
    else:
        print("\nOK — all pages load, all images resolve, all internal links exist.")


if __name__ == "__main__":
    main()
