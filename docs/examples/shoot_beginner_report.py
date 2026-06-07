"""Playwright: screenshot the generated weakness report (offline file://) for the beginner docs.
Captures the full page + key sections (TL;DR health, per-class AP with Δ, confusion, confidently-wrong).

Run (Tier-2 .venv311 with playwright):  python docs/examples/shoot_beginner_report.py
Out: docs/guide/site/img/report_*.png
"""
from __future__ import annotations

from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent / "guide" / "site"
REPORT = (ROOT / "_artifacts" / "weakness_report.html").resolve()
IMG = ROOT / "img"
IMG.mkdir(parents=True, exist_ok=True)


def _shot(page, selector, out):
    el = page.query_selector(selector)
    if el is None:
        print("  (skip, not found:", selector, ")")
        return
    el.screenshot(path=str(IMG / out))
    print("  wrote", out)


def main():
    if not REPORT.exists():
        raise SystemExit(f"missing {REPORT} — run gen_beginner_report.py first")
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1120, "height": 900}, device_scale_factor=2)
        pg.goto(REPORT.as_uri(), wait_until="domcontentloaded")
        pg.wait_for_timeout(300)
        pg.screenshot(path=str(IMG / "report_full.png"), full_page=True)
        print("  wrote report_full.png")
        _shot(pg, "#tldr", "report_tldr.png")
        _shot(pg, "#per-class", "report_perclass.png")
        _shot(pg, "#consistency", "report_consistency.png")
        _shot(pg, "#confident-wrong", "report_confident_wrong.png")
        b.close()
    print("done ->", IMG)


if __name__ == "__main__":
    main()
