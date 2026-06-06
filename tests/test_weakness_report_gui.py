"""Playwright GUI test for the weakness-report's browsable HTML surface.

The consistency-attribution layer's human-facing artifact is an HTML report. This test renders a
real report (end-to-end through the pipeline: golden embeddings + eval confusion -> a `taxonomy`
verdict), opens it in a headless Chromium, and asserts the consistency table + the verdict actually
render in a browser. Skips cleanly if Playwright's browser isn't installed (Tier-2 dependency).
"""

import json

import numpy as np
import pytest

pytest.importorskip("playwright.sync_api")
from playwright.sync_api import sync_playwright  # noqa: E402

from vix import pipeline  # noqa: E402
from vix.adapters.memory import InMemoryAdapter  # noqa: E402
from vix.config import Config  # noqa: E402
from vix.types import BBox, Detection, Tag  # noqa: E402


def _cluster(center, n, jit, seed):
    rng = np.random.RandomState(seed)
    return np.asarray(center, float) + jit * rng.randn(n, len(center))


def _det(label, emb):
    return Detection(label, 0.9, BBox(0.5, 0.5, 0.2, 0.2), embedding=np.asarray(emb, float))


def _build_report(tmp_path):
    """Real pipeline render: overlapping golden a/b + matching eval confusion -> taxonomy verdict."""
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    ad = InMemoryAdapter()
    for i, v in enumerate(_cluster([1, 0, 0, 0], 25, 0.6, 1)):
        ad.seed(f"a{i}", "a.png", [_det("a", v)], tags=[Tag.GOLDEN])
    for i, v in enumerate(_cluster([1, 0, 0, 0], 25, 0.6, 2)):
        ad.seed(f"b{i}", "b.png", [_det("b", v)], tags=[Tag.GOLDEN])
    box = [0.5, 0.5, 0.4, 0.4]
    imgs = [{"vix_hash": f"e{i}", "gt": [{"label": "a", "bbox": box}],
             "pred": ([{"label": "b", "bbox": box, "conf": 0.9}] if i < 10 else [])} for i in range(20)]
    (tmp_path / "res.jsonl").write_text("\n".join(json.dumps(x) for x in imgs), encoding="utf-8")
    pipeline.eval_ingest(ad, cfg, str(tmp_path / "res.jsonl"))
    wr = pipeline.weakness_report(ad, cfg)
    return wr["html"]


def test_weakness_report_html_renders_consistency_in_browser(tmp_path):
    html_path = _build_report(tmp_path)
    with sync_playwright() as pw:
        try:
            browser = pw.chromium.launch()
        except Exception as e:  # browser binary not installed -> Tier-2 skip, not a failure
            pytest.skip(f"chromium unavailable (run: playwright install chromium): {e}")
        try:
            page = browser.new_page()
            page.goto(f"file://{html_path}", wait_until="domcontentloaded")
            assert page.title() == "YOLO 弱點報告"
            assert page.locator("#consistency").count() == 1          # the section heading renders
            table = page.locator("#consistency-table")
            assert table.count() == 1                                 # the attribution table renders
            body = table.inner_text()
            assert "a↔b" in body and "taxonomy" in body               # the pair + its verdict are shown
            assert page.locator("td.v.tax").count() >= 1              # verdict cell is styled (real DOM)
            assert "PROXY" in page.locator("body").inner_text()       # honesty stamp is visible
        finally:
            browser.close()
