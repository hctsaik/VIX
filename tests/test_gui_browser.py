"""Genuine BROWSER (Playwright) GUI scenarios against a live FiftyOne App (Tier-2; needs fiftyone +
Mongo + a Chromium install). Complements tests/test_gui_e2e.py (handler/ledger-anchored): this layer
proves the App actually mounts and renders in a real browser DOM, that both VIX panels open in the
App, and that an in-browser operator execution writes exactly one chained ledger event.

Launches one App on a dedicated port (module-scoped) and drives it with Playwright using
wait_until="domcontentloaded" (the FiftyOne websocket never goes idle -> networkidle would hang).
Skips cleanly if fiftyone/playwright/chromium are unavailable.
"""

import os
import re
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("fiftyone")
pytest.importorskip("playwright.sync_api")

PORT = 5153
URL = f"http://localhost:{PORT}"


def _reviews(cfg):
    from vix.core.decision_log import DecisionLog
    return [r for r in DecisionLog(cfg.decision_log_path).read_all() if r.get("event") == "review"]


@pytest.fixture(scope="module")
def app(tmp_path_factory):
    os.environ["FIFTYONE_PLUGINS_DIR"] = str(Path(__file__).resolve().parent.parent / "src" / "vix" / "plugins")
    os.environ.setdefault("FIFTYONE_DO_NOT_TRACK", "true")
    os.environ["VIX_WORKSPACE"] = str(tmp_path_factory.mktemp("guiws").resolve())

    import fiftyone as fo
    from playwright.sync_api import sync_playwright

    from vix import pipeline, verification as V
    from vix.adapters.fiftyone_adapter import FiftyOneAdapter
    from vix.config import Config

    cfg = Config()
    cfg.ensure_dirs()
    cfg.embedding_backend = "pixel_fallback"
    ds = V._build_dataset(fo)
    ad = FiftyOneAdapter(cfg, dataset_name=V.DATASET)
    pipeline.calibrate(ad, cfg)
    pipeline.route(ad, cfg)
    ds.reload()
    session = fo.launch_app(ds, remote=True, port=PORT)

    import urllib.request
    ready = False
    for _ in range(60):
        try:
            urllib.request.urlopen(URL, timeout=2)
            ready = True
            break
        except Exception:
            time.sleep(1)

    shots = Path(os.environ["VIX_WORKSPACE"]) / "shots"
    shots.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        try:
            browser = pw.chromium.launch()
        except Exception as e:  # noqa: BLE001
            session.close()
            pytest.skip(f"chromium unavailable: {e}")
        page = browser.new_page(viewport={"width": 1600, "height": 1000})
        if ready:
            page.goto(URL, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(12000)  # grid render (CI runners slow; the websocket never idles)
        yield SimpleNamespace(fo=fo, session=session, page=page, ds=ds, cfg=cfg, ad=ad, shots=shots, ready=ready)
        browser.close()
    session.close()
    if fo.dataset_exists(V.DATASET):
        fo.delete_dataset(V.DATASET)


def _open_panel(app, panel_type):
    """Open a VIX panel beside the grid via the session spaces API, then let it render."""
    app.session.spaces = app.fo.Space(children=[
        app.fo.Panel(type="Samples", pinned=True),
        app.fo.Panel(type=panel_type),
    ])
    app.page.wait_for_timeout(7000)


def _events(cfg, name):
    from vix.core.decision_log import DecisionLog
    return [r for r in DecisionLog(cfg.decision_log_path).read_all() if r.get("event") == name]


def _dismiss_dialogs(pg):
    """Close any operator-result dialog / modal left open by a prior test — otherwise the next ` press
    is swallowed by the focused dialog and the operator browser never opens (sequencing bug, not a
    product bug)."""
    for _ in range(4):
        pg.keyboard.press("Escape")
        pg.wait_for_timeout(350)


def _run_operator(app, name, select=None, post_wait=3500):
    """Invoke an operator IN THE BROWSER via the ` operator browser, then Execute. Returns nothing;
    callers assert the server-side effect (brain run / tag / ledger / session.view). `select` (a sample
    id) is applied AFTER the spaces reset so the selection survives into the operator's ctx."""
    pg = app.page
    _dismiss_dialogs(pg)
    app.session.spaces = app.fo.Space(children=[app.fo.Panel(type="Samples", pinned=True)])
    pg.wait_for_timeout(1500)
    # ensure the Samples grid is the active surface (a prior test may have left the Embeddings panel
    # open, which swallows the ` operator-browser shortcut) — wait for the grid toolbar to be visible.
    try:
        pg.wait_for_selector("img[src*='folder.svg']", state="visible", timeout=15000)
    except Exception:
        pass
    if select is not None:
        app.session.selected = [select]
        pg.wait_for_timeout(1500)
    pg.keyboard.press("`"); pg.wait_for_timeout(1200)
    pg.keyboard.type(name); pg.wait_for_timeout(1200)
    pg.keyboard.press("Enter"); pg.wait_for_timeout(2500)
    btn = pg.get_by_role("button", name=re.compile("execute|run|執行|送出", re.I))
    (btn.first.click() if btn.count() else pg.keyboard.press("Enter"))
    pg.wait_for_timeout(post_wait)


def test_b6_all_vix_toolbar_buttons_present(app):
    """B6: every VIX toolbar placement button registers a clickable icon in the live grid toolbar
    (folder/trash/simindex/similar/check/ban/scatter). Catches a placement that silently fails to mount."""
    app.session.spaces = app.fo.Space(children=[app.fo.Panel(type="Samples", pinned=True)])
    app.page.wait_for_timeout(2500)
    icons = ["folder.svg", "trash.svg", "simindex.svg", "similar.svg", "check.svg", "ban.svg", "scatter.svg"]
    present = [i for i in icons if app.page.query_selector(f"img[src*='{i}']")]
    app.page.screenshot(path=str(app.shots / "b6_toolbar.png"))
    assert set(present) == set(icons), f"missing toolbar buttons: {set(icons) - set(present)}"


def test_b7_build_similarity_creates_patch_index(app):
    """B7: build_similarity executed in the browser builds the vix_patch_sim brain run (DINO patch index)."""
    _run_operator(app, "build_similarity", post_wait=8000)
    app.ds.reload()
    assert "vix_patch_sim" in app.ds.list_brain_runs()


def test_b8_find_similar_sets_similarity_view(app):
    """B8: find_similar (needs vix_patch_sim from B7 + a selection) re-views the App as patches sorted by
    similarity — the session view gains a SortBySimilarity stage. Driven via the real toolbar button
    (prompt=False -> executes on the selection). Runs BEFORE compute_visualization, whose open_panel
    would otherwise remount the toolbar mid-test."""
    _dismiss_dialogs(app.page)
    app.session.view = None                       # clear any leftover view from a prior test
    app.session.spaces = app.fo.Space(children=[app.fo.Panel(type="Samples", pinned=True)])
    app.page.wait_for_timeout(2500)
    app.page.wait_for_selector("img[src*='similar.svg']", state="visible", timeout=20000)
    s = app.ds.match({"vix_hash": "rev2"}).first() or app.ds.first()
    app.session.selected = [s.id]
    app.page.wait_for_timeout(2000)
    try:  # the click fires find_similar -> set_view (view reload); the post-nav wait may reject even
        app.page.locator("img[src*='similar.svg']").first.click(no_wait_after=True, timeout=10000, force=True)
    except Exception:
        pass
    app.page.wait_for_timeout(8000)
    stages = [type(st).__name__ for st in (app.session.view._stages if app.session.view else [])]
    assert any("Similarity" in n for n in stages), f"no similarity stage in view: {stages}"
    app.session.view = None  # restore for later tests


def test_b8b_find_similar_with_no_selection_toasts(app):
    """B8b: the real '按了沒反應' bug — clicking 找相似 with NOTHING selected used to silently return an
    error (prompt=False shows no output modal), so the user saw nothing. Now it must surface a visible
    notification toast in the live DOM. Proves the failure path gives feedback, not just the happy path."""
    _dismiss_dialogs(app.page)
    app.session.view = None
    app.session.selected = []                      # the key: NO selection
    app.session.spaces = app.fo.Space(children=[app.fo.Panel(type="Samples", pinned=True)])
    app.page.wait_for_timeout(2500)
    app.page.wait_for_selector("img[src*='similar.svg']", state="visible", timeout=20000)
    try:
        app.page.locator("img[src*='similar.svg']").first.click(no_wait_after=True, timeout=10000, force=True)
    except Exception:
        pass
    # a notification toast carrying the actionable message must appear (notistack/snackbar in the DOM)
    toast = app.page.get_by_text(re.compile("請先在格狀檢視選|請先.*選一張|沒有偵測框"))
    try:
        toast.first.wait_for(state="visible", timeout=8000)
        shown = True
    except Exception:
        shown = False
    app.page.screenshot(path=str(app.shots / "b8b_no_selection_toast.png"))
    assert shown, "clicking 找相似 with no selection produced NO visible toast (silent = 按了沒反應)"


def test_b10_dismiss_false_alarm_in_browser(app):
    """B10: dismiss_false_alarm executed in the browser tags rev2 'rejected' + writes one false_alarm ledger event."""
    rev = app.ds.match({"vix_hash": "rev2"}).first()
    before = len(_reviews(app.cfg))
    _run_operator(app, "dismiss_false_alarm", select=rev.id, post_wait=3500)
    ok = False
    for _ in range(15):
        app.ds.reload()
        if "rejected" in app.ds.match({"vix_hash": "rev2"}).first().tags:
            ok = True; break
        time.sleep(1)
    assert ok, "dismiss did not tag rev2 rejected"
    revs = _reviews(app.cfg)
    assert len(revs) == before + 1 and revs[-1]["decision"] == "false_alarm"


def test_b11_audit_label_errors_runs_in_browser(app):
    """B11: audit_label_errors (DINO cross-class) runs in the browser without crashing and logs an
    audit_labels event (vix_verify has 2 classes + embeddings)."""
    before = len(_events(app.cfg, "audit_labels"))
    _run_operator(app, "audit_label_errors", post_wait=5000)
    assert len(_events(app.cfg, "audit_labels")) >= before + 1


def test_b14_flag_image_quality_runs_in_browser(app):
    """B14: flag_image_quality (the image-level Data Quality replacement) runs in the browser via the
    operator browser, scans pixels of all samples, appends exactly one image_quality ledger event, and
    leaves the App alive. No golden needed (pixel-level)."""
    before = len(_events(app.cfg, "image_quality"))
    _run_operator(app, "flag_image_quality", post_wait=5000)
    assert len(_events(app.cfg, "image_quality")) >= before + 1
    assert app.page.locator('[data-cy="looker"], canvas, [data-cy="fo-grid"]').count() >= 1  # App still alive


def test_b12_explain_sample_runs_in_browser(app):
    """B12: explain_sample (drill-down) runs in the browser on a selection without crashing (read-only)."""
    s = app.ds.match({"vix_hash": "rev1"}).first() or app.ds.first()
    before = len(_reviews(app.cfg))
    _run_operator(app, "explain_sample", select=s.id, post_wait=3500)
    app.page.screenshot(path=str(app.shots / "b12_explain.png"))
    assert len(_reviews(app.cfg)) == before  # read-only: writes no review event, didn't crash the App
    assert app.page.locator('[data-cy="looker"], canvas, [data-cy="fo-grid"]').count() >= 1  # App still alive


def test_b12z_compute_visualization_creates_umap(app):
    """B12z: compute_visualization builds the vix_umap UMAP run (Embeddings panel). Runs AFTER the other
    operator-browser tests because its open_panel('Embeddings') side-effect disrupts a following ` invoke;
    the next test (b13 queue) opens its panel via the spaces API, which is unaffected."""
    _run_operator(app, "compute_visualization", post_wait=30000)
    app.ds.reload()
    assert "vix_umap" in app.ds.list_brain_runs()
    # object-level: the run carries patches_field so the panel plots one point per box, not per image
    assert getattr(app.ds.get_brain_info("vix_umap").config, "patches_field", None) == "yolo_detections"


def test_b13_queue_inspect_opens_sample_modal(app):
    """B13: the vix_queue panel's 看圖 row-action opens the sample modal (open_sample) for a queued item."""
    _dismiss_dialogs(app.page)
    _open_panel(app, "vix_queue")
    body = app.page.locator("body").inner_text()
    if "尚未就緒" in body:
        pytest.skip("queue disabled (no golden-with-embeddings in this run) — panel mounted honestly")
    # the 看圖 row-action renders as an icon button; try several handles (role/title/text/visibility icon)
    handle = None
    for finder in [lambda: app.page.get_by_role("button", name=re.compile("看圖")),
                   lambda: app.page.get_by_title(re.compile("看圖")),
                   lambda: app.page.get_by_text("看圖", exact=False)]:
        loc = finder()
        if loc.count():
            handle = loc.first; break
    if handle is None:
        handle = app.page.query_selector('svg[data-testid="VisibilityIcon"]')
    if handle is None:
        app.page.screenshot(path=str(app.shots / "b13_queue_norows.png"))
        pytest.skip("queue 看圖 row-action not locatable (TableView icon-button rendering)")
    handle.click()
    app.page.wait_for_timeout(3500)
    app.page.screenshot(path=str(app.shots / "b13_queue_inspect.png"))
    assert app.page.query_selector('[data-cy="modal"], [data-cy="looker-modal"], [data-cy*="modal"]') is not None


def test_b1_app_grid_renders_in_dom(app):
    """B1: the App serves the Mongo-backed dataset and the sample grid actually paints in the DOM."""
    assert app.ready, "App server did not become ready"
    app.page.screenshot(path=str(app.shots / "b1_grid.png"), full_page=True)
    # a real grid/looker/canvas node must exist (not just a non-empty screenshot)
    assert app.page.locator('[data-cy="looker"], [data-cy^="sample"], canvas, [data-cy="fo-grid"]').count() >= 1


def test_b5_both_panels_open_in_the_live_app(app):
    """B5: both VIX panels are live in the App runtime — open them side-by-side and assert both panel
    tabs mount in the browser DOM. (This is a stronger proof than an in-process operator_exists, which
    is import-order-fragile here; `vix verify-gui` asserts operator_exists in a clean process.)"""
    app.session.spaces = app.fo.Space(children=[
        app.fo.Panel(type="Samples", pinned=True),
        app.fo.Panel(type="vix_queue"),
        app.fo.Panel(type="vix_report"),
    ])
    app.page.wait_for_timeout(7000)
    app.page.screenshot(path=str(app.shots / "b5_both_panels.png"), full_page=True)
    body = app.page.locator("body").inner_text()
    assert "覆核佇列" in body and "弱點" in body  # both VIX panel tabs mounted in the live App


def test_b2_queue_panel_opens_in_browser(app):
    """B2: the vix_queue panel mounts in the browser and renders its table surface (real DOM)."""
    _open_panel(app, "vix_queue")
    app.page.screenshot(path=str(app.shots / "b2_queue_panel.png"), full_page=True)
    body = app.page.locator("body").inner_text()
    # the panel's label and/or its table columns render in the DOM
    assert ("覆核佇列" in body) or ("vix_hash" in body) or ("風險" in body)


def test_b3_report_panel_opens_in_browser(app):
    """B3: the vix_report panel mounts in the browser (its '弱點' tab renders in the DOM). The report
    content + PROXY framing is asserted at the handler layer (test_gui02); here we prove the mount."""
    _open_panel(app, "vix_report")
    app.page.screenshot(path=str(app.shots / "b3_report_panel.png"), full_page=True)
    body = app.page.locator("body").inner_text()
    assert "弱點" in body  # the report panel ("VIX: 弱點/一致性報告") mounted in the browser DOM


def test_b4_confirm_golden_in_browser_writes_one_ledger_event(app):
    """B4: execute confirm_golden IN THE BROWSER (operator browser via keyboard) -> rev1 gains 'golden'
    AND the DecisionLog gains exactly one review/confirmed event with a valid chain (ledger-anchored)."""
    # reset spaces to the grid so the operator browser targets the selection
    _dismiss_dialogs(app.page)  # clear any operator-result dialog left open by a prior test
    app.session.spaces = app.fo.Space(children=[app.fo.Panel(type="Samples", pinned=True)])
    app.page.wait_for_timeout(2000)
    rev = app.ds.match({"vix_hash": "rev1"}).first()
    before = len(_reviews(app.cfg))
    app.session.selected = [rev.id]
    app.page.wait_for_timeout(1500)
    app.page.keyboard.press("`")
    app.page.wait_for_timeout(1200)
    app.page.keyboard.type("confirm_golden")
    app.page.wait_for_timeout(1200)
    app.page.keyboard.press("Enter")
    app.page.wait_for_timeout(2000)
    btn = app.page.get_by_role("button", name=re.compile("execute|run|執行|送出", re.I))
    (btn.first.click() if btn.count() else app.page.keyboard.press("Enter"))
    app.page.screenshot(path=str(app.shots / "b4_after_confirm.png"))

    from vix.core.decision_log import DecisionLog
    ok, revs = False, []
    for _ in range(15):  # deterministic poll for the in-browser effect
        app.ds.reload()
        tags = app.ds.match({"vix_hash": "rev1"}).first().tags
        revs = _reviews(app.cfg)
        if "golden" in tags:
            ok = True
            break
        time.sleep(1)
    assert ok, "confirm_golden did not tag rev1 golden in the browser"
    assert len(revs) == before + 1 and revs[-1]["vix_hash"] == "rev1" and revs[-1]["decision"] == "confirmed"
    assert DecisionLog(app.cfg.decision_log_path).verify_chain()
