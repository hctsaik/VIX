"""Comprehensive live-App GUI scenario suite (Tier-2; needs fiftyone + a running Mongo).

Drives the ACTUAL plugin panel/operator handlers (VixReportPanel / VixQueuePanel / ConfirmGolden /
DismissFalseAlarm / ExplainSample) against a LIVE Mongo-backed FiftyOne dataset through the real
FiftyOneAdapter + pipeline + hash-chained DecisionLog — the same code the browser invokes, minus the
DOM. Assertions are LEDGER-ANCHORED (tags + exact DecisionLog event count/shape + verify_chain +
is_truncated + reviewer provenance + adverse twin + no-spurious-write), per the multi-agent rubric:
"perfect" = correct effect + graceful failure + audit integrity + honest framing.

Genuine browser/Playwright acceptance (App loads, grid renders, panels mount, confirm executes in the
DOM) lives in tests/test_gui_browser.py + `vix verify-gui`. Scenario IDs map to docs/discussion/gui-test-plan.md.
"""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("fiftyone")

import fiftyone as fo  # noqa: E402

from vix import pipeline, verification as V  # noqa: E402
from vix.adapters.fiftyone_adapter import FiftyOneAdapter  # noqa: E402
from vix.config import Config  # noqa: E402
from vix.core.decision_log import DecisionLog  # noqa: E402
from vix.types import Tag  # noqa: E402


def _load_plugin():
    path = Path(__file__).resolve().parent.parent / "src" / "vix" / "plugins" / "vix_review" / "__init__.py"
    spec = importlib.util.spec_from_file_location("vix_review_plugin_e2e", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


PLUGIN = _load_plugin()


class _Ops:
    """Records ctx.ops.* calls so navigation/reload effects are assertable."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def f(*a, **k):
            self.calls.append((name, a, k))
        return f


def _ctx(ds, selected=None, params=None, user_id="tester"):
    panel = SimpleNamespace(state=SimpleNamespace(md=None, err=None, rows=[]),
                            data=SimpleNamespace(rows=[]))
    return SimpleNamespace(dataset=ds, selected=list(selected or []), params=params or {},
                           user_id=user_id, ops=_Ops(), panel=panel, results=None, selected_labels=[])


def _log(cfg):
    return DecisionLog(cfg.decision_log_path)


def _reviews(cfg):
    return [r for r in _log(cfg).read_all() if r.get("event") == "review"]


def _events(cfg, name):
    return [r for r in _log(cfg).read_all() if r.get("event") == name]


def _chain_ok(cfg):
    return _log(cfg).verify_chain()


def _sid(ds, h):
    return ds.match({"vix_hash": h}).first().id


@pytest.fixture
def live(tmp_path, monkeypatch):
    """A calibrated+routed live dataset (16 golden + rev1/rev2/cand_low review candidates)."""
    monkeypatch.setenv("VIX_WORKSPACE", str((tmp_path / "ws").resolve()))
    cfg = Config()
    cfg.ensure_dirs()
    cfg.embedding_backend = "pixel_fallback"
    ds = V._build_dataset(fo)  # dataset "vix_verify" (persistent, pixel-fallback embeddings)
    ad = FiftyOneAdapter(cfg, dataset_name=V.DATASET)
    pipeline.calibrate(ad, cfg)
    pipeline.route(ad, cfg)
    ds.reload()
    yield SimpleNamespace(ds=ds, cfg=cfg, ad=ad)
    if fo.dataset_exists(V.DATASET):
        fo.delete_dataset(V.DATASET)


@pytest.fixture
def bare(tmp_path, monkeypatch):
    """A dataset with NO golden and NO calibration (degraded day-0 state) for graceful-failure tests."""
    monkeypatch.setenv("VIX_WORKSPACE", str((tmp_path / "ws2").resolve()))
    cfg = Config()
    cfg.ensure_dirs()
    cfg.embedding_backend = "pixel_fallback"
    name = "vix_gui_bare"
    if fo.dataset_exists(name):
        fo.delete_dataset(name)
    ds = fo.Dataset(name, persistent=True)
    samples = []
    for i in range(3):
        det = fo.Detection(label="x", bounding_box=[0, 0, 1, 1], confidence=0.3)
        det["dino_embedding"] = [float(i), 0.0, 0.0]
        s = fo.Sample(filepath=f"/tmp/bare{i}.png", tags=["review"])
        s["vix_hash"] = f"bare{i}"
        s["yolo_detections"] = fo.Detections(detections=[det])
        samples.append(s)
    ds.add_samples(samples)
    yield SimpleNamespace(ds=ds, cfg=cfg, name=name)
    if fo.dataset_exists(name):
        fo.delete_dataset(name)


# ============================ Happy-path / rendering (GUI-02..GUI-10) ============================

def test_gui05_queue_table_renders_rows(live):
    """GUI-05: vix_queue.on_load binds review-queue rows (id/risk/why) == pipeline.review_queue; read-only."""
    p = PLUGIN.VixQueuePanel()
    ctx = _ctx(live.ds)
    before = len(_reviews(live.cfg))
    p.on_load(ctx)
    rows = ctx.panel.data.rows
    assert rows and ctx.panel.state.err is None
    assert {"id", "risk", "why"} <= set(rows[0])
    assert {r["id"] for r in rows} == {r["id"] for r in pipeline.review_queue(live.ad, live.cfg)}  # exact
    assert len(_reviews(live.cfg)) == before and _chain_ok(live.cfg) and not _log(live.cfg).is_truncated()


def test_gui06_queue_inspect_navigates_to_sample(live):
    """GUI-06: inspect row -> ctx.ops.open_sample pops EXACTLY that sample's image in the modal (works
    from the panel tab; set_view only filtered the hidden grid behind it)."""
    p = PLUGIN.VixQueuePanel()
    ctx = _ctx(live.ds)
    p.on_load(ctx)
    target = ctx.panel.state.rows[0]["id"]                 # vix_hash of the clicked row
    ctx.params = {"row": 0}
    p.on_inspect(ctx)
    opens = [c for c in ctx.ops.calls if c[0] == "open_sample"]
    assert len(opens) == 1
    assert opens[0][2].get("id") == _sid(live.ds, target)  # opens exactly the clicked sample's image
    assert not _reviews(live.cfg)                          # navigation is read-only


def test_gui_build_similarity_patch_index(live):
    """Similarity-A: BuildSimilarity creates the OBJECT-BOX (patch) index over DINO crop embeddings so
    the App's native sort-by-similarity ranks by object look, not whole scene. Idempotent; read-only."""
    op = PLUGIN.BuildSimilarity()
    before = len(_reviews(live.cfg))
    out = op.execute(_ctx(live.ds))                                   # embeddings already present -> just indexes
    assert out.get("brain_key") == "vix_patch_sim" and not out.get("error")
    live.ds.reload()
    assert "vix_patch_sim" in live.ds.list_brain_runs()
    info = live.ds.get_brain_info("vix_patch_sim")
    assert getattr(info.config, "patches_field", None) == "yolo_detections"   # OBJECT-level, not whole-image
    out2 = op.execute(_ctx(live.ds))                                  # idempotent: re-click replaces, no crash
    assert out2.get("brain_key") == "vix_patch_sim"
    assert len(_reviews(live.cfg)) == before and _chain_ok(live.cfg)  # index build writes no review decision


def test_gui_find_similar_uses_dino_index(live):
    """Find-similar (OSS, no Enterprise replacement): with the patch index built, selecting a sample and
    running find_similar drives the App (set_view) to a patches view sorted by that object's DINO
    similarity. Read-only; writes no review; never touches a zoo model / Enterprise."""
    live.ad.build_patch_similarity()
    live.ds.reload()
    op = PLUGIN.FindSimilar()
    ctx = _ctx(live.ds, selected=[live.ds.first().id])
    out = op.execute(ctx)
    assert not out.get("error"), out
    assert out.get("shown", 0) >= 1
    setviews = [c for c in ctx.ops.calls if c[0] == "set_view"]
    assert len(setviews) == 1
    view = setviews[0][2].get("view")
    assert any("Similarity" in st.__class__.__name__ for st in view._stages)  # sorted by similarity
    # success must TOAST too (prompt=False shows no output form) — else a working find still looks "dead"
    assert any(c[0] == "notify" and c[2].get("variant") == "success" for c in ctx.ops.calls)
    assert not _reviews(live.cfg) and _chain_ok(live.cfg)


def test_gui_find_similar_needs_index_and_selection(live):
    """Regression guard for '按了沒反應': because find_similar is prompt=False (no output modal), a bare
    `return {"error": ...}` is INVISIBLE — every failure path MUST ctx.ops.notify or the user sees nothing.
    Assert the notify fires (not just that an error string is returned)."""
    op = PLUGIN.FindSimilar()
    ctx1 = _ctx(live.ds, selected=[live.ds.first().id])          # index not built yet
    assert "建立相似" in (op.execute(ctx1).get("error") or "")
    assert any(c[0] == "notify" for c in ctx1.ops.calls), "no-index path was SILENT (no notify) -> 按了沒反應"
    live.ad.build_patch_similarity(); live.ds.reload()
    ctx2 = _ctx(live.ds, selected=[])                            # nothing selected
    assert (op.execute(ctx2).get("error") or "")
    assert any(c[0] == "notify" for c in ctx2.ops.calls), "no-selection path was SILENT (no notify) -> 按了沒反應"
    assert _chain_ok(live.cfg)


def test_gui_find_similar_self_heals_stale_index(live):
    """User-reported error: clicking 找相似 raised 'Query IDs [...] do not exist in this index' — the
    selected box wasn't in the (stale) index, NOT a 'no similar found'. find_similar must SELF-HEAL:
    rebuild the index from existing embeddings and still return a similarity view, never the raw error."""
    live.ad.build_patch_similarity()                          # index over the CURRENT detections only
    live.ds.reload()
    ref = None                                                # borrow a real box embedding (right field+dim)
    for smp in live.ds:
        d = smp["yolo_detections"].detections if smp["yolo_detections"] else []
        if d and d[0].has_field("dino_embedding") and d[0]["dino_embedding"] is not None:
            ref = list(d[0]["dino_embedding"]); break
    assert ref is not None
    s = fo.Sample(filepath="/tmp/vix_stale_box.png")          # a NEW box the stale index has never seen
    s["vix_hash"] = "stalebox"
    s["yolo_detections"] = fo.Detections(detections=[
        fo.Detection(label="stale", bounding_box=[0.1, 0.1, 0.2, 0.2], confidence=0.9, dino_embedding=ref)])
    live.ds.add_sample(s)
    live.ds.reload()
    op = PLUGIN.FindSimilar()
    ctx = _ctx(live.ds, selected=[s.id])                      # its id is NOT in the index -> would error
    out = op.execute(ctx)
    assert not out.get("error"), out                          # self-healed, not the cryptic index error
    assert out.get("shown", 0) >= 1
    assert any(c[0] == "set_view" for c in ctx.ops.calls)
    assert any(c[0] == "notify" and c[2].get("variant") == "success" for c in ctx.ops.calls)


def test_build_similarity_survives_stale_adapter(live, monkeypatch):
    """A long-running App reloads the plugin but CACHES `import vix.adapters...`; if that cached adapter
    predates has_full_embeddings, BuildSimilarity must FALL BACK to has_embeddings — not crash with
    \"'FiftyOneAdapter' object has no attribute 'has_full_embeddings'\". Simulate the stale adapter by
    removing the new method. (Fresh-process tests never reproduce plugin-new/adapter-old skew — this does.)"""
    monkeypatch.delattr(PLUGIN.FiftyOneAdapter, "has_full_embeddings", raising=False)
    op = PLUGIN.BuildSimilarity()
    out = op.execute(_ctx(live.ds))
    assert not out.get("error"), out               # fell back, did not AttributeError-crash
    assert out.get("brain_key") == "vix_patch_sim"


def test_gui_compute_visualization_builds_umap(live):
    """Embeddings-viz (OSS replacement for Enterprise 'Create Embeddings'): ComputeVisualization builds
    a UMAP brain run (vix_umap) from the DINO vectors so the native Embeddings panel can plot it.
    Read-only w.r.t. reviews; idempotent."""
    op = PLUGIN.ComputeVisualization()
    before = len(_reviews(live.cfg))
    out = op.execute(_ctx(live.ds))
    assert out.get("brain_key") == "vix_umap" and not out.get("error"), out
    live.ds.reload()
    assert "vix_umap" in live.ds.list_brain_runs()
    info = live.ds.get_brain_info("vix_umap")
    assert "Visualization" in info.config.cls or "umap" in info.config.cls.lower()  # a viz run, not similarity
    assert op.execute(_ctx(live.ds)).get("brain_key") == "vix_umap"   # idempotent re-run, no crash
    assert len(_reviews(live.cfg)) == before and _chain_ok(live.cfg)


def test_adapter_patch_similarity_and_has_embeddings(live):
    """Adapter seam: build_patch_similarity returns the patch brain key; has_embeddings detects the
    per-detection DINO vectors (so the operator can skip the expensive recompute)."""
    assert live.ad.has_embeddings() is True
    bk = live.ad.build_patch_similarity()
    assert bk == "vix_patch_sim" and bk in live.ds.list_brain_runs()


def test_gui07_queue_confirm_golden_one_event_and_drops(live):
    """GUI-07: confirm row -> golden + EXACTLY ONE review/confirmed event + reviewer + chain + untruncated + drops."""
    p = PLUGIN.VixQueuePanel()
    ctx = _ctx(live.ds)
    p.on_load(ctx)
    h = ctx.panel.state.rows[0]["id"]
    before = len(_reviews(live.cfg))
    ctx.params = {"row": 0}
    p.on_confirm(ctx)
    live.ds.reload()
    after = _reviews(live.cfg)
    assert len(after) == before + 1                               # exactly one write
    assert after[-1]["vix_hash"] == h and after[-1]["decision"] == "confirmed"
    assert after[-1]["reviewer_id"] == "tester"
    assert "golden" in live.ds.match({"vix_hash": h}).first().tags
    assert _chain_ok(live.cfg) and not _log(live.cfg).is_truncated()
    assert h not in {r["id"] for r in pipeline.review_queue(live.ad, live.cfg)}


def test_gui08_queue_dismiss_rejected_one_event(live):
    """GUI-08: dismiss row -> rejected + one false_alarm event + reviewer + chain + untruncated + drops (twin of 07)."""
    p = PLUGIN.VixQueuePanel()
    ctx = _ctx(live.ds)
    p.on_load(ctx)
    h = ctx.panel.state.rows[0]["id"]
    before = len(_reviews(live.cfg))
    ctx.params = {"row": 0}
    p.on_dismiss(ctx)
    live.ds.reload()
    after = _reviews(live.cfg)
    assert len(after) == before + 1 and after[-1]["decision"] == "false_alarm"
    assert after[-1]["vix_hash"] == h and after[-1]["reviewer_id"] == "tester"
    assert "rejected" in live.ds.match({"vix_hash": h}).first().tags
    assert _chain_ok(live.cfg) and not _log(live.cfg).is_truncated()
    assert h not in {r["id"] for r in pipeline.review_queue(live.ad, live.cfg)}


def test_gui02_report_panel_renders_markdown(live):
    """GUI-02: vix_report.on_load fills markdown with the report (PROXY honesty stamp); read-only, chain intact."""
    p = PLUGIN.VixReportPanel()
    ctx = _ctx(live.ds)
    before = len(_reviews(live.cfg))
    p.on_load(ctx)
    assert ctx.panel.state.md and "健康度" in ctx.panel.state.md  # compact panel layout leads with the verdict badge
    assert "PROXY" in ctx.panel.state.md                    # honest framing preserved on the GUI surface
    assert len(_reviews(live.cfg)) == before and _chain_ok(live.cfg) and not _log(live.cfg).is_truncated()


def test_gui02b_report_filename_links_to_image(live):
    """GUI-02b: a filename row in the report panel is CLICKABLE — 看圖 drives the grid to that image
    (the owner's 'can I link a filename back to its picture?' ask). Read-only; writes nothing; chain intact."""
    p = PLUGIN.VixReportPanel()
    ctx = _ctx(live.ds, params={"row": 0})
    before = len(_reviews(live.cfg))
    # a navigable confident-wrong row (eval-derived in production; here seeded to a real sample's hash)
    ctx.panel.state.cw = [{"file": "rev1.png", "hash": "rev1", "pred_class": "a", "conf": 0.9, "fp_type": "-"}]
    p.on_inspect_cw(ctx)
    opens = [c for c in ctx.ops.calls if c[0] == "open_sample"]
    assert opens, "看圖 must open the clicked image in the sample modal"  # the filename links back to the picture
    assert opens[0][2].get("id") == _sid(live.ds, "rev1")                 # opens exactly that image
    assert len(_reviews(live.cfg)) == before and _chain_ok(live.cfg)      # navigation is read-only


def test_gui02c_report_bad_row_navigates_nowhere(live):
    """GUI-02c: a stale/unknown filename row no-ops (never crashes, never set_view to nothing)."""
    p = PLUGIN.VixReportPanel()
    ctx = _ctx(live.ds, params={"row": 0})
    ctx.panel.state.cw = [{"file": "gone.png", "hash": "no_such_hash", "pred_class": "a", "conf": 0.9}]
    p.on_inspect_cw(ctx)                                                  # must not raise
    assert not [c for c in ctx.ops.calls if c[0] == "open_sample"]        # unknown hash -> open nothing
    assert _chain_ok(live.cfg)


def test_gui03_report_regen_appends_audit(live):
    """GUI-03: regen re-runs weakness_report (exactly one fresh audit event); chain valid + untruncated."""
    p = PLUGIN.VixReportPanel()
    ctx = _ctx(live.ds)
    before = len(_events(live.cfg, "weakness_report"))
    p.on_regen(ctx)
    assert len(_events(live.cfg, "weakness_report")) == before + 1
    assert ctx.panel.state.md and "健康度" in ctx.panel.state.md  # compact panel layout leads with the verdict badge
    assert _chain_ok(live.cfg) and not _log(live.cfg).is_truncated()


def test_gui04_worklist_tags_match_views(live):
    """GUI-04 / DI-5: worklist tags vixq:* EXACTLY map to worklist_views() specs; no review write; chain valid."""
    p = PLUGIN.VixReportPanel()
    ctx = _ctx(live.ds)
    before = len(_reviews(live.cfg))
    p.on_worklist(ctx)
    live.ds.reload()
    all_tags = {t for s in live.ds for t in s.tags}
    vixq = {t for t in all_tags if t.startswith("vixq:")}
    assert vixq == set(pipeline.worklist_views(all_tags).values())  # 1:1, both directions
    assert len(_reviews(live.cfg)) == before                        # worklist writes no review decision
    assert _chain_ok(live.cfg)


def test_gui10_explain_sample_drilldown(live):
    """GUI-10: explain returns a non-empty drill-down; writes NO review (audit_labels bookkeeping ok); chain valid."""
    op = PLUGIN.ExplainSample()
    before = len(_reviews(live.cfg))
    out = op.execute(_ctx(live.ds, selected=[_sid(live.ds, "rev1")]))
    assert out.get("explanation")
    assert len(_reviews(live.cfg)) == before and _chain_ok(live.cfg)


# ============================ Error / edge / empty (S1..S7, E1..E3) ============================

def test_s1_report_no_golden_is_graceful(bare):
    """S1: report panel on a no-golden dataset renders something, never crashes, never writes a review."""
    p = PLUGIN.VixReportPanel()
    ctx = _ctx(bare.ds)
    p.on_load(ctx)  # must not raise
    assert ctx.panel.state.md
    assert _chain_ok(bare.cfg) and not _reviews(bare.cfg) and not _log(bare.cfg).is_truncated()


def test_s3_queue_uncalibrated_is_graceful(bare):
    """S3: queue panel without calibration renders without crashing (rows or named error), no write, chain valid."""
    p = PLUGIN.VixQueuePanel()
    ctx = _ctx(bare.ds)
    p.on_load(ctx)  # must not raise
    if ctx.panel.state.err:
        assert "確認" in ctx.panel.state.err or "golden" in ctx.panel.state.err
    assert isinstance(ctx.panel.data.rows, list)
    assert not _reviews(bare.cfg) and _chain_ok(bare.cfg)


def test_resolve_review_clears_opposing_tag(live):
    """Re-confirm after dismiss (and vice-versa) must clear the opposing terminal tag, so a sample never
    holds GOLDEN∧REJECTED — that contradictory state is silently dropped by export."""
    h = next((hh for hh, _s, _d, t in live.ad.samples() if "golden" not in t and "rejected" not in t), None)
    assert h, "need a not-yet-resolved sample"
    pipeline.resolve_review(live.ad, live.cfg, h, "false_alarm")
    live.ds.reload()
    assert "rejected" in live.ds.match({"vix_hash": h}).first().tags
    pipeline.resolve_review(live.ad, live.cfg, h, "confirm")           # change of mind
    live.ds.reload()
    tags = live.ds.match({"vix_hash": h}).first().tags
    assert "golden" in tags and "rejected" not in tags                 # clean transition, no contradiction


def test_confirm_golden_relabel_preserves_embeddings(live):
    """Regression (the reported bug): confirming→golden WITH a relabel must NOT wipe the DINO crop
    embeddings. set_detections used to rebuild fo.Detection without the embedding field, so relabelled
    golden lost its vector -> _image_items skipped it -> review_queue saw 'no golden' despite the tag."""
    import fiftyone as fo  # noqa: F401
    h = next((hh for hh, _s, dets, tags in live.ad.samples()
              if "golden" not in tags and any(d.embedding is not None for d in dets)), None)
    assert h, "fixture needs a non-golden sample with detection embeddings"
    pipeline.resolve_review(live.ad, live.cfg, h, "confirm", label="relabeled_x")  # exercises set_detections
    live.ds.reload()
    dets_after = next(dets for hh, _s, dets, _t in live.ad.samples() if hh == h)
    assert any(d.embedding is not None for d in dets_after), "relabel wiped the detection embedding"
    assert any(d.label == "relabeled_x" for d in dets_after)                       # relabel still applied
    assert h in {it.id for it in pipeline._image_items(live.ad, want_tags=[Tag.GOLDEN])}  # counts as golden ref


def test_gui_queue_warns_no_golden_instead_of_fake_rows(bare):
    """Honesty guard (the reported bug): with NO golden reference the queue panel must show a loud
    warning and emit ZERO rows — not a uniform fake-confident `far_from_known` table."""
    p = PLUGIN.VixQueuePanel()
    ctx = _ctx(bare.ds)
    p.on_load(ctx)
    assert ctx.panel.data.rows == []                                  # no degenerate rows
    assert ctx.panel.state.err and ("golden" in ctx.panel.state.err or "calibrate" in ctx.panel.state.err)
    assert not _reviews(bare.cfg) and _chain_ok(bare.cfg)


def test_s4_empty_queue_has_no_error_block(live):
    """S4: when all candidates are resolved, the queue is empty WITH NO error block (empty != broken); chain valid."""
    for r in pipeline.review_queue(live.ad, live.cfg):
        pipeline.resolve_review(live.ad, live.cfg, r["id"], "false_alarm")
    p = PLUGIN.VixQueuePanel()
    ctx = _ctx(live.ds)
    p.on_load(ctx)
    assert ctx.panel.data.rows == [] and ctx.panel.state.err is None
    assert _chain_ok(live.cfg) and not _log(live.cfg).is_truncated()


def test_s5_confirm_no_selection_no_write(live):
    """S5: confirm_golden with nothing selected -> friendly error, ZERO writes, chain valid.
    ConfirmGolden has no resolve_output, so the error is INVISIBLE unless it also notifies (按了沒反應)."""
    op = PLUGIN.ConfirmGolden()
    before = len(_reviews(live.cfg))
    ctx = _ctx(live.ds, selected=[])
    out = op.execute(ctx)
    assert "error" in out and len(_reviews(live.cfg)) == before
    assert any(c[0] == "notify" for c in ctx.ops.calls), "no-selection was SILENT (no notify)"
    assert _chain_ok(live.cfg) and not _log(live.cfg).is_truncated()


def test_s6_explain_no_selection_friendly_error(live):
    """S6: explain_sample with nothing selected -> friendly error, no write, chain valid."""
    op = PLUGIN.ExplainSample()
    before = len(_reviews(live.cfg))
    out = op.execute(_ctx(live.ds, selected=[]))
    assert "error" in out and "選取" in out["error"]
    assert len(_reviews(live.cfg)) == before and _chain_ok(live.cfg)


def test_s7_inspect_stale_hash_is_noop(live):
    """S7: inspect a hash not in the dataset -> no open_sample, no crash, no write, chain valid (degrade path)."""
    p = PLUGIN.VixQueuePanel()
    before = len(_reviews(live.cfg))
    ctx = _ctx(live.ds, params={"id": "does-not-exist"})
    p.on_inspect(ctx)  # must not raise
    assert not any(c[0] == "open_sample" for c in ctx.ops.calls)
    assert len(_reviews(live.cfg)) == before and _chain_ok(live.cfg)


def test_e1_queue_confirm_unknown_hash_is_graceful(live):
    """E1 (bug fixed): confirm an unknown hash -> caught, friendly err, ZERO write, chain valid + untruncated."""
    p = PLUGIN.VixQueuePanel()
    ctx = _ctx(live.ds, params={"id": "ghost-hash"})
    before = len(_reviews(live.cfg))
    p.on_confirm(ctx)
    assert ctx.panel.state.err and "重新整理" in ctx.panel.state.err
    assert len(_reviews(live.cfg)) == before
    assert _chain_ok(live.cfg) and not _log(live.cfg).is_truncated()


def test_e2_selection_without_vix_hash_skipped(live):
    """E2 (bug fixed): a selected sample lacking vix_hash is skipped (not KeyError); no write, chain valid."""
    s = fo.Sample(filepath="/tmp/non_vix.png")
    live.ds.add_sample(s)
    op = PLUGIN.ConfirmGolden()
    before = len(_reviews(live.cfg))
    out = op.execute(_ctx(live.ds, selected=[s.id]))  # must not raise
    assert "error" in out and len(_reviews(live.cfg)) == before and _chain_ok(live.cfg)


def test_e3_worklist_no_golden_is_graceful(bare):
    """E3 (bug fixed): worklist button on a no-golden dataset never crashes / corrupts the log."""
    p = PLUGIN.VixReportPanel()
    ctx = _ctx(bare.ds)
    p.on_worklist(ctx)  # must not raise
    assert ctx.panel.state.md
    assert _chain_ok(bare.cfg) and not _log(bare.cfg).is_truncated() and not _reviews(bare.cfg)


# ============================ Data-integrity / audit (DI-1..DI-4) ============================

def test_di1_di3_gui_confirm_matches_cli_shape(live):
    """DI-1/DI-3: a GUI confirm writes one review event whose VALUES match the CLI path field-for-field
    (except ts/reviewer/chain) -> the GUI is presentation over the same core, not a second write path."""
    pipeline.resolve_review(live.ad, live.cfg, "rev2", "confirm", reviewer_id="cli")
    cli_rec = _reviews(live.cfg)[-1]
    PLUGIN.ConfirmGolden().execute(_ctx(live.ds, selected=[_sid(live.ds, "rev1")], user_id="gui"))
    gui_rec = _reviews(live.cfg)[-1]
    assert set(gui_rec) == set(cli_rec)  # identical schema
    differ = {"ts_utc", "reviewer_id", "entry_hash", "prev_hash", "vix_hash"}
    for k in set(gui_rec) - differ:      # identical VALUES on every semantic field
        assert gui_rec[k] == cli_rec[k], k
    assert gui_rec["decision"] == cli_rec["decision"] == "confirmed"
    assert gui_rec["vix_hash"] == "rev1" and gui_rec["reviewer_id"] == "gui"


def test_di2_chain_valid_across_mixed_sequence(live):
    """DI-2: a mixed GUI sequence (confirm, dismiss, regen) keeps the chain valid + untruncated throughout."""
    qp, rp = PLUGIN.VixQueuePanel(), PLUGIN.VixReportPanel()
    ctx = _ctx(live.ds)
    qp.on_load(ctx)
    ctx.params = {"row": 0}
    qp.on_confirm(ctx)
    assert _chain_ok(live.cfg) and not _log(live.cfg).is_truncated()
    ctx2 = _ctx(live.ds)
    qp.on_load(ctx2)
    assert ctx2.panel.state.rows  # 3 candidates - 1 confirmed = >=2 remain, so the dismiss leg always runs
    ctx2.params = {"row": 0}
    qp.on_dismiss(ctx2)
    assert _chain_ok(live.cfg) and not _log(live.cfg).is_truncated()
    rp.on_regen(_ctx(live.ds))
    assert _chain_ok(live.cfg) and not _log(live.cfg).is_truncated()


def test_di4_reconfirm_is_logged_and_chain_valid(live):
    """DI-4: re-confirming logs each human action (append-only), golden stays once, chain valid."""
    op = PLUGIN.ConfirmGolden()
    sid = _sid(live.ds, "rev1")
    op.execute(_ctx(live.ds, selected=[sid]))
    n1 = len(_reviews(live.cfg))
    live.ds.reload()
    op.execute(_ctx(live.ds, selected=[sid]))
    assert len(_reviews(live.cfg)) == n1 + 1
    live.ds.reload()
    assert live.ds.match({"vix_hash": "rev1"}).first().tags.count("golden") == 1
    assert _chain_ok(live.cfg)


# ============================ Round 2: relabel / multi-select / operators ============================

def test_r2_confirm_with_relabel(live):
    """R2-21: confirm WITH a label relabels detections, writes a reversible relabel_changes record, and the
    review event's decision carries the new label; chain valid."""
    op = PLUGIN.ConfirmGolden()
    before = len(_reviews(live.cfg))
    op.execute(_ctx(live.ds, selected=[_sid(live.ds, "rev1")], params={"label": "zzz"}))
    live.ds.reload()
    dets = live.ds.match({"vix_hash": "rev1"}).first()["yolo_detections"].detections
    assert all(d.label == "zzz" for d in dets)                       # relabeled
    rc = live.cfg.workspace / "relabel_changes.jsonl"
    assert rc.exists() and any(json.loads(l)["new"] == "zzz" for l in rc.read_text().splitlines())  # reversible record
    after = _reviews(live.cfg)
    assert len(after) == before + 1 and after[-1]["decision"] == "zzz"  # decision carries the new label
    assert _chain_ok(live.cfg)


def test_r2_confirm_relabel_same_label_no_change_record(live):
    """R2-22: confirm with a label EQUAL to the existing one writes no spurious relabel_changes line."""
    op = PLUGIN.ConfirmGolden()
    rc = live.cfg.workspace / "relabel_changes.jsonl"
    n0 = len(rc.read_text().splitlines()) if rc.exists() else 0
    op.execute(_ctx(live.ds, selected=[_sid(live.ds, "rev1")], params={"label": "vert"}))  # rev1 already 'vert'
    n1 = len(rc.read_text().splitlines()) if rc.exists() else 0
    assert n1 == n0 and _chain_ok(live.cfg)


def test_r2_multiselect_confirm(live):
    """R2-23: confirming 2 selected samples -> 2 events, 2 golden tags, {'confirmed': 2}; chain valid."""
    op = PLUGIN.ConfirmGolden()
    before = len(_reviews(live.cfg))
    out = op.execute(_ctx(live.ds, selected=[_sid(live.ds, "rev1"), _sid(live.ds, "cand_low")]))
    assert out["confirmed"] == 2 and len(_reviews(live.cfg)) == before + 2
    live.ds.reload()
    assert "golden" in live.ds.match({"vix_hash": "rev1"}).first().tags
    assert "golden" in live.ds.match({"vix_hash": "cand_low"}).first().tags
    assert _chain_ok(live.cfg)


def test_r2_multiselect_dismiss(live):
    """R2-24: dismissing 2 selected samples -> 2 false_alarm events, 2 rejected tags; chain valid."""
    op = PLUGIN.DismissFalseAlarm()
    before = len(_reviews(live.cfg))
    out = op.execute(_ctx(live.ds, selected=[_sid(live.ds, "rev1"), _sid(live.ds, "cand_low")]))
    assert out["dismissed"] == 2 and len(_reviews(live.cfg)) == before + 2
    live.ds.reload()
    assert "rejected" in live.ds.match({"vix_hash": "rev1"}).first().tags
    assert _chain_ok(live.cfg)


def test_r2_dismiss_operator_happy(live):
    """R2-25: the dismiss_false_alarm OPERATOR (grid path) -> rejected + one false_alarm event."""
    op = PLUGIN.DismissFalseAlarm()
    before = len(_reviews(live.cfg))
    out = op.execute(_ctx(live.ds, selected=[_sid(live.ds, "rev2")]))
    assert out["dismissed"] == 1 and len(_reviews(live.cfg)) == before + 1
    assert _reviews(live.cfg)[-1]["decision"] == "false_alarm"
    live.ds.reload()
    assert "rejected" in live.ds.match({"vix_hash": "rev2"}).first().tags
    assert _chain_ok(live.cfg)


def test_r2_dismiss_operator_no_selection(live):
    """R2-26: dismiss operator with nothing selected -> friendly error, zero writes, chain valid.
    DismissFalseAlarm has no resolve_output -> must notify or the user sees nothing (按了沒反應)."""
    op = PLUGIN.DismissFalseAlarm()
    before = len(_reviews(live.cfg))
    ctx = _ctx(live.ds, selected=[])
    out = op.execute(ctx)
    assert "error" in out and len(_reviews(live.cfg)) == before and _chain_ok(live.cfg)
    assert any(c[0] == "notify" for c in ctx.ops.calls), "no-selection was SILENT (no notify)"


def test_report_panel_jump_miss_toasts():
    """Regression for 按了沒反應: VixReportPanel '看圖' on a vanished/out-of-range row must toast, not no-op.
    No live dataset needed — the miss path never touches FiftyOne (idx out of range -> sid None -> notify)."""
    op = PLUGIN.VixReportPanel()
    ctx = _ctx(None, params={})              # no 'row' param, empty rows -> miss
    op._jump(ctx, [])
    assert any(c[0] == "notify" for c in ctx.ops.calls)


def test_queue_panel_resolve_unresolvable_row_toasts():
    """Regression for 按了沒反應: VixQueuePanel resolve on an unresolvable row toasts, not a silent no-op."""
    op = PLUGIN.VixQueuePanel()
    ctx = _ctx(None, params={})              # _row_hash -> None (no row idx, empty rows)
    op._resolve(ctx, "confirm")
    assert any(c[0] == "notify" for c in ctx.ops.calls)


def test_r2_mixed_batch_skips_non_vix(live):
    """R2-27: a selection mixing a valid sample + a non-VIX sample -> only the valid one resolves, no crash."""
    s = fo.Sample(filepath="/tmp/mixed_non_vix.png")
    live.ds.add_sample(s)
    op = PLUGIN.ConfirmGolden()
    before = len(_reviews(live.cfg))
    out = op.execute(_ctx(live.ds, selected=[_sid(live.ds, "rev1"), s.id]))
    assert out["confirmed"] == 1 and len(_reviews(live.cfg)) == before + 1  # exactly the resolvable one
    assert _chain_ok(live.cfg)


def test_r2_queue_refresh_reflects_external_resolution(live):
    """R2-28: a CLI-side resolution between on_load and on_refresh is reflected (the row drops out)."""
    p = PLUGIN.VixQueuePanel()
    ctx = _ctx(live.ds)
    p.on_load(ctx)
    n0 = len(ctx.panel.data.rows)
    h = ctx.panel.state.rows[0]["id"]
    pipeline.resolve_review(live.ad, live.cfg, h, "false_alarm")  # external (CLI) change
    p.on_refresh(ctx)
    assert h not in {r["id"] for r in ctx.panel.data.rows} and len(ctx.panel.data.rows) == n0 - 1


def test_r2_two_sequential_confirms_distinct_events(live):
    """R2-29: two confirms on different samples -> two distinct review events, both golden, chain valid."""
    op = PLUGIN.ConfirmGolden()
    before = len(_reviews(live.cfg))
    op.execute(_ctx(live.ds, selected=[_sid(live.ds, "rev1")]))
    live.ds.reload()
    op.execute(_ctx(live.ds, selected=[_sid(live.ds, "cand_low")]))
    revs = _reviews(live.cfg)
    assert len(revs) == before + 2 and revs[-1]["vix_hash"] != revs[-2]["vix_hash"]
    assert _chain_ok(live.cfg)


# ============================ Round 2: audit tamper-detection canaries ============================

def test_r2_chain_detects_middle_edit(live):
    """R2-30: editing a middle log record makes verify_chain() FALSE — proves the chain actually guards
    the records the GUI writes (not just that it returns True on happy data)."""
    PLUGIN.ConfirmGolden().execute(_ctx(live.ds, selected=[_sid(live.ds, "rev1")]))
    assert _chain_ok(live.cfg)
    lines = live.cfg.decision_log_path.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[0])
    rec["decision"] = "TAMPERED"
    lines[0] = json.dumps(rec)
    live.cfg.decision_log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert _log(live.cfg).verify_chain() is False  # tamper detected


def test_r2_truncation_detected(live):
    """R2-31: dropping the last log line trips is_truncated() (the .hwm anchor) — the GUI's writes can't
    be silently rolled back without detection."""
    PLUGIN.ConfirmGolden().execute(_ctx(live.ds, selected=[_sid(live.ds, "rev1")]))
    assert not _log(live.cfg).is_truncated()
    lines = live.cfg.decision_log_path.read_text(encoding="utf-8").splitlines()
    live.cfg.decision_log_path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    assert _log(live.cfg).is_truncated() is True


# ============================ Round 2: operator schema / config ============================

def test_r2_confirm_operator_config_and_input_schema(live):
    """R2-32: confirm_golden config (name/dynamic) + resolve_input builds the optional relabel field."""
    op = PLUGIN.ConfirmGolden()
    assert op.config.name == "confirm_golden"
    prop = op.resolve_input(_ctx(live.ds))
    assert prop is not None and prop.to_json()  # input schema serializes (the relabel 'label' field)


def test_r2_explain_output_schema_builds(live):
    """R2-33: explain_sample resolve_output builds a Property (the drill-down render schema)."""
    op = PLUGIN.ExplainSample()
    prop = op.resolve_output(_ctx(live.ds))
    assert prop is not None and prop.to_json()


def test_r2_queue_panel_row_fallback_to_id(live):
    """R2-34: _row_hash falls back to ctx.params['id'] when the row index is stale/out-of-range (frontend
    contract degrade path), so a confirm still targets the right sample."""
    p = PLUGIN.VixQueuePanel()
    ctx = _ctx(live.ds)
    p.on_load(ctx)
    h = ctx.panel.state.rows[0]["id"]
    ctx.params = {"row": 999, "id": h}        # stale index, valid id
    assert p._row_hash(ctx) == h               # falls back to the id


def test_op_generate_weakness_report(live):
    """GUI 'generate model-weakness report' operator: pick an eval JSONL -> eval-ingest + weakness-report
    (the in-App equivalent of `vix eval-ingest` + `vix weakness-report`); zero new core logic."""
    box = [0.5, 0.5, 0.4, 0.4]
    rows = [
        {"vix_hash": "rev1", "gt": [{"label": "vert", "bbox": box}], "pred": [{"label": "vert", "bbox": box, "conf": 0.9}]},
        {"vix_hash": "rev2", "gt": [{"label": "horiz", "bbox": box}], "pred": []},                       # missed
        {"vix_hash": "cand_low", "gt": [], "pred": [{"label": "vert", "bbox": box, "conf": 0.85}]},      # background FP
    ]
    jl = live.cfg.workspace / "gui_eval.jsonl"
    jl.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    out = PLUGIN.GenerateWeaknessReport().execute(_ctx(live.ds, params={"custom_path": str(jl)}))
    assert "error" not in out and out.get("mAP") is not None and out.get("health")
    assert live.cfg.eval_results_path.exists()
    assert (live.cfg.workspace / "weakness_report.md").exists()
    assert _chain_ok(live.cfg)


def test_op_audit_label_errors(live):
    """GUI cross-class label-error audit: mislabel a golden sample's class -> the operator flags it via
    embedding kNN and reports given->suggested (the standout 標成 X 但鄰居多為 Y), and tags vixq:label_error."""
    s = live.ds.match({"vix_hash": "vert0"}).first()
    s["yolo_detections"].detections[0].label = "horiz"  # wrong class (its embedding is a 'vert')
    s.save()
    out = PLUGIN.AuditLabelErrors().execute(_ctx(live.ds, params={"top": 20}))
    assert "error" not in out
    row = next((r for r in out["rows"] if r["id"] == "vert0"), None)
    assert row and row["given"] == "horiz" and row["suggested"] == "vert"  # DINO/embedding suggests the true class
    assert "vixq:label_error" in live.ds.match({"vix_hash": "vert0"}).first().tags


def test_op_flag_label_issues(live):
    """GUI 'flag inaccurate labels' operator: audit_labels + box_qa -> vixq:* tags; no crash, chain valid,
    no review write (it's a read-only audit that only tags)."""
    op = PLUGIN.FlagLabelIssues()
    before = len(_reviews(live.cfg))
    out = op.execute(_ctx(live.ds))
    assert "error" not in out
    assert isinstance(out["label_suspect"], int) and isinstance(out["box_issue"], int)
    assert len(_reviews(live.cfg)) == before and _chain_ok(live.cfg)  # audit only, no human-decision write


def test_r2_saved_views_from_worklist_tags(live):
    """R2-35 (the saved-views-in-sidebar gap): a vixq:* worklist tag -> a NAMED saved view (the exact path
    `vix app` uses to build the clickable sidebar) that resolves to the tagged sample. Non-vacuous."""
    live.ad.apply_tags("rev1", ["vixq:label:vert"])
    live.ds.reload()
    all_tags = {t for s in live.ds for t in s.tags}
    views = pipeline.worklist_views(all_tags)
    assert "工作清單 label:vert" in views                       # tag -> named view spec
    for name, tag in views.items():
        live.ds.save_view(name, live.ds.match_tags(tag))         # what cli.py does at `vix app` launch
    assert "工作清單 label:vert" in live.ds.list_saved_views()   # appears in the App's saved-views sidebar
    assert live.ds.load_saved_view("工作清單 label:vert").first()["vix_hash"] == "rev1"  # resolves to the sample
