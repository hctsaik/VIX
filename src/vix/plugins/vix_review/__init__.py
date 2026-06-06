"""VIX review workstation — FiftyOne plugin (Tier 2; requires `fiftyone`).

Turns the generic FiftyOne App into a VIX-specific review workstation: buttons to
**confirm → golden** (optionally relabel), **mark false alarm → dismiss**, and a
**"why was this flagged" drill-down**. Every action calls the same
`vix.pipeline` functions the CLI uses, so it writes to the same append-only,
hash-chained audit log — the GUI is pure presentation over the decoupled core.

Enable (on a machine with `pip install 'vix[fiftyone]'`):
    $env:FIFTYONE_PLUGINS_DIR = "C:\\code\\claude\\VIX\\src\\vix\\plugins"
    fiftyone plugins list        # should show '@vix/review'
    vix app                      # operators appear in the App's operator browser

NOTE: this module imports `fiftyone` and is loaded ONLY by the FiftyOne plugin
system — never by `import vix` or the test suite.
"""

from __future__ import annotations

import fiftyone.operators as foo
import fiftyone.operators.types as types

from vix import pipeline
from vix.adapters.fiftyone_adapter import FiftyOneAdapter
from vix.config import Config


def _adapter(ctx):
    return FiftyOneAdapter(Config(), dataset_name=ctx.dataset.name)


def _selected_hashes(ctx):
    return [ctx.dataset[sid]["vix_hash"] for sid in (ctx.selected or [])]


def _sample_id_for_hash(ctx, h):
    """vix_hash -> FiftyOne sample id (inverse of _selected_hashes). The one bit of live-only glue
    the queue panel needs to navigate; kept tiny so it's the obvious thing to find if FiftyOne drifts."""
    s = ctx.dataset.match({"vix_hash": h}).first()
    return s.id if s else None


class ConfirmGolden(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(name="confirm_golden", label="VIX: 確認 → 併入 golden", dynamic=True)

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.str("label", label="(選填)更正類別,留空則沿用原標籤", required=False)
        return types.Property(inputs, view=types.View(label="確認選取影像為 golden"))

    def execute(self, ctx):
        cfg, ad = Config(), _adapter(ctx)
        label = ctx.params.get("label") or None
        n = 0
        for h in _selected_hashes(ctx):
            pipeline.resolve_review(ad, cfg, h, "confirm", label, reviewer_id=ctx.user_id or "reviewer")
            n += 1
        ctx.ops.reload_dataset()
        return {"confirmed": n}


class DismissFalseAlarm(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(name="dismiss_false_alarm", label="VIX: 標記誤報並排除", dynamic=True)

    def execute(self, ctx):
        cfg, ad = Config(), _adapter(ctx)
        n = 0
        for h in _selected_hashes(ctx):
            pipeline.resolve_review(ad, cfg, h, "false_alarm", reviewer_id=ctx.user_id or "reviewer")
            n += 1
        ctx.ops.reload_dataset()
        return {"dismissed": n}


class ExplainSample(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(name="explain_sample", label="VIX: 為何被攔(下鑽解釋)", dynamic=True)

    def execute(self, ctx):
        cfg, ad = Config(), _adapter(ctx)
        hashes = _selected_hashes(ctx)
        if not hashes:
            return {"error": "請先在格狀檢視選取一張影像"}
        return {"explanation": pipeline.explain_one(ad, cfg, hashes[0])}

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.obj("explanation", label="VIX 下鑽解釋")
        return types.Property(outputs)


def _report_md(ctx, regenerate=False):
    """Render the weakness report (per-class AP + consistency + hit-rate + TL;DR) as markdown for the
    panel. Reuses pipeline.weakness_report (the same tested artifact the CLI writes)."""
    cfg, ad = Config(), _adapter(ctx)
    md_path = cfg.workspace / "weakness_report.md"
    if regenerate or not md_path.exists():
        try:
            pipeline.weakness_report(ad, cfg)
        except Exception as exc:  # noqa: BLE001 - surface the reason in-panel rather than crash the App
            return f"# VIX 弱點報告\n\n產生失敗:`{exc}`\n\n需先有 golden,並(選用)`vix eval-ingest <val.jsonl>`。"
    return md_path.read_text(encoding="utf-8") if md_path.exists() else "# VIX 弱點報告\n\n(尚無報告)"


class VixReportPanel(foo.Panel):
    """In-App panel surfacing the VIX weakness/consistency report (Tier 2 GUI). Pure presentation over
    pipeline.weakness_report — same audit-logged core the CLI uses."""

    @property
    def config(self):
        return foo.PanelConfig(name="vix_report", label="VIX: 弱點/一致性報告", surfaces="grid")

    def on_load(self, ctx):
        ctx.panel.state.md = _report_md(ctx)

    def on_regen(self, ctx):
        ctx.panel.state.md = _report_md(ctx, regenerate=True)

    def on_worklist(self, ctx):
        cfg, ad = Config(), _adapter(ctx)
        pipeline.weakness_report(ad, cfg, worklist=True)  # tag vixq:* so saved views become clickable
        ctx.panel.state.md = _report_md(ctx)
        ctx.ops.reload_dataset()

    def render(self, ctx):
        panel = types.Object()
        panel.md(ctx.panel.state.md or "_載入中…_", name="report")
        panel.btn("regen", label="產生 / 重新整理報告", on_click=self.on_regen)
        panel.btn("worklist", label="標記工作清單(供 saved views 點選)", on_click=self.on_worklist)
        return types.Property(panel, view=types.GridView(height=100, width=100))


def _queue_rows(ctx, top=50):
    """The risk-ranked review queue as table rows. Pure render of pipeline.review_queue (tested core):
    no ranking/decision logic lives here. Returns (rows, error_str)."""
    cfg, ad = Config(), _adapter(ctx)
    try:
        q = pipeline.review_queue(ad, cfg, top=top)
    except Exception as exc:  # noqa: BLE001 - surface in-panel rather than crash the App
        return [], str(exc)
    return [{"id": r["id"], "risk": round(r.get("risk", 0.0), 3), "why": (r.get("why") or "")[:90]} for r in q], None


class VixQueuePanel(foo.Panel):
    """The review queue as a CLICKABLE table (Tier 2 GUI): each row jumps the App view to that sample
    and can confirm→golden / dismiss in place — turning the App's one unused superpower (a click that
    drives the view) on. ZERO logic here: rows come from pipeline.review_queue, actions call
    pipeline.resolve_review — both tested core. If navigation ever breaks on a FiftyOne bump, the row
    still shows the vix_hash so the operator can fall back to the CLI (no dead-end)."""

    @property
    def config(self):
        return foo.PanelConfig(name="vix_queue", label="VIX: 覆核佇列(點列跳到該圖)", surfaces="grid")

    def on_load(self, ctx):
        rows, err = _queue_rows(ctx)
        ctx.panel.state.rows = rows
        ctx.panel.data.rows = rows  # TableView binds to the data path
        ctx.panel.state.err = err

    on_refresh = on_load

    def _row_hash(self, ctx):
        rows = ctx.panel.state.rows or []
        idx = ctx.params.get("row")
        if isinstance(idx, int) and 0 <= idx < len(rows):
            return rows[idx]["id"]
        return ctx.params.get("id")  # robust fallback if the frontend passes the id directly

    def on_inspect(self, ctx):
        h = self._row_hash(ctx)
        sid = _sample_id_for_hash(ctx, h) if h else None
        if sid:
            ctx.ops.set_view(view=ctx.dataset.select(sid))  # drive the grid to the clicked sample

    def _resolve(self, ctx, decision):
        h = self._row_hash(ctx)
        if not h:
            return
        cfg, ad = Config(), _adapter(ctx)
        pipeline.resolve_review(ad, cfg, h, decision, reviewer_id=ctx.user_id or "reviewer")
        self.on_load(ctx)  # resolved item drops out of the queue (review_queue excludes golden/rejected)
        ctx.ops.reload_dataset()

    def on_confirm(self, ctx):
        self._resolve(ctx, "confirm")

    def on_dismiss(self, ctx):
        self._resolve(ctx, "false_alarm")

    def render(self, ctx):
        panel = types.Object()
        if ctx.panel.state.err:
            panel.md(f"佇列產生失敗:`{ctx.panel.state.err}`\n\n需先 `vix calibrate` + `vix route`。", name="qerr")
        table = types.TableView()
        table.add_column("risk", label="風險")
        table.add_column("id", label="vix_hash")
        table.add_column("why", label="原因(proxy)")
        table.add_row_action("inspect", self.on_inspect, label="看圖", icon="visibility")
        table.add_row_action("confirm", self.on_confirm, label="確認→golden", icon="check")
        table.add_row_action("dismiss", self.on_dismiss, label="誤報排除", icon="block")
        panel.list("rows", types.Object(), view=table)
        panel.btn("refresh", label="重新整理佇列", on_click=self.on_refresh)
        return types.Property(panel, view=types.GridView(height=100, width=100))


def register(p):
    p.register(ConfirmGolden)
    p.register(DismissFalseAlarm)
    p.register(ExplainSample)
    p.register(VixReportPanel)
    p.register(VixQueuePanel)
