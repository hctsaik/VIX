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

from pathlib import Path

import fiftyone.operators as foo
import fiftyone.operators.types as types

from vix import pipeline
from vix.adapters.fiftyone_adapter import FiftyOneAdapter
from vix.config import Config


def _adapter(ctx):
    return FiftyOneAdapter(Config(), dataset_name=ctx.dataset.name)


def _selected_hashes(ctx):
    # skip selected samples that carry no vix_hash (e.g. added outside VIX) instead of KeyError-crashing
    out = []
    for sid in (ctx.selected or []):
        try:
            h = ctx.dataset[sid].get_field("vix_hash")
        except Exception:  # noqa: BLE001 - a non-VIX / vanished sample must not crash the operator
            h = None
        if h:
            out.append(h)
    return out


def _sample_id_for_hash(ctx, h):
    """vix_hash -> FiftyOne sample id (inverse of _selected_hashes). The one bit of live-only glue
    the queue panel needs to navigate; kept tiny so it's the obvious thing to find if FiftyOne drifts.
    Returns None for a vanished/unknown hash (.first() raises on an empty view) so inspect no-ops."""
    try:
        return ctx.dataset.match({"vix_hash": h}).first().id
    except Exception:  # noqa: BLE001 - empty match / vanished sample -> navigate nowhere, never crash
        return None


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
        hashes = _selected_hashes(ctx)
        if not hashes:  # parity with explain_sample: a friendly message, never a phantom 0-write
            return {"error": "請先在格狀檢視選取影像"}
        for h in hashes:
            pipeline.resolve_review(ad, cfg, h, "confirm", label, reviewer_id=ctx.user_id or "reviewer")
        ctx.ops.reload_dataset()
        return {"confirmed": len(hashes)}


class DismissFalseAlarm(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(name="dismiss_false_alarm", label="VIX: 標記誤報並排除", dynamic=True)

    def execute(self, ctx):
        cfg, ad = Config(), _adapter(ctx)
        hashes = _selected_hashes(ctx)
        if not hashes:
            return {"error": "請先在格狀檢視選取影像"}
        for h in hashes:
            pipeline.resolve_review(ad, cfg, h, "false_alarm", reviewer_id=ctx.user_id or "reviewer")
        ctx.ops.reload_dataset()
        return {"dismissed": len(hashes)}


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
        try:  # tag vixq:* so saved views become clickable — must not crash on a no-golden dataset
            pipeline.weakness_report(ad, cfg, worklist=True)
        except Exception as exc:  # noqa: BLE001
            ctx.panel.state.md = (f"# VIX 弱點報告\n\n標記工作清單失敗:`{exc}`\n\n"
                                  "需先有 golden,並(選用)`vix eval-ingest <val.jsonl>`。")
            return
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
        try:  # a stale/unknown row (resolve_review fail-closes via _require_known) must not crash the panel
            pipeline.resolve_review(ad, cfg, h, decision, reviewer_id=ctx.user_id or "reviewer")
        except Exception as exc:  # noqa: BLE001
            ctx.panel.state.err = f"此列無法處理({exc});請按「重新整理佇列」"
            return
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


class GenerateWeaknessReport(foo.Operator):
    """Pick an eval file in the App and generate the model-weakness report — the GUI equivalent of
    `vix eval-ingest <val.jsonl>` + `vix weakness-report`. Zero new logic: calls the tested pipeline.*
    and the existing vix_report panel renders the result."""

    @property
    def config(self):
        return foo.OperatorConfig(name="generate_weakness_report", label="VIX: 產生模型弱點報告(選 eval)",
                                  dynamic=True)

    def _candidates(self):
        cfg = Config()
        cands: set[str] = set()
        for d in (cfg.workspace, Path.cwd()):
            try:
                cands |= {str(p) for p in Path(d).glob("*.jsonl")}
            except Exception:  # noqa: BLE001
                pass
        return sorted(cands)

    def resolve_input(self, ctx):
        inputs = types.Object()
        cands = self._candidates()
        if cands:
            inputs.enum("eval_file", cands, label="選一個 eval JSONL(每行 {vix_hash, gt, pred})",
                        view=types.DropdownView())
        inputs.str("custom_path", required=False, label="或自訂 eval JSONL 路徑(優先)")
        return types.Property(inputs, view=types.View(label="產生模型弱點報告(eval-ingest → weakness-report)"))

    def execute(self, ctx):
        cfg, ad = Config(), _adapter(ctx)
        path = (ctx.params.get("custom_path") or "").strip() or ctx.params.get("eval_file")
        if not path or not Path(path).exists():
            return {"error": f"找不到 eval 檔:{path!r}(需每行 {{vix_hash, gt, pred}} 的 JSONL)"}
        try:
            ev = pipeline.eval_ingest(ad, cfg, path)            # writes eval_results.json (tested)
            wr = pipeline.weakness_report(ad, cfg)["data"]       # writes weakness_report.md/.html (tested)
        except Exception as exc:  # noqa: BLE001
            return {"error": f"產生失敗:{exc}"}
        try:
            ctx.ops.open_panel("vix_report")  # surface the report; harmless if unsupported
        except Exception:  # noqa: BLE001
            pass
        return {"mAP": ev.get("mAP"), "health": wr["summary"]["health"],
                "weakest": wr["summary"].get("weakest") or "-",
                "report": str(cfg.workspace / "weakness_report.md")}

    def resolve_output(self, ctx):
        out = types.Object()
        out.str("error", label="錯誤")
        out.float("mAP", label="mAP@0.5")
        out.str("health", label="健康度")
        out.str("weakest", label="最弱類別")
        out.str("report", label="報告檔(或開「VIX: 弱點/一致性報告」面板)")
        return types.Property(out)


class FlagLabelIssues(foo.Operator):
    """One click: surface which golden images have likely-INACCURATE labels — suspected wrong class
    (embedding-neighbour disagreement) and bad box geometry (degenerate/truncated/outlier) — by tagging
    them vixq:label_suspect / vixq:box_issue so they become a filterable, clickable worklist in the App.
    Zero new logic: calls the tested audit_labels + box_qa. Honest: these are PROXY suspicions to review,
    never auto-edits. (Pixel-level box-tightness needs an optional SAM add-on — not included.)"""

    @property
    def config(self):
        return foo.OperatorConfig(name="flag_label_issues", label="VIX: 標出疑似不準的標註", dynamic=True)

    def execute(self, ctx):
        cfg, ad = Config(), _adapter(ctx)

        def _imgs(items):
            out = set()
            for it in items or []:
                i = it.get("id") if isinstance(it, dict) else getattr(it, "id", None)
                if i:
                    out.add(str(i).split(":")[0])  # box/label ids are "<vix_hash>[:idx]"
            return out
        try:
            label_susp = _imgs(pipeline.audit_labels(ad, cfg))   # suspected wrong CLASS (kNN disagreement)
            box_susp = _imgs(pipeline.box_qa(ad, cfg))            # suspected bad BOX geometry
        except Exception as exc:  # noqa: BLE001
            return {"error": f"分析失敗:{exc}"}
        for h in label_susp:
            try:
                ad.apply_tags(h, ["vixq:label_suspect"])
            except Exception:  # noqa: BLE001
                pass
        for h in box_susp:
            try:
                ad.apply_tags(h, ["vixq:box_issue"])
            except Exception:  # noqa: BLE001
                pass
        ctx.ops.reload_dataset()
        return {"label_suspect": len(label_susp), "box_issue": len(box_susp),
                "hint": "在 App 用 sample tags / saved view 篩 vixq:label_suspect、vixq:box_issue 逐張檢查並修正"}

    def resolve_output(self, ctx):
        out = types.Object()
        out.str("error", label="錯誤")
        out.int("label_suspect", label="疑似標錯類別(張)")
        out.int("box_issue", label="框幾何問題(張)")
        out.str("hint", label="怎麼看")
        return types.Property(out)


def register(p):
    p.register(ConfirmGolden)
    p.register(DismissFalseAlarm)
    p.register(ExplainSample)
    p.register(GenerateWeaknessReport)
    p.register(FlagLabelIssues)
    p.register(VixReportPanel)
    p.register(VixQueuePanel)
