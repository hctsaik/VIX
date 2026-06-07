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
    panel_path = cfg.workspace / "weakness_report_panel.md"  # compact, panel-optimized layout
    if regenerate or not panel_path.exists():
        try:
            pipeline.weakness_report(ad, cfg)
        except Exception as exc:  # noqa: BLE001 - surface the reason in-panel rather than crash the App
            return f"### VIX 弱點報告\n\n產生失敗:`{exc}`\n\n需先有 golden,並(選用)`vix eval-ingest <val.jsonl>`。"
    return panel_path.read_text(encoding="utf-8") if panel_path.exists() else "### VIX 弱點報告\n\n(尚無報告)"


def _panel_nav(ctx):
    """Navigable rows (confident_wrong / overturns) for the report panel's CLICKABLE tables — written
    by pipeline.weakness_report alongside the panel .md. Each row carries a readable `file` to show and
    the `hash` to jump to. Empty (table hidden) for an old report with no sidecar."""
    import json
    p = Config().workspace / "weakness_report_panel.json"
    if not p.exists():
        return {"confident_wrong": [], "overturns": []}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - a malformed/partial sidecar must not crash the panel
        return {"confident_wrong": [], "overturns": []}


class VixReportPanel(foo.Panel):
    """In-App panel surfacing the VIX weakness/consistency report (Tier 2 GUI). Pure presentation over
    pipeline.weakness_report — same audit-logged core the CLI uses. The confident-wrong / overturn rows
    are rendered as CLICKABLE tables (point『看圖』jumps the grid to that image) so a filename in the
    report links straight back to its picture; zero logic here (nav reuses _sample_id_for_hash)."""

    @property
    def config(self):
        return foo.PanelConfig(name="vix_report", label="VIX: 弱點/一致性報告", surfaces="grid")

    def _refresh(self, ctx, regenerate=False):
        ctx.panel.state.md = _report_md(ctx, regenerate=regenerate)
        nav = _panel_nav(ctx)
        cw, ov = nav.get("confident_wrong") or [], nav.get("overturns") or []
        ctx.panel.state.cw = cw
        ctx.panel.data.cw = cw       # TableView binds to the data path
        ctx.panel.state.ov = ov
        ctx.panel.data.ov = ov

    def on_load(self, ctx):
        self._refresh(ctx)

    def on_regen(self, ctx):
        self._refresh(ctx, regenerate=True)

    def on_worklist(self, ctx):
        cfg, ad = Config(), _adapter(ctx)
        try:  # tag vixq:* so saved views become clickable — must not crash on a no-golden dataset
            pipeline.weakness_report(ad, cfg, worklist=True)
        except Exception as exc:  # noqa: BLE001
            ctx.panel.state.md = (f"# VIX 弱點報告\n\n標記工作清單失敗:`{exc}`\n\n"
                                  "需先有 golden,並(選用)`vix eval-ingest <val.jsonl>`。")
            return
        self._refresh(ctx)
        ctx.ops.reload_dataset()

    def _jump(self, ctx, rows):
        """Open the clicked row's image (filename -> picture). Uses open_sample so the picture pops in
        the App's sample modal OVER whatever you're looking at — set_view only filtered the grid behind
        this panel (you're on the panel tab, so you'd see nothing change)."""
        idx = ctx.params.get("row")
        if isinstance(idx, int) and 0 <= idx < len(rows):
            sid = _sample_id_for_hash(ctx, rows[idx].get("hash"))
            if sid:
                ctx.ops.open_sample(id=sid)

    def on_inspect_cw(self, ctx):
        self._jump(ctx, ctx.panel.state.cw or [])

    def on_inspect_ov(self, ctx):
        self._jump(ctx, ctx.panel.state.ov or [])

    def render(self, ctx):
        panel = types.Object()
        panel.md(ctx.panel.state.md or "_載入中…_", name="report")
        cw = ctx.panel.state.cw or []
        if cw:
            t = types.TableView()
            t.add_column("file", label="影像")
            t.add_column("pred_class", label="類別")
            t.add_column("conf", label="信心")
            t.add_column("fp_type", label="型態")
            t.add_row_action("inspect", self.on_inspect_cw, label="看圖", icon="visibility")
            panel.list("cw", types.Object(), view=t, label="最自信卻錯(點『看圖』跳到該張)")
        ov = ctx.panel.state.ov or []
        if ov:
            t2 = types.TableView()
            t2.add_column("file", label="影像")
            t2.add_column("pred_class", label="類別")
            t2.add_column("conf", label="信心")
            t2.add_column("wrongness", label="可疑度")
            t2.add_row_action("inspect", self.on_inspect_ov, label="看圖", icon="visibility")
            panel.list("ov", types.Object(), view=t2, label="高信心但長得不像該類(點『看圖』跳到該張)")
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
            ctx.ops.open_sample(id=sid)  # pop the image in the App's sample modal (works from any tab)

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


class AuditLabelErrors(foo.Operator):
    """DINO cross-class label-error audit: lists samples whose DINOv2 kNN-majority label disagrees with
    their given label ("標成 X 但鄰居多為 Y"), tags them vixq:label_error, and shows the suggested class.
    Needs >=2 classes + computed embeddings (run `vix embed` first). PROXY — review, never auto-relabel."""

    @property
    def config(self):
        return foo.OperatorConfig(name="audit_label_errors", label="VIX: 找標錯的類別(DINO)", dynamic=True)

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.int("top", default=50, label="最多列出幾筆")
        return types.Property(inputs, view=types.View(label="DINO 跨類標錯稽核(需先 vix embed,≥2 類)"))

    def execute(self, ctx):
        cfg, ad = Config(), _adapter(ctx)
        try:
            issues = pipeline.audit_labels(ad, cfg)
        except Exception as exc:  # noqa: BLE001
            return {"error": f"稽核失敗(先 vix embed 計算嵌入):{exc}"}
        for i in issues:
            try:
                ad.apply_tags(i.id.split(":")[0], ["vixq:label_error"])
            except Exception:  # noqa: BLE001
                pass
        rows = [{"id": i.id.split(":")[0], "given": i.given_label, "suggested": i.suggested_label,
                 "disagreement": round(i.disagreement, 2)} for i in issues[:int(ctx.params.get("top") or 50)]]
        return {"n": len(issues), "rows": rows,
                "hint": "vixq:label_error 已標記;PROXY,逐筆覆核是否真的標錯類別,勿自動改標" if rows
                else "未發現跨類標錯(或單一類別 / 尚未 vix embed)"}

    def resolve_output(self, ctx):
        out = types.Object()
        out.str("error", label="錯誤")
        out.int("n", label="疑似標錯類別數")
        tbl = types.TableView()
        tbl.add_column("id", label="樣本")
        tbl.add_column("given", label="目前標成")
        tbl.add_column("suggested", label="DINO 建議(鄰居多為)")
        tbl.add_column("disagreement", label="不一致度")
        out.list("rows", types.Object(), view=tbl, label="標成 X → 建議 Y")
        out.str("hint", label="怎麼看")
        return types.Property(out)


class FlagLooseBoxes(foo.Operator):
    """Opt-in PIXEL-level box-tightness audit (the one check box_qa structurally can't do): prompts a
    SAM mask per golden box and flags boxes whose GT doesn't hug the object (low IoU) as vixq:loose_box.
    Needs ultralytics SAM (one-time weights download); SAM is ~1s/box on CPU so it samples. PROXY (the
    mask is itself a model's guess) — tags suspects to review, never auto-edits the boxes."""

    @property
    def config(self):
        return foo.OperatorConfig(name="flag_loose_boxes", label="VIX: 標出太鬆的框(SAM,選用)", dynamic=True)

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.int("limit", default=40, label="抽樣張數(SAM 較慢,~1s/框)")
        inputs.float("iou_thr", default=0.6, label="IoU 門檻(GT 框與物件遮罩低於此=太鬆)")
        return types.Property(inputs, view=types.View(label="SAM 框緊度稽核(需 ultralytics SAM 權重)"))

    def execute(self, ctx):
        cfg, ad = Config(), _adapter(ctx)
        try:
            loose = pipeline.box_tightness(ad, cfg, limit=int(ctx.params.get("limit") or 40),
                                           iou_thr=float(ctx.params.get("iou_thr") or 0.6))
        except Exception as exc:  # noqa: BLE001 - missing SAM weights / deps surface as a friendly message
            return {"error": f"需要 ultralytics SAM:{exc}"}
        ids = {it["id"] for it in loose}
        for h in ids:
            try:
                ad.apply_tags(h, ["vixq:loose_box"])
            except Exception:  # noqa: BLE001
                pass
        ctx.ops.reload_dataset()
        return {"loose_boxes": len(loose), "images": len(ids),
                "hint": "篩 vixq:loose_box 逐張檢查並收緊框;PROXY(SAM 也是猜的),勿自動改框"}

    def resolve_output(self, ctx):
        out = types.Object()
        out.str("error", label="錯誤")
        out.int("loose_boxes", label="疑似太鬆的框")
        out.int("images", label="影像數")
        out.str("hint", label="怎麼看")
        return types.Property(out)


class BuildSimilarity(foo.Operator):
    """Build the OBJECT-BOX (patch) similarity index so the App's native sort-by-similarity works on
    YOUR boxes: pick a box → the magnifying glass → the whole dataset re-ranks by how that OBJECT looks
    (DINOv2 crop embeddings, sklearn exact-NN — offline, no FiftyOne Enterprise). Object-level, not
    whole-scene, so it finds similar *defects*, not just similar backgrounds. Zero new logic: reuses the
    same DINO embeddings VIX already computes; only wraps fob.compute_similarity(patches_field=...)."""

    @property
    def config(self):
        return foo.OperatorConfig(name="build_similarity",
                                  label="VIX: 建立相似搜尋索引(DINO,物件框)", dynamic=True)

    def resolve_placement(self, ctx):
        return types.Placement(
            types.Places.SAMPLES_GRID_ACTIONS,
            types.Button(label="VIX: 建立相似搜尋索引", icon="/assets/simindex.svg", prompt=True),
        )

    def resolve_input(self, ctx):
        ad = _adapter(ctx)
        try:
            have = ad.has_embeddings()
        except Exception:  # noqa: BLE001
            have = False
        inputs = types.Object()
        if have:
            inputs.view("info", types.Notice(label="偵測框已有 DINO 嵌入 → 直接建索引(快)。"))
        else:
            try:
                from vix.embedding.dinov2_torch import device_report
                dev = device_report()
            except Exception:  # noqa: BLE001
                dev = "將自動偵測加速硬體(CUDA/MPS/CPU)"
            inputs.view("warn", types.Notice(
                label=f"偵測框尚無 DINO 嵌入,會先對每個框算 DINOv2。{dev}。"
                      "GPU 上很快;純 CPU 整個資料集可能數分鐘。完成後即可用原生『放大鏡』排相似。"))
        return types.Property(inputs, view=types.View(label="建立相似搜尋索引(物件框 / DINOv2)"))

    def execute(self, ctx):
        cfg, ad = Config(), _adapter(ctx)
        try:
            if not ad.has_embeddings():        # one-time: embed every box crop with DINOv2 (real, offline)
                ad.compute_embeddings(cfg.dinov2_model_key)
            brain_key = ad.build_patch_similarity()   # sklearn exact-NN over the crop embeddings
        except Exception as exc:  # noqa: BLE001 - missing deps / no detections -> friendly message
            return {"error": f"建立失敗:{exc}"}
        try:
            ctx.ops.reload_dataset()
            ctx.ops.notify("相似搜尋索引完成。選一張有框的影像(或在展開圖選一個 label)→ 點工具列的"
                           "🔎『找相似物件』按鈕,就用你的 DINO 排出最像的物件(不需 Enterprise)。", variant="success")
        except Exception:  # noqa: BLE001
            pass
        return {"brain_key": brain_key,
                "hint": "用法:選一張有框的影像 → 工具列『VIX: 找相似物件』→ App 切到用你 DINO 排序的相似物件。"
                        "這是物件級(非整張圖),找的是長得像的瑕疵;不要用內建的『Similarity Search』面板(那是 Enterprise)。"}

    def resolve_output(self, ctx):
        out = types.Object()
        r = ctx.results or {}
        if r.get("error"):
            out.str("error", label="錯誤")
            return types.Property(out)
        out.str("brain_key", label="索引名稱(brain key)")
        out.str("hint", label="怎麼用")
        return types.Property(out)


class FindSimilar(foo.Operator):
    """Find-similar using YOUR DINO index — the OSS replacement for the App's Enterprise-gated
    'Similarity Search' panel. Select an image (or a label) and this re-views the dataset as the
    object PATCHES sorted by similarity to that object, via the vix_patch_sim index (sklearn exact-NN
    over DINOv2 crops). No Enterprise, no zoo model, fully offline. Zero ranking logic here — it just
    calls FiftyOne's sort_by_similarity on the index BuildSimilarity created and drives the view."""

    @property
    def config(self):
        return foo.OperatorConfig(name="find_similar", label="VIX: 找相似物件(我的 DINO)", dynamic=True)

    def resolve_placement(self, ctx):
        return types.Placement(
            types.Places.SAMPLES_GRID_ACTIONS,
            types.Button(label="VIX: 找相似物件", icon="/assets/similar.svg", prompt=False),
        )

    def _query_label_id(self, ctx):
        """The patch (label) id to query by: a selected label wins; else the highest-confidence box of a
        selected sample. Returns None with a reason for a friendly message."""
        for sl in (getattr(ctx, "selected_labels", None) or []):  # user selected a specific box in the expanded view
            lid = sl.get("label_id") or sl.get("id")
            if lid:
                return lid, None
        if ctx.selected:  # a sample is selected in the grid -> use its most-confident detection
            try:
                s = ctx.dataset[ctx.selected[0]]
                dets = s["yolo_detections"].detections if s["yolo_detections"] else []
            except Exception:  # noqa: BLE001
                dets = []
            if dets:
                return max(dets, key=lambda d: d.confidence or 0.0).id, None
            return None, "選取的影像沒有偵測框"
        return None, "請先在格狀檢視選一張有框的影像(或在展開圖選一個 label)"

    def execute(self, ctx):
        ds = ctx.dataset
        k = int(ctx.params.get("k") or 50)
        try:
            has_index = "vix_patch_sim" in ds.list_brain_runs()
        except Exception:  # noqa: BLE001
            has_index = False
        if not has_index:
            return {"error": "尚未建立索引 — 請先點工具列的『VIX: 建立相似搜尋索引』。"}
        qid, why = self._query_label_id(ctx)
        if not qid:
            return {"error": why}
        try:
            view = ds.to_patches("yolo_detections").sort_by_similarity(qid, k=k, brain_key="vix_patch_sim")
            ctx.ops.set_view(view=view)  # drive the App to the DINO-sorted object patches (no Enterprise)
        except Exception as exc:  # noqa: BLE001 - stale id / index mismatch -> friendly message
            return {"error": f"找相似失敗:{exc}"}
        return {"shown": len(view),
                "hint": "已切到『用你的 DINO 排序的相似物件』檢視(最像的在前)。要還原:清掉檢視列的階段。"}

    def resolve_output(self, ctx):
        out = types.Object()
        r = ctx.results or {}
        if r.get("error"):
            out.str("error", label="錯誤")
            return types.Property(out)
        out.int("shown", label="顯示相似物件數")
        out.str("hint", label="說明")
        return types.Property(out)


class LoadDataset(foo.Operator):
    """Load YOUR dataset from the App: point at a folder of images + their YOLO/VOC/COCO labels
    (and optionally your model .pt), and VIX imports them into a FiftyOne dataset and switches to it.
    GUI equivalent of `vix diagnose` / `vix import-labels` — same tested pipeline.* functions, same
    honesty: imported labels are tagged an UNVERIFIED reference (provisional), never golden."""

    @property
    def config(self):
        return foo.OperatorConfig(name="load_dataset", label="VIX: 載入我的資料集(影像+標籤)", dynamic=True)

    def resolve_placement(self, ctx):
        # A visible button in the grid toolbar (no need to press ` and search). Clicking opens the form.
        return types.Placement(
            types.Places.SAMPLES_GRID_ACTIONS,
            types.Button(label="VIX: 載入我的資料集", icon="/assets/folder.svg", prompt=True),
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.str("folder", required=True, label="影像資料夾路徑",
                   description="貼上資料夾路徑即可 — VIX 會自動判斷 YOLO / VOC / COCO 與類別名稱")
        inputs.str("weights", required=False, label="(選填)你的模型 .pt — 填了就順便跑模型弱點評估")
        return types.Property(inputs, view=types.View(label="載入我的資料集(只要資料夾,自動判斷格式)"))

    def execute(self, ctx):
        folder = (ctx.params.get("folder") or "").strip().strip('"')
        if not folder or not Path(folder).exists():
            return {"error": f"找不到資料夾:{folder!r}"}
        weights = (ctx.params.get("weights") or "").strip().strip('"') or None
        name = "".join(c if (c.isalnum() or c in "-_") else "_" for c in Path(folder).name) or "vix"
        cfg, ad = Config(), FiftyOneAdapter(Config(), dataset_name=name)
        try:
            if weights:  # Tier A: auto-detect format/names + run YOUR model + write the weakness report
                res = pipeline.diagnose(ad, cfg, folder, labels_fmt="auto", weights=weights)
                imp = res["import"]
            else:        # just load the labels so you can browse them (format auto-detected)
                imp = pipeline.import_labels(ad, cfg, folder, fmt="auto")
        except Exception as exc:  # noqa: BLE001 - bad path / no labels found -> friendly message
            return {"error": f"載入失敗:{exc}"}
        try:
            ctx.ops.open_dataset(name)             # switch the App to the freshly loaded dataset
            ctx.ops.notify(f"已載入 {imp['n_images']} 張 / {imp['n_boxes']} 框 → dataset『{name}』", variant="success")
            if weights:
                ctx.ops.open_panel("vix_report")  # show the weakness report panel
        except Exception:  # noqa: BLE001
            pass
        return {"dataset": name, "n_images": imp["n_images"], "n_boxes": imp["n_boxes"],
                "classes": ", ".join(imp.get("classes", [])) or "-",
                "hint": f"已切到 dataset『{name}』。匯入標籤為未覆核參照(非 golden)。"
                        + ("已開弱點報告面板。" if weights else "想跑模型評估?重跑並填 .pt 路徑。")}

    def resolve_output(self, ctx):
        out = types.Object()
        r = ctx.results or {}
        if r.get("error"):  # only show fields that have a value (avoids "No value provided" on success)
            out.str("error", label="錯誤")
            return types.Property(out)
        out.str("dataset", label="載入到 dataset")
        out.int("n_images", label="影像數")
        out.int("n_boxes", label="標註框數")
        out.str("classes", label="類別")
        out.str("hint", label="提示")
        return types.Property(out)


class DeleteDataset(foo.Operator):
    """Delete the CURRENTLY OPEN FiftyOne dataset from the App (symmetric to LoadDataset). Requires an
    explicit confirm checkbox. Only removes the FiftyOne/Mongo dataset records — your image and label
    files on disk are never touched (VIX is read-only over your data). After delete, switches to another
    remaining dataset."""

    @property
    def config(self):
        return foo.OperatorConfig(name="delete_dataset", label="VIX: 刪除目前的 dataset", dynamic=True)

    def resolve_placement(self, ctx):
        return types.Placement(
            types.Places.SAMPLES_GRID_ACTIONS,
            types.Button(label="VIX: 刪除目前的 dataset", icon="/assets/trash.svg", prompt=True),
        )

    def resolve_input(self, ctx):
        name = ctx.dataset.name if ctx.dataset else ""
        inputs = types.Object()
        inputs.view("warn", types.Notice(
            label=f"將永久刪除 dataset『{name}』。只刪 FiftyOne 記錄,你的影像/標籤檔不受影響(可重新載入)。"))
        inputs.bool("confirm", label=f"我確定要刪除『{name}』", default=False, required=True)
        return types.Property(inputs, view=types.View(label=f"刪除 dataset『{name}』"))

    def execute(self, ctx):
        import fiftyone as fo
        name = ctx.dataset.name if ctx.dataset else None
        if not name:
            return {"error": "目前沒有開啟的 dataset"}
        if not ctx.params.get("confirm"):
            return {"error": "未確認:請勾選「我確定要刪除」再執行"}
        try:
            fo.delete_dataset(name)
        except Exception as exc:  # noqa: BLE001
            return {"error": f"刪除失敗:{exc}"}
        remaining = list(fo.list_datasets())
        try:
            if remaining:
                ctx.ops.open_dataset(remaining[0])
            ctx.ops.notify(f"已刪除 dataset『{name}』(你的檔案不受影響)", variant="success")
        except Exception:  # noqa: BLE001
            pass
        return {"deleted": name, "switched_to": (remaining[0] if remaining else "(已無其他 dataset)"),
                "note": "只刪 FiftyOne 記錄;磁碟上的影像/標籤未更動。"}

    def resolve_output(self, ctx):
        out = types.Object()
        r = ctx.results or {}
        if r.get("error"):
            out.str("error", label="錯誤")
            return types.Property(out)
        out.str("deleted", label="已刪除")
        out.str("switched_to", label="已切換到")
        out.str("note", label="說明")
        return types.Property(out)


def register(p):
    p.register(LoadDataset)
    p.register(DeleteDataset)
    p.register(BuildSimilarity)
    p.register(FindSimilar)
    p.register(ConfirmGolden)
    p.register(DismissFalseAlarm)
    p.register(ExplainSample)
    p.register(GenerateWeaknessReport)
    p.register(FlagLabelIssues)
    p.register(AuditLabelErrors)
    p.register(FlagLooseBoxes)
    p.register(VixReportPanel)
    p.register(VixQueuePanel)
