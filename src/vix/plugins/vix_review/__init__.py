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


def register(p):
    p.register(ConfirmGolden)
    p.register(DismissFalseAlarm)
    p.register(ExplainSample)
