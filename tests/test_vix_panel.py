"""Tier-2: the in-App VIX report Panel loads + registers correctly under FiftyOne 1.16's Panel API.
Skips cleanly without fiftyone. The live browser render is verified by `vix verify-gui` (needs a
running App); this catches Panel-API / registration regressions offline."""

from pathlib import Path

import pytest

pytest.importorskip("fiftyone")


def _load_plugin():
    import importlib.util
    path = Path(__file__).resolve().parent.parent / "src" / "vix" / "plugins" / "vix_review" / "__init__.py"
    spec = importlib.util.spec_from_file_location("vix_review_plugin", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_report_panel_config():
    m = _load_plugin()
    panel = m.VixReportPanel()
    assert panel.config.name == "vix_report" and panel.config.surfaces == "grid"


def test_report_panel_render_schema_builds():
    import fiftyone.operators.types as types
    m = _load_plugin()
    panel = m.VixReportPanel()
    # build the render schema the way render() does — catches view/kwarg API drift without an App
    obj = types.Object()
    obj.md("# report", name="report")
    obj.btn("regen", label="x", on_click=panel.on_regen)
    prop = types.Property(obj, view=types.GridView(height=100, width=100))
    assert prop is not None


def test_plugin_registers_panel():
    m = _load_plugin()
    registered = []

    class _Collector:
        def register(self, cls):
            registered.append(cls.__name__)

    m.register(_Collector())
    assert "VixReportPanel" in registered and "ConfirmGolden" in registered
    assert "VixQueuePanel" in registered  # P1: the clickable review-queue panel


def test_queue_panel_config():
    m = _load_plugin()
    panel = m.VixQueuePanel()
    assert panel.config.name == "vix_queue" and panel.config.surfaces == "grid"


def test_queue_panel_render_schema_builds_with_row_actions():
    import fiftyone.operators.types as types
    m = _load_plugin()
    panel = m.VixQueuePanel()
    # build the TableView render schema the way render() does — catches row-action/list API drift offline
    obj = types.Object()
    table = types.TableView()
    table.add_column("risk", label="風險")
    table.add_column("id", label="vix_hash")
    table.add_row_action("inspect", panel.on_inspect, label="看圖", icon="visibility")
    table.add_row_action("confirm", panel.on_confirm, label="確認", icon="check")
    obj.list("rows", types.Object(), view=table)
    js = types.Property(obj, view=types.GridView(height=100, width=100)).to_json()
    actions = js["type"]["properties"]["rows"]["view"]["row_actions"]
    assert [a["name"] for a in actions] == ["inspect", "confirm"]
