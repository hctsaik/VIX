"""Round 5: in-report per-class before/after (Δ) — shown ONLY when the eval set is unchanged
(eval_set_hash includes GT, so a relabelled eval set is correctly NOT comparable -> no Δ)."""

import sys
import types

from vix.adapters.memory import InMemoryAdapter
from vix.config import Config
from vix.pipeline import _report_provenance, diagnose, eval_ingest, weakness_report
from vix.types import BBox, Detection, Tag

_PNG = (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
        b"\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82")


def _eval_imgs(ap_good):
    # one image, one GT 'a'; pred matches when ap_good else misses -> AP 1.0 vs 0.0, SAME gt -> same eval_set_hash
    pred = [{"label": "a", "bbox": [0.5, 0.5, 0.2, 0.2], "conf": 0.9}] if ap_good else []
    return [{"vix_hash": "h1", "gt": [{"label": "a", "bbox": [0.5, 0.5, 0.2, 0.2]}], "pred": pred}]


def test_provenance_carries_prev_per_class_when_comparable(tmp_path):
    cfg = Config(workspace=tmp_path / "ws"); cfg.ensure_dirs()
    ad = InMemoryAdapter(); ad.seed("h1", "h1.png", [Detection("a", 0.9, BBox(0.5, 0.5, 0.2, 0.2))], tags=[Tag.EVAL])
    eval_ingest(ad, cfg, _eval_imgs(False))   # cycle 1: AP 0.0 (same eval set)
    eval_ingest(ad, cfg, _eval_imgs(True))    # cycle 2: AP 1.0 (same GT -> comparable)
    cur_hash = json_hash = __import__("json").loads(cfg.eval_results_path.read_text(encoding="utf-8"))["eval_set_hash"]
    prov = _report_provenance(cfg, cur_hash)
    assert prov["comparable"] is True
    assert prov["prev_per_class_ap"] == {"a": 0.0}  # the prior comparable run's per-class AP


def test_no_delta_when_eval_set_changed(tmp_path):
    cfg = Config(workspace=tmp_path / "ws"); cfg.ensure_dirs()
    ad = InMemoryAdapter(); ad.seed("h1", "h1.png", [Detection("a", 0.9, BBox(0.5, 0.5, 0.2, 0.2))], tags=[Tag.EVAL])
    eval_ingest(ad, cfg, _eval_imgs(True))
    # change the GT (relabel) -> different eval_set_hash -> must NOT be comparable
    changed = [{"vix_hash": "h1", "gt": [{"label": "b", "bbox": [0.5, 0.5, 0.2, 0.2]}],
                "pred": [{"label": "b", "bbox": [0.5, 0.5, 0.2, 0.2], "conf": 0.9}]}]
    eval_ingest(ad, cfg, changed)
    cur_hash = __import__("json").loads(cfg.eval_results_path.read_text(encoding="utf-8"))["eval_set_hash"]
    prov = _report_provenance(cfg, cur_hash)
    assert prov["comparable"] is False
    assert prov["prev_per_class_ap"] is None  # eval set changed -> no dishonest delta


def _ds(tmp_path, label):
    (tmp_path / "images").mkdir(exist_ok=True); (tmp_path / "labels").mkdir(exist_ok=True)
    (tmp_path / "images" / "a.png").write_bytes(_PNG)
    (tmp_path / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    return tmp_path


def test_report_shows_delta_column_across_comparable_diagnose_runs(tmp_path, monkeypatch):
    folder = _ds(tmp_path, "pothole")
    cfg = Config(workspace=tmp_path / "ws"); cfg.ensure_dirs()
    ad = InMemoryAdapter()

    def fake_yolo(rows):
        mod = types.ModuleType("ultralytics")

        class _V(list):
            def tolist(self):
                return list(self)

        class _B:
            def __init__(s):
                s._r = rows

            def __len__(s):
                return len(s._r)

            @property
            def cls(s):
                return [r[0] for r in s._r]

            @property
            def conf(s):
                return [r[1] for r in s._r]

            @property
            def xywhn(s):
                return [_V(r[2]) for r in s._r]

        r = types.SimpleNamespace(boxes=_B(), names={0: "pothole"})
        mod.YOLO = lambda *a, **k: types.SimpleNamespace(predict=lambda src, **kw: [r])
        monkeypatch.setitem(sys.modules, "ultralytics", mod)

    # cycle 1: model misses -> AP 0
    fake_yolo([])
    diagnose(ad, cfg, folder, labels_fmt="yolo", weights="m.pt", names=["pothole"],
             out_path=cfg.workspace / "wr.md")
    # cycle 2: model now hits -> AP 1 (GT unchanged -> comparable -> Δ shown)
    fake_yolo([(0, 0.9, [0.5, 0.5, 0.2, 0.2])])
    diagnose(ad, cfg, folder, labels_fmt="yolo", weights="m.pt", names=["pothole"],
             out_path=cfg.workspace / "wr.md")
    md = (cfg.workspace / "wr.md").read_text(encoding="utf-8")
    # before/after delta rendered; this class has n_gt=1 (<min_support) so it must be hedged, NOT a bare ↑
    assert "Δ(同 eval set)" in md and "n少不穩" in md
    assert "+1.0 ↑" not in md  # a tiny-support swing must not look like a confident gain


def test_delta_cell_arrow_only_when_supported():
    from vix.core.weakness_report import _delta_cell, _MIN_SUPPORT
    assert _delta_cell(0.3, _MIN_SUPPORT + 5) == "+0.3 ↑"        # enough support -> confident arrow
    assert _delta_cell(-0.2, _MIN_SUPPORT + 5) == "-0.2 ↓"
    assert "↑" not in _delta_cell(0.3, 3) and "n少不穩" in _delta_cell(0.3, 3)  # tiny support -> hedged, no arrow
    assert _delta_cell(None, 999) == "-"
