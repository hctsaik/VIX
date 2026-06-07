"""eval-run + diagnose orchestrator. Fake YOLO (no torch/ultralytics, no FiftyOne, no DINOv2):
proves the content-hash join (KS3), Tier-A runs offline, and the honesty banner is emitted."""

import sys
import types

import pytest

from vix.adapters.memory import InMemoryAdapter
from vix.config import Config
from vix.core.manifest import compute_hash
from vix.pipeline import diagnose, eval_run, import_labels

_PNG = (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
        b"\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82")


class _Vec(list):
    def tolist(self):
        return list(self)


class _Boxes:
    def __init__(self, rows):  # rows: (cls_id, conf, [cx,cy,w,h])
        self._r = rows

    def __len__(self):
        return len(self._r)

    @property
    def cls(self):
        return [r[0] for r in self._r]

    @property
    def conf(self):
        return [r[1] for r in self._r]

    @property
    def xywhn(self):
        return [_Vec(r[2]) for r in self._r]


class _Result:
    names = {0: "pothole"}

    def __init__(self, rows):
        self.boxes = _Boxes(rows)


def _install_fake_yolo(monkeypatch, rows):
    mod = types.ModuleType("ultralytics")
    mod.YOLO = lambda *a, **k: types.SimpleNamespace(
        predict=lambda src, **kw: [_Result(rows)])
    monkeypatch.setitem(sys.modules, "ultralytics", mod)


def _ds(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    (tmp_path / "images" / "a.png").write_bytes(_PNG)
    # GT: one pothole top-left; model will predict a different box -> FP + FN
    (tmp_path / "labels" / "a.txt").write_text("0 0.3 0.3 0.2 0.2\n", encoding="utf-8")
    return tmp_path


def test_eval_run_keys_predictions_by_content_hash(tmp_path, monkeypatch):
    folder = _ds(tmp_path)
    cfg = Config(workspace=tmp_path / "ws"); cfg.ensure_dirs()
    ad = InMemoryAdapter()
    import_labels(ad, cfg, folder, fmt="yolo", names=["pothole"], batch="b1")
    _install_fake_yolo(monkeypatch, [(0, 0.95, [0.8, 0.8, 0.1, 0.1])])  # a background FP
    res = eval_run(ad, cfg, "fake.pt")
    h = compute_hash(folder / "images" / "a.png")
    assert h in res["per_image"], "eval keyed by stem, not content hash (KS3 regression)"
    assert ad.fields(h)["eval_fp"] == 1 and ad.fields(h)["eval_fn"] == 1  # attached to the right image


def test_diagnose_tier_a_offline_and_honest(tmp_path, monkeypatch):
    folder = _ds(tmp_path)
    cfg = Config(workspace=tmp_path / "ws"); cfg.ensure_dirs()
    ad = InMemoryAdapter()
    _install_fake_yolo(monkeypatch, [(0, 0.95, [0.8, 0.8, 0.1, 0.1])])
    # Tier A must NOT require FiftyOne: make `import fiftyone` HARD-FAIL for this call (isolation-proof,
    # unlike a global sys.modules check which other suite tests pollute). If diagnose tried to import
    # it, this raises ImportError and the test fails.
    monkeypatch.setitem(sys.modules, "fiftyone", None)
    out = diagnose(ad, cfg, folder, labels_fmt="yolo", weights="fake.pt",
                   names=["pothole"], out_path=cfg.workspace / "wr.md")
    assert "A" in out["tiers"] and "eval" in out
    # honesty F1: the report frames the reference as your UNVERIFIED labels
    html = (cfg.workspace / "wr.html").read_text(encoding="utf-8")
    assert "未經 VIX 覆核" in html and "unverified-ref" in html


def test_diagnose_requires_weights_or_audit(tmp_path):
    folder = _ds(tmp_path)
    cfg = Config(workspace=tmp_path / "ws"); cfg.ensure_dirs()
    with pytest.raises(ValueError, match="weights.*audit|audit.*weights|其一"):
        diagnose(InMemoryAdapter(), cfg, folder, labels_fmt="yolo", names=["pothole"])
