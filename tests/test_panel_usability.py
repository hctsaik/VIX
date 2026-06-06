"""First-batch usability features from the panel/usability multi-agent review (R1-R3 consensus):
U1 (bare/unknown `vix` prints the golden path, not the 70-verb wall), U2 (precondition guards name
the prerequisite instead of leaking `Errno 2`), U4 (`vix status` = where am I + next step). All
fully offline-testable core/CLI — deliberately NOT the live-App Panel (that's verified by verify-gui)."""

import numpy as np
import pytest

from vix import pipeline
from vix.adapters.memory import InMemoryAdapter
from vix.cli import main
from vix.config import Config
from vix.types import BBox, Detection, Tag


def _det(label, emb, conf=0.9):
    return Detection(label, conf, BBox(0.5, 0.5, 0.2, 0.2), embedding=np.asarray(emb, float))


# ---- U1: golden-path on-ramp ----------------------------------------------------------------

def test_bare_vix_prints_golden_path(capsys):
    rc = main([])
    assert rc == 0
    out = capsys.readouterr().out
    assert "快速上手" in out and "vix calibrate" in out  # the 9-step path, not an argparse verb wall


def test_unknown_verb_shows_golden_path_not_wall(capsys):
    with pytest.raises(SystemExit) as e:
        main(["bogus-verb"])
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert "快速上手" in err and "vix --help" in err


# ---- U2: precondition guards name the prerequisite ------------------------------------------

def test_route_before_calibrate_names_prerequisite(tmp_path, capsys):
    rc = main(["--workspace", str(tmp_path / "ws"), "--adapter", "memory", "route"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "請先執行 vix calibrate" in err and "Errno" not in err  # named step, not a raw filesystem error


def test_calibrate_without_golden_names_prerequisite(tmp_path):
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    with pytest.raises(ValueError, match="尚無 golden"):
        pipeline.calibrate(InMemoryAdapter(), cfg)


def test_export_without_golden_names_prerequisite(tmp_path):
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    with pytest.raises(ValueError, match="尚無 golden 可匯出"):
        pipeline.export(InMemoryAdapter(), cfg, ["a"], tmp_path / "out")


# ---- U4: vix status (where am I + branching next step) --------------------------------------

def test_status_next_step_branches_on_state(tmp_path):
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    ad = InMemoryAdapter()
    assert pipeline.status(ad, cfg)["next"]["stage"] == "ingest"          # empty workspace
    ad.seed("g0", "g0.png", [_det("a", [1, 0, 0])], tags=[Tag.GOLDEN])    # has detections + embeddings
    st = pipeline.status(ad, cfg)
    assert st["counts"]["golden"] == 1 and st["has_embeddings"]
    assert st["next"]["stage"] == "calibrate"                            # emb present, no thresholds yet
    pipeline.calibrate(ad, cfg)
    assert pipeline.status(ad, cfg)["next"]["stage"] == "route"          # thresholds now exist, not routed


def test_status_cli_smoke(tmp_path, capsys):
    rc = main(["--workspace", str(tmp_path / "ws"), "--adapter", "memory", "status"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "下一步" in out and "vix ingest" in out
