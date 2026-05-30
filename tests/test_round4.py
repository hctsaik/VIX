import json

import numpy as np
from PIL import Image

from vix import pipeline
from vix.adapters.memory import InMemoryAdapter
from vix.config import Config
from vix.core.analytics import EmbItem, coverage_gaps
from vix.core.manifest import compute_hash
from vix.core.report import build_report
from vix.types import BBox, Detection, Tag


def _det(label, conf, emb):
    return Detection(label, conf, BBox(0.5, 0.5, 0.2, 0.2), embedding=np.array(emb, dtype=float))


def _it(id, label, emb):
    return EmbItem(id, label, np.array(emb, dtype=float))


def test_quality_score_and_gate_in_report():
    rep = build_report(
        version="v", total=100, class_dist={"a": 50, "b": 50}, pass_count=90, review_count=10,
        duplicate_groups=[["x", "y"]], label_issues=[1],
        coverage={"a": {"under_represented": False}}, gate_verdict="GO",
    )
    assert 0 <= rep["quality_score"] <= 100
    assert rep["gate_verdict"] == "GO"


def test_coverage_need_quantified():
    items = [_it(f"a{i}", "a", [1, 0]) for i in range(10)] + [_it("b0", "b", [0, 1])]
    gaps = coverage_gaps(items, k=3)
    assert gaps["b"]["under_represented"] and gaps["b"]["need"] > 0
    assert gaps["a"]["need"] == 0


def test_run_pipeline_end_to_end(tmp_path):
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    ad = InMemoryAdapter()
    for i in range(8):
        ad.seed(f"a{i}", f"a{i}.png", [_det("a", 0.9, [1, 0])], tags=[Tag.GOLDEN])
        ad.seed(f"b{i}", f"b{i}.png", [_det("b", 0.9, [0, 1])], tags=[Tag.GOLDEN])
    ad.seed("c1", "c1.png", [_det("a", 0.92, [1, 0])])
    s = pipeline.run_pipeline(ad, cfg)
    assert s["gate"] in ("GO", "NO-GO")
    assert 0 <= s["quality_score"] <= 100
    assert all(st["ok"] for st in s["steps"])


def test_history_tracks_resubmission(tmp_path):
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    Image.new("RGB", (8, 8), (1, 2, 3)).save(imgs / "x.png")
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    ad = InMemoryAdapter()
    pipeline.ingest(ad, cfg, imgs, "w1")
    pipeline.ingest(ad, cfg, imgs, "w2")  # re-submit same file
    hist = pipeline.history(cfg, compute_hash(imgs / "x.png"))
    assert len(hist) >= 2
    assert any(r["decision"] == "skipped" for r in hist)


def test_routing_diff_runs(tmp_path):
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    ad = InMemoryAdapter()
    for i in range(5):
        ad.seed(f"g{i}", f"g{i}.png", [_det("a", 0.9, [1, 0])], tags=[Tag.GOLDEN])
    ad.seed("c", "c.png", [_det("a", 0.9, [1, 0])])
    pol = pipeline.calibrate(ad, cfg)
    pipeline.route(ad, cfg, pol)
    pipeline.route(ad, cfg, pol)
    assert pipeline.routing_diff(cfg)["n_changed"] == 0


def test_dismiss_excludes_and_fp_rate(tmp_path):
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    ad = InMemoryAdapter()
    for i in range(5):
        ad.seed(f"g{i}", f"g{i}.png", [_det("a", 0.9, [1, 0])], tags=[Tag.GOLDEN])
    ad.seed("bad", "bad.png", [_det("a", 0.01, [1, 0])])
    pol = pipeline.calibrate(ad, cfg)
    pipeline.route(ad, cfg, pol)
    assert pipeline.dismiss(ad, cfg, ["bad"]) == 1
    assert all(r["id"] != "bad" for r in pipeline.review_queue(ad, cfg))
    assert pipeline.false_positive_rate(cfg)["dismissed_false_alarms"] == 1


def test_relabel_rollback(tmp_path):
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    ad = InMemoryAdapter()
    ad.seed("s1", "s1.png", [_det("sedan", 0.9, [1, 0])], tags=[Tag.GOLDEN])
    pipeline.relabel_dataset(ad, cfg, {"sedan": "car"})
    assert {d.label for _h, _s, dets, _t in ad.samples() for d in dets} == {"car"}
    pipeline.relabel_rollback(ad, cfg)
    assert {d.label for _h, _s, dets, _t in ad.samples() for d in dets} == {"sedan"}


def test_export_manifest_includes_labels_and_yaml(tmp_path):
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    ad = InMemoryAdapter()
    ad.seed("g1", "g1.png", [_det("a", 0.9, [1, 0])], tags=[Tag.GOLDEN])
    pipeline.export(ad, cfg, ["a"], tmp_path / "out")
    man = tmp_path / "out" / "export_manifest.jsonl"
    files = {json.loads(line)["file"] for line in man.read_text().splitlines() if line.strip()}
    assert "data.yaml" in files
    assert any(f.endswith(".txt") for f in files)
