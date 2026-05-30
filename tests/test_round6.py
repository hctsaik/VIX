import numpy as np
from PIL import Image

from vix import pipeline
from vix.adapters.memory import InMemoryAdapter
from vix.cli import main
from vix.config import Config
from vix.core.manifest import Manifest, ManifestEntry
from vix.core.snapshot import create_snapshot
from vix.embedding.simple import pixel_embedding
from vix.types import BBox, Detection, Tag


def _det(label, conf, emb=None):
    return Detection(label, conf, BBox(0.5, 0.5, 1.0, 1.0),
                     embedding=np.array(emb, dtype=float) if emb is not None else None)


def test_run_pipeline_offline_real_files(tmp_path):  # X5: pixel_fallback end-to-end via run
    imgs = tmp_path / "imgs"
    imgs.mkdir()

    def vert(path, split):
        a = np.zeros((32, 32, 3), np.uint8)
        a[:, :split] = 255
        Image.fromarray(a).save(path)

    def horiz(path, split):
        a = np.zeros((32, 32, 3), np.uint8)
        a[:split, :] = 255
        Image.fromarray(a).save(path)

    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    cfg.embedding_backend = "pixel_fallback"
    ad = InMemoryAdapter(embedder=lambda im: pixel_embedding(im, size=8))
    for i in range(6):
        pv = imgs / f"v{i}.png"; vert(pv, 8 + i)
        ad.seed(f"v{i}", str(pv), [_det("vert", 0.9)], tags=[Tag.GOLDEN])
        ph = imgs / f"h{i}.png"; horiz(ph, 8 + i)
        ad.seed(f"h{i}", str(ph), [_det("horiz", 0.9)], tags=[Tag.GOLDEN])
    pc = imgs / "c.png"; vert(pc, 20)
    ad.seed("c", str(pc), [_det("vert", 0.5)])

    s = pipeline.run_pipeline(ad, cfg)  # no weights -> infer skipped; embed via pixel embedder
    assert all(st["ok"] for st in s["steps"])
    assert "quality_score" in s and s["gate"] in ("GO", "NO-GO")


def test_restore_apply(tmp_path):  # X10
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    m = Manifest(cfg.manifest_path)
    m.append(ManifestEntry.create("imgs/g1.png", "b1", tags=["golden"], vix_hash="g1"))
    m.append(ManifestEntry.create("imgs/g2.png", "b1", tags=["golden"], vix_hash="g2"))
    snap = cfg.workspace / "snap.json"
    create_snapshot(cfg.manifest_path, snap, "v1")

    ad = InMemoryAdapter()
    res = pipeline.restore_apply(ad, cfg, snap)
    assert res["n_restored"] == 2
    assert {"g1", "g2"} <= {h for h, _s, _d, _t in ad.samples()}


def test_harmful_remove_audited(tmp_path):  # X9
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    ad = InMemoryAdapter()
    for i in range(4):
        ad.seed(f"g{i}", f"g{i}.png", [_det("a", 0.9, [1, 0])], tags=[Tag.GOLDEN])
    ad.seed("dupA", "dupA.png", [_det("a", 0.9, [1, 0])], tags=[Tag.GOLDEN])
    ad.seed("dupB", "dupB.png", [_det("a", 0.9, [1, 0])], tags=[Tag.GOLDEN])
    ids = pipeline.harmful_remove(ad, cfg, top=2, note="QA decision")
    assert len(ids) >= 1
    rejected = {h for h, _s, _d, t in ad.samples() if Tag.REJECTED in t}
    assert set(ids) <= rejected


def test_quickstart_cli(tmp_path, capsys):  # X8
    main(["--workspace", str(tmp_path / "ws"), "--adapter", "memory", "quickstart"])
    out = capsys.readouterr().out
    assert "golden" in out and "vix run" in out
