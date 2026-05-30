"""End-to-end on REAL image files, FiftyOne-free, via the pixel embedder.

Proves the whole embedding -> analytics chain actually runs on disk images
(addresses the Round-1 reviewer concern that the embedding pipeline had never
been executed end-to-end).
"""

import numpy as np
from PIL import Image

from vix import pipeline
from vix.adapters.memory import InMemoryAdapter
from vix.config import Config
from vix.embedding.simple import pixel_embedding
from vix.types import BBox, Detection, Tag


def _vert(path, split):  # white left columns -> distinct "direction" per split
    a = np.zeros((32, 32, 3), np.uint8)
    a[:, :split] = 255
    Image.fromarray(a).save(path)


def _horiz(path, split):
    a = np.zeros((32, 32, 3), np.uint8)
    a[:split, :] = 255
    Image.fromarray(a).save(path)


def _det(label, conf=0.9):
    return Detection(label, conf, BBox(0.5, 0.5, 1.0, 1.0))  # whole-image box


def test_end_to_end_real_files(tmp_path):
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    ad = InMemoryAdapter(embedder=lambda im: pixel_embedding(im, size=8))

    for i in range(6):
        pv = imgs / f"vert{i}.png"
        _vert(pv, 8 + i)
        ad.seed(f"vert{i}", str(pv), [_det("vert")], tags=[Tag.GOLDEN])
        ph = imgs / f"horiz{i}.png"
        _horiz(ph, 8 + i)
        ad.seed(f"horiz{i}", str(ph), [_det("horiz")], tags=[Tag.GOLDEN])

    # an exact-duplicate pair (identical bytes -> cosine distance 0)
    for name in ("dupA", "dupB"):
        p = imgs / f"{name}.png"
        _vert(p, 24)
        ad.seed(name, str(p), [_det("vert")], tags=[Tag.GOLDEN])

    # a candidate (non-golden) novel image
    pc = imgs / "cand.png"
    _horiz(pc, 28)
    ad.seed("cand", str(pc), [_det("horiz", 0.2)])

    # REAL embedding from files
    ad.compute_embeddings("pixel")
    embedded = [d.embedding is not None for _h, _s, dets, _t in ad.samples() for d in dets]
    assert embedded and all(embedded)

    # dedup catches the exact-duplicate pair
    groups = pipeline.dedup(ad, cfg, max_distance=0.01)
    assert any({"dupA", "dupB"} <= set(g) for g in groups)

    # coverage sees both classes
    cov = pipeline.coverage(ad, cfg)
    assert {"vert", "horiz"} <= set(cov["distribution"])

    # active learning runs and returns the candidate (now with reasons)
    assert pipeline.active_learn(ad, cfg, budget=1)[0]["id"] == "cand"

    # one-click health report writes artifacts
    rep, paths = pipeline.health_report(ad, cfg, tmp_path / "rep")
    assert (tmp_path / "rep" / "health_report.md").exists()
    assert rep["total_images"] == 15
    assert rep["n_duplicate_groups"] >= 1
