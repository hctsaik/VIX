"""Domain-adapted embedding: a supervised LDA projection of frozen embeddings turns an
'inseparable in frozen space' pair into a separable one WHEN the separation exists but is swamped
by noise — and CV stays honest (a genuinely inseparable pair is NOT falsely rescued)."""

import numpy as np

from vix import pipeline
from vix.adapters.memory import InMemoryAdapter
from vix.config import Config
from vix.core.embed_adapt import (cv_pair_separability, fit_projection, load_projection,
                                  save_projection, transform)
from vix.types import BBox, Detection, Tag


def _noise_swamped(sign, n, d=10, seed=0):
    """Discriminative info in dim0 (small variance), swamped by large isotropic noise elsewhere.
    Frozen Euclidean kNN drowns in the noise; LDA recovers dim0."""
    rng = np.random.RandomState(seed)
    X = 2.5 * rng.randn(n, d)                 # large isotropic noise dominates raw distance
    X[:, 0] = sign * 0.6 + 0.1 * rng.randn(n)  # clean separation in dim0 (LDA recovers it)
    return X


_A = _noise_swamped(+1, 80, seed=1)
_B = _noise_swamped(-1, 80, seed=2)


def test_fit_projection_shape_and_transform():
    X = np.vstack([_A, _B]); y = np.array([0] * 80 + [1] * 80)
    proj = fit_projection(X, y)
    assert proj["W"].shape == (10, 1)                       # K-1 = 1 LDA dim
    assert transform(proj, X).shape == (160, 1)


def test_lda_rescues_noise_swamped_separation():
    fr, ad, n = cv_pair_separability(_A, _B, folds=5)
    assert fr > 0.38                                        # frozen kNN drowns in noise (inseparable)
    assert ad < 0.35 and (fr - ad) > 0.15                  # projection rescues it across the separable bar (CV'd)


def test_cv_does_not_falsely_rescue_truly_inseparable():
    # both classes drawn from the SAME distribution -> no real boundary; CV must NOT report low error
    a = _noise_swamped(+1, 80, seed=3)
    b = _noise_swamped(+1, 80, seed=4)  # same sign/center as a
    fr, ad, n = cv_pair_separability(a, b, folds=5)
    assert ad > 0.35                                        # honest: refit-per-fold can't overfit a non-boundary


def test_save_load_roundtrip(tmp_path):
    proj = fit_projection(np.vstack([_A, _B]), np.array([0] * 80 + [1] * 80))
    p = tmp_path / "proj.npz"
    save_projection(p, proj)
    loaded = load_projection(p)
    assert loaded is not None and loaded["W"].shape == proj["W"].shape
    assert np.allclose(transform(loaded, _A), transform(proj, _A))
    assert load_projection(tmp_path / "nope.npz") is None


def _det(label, emb):
    return Detection(label, 0.9, BBox(0.5, 0.5, 0.2, 0.2), embedding=np.asarray(emb, float))


def test_pipeline_adapt_embedding_rescues_and_saves(tmp_path):
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    ad = InMemoryAdapter()
    for i, v in enumerate(_A):
        ad.seed(f"a{i}", "a.png", [_det("a", v)], tags=[Tag.GOLDEN])
    for i, v in enumerate(_B):
        ad.seed(f"b{i}", "b.png", [_det("b", v)], tags=[Tag.GOLDEN])
    r = pipeline.adapt_embedding(ad, cfg, save=True)
    assert r["out_dim"] == 1 and r["n_rescued"] >= 1
    pair = next(p for p in r["pairs"] if set(p["pair"]) == {"a", "b"})
    assert pair["rescued"] and pair["adapted_sep_err"] < pair["frozen_sep_err"]
    assert load_projection(cfg.workspace / "embed_projection.npz") is not None
