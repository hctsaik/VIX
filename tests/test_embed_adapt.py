"""Domain-adapted embedding: a supervised LDA projection of frozen embeddings turns an
'inseparable in frozen space' pair into a separable one WHEN the separation exists but is swamped
by noise — and CV stays honest (a genuinely inseparable pair is NOT falsely rescued)."""

import numpy as np

from vix import pipeline
from vix.adapters.memory import InMemoryAdapter
from vix.config import Config
from vix.core.embed_adapt import (cv_pair_separability, fit_projection, load_projection,
                                  projection_gate, save_projection, transform)
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


# --- Feature 1: gate-validated apply across the stack ---

def test_projection_gate_go_and_nogo():
    go, _r, s = projection_gate([{"pair": ["a", "b"], "frozen_sep_err": 0.45, "adapted_sep_err": 0.10, "rescued": True}])
    assert go and s["n_rescued"] == 1
    nogo, reasons, _ = projection_gate([{"pair": ["a", "b"], "frozen_sep_err": 0.20, "adapted_sep_err": 0.30, "rescued": False}])
    assert not nogo and reasons                                    # a pair regressed -> NO-GO
    flat, _r2, _ = projection_gate([{"pair": ["a", "b"], "frozen_sep_err": 0.30, "adapted_sep_err": 0.30, "rescued": False}])
    assert not flat                                                # no macro gain -> NO-GO


def test_adapt_embedding_gate_go_enables(tmp_path):
    cfg = Config(workspace=tmp_path / "ws"); cfg.ensure_dirs()
    ad = InMemoryAdapter()
    for i, v in enumerate(_A):
        ad.seed(f"a{i}", "a.png", [_det("a", v)], tags=[Tag.GOLDEN])
    for i, v in enumerate(_B):
        ad.seed(f"b{i}", "b.png", [_det("b", v)], tags=[Tag.GOLDEN])
    r = pipeline.adapt_embedding(ad, cfg, save=True, enable=True)
    assert r["gate"]["go"] and r["enabled"]
    assert cfg.embed_projection_enabled_path.exists() and cfg.adapt_report_path.exists()


def test_adapt_embedding_nogo_does_not_enable(tmp_path):
    cfg = Config(workspace=tmp_path / "ws"); cfg.ensure_dirs()
    ad = InMemoryAdapter()
    rng = np.random.RandomState(7)  # already cleanly separable -> frozen sep_err ~0 -> no gain -> gate NO-GO
    a = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0.]) + 0.05 * rng.randn(80, 10)
    b = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0.]) + 0.05 * rng.randn(80, 10)
    for i, v in enumerate(a):
        ad.seed(f"a{i}", "a.png", [_det("a", v)], tags=[Tag.GOLDEN])
    for i, v in enumerate(b):
        ad.seed(f"b{i}", "b.png", [_det("b", v)], tags=[Tag.GOLDEN])
    r = pipeline.adapt_embedding(ad, cfg, save=True, enable=True)
    assert not r["gate"]["go"] and not r["enabled"]
    assert not cfg.embed_projection_enabled_path.exists()         # gate NO-GO -> never enabled


def _tri(seed):
    rng = np.random.RandomState(seed)
    return (np.array([2, 0, 0, 0.]) + 0.1 * rng.randn(8, 4),
            np.array([0, 2, 0, 0.]) + 0.1 * rng.randn(8, 4),
            np.array([0, 0, 2, 0.]) + 0.1 * rng.randn(8, 4))


def test_active_projection_dim_and_enable_guard(tmp_path):
    cfg = Config(workspace=tmp_path / "ws"); cfg.ensure_dirs()
    # 2-class projection -> 1-d -> guarded out (cosine ranking degenerate) even when enabled
    save_projection(cfg.embed_projection_path, fit_projection(np.vstack([_A, _B]), np.array([0] * 80 + [1] * 80)))
    cfg.embed_projection_enabled_path.write_text("gate=GO", encoding="utf-8")
    assert pipeline._active_projection(cfg) is None               # out_dim 1 -> skip
    # 3-class projection -> 2-d -> active
    A, B, C = _tri(0)
    X = np.vstack([A, B, C]); y = np.array(["a"] * 8 + ["b"] * 8 + ["c"] * 8)
    save_projection(cfg.embed_projection_path, fit_projection(X, y))
    assert pipeline._active_projection(cfg) is not None
    # not enabled -> None regardless
    cfg.embed_projection_enabled_path.unlink()
    assert pipeline._active_projection(cfg) is None


def test_error_mine_applies_projection_when_enabled(tmp_path):
    import json
    cfg = Config(workspace=tmp_path / "ws"); cfg.ensure_dirs()
    ad = InMemoryAdapter()
    A, B, C = _tri(1)
    for grp, lab in ((A, "a"), (B, "b"), (C, "c")):                # golden for the projection (3 classes -> 2-d)
        for i, v in enumerate(grp):
            ad.seed(f"{lab}{i}", "g.png", [_det(lab, v)], tags=[Tag.GOLDEN])
    save_projection(cfg.embed_projection_path, fit_projection(
        np.vstack([A, B, C]), np.array(["a"] * 8 + ["b"] * 8 + ["c"] * 8)))
    cfg.embed_projection_enabled_path.write_text("gate=GO", encoding="utf-8")
    ad.seed("e", "e.png", [_det("a", A[0])], tags=[Tag.EVAL])      # error image carries an 'a' region
    ad.seed("c0", "c0.png", [_det("a", A[1])], tags=[])           # near 'a' -> should rank first
    ad.seed("c1", "c1.png", [_det("b", B[0])], tags=[])           # near 'b'
    (tmp_path / "res.jsonl").write_text(json.dumps(
        {"vix_hash": "e", "gt": [{"label": "a", "bbox": [0.5, 0.5, 0.4, 0.4]}], "pred": []}), encoding="utf-8")
    pipeline.eval_ingest(ad, cfg, str(tmp_path / "res.jsonl"))
    mined = pipeline.error_mine(ad, cfg, top=5)
    ids = [m["id"].split(":")[0] for m in mined]
    assert ids and ids.index("c0") < ids.index("c1")             # projected ranking still correct
    assert any("投影" in m["why"] for m in mined)                 # projection path was applied
