"""Domain-adapted embedding (gt-consistency amplification step #2).

The consistency layer can verdict a class pair "not separable IN THE CURRENT EMBEDDING SPACE" —
but frozen DINOv2 is a generic encoder, so that may be an *encoder* limit, not a true taxonomy
dead-end. This module learns a lightweight, supervised projection of the frozen DINOv2 embeddings
from the human-confirmed GOLDEN labels (regularized LDA: PCA pre-reduction + shrinkage), turning
"inseparable in frozen DINO" into a falsifiable, fixable question: *does a learned projection
separate them?* If yes, it's a representation problem (apply the projection / it's fixable), not a
class-definition dead-end.

OFFLINE, $0, closed-form — this is NOT training YOLO; it's a few matrix ops on already-computed
embeddings (seconds on CPU). Honesty is enforced by measuring before/after separability with k-fold
CROSS-VALIDATION (the projection is refit on train folds only) — fitting LDA and scoring it on the
same tens of points would look artificially separable; CV is the anti-overfit guard. Pure numpy.
"""

from __future__ import annotations

import numpy as np

SEP_SEPARABLE = 0.35  # kNN error below this => "separable" (matches consistency.SEP_INSEPARABLE)


def fit_projection(X: np.ndarray, y, max_pca: int = 64, shrink: float = 0.3,
                   max_lda: int | None = None) -> dict:
    """Regularized LDA projection of frozen embeddings, supervised by golden class labels.
    Pipeline: center -> PCA (handle p>>n) -> shrinkage-LDA (Sw + lambda*I). Returns {mean, W};
    transform is (X - mean) @ W. None if <2 classes."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    classes = sorted(set(y.tolist()))
    K = len(classes)
    if K < 2 or X.shape[0] < K + 1:
        return {"mean": X.mean(0) if X.size else np.zeros(X.shape[1:]), "W": None, "classes": classes}
    mu = X.mean(0)
    Xc = X - mu
    n, d = Xc.shape
    n_pca = max(1, min(max_pca, n - 1, d))
    _U, _S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Vt = Vt[:n_pca]                       # (n_pca, d)
    Xp = Xc @ Vt.T                        # (n, n_pca)
    overall = Xp.mean(0)
    Sw = np.zeros((n_pca, n_pca))
    Sb = np.zeros((n_pca, n_pca))
    for c in classes:
        Xk = Xp[y == c]
        mk = Xk.mean(0)
        diff = Xk - mk
        Sw += diff.T @ diff
        m = (mk - overall)[:, None]
        Sb += len(Xk) * (m @ m.T)
    Sw += shrink * (np.trace(Sw) / n_pca + 1e-9) * np.eye(n_pca)  # shrinkage -> invertible at small n
    evals, evecs = np.linalg.eig(np.linalg.solve(Sw, Sb))
    order = np.argsort(-evals.real)
    n_lda = min(max_lda or (K - 1), n_pca, K - 1)
    W_lda = evecs.real[:, order[:max(1, n_lda)]]   # (n_pca, n_lda)
    W = Vt.T @ W_lda                                # (d, n_lda)
    return {"mean": mu, "W": W, "classes": classes}


def transform(proj: dict, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if proj.get("W") is None:
        return X
    return (X - proj["mean"]) @ proj["W"]


def _knn_err(Xtr, ytr, Xte, yte, k):
    """Euclidean kNN classification error of test points against train points."""
    if len(Xtr) == 0 or len(Xte) == 0:
        return 0.0
    kk = min(k, len(Xtr))
    d2 = ((Xte[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1)   # (n_te, n_tr)
    nn = np.argpartition(d2, kk - 1, axis=1)[:, :kk]
    wrong = 0
    for i in range(len(Xte)):
        labs = ytr[nn[i]]
        vals, cnts = np.unique(labs, return_counts=True)
        pred = vals[np.argmax(cnts)]
        wrong += int(pred != yte[i])
    return wrong / len(Xte)


def _folds(n, n_folds, seed=0):
    idx = np.random.RandomState(seed).permutation(n)
    return [idx[i::n_folds] for i in range(n_folds)]


def cv_pair_separability(emb_i, emb_j, folds: int = 5, k: int | None = None,
                         max_pca: int = 64, seed: int = 0):
    """Honest before/after kNN separability of a class pair via k-fold CV. For each fold the LDA
    projection is refit on the train folds ONLY (never sees the test fold). Returns
    (frozen_err, adapted_err, n_min). Euclidean kNN throughout so before/after are comparable."""
    Ei, Ej = np.asarray(emb_i, float), np.asarray(emb_j, float)
    ni, nj = len(Ei), len(Ej)
    n_min = min(ni, nj)
    if n_min < 2:
        return (0.0, 0.0, n_min)
    nf = max(2, min(folds, n_min))
    k = k or max(1, min(5, n_min - 1))
    fi, fj = _folds(ni, nf, seed), _folds(nj, nf, seed + 1)
    fr_w = fr_n = ad_w = ad_n = 0
    for f in range(nf):
        te_i, te_j = fi[f], fj[f]
        tr_i = np.concatenate([fi[g] for g in range(nf) if g != f]) if nf > 1 else fi[f]
        tr_j = np.concatenate([fj[g] for g in range(nf) if g != f]) if nf > 1 else fj[f]
        if len(tr_i) == 0 or len(tr_j) == 0 or len(te_i) == 0 or len(te_j) == 0:
            continue
        Xtr = np.vstack([Ei[tr_i], Ej[tr_j]]); ytr = np.array([0] * len(tr_i) + [1] * len(tr_j))
        Xte = np.vstack([Ei[te_i], Ej[te_j]]); yte = np.array([0] * len(te_i) + [1] * len(te_j))
        fr = _knn_err(Xtr, ytr, Xte, yte, k)                       # frozen: raw coords
        fr_w += fr * len(yte); fr_n += len(yte)
        proj = fit_projection(Xtr, ytr, max_pca=max_pca)           # refit on train only
        ad = _knn_err(transform(proj, Xtr), ytr, transform(proj, Xte), yte, k)
        ad_w += ad * len(yte); ad_n += len(yte)
    frozen = (fr_w / fr_n) if fr_n else 0.0
    adapted = (ad_w / ad_n) if ad_n else 0.0
    return (round(frozen, 4), round(adapted, 4), n_min)


def save_projection(path, proj: dict) -> None:
    if proj.get("W") is None:
        raise ValueError("no projection to save (need >=2 classes)")
    np.savez(path, mean=proj["mean"], W=proj["W"], classes=np.array(proj["classes"], dtype=object))


def load_projection(path) -> dict | None:
    from pathlib import Path
    if not Path(path).exists():
        return None
    z = np.load(path, allow_pickle=True)
    return {"mean": z["mean"], "W": z["W"], "classes": list(z["classes"])}
