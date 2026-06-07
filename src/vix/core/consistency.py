"""GT-powered consistency ATTRIBUTION (model-loop-v2 / safe-vix-merge follow-up).

VIX is otherwise embedding-self-referential: its label signals can only say "this disagrees with
its neighbours", so they are structurally blind exactly where two classes overlap. Ground-truth
labels (here: the human-confirmed GOLDEN embeddings per class, plus the eval-ingest confusion
matrix) break that circularity and let us ATTRIBUTE a class-pair failure to its cause:

  * separability   — LOO-kNN error of class i vs j in DINOv2 space. High error => the two classes
                     are not separable IN THE CURRENT EMBEDDING SPACE (an encoder-hedged verdict,
                     not "your taxonomy is broken").
  * overlap O[i->j] — fraction of true-i golden points whose k nearest (within {i,j}) are class j.
                     Unit-matched to the model confusion rate C[i->j], so they can be compared.
  * 2x2 attribution — join O (embedding) with C (model): taxonomy / model / label_noise / clean.

Everything is offline, advisory, and support-gated with CIs. On a small GT set the honest output is
`insufficient_support` or `*_watch`, never a confident merge recommendation. Pure numpy / stdlib.
"""

from __future__ import annotations

import math

import numpy as np

SEP_INSEPARABLE = 0.35  # LOO-kNN error above this => "not separable in the current embedding space"


def _l2norm(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[None, :]
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-12)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion k/n (correct coverage at small n / near 0,1)."""
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (centre - half) / denom), min(1.0, (centre + half) / denom))


def k_rule(n_min: int) -> int:
    """k = max(3, floor(sqrt(n_min))) forced odd (no 2-class ties); capped by callers."""
    k = max(3, int(math.isqrt(max(1, int(n_min)))))
    return k if k % 2 == 1 else k + 1


def _tier(n_min: int, n_pair: int) -> str:
    if n_min < 10 or n_pair < 25:
        return "insufficient_support"
    return "provisional" if n_min < 20 else "supported"


def pair_stats(emb_i: np.ndarray, emb_j: np.ndarray, k: int | None = None,
               n_boot: int = 400, seed: int = 0) -> dict | None:
    """LOO-kNN separability + directional overlap for one class pair (on golden embeddings).
    Returns sep_err (+Wilson CI), O_ij/O_ji (+bootstrap CI), supports. None if a class is empty."""
    Ei, Ej = _l2norm(emb_i), _l2norm(emb_j)
    ni, nj = Ei.shape[0], Ej.shape[0]
    if ni == 0 or nj == 0:
        return None
    k = k or k_rule(min(ni, nj))
    E = np.vstack([Ei, Ej])
    y = np.array([0] * ni + [1] * nj)
    n = ni + nj
    S = E @ E.T
    np.fill_diagonal(S, -np.inf)  # leave-one-out
    kk = min(k, n - 1)
    idx = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
    nbr = y[idx]  # n x kk neighbour labels
    frac_other = nbr.mean(axis=1)  # for class-0 rows: frac of neighbours that are class 1
    maj = (frac_other > 0.5).astype(int)
    mis = maj != y
    sep_err = float(mis.mean())
    sep_ci = wilson(int(mis.sum()), n)

    frac_i = frac_other[:ni]          # class-i points: frac of neighbours that are j
    frac_j = 1.0 - frac_other[ni:]    # class-j points: frac of neighbours that are i
    rng = np.random.RandomState(seed)

    def _boot(fr: np.ndarray) -> list[float]:
        if len(fr) < 2:
            m = float(fr.mean()) if len(fr) else 0.0
            return [round(m, 4), round(m, 4)]
        means = [fr[rng.randint(0, len(fr), len(fr))].mean() for _ in range(n_boot)]
        return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]

    return {
        "ni": ni, "nj": nj, "k": int(kk),
        "sep_err": round(sep_err, 4), "sep_ci": [round(x, 4) for x in sep_ci],
        "O_ij": round(float(frac_i.mean()), 4), "O_ij_ci": _boot(frac_i),
        "O_ji": round(float(frac_j.mean()), 4), "O_ji_ci": _boot(frac_j),
    }


def _apply_reference_firewall(f: dict) -> dict:
    """Honesty ruling F2: when the per-class reference is human-UNVERIFIED (imported labels, not
    confirmed golden), VIX must NOT use the labels-under-audit as the trusted oracle to convict
    them. Separability/clean (falsifiable geometry, claims nothing about label correctness) survive
    unchanged; the label-trust verdicts are softened to watch/audit and never block a retrain."""
    v = f["verdict"]
    li, lj = f["pair"]
    if v == "label_noise":  # pure circularity -> never fire on an unverified reference
        f["verdict"] = "label_audit_needed"
        f["action"] = (f"嵌入難分且與你的標籤分歧,但這些標籤未覆核 → 先人工覆核 {li}/{lj}"
                       "(確認後升級為 golden 再判定是否標籤雜訊);現不宣稱標籤錯誤")
    elif v == "taxonomy":
        f["verdict"] = "taxonomy_watch"
        f["action"] = (f"重疊與混淆一致,但參照標籤未覆核 → 僅監看 {li}/{lj};"
                       "先覆核成 golden 再考慮 merge/重寫規則(勿做不可逆動作)")
    elif v == "model":
        f["verdict"] = "model_watch"
        f["action"] = (f"{li}/{lj} 可分但模型混淆(參照未覆核)→ 監看;可沿邊界補資料,先覆核標籤再下結論")
    f["reference_trusted"] = False
    return f


def _attribute(li: str, lj: str, *, ni: int, nj: int, k: int, sep_err: float, sep_ci: list,
               O: float, O_ci: list, confusion: dict | None, n_gt: dict | None,
               reference_trusted: bool = True) -> dict:
    """Directed i->j verdict from overlap O[i->j] (embedding) vs C[i->j] (model confusion)."""
    tier = _tier(min(ni, nj), ni + nj)
    n_gt_i = int((n_gt or {}).get(li, 0))
    c_cnt = int((confusion or {}).get(f"{li}->{lj}", 0))
    have_C = bool(n_gt) and n_gt_i > 0
    C = (c_cnt / n_gt_i) if have_C else None
    C_ci = list(wilson(c_cnt, n_gt_i)) if have_C else None
    f = {
        "pair": [li, lj], "tier": tier,
        "support": {"golden_i": ni, "golden_j": nj, "n_gt_i": n_gt_i, "k": k},
        "separable_in_embedding": ("no" if sep_err > SEP_INSEPARABLE else "yes"),
        "sep_err": sep_err, "sep_ci": sep_ci,
        "O_ij": O, "O_ci": [round(x, 4) for x in O_ci],
        "C_ij": (round(C, 4) if C is not None else None),
        "C_ci": ([round(x, 4) for x in C_ci] if C_ci else None),
        "delta": None, "delta_ci": None,
    }
    if tier == "insufficient_support":
        f.update(verdict="insufficient_support", action="golden/GT 樣本不足,先補該對樣本再判(避免假性結論)")
        return f
    if C is None:  # no eval confusion -> separability-only (still GT via golden labels)
        if sep_err > SEP_INSEPARABLE:
            f.update(verdict="inseparable_embedding",
                     action=f"{li}/{lj} 在目前 embedding 空間難分;接 eval-ingest 才能歸因 taxonomy/model/label")
        else:
            f.update(verdict="separable_embedding", action=f"{li}/{lj} 在目前 embedding 空間可分")
        return f
    tau = max(0.10, 2.0 / max(1, min(n_gt_i, ni)))
    d = O - C
    d_ci = [round(O_ci[0] - C_ci[1], 4), round(O_ci[1] - C_ci[0], 4)]
    f["delta"], f["delta_ci"] = round(d, 4), d_ci
    zero_in = d_ci[0] <= 0 <= d_ci[1]
    c_hi, o_hi = C >= tau, O >= tau
    has_conf = c_cnt > 0            # the model actually confused i->j at least once
    inseparable = sep_err > SEP_INSEPARABLE  # robust (majority-vote) overlap signal, not the mean O
    if c_hi and o_hi and zero_in:
        if tier == "provisional":
            f.update(verdict="taxonomy_watch", action="重疊與混淆一致但支撐不足:僅監看,勿 merge")
        else:
            f.update(verdict="taxonomy",
                     action=f"在目前 embedding 空間 {li}/{lj} 難分且模型同步混淆 → 停止多標;考慮 merge 或重寫判別規則(先看例圖)")
    elif c_hi and not o_hi and d_ci[1] < 0:
        f.update(verdict="model", action=f"{li}/{lj} 可分但模型混淆 → 沿邊界補硬負樣本/資料(標註是對的槓桿)")
    # label_noise requires POSITIVE model confusion (c_cnt>0) AND genuine embedding entanglement
    # (sep_err inseparable) — without confusion there is no model-vs-label disagreement to attribute,
    # and the mean O alone over-counts overlap under noise on a majority-separable pair (the bug fix).
    elif o_hi and not c_hi and d_ci[0] > 0 and has_conf and inseparable:
        f.update(verdict="label_noise",
                 action=f"嵌入難分且標籤分歧(模型偶混淆)→ 重新裁決 {li}/{lj} 並寫下規則(勿 merge、勿多標)")
    elif not has_conf and not inseparable:
        f.update(verdict="clean", action="可分且模型無混淆:無歸因需要")
    elif not c_hi and not o_hi:
        f.update(verdict="clean", action="無顯著混淆/重疊")
    else:
        f.update(verdict="taxonomy_watch",
                 action="訊號混合/證據不足:監看,勿做不可逆動作(無模型混淆時不可宣稱標籤雜訊)")
    if not reference_trusted:  # imported/unverified reference -> firewall the label-trust verdicts
        _apply_reference_firewall(f)
    return f


def consistency_findings(emb_by_class: dict, confusion: dict | None = None,
                         n_gt: dict | None = None, max_pairs: int = 20,
                         min_sep_report: float = 0.2, adapt_rescued: dict | None = None,
                         reference_trusted: bool = True) -> list[dict]:
    """Per class-pair attribution over golden embeddings (+ optional eval confusion).
    Orients each pair toward its stronger model-confusion direction. Keeps only actionable pairs
    (a verdict beyond 'clean', or sep_err high enough to be worth showing). Sorted worst-first.

    ``adapt_rescued`` (from a saved adapt-embedding report; {frozenset(pair): rescued_bool}): when a
    pair the consistency layer calls a taxonomy/inseparable dead-end was RESCUED by a learned
    projection (CV-verified in adapt-embedding), flip it to representation_fixable — "don't merge,
    it's an encoder limit, not a definition dead-end"."""
    classes = sorted(c for c, e in emb_by_class.items() if np.atleast_2d(np.asarray(e)).shape[0] >= 1)
    conf = confusion or {}
    out: list[dict] = []
    for a in range(len(classes)):
        for b in range(a + 1, len(classes)):
            ci, cj = classes[a], classes[b]
            st = pair_stats(emb_by_class[ci], emb_by_class[cj])
            if st is None:
                continue
            # orient i->j toward the stronger confusion direction (so C_ij is the dominant cell)
            if conf.get(f"{cj}->{ci}", 0) > conf.get(f"{ci}->{cj}", 0):
                li, lj, ni, nj, O, O_ci = cj, ci, st["nj"], st["ni"], st["O_ji"], st["O_ji_ci"]
            else:
                li, lj, ni, nj, O, O_ci = ci, cj, st["ni"], st["nj"], st["O_ij"], st["O_ij_ci"]
            f = _attribute(li, lj, ni=ni, nj=nj, k=st["k"], sep_err=st["sep_err"], sep_ci=st["sep_ci"],
                           O=O, O_ci=O_ci, confusion=confusion, n_gt=n_gt,
                           reference_trusted=reference_trusted)
            if adapt_rescued is not None:
                resc = adapt_rescued.get(frozenset(f["pair"]))
                if resc is not None:
                    f["representation_fixable"] = bool(resc)
                    if resc and f["verdict"] in ("taxonomy", "taxonomy_watch", "inseparable_embedding"):
                        f["action"] = (f"別 merge:學到的投影已能分開 {li}/{lj}(CV 驗證)→ 表徵問題,"
                                       f"非 taxonomy 死路;套用 adapt-embedding 或換更強編碼器")
            if f["verdict"] in ("clean", "separable_embedding") and f["sep_err"] <= min_sep_report:
                continue
            out.append(f)
    out.sort(key=lambda r: -r["sep_err"])
    return out[:max_pairs]
