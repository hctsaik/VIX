"""Dataset analytics — the dataset-management workhorse (pure, FiftyOne-free).

Operates on lightweight ``EmbItem`` records (id, label, embedding, confidence)
so every function is unit-testable without FiftyOne/GPU. The "raw material"
(embeddings, labels, confidence) comes from the adapter; these turn it into the
answers a CV engineer actually wants:

    suspected_label_errors   S2  kNN-majority label disagreement
    near_duplicate_groups    S3  cosine clustering (union-find)
    class_distribution       S5  label counts
    coverage_gaps            S5  per-class sparse subgroups + under-represented
    coverage_delta           S4  how much a new batch covers unseen regions
    active_learning_ranking  S6  uncertainty + diversity ranking
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from .scorer import _l2norm


@dataclass
class EmbItem:
    id: str
    label: str
    embedding: np.ndarray
    confidence: float = 1.0
    batch: str = ""   # time dimension (e.g. "2026w22")
    split: str = ""   # "train" / "val" / "test"


@dataclass
class LabelIssue:
    id: str
    given_label: str
    suggested_label: str
    disagreement: float  # fraction of kNN majority that disagrees, [0,1]


def _stack(items: list[EmbItem]) -> tuple[list[str], np.ndarray, np.ndarray]:
    ids = [it.id for it in items]
    labels = np.array([it.label for it in items])
    mat = _l2norm(np.vstack([np.asarray(it.embedding, dtype=float) for it in items]))
    return ids, labels, mat


def _label_issue(i, ids, labels, nn, kk) -> "LabelIssue | None":
    nn_labels = labels[nn]
    vals, counts = np.unique(nn_labels, return_counts=True)
    maj = vals[int(np.argmax(counts))]
    if maj != labels[i]:
        maj_count = int(counts.max())
        own_count = int((nn_labels == labels[i]).sum())
        return LabelIssue(ids[i], str(labels[i]), str(maj), (maj_count - own_count) / kk)
    return None


def suspected_label_errors(
    items: list[EmbItem], k: int = 10, use_lsh: bool | None = None
) -> list[LabelIssue]:
    """Flag samples whose kNN-majority label disagrees with their given label.

    ``use_lsh`` restricts the neighbour search to LSH bucket-mates so this scales
    sub-quadratically on large datasets (auto-enabled above ~2000 items).
    """
    n = len(items)
    if n < 2:
        return []
    ids, labels, M = _stack(items)
    if use_lsh is None:
        use_lsh = n > 2000
    kk = min(k, n - 1)
    issues: list[LabelIssue] = []

    if use_lsh:
        from .lsh import neighbor_map

        nbrs = neighbor_map(M)
        for i in range(n):
            cand = list(nbrs.get(i, ()))
            if not cand:
                continue
            sims = M[cand] @ M[i]
            top = np.argsort(-sims)[: min(kk, len(cand))]
            issue = _label_issue(i, ids, labels, [cand[t] for t in top], min(kk, len(cand)))
            if issue:
                issues.append(issue)
    else:
        sims = M @ M.T
        np.fill_diagonal(sims, -np.inf)
        for i in range(n):
            nn = np.argpartition(-sims[i], kk - 1)[:kk]
            issue = _label_issue(i, ids, labels, nn, kk)
            if issue:
                issues.append(issue)

    issues.sort(key=lambda x: -x.disagreement)
    return issues


def near_duplicate_groups(
    items: list[EmbItem], max_distance: float = 0.05, use_lsh: bool | None = None
) -> list[list[str]]:
    """Group items whose pairwise cosine distance < ``max_distance`` (union-find).

    ``use_lsh`` only compares LSH bucket-mates instead of all O(n²) pairs, so
    this scales to large datasets (auto-enabled above ~2000 items).
    """
    n = len(items)
    if n < 2:
        return []
    ids, _labels, M = _stack(items)
    if use_lsh is None:
        use_lsh = n > 2000
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    if use_lsh:
        from .lsh import candidate_pairs

        for i, j in candidate_pairs(M):
            if (1.0 - float(M[i] @ M[j])) < max_distance:
                union(i, j)
    else:
        dist = 1.0 - (M @ M.T)
        for i in range(n):
            for j in range(i + 1, n):
                if dist[i, j] < max_distance:
                    union(i, j)

    groups: dict[int, list[str]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(ids[i])
    return [g for g in groups.values() if len(g) >= 2]


def near_dup_label_conflicts(items: list[EmbItem], max_distance: float = 0.03) -> list[dict]:
    """Near-duplicate groups whose members carry CONFLICTING labels — a causal-certain label error:
    near-identical images cannot legitimately have different labels, so at least one annotation is wrong.
    Tight ``max_distance`` (default 0.03) keeps it to genuinely near-identical pairs, not merely similar
    ones. ADVISORY: surfaces the contradiction (which to review), never resolves it. Returns
    [{"ids": [...], "labels": {label: count}}] sorted by group size."""
    by_id = {it.id: it.label for it in items}
    out = []
    for g in near_duplicate_groups(items, max_distance):
        labs = [by_id.get(i, "") for i in g]
        if len(set(labs)) > 1:  # near-identical yet labelled differently => at least one is wrong
            out.append({"ids": g, "labels": dict(Counter(labs))})
    out.sort(key=lambda r: -len(r["ids"]))
    return out


def class_distribution(items: list[EmbItem]) -> dict[str, int]:
    return dict(Counter(it.label for it in items))


def coverage_gaps(
    items: list[EmbItem], k: int = 5, sparse_quantile: float = 0.9, target: float | None = None
) -> dict[str, dict]:
    """Per-class: count, under-represented flag, sparse ids, and 'need' = how many
    more samples to reach the target (default: median class size)."""
    by: dict[str, list[EmbItem]] = defaultdict(list)
    for it in items:
        by[it.label].append(it)
    counts = {c: len(v) for c, v in by.items()}
    median = float(np.median(list(counts.values()))) if counts else 0.0
    tgt = target if target is not None else median
    out: dict[str, dict] = {}
    for c, group in by.items():
        sparse_ids: list[str] = []
        if len(group) >= 2:
            M = _l2norm(np.vstack([np.asarray(g.embedding, float) for g in group]))
            d = 1.0 - (M @ M.T)
            np.fill_diagonal(d, np.inf)
            kk = min(k, len(group) - 1)
            dens = np.array([np.partition(d[i], kk - 1)[:kk].mean() for i in range(len(group))])
            thr = float(np.quantile(dens, sparse_quantile))
            sparse_ids = [group[i].id for i in range(len(group)) if dens[i] >= thr]
        # explicit --target: under = below the absolute target (even if classes are balanced);
        # default (no target): the relative-imbalance heuristic vs the median class size.
        under = counts[c] < tgt if target is not None else counts[c] < 0.5 * median
        out[c] = {
            "count": counts[c],
            "under_represented": under,
            "need": max(0, int(np.ceil(tgt)) - counts[c]) if under else 0,
            "sparse_ids": sparse_ids,
        }
    return out


def coverage_delta(
    new_items: list[EmbItem], existing_items: list[EmbItem], radius: float = 0.2
) -> dict:
    """Fraction of the new batch that falls in regions not covered by existing data."""
    if not new_items:
        return {"novel_fraction": 0.0, "novel_ids": []}
    if not existing_items:
        return {"novel_fraction": 1.0, "novel_ids": [it.id for it in new_items]}
    E = _l2norm(np.vstack([np.asarray(e.embedding, float) for e in existing_items]))
    novel: list[str] = []
    for it in new_items:
        q = _l2norm(np.asarray(it.embedding, float).reshape(1, -1)).ravel()
        nearest = 1.0 - float((E @ q).max())
        if nearest > radius:
            novel.append(it.id)
    return {"novel_fraction": len(novel) / len(new_items), "novel_ids": novel}


def active_learning_ranking(
    candidates: list[EmbItem],
    existing_items: list[EmbItem],
    budget: int,
    alpha: float = 0.5,
    return_reasons: bool = False,
):
    """Rank unlabeled candidates by uncertainty (1-conf) + diversity (novelty + spread).

    Returns a list of ids, or (with ``return_reasons``) a list of dicts carrying the
    per-sample uncertainty / novelty / score so a labeler understands why it was picked.
    """
    if not candidates:
        return []
    C = _l2norm(np.vstack([np.asarray(c.embedding, float) for c in candidates]))
    if existing_items:
        E = _l2norm(np.vstack([np.asarray(e.embedding, float) for e in existing_items]))
        novelty = np.array([1.0 - float((E @ C[i]).max()) for i in range(len(candidates))])
    else:
        novelty = np.ones(len(candidates))
    span = np.ptp(novelty)
    novelty = (novelty - novelty.min()) / (span + 1e-12)
    uncertainty = np.array([1.0 - c.confidence for c in candidates])
    score = alpha * uncertainty + (1.0 - alpha) * novelty

    selected: list = []
    remaining = set(range(len(candidates)))
    while remaining and len(selected) < budget:
        i = max(remaining, key=lambda x: score[x])
        if return_reasons:
            selected.append({
                "id": candidates[i].id,
                "uncertainty": round(float(uncertainty[i]), 3),
                "novelty": round(float(novelty[i]), 3),
                "score": round(float(score[i]), 3),
            })
        else:
            selected.append(candidates[i].id)
        remaining.discard(i)
        sim_to_i = C @ C[i]  # penalise candidates similar to the just-picked (diversity)
        for j in remaining:
            score[j] -= (1.0 - alpha) * max(0.0, float(sim_to_i[j])) * 0.5
    return selected


def cross_period_drift(
    period_a: list[EmbItem],
    period_b: list[EmbItem],
    shift_threshold: float = 0.15,
    top: int = 3,
) -> dict[str, dict]:
    """Per-class definition drift across two time periods (S7).

    Returns centroid shift, an alert flag, and the period-B samples that drifted
    farthest from the period-A centroid (the "most different sample" evidence).
    """
    a_by: dict[str, list[EmbItem]] = defaultdict(list)
    b_by: dict[str, list[EmbItem]] = defaultdict(list)
    for it in period_a:
        a_by[it.label].append(it)
    for it in period_b:
        b_by[it.label].append(it)

    out: dict[str, dict] = {}
    for c in set(a_by) & set(b_by):
        A = _l2norm(np.vstack([np.asarray(x.embedding, float) for x in a_by[c]]))
        ca = _l2norm(A.mean(axis=0, keepdims=True)).ravel()
        b_items = b_by[c]
        B = _l2norm(np.vstack([np.asarray(x.embedding, float) for x in b_items]))
        cb = _l2norm(B.mean(axis=0, keepdims=True)).ravel()
        shift = float(1.0 - float(np.dot(ca, cb)))
        dists = 1.0 - (B @ ca)
        order = np.argsort(-dists)[:top]
        out[c] = {
            "shift": shift,
            "alert": shift > shift_threshold,
            "representatives": [(b_items[i].id, float(dists[i])) for i in order],
        }
    return out


def suspected_new_classes(
    query_items: list[EmbItem],
    reference_items: list[EmbItem],
    novelty_radius: float = 0.3,
    cluster_distance: float = 0.2,
) -> list[dict]:
    """Open-set detection (U1): query items far from EVERY known sample, clustered.

    Returns candidate new-class clusters so they are surfaced explicitly instead
    of being silently absorbed into the nearest existing class.
    """
    if not reference_items or not query_items:
        return []
    R = _l2norm(np.vstack([np.asarray(r.embedding, float) for r in reference_items]))
    novel = [
        it
        for it in query_items
        if (1.0 - float((R @ _l2norm(np.asarray(it.embedding, float).reshape(1, -1)).ravel()).max()))
        > novelty_radius
    ]
    if not novel:
        return []
    groups = near_duplicate_groups(novel, max_distance=cluster_distance) if len(novel) > 1 else []
    grouped = {i for g in groups for i in g}
    clusters = [{"cluster": f"new_cluster_{k}", "ids": g} for k, g in enumerate(groups)]
    for k, it in enumerate(i for i in novel if i.id not in grouped):
        clusters.append({"cluster": f"new_singleton_{k}", "ids": [it.id]})
    return clusters


def cross_split_leakage(items: list[EmbItem], max_distance: float = 0.05) -> list[dict]:
    """Train/val/test leakage (U3): near-duplicate groups spanning >1 split."""
    split_of = {it.id: it.split for it in items}
    leaks = []
    for g in near_duplicate_groups(items, max_distance):
        splits = {split_of.get(i, "") for i in g} - {""}
        if len(splits) > 1:
            leaks.append({"ids": g, "splits": sorted(splits)})
    return leaks


def harmful_ranking(
    items: list[EmbItem],
    label_issue_ids: set[str] | None = None,
    duplicate_ids: set[str] | None = None,
    k: int = 5,
    weights: tuple[float, float, float] = (0.5, 0.25, 0.25),
    top: int = 50,
) -> list[dict]:
    """Rank the most harmful samples (U5): label error + duplicate + outlier."""
    label_issue_ids = label_issue_ids or set()
    duplicate_ids = duplicate_ids or set()
    w_lbl, w_dup, w_out = weights
    n = len(items)
    if n == 0:
        return []
    ids, _labels, M = _stack(items)
    sims = M @ M.T
    np.fill_diagonal(sims, -np.inf)
    kk = min(k, n - 1) if n > 1 else 0

    out = []
    for i, it in enumerate(items):
        if kk:
            nn = np.argpartition(-sims[i], kk - 1)[:kk]
            outlier = float((1.0 - sims[i][nn]).mean())
        else:
            outlier = 1.0
        is_lbl = it.id in label_issue_ids
        is_dup = it.id in duplicate_ids
        harm = w_lbl * is_lbl + w_dup * is_dup + w_out * outlier
        reasons = []
        if is_lbl:
            reasons.append("suspected_label_error")
        if is_dup:
            reasons.append("duplicate")
        if outlier > 0.5:
            reasons.append("outlier")
        out.append({"id": it.id, "harm": float(harm), "reasons": reasons})
    out.sort(key=lambda x: -x["harm"])
    return out[:top]
