import numpy as np

from vix.core.analytics import EmbItem, cross_split_leakage, harmful_ranking, suspected_new_classes
from vix.core.explain import explain_image


def _it(id, label, emb, conf=1.0, split=""):
    return EmbItem(id, label, np.array(emb, dtype=float), conf, split=split)


def test_suspected_new_classes():
    ref = [_it(f"a{i}", "a", [1, 0, 0]) for i in range(5)] + [_it(f"b{i}", "b", [0, 1, 0]) for i in range(5)]
    query = [_it("n1", "a", [0, 0, 1]), _it("n2", "a", [0, 0, 1]), _it("known", "a", [1, 0, 0])]
    clusters = suspected_new_classes(query, ref, novelty_radius=0.3, cluster_distance=0.2)
    ids = {i for c in clusters for i in c["ids"]}
    assert "n1" in ids and "n2" in ids
    assert "known" not in ids


def test_cross_split_leakage():
    items = [
        _it("x", "a", [1, 0], split="train"),
        _it("xv", "a", [1, 0], split="val"),
        _it("y", "a", [0, 1], split="train"),
    ]
    leaks = cross_split_leakage(items, max_distance=0.001)
    assert any(set(lk["splits"]) == {"train", "val"} for lk in leaks)


def test_harmful_ranking_puts_worst_first():
    items = [_it("bad", "a", [1, 0]), _it("ok", "a", [1, 0.01]), _it("ok2", "a", [1, 0.02])]
    res = harmful_ranking(items, label_issue_ids={"bad"}, duplicate_ids={"bad"})
    assert res[0]["id"] == "bad"
    assert "suspected_label_error" in res[0]["reasons"]


def test_explain_image_drilldown():
    r = explain_image("a", confidence=0.1, knn_dist=0.9, conf_thr=0.5, dist_thr=0.5)
    axes = [a["axis"] for a in r["axes"]]
    assert "confidence" in axes and "knn_dist" in axes
    assert set(r["failing_axes"]) >= {"confidence", "knn_dist"}
    assert "被攔" in r["summary"]
