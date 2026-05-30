import numpy as np

from vix.core.analytics import EmbItem
from vix.core.explain import explain
from vix.core.triage import review_queue


def _it(id, emb, conf):
    return EmbItem(id, "a", np.array(emb, dtype=float), conf)


def test_review_queue_ranks_risky_first():
    ref = [_it(f"g{i}", [1, 0], 1.0) for i in range(5)]
    cands = [_it("safe", [1, 0], 0.95), _it("lowconf", [1, 0], 0.1), _it("novel", [0, 1], 0.95)]
    q = review_queue(cands, ref, k=3, dist_norm=0.5)
    ids = [r.id for r in q]
    assert ids[0] in ("lowconf", "novel")
    assert ids[-1] == "safe"
    assert q[0].reasons  # carries reasons for the rank


def test_explain_sentence():
    s = explain(["low_conf", "far_from_known"], {"conf": 0.2, "knn_dist": 0.6})
    assert "信心" in s and "攔" in s
    assert explain([]).startswith("通過")
