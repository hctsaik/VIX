import numpy as np

from vix.core.analytics import EmbItem, near_duplicate_groups, suspected_label_errors


def _two_clusters(n=40, d=8, seed=0):
    rng = np.random.RandomState(seed)
    items = []
    for i in range(n):
        v = np.zeros(d); v[0] = 1.0; v += 0.01 * rng.randn(d)
        items.append(EmbItem(f"a{i}", "a", v))
    for i in range(n):
        v = np.zeros(d); v[1] = 1.0; v += 0.01 * rng.randn(d)
        items.append(EmbItem(f"b{i}", "b", v))
    return items


def test_lsh_label_errors_agree_with_brute_on_mislabel():
    items = _two_clusters()
    items.append(EmbItem("bad", "a", np.eye(8)[1]))  # 'a' sitting in 'b' cluster
    brute = suspected_label_errors(items, k=5, use_lsh=False)
    lsh = suspected_label_errors(items, k=5, use_lsh=True)
    assert "bad" in [x.id for x in brute]
    assert "bad" in [x.id for x in lsh]


def test_lsh_dedup_finds_exact_duplicate_pair():
    items = _two_clusters(n=30)
    items.append(EmbItem("dupX", "a", np.eye(8)[0]))
    items.append(EmbItem("dupY", "a", np.eye(8)[0]))
    groups = near_duplicate_groups(items, max_distance=0.001, use_lsh=True)
    assert any({"dupX", "dupY"} <= set(g) for g in groups)
