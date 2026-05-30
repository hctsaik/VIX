import numpy as np

from vix.core.scorer import OutlierScorer, cosine_knn_distance, intra_class_knn_distances
from vix.types import BBox, Detection


def test_cosine_knn_distance_identical():
    nb = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    assert cosine_knn_distance([1.0, 0.0], nb, 3) < 1e-6


def test_cosine_knn_distance_orthogonal():
    nb = np.array([[0.0, 1.0], [0.0, 1.0]])
    assert abs(cosine_knn_distance([1.0, 0.0], nb, 2) - 1.0) < 1e-6


def test_empty_neighbors_is_inf():
    assert cosine_knn_distance([1.0, 0.0], np.zeros((0, 2)), 3) == float("inf")


def test_intra_class_excludes_self():
    E = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    out = intra_class_knn_distances(E, 2)
    assert out.shape == (3,)
    assert np.all(out < 1e-6)


def test_score_image_two_axis():
    scorer = OutlierScorer({"a": np.array([[1.0, 0.0]] * 5)}, k=3)
    dets = [Detection("a", 0.9, BBox(0.5, 0.5, 0.1, 0.1), embedding=np.array([1.0, 0.001]))]
    s = scorer.score_image(dets)
    assert s.conf_max == 0.9
    assert s.knn_dist < 1e-2
    assert dets[0].low_support is False  # 5 >= k=3


def test_unknown_class_is_maximally_novel():
    scorer = OutlierScorer({"a": np.array([[1.0, 0.0]] * 5)}, k=3)
    dets = [Detection("z", 0.9, BBox(0.5, 0.5, 0.1, 0.1), embedding=np.array([1.0, 0.0]))]
    s = scorer.score_image(dets)
    assert s.knn_dist == float("inf")
    assert dets[0].low_support is True
