import numpy as np

from vix.core.reference import FrozenReference, label_consistency


def _anchor():
    return {"a": np.array([[1.0, 0.0]] * 10), "b": np.array([[0.0, 1.0]] * 10)}


def test_centroid_shift_zero_when_same():
    ref = FrozenReference.build(_anchor())
    shifts = ref.centroid_shift(_anchor())
    assert all(v < 1e-9 for v in shifts.values())


def test_label_consistency_high_for_separable():
    assert label_consistency(_anchor(), _anchor(), k=3) > 0.99


def test_guard_triggers_on_definition_drift():
    ref = FrozenReference.build(_anchor())
    # New 'a' data that looks like 'b' -> centroid shift + consistency collapse
    rep = ref.guard({"a": np.array([[0.0, 1.0]] * 10)},
                    shift_threshold=0.15, consistency_drop_threshold=0.05)
    assert rep.triggered
    assert rep.max_shift > 0.5
    assert "centroid_shift" in rep.reasons


def test_guard_quiet_when_consistent():
    ref = FrozenReference.build(_anchor())
    rep = ref.guard({"a": np.array([[1.0, 0.02]] * 5)},
                    shift_threshold=0.15, consistency_drop_threshold=0.05)
    assert rep.triggered is False


def test_save_load_roundtrip(tmp_path):
    rng = np.random.RandomState(0)
    anchor = {"a": rng.randn(8, 4), "b": rng.randn(8, 4)}
    ref = FrozenReference.build(anchor)
    p = tmp_path / "ref.npz"
    ref.save(p)
    ref2 = FrozenReference.load(p)
    assert set(ref2.anchor_per_class) == {"a", "b"}
    assert abs(ref2.baseline_consistency - ref.baseline_consistency) < 1e-9
