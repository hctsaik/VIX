from vix.core.labelmap import merge_class_maps, migration_diff, relabel, rollback


def test_merge_with_overrides():
    a = {0: "car", 1: "truck"}
    b = {0: "vehicle_car", 1: "heavy_vehicle"}
    res = merge_class_maps(a, b, name_overrides={"vehicle_car": "car", "heavy_vehicle": "truck"})
    assert res["unified_names"] == ["car", "truck"]
    assert res["remap_a"] == {0: 0, 1: 1}
    assert res["remap_b"] == {0: 0, 1: 1}
    assert res["common"] == ["car", "truck"]
    assert res["orphans"] == []


def test_merge_conflict_report():
    res = merge_class_maps({0: "car"}, {0: "bus"})
    assert "car" in res["only_in_a"] and "bus" in res["only_in_b"]
    assert set(res["needs_decision"]) == {"car", "bus"}
    assert res["unified_names"] == ["bus", "car"]


def test_relabel_merge_and_rollback():
    recs = [("1", "sedan"), ("2", "hatchback"), ("3", "truck")]
    new, changes = relabel(recs, {"sedan": "passenger_car", "hatchback": "passenger_car"})
    assert dict(new) == {"1": "passenger_car", "2": "passenger_car", "3": "truck"}
    assert migration_diff(changes)["total_changed"] == 2
    assert rollback(new, changes) == recs  # merge undone via change log
