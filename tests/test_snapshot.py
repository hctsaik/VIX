from vix.core.manifest import Manifest, ManifestEntry
from vix.core.snapshot import create_snapshot, restore


def test_snapshot_and_restore(tmp_path):
    mp = tmp_path / "manifest.jsonl"
    m = Manifest(mp)
    m.append(ManifestEntry.create("imgs/g1.png", "b1", tags=["golden"], vix_hash="g1"))
    m.append(ManifestEntry.create("imgs/g2.png", "b1", tags=["golden"], vix_hash="g2"))
    m.append(ManifestEntry.create("imgs/r1.png", "b1", tags=["review"], vix_hash="r1"))

    out = tmp_path / "snap.json"
    snap = create_snapshot(mp, out, "v1", thresholds_meta={"conf_pct": 5})
    assert snap["n_golden"] == 2 and snap["n_excluded"] == 1

    r = restore(out)
    assert {c["vix_hash"] for c in r["composition"]} == {"g1", "g2"}
    assert r["params"] == {"conf_pct": 5}
    assert len(r["excluded"]) == 1


def test_snapshot_content_hash_is_deterministic(tmp_path):
    mp = tmp_path / "m.jsonl"
    m = Manifest(mp)
    m.append(ManifestEntry.create("imgs/g1.png", "b1", tags=["golden"], vix_hash="g1"))
    s1 = create_snapshot(mp, tmp_path / "s1.json", "v1", thresholds_meta={"x": 1})
    s2 = create_snapshot(mp, tmp_path / "s2.json", "v1", thresholds_meta={"x": 1})
    assert s1["content_hash"] == s2["content_hash"]
