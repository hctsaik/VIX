from vix.core.manifest import Manifest, ManifestEntry, compute_hash


def test_compute_hash(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"hello world")
    h = compute_hash(f)
    assert len(h) == 64
    assert h == compute_hash(f)  # stable


def test_append_dedup_and_reload(tmp_path):
    mp = tmp_path / "manifest.jsonl"
    m = Manifest(mp)
    e1 = ManifestEntry.create("imgs/1.png", "batch1", vix_hash="h1")
    assert m.append(e1) is True
    assert m.append(e1) is False  # duplicate hash skipped
    assert len(m) == 1

    m2 = Manifest.load(mp)
    assert m2.has("h1")
    assert len(m2) == 1
    e2 = ManifestEntry.create("imgs/2.png", "batch1", vix_hash="h2")
    assert m2.append(e2) is True
    assert len(Manifest.load(mp)) == 2
