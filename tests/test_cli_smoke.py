from PIL import Image

from vix.cli import main
from vix.core.manifest import Manifest


def test_cli_ingest_memory(tmp_path):
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    for i in range(3):
        Image.new("RGB", (16, 16), (i * 10, 0, 0)).save(imgs / f"{i}.png")

    ws = tmp_path / "ws"
    rc = main(
        ["--workspace", str(ws), "--adapter", "memory", "ingest", str(imgs), "--batch", "b1", "--golden"]
    )
    assert rc == 0

    m = Manifest.load(ws / "manifest.jsonl")
    assert len(m) == 3
    assert all("golden" in e.tags for e in m.entries())
    assert (ws / "vix.log").exists()  # logging wired
