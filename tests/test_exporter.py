import yaml

from vix.core.exporter import DatasetExporter
from vix.types import BBox, Detection


def test_export_yolo_txt_and_yaml(tmp_path):
    exp = DatasetExporter(["bubble", "reflection"])
    records = [
        ("imgs/x1.png", [Detection("bubble", 0.9, BBox(0.5, 0.5, 0.2, 0.2))]),
        (
            "imgs/x2.png",
            [
                Detection("reflection", 0.8, BBox(0.1, 0.1, 0.05, 0.05)),
                Detection("unknown_class", 0.5, BBox(0.0, 0.0, 0.0, 0.0)),
            ],
        ),
    ]
    res = exp.export(records, tmp_path / "out", split="train")

    assert res["n_images"] == 2
    assert res["n_labels"] == 2     # two known-class detections written
    assert res["n_skipped"] == 1    # unknown_class skipped

    x1 = (tmp_path / "out" / "labels" / "train" / "x1.txt").read_text(encoding="utf-8").strip()
    assert x1.startswith("0 ")      # bubble -> index 0

    data = yaml.safe_load((tmp_path / "out" / "data.yaml").read_text(encoding="utf-8"))
    assert data["names"] == {0: "bubble", 1: "reflection"}
    assert data["train"] == "images/train"
