"""Round 22 補強的回歸測試:
  - export 不再因不同子目錄同名檔(basename 碰撞)而靜默覆寫/遺失資料
"""

import numpy as np

from vix.core.exporter import DatasetExporter
from vix.types import BBox, Detection


def _det(label):
    return Detection(label, 0.9, BBox(0.5, 0.5, 0.2, 0.2), embedding=np.array([1.0, 0.0]))


def test_export_no_basename_collision(tmp_path):
    recs = [("a/img.png", [_det("x")]), ("b/img.png", [_det("x")])]  # same basename, different dirs
    res = DatasetExporter(["x"]).export(recs, tmp_path / "out")
    assert res["n_images"] == 2
    labels = sorted(p.name for p in (tmp_path / "out" / "labels" / "train").glob("*.txt"))
    assert len(labels) == 2  # both survive (img.txt + img_1.txt), no silent overwrite
    assert res["n_labels"] == 2
