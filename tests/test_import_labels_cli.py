"""import-labels: content-hash join (KS3) + loud-fail + PROVISIONAL (never golden)."""

import pytest

from vix.adapters.memory import InMemoryAdapter
from vix.config import Config
from vix.core.manifest import compute_hash
from vix.pipeline import import_labels
from vix.types import Tag

_PNG = (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
        b"\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82")


def _ds(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    (tmp_path / "images" / "a.png").write_bytes(_PNG)
    (tmp_path / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    return tmp_path


def test_import_labels_joins_by_content_hash(tmp_path):
    folder = _ds(tmp_path)
    cfg = Config(workspace=tmp_path / "ws"); cfg.ensure_dirs()
    ad = InMemoryAdapter()
    res = import_labels(ad, cfg, folder, fmt="yolo", names=["pothole"], batch="b1")
    assert res["n_images"] == 1 and res["n_boxes"] == 1 and res["classes"] == ["pothole"]
    h = compute_hash(folder / "images" / "a.png")  # the content-hash key, NOT the stem "a"
    found = {hh: (dets, tags) for hh, _s, dets, tags in ad.samples()}
    assert h in found, "labels did not join to the image by content hash"
    dets, tags = found[h]
    assert dets[0].label == "pothole" and Tag.PROVISIONAL in tags
    assert Tag.GOLDEN not in tags  # honesty: imported labels are NEVER golden


def test_import_labels_unknown_image_fails_loud(tmp_path):
    import json
    (tmp_path / "a.png").write_bytes(_PNG)  # an image exists, but COCO references a missing one
    doc = {"images": [{"id": 1, "file_name": "ghost.png", "width": 10, "height": 10}],
           "annotations": [{"image_id": 1, "category_id": 1, "bbox": [1, 1, 2, 2]}],
           "categories": [{"id": 1, "name": "x"}]}
    (tmp_path / "instances.json").write_text(json.dumps(doc), encoding="utf-8")
    cfg = Config(workspace=tmp_path / "ws"); cfg.ensure_dirs()
    with pytest.raises(ValueError, match="未匯入|不存在"):
        import_labels(InMemoryAdapter(), cfg, tmp_path, fmt="coco", batch="b1")


def test_export_after_diagnose_gives_diagnose_aware_error(tmp_path):
    # diagnose imports labels as PROVISIONAL (never golden) -> export must explain, not just "no golden"
    import pytest
    from vix.pipeline import export
    folder = _ds(tmp_path)
    cfg = Config(workspace=tmp_path / "ws"); cfg.ensure_dirs()
    ad = InMemoryAdapter()
    import_labels(ad, cfg, folder, fmt="yolo", names=["pothole"], batch="b1")  # PROVISIONAL
    with pytest.raises(ValueError, match="參照標籤|未覆核|resolve"):
        export(ad, cfg, ["pothole"], tmp_path / "out")


def test_import_labels_custom_label_dir(tmp_path):
    # labels live in a non-standard dir (not sibling labels/) -> --label-dir must find them (docs promise this)
    (tmp_path / "images").mkdir()
    (tmp_path / "ann").mkdir()
    (tmp_path / "images" / "a.png").write_bytes(_PNG)
    (tmp_path / "ann" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    cfg = Config(workspace=tmp_path / "ws"); cfg.ensure_dirs()
    ad = InMemoryAdapter()
    res = import_labels(ad, cfg, tmp_path, fmt="yolo", names=["pothole"], batch="b1",
                        label_dir=str(tmp_path / "ann"))
    assert res["n_images"] == 1 and res["n_boxes"] == 1


def test_import_labels_as_eval_tags_eval(tmp_path):
    folder = _ds(tmp_path)
    cfg = Config(workspace=tmp_path / "ws"); cfg.ensure_dirs()
    ad = InMemoryAdapter()
    import_labels(ad, cfg, folder, fmt="yolo", names=["pothole"], batch="b1", as_="eval")
    tags = next(t for _h, _s, _d, t in ad.samples())
    assert Tag.EVAL in tags and Tag.PROVISIONAL not in tags
