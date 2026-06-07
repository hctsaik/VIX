"""Pure GT-label importer (yolo/voc/coco -> normalised Detections). The on-ramp seam.

No FiftyOne / torch / network. Closes "the parser must actually invert our own export".
"""

import json

import pytest

from vix.core.exporter import DatasetExporter
from vix.core.label_import import (
    coco_to_dets,
    parse_labels,
    voc_to_dets,
    yolo_txt_to_dets,
)
from vix.types import BBox, Detection


def _png(p):
    # 1x1 PNG bytes — parsers never decode, but _images_under globs by extension and exporter copies.
    p.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
        b"\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )


def test_yolo_txt_to_dets_maps_class_and_box(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    _png(tmp_path / "images" / "a.png")
    (tmp_path / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    out = yolo_txt_to_dets(tmp_path / "images", names=["pothole"])
    dets = out[str(tmp_path / "images" / "a.png")]
    assert len(dets) == 1 and dets[0].label == "pothole"
    assert dets[0].bbox.as_tuple() == (0.5, 0.5, 0.2, 0.2)
    assert dets[0].confidence == 1.0  # ground-truth ingest


def test_yolo_bom_label_file_tolerated(tmp_path):
    _png(tmp_path / "a.png")  # Windows editors / PowerShell Set-Content prepend a UTF-8 BOM
    (tmp_path / "a.txt").write_bytes("﻿0 0.5 0.5 0.2 0.2\n".encode("utf-8"))
    out = yolo_txt_to_dets(tmp_path, names=["pothole"])
    assert out[str(tmp_path / "a.png")][0].label == "pothole"


def test_yolo_empty_label_file_yields_empty_list(tmp_path):
    _png(tmp_path / "a.png")  # no sibling .txt
    out = yolo_txt_to_dets(tmp_path, names=["x"])
    assert out[str(tmp_path / "a.png")] == []  # key present, no boxes (not missing)


def test_yolo_unknown_class_index_raises(tmp_path):
    _png(tmp_path / "a.png")
    (tmp_path / "a.txt").write_text("7 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="類別索引"):
        yolo_txt_to_dets(tmp_path, names=["only_class_0"])


def test_yolo_nonnumeric_token_raises(tmp_path):
    _png(tmp_path / "a.png")
    (tmp_path / "a.txt").write_text("0 left 0.5 0.1 0.1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="非數值"):
        yolo_txt_to_dets(tmp_path, names=["c"])


def test_coco_xywh_to_normalised_box(tmp_path):
    doc = {
        "images": [{"id": 1, "file_name": "a.png", "width": 100, "height": 200}],
        "annotations": [{"image_id": 1, "category_id": 5, "bbox": [10, 20, 40, 60]}],
        "categories": [{"id": 5, "name": "crack"}],
    }
    jp = tmp_path / "instances.json"
    jp.write_text(json.dumps(doc), encoding="utf-8")
    out = coco_to_dets(jp)
    dets = next(v for v in out.values() if v)
    b = dets[0].bbox
    assert dets[0].label == "crack"
    assert abs(b.cx - 0.30) < 1e-6 and abs(b.cy - 0.25) < 1e-6  # (10+20)/100, (20+30)/200
    assert abs(b.w - 0.40) < 1e-6 and abs(b.h - 0.30) < 1e-6


def test_coco_zero_size_with_annotation_raises(tmp_path):
    doc = {  # an ANNOTATED image with no size must not silently drop its box
        "images": [{"id": 1, "file_name": "a.png", "width": 0, "height": 0}],
        "annotations": [{"image_id": 1, "category_id": 1, "bbox": [1, 1, 2, 2]}],
        "categories": [{"id": 1, "name": "x"}],
    }
    jp = tmp_path / "instances.json"
    jp.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="width/height"):
        coco_to_dets(jp)


def test_coco_zero_size_no_annotation_ok(tmp_path):
    doc = {  # a box-free image with no size is fine (nothing to drop)
        "images": [{"id": 1, "file_name": "a.png", "width": 0, "height": 0}],
        "annotations": [], "categories": [{"id": 1, "name": "x"}],
    }
    jp = tmp_path / "instances.json"
    jp.write_text(json.dumps(doc), encoding="utf-8")
    assert coco_to_dets(jp) == {str(tmp_path / "a.png") if (tmp_path / "a.png").exists() else "a.png": []}


def test_voc_xyxy_to_normalised_box(tmp_path):
    xml = """<annotation><size><width>100</width><height>100</height></size>
    <object><name>pothole</name><bndbox><xmin>10</xmin><ymin>20</ymin>
    <xmax>30</xmax><ymax>60</ymax></bndbox></object></annotation>"""
    (tmp_path / "a.xml").write_text(xml, encoding="utf-8")
    out = voc_to_dets(tmp_path)
    dets = next(v for v in out.values() if v)
    b = dets[0].bbox
    # matches dogfood_eval_yolo._gt math: ((10+30)/2)/100, ((20+60)/2)/100, 20/100, 40/100
    assert abs(b.cx - 0.20) < 1e-6 and abs(b.cy - 0.40) < 1e-6
    assert abs(b.w - 0.20) < 1e-6 and abs(b.h - 0.40) < 1e-6


def test_voc_zero_size_with_object_raises(tmp_path):
    xml = "<annotation><size><width>0</width><height>0</height></size>" \
          "<object><name>x</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>2</xmax><ymax>2</ymax></bndbox></object></annotation>"
    (tmp_path / "a.xml").write_text(xml, encoding="utf-8")
    with pytest.raises(ValueError, match="<size>"):
        voc_to_dets(tmp_path)


def test_voc_no_size_no_object_ok(tmp_path):
    (tmp_path / "a.xml").write_text("<annotation><size><width>0</width><height>0</height></size></annotation>", encoding="utf-8")
    assert all(v == [] for v in voc_to_dets(tmp_path).values())  # box-free -> nothing to drop


def test_export_then_import_roundtrip_identical_boxes(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _png(src / "img1.png")
    orig = [Detection("pothole", 1.0, BBox(0.5, 0.5, 0.25, 0.4)),
            Detection("crack", 1.0, BBox(0.2, 0.3, 0.1, 0.1))]
    dst = tmp_path / "out"
    DatasetExporter(["pothole", "crack"]).export(
        [(str(src / "img1.png"), orig)], dst, copy_images=True)
    # import back from the exported tree (ultralytics images/<->labels/ layout + data.yaml names)
    import yaml
    names = yaml.safe_load((dst / "data.yaml").read_text(encoding="utf-8"))["names"]
    out = yolo_txt_to_dets(dst / "images", names=names)
    got = next(v for v in out.values() if v)
    assert sorted(d.label for d in got) == ["crack", "pothole"]
    by = {d.label: d.bbox for d in got}
    assert by["pothole"].as_tuple() == pytest.approx((0.5, 0.5, 0.25, 0.4), abs=1e-6)
    assert by["crack"].as_tuple() == pytest.approx((0.2, 0.3, 0.1, 0.1), abs=1e-6)


def test_detect_format_yolo_voc_coco(tmp_path):
    from vix.core.label_import import detect_format
    # YOLO: labels/ + data.yaml
    (tmp_path / "y" / "images").mkdir(parents=True); (tmp_path / "y" / "labels").mkdir()
    _png(tmp_path / "y" / "images" / "a.png")
    (tmp_path / "y" / "labels" / "a.txt").write_text("0 .5 .5 .2 .2", encoding="utf-8")
    (tmp_path / "y" / "data.yaml").write_text("names: [pothole]", encoding="utf-8")
    dy = detect_format(tmp_path / "y")
    assert dy["fmt"] == "yolo" and dy["names"] == ["pothole"]
    # VOC: annotations/*.xml
    (tmp_path / "v" / "annotations").mkdir(parents=True)
    (tmp_path / "v" / "annotations" / "a.xml").write_text(
        "<annotation><size><width>10</width><height>10</height></size>"
        "<object><name>car</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>3</xmax><ymax>3</ymax></bndbox></object></annotation>",
        encoding="utf-8")
    dv = detect_format(tmp_path / "v")
    assert dv["fmt"] == "voc" and dv["names"] == ["car"]
    # COCO: instances.json
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / "instances.json").write_text(json.dumps(
        {"images": [{"id": 1, "file_name": "a.png", "width": 10, "height": 10}],
         "annotations": [{"image_id": 1, "category_id": 1, "bbox": [1, 1, 2, 2]}],
         "categories": [{"id": 1, "name": "dog"}]}), encoding="utf-8")
    dc = detect_format(tmp_path / "c")
    assert dc["fmt"] == "coco" and dc["json_path"] and dc["names"] == ["dog"]
    # nothing -> None
    (tmp_path / "empty").mkdir()
    assert detect_format(tmp_path / "empty")["fmt"] is None


def test_parse_labels_dispatch_voc(tmp_path):
    (tmp_path / "annotations").mkdir()
    _png(tmp_path / "a.png")
    (tmp_path / "annotations" / "a.xml").write_text(
        "<annotation><size><width>10</width><height>10</height></size>"
        "<object><name>z</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>3</xmax><ymax>3</ymax></bndbox></object></annotation>",
        encoding="utf-8")
    out = parse_labels(tmp_path, "voc")
    assert any(v and v[0].label == "z" for v in out.values())
