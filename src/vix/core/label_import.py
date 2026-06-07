"""Ground-truth label IMPORT — the missing on-ramp seam (landable-system Round 1-3).

`ingest` records images only; until now the only way a box entered VIX was a YOLO
prediction or a synthetic full-image box. A CV engineer arriving with an EXISTING
labelled dataset (the normal case) had no way to load their annotations to audit
them — the VOC/YOLO/COCO parsers lived only inside example scripts. This promotes
them into tested, importable core.

Pure / stdlib + the shared ``BBox``/``Detection`` types. No FiftyOne, no torch, no
numpy. Boxes come out YOLO-normalised (cx, cy, w, h) in [0, 1] — the same geometry
``exporter.py`` writes and ``eval_ingest`` consumes, so export -> import round-trips.

Each parser returns ``dict[str_image_path -> list[Detection]]`` keyed by the resolved
image path; the pipeline layer joins those paths to ingested images by CONTENT HASH
(never by filename stem — that silent mismatch is the bug this whole effort fixes).

HONESTY: imported labels are human-UNVERIFIED. Callers tag them ``Tag.PROVISIONAL``
(a diagnosis-only reference), never ``Tag.GOLDEN``.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

from ..types import BBox, Detection

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _names_map(names) -> dict[int, str]:
    """Accept a list (index->name), a dict (int or str keys -> name), or None."""
    if names is None:
        return {}
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(n) for i, n in enumerate(names)}


def _images_under(folder: Path) -> list[Path]:
    return [p for p in sorted(folder.rglob("*")) if p.suffix.lower() in IMG_EXTS]


# --- YOLO -------------------------------------------------------------------

def _yolo_label_path(img: Path, image_root: Path, label_dir: Path | None) -> Path | None:
    """Locate the .txt for an image across the common ultralytics layouts."""
    stem = img.stem + ".txt"
    cands: list[Path] = []
    if label_dir is not None:
        label_dir = Path(label_dir)
        try:
            rel = img.relative_to(image_root)
            cands.append(label_dir / rel.with_suffix(".txt"))
        except ValueError:
            pass
        cands.append(label_dir / stem)
    # ultralytics convention: .../images/.../x.jpg <-> .../labels/.../x.txt
    parts = list(img.parts)
    if "images" in parts:
        i = len(parts) - 1 - parts[::-1].index("images")
        parts[i] = "labels"
        cands.append(Path(*parts).with_suffix(".txt"))
    cands.append(img.with_suffix(".txt"))            # sibling
    cands.append(image_root / "labels" / stem)        # flat labels/ dir
    for c in cands:
        if c.exists():
            return c
    return None


def _parse_yolo_line(line: str, nmap: dict[int, str], src: Path) -> Detection:
    toks = line.split()
    if len(toks) < 5:
        raise ValueError(f"格式錯誤的 YOLO 標籤行(需 'cls cx cy w h'):{src}:「{line}」")
    try:
        cls = int(float(toks[0]))
        cx, cy, w, h = (float(t) for t in toks[1:5])
    except ValueError as exc:
        raise ValueError(f"YOLO 標籤含非數值 token:{src}:「{line}」") from exc
    if cls not in nmap:
        raise ValueError(
            f"YOLO 類別索引 {cls} 不在 names 對照中(請給 --data-yaml 或 --names):{src}"
        )
    return Detection(nmap[cls], 1.0, BBox(cx, cy, w, h))


def yolo_txt_to_dets(image_dir, names=None, label_dir=None) -> dict[str, list[Detection]]:
    """Read sibling/ultralytics YOLO .txt labels for every image under ``image_dir``."""
    image_dir = Path(image_dir)
    nmap = _names_map(names)
    out: dict[str, list[Detection]] = {}
    for img in _images_under(image_dir):
        txt = _yolo_label_path(img, image_dir, label_dir)
        dets: list[Detection] = []
        if txt is not None:
            for raw in txt.read_text(encoding="utf-8-sig").splitlines():  # tolerate a Windows BOM
                raw = raw.strip()
                if raw:
                    dets.append(_parse_yolo_line(raw, nmap, txt))
        out[str(img)] = dets
    return out


# --- COCO -------------------------------------------------------------------

def coco_to_dets(json_path, image_dir=None) -> dict[str, list[Detection]]:
    """Read a COCO ``instances.json`` (absolute xywh top-left) -> normalised dets."""
    json_path = Path(json_path)
    doc = json.loads(json_path.read_text(encoding="utf-8-sig"))
    base = Path(image_dir) if image_dir else json_path.parent
    cats = {c["id"]: c["name"] for c in doc.get("categories", [])}
    imgs = {im["id"]: im for im in doc.get("images", [])}
    by_img: dict[int, list[Detection]] = {}
    bad_size: set = set()
    for ann in doc.get("annotations", []):
        im = imgs.get(ann["image_id"])
        if im is None:
            continue
        W, H = float(im.get("width", 0)), float(im.get("height", 0))
        if W <= 0 or H <= 0:
            bad_size.add(ann["image_id"])  # an annotated image with no size -> can't normalise; don't silently drop
            continue
        x, y, w, h = (float(v) for v in ann["bbox"])
        label = cats.get(ann["category_id"], str(ann["category_id"]))
        det = Detection(label, 1.0, BBox((x + w / 2) / W, (y + h / 2) / H, w / W, h / H))
        by_img.setdefault(ann["image_id"], []).append(det)
    if bad_size:  # honesty: never silently discard boxes for images missing width/height
        eg = imgs[next(iter(bad_size))].get("file_name", "?")
        raise ValueError(f"COCO 有 {len(bad_size)} 張帶標註的影像缺有效 width/height,無法正規化(例:{eg}):{json_path}")
    out: dict[str, list[Detection]] = {}
    for iid, im in imgs.items():
        fn = im.get("file_name", "")
        p = (base / fn)
        key = str(p if p.exists() else fn)
        out[key] = by_img.get(iid, [])
    return out


# --- Pascal VOC -------------------------------------------------------------

def _voc_image_path(root, xml: Path, image_dir: Path | None) -> str:
    """Resolve the image a VOC xml describes (path / filename / stem search)."""
    p = root.findtext("path")
    if p and Path(p).exists():
        return str(Path(p))
    fn = root.findtext("filename")
    if image_dir is not None:
        image_dir = Path(image_dir)
        if fn and (image_dir / fn).exists():
            return str(image_dir / fn)
        for ext in IMG_EXTS:
            cand = image_dir / (xml.stem + ext)
            if cand.exists():
                return str(cand)
    if fn:
        return str(Path(fn))
    return str(xml.with_suffix(".jpg"))


def voc_to_dets(xml_dir, image_dir=None) -> dict[str, list[Detection]]:
    """Read per-image Pascal VOC xml (absolute xyxy) -> normalised dets."""
    xml_dir = Path(xml_dir)
    out: dict[str, list[Detection]] = {}
    for xml in sorted(xml_dir.rglob("*.xml")):
        root = ET.parse(xml).getroot()
        sz = root.find("size")
        W = float(sz.findtext("width", "0")) if sz is not None else 0.0
        H = float(sz.findtext("height", "0")) if sz is not None else 0.0
        key = _voc_image_path(root, xml, image_dir)
        dets: list[Detection] = []
        objs = root.findall("object")
        if (W <= 0 or H <= 0) and objs:  # objects present but no valid <size> -> would silently drop boxes
            raise ValueError(f"VOC 缺有效 <size>(width/height),無法正規化 {len(objs)} 個框:{xml}")
        if W > 0 and H > 0:
            for o in objs:
                b = o.find("bndbox")
                if b is None:
                    continue
                try:
                    x1, y1, x2, y2 = (float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"VOC bndbox 缺座標或非數值:{xml}") from exc
                label = o.findtext("name") or "object"
                dets.append(Detection(label, 1.0, BBox(
                    ((x1 + x2) / 2) / W, ((y1 + y2) / 2) / H, abs(x2 - x1) / W, abs(y2 - y1) / H)))
        out[key] = dets
    return out


def detect_format(folder) -> dict:
    """Sniff a folder and figure out the label format + helpers, so the user only supplies the path.
    Returns {"fmt": yolo|voc|coco|None, "json_path": str|None, "data_yaml": str|None, "names": list|None}.

    Order: COCO (a *.json with images+annotations) > VOC (*.xml with a bndbox) > YOLO (labels/ or sibling
    *.txt). For YOLO it also picks up a data.yaml's names; VOC/COCO class names come from the labels."""
    folder = Path(folder)
    out = {"fmt": None, "json_path": None, "data_yaml": None, "names": None}
    # COCO: a json holding images[] + annotations[] (+ categories[])
    for j in sorted(folder.rglob("*.json")):
        try:
            d = json.loads(j.read_text(encoding="utf-8-sig"))
        except (ValueError, OSError):
            continue
        if isinstance(d, dict) and isinstance(d.get("annotations"), list) and isinstance(d.get("images"), list):
            out.update(fmt="coco", json_path=str(j),
                       names=[c["name"] for c in d.get("categories", [])] or None)
            return out
    # VOC: an xml with an <object>/<bndbox>
    for x in sorted(folder.rglob("*.xml")):
        try:
            root = ET.parse(x).getroot()
        except ET.ParseError:
            continue
        if root.find(".//bndbox") is not None:
            names = sorted({o.findtext("name") for o in folder_objects(folder) if o.findtext("name")})
            out.update(fmt="voc", names=names or None)
            return out
    # YOLO: a labels/ dir or any sibling .txt next to an image; names from data.yaml if present
    has_txt = (folder / "labels").exists() or any(folder.rglob("*.txt"))
    if has_txt:
        dy = next((p for p in folder.rglob("*.yaml") if "data" in p.name.lower()), None) \
            or next(iter(folder.rglob("*.yaml")), None)
        names = None
        if dy is not None:
            try:
                import yaml
                names = yaml.safe_load(dy.read_text(encoding="utf-8")).get("names")
            except (ValueError, OSError):
                names = None
        out.update(fmt="yolo", data_yaml=(str(dy) if dy else None), names=names)
        return out
    return out


def folder_objects(folder):
    """Yield all VOC <object> elements under a folder (for class-name discovery)."""
    folder = Path(folder)
    for x in sorted(folder.rglob("*.xml")):
        try:
            root = ET.parse(x).getroot()
        except ET.ParseError:
            continue
        yield from root.findall("object")


def parse_labels(folder, fmt: str, names=None, label_dir=None, json_path=None) -> dict[str, list[Detection]]:
    """Dispatch by format. ``folder`` is the image root; VOC looks in folder/annotations
    (or folder) for xml, COCO uses ``json_path`` (or an instances*.json under folder)."""
    folder = Path(folder)
    fmt = fmt.lower()
    if fmt == "yolo":
        return yolo_txt_to_dets(folder, names=names, label_dir=label_dir)
    if fmt == "voc":
        xmls = next((d for d in (folder / "annotations", folder / "Annotations", folder)
                     if d.exists() and any(d.rglob("*.xml"))), folder)
        return voc_to_dets(xmls, image_dir=folder)
    if fmt == "coco":
        jp = Path(json_path) if json_path else next(
            (p for p in sorted(folder.rglob("*.json")) if "instance" in p.name.lower()
             or "annotation" in p.name.lower()), None)
        if jp is None:
            raise ValueError(f"找不到 COCO json(請用 --json 指定):{folder}")
        return coco_to_dets(jp, image_dir=folder)
    raise ValueError(f"未知標籤格式:{fmt}(支援 yolo|voc|coco)")
