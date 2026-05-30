"""YOLO inference -> VIX detections (requires the ``yolo`` extra: ultralytics).

Kept thin and separate from embedding. Runs on the sample filepaths the adapter
already knows about and writes detections back through the adapter.
"""

from __future__ import annotations

from .adapters.base import DatasetAdapter
from .config import Config
from .logging_setup import get_logger
from .types import BBox, Detection

log = get_logger("vix.detect")


def run_yolo(adapter: DatasetAdapter, cfg: Config, weights: str, conf: float = 0.001) -> int:
    """Run a YOLO model over every sample and store detections.

    Low ``conf`` floor on purpose: VIX's own per-class threshold does the
    routing decision, so we want YOLO to surface even low-confidence boxes.
    """
    from ultralytics import YOLO

    model = YOLO(weights)
    n = 0
    for vix_hash, src_path, _dets, _tags in list(adapter.samples()):
        result = model(src_path, conf=conf, verbose=False)[0]
        names = result.names
        out: list[Detection] = []
        for box in result.boxes:
            cx, cy, w, h = box.xywhn[0].tolist()
            out.append(
                Detection(
                    label=names[int(box.cls)],
                    confidence=float(box.conf),
                    bbox=BBox(cx=cx, cy=cy, w=w, h=h),
                )
            )
        adapter.set_detections(vix_hash, out)
        n += 1
    log.info("detect.run_yolo: inferred %d images (weights=%s)", n, weights)
    return n
