"""DINOv2 ViT-B/14 (locked) — crop preprocessing + model key.

The actual embedding extraction runs inside FiftyOneAdapter via the Model Zoo;
this module holds the pieces that are pure and unit-testable: the model key and
the per-detection crop. ViT-B/14 -> 768-d embeddings.
"""

from __future__ import annotations

from PIL import Image

from ..types import BBox

MODEL_KEY = "dinov2-vitb14-torch"
EMBEDDING_DIM = 768


def crop_detection(image: "Image.Image | str", bbox: BBox) -> "Image.Image":
    """Crop a normalised (cx, cy, w, h) box out of an image.

    Accepts a PIL image or a path. Always returns at least a 1x1 region.
    """
    if not hasattr(image, "size"):
        image = Image.open(image).convert("RGB")
    W, H = image.size
    cx, cy, w, h = bbox.cx * W, bbox.cy * H, bbox.w * W, bbox.h * H
    x1 = max(0, int(round(cx - w / 2)))
    y1 = max(0, int(round(cy - h / 2)))
    x2 = min(W, int(round(cx + w / 2)))
    y2 = min(H, int(round(cy + h / 2)))
    if x2 <= x1:
        x2 = min(W, x1 + 1)
    if y2 <= y1:
        y2 = min(H, y1 + 1)
    return image.crop((x1, y1, x2, y2))
