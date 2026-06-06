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


def crop_detection(
    image: "Image.Image | str", bbox: BBox, pad: float = 0.0, min_size: int = 0
) -> "Image.Image":
    """Crop a normalised (cx, cy, w, h) box out of an image.

    Accepts a PIL image or a path. Always returns at least a 1x1 region.

    ``pad`` / ``min_size`` default to 0 (unchanged behaviour for all existing
    callers / banks). The bank-audit path may set ``pad`` (context dilation: pad
    each side by ``pad`` * box w/h, clamped) and ``min_size`` (BICUBIC-upscale so
    the short side >= min_size px) to give DINOv2 signal on tiny/transparent
    defects. IMPORTANT: if used, banks AND proposals must be embedded with the
    SAME transform, or the cosine vote compares different preprocessing.
    """
    if not hasattr(image, "size"):
        image = Image.open(image).convert("RGB")
    W, H = image.size
    bw, bh = bbox.w * W, bbox.h * H
    cx, cy = bbox.cx * W, bbox.cy * H
    px, py = (pad * bw, pad * bh) if pad else (0.0, 0.0)
    x1 = max(0, int(round(cx - bw / 2 - px)))
    y1 = max(0, int(round(cy - bh / 2 - py)))
    x2 = min(W, int(round(cx + bw / 2 + px)))
    y2 = min(H, int(round(cy + bh / 2 + py)))
    if x2 <= x1:
        x2 = min(W, x1 + 1)
    if y2 <= y1:
        y2 = min(H, y1 + 1)
    crop = image.crop((x1, y1, x2, y2))
    if min_size:
        cw, ch = crop.size
        short = min(cw, ch)
        if 0 < short < min_size:
            scale = min_size / short
            crop = crop.resize((max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))), Image.BICUBIC)
    return crop
