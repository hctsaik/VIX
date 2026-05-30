"""FiftyOne-free deterministic pixel embedding.

A lightweight, dependency-only-PIL+numpy embedder used for (a) air-gapped /
no-GPU smoke runs and (b) end-to-end tests that exercise the *whole* pipeline on
real image files without FiftyOne or DINOv2. Not a replacement for DINOv2 in
production — it's the runnable fallback that proves the plumbing works.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def pixel_embedding(image: "Image.Image | str", size: int = 16) -> np.ndarray:
    """Grayscale-resize to size×size and flatten to a deterministic vector."""
    if not hasattr(image, "size"):
        image = Image.open(image)
    image = image.convert("L").resize((size, size))
    return np.asarray(image, dtype=float).ravel() / 255.0
