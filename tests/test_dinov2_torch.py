"""Built-in torch.hub DINOv2 embedder (used when the FiftyOne zoo is unavailable / offline). The pure
key->variant mapping always runs; the actual model load is Tier-2 (needs torch + a hub cache or network)
and skips cleanly otherwise."""

import numpy as np
import pytest

from vix.embedding.dinov2_torch import variant_for


def test_variant_for():
    assert variant_for("dinov2-vits14-torch") == ("dinov2_vits14", 384)
    assert variant_for("dinov2-vitb14-torch") == ("dinov2_vitb14", 768)
    assert variant_for("dinov2-vitl14-torch") == ("dinov2_vitl14", 1024)
    assert variant_for("anything-else") == ("dinov2_vitb14", 768)  # safe default


def test_embedder_loads_and_embeds():
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from PIL import Image

    from vix.embedding.dinov2_torch import DinoV2Embedder
    try:
        emb = DinoV2Embedder("dinov2-vits14-torch")  # VIX_DINOV2_HUB_DIR if set, else downloads
    except Exception as e:  # noqa: BLE001 - air-gapped with no cache -> skip, not fail
        pytest.skip(f"DINOv2 hub unavailable: {str(e).splitlines()[0][:80]}")
    v = emb.embed(Image.fromarray((np.random.rand(80, 80, 3) * 255).astype("uint8")))
    assert v.shape == (384,) and v.dtype == float and np.linalg.norm(v) > 0
