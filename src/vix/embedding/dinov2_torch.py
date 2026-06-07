"""Built-in DINOv2 embedder via torch.hub — offline-capable, used when the FiftyOne Model Zoo is
unavailable (broken manifest / air-gapped). Lazy torch import so importing VIX stays light.

Weights + repo come from the torch hub cache: set ``VIX_DINOV2_HUB_DIR`` (or cfg.dinov2_hub_dir) to a
pre-populated cache for fully-offline use; otherwise torch downloads them on first use. Same per-crop
contract the FiftyOne zoo model exposes — `model.embed(crop)` and a `with model:` context — so the
adapter can use either interchangeably.
"""

from __future__ import annotations

import os

import numpy as np

from .dinov2 import MODEL_KEY

# VIX model_key -> (torch.hub entrypoint, embedding dim)
_VARIANTS = {
    "dinov2-vits14-torch": ("dinov2_vits14", 384),
    "dinov2-vitb14-torch": ("dinov2_vitb14", 768),
    "dinov2-vitl14-torch": ("dinov2_vitl14", 1024),
}


def variant_for(model_key: str) -> tuple[str, int]:
    """Map a VIX model_key to the (torch.hub entrypoint, dim); defaults to ViT-B/14 (768-d)."""
    return _VARIANTS.get(model_key, ("dinov2_vitb14", 768))


class DinoV2Embedder:
    """Loads DINOv2 once; `embed(crop)` returns its CLS embedding as a float np.ndarray. `crop` may be a
    PIL image or an np array (so it's drop-in for the zoo model's `embed(np.array(crop))` call site)."""

    def __init__(self, model_key: str = MODEL_KEY, hub_dir: str | None = None):
        import torch
        import torchvision.transforms as T

        self._torch = torch
        entry, self.dim = variant_for(model_key)
        hub_dir = hub_dir or os.environ.get("VIX_DINOV2_HUB_DIR")
        if hub_dir:
            torch.hub.set_dir(hub_dir)
        self.model = torch.hub.load("facebookresearch/dinov2", entry, source="github",
                                    verbose=False, trust_repo=True).eval()
        self._tf = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                              T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def fingerprint_material(self) -> dict:
        """Extra encoder-identity material for the audit fingerprint (best-effort; behaviour is captured
        separately by a probe embedding). Weights digest is a cheap strided checksum, not a full byte hash."""
        import torch
        mat: dict = {"vix_tag": getattr(self, "_vix_tag", None), "embedding_dim": self.dim}
        try:
            mat["torch_version"] = torch.__version__
            mat["device"] = str(next(self.model.parameters()).device.type)
            mat["preproc"] = "resize224;imagenet-norm"
            sd = self.model.state_dict()
            meta = sorted((k, tuple(v.shape), str(v.dtype)) for k, v in sd.items())
            strided = float(sum(float(v.detach().float().flatten()[::997].sum()) for v in sd.values()))
            mat["weights_digest"] = __import__("hashlib").sha256(
                (str(meta) + f"|{strided:.4f}").encode("utf-8")).hexdigest()[:16]
        except Exception:  # noqa: BLE001
            pass
        return mat

    def embed(self, crop) -> np.ndarray:
        from PIL import Image
        if isinstance(crop, np.ndarray):  # adapter passes np.array(crop); np arrays also have a .size attr
            a = crop if crop.dtype == np.uint8 else crop.astype("uint8")
            crop = Image.fromarray(a)
        crop = crop.convert("RGB")
        x = self._tf(crop).unsqueeze(0)
        with self._torch.no_grad():
            return self.model(x)[0].cpu().numpy().astype(float)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
