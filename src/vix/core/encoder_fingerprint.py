"""Encoder fingerprint — bind the embedding encoder into the audit truth.

Every PROXY number VIX produces (kNN distance, separability, drift, the distance thresholds themselves)
is a pure function of the encoder, yet historically the encoder identity was computed and then discarded
to a log line while only a static `embedding_backend` string entered the audit — so a re-pulled torch.hub
cache, a torch upgrade, a CPU<->GPU move, or a swapped checkpoint stamped the identical audit identity
while every threshold silently drifted. That is the same class of hole as the earlier box-level audit gap.

This module computes a small, behaviour-anchored fingerprint so a materially-changed encoder changes the
audit identity (and trips the gate) instead of passing silently. Pure / stdlib+numpy only — the adapter
supplies the raw material (a probe embedding + optional weights/version metadata).

HONEST scope: this is a reproducibility/identity check, NOT a security signature. The fingerprint is keyed
on BEHAVIOUR (a probe embedding, coarsely rounded) so a no-op torch point-release does not cry wolf, while
a real encoder/weights/preprocessing change does. PKI signing / bit-exact weight hashing are deliberately
out of scope.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np

_PROBE_DECIMALS = 3  # round the probe embedding so trivial numeric noise doesn't trip the fingerprint


def probe_image(size: int = 64):
    """A fixed, deterministic synthetic RGB image (no asset file) to behaviourally probe any encoder."""
    from PIL import Image
    rng = np.random.RandomState(12345)
    a = (rng.rand(size, size, 3) * 255).astype("uint8")
    a[: size // 2, : size // 2] = [200, 30, 30]      # stable colour blocks so the probe is structured
    a[size // 2:, size // 2:] = [30, 30, 200]
    return Image.fromarray(a)


def probe_digest(embed_fn) -> str | None:
    """Behavioural digest: embed the fixed probe image, L2-normalise, round, hash. Captures weights +
    preprocessing + device-numerics + torch-version drift in one number, for ANY embedder. None on failure."""
    try:
        v = np.asarray(embed_fn(probe_image()), dtype=float).ravel()
    except Exception:  # noqa: BLE001
        return None
    if v.size == 0:
        return None
    n = np.linalg.norm(v)
    if n > 0:
        v = v / n
    rounded = np.round(v, _PROBE_DECIMALS) + 0.0  # +0.0 normalises -0.0
    return hashlib.sha256(rounded.tobytes()).hexdigest()[:16]


def encoder_fingerprint(material: dict) -> dict:
    """Canonical fingerprint of an encoder from supplied material. Returns {"fp", "components"}.
    `fp` is sha256 over the non-null components (so partial material — e.g. an opaque zoo model with only
    backend+dim+probe — still yields a stable, meaningful identity). Omitting None keeps it backward-compatible."""
    components = {k: v for k, v in (material or {}).items() if v is not None}
    fp = hashlib.sha256(json.dumps(components, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return {"fp": fp, "components": components}
