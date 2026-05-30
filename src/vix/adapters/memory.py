"""InMemoryAdapter — FiftyOne-free backend for tests and dry-runs.

Embeddings are expected to be pre-seeded on the detections (tests inject
deterministic vectors), so the whole VIX pipeline can run end-to-end without
FiftyOne, MongoDB or a GPU.
"""

from __future__ import annotations

from typing import Iterable

from ..core.manifest import ManifestEntry
from ..types import Decision, Detection
from .base import DatasetAdapter, SampleRow


class InMemoryAdapter(DatasetAdapter):
    def __init__(self, embedder=None) -> None:
        self._s: dict[str, dict] = {}
        self._decisions: list[Decision] = []
        self._embedder = embedder  # optional callable(image|path) -> np.ndarray

    # --- test helper ---
    def seed(self, vix_hash: str, src_path: str, detections: list[Detection], tags=None) -> None:
        self._s[vix_hash] = {
            "src_path": src_path,
            "detections": list(detections),
            "fields": {},
            "tags": list(tags or []),
        }

    def stage_decision(self, decision: Decision) -> None:
        self._decisions.append(decision)

    # --- DatasetAdapter ---
    def sync(self, entries: Iterable[ManifestEntry]) -> None:
        for e in entries:
            self._s.setdefault(
                e.vix_hash,
                {"src_path": e.src_path, "detections": [], "fields": {}, "tags": list(e.tags)},
            )

    def set_detections(self, vix_hash: str, detections: list[Detection]) -> None:
        self._s[vix_hash]["detections"] = list(detections)

    def compute_embeddings(self, model_key: str) -> None:
        """If an embedder was provided, embed each detection crop from the real
        file; otherwise embeddings are assumed pre-seeded on the detections."""
        if self._embedder is None:
            return
        from PIL import Image

        from ..embedding.dinov2 import crop_detection

        for _h, d in self._s.items():
            if not d["detections"]:
                continue
            img = Image.open(d["src_path"]).convert("RGB")
            for det in d["detections"]:
                det.embedding = self._embedder(crop_detection(img, det.bbox))

    def samples(self) -> Iterable[SampleRow]:
        for h, d in self._s.items():
            yield h, d["src_path"], d["detections"], d["tags"]

    def attach_fields(self, vix_hash: str, fields: dict) -> None:
        self._s[vix_hash]["fields"].update(fields)

    def apply_tags(self, vix_hash: str, tags: list[str]) -> None:
        cur = self._s[vix_hash]["tags"]
        for t in tags:
            if t not in cur:
                cur.append(t)

    def get_by_tag(self, tag: str) -> Iterable[tuple[str, str, list[Detection]]]:
        for h, d in self._s.items():
            if tag in d["tags"]:
                yield h, d["src_path"], d["detections"]

    def pull_review_decisions(self) -> list[Decision]:
        return list(self._decisions)

    def fields(self, vix_hash: str) -> dict:
        """Inspect attached fields (test convenience)."""
        return self._s[vix_hash]["fields"]
