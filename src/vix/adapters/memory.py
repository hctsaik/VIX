"""InMemoryAdapter — FiftyOne-free backend for tests and dry-runs.

Embeddings are expected to be pre-seeded on the detections (tests inject
deterministic vectors), so the whole VIX pipeline can run end-to-end without
FiftyOne, MongoDB or a GPU.

Optional file persistence: pass ``state_path`` and the per-sample state
(detections incl. embeddings, attached fields, tags) plus staged decisions are
saved to disk and reloaded on the next process — so the documented
``--adapter memory`` dry-run works across *separate* CLI invocations
(``vix embed`` then ``vix route`` then ``vix explain`` …), not only inside a
single ``vix run``. With ``state_path=None`` it stays purely in-memory, so the
test suite is unaffected.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable

from ..core.manifest import ManifestEntry
from ..types import Decision, Detection
from .base import DatasetAdapter, SampleRow


class InMemoryAdapter(DatasetAdapter):
    def __init__(self, embedder=None, state_path: str | Path | None = None) -> None:
        self._s: dict[str, dict] = {}
        self._decisions: list[Decision] = []
        self._embedder = embedder  # optional callable(image|path) -> np.ndarray
        self._state_path = Path(state_path) if state_path else None
        if self._state_path and self._state_path.exists():
            self._load()

    # --- persistence (no-op unless state_path was given) ---
    def _load(self) -> None:
        try:
            with self._state_path.open("rb") as fh:
                self._s, self._decisions = pickle.load(fh)
        except Exception:  # noqa: BLE001 - a corrupt cache should not crash the CLI
            self._s, self._decisions = {}, []

    def _persist(self) -> None:
        if not self._state_path:
            return
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._state_path.with_suffix(self._state_path.suffix + ".tmp")
        with tmp.open("wb") as fh:
            pickle.dump((self._s, self._decisions), fh)
        tmp.replace(self._state_path)  # atomic

    # --- test helper ---
    def seed(self, vix_hash: str, src_path: str, detections: list[Detection], tags=None) -> None:
        self._s[vix_hash] = {
            "src_path": src_path,
            "detections": list(detections),
            "fields": {},
            "tags": list(tags or []),
        }
        self._persist()

    def stage_decision(self, decision: Decision) -> None:
        self._decisions.append(decision)
        self._persist()

    # --- DatasetAdapter ---
    def sync(self, entries: Iterable[ManifestEntry]) -> None:
        for e in entries:
            self._s.setdefault(
                e.vix_hash,
                {"src_path": e.src_path, "detections": [], "fields": {}, "tags": list(e.tags)},
            )
        self._persist()

    def set_detections(self, vix_hash: str, detections: list[Detection]) -> None:
        self._s[vix_hash]["detections"] = list(detections)
        self._persist()

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
        self._persist()

    def samples(self) -> Iterable[SampleRow]:
        for h, d in self._s.items():
            yield h, d["src_path"], d["detections"], d["tags"]

    def attach_fields(self, vix_hash: str, fields: dict) -> None:
        self._s[vix_hash]["fields"].update(fields)
        self._persist()

    def apply_tags(self, vix_hash: str, tags: list[str]) -> None:
        cur = self._s[vix_hash]["tags"]
        for t in tags:
            if t not in cur:
                cur.append(t)
        self._persist()

    def get_by_tag(self, tag: str) -> Iterable[tuple[str, str, list[Detection]]]:
        for h, d in self._s.items():
            if tag in d["tags"]:
                yield h, d["src_path"], d["detections"]

    def pull_review_decisions(self) -> list[Decision]:
        return list(self._decisions)

    def fields(self, vix_hash: str) -> dict:
        """Inspect attached fields (test convenience)."""
        return self._s[vix_hash]["fields"]
