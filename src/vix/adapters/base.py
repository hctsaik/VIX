"""DatasetAdapter — the boundary between VIX core logic and FiftyOne."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from ..core.manifest import ManifestEntry
from ..types import Decision, Detection

# A sample row as seen by the pure pipeline: (vix_hash, src_path, detections, tags)
SampleRow = tuple[str, str, list[Detection], list[str]]


class DatasetAdapter(ABC):
    """Everything the pipeline needs from a dataset backend.

    No FiftyOne types cross this boundary — only plain VIX types / numpy.
    """

    @abstractmethod
    def sync(self, entries: Iterable[ManifestEntry]) -> None:
        """Create/refresh dataset samples from manifest entries (images)."""

    @abstractmethod
    def set_detections(self, vix_hash: str, detections: list[Detection]) -> None:
        """Store YOLO detections for a sample."""

    @abstractmethod
    def compute_embeddings(self, model_key: str) -> None:
        """Fill ``detection.embedding`` for every sample (real: DINOv2 crops)."""

    @abstractmethod
    def samples(self) -> Iterable[SampleRow]:
        """Yield (vix_hash, src_path, detections, tags) for all samples."""

    @abstractmethod
    def attach_fields(self, vix_hash: str, fields: dict) -> None:
        """Attach read-only scalar fields (scores, routing decision, reasons)."""

    @abstractmethod
    def apply_tags(self, vix_hash: str, tags: list[str]) -> None:
        """Add tags (idempotent)."""

    def remove_tags(self, vix_hash: str, tags: list[str]) -> None:  # optional (un-reject / restore)
        raise NotImplementedError("This adapter cannot remove tags")

    @abstractmethod
    def get_by_tag(self, tag: str) -> Iterable[tuple[str, str, list[Detection]]]:
        """Yield (vix_hash, src_path, detections) for samples carrying ``tag``."""

    @abstractmethod
    def pull_review_decisions(self) -> list[Decision]:
        """Read human review decisions back from the App."""

    def build_knn_index(self, embeddings_field: str = "dino_embedding") -> str:  # optional
        """Build the similarity index (real adapter uses compute_similarity)."""
        return ""

    def launch_app(self, saved_views: dict | None = None) -> None:  # optional
        raise NotImplementedError("This adapter has no UI")
