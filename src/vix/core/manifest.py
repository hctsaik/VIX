"""Manifest — the DVC-trackable source of truth for ingested images.

The live FiftyOne/MongoDB dataset is a *rebuildable derivative*; this JSONL file
(+ the media + thresholds.json + anchor_ref.npz + decision_log.jsonl) is what gets
version-controlled. ``ingest`` reconstructs the FiftyOne dataset from here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


def compute_hash(path: str | Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of file bytes — stable primary key for an image."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class ManifestEntry:
    vix_hash: str
    src_path: str
    batch_id: str
    ingested_at: str
    label_version: str = "v0"
    tags: list[str] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        src_path: str | Path,
        batch_id: str,
        label_version: str = "v0",
        tags: list[str] | None = None,
        vix_hash: str | None = None,
    ) -> "ManifestEntry":
        src_path = Path(src_path)
        return cls(
            vix_hash=vix_hash or compute_hash(src_path),
            src_path=str(src_path),
            batch_id=batch_id,
            ingested_at=datetime.now(timezone.utc).isoformat(),
            label_version=label_version,
            tags=list(tags or []),
        )


class Manifest:
    """Append-only, deduplicated by ``vix_hash``."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._entries: list[ManifestEntry] = []
        self._hashes: set[str] = set()

    @classmethod
    def load(cls, path: str | Path) -> "Manifest":
        m = cls(path)
        if m.path.exists():
            for line in m.path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                entry = ManifestEntry(**json.loads(line))
                m._entries.append(entry)
                m._hashes.add(entry.vix_hash)
        return m

    def has(self, vix_hash: str) -> bool:
        return vix_hash in self._hashes

    def append(self, entry: ManifestEntry) -> bool:
        """Append unless the hash is already present. Returns True if written."""
        if entry.vix_hash in self._hashes:
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
        self._entries.append(entry)
        self._hashes.add(entry.vix_hash)
        return True

    def entries(self) -> Iterator[ManifestEntry]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)
