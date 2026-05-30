"""Snapshot / restore — reproduce a historical dataset version (S9).

The manifest + decision log already hold everything needed to reconstruct a past
golden set; this wraps them into an immutable, content-hashed snapshot and a
restore that returns the exact composition, the params used, and why items were
excluded. Pure / file-based / testable.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .decision_log import DecisionLog
from .manifest import Manifest


def _content_hash(golden_hashes: list[str], thr_meta: dict) -> str:
    payload = json.dumps({"golden": sorted(golden_hashes), "thr": thr_meta}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _latest_reasons(log_path: Path) -> dict[str, list[str]]:
    reasons: dict[str, list[str]] = {}
    dlog = DecisionLog(log_path)
    for rec in dlog.read_all():
        h = rec.get("vix_hash")
        if h:
            reasons[h] = rec.get("extra", {}).get("reasons", []) or [rec.get("decision", "")]
    return reasons


def create_snapshot(
    manifest_path: str | Path,
    out_path: str | Path,
    version: str,
    thresholds_meta: dict | None = None,
    decision_log_path: str | Path | None = None,
    golden_tag: str = "golden",
) -> dict:
    man = Manifest.load(manifest_path)
    reasons = _latest_reasons(Path(decision_log_path)) if decision_log_path else {}

    composition, excluded, golden_hashes = [], [], []
    for e in man.entries():
        row = {
            "vix_hash": e.vix_hash,
            "src_path": e.src_path,
            "batch_id": e.batch_id,
            "label_version": e.label_version,
            "tags": e.tags,
        }
        if golden_tag in e.tags:
            composition.append(row)
            golden_hashes.append(e.vix_hash)
        else:
            excluded.append({**row, "reason": reasons.get(e.vix_hash, [])})

    thr_meta = thresholds_meta or {}
    snap = {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_golden": len(golden_hashes),
        "n_excluded": len(excluded),
        "content_hash": _content_hash(golden_hashes, thr_meta),
        "thresholds_meta": thr_meta,
        "composition": composition,
        "excluded": excluded,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(snap, indent=2, ensure_ascii=False), encoding="utf-8")
    return snap


def restore(path: str | Path) -> dict:
    """Return the historical composition + params + exclusion reasons."""
    snap = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        "version": snap["version"],
        "created_at": snap["created_at"],
        "content_hash": snap["content_hash"],
        "params": snap.get("thresholds_meta", {}),
        "composition": snap["composition"],
        "excluded": snap.get("excluded", []),
    }
