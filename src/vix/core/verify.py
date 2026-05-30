"""Export integrity (U8): per-file SHA-256 manifest + verification.

Lets a recipient confirm a transferred dataset is bit-identical to what was
exported — catching corruption, substitution, or missing files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .manifest import compute_hash


def write_export_manifest(records: Iterable[tuple[str, list]], dst: str | Path) -> Path:
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    lines = []
    for src, _dets in records:
        p = Path(src)
        if p.exists():
            lines.append({"file": p.name, "sha256": compute_hash(p)})
    out = dst / "export_manifest.jsonl"
    out.write_text(
        "\n".join(json.dumps(line) for line in lines) + ("\n" if lines else ""), encoding="utf-8"
    )
    return out


def write_dir_manifest(dst: str | Path) -> Path:
    """Hash EVERY exported file (images + labels/*.txt + data.yaml), not just images."""
    dst = Path(dst)
    lines = []
    for p in sorted(dst.rglob("*")):
        if p.is_file() and p.name != "export_manifest.jsonl":
            lines.append({"file": p.name, "sha256": compute_hash(p)})
    out = dst / "export_manifest.jsonl"
    out.write_text(
        "\n".join(json.dumps(line) for line in lines) + ("\n" if lines else ""), encoding="utf-8"
    )
    return out


def verify_export(manifest_path: str | Path, data_dir: str | Path) -> dict:
    recorded = [
        json.loads(line)
        for line in Path(manifest_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    files = {p.name: p for p in Path(data_dir).rglob("*") if p.is_file()}
    mismatched, missing = [], []
    for rec in recorded:
        p = files.get(rec["file"])
        if p is None:
            missing.append(rec["file"])
        elif compute_hash(p) != rec["sha256"]:
            mismatched.append(rec["file"])
    return {
        "ok": not mismatched and not missing,
        "n_checked": len(recorded),
        "mismatched": mismatched,
        "missing": missing,
    }
