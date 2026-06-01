"""Export integrity (U8): per-file SHA-256 manifest + verification.

Lets a recipient confirm a transferred dataset is bit-identical to what was
exported — catching corruption, substitution, missing files, AND unexpected
extra files injected into a "verified" export.

Manifest keys are paths **relative to the export root** (POSIX), so two files
with the same basename in different subdirectories cannot collide and silently
escape verification.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .manifest import compute_hash

_MANIFEST_NAME = "export_manifest.jsonl"


def write_export_manifest(records: Iterable[tuple[str, list]], dst: str | Path) -> Path:
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    lines = []
    for src, _dets in records:
        p = Path(src)
        if p.exists():
            lines.append({"file": p.name, "sha256": compute_hash(p)})
    out = dst / _MANIFEST_NAME
    out.write_text(
        "\n".join(json.dumps(line) for line in lines) + ("\n" if lines else ""), encoding="utf-8"
    )
    return out


def write_dir_manifest(dst: str | Path) -> Path:
    """Hash EVERY exported file (images + labels/*.txt + data.yaml), keyed by
    path relative to the export root, so subdir basename collisions are safe."""
    dst = Path(dst)
    lines = []
    for p in sorted(dst.rglob("*")):
        if p.is_file() and p.name != _MANIFEST_NAME:
            lines.append({"file": p.relative_to(dst).as_posix(), "sha256": compute_hash(p)})
    out = dst / _MANIFEST_NAME
    out.write_text(
        "\n".join(json.dumps(line) for line in lines) + ("\n" if lines else ""), encoding="utf-8"
    )
    return out


def verify_export(manifest_path: str | Path, data_dir: str | Path) -> dict:
    data_dir = Path(data_dir)
    recorded = [
        json.loads(line)
        for line in Path(manifest_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # files keyed by path relative to the export root (collision-safe), manifest excluded
    files = {
        p.relative_to(data_dir).as_posix(): p
        for p in data_dir.rglob("*")
        if p.is_file() and p.name != _MANIFEST_NAME
    }
    by_base: dict[str, list[str]] = {}
    for rel in files:
        by_base.setdefault(rel.rsplit("/", 1)[-1], []).append(rel)

    matched: set[str] = set()
    mismatched, missing = [], []
    for rec in recorded:
        key = rec["file"]
        rel = None
        if key in files:  # relative-path manifest (write_dir_manifest)
            rel = key
        elif "/" not in key and by_base.get(key):  # basename manifest (write_export_manifest)
            cands = [r for r in by_base[key] if r not in matched]
            rel = cands[0] if cands else by_base[key][0]
        if rel is None:
            missing.append(key)
        else:
            matched.add(rel)
            if compute_hash(files[rel]) != rec["sha256"]:
                mismatched.append(key)
    # completeness: any present file that no manifest entry accounted for is injected/unexpected
    unexpected = sorted(r for r in files if r not in matched)
    return {
        "ok": not mismatched and not missing and not unexpected,
        "n_checked": len(recorded),
        "mismatched": mismatched,
        "missing": missing,
        "unexpected": unexpected,
    }
