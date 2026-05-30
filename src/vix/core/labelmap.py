"""Label-map reconciliation (T2) and class rename/merge migration (T4).

Pure functions over class maps and (id, label) records — testable without
FiftyOne. Every relabel produces a change log that is also the rollback data, so
even non-invertible merges can be undone.
"""

from __future__ import annotations

from dataclasses import dataclass


# --- T2: merge two datasets' class maps ---------------------------------

def merge_class_maps(
    map_a: dict[int, str],
    map_b: dict[int, str],
    name_overrides: dict[str, str] | None = None,
) -> dict:
    """Align two class maps into one unified namespace.

    ``name_overrides`` lets a human reconcile differing names
    (e.g. ``{"vehicle_car": "car", "heavy_vehicle": "truck"}``). Returns the
    unified index, per-dataset old-id -> new-id remaps, a conflict report
    (names present in only one side), and an orphan check.
    """
    overrides = name_overrides or {}

    def canon(n: str) -> str:
        return overrides.get(n, n)

    names = sorted({canon(n) for n in list(map_a.values()) + list(map_b.values())})
    unified = {n: i for i, n in enumerate(names)}
    remap_a = {oid: unified[canon(n)] for oid, n in map_a.items()}
    remap_b = {oid: unified[canon(n)] for oid, n in map_b.items()}
    orphans = [oid for oid, nid in {**remap_a, **remap_b}.items() if nid is None]

    a_names = {canon(n) for n in map_a.values()}
    b_names = {canon(n) for n in map_b.values()}
    return {
        "unified_names": names,
        "unified_index": unified,
        "remap_a": remap_a,
        "remap_b": remap_b,
        "common": sorted(a_names & b_names),
        "only_in_a": sorted(a_names - b_names),
        "only_in_b": sorted(b_names - a_names),
        "needs_decision": sorted((a_names - b_names) | (b_names - a_names)),
        "orphans": orphans,
    }


# --- T4: class rename / merge migration with rollback -------------------

@dataclass
class LabelChange:
    id: str
    old: str
    new: str


def relabel(
    records: list[tuple[str, str]], mapping: dict[str, str]
) -> tuple[list[tuple[str, str]], list[LabelChange]]:
    """Apply a name->name mapping (rename or merge) across (id, label) records.

    Returns the new records plus a change log; the change log records each
    original value, so :func:`rollback` can undo merges too.
    """
    changes: list[LabelChange] = []
    out: list[tuple[str, str]] = []
    for rid, label in records:
        new = mapping.get(label, label)
        if new != label:
            changes.append(LabelChange(rid, label, new))
        out.append((rid, new))
    return out, changes


def rollback(records: list[tuple[str, str]], changes: list[LabelChange]) -> list[tuple[str, str]]:
    """Restore records to their pre-relabel state using the change log."""
    undo = {c.id: c.old for c in changes}
    return [(rid, undo.get(rid, label)) for rid, label in records]


def preview_merged_distribution(
    counts_a: dict[str, int], counts_b: dict[str, int], name_overrides: dict[str, str] | None = None
) -> dict[str, int]:
    """Preview the per-class sample counts AFTER merging two datasets (W9),
    so surprises are seen before committing the merge."""
    overrides = name_overrides or {}
    merged: dict[str, int] = {}
    for counts in (counts_a, counts_b):
        for name, c in counts.items():
            canon = overrides.get(name, name)
            merged[canon] = merged.get(canon, 0) + c
    return merged


def migration_diff(changes: list[LabelChange]) -> dict:
    """Summarise a migration: per (old->new) count + total changed."""
    summary: dict[str, int] = {}
    for c in changes:
        summary[f"{c.old}->{c.new}"] = summary.get(f"{c.old}->{c.new}", 0) + 1
    return {"total_changed": len(changes), "by_transition": summary}
