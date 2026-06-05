"""DecisionLog — append-only JSONL audit trail with an optional hash-chain.

Every routing / review / guard / export event is appended, never rewritten.
Each record links to the previous via ``prev_hash`` and carries its own
``entry_hash`` (sha256 of the canonical record), giving near-zero-cost
tamper-evidence without needing FiftyOne Enterprise audit logging.

Robustness (Round 13): appends are serialized by a sidecar lock file and the
record + newline is flushed + fsync'd, so a crash or a second writer cannot
interleave bytes or break the chain (``prev_hash`` is read *under* the lock).
Reads tolerate a UTF-8 BOM and a torn final line (a half-written last record
from a crash is skipped) so ``verify_chain`` / ``audit`` / ``gate`` degrade
gracefully instead of raising.

LIMITATIONS (documented, not silently assumed):
- The chain detects insert / edit / delete in the middle, but an append-only
  chain cannot by itself detect *tail truncation* (dropping the last N records
  leaves a valid shorter chain). Pair with periodic external snapshots.
- The lock is single-machine advisory (sidecar file). VIX v0.1 is single-writer;
  concurrent writers on the same workspace are serialized best-effort, not
  guaranteed across network filesystems.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path


def _canonical(d: dict) -> str:
    return json.dumps(d, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _hash_record(rec: dict) -> str:
    body = {k: v for k, v in rec.items() if k != "entry_hash"}
    return hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest()


class _LockFile:
    """Best-effort cross-platform advisory lock via an O_EXCL sidecar file."""

    def __init__(self, target: Path, timeout: float = 5.0):
        self.lock_path = target.with_suffix(target.suffix + ".lock")
        self.timeout = timeout
        self.fd = None

    def __enter__(self):
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                self.fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                return self
            except FileExistsError:
                if time.monotonic() >= deadline:
                    self.fd = None  # give up rather than deadlock (best-effort)
                    return self
                time.sleep(0.01)

    def __exit__(self, *exc):
        if self.fd is not None:
            os.close(self.fd)
            try:
                os.unlink(self.lock_path)
            except OSError:
                pass


class DecisionLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _iter_records(self):
        """Yield parsed records, tolerating a BOM and skipping torn/invalid lines."""
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue  # torn final line from a crash, or corruption -> skip

    def _last_hash(self) -> str:
        last = ""
        for rec in self._iter_records():
            last = rec.get("entry_hash", last)
        return last

    def _ends_clean(self) -> bool:
        """True if the file is empty/missing or ends with a newline (safe to append)."""
        try:
            with open(self.path, "rb") as f:
                f.seek(-1, os.SEEK_END)
                return f.read(1) == b"\n"
        except (OSError, ValueError):
            return True

    def append(
        self,
        event: str,
        vix_hash: str = "",
        batch_id: str = "",
        reviewer_id: str = "auto",
        decision: str = "",
        scores: dict | None = None,
        thr_version: str = "",
        extra: dict | None = None,
        ts: datetime | None = None,
    ) -> dict:
        rec: dict = {
            "ts_utc": (ts or datetime.now(timezone.utc)).isoformat(),
            "event": event,
            "vix_hash": vix_hash,
            "batch_id": batch_id,
            "reviewer_id": reviewer_id,
            "decision": decision,
            "scores": scores or {},
            "thr_version": thr_version,
            "extra": extra or {},
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _LockFile(self.path):  # serialize: read prev_hash + write atomically under the lock
            rec["prev_hash"] = self._last_hash()
            rec["entry_hash"] = _hash_record(rec)
            with open(self.path, "a", encoding="utf-8") as f:
                if not self._ends_clean():  # heal a torn line left by a crash before appending
                    f.write("\n")
                f.write(_canonical(rec) + "\n")
                f.flush()
                os.fsync(f.fileno())
        return rec

    def read_all(self) -> list[dict]:
        return list(self._iter_records())

    def verify_chain(self) -> bool:
        """True iff the hash-chain is intact (no insert/edit/delete in the middle)."""
        prev = ""
        for rec in self.read_all():
            if rec.get("prev_hash") != prev:
                return False
            if rec.get("entry_hash") != _hash_record(rec):
                return False
            prev = rec["entry_hash"]
        return True
