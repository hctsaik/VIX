"""DecisionLog — append-only JSONL audit trail with an optional hash-chain.

Every routing / review / guard / export event is appended, never rewritten.
Each record links to the previous via ``prev_hash`` and carries its own
``entry_hash`` (sha256 of the canonical record), giving near-zero-cost
tamper-evidence without needing FiftyOne Enterprise audit logging.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def _canonical(d: dict) -> str:
    return json.dumps(d, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _hash_record(rec: dict) -> str:
    body = {k: v for k, v in rec.items() if k != "entry_hash"}
    return hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest()


class DecisionLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _last_hash(self) -> str:
        if not self.path.exists():
            return ""
        last = ""
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                last = json.loads(line).get("entry_hash", "")
            except json.JSONDecodeError:
                continue
        return last

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
            "prev_hash": self._last_hash(),
        }
        rec["entry_hash"] = _hash_record(rec)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(_canonical(rec) + "\n")
        return rec

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out

    def verify_chain(self) -> bool:
        """True iff the hash-chain is intact (no insert/edit/delete)."""
        prev = ""
        for rec in self.read_all():
            if rec.get("prev_hash") != prev:
                return False
            if rec.get("entry_hash") != _hash_record(rec):
                return False
            prev = rec["entry_hash"]
        return True
