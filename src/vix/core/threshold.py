"""ThresholdPolicy — per-class percentile routing (no global 0.85).

Calibrated from the golden/training distribution (binding condition A#3):
    conf_thr[c]  = p-th percentile of golden confidences for class c   (flag low_conf below)
    dist_thr[c]  = q-th percentile of intra-class kNN distances for c  (flag far_from_known above)

Serialised to thresholds.json (DVC-tracked) with the reference snapshot + params.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..types import Flag, RouteResult, Routing


@dataclass
class ClassThreshold:
    conf_thr: float
    dist_thr: float
    n_support: int


class ThresholdPolicy:
    def __init__(self, thresholds: dict[str, ClassThreshold], meta: dict | None = None):
        self.thresholds = thresholds
        self.meta = meta or {}

    @property
    def version(self) -> str:
        return str(self.meta.get("calibrated_at", ""))

    @classmethod
    def calibrate(
        cls,
        per_class_conf: dict[str, np.ndarray],
        per_class_intra_dist: dict[str, np.ndarray],
        conf_pct: float = 5.0,
        dist_pct: float = 95.0,
        ref_snapshot: str = "",
    ) -> "ThresholdPolicy":
        thresholds: dict[str, ClassThreshold] = {}
        for c in set(per_class_conf) | set(per_class_intra_dist):
            confs = np.asarray(per_class_conf.get(c, []), dtype=float)
            dists = np.asarray(per_class_intra_dist.get(c, []), dtype=float)
            conf_thr = float(np.percentile(confs, conf_pct)) if confs.size else 0.0
            dist_thr = float(np.percentile(dists, dist_pct)) if dists.size else float("inf")
            thresholds[c] = ClassThreshold(conf_thr, dist_thr, int(max(confs.size, dists.size)))
        meta = {
            "calibrated_at": datetime.now(timezone.utc).isoformat(),
            "conf_pct": conf_pct,
            "dist_pct": dist_pct,
            "ref_snapshot": ref_snapshot,
            "basis": "golden/training embedding distribution",
        }
        return cls(thresholds, meta)

    def route(
        self,
        predicted_class: str,
        conf: float,
        knn_dist: float,
        low_support: bool = False,
    ) -> RouteResult:
        """Two independent axes -> a routing decision + human-readable reasons."""
        reasons: list[str] = []
        ct = self.thresholds.get(predicted_class)
        if ct is None:
            # Unknown class with no calibrated reference == maximally novel.
            reasons.append(Flag.FAR_FROM_KNOWN)
        else:
            if conf < ct.conf_thr:
                reasons.append(Flag.LOW_CONF)
            if knn_dist > ct.dist_thr:
                reasons.append(Flag.FAR_FROM_KNOWN)
        if low_support and Flag.LOW_SUPPORT not in reasons:
            reasons.append(Flag.LOW_SUPPORT)
        decision = Routing.REVIEW if reasons else Routing.PASS
        return RouteResult(decision=decision, reasons=reasons)

    # --- persistence ---
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": self.meta,
            "thresholds": {c: asdict(t) for c, t in self.thresholds.items()},
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ThresholdPolicy":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        thresholds = {c: ClassThreshold(**t) for c, t in data.get("thresholds", {}).items()}
        return cls(thresholds, data.get("meta", {}))
