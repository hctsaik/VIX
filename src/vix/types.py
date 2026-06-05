"""Shared, dependency-light domain types.

Only depends on stdlib + numpy. No FiftyOne. These flow between ``core`` and the
adapters so that no FiftyOne object ever leaks into the pure logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


class Flag:
    """Human-readable reasons a sample was routed to review (interpretability)."""

    LOW_CONF = "low_conf"            # YOLO confidence below the per-class floor
    FAR_FROM_KNOWN = "far_from_known"  # DINOv2 kNN distance above the per-class ceiling
    LOW_SUPPORT = "low_support"      # too few golden neighbours of this class to trust


class Routing:
    PASS = "pass"
    REVIEW = "review"


class Tag:
    GOLDEN = "golden"
    REVIEW = "review"
    PASS = "pass"
    ANCHOR = "anchor"
    REJECTED = "rejected"
    EVAL = "eval"  # held-out evaluation/regression set: never calibrated/routed/exported on


@dataclass
class BBox:
    """Normalised YOLO-style box: centre x/y and width/height in [0, 1]."""

    cx: float
    cy: float
    w: float
    h: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.cx, self.cy, self.w, self.h)


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: BBox
    embedding: Optional[np.ndarray] = None  # DINOv2 vector of the cropped region
    knn_dist: Optional[float] = None        # filled by OutlierScorer
    low_support: bool = False


@dataclass
class Scores:
    """Image-level routing signals — kept as two independent axes on purpose."""

    conf_max: float
    knn_dist: float


@dataclass
class RouteResult:
    decision: str               # Routing.PASS | Routing.REVIEW
    reasons: list[str] = field(default_factory=list)  # subset of Flag.*


@dataclass
class Decision:
    """A human (or auto) review decision pulled back from the App."""

    vix_hash: str
    decision: str               # e.g. a class label, or "false_alarm"
    reviewer_id: str = "auto"
