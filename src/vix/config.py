"""Runtime configuration for VIX v0.1.

Locked parameters (see docs/spec/v0.1-technical-spec.md §13):
    * DINOv2 ViT-B/14  -> embedding dim 768
    * vector backend   -> LanceDB
    * export format    -> YOLO txt + data.yaml

All values can be overridden via ``VIX_*`` environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


@dataclass
class Config:
    # --- workspace layout (VIX core artifacts are the source of truth, DVC-tracked) ---
    workspace: Path = field(default_factory=lambda: Path(_env("VIX_WORKSPACE", "./vix_workspace")))

    # --- model / algorithm (locked) ---
    dinov2_model_key: str = field(default_factory=lambda: _env("VIX_DINOV2_KEY", "dinov2-vitb14-torch"))
    embedding_dim: int = field(default_factory=lambda: _env_int("VIX_EMBED_DIM", 768))
    similarity_backend: str = field(default_factory=lambda: _env("VIX_SIM_BACKEND", "lancedb"))
    # which embedder actually produced the vectors — recorded in audit/report/export
    # so an offline 'pixel_fallback' run is never mistaken for a production DINOv2 run.
    embedding_backend: str = field(default_factory=lambda: _env("VIX_EMBED_BACKEND", "dinov2-vitb14-torch"))

    # --- routing / scoring params ---
    knn_k: int = field(default_factory=lambda: _env_int("VIX_KNN_K", 10))
    conf_percentile: float = field(default_factory=lambda: _env_float("VIX_CONF_PCT", 5.0))
    dist_percentile: float = field(default_factory=lambda: _env_float("VIX_DIST_PCT", 95.0))
    # domain-adapted embedding (gt-consistency #2): apply the learned LDA projection to ranking
    # consumers (error-mine). Off unless explicitly enabled AND the projection passed its gate.
    use_embed_projection: bool = field(
        default_factory=lambda: _env("VIX_USE_PROJECTION", "0") not in ("0", "", "false", "False"))

    # --- guard (frozen reference) thresholds ---
    drift_shift_threshold: float = field(default_factory=lambda: _env_float("VIX_DRIFT_SHIFT", 0.15))
    consistency_drop_threshold: float = field(default_factory=lambda: _env_float("VIX_CONSIST_DROP", 0.05))

    def __post_init__(self) -> None:
        self.workspace = Path(self.workspace)

    # --- derived paths ---
    @property
    def manifest_path(self) -> Path:
        return self.workspace / "manifest.jsonl"

    @property
    def thresholds_path(self) -> Path:
        return self.workspace / "thresholds.json"

    @property
    def decision_log_path(self) -> Path:
        return self.workspace / "decision_log.jsonl"

    @property
    def anchor_ref_path(self) -> Path:
        return self.workspace / "anchor_ref.npz"

    @property
    def calibration_path(self) -> Path:
        return self.workspace / "calibration.json"

    @property
    def eval_results_path(self) -> Path:
        return self.workspace / "eval_results.json"

    @property
    def eval_baseline_path(self) -> Path:  # frozen mAP/AP + eval_set_hash for challenge-guard
        return self.workspace / "eval_baseline.json"

    @property
    def embed_projection_path(self) -> Path:  # saved LDA projection (domain-adapted embedding)
        return self.workspace / "embed_projection.npz"

    @property
    def adapt_report_path(self) -> Path:  # per-pair frozen->adapted separability + gate verdict
        return self.workspace / "adapt_report.json"

    @property
    def embed_projection_enabled_path(self) -> Path:  # gate-validated enable marker (written iff gate GO)
        return self.workspace / "embed_projection.enabled"

    @property
    def lancedb_dir(self) -> Path:
        return self.workspace / "lancedb"

    @property
    def log_path(self) -> Path:
        return self.workspace / "vix.log"

    def ensure_dirs(self) -> None:
        self.workspace.mkdir(parents=True, exist_ok=True)
