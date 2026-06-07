"""Calibration↔dataset coverage check (honesty guard).

VIX's routing/review-queue lean on a calibration (thresholds.json) computed from one dataset's
golden distribution. If that calibration belongs to a DIFFERENT dataset — its classes don't cover the
data, a different embedding backend, or a drifted encoder — the distance/confidence thresholds are
meaningless, yet the surfaces would happily emit a confident-looking-but-junk routing/queue. This pure
helper detects that mismatch so each surface can fail LOUD ("I'm not calibrated for this") instead of
faking confidence — VIX's core honest-boundaries rule. No I/O; the pipeline supplies the inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CoverageVerdict:
    ok: bool                                       # True iff the calibration is usable for THIS dataset
    no_thresholds: bool = False                    # thresholds.json absent entirely
    uncovered_classes: list[str] = field(default_factory=list)  # det classes with no calibrated threshold
    covered_classes: list[str] = field(default_factory=list)
    backend_mismatch: bool = False                 # calibrated under a different embedding backend
    fp_mismatch: bool = False                      # encoder fingerprint differs from calibration time
    reason: str = ""                               # human zh-Hant one-liner (empty when ok)


def _short(names: list[str], n: int = 5) -> str:
    names = sorted(names)
    return "、".join(names[:n]) + ("…" if len(names) > n else "")


def assess_coverage(thresholds, meta, detection_classes, current_backend,
                    current_encoder_fp=None) -> CoverageVerdict:
    """Decide whether the loaded calibration matches the current dataset.

    thresholds: mapping ``{class: obj-with-.n_support}`` (a ThresholdPolicy.thresholds) or None.
    meta: ThresholdPolicy.meta (reads ``embedding_backend``/``encoder_fp``) or None.
    detection_classes: the classes actually present in the dataset's detections.
    current_backend / current_encoder_fp: the live embedding backend / encoder fingerprint.

    A class counts as covered only if it has a calibrated threshold with ``n_support > 0`` (reusing the
    "n_support == 0 == uncalibrated" convention from threshold.route). fp is fail-OPEN when either side
    is absent (matches the gate's legacy behaviour) so pre-fingerprint thresholds don't cry wolf. An
    empty class set is OK (nothing to cover)."""
    meta = meta or {}
    det = {c for c in (detection_classes or []) if c}
    if thresholds is None:
        return CoverageVerdict(
            ok=False, no_thresholds=True,
            reason="尚未針對此資料集校準(找不到 thresholds.json):請先 vix calibrate。")
    covered = {c for c, t in thresholds.items() if getattr(t, "n_support", 0) > 0}
    uncovered = sorted(det - covered)
    cal_backend = meta.get("embedding_backend")
    backend_mismatch = bool(cal_backend) and bool(current_backend) and cal_backend != current_backend
    cal_fp = meta.get("encoder_fp")
    fp_mismatch = bool(cal_fp) and bool(current_encoder_fp) and cal_fp != current_encoder_fp

    reason = ""
    if backend_mismatch:
        reason = (f"校準後端({cal_backend})≠ 目前後端({current_backend}):距離門檻不可比,"
                  "請以同一後端重新 vix embed + vix calibrate。")
    elif fp_mismatch:
        reason = ("資料目前的編碼器指紋與 calibrate 時不一致(權重/前處理/行為已改):"
                  "距離門檻不可比,請以同一編碼器重新 vix embed + vix calibrate。")
    elif uncovered:
        reason = (f"此校準未涵蓋目前的偵測類別:{_short(uncovered)}"
                  f"(thresholds.json 的類別為 {_short(sorted(covered)) or '(空)'})。"
                  "這份校準屬於別的資料集,對這些類別無法給出可信的距離/信心判斷;"
                  "請先取得 golden 樣本再 vix calibrate。")
    ok = not (backend_mismatch or fp_mismatch or uncovered)
    return CoverageVerdict(ok=ok, uncovered_classes=uncovered, covered_classes=sorted(covered),
                           backend_mismatch=backend_mismatch, fp_mismatch=fp_mismatch, reason=reason)
