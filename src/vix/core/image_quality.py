"""Image-level pixel quality scan — the offline replacement for FiftyOne Enterprise's "Data Quality"
panel, which VIX otherwise lacks at the pixel level (``box_qa`` is box-geometry only).

Three cheap, dependency-free per-image signals (numpy + PIL only — NO cv2/scipy, confirmed not core deps):
    blur      variance-of-Laplacian (low = blurry)   — on [0,255] luma so the absolute floor matches the
                                                        classic ~100 threshold
    exposure  fraction of pixels clipped at the dark/bright histogram ends (over/under-exposed)
    aspect    width/height outliers relative to THIS dataset

Pure: takes ``np.ndarray`` images, returns metrics + a ranked advisory list. No FiftyOne, no I/O (the
adapter/pipeline decodes files) — unit-testable on synthetic arrays.

HONESTY (mirrors box_qa / weakness_report): advisory/PROXY ONLY — never mutates, never deletes. Flags are
percentile-WITHIN-this-dataset AND-gated by an optional absolute floor, so a uniformly sharp / well-exposed
dataset returns ZERO flags instead of the mathematically-guaranteed "bottom decile" a pure-percentile
design always emits. Variance-of-Laplacian is scene-dependent (a real sky / wall / bokeh background is
smooth, not "out of focus") — so a low value is *suspect, not a verdict*.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_MAX_DIM = 512   # downscale cap before the Laplacian -> O(512^2), independent of input size
_MIN_DIM = 32    # below this, variance-of-Laplacian is unreliable -> blur metric is None (excluded)
_DARK = 5.0      # luma <= ~2% of 255 counts as black-clipped
_BRIGHT = 250.0  # luma >= ~98% of 255 counts as white-clipped

_SEV = {"exposure": 0, "blur": 1, "aspect": 2}  # severity order for the ranked list (cf. box_qa)

# zh-Hant honest caveats (reuse the PROXY/advisory register of weakness_report._PROXY)
_CAVEATS = [
    "_(PROXY:像素層級的「嫌疑/排序」,非品質判決;一律建議性,永不自動刪檔/改標,人工覆核後才處置。)_",
    "模糊以 Laplacian 變異數估計,與場景強相關:合法的天空/純色背景/散景本來就平滑、高頻少→會算成低變異,"
    "但並非拍糊。門檻取「此資料集最低十分位」並加一道絕對下限,避免在整體都銳利的資料集裡亂報。",
    "過/欠曝以「直方圖兩端被截斷的像素比例」估計,門檻為經驗值:刻意的高調/低調攝影、夜景、純白掃描都可能合法觸發。",
    "長寬比僅以「相對此資料集的分位」判離群,沒有絕對對錯:4:1 只有在一堆 16:9 裡才算離群。",
]
CAVEATS = _CAVEATS  # public: the CLI/operator surface these honest caveats to the user


@dataclass
class ImageMetrics:
    id: str
    blur_var: float | None          # variance-of-Laplacian on [0,255] luma; None if too small/undecodable
    dark_frac: float                # fraction clipped at the dark end
    bright_frac: float              # fraction clipped at the bright end
    aspect: float                   # w/h of the ORIGINAL image (nan if undecodable)
    width: int
    height: int
    flags: list[str] = field(default_factory=list)        # subset of {"blur","exposure","aspect"}
    why: dict[str, str] = field(default_factory=dict)      # flag -> zh-Hant reason


@dataclass
class QualityScan:
    metrics: list[ImageMetrics]     # per-image, input order
    ranked: list[dict]              # advisory list, worst-first: {"id","issue","value","why","severity"}
    envelopes: dict                 # the percentile cutoffs actually used (transparency / repro)
    n_scored: int                   # images that yielded a blur var (excludes too-small/undecodable)
    caveats: list[str]              # the honest strings, always attached


def _to_luma(img: np.ndarray) -> np.ndarray | None:
    """RGB/RGBA/grayscale ndarray -> single-channel luma in [0,255] float (alpha dropped)."""
    a = np.asarray(img)
    if a.ndim == 2:
        g = a.astype(np.float32)
    elif a.ndim == 3:
        if a.shape[2] >= 3:
            a3 = a[..., :3].astype(np.float32)                          # drop alpha
            g = 0.299 * a3[..., 0] + 0.587 * a3[..., 1] + 0.114 * a3[..., 2]  # Rec.601 luma
        else:
            g = a[..., 0].astype(np.float32)                           # LA / single channel
    else:
        return None
    # Only a FLOAT array can be a normalised [0,1] image. A uint8 array is by definition already [0,255]
    # and must NEVER be rescaled — a genuinely near-black uint8 image (max<=1) is not "normalised", and
    # multiplying it x255 would invert it to pure white (→ falsely "over-exposed"). Gate on dtype.
    if np.issubdtype(a.dtype, np.floating) and g.max() <= 1.5:
        g = g * 255.0
    return g


def _downscale(g: np.ndarray, max_dim: int) -> np.ndarray:
    h, w = g.shape
    if max(h, w) <= max_dim:
        return g
    from PIL import Image
    scale = max_dim / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    u8 = np.clip(g, 0, 255).astype(np.uint8)  # clip first so HDR/raw float luma >255 doesn't wrap mod 256
    return np.asarray(Image.fromarray(u8).resize((nw, nh), Image.BILINEAR), dtype=np.float32)


def blur_var(img: np.ndarray, max_dim: int = _MAX_DIM) -> float | None:
    """Variance-of-Laplacian on [0,255] luma (pure-numpy 3x3 conv). None if the image is too small."""
    g = _to_luma(img)
    if g is None or min(g.shape) < _MIN_DIM:
        return None
    g = _downscale(g, max_dim)
    p = np.pad(g, 1, mode="reflect")  # reflect-pad so borders don't inject artificial edges
    lap = p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - 4.0 * p[1:-1, 1:-1]
    return float(lap.var())


def exposure_frac(img: np.ndarray) -> tuple[float, float]:
    """(dark_fraction, bright_fraction) — share of pixels clipped at each histogram end."""
    g = _to_luma(img)
    if g is None or g.size == 0:
        return 0.0, 0.0
    n = g.size
    return float(np.count_nonzero(g <= _DARK)) / n, float(np.count_nonzero(g >= _BRIGHT)) / n


def aspect_ratio(img: np.ndarray) -> float:
    a = np.asarray(img)
    if a.ndim < 2 or a.shape[0] == 0:
        return float("nan")
    return float(a.shape[1]) / float(a.shape[0])


def _flag_low(value: float, cut: float | None, floor: float | None) -> bool:
    """Low-is-bad flag: AND-gate relative(<=cut) with absolute(<floor) when BOTH present; else use
    whichever is present; if neither, never flag (so a healthy dataset can return zero)."""
    rel = cut is not None and value <= cut
    ab = floor is not None and value < floor
    if cut is not None and floor is not None:
        return rel and ab
    return rel or ab


def _flag_high(value: float, cut: float | None, floor: float | None) -> bool:
    rel = cut is not None and value >= cut
    ab = floor is not None and value > floor
    if cut is not None and floor is not None:
        return rel and ab
    return rel or ab


def scan_images(
    images: list[tuple[str, np.ndarray]],
    *,
    blur_pctile: float = 10.0,
    blur_abs_floor: float | None = 100.0,
    expo_pctile: float = 90.0,
    expo_abs_floor: float | None = 0.10,
    aspect_lo_pct: float = 2.0,
    aspect_hi_pct: float = 98.0,
    min_support: int = 8,
    max_dim: int = _MAX_DIM,
) -> QualityScan:
    """Per-image quality PROXY scan (blur / exposure / aspect). Pure; advisory; never mutates/deletes.

    Relative percentile-within-THIS-dataset AND-gated by an absolute floor so a uniformly clean dataset
    isn't falsely flagged. Below ``min_support`` images no percentile envelope is built (only the absolute
    floors fire). Returns metrics + a worst-first ranked advisory list + the cutoffs used + caveats."""
    metrics: list[ImageMetrics] = []
    for id_, img in images:
        a = np.asarray(img)
        h = int(a.shape[0]) if a.ndim >= 2 else 0
        w = int(a.shape[1]) if a.ndim >= 2 else 0
        d, b = exposure_frac(a)
        metrics.append(ImageMetrics(id_, blur_var(a, max_dim), d, b, aspect_ratio(a), w, h))

    vars_ = [m.blur_var for m in metrics if m.blur_var is not None]
    clips = [max(m.dark_frac, m.bright_frac) for m in metrics]
    asps = [m.aspect for m in metrics if m.aspect == m.aspect and m.aspect > 0]
    enough = len(metrics) >= min_support

    # zero-spread guard: if a metric is identical across the dataset there are no relative outliers, and an
    # inclusive percentile cut would otherwise flag the WHOLE dataset. ptp==0 -> no relative cut.
    blur_cut = float(np.percentile(vars_, blur_pctile)) if (enough and vars_ and np.ptp(vars_) > 0) else None
    expo_cut = float(np.percentile(clips, expo_pctile)) if (enough and clips and np.ptp(clips) > 0) else None
    asp_ok = enough and len(asps) >= min_support and np.ptp(asps) > 0
    asp_lo = float(np.percentile(asps, aspect_lo_pct)) if asp_ok else None
    asp_hi = float(np.percentile(asps, aspect_hi_pct)) if asp_ok else None
    med_asp = float(np.median(asps)) if asps else 1.0

    ranked: list[dict] = []
    for m in metrics:
        if m.blur_var is not None and _flag_low(m.blur_var, blur_cut, blur_abs_floor):
            m.flags.append("blur")
            m.why["blur"] = f"清晰度(Laplacian 變異數){m.blur_var:.0f} 偏低,疑似模糊(場景相關,非定罪)"
            sev = 1.0 - min(1.0, m.blur_var / (blur_abs_floor or blur_cut or (m.blur_var + 1.0)))
            ranked.append({"id": m.id, "issue": "blur", "value": round(m.blur_var, 1),
                           "why": m.why["blur"], "severity": round(float(sev), 3)})
        clip = max(m.dark_frac, m.bright_frac)
        if _flag_high(clip, expo_cut, expo_abs_floor):
            m.flags.append("exposure")
            side = "過曝" if m.bright_frac >= m.dark_frac else "過暗/欠曝"
            m.why["exposure"] = f"{side}:{clip:.0%} 像素被截斷在直方圖端點(經驗門檻,非標準)"
            ranked.append({"id": m.id, "issue": "exposure", "value": round(clip, 3),
                           "why": m.why["exposure"], "severity": round(float(clip), 3)})
        if asp_lo is not None and m.aspect == m.aspect and (m.aspect < asp_lo or m.aspect > asp_hi):
            m.flags.append("aspect")
            m.why["aspect"] = f"長寬比 {m.aspect:.2f} 相對此資料集離群(中位 {med_asp:.2f};可能裁切錯/誤旋轉)"
            dev = abs(np.log((m.aspect or 1e-6) / (med_asp or 1e-6)))
            ranked.append({"id": m.id, "issue": "aspect", "value": round(m.aspect, 2),
                           "why": m.why["aspect"], "severity": round(float(min(1.0, dev)), 3)})

    ranked.sort(key=lambda r: (_SEV[r["issue"]], -r["severity"]))
    return QualityScan(
        metrics=metrics, ranked=ranked, n_scored=len(vars_), caveats=list(_CAVEATS),
        envelopes={"blur_cut": blur_cut, "blur_abs_floor": blur_abs_floor, "expo_cut": expo_cut,
                   "expo_abs_floor": expo_abs_floor, "aspect_lo": asp_lo, "aspect_hi": asp_hi,
                   "min_support": min_support, "n_images": len(metrics)},
    )
