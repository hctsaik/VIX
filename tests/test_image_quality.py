"""Pure-core tests for the image-quality scan (blur / exposure / aspect) on synthetic numpy images,
plus the read-only pipeline stage + the two-step tagging. FiftyOne-free (InMemoryAdapter)."""

import numpy as np
from PIL import Image

from vix import pipeline
from vix.adapters.memory import InMemoryAdapter
from vix.config import Config
from vix.core.image_quality import aspect_ratio, blur_var, exposure_frac, scan_images
from vix.types import Tag


# --- synthetic image recipes (deterministic, no RNG) ---
def _checker(n=128, sq=8):
    yy, xx = np.indices((n, n))
    g = (((xx // sq + yy // sq) % 2) * 255).astype(np.uint8)   # maximal high-frequency edges
    return np.stack([g, g, g], axis=-1)


def _box_blur(a, k=15):
    f = a.astype(np.float32)
    p = np.pad(f, ((k // 2, k // 2), (k // 2, k // 2), (0, 0)), mode="reflect")
    out = np.zeros_like(f)
    for i in range(k):
        for j in range(k):
            out += p[i:i + f.shape[0], j:j + f.shape[1], :]
    return (out / (k * k)).astype(np.uint8)


def _solid(h, w, val):
    return np.full((h, w, 3), val, np.uint8)


def _checker_mid(h, w, sq=8, lo=80, hi=180):
    """Sharp (high variance-of-Laplacian) but NOT clipped (mid-tones): a 'clean' image for both blur
    and exposure, so only aspect can flag it."""
    yy, xx = np.indices((h, w))
    g = np.where((xx // sq + yy // sq) % 2 == 0, lo, hi).astype(np.uint8)
    return np.stack([g, g, g], axis=-1)


# --- core: blur ---
def test_blur_var_sharp_beats_blurred():
    sharp = _checker()
    blurry = _box_blur(sharp, k=15)
    assert blur_var(sharp) > 10 * blur_var(blurry)            # variance collapses under blur


def test_blur_tiny_image_is_none():
    assert blur_var(_solid(10, 10, 128)) is None              # below _MIN_DIM -> excluded, not "blurriest"


def test_scan_flags_blurry_not_sharp():
    imgs = [("sharp", _checker())] + [(f"b{i}", _box_blur(_checker(), 15)) for i in range(9)]
    scan = scan_images(imgs, blur_abs_floor=None)             # pure percentile to isolate the relative test
    blur_ids = {r["id"] for r in scan.ranked if r["issue"] == "blur"}
    assert "b0" in blur_ids and "sharp" not in blur_ids


# --- core: exposure ---
def test_exposure_frac_extremes():
    d_b, br_b = exposure_frac(_solid(64, 64, 255)); assert br_b == 1.0 and d_b == 0.0
    d_d, br_d = exposure_frac(_solid(64, 64, 0));   assert d_d == 1.0 and br_d == 0.0
    d_m, br_m = exposure_frac(_solid(64, 64, 128)); assert d_m == 0.0 and br_m == 0.0


def test_near_black_uint8_not_inverted_to_overexposed():
    """Regression (review B1): a near-black uint8 image (max<=1) must NOT be rescaled x255 to white and
    mislabelled over-exposed. dark_frac must dominate, bright_frac must be ~0."""
    dark, bright = exposure_frac(np.ones((64, 64, 3), np.uint8))   # all 1s on a [0,255] scale = near-black
    assert dark == 1.0 and bright == 0.0


def test_all_identical_blur_no_flag_without_floor():
    """Regression (review S1): with the absolute floor disabled, an all-identical-sharpness dataset has
    zero spread -> no relative outliers -> ZERO blur flags (not the whole dataset)."""
    imgs = [(f"x{i}", _checker()) for i in range(10)]            # identical sharpness
    scan = scan_images(imgs, blur_abs_floor=None)
    assert not [r for r in scan.ranked if r["issue"] == "blur"]


def test_scan_flags_over_and_under_exposed_not_mid():
    imgs = [("bright", _solid(64, 64, 255)), ("dark", _solid(64, 64, 0))] + \
           [(f"m{i}", _solid(64, 64, 128)) for i in range(8)]
    scan = scan_images(imgs)
    flagged = {r["id"] for r in scan.ranked if r["issue"] == "exposure"}
    assert flagged == {"bright", "dark"}                      # mid-gray (0% clipped) never trips


# --- core: aspect ---
def test_aspect_outlier_strip():
    assert aspect_ratio(_solid(100, 400, 0)) == 4.0
    imgs = [(f"sq{i}", _solid(100, 100, 128)) for i in range(10)] + [("strip", _solid(100, 400, 128))]
    scan = scan_images(imgs)
    flagged = {r["id"] for r in scan.ranked if r["issue"] == "aspect"}
    assert flagged == {"strip"}


# --- honesty guardrails ---
def test_uniformly_sharp_dataset_no_blur_false_positive():
    imgs = [(f"s{i}", _checker(sq=4 + (i % 3))) for i in range(20)]  # ALL sharp
    scan = scan_images(imgs, blur_abs_floor=100.0)                  # floor ON (default)
    assert not [r for r in scan.ranked if r["issue"] == "blur"]     # absolute floor vetoes the bottom decile


def test_scan_never_mutates_input():
    a = _checker()
    before = a.copy()
    scan_images([("a", a)])
    assert np.array_equal(a, before)


def test_caveats_always_attached():
    scan = scan_images([("a", _checker())])
    assert scan.caveats and any("PROXY" in c for c in scan.caveats)


# --- pipeline seam (read-only, scans ALL samples, not golden-only) ---
def _png(tmp_path, name, arr):
    p = tmp_path / name
    Image.fromarray(arr).save(p)
    return str(p)


def test_pipeline_image_quality_scans_all_and_is_readonly(tmp_path):
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    ad = InMemoryAdapter()
    # a blurry image tagged REVIEW (NOT golden) must still be scanned -> proves NOT golden-scoped
    ad.seed("blurry", _png(tmp_path, "blurry.png", _box_blur(_checker(), 19)), [], tags=[Tag.REVIEW])
    for i in range(9):
        ad.seed(f"sharp{i}", _png(tmp_path, f"sharp{i}.png", _checker()), [])
    from vix.core.decision_log import DecisionLog
    before = len(DecisionLog(cfg.decision_log_path).read_all())
    issues = pipeline.image_quality(ad, cfg)
    assert any(it["id"] == "blurry" and it["issue"] == "blur" for it in issues)
    assert len(DecisionLog(cfg.decision_log_path).read_all()) == before   # read-only: no ledger write
    tags = {h: set(t) for h, _s, _d, t in ad.samples()}
    assert all("vixq:blurry" not in ts for ts in tags.values())           # read-only: no tag write


def test_flag_image_quality_two_step(tmp_path):
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    ad = InMemoryAdapter()
    ad.seed("bright", _png(tmp_path, "bright.png", _solid(64, 64, 255)), [])
    for i in range(9):
        ad.seed(f"m{i}", _png(tmp_path, f"m{i}.png", _solid(64, 64, 128)), [])
    res = pipeline.flag_image_quality(ad, cfg, confirm=False)
    assert res["confirmed"] is False
    assert all("vixq:" not in t for _h, _s, _d, ts in ad.samples() for t in ts)  # no tag without confirm
    res2 = pipeline.flag_image_quality(ad, cfg, confirm=True)
    assert res2["confirmed"] is True and res2["tagged"]["exposure"] >= 1
    tags = {h: set(t) for h, _s, _d, t in ad.samples()}
    assert "vixq:exposed" in tags["bright"]
    from vix.core.decision_log import DecisionLog
    evs = [r for r in DecisionLog(cfg.decision_log_path).read_all() if r.get("event") == "image_quality"]
    assert len(evs) == 1


def test_flag_image_quality_tags_aspect(tmp_path):
    """Review M1: exercise the aspect tagging path end-to-end (the unit test only covered core ranking).
    9 square + 1 wide-strip sharp mid-tone images -> only the strip gets vixq:aspect."""
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    ad = InMemoryAdapter()
    for i in range(9):
        ad.seed(f"sq{i}", _png(tmp_path, f"sq{i}.png", _checker_mid(100, 100)), [])
    ad.seed("strip", _png(tmp_path, "strip.png", _checker_mid(100, 400)), [])   # w/h = 4.0 outlier
    res = pipeline.flag_image_quality(ad, cfg, confirm=True)
    assert res["tagged"]["aspect"] >= 1
    tags = {h: set(t) for h, _s, _d, t in ad.samples()}
    assert "vixq:aspect" in tags["strip"]
    assert all("vixq:aspect" not in tags[f"sq{i}"] for i in range(9))           # squares not flagged


def test_flag_image_quality_all_clean_zero_flags(tmp_path):
    """Review M2: the headline honesty property at the SEAM — a uniformly clean dataset (sharp, mid-tone,
    square) yields ZERO flags and ZERO vixq tags (still exactly one ledger event, like prune --confirm)."""
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    ad = InMemoryAdapter()
    for i in range(10):
        ad.seed(f"c{i}", _png(tmp_path, f"c{i}.png", _checker_mid(100, 100, sq=6 + (i % 3))), [])
    res = pipeline.flag_image_quality(ad, cfg, confirm=True)
    assert res["tagged"] == {"blur": 0, "exposure": 0, "aspect": 0}
    assert all(not any(t.startswith("vixq:") for t in ts) for _h, _s, _d, ts in ad.samples())
    from vix.core.decision_log import DecisionLog
    evs = [r for r in DecisionLog(cfg.decision_log_path).read_all() if r.get("event") == "image_quality"]
    assert len(evs) == 1                                                        # confirm logs once even at 0


def test_image_quality_skips_unreadable_image(tmp_path):
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    ad = InMemoryAdapter()
    ad.seed("missing", str(tmp_path / "does_not_exist.png"), [])
    ad.seed("ok", _png(tmp_path, "ok.png", _checker()), [])
    issues = pipeline.image_quality(ad, cfg)                  # must not crash on the unreadable one
    assert all(it["id"] != "missing" for it in issues)
