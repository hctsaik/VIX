"""Pipeline — orchestrates VIX stages over a DatasetAdapter.

Depends only on ``core`` + an adapter, so the whole flow is testable with
InMemoryAdapter (no FiftyOne). Each stage logs what it did and appends to the
append-only DecisionLog.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .adapters.base import DatasetAdapter
from .config import Config
from .core import snapshot as snap_mod
from .core import verify as verify_mod
from .core.analytics import (
    EmbItem,
    active_learning_ranking,
    class_distribution,
    coverage_delta,
    coverage_gaps,
    cross_period_drift,
    cross_split_leakage,
    harmful_ranking,
    near_duplicate_groups,
    suspected_label_errors,
    suspected_new_classes,
)
from .core.decision_log import DecisionLog
from .core.explain import explain, explain_image
from .core.exporter import DatasetExporter
from .core.calibration import apply_temperature, expected_calibration_error, fit_temperature
from .core.confident_learning import confident_joint, find_label_issues, noise_rates
from .core.drift_types import diagnose_drift_type
from .core.gate import cost_gate, pre_train_gate, regression_check
from .core.geometry import geometry_drift
from .core.parity import performance_parity
from .core.scorer import _l2norm
from .core import spc as spc_mod
from .core.labelmap import merge_class_maps, migration_diff, preview_merged_distribution
from .core.labelmap import relabel as _relabel
from .core.manifest import Manifest, ManifestEntry
from .core.quality import class_quality_trend, reviewer_consistency
from .core.reference import FrozenReference, GuardReport
from .core.report import build_report, write_report
from .core.scorer import OutlierScorer, intra_class_knn_distances
from .core.threshold import ClassThreshold, ThresholdPolicy
from .core.triage import review_queue as _review_queue
from .logging_setup import get_logger
from .types import Routing, Tag

log = get_logger("vix.pipeline")

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# --- helpers -------------------------------------------------------------

def _emb_by_class(adapter: DatasetAdapter, want_tags: set[str]) -> dict[str, np.ndarray]:
    acc: dict[str, list] = defaultdict(list)
    for _h, _src, dets, tags in adapter.samples():
        if want_tags and not (want_tags & set(tags)):
            continue
        for d in dets:
            if d.embedding is not None:
                acc[d.label].append(np.asarray(d.embedding, dtype=float))
    return {c: np.vstack(v) for c, v in acc.items() if v}


# --- stages --------------------------------------------------------------

def ingest(
    adapter: DatasetAdapter,
    cfg: Config,
    folder: str | Path,
    batch_id: str,
    label_version: str = "v0",
    tags: list[str] | None = None,
) -> tuple[int, int]:
    cfg.ensure_dirs()
    tags = tags or []
    if Tag.GOLDEN in tags and Tag.EVAL in tags:  # held-out eval must never be trainable
        raise ValueError("一個樣本不可同時是 golden 與 eval(eval 為 held-out 評估集,不可進訓練)")
    # auto-apply a batch:<id> sample tag so compare/drift-type/parity/geometry work without manual tagging
    entry_tags = list(tags) + ([f"batch:{batch_id}"] if batch_id else [])
    manifest = Manifest.load(cfg.manifest_path)
    folder = Path(folder)
    dlog = DecisionLog(cfg.decision_log_path)
    n_new = n_scanned = 0
    for p in sorted(folder.rglob("*")):
        if p.suffix.lower() in _IMAGE_EXTS:
            n_scanned += 1
            entry = ManifestEntry.create(p, batch_id, label_version, entry_tags)
            added = manifest.append(entry)
            n_new += int(added)
            # log every submission (incl. skipped re-submissions) -> per-image history
            dlog.append("ingest", vix_hash=entry.vix_hash, batch_id=batch_id,
                        decision="new" if added else "skipped")
    n_skipped = n_scanned - n_new  # idempotency: re-running a batch skips known hashes
    adapter.sync(manifest.entries())
    log.info("ingest: %d new, %d skipped (batch=%s, total=%d)", n_new, n_skipped, batch_id, len(manifest))
    return n_new, n_skipped


def import_labels(adapter, cfg, folder, fmt="auto", names=None, batch="import",
                  as_="reference", json_path=None, label_dir=None):
    """Import an EXISTING labelled dataset's ground truth (the on-ramp seam).

    Ingests ``folder`` (idempotent, content-hash keyed) then parses yolo/voc/coco labels and
    attaches them as detections, JOINED TO IMAGES BY CONTENT HASH (compute_hash) — never by
    filename stem. A label set referencing an image that wasn't ingested fails loudly.

    HONESTY: imported labels are human-UNVERIFIED. ``as_='reference'`` tags them
    ``Tag.PROVISIONAL`` (a diagnosis-only reference that NEVER feeds calibrate/route/gate/
    snapshot/export — those read GOLDEN); ``as_='eval'`` tags ``Tag.EVAL``. Promotion to
    golden is only via the human-confirm path."""
    from .core.label_import import detect_format, parse_labels
    from .core.manifest import compute_hash

    if fmt in (None, "auto"):  # let the user just give a folder — sniff format + class names
        det = detect_format(folder)
        fmt = det["fmt"]
        names = names or det["names"]
        json_path = json_path or det["json_path"]
        if not fmt:
            raise ValueError("無法自動判斷標籤格式:資料夾裡找不到 YOLO labels/(.txt)、"
                             "VOC annotations/(.xml) 或 COCO json")
    ingest(adapter, cfg, folder, batch)  # images in by content hash
    manifest = Manifest.load(cfg.manifest_path)
    # Robust join: a label set references its image by name; the image may sit in a sibling images/ dir
    # (e.g. VOC <filename> + images/). Map each label to the INGESTED image's content hash by exact path
    # first (precise), else by basename, else by stem — so the layout doesn't matter.
    by_base: dict[str, str] = {}
    by_stem: dict[str, str] = {}
    for e in manifest.entries():
        sp = Path(e.src_path)
        by_base.setdefault(sp.name, e.vix_hash)
        by_stem.setdefault(sp.stem, e.vix_hash)
    parsed = parse_labels(folder, fmt, names=names, label_dir=label_dir, json_path=json_path)
    tag = Tag.EVAL if as_ == "eval" else Tag.PROVISIONAL

    missing: list[str] = []
    n_imgs = n_boxes = 0
    classes: set[str] = set()
    for img_path, dets in parsed.items():
        p = Path(img_path)
        h = None
        if p.exists():
            ch = compute_hash(p)
            if manifest.has(ch):
                h = ch
        if h is None:  # fall back to matching by file name against the ingested images
            h = by_base.get(p.name) or by_stem.get(p.stem)
        if h is None:
            missing.append(img_path)  # label references an image not under the ingested folder
            continue
        adapter.set_detections(h, dets)
        if dets:
            adapter.apply_tags(h, [tag])
            n_imgs += 1
            n_boxes += len(dets)
            classes.update(d.label for d in dets)
    if missing:
        raise ValueError(
            f"{len(missing)} 個標籤檔指向未匯入/不存在的影像(例:{missing[0]});"
            "請確認標籤與影像在同一資料夾、或用 --json/--label-dir 指定正確路徑。"
        )
    DecisionLog(cfg.decision_log_path).append(
        "import_labels", decision=f"{as_}:{fmt}",
        extra={"n_images": n_imgs, "n_boxes": n_boxes, "n_classes": len(classes), "tag": tag})
    log.info("import_labels: %d images, %d boxes, %d classes (fmt=%s, tag=%s)",
             n_imgs, n_boxes, len(classes), fmt, tag)
    return {"n_images": n_imgs, "n_boxes": n_boxes, "classes": sorted(classes),
            "n_scanned": len(parsed)}  # images seen (for a precise 0-box diagnostic)


def infer_synthetic(adapter, cfg):  # offline demo/CI: seed deterministic synthetic detections + embeddings
    """No-YOLO fallback so `--adapter memory` dry-runs / CI produce a non-empty pipeline.
    Label = source parent-folder name (common demo layout); one full-image box per sample.
    Clearly NOT real inference — use a real YOLO weight via `vix infer --weights` for judgments."""
    import hashlib

    from .embedding.simple import pixel_embedding
    from .types import BBox, Detection

    n = 0
    for h, src, _dets, _tags in list(adapter.samples()):
        label = Path(src).parent.name or "obj"
        try:
            emb = np.asarray(pixel_embedding(src, size=8), dtype=float)
        except Exception:  # noqa: BLE001
            emb = np.zeros(192, dtype=float)
        seed = int(hashlib.sha256(h.encode()).hexdigest()[:8], 16) % 1000
        conf = round(0.4 + 0.6 * seed / 1000.0, 3)  # deterministic pseudo-confidence
        adapter.set_detections(h, [Detection(label, conf, BBox(0.5, 0.5, 1.0, 1.0), embedding=emb)])
        n += 1
    cfg.embedding_backend = "pixel_fallback"
    log.info("infer_synthetic: seeded %d images (offline demo, NOT real inference)", n)
    return n


def calibrate(adapter: DatasetAdapter, cfg: Config) -> ThresholdPolicy:
    per_conf: dict[str, list] = defaultdict(list)
    for _h, _src, dets, tags in adapter.samples():
        if Tag.GOLDEN in tags:
            for d in dets:
                per_conf[d.label].append(d.confidence)
    if not per_conf:  # U2: no golden detections to calibrate from -> name the prerequisite
        raise ValueError("尚無 golden 偵測可校準:請先 vix ingest --golden,再 vix infer / vix embed")
    golden = _emb_by_class(adapter, {Tag.GOLDEN})
    per_dist = {c: intra_class_knn_distances(emb, cfg.knn_k) for c, emb in golden.items()}
    policy = ThresholdPolicy.calibrate(
        {c: np.asarray(v, dtype=float) for c, v in per_conf.items()},
        per_dist,
        cfg.conf_percentile,
        cfg.dist_percentile,
        ref_snapshot=str(cfg.manifest_path),
    )
    policy.meta["embedding_backend"] = cfg.embedding_backend  # AI6: so route/gate can catch a backend mismatch
    try:  # bind the ENCODER identity (weights/version/preprocessing/behaviour) into the audit truth — it
        policy.meta["encoder_fp"] = adapter.encoder_fingerprint().get("fp")  # flows into thresholds.json
    except Exception:  # noqa: BLE001                                         # -> snapshot content_hash
        pass
    policy.save(cfg.thresholds_path)
    log.info("calibrate: %d classes -> %s", len(policy.thresholds), cfg.thresholds_path)
    return policy


def route(adapter: DatasetAdapter, cfg: Config, policy: ThresholdPolicy | None = None) -> dict:
    if policy is None and not cfg.thresholds_path.exists():  # U2: name the prerequisite, not "Errno 2"
        raise ValueError("尚未校準:請先執行 vix calibrate(會產生 thresholds.json)")
    policy = policy or ThresholdPolicy.load(cfg.thresholds_path)
    cal_backend = policy.meta.get("embedding_backend")  # AI6: distance thresholds are backend-specific
    backend_mismatch = bool(cal_backend) and cal_backend != cfg.embedding_backend
    if backend_mismatch:
        log.warning("route: 校準後端(%s)≠ 目前後端(%s);距離門檻不可靠,請以同一後端重新 calibrate",
                    cal_backend, cfg.embedding_backend)
    scorer = OutlierScorer(_emb_by_class(adapter, {Tag.GOLDEN}), k=cfg.knn_k)
    dlog = DecisionLog(cfg.decision_log_path)
    counts = {Routing.PASS: 0, Routing.REVIEW: 0}
    decisions: dict[str, str] = {}

    for h, _src, dets, tags in adapter.samples():
        if Tag.GOLDEN in tags or Tag.ANCHOR in tags or Tag.EVAL in tags:
            continue  # never re-route reference / held-out eval data
        scores = scorer.score_image(dets)  # fills det.knn_dist / low_support
        reasons: set[str] = set()
        decision = Routing.PASS
        if not dets:
            decision = Routing.REVIEW
            reasons.add("no_detection")
        for d in dets:
            rr = policy.route(
                d.label,
                d.confidence,
                d.knn_dist if d.knn_dist is not None else float("inf"),
                d.low_support,
            )
            if rr.decision == Routing.REVIEW:
                decision = Routing.REVIEW
                reasons.update(rr.reasons)

        adapter.attach_fields(
            h,
            {
                "yolo_conf_max": scores.conf_max,
                "knn_dist": scores.knn_dist,
                "routing_decision": decision,
                "flag_reason": sorted(reasons),
            },
        )
        adapter.apply_tags(h, [decision])
        decisions[h] = decision
        dlog.append(
            "route",
            vix_hash=h,
            decision=decision,
            scores={"conf_max": scores.conf_max, "knn_dist": scores.knn_dist},
            thr_version=policy.version,
            extra={"reasons": sorted(reasons), "embedding_backend": cfg.embedding_backend},
        )
        counts[decision] += 1

    # routing snapshot (for before/after diff) — rotate current -> prev
    cur = cfg.workspace / "routing_current.json"
    if cur.exists():
        cur.replace(cfg.workspace / "routing_prev.json")
    cur.write_text(json.dumps(decisions), encoding="utf-8")

    total = counts[Routing.PASS] + counts[Routing.REVIEW]
    flag_rate = counts[Routing.REVIEW] / total if total else 0.0
    warning = None
    if total and flag_rate > 0.8:
        warning = f"覆核率 {flag_rate:.0%} 過高,門檻可能過嚴,建議複查"
    elif total and flag_rate < 0.05:
        warning = f"覆核率 {flag_rate:.0%} 過低,門檻可能過鬆,可能放過邊界樣本"
    if warning:
        log.warning("route: %s", warning)
    counts["flag_rate"] = round(flag_rate, 3)
    counts["warning"] = warning
    counts["backend_mismatch"] = backend_mismatch
    log.info("route: %d pass, %d review (flag_rate=%.2f)",
             counts[Routing.PASS], counts[Routing.REVIEW], flag_rate)
    return counts


def build_reference(adapter: DatasetAdapter, cfg: Config) -> FrozenReference:
    anchor = _emb_by_class(adapter, {Tag.ANCHOR})
    golden = _emb_by_class(adapter, {Tag.GOLDEN})
    if not anchor:
        raise ValueError("no anchor samples (tag 'anchor') found to build FrozenReference")
    ref = FrozenReference.build(anchor, golden or None, cfg.knn_k)
    ref.save(cfg.anchor_ref_path)
    log.info("build_reference: %d anchor classes, baseline_consistency=%.3f",
             len(anchor), ref.baseline_consistency)
    return ref


def guard(adapter: DatasetAdapter, cfg: Config, ack: str | None = None) -> GuardReport:
    ref = FrozenReference.load(cfg.anchor_ref_path)
    new = _emb_by_class(adapter, {Tag.REVIEW, Tag.PASS})  # candidate (non-reference) data
    if not new:
        new = {
            c: e
            for c, e in _emb_by_class(adapter, set()).items()
        }  # fall back to everything if untagged
    report = ref.guard(new, cfg.drift_shift_threshold, cfg.consistency_drop_threshold, cfg.knn_k)
    dlog = DecisionLog(cfg.decision_log_path)
    if report.triggered:
        dlog.append(
            "guard_alert",
            decision="ACK" if ack else "HOLD",
            extra={
                "reasons": report.reasons,
                "max_shift": report.max_shift,
                "consistency_drop": report.consistency_drop,
                "ack": ack or "",
            },
        )
        if ack:
            log.warning("guard ACK: %s | reasons=%s shift=%.3f drop=%.3f",
                        ack, report.reasons, report.max_shift, report.consistency_drop)
        else:
            log.warning(
                "GUARD TRIGGERED reasons=%s shift=%.3f drop=%.3f — re-run with --ack '<reason>' to proceed",
                report.reasons, report.max_shift, report.consistency_drop,
            )
    else:
        log.info("guard ok (max_shift=%.3f drop=%.3f)", report.max_shift, report.consistency_drop)
    return report


def export(
    adapter: DatasetAdapter,
    cfg: Config,
    class_names: list[str],
    dst: str | Path,
    copy_images: bool = False,
) -> dict:
    # exclude samples marked rejected/dismissed (PII removal, harmful) — they must NOT export (AD8)
    records = [
        (src, dets)
        for _h, src, dets, tags in adapter.samples()
        if Tag.GOLDEN in tags and Tag.REJECTED not in tags
    ]
    if not records:  # U2: nothing to export -> name the prerequisite instead of writing an empty dir
        has_provisional = any(Tag.PROVISIONAL in tags for _h, _s, _d, tags in adapter.samples())
        if has_provisional:  # diagnose flow: imported labels are PROVISIONAL refs, not golden (honesty firewall)
            raise ValueError(
                "尚無 golden 可匯出。這個資料集是 diagnose 匯入的「參照標籤(未覆核,非 golden)」——"
                "你本就擁有這些標籤檔,在你的標註工具修正後直接重訓即可;"
                "若要透過 VIX 匯出,請先 vix resolve <hash> --confirm 將覆核過的樣本併入 golden。")
        raise ValueError("尚無 golden 可匯出:請先覆核並併入 golden(vix review-queue → confirm)")
    res = DatasetExporter(class_names).export(records, dst, copy_images=copy_images)
    manifest = verify_mod.write_dir_manifest(dst)  # hashes images + labels + data.yaml (U8/V8)
    res["export_manifest"] = str(manifest)
    # boxes_hash: fingerprint the EXPORTED detections (label+geometry) so the audit log records WHAT
    # boxes trained, not just how many — a native box edit now changes this on the export event (audit hole)
    import hashlib as _hl
    _canon = sorted([[Path(src).name, sorted([[d.label, *(round(v, 4) for v in d.bbox.as_tuple())] for d in dets])]
                     for src, dets in records])
    res["boxes_hash"] = _hl.sha256(json.dumps(_canon, sort_keys=True).encode("utf-8")).hexdigest()
    DecisionLog(cfg.decision_log_path).append(
        "export", decision="golden",
        extra={"dst": str(dst), "embedding_backend": cfg.embedding_backend, **res},
    )
    log.info("export: %d images, %d labels -> %s", res["n_images"], res["n_labels"], dst)
    return res


# --- dataset analytics stages (S2–S10) -----------------------------------

def _parse_meta(tags) -> tuple[str, str]:
    """Extract batch / split from convention tags 'batch:<id>' / 'split:<name>'."""
    batch = split = ""
    for t in tags:
        if t.startswith("batch:"):
            batch = t[len("batch:"):]
        elif t.startswith("split:"):
            split = t[len("split:"):]
    return batch, split


def _resolve_tag(adapter, tag):
    """Accept a bare batch id: resolve 'w23' -> 'batch:w23' if that's what samples carry;
    warn if a tag matches nothing (instead of silently comparing 0 samples)."""
    have: set[str] = set()
    for _h, _s, _d, tags in adapter.samples():
        have.update(tags)
    if tag in have:
        return tag
    if f"batch:{tag}" in have:
        return f"batch:{tag}"
    log.warning("tag '%s' 未匹配任何樣本(也無 batch:%s);請確認標籤名稱", tag, tag)
    return tag


def _detection_items(adapter, want_tags=None, exclude_tags=None) -> list[EmbItem]:
    items: list[EmbItem] = []
    for h, _src, dets, tags in adapter.samples():
        ts = set(tags)
        if want_tags and not (set(want_tags) & ts):
            continue
        if exclude_tags and (set(exclude_tags) & ts):
            continue
        batch, split = _parse_meta(tags)
        for i, det in enumerate(dets):
            if det.embedding is not None:
                items.append(EmbItem(f"{h}:{i}", det.label, np.asarray(det.embedding, float),
                                     det.confidence, batch=batch, split=split))
    return items


def _image_items(adapter, want_tags=None, exclude_tags=None) -> list[EmbItem]:
    items: list[EmbItem] = []
    for h, _src, dets, tags in adapter.samples():
        ts = set(tags)
        if want_tags and not (set(want_tags) & ts):
            continue
        if exclude_tags and (set(exclude_tags) & ts):
            continue
        embs = [np.asarray(d.embedding, float) for d in dets if d.embedding is not None]
        if embs:
            label = dets[0].label if dets else ""
            conf = max((d.confidence for d in dets), default=0.0)
            batch, split = _parse_meta(tags)
            items.append(EmbItem(h, label, np.mean(np.vstack(embs), axis=0), conf,
                                 batch=batch, split=split))
    return items


def audit_labels(adapter, cfg, k=None):  # S2
    issues = suspected_label_errors(_detection_items(adapter, want_tags=[Tag.GOLDEN]), k or cfg.knn_k)
    DecisionLog(cfg.decision_log_path).append(
        "audit_labels", decision=str(len(issues)), extra={"top": [i.id for i in issues[:20]]}
    )
    log.info("audit_labels: %d suspected label errors", len(issues))
    return issues


def near_dup_label_conflicts(adapter, cfg, max_distance=0.03):  # S2b: causal-certain label errors via near-dups
    """Golden images that are near-duplicates (DINO) yet carry CONFLICTING labels — at least one is
    mislabelled. Causal-certain (not a proxy): near-identical pixels can't legitimately differ in label.
    Advisory — surfaces the contradiction to review, never auto-resolves."""
    from .core.analytics import near_dup_label_conflicts as _conf
    conflicts = _conf(_detection_items(adapter, want_tags=[Tag.GOLDEN]), max_distance)
    log.info("near_dup_label_conflicts: %d conflicting near-dup groups", len(conflicts))
    return conflicts


def dedup(adapter, cfg, max_distance=0.05):  # S3
    groups = near_duplicate_groups(_image_items(adapter), max_distance)
    redundant = sum(len(g) - 1 for g in groups)
    log.info("dedup: %d near-duplicate groups (%d redundant images)", len(groups), redundant)
    return groups


def coverage(adapter, cfg, target=None):  # S5
    items = _detection_items(adapter, want_tags=[Tag.GOLDEN])
    dist = class_distribution(items)
    gaps = coverage_gaps(items, k=min(5, cfg.knn_k), target=target)
    under = [c for c, v in gaps.items() if v["under_represented"]]
    log.info("coverage: %d classes; under-represented=%s", len(dist), under)
    return {"distribution": dist, "gaps": gaps}


def coverage_value(adapter, cfg, radius=0.2):  # S4
    new = _image_items(adapter, exclude_tags=[Tag.GOLDEN, Tag.ANCHOR, Tag.EVAL])
    existing = _image_items(adapter, want_tags=[Tag.GOLDEN])
    res = coverage_delta(new, existing, radius)
    log.info("coverage_value: %.1f%% of %d new images cover novel regions",
             res["novel_fraction"] * 100, len(new))
    return res


def active_learn(adapter, cfg, budget):  # S6
    cands = _image_items(adapter, exclude_tags=[Tag.GOLDEN, Tag.ANCHOR, Tag.EVAL])
    existing = _image_items(adapter, want_tags=[Tag.GOLDEN])
    ranked = active_learning_ranking(cands, existing, budget, return_reasons=True)
    for r in ranked:  # build the rationale from whichever signals are actually high (no hardcoded claim)
        parts = []
        if r["uncertainty"] >= 0.3:
            parts.append(f"模型信心低(不確定度 {r['uncertainty']})")
        if r["novelty"] >= 0.1:
            parts.append(f"與既有資料差異大(新穎度 {r['novelty']})")
        if not parts:
            parts.append(f"綜合分數 {r['score']}(不確定度 {r['uncertainty']}、新穎度 {r['novelty']})")
        r["why"] = " 且 ".join(parts) + " — 標注效益相對較高"
    log.info("active_learn: %d candidates -> top %d selected", len(cands), len(ranked))
    return ranked


def drift_periods(adapter, cfg, tag_a, tag_b, top=3):  # S7 (cross-time)
    tag_a, tag_b = _resolve_tag(adapter, tag_a), _resolve_tag(adapter, tag_b)
    a = _detection_items(adapter, want_tags=[tag_a])
    b = _detection_items(adapter, want_tags=[tag_b])
    result = cross_period_drift(a, b, cfg.drift_shift_threshold, top)
    alerts = [c for c, v in result.items() if v["alert"]]
    if alerts:
        DecisionLog(cfg.decision_log_path).append(
            "drift_periods", decision="ALERT", extra={"classes": alerts}
        )
    log.info("drift_periods: %d classes compared, alerts=%s", len(result), alerts)
    return result


def _box_digests(adapter, want_tags: set) -> dict:
    """Canonical per-sample box fingerprint (label + rounded bbox) over tagged samples. Folded into the
    snapshot / training-pool content hash so a native-editor box edit (tighten/add/delete/relabel)
    changes the audit identity — closing the box-level audit hole (vix_hash hashes image bytes only,
    so box edits flow into export but were previously invisible to content_hash + the DecisionLog)."""
    want = set(want_tags)
    out: dict = {}
    for h, _s, dets, t in adapter.samples():
        if want & set(t):
            out[h] = sorted([[d.label, *(round(v, 4) for v in d.bbox.as_tuple())] for d in dets])
    return out


def snapshot(adapter, cfg, version):  # S9
    # freeze the FULL thresholds (values, not just meta) so the decision is reproducible
    thr_meta = (
        json.loads(cfg.thresholds_path.read_text(encoding="utf-8"))
        if cfg.thresholds_path.exists()
        else {}
    )
    # also freeze the anchor reference fingerprint so guard's baseline is reproducible
    if cfg.anchor_ref_path.exists():
        from .core.manifest import compute_hash

        thr_meta = {**thr_meta, "anchor_ref_sha256": compute_hash(cfg.anchor_ref_path)}
    out = cfg.workspace / f"snapshot_{version}.json"
    snap = snap_mod.create_snapshot(cfg.manifest_path, out, version, thr_meta, cfg.decision_log_path,
                                    box_digests=_box_digests(adapter, {Tag.GOLDEN}))  # bind box content (audit hole)
    log.info("snapshot: %s (%d golden, %d excluded) -> %s",
             version, snap["n_golden"], snap["n_excluded"], out)
    return snap, out


def restore(cfg, path):  # S9
    r = snap_mod.restore(path)
    log.info("restore: version=%s, %d golden, %d excluded", r["version"], len(r["composition"]), len(r["excluded"]))
    return r


def _latest_prior_report(cfg):
    d = cfg.workspace / "reports"
    files = sorted(d.glob("report_*.json")) if d.exists() else []
    return json.loads(files[-1].read_text(encoding="utf-8")) if files else None


def health_report(adapter, cfg, out_dir, version="current", prev=None):  # S10
    if prev is None:
        prev = _latest_prior_report(cfg)  # auto week-over-week baseline (W4)
    gold = _detection_items(adapter, want_tags=[Tag.GOLDEN])
    dist = class_distribution(gold)
    gaps = coverage_gaps(gold, k=min(5, cfg.knn_k))
    dups = near_duplicate_groups(_image_items(adapter, want_tags=[Tag.GOLDEN]))
    issues = suspected_label_errors(gold, cfg.knn_k)
    rows = list(adapter.samples())
    total = len(rows)
    pass_c = sum(1 for _h, _s, _d, t in rows if Tag.PASS in t)
    review_c = sum(1 for _h, _s, _d, t in rows if Tag.REVIEW in t)
    n_batches = len({_parse_meta(t)[0] for _h, _s, _d, t in rows} - {""})

    under = [c for c, v in gaps.items() if v["under_represented"]]
    ranked: list[tuple[int, str]] = []  # (magnitude, action) -> highest-impact first (AH3)
    if issues:
        ranked.append((len(issues), f"覆核疑似標錯 {min(len(issues), 20)} 筆(vix audit-labels)"))
    if dups:
        ranked.append((len(dups), f"處理 {len(dups)} 群近似重複(vix dedup)"))
    if under:
        ranked.append((len(under), f"補採樣本不足的類別: {under}(vix coverage --target)"))
    ranked.sort(key=lambda x: -x[0])
    suggestions = [t for _n, t in ranked]
    if suggestions:
        suggestions[0] = "本週首要(triage 排序,非實測 mAP): " + suggestions[0]
    else:
        suggestions.append("資料集健康,可直接 vix export 進訓練")

    gate_verdict = pre_train_gate(
        n_review_open=review_c, under_represented=under
    ).verdict
    report = build_report(
        version=version, total=total, class_dist=dist, pass_count=pass_c, review_count=review_c,
        duplicate_groups=dups, label_issues=issues, coverage=gaps, prev=prev,
        n_batches=n_batches, suggestions=suggestions, gate_verdict=gate_verdict,
        embedding_backend=cfg.embedding_backend,
    )
    paths = write_report(report, out_dir)
    # keep a timestamped copy so the next run can auto-diff against it
    hist = cfg.workspace / "reports"
    hist.mkdir(parents=True, exist_ok=True)
    stamp = report["generated_at"].replace(":", "").replace("-", "")
    (hist / f"report_{stamp}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    DecisionLog(cfg.decision_log_path).append("report", decision=version, extra=paths)
    log.info("health_report: -> %s", paths["md"])
    return report, paths


def review_queue(adapter, cfg, top=50):  # T3 + T7
    cands = _image_items(adapter, exclude_tags=[Tag.GOLDEN, Tag.ANCHOR, Tag.REJECTED, Tag.EVAL])
    ref = _image_items(adapter, want_tags=[Tag.GOLDEN])
    label_issue_ids = {i.id.split(":")[0] for i in audit_labels(adapter, cfg)}  # AJ2/AJ6: activate label-error risk
    ranked = _review_queue(cands, ref, k=cfg.knn_k, label_issue_ids=label_issue_ids)[:top]
    log.info("review_queue: %d candidates ranked, returning top %d", len(cands), len(ranked))
    return [
        {"id": r.id, "risk": r.risk, "reasons": r.reasons, "why": explain(r.reasons, r.scores)}
        for r in ranked
    ]


def status(adapter, cfg):
    """'Where am I in the loop + what to run next' (U4). Reads only tags + which workspace artifacts
    exist — no computation, no claims about quality/mAP. The daily re-entry verb; the next-step rule
    is conditional (branches on state), not a hardcoded line. Pure: InMemory-testable."""
    n = {"total": 0, "golden": 0, "anchor": 0, "review": 0, "rejected": 0, "pass": 0, "eval": 0, "admitted": 0}
    key = {Tag.GOLDEN: "golden", Tag.ANCHOR: "anchor", Tag.REVIEW: "review", Tag.REJECTED: "rejected",
           Tag.PASS: "pass", Tag.EVAL: "eval", Tag.ADMITTED: "admitted"}
    has_det = has_emb = False
    for _h, _s, dets, tags in adapter.samples():
        n["total"] += 1
        for t, k in key.items():
            if t in tags:
                n[k] += 1
        if dets:
            has_det = True
            if any(d.embedding is not None for d in dets):
                has_emb = True
    has_thr = cfg.thresholds_path.exists()
    has_eval = cfg.eval_results_path.exists()
    routed = (n["pass"] + n["review"]) > 0
    if n["total"] == 0:
        nxt = ("ingest", "vix ingest ./golden --batch init --golden")
    elif not has_det:
        nxt = ("infer", "vix infer --weights yolo.pt")
    elif not has_emb:
        nxt = ("embed", "vix embed")
    elif not has_thr:
        nxt = ("calibrate", "vix calibrate")
    elif not routed:
        nxt = ("route", "vix route")
    elif n["review"] > 0:
        nxt = ("review", f"vix review-queue --top 40   # 尚有 {n['review']} 筆待覆核")
    elif not has_eval:
        nxt = ("eval-ingest", "(選用) vix eval-ingest <val.jsonl> → vix weakness-report")
    else:
        nxt = ("gate", "vix gate   # 能不能訓練? 然後 vix export")
    return {"counts": n, "has_detections": has_det, "has_embeddings": has_emb,
            "has_thresholds": has_thr, "has_eval": has_eval, "routed": routed,
            "next": {"stage": nxt[0], "cmd": nxt[1]}}


def audit(cfg, since=None, until=None, event=None, reviewer=None):  # T8
    recs = DecisionLog(cfg.decision_log_path).read_all()

    def keep(r):
        if event and r.get("event") != event:
            return False
        if reviewer and r.get("reviewer_id") != reviewer:
            return False
        if since and r.get("ts_utc", "") < since:
            return False
        if until and r.get("ts_utc", "") > until:
            return False
        return True

    out = [r for r in recs if keep(r)]
    log.info("audit: %d/%d records match", len(out), len(recs))
    return out


def merge_maps(map_a: dict, map_b: dict, overrides: dict | None = None):  # T2
    res = merge_class_maps(map_a, map_b, overrides)
    log.info("merge_maps: %d unified classes, %d need decision, %d orphans",
             len(res["unified_names"]), len(res["needs_decision"]), len(res["orphans"]))
    return res


def relabel_dataset(adapter, cfg, mapping: dict[str, str], change_log_path=None):  # T4
    records, det_refs, dets_by_h = [], {}, {}
    for h, _src, dets, _t in adapter.samples():
        dets_by_h[h] = dets
        for i, det in enumerate(dets):
            rid = f"{h}:{i}"
            records.append((rid, det.label))
            det_refs[rid] = det
    new, changes = _relabel(records, mapping)
    for rid, label in new:
        det_refs[rid].label = label  # mutate in place (in-memory refs)
    for h in {c.id.split(":")[0] for c in changes}:  # persist so the change survives across CLI invocations
        adapter.set_detections(h, dets_by_h[h])
    diff = migration_diff(changes)
    path = Path(change_log_path or (cfg.workspace / "relabel_changes.jsonl"))
    with open(path, "a", encoding="utf-8") as f:
        for c in changes:
            f.write(json.dumps({"id": c.id, "old": c.old, "new": c.new}) + "\n")
    DecisionLog(cfg.decision_log_path).append(
        "relabel", decision=str(diff["total_changed"]), extra=diff["by_transition"]
    )
    log.info("relabel: %d labels changed -> %s", diff["total_changed"], path)
    return diff


# --- round-3 additions (U1–U10) ------------------------------------------

def new_classes(adapter, cfg, novelty_radius=0.3, cluster_distance=0.2):  # U1
    ref = _detection_items(adapter, want_tags=[Tag.GOLDEN])
    query = _detection_items(adapter, exclude_tags=[Tag.GOLDEN, Tag.ANCHOR, Tag.EVAL])
    clusters = suspected_new_classes(query, ref, novelty_radius, cluster_distance)
    for c in clusters:
        n = len(c["ids"])
        c["size"] = n
        c["suggestion"] = (
            f"處置建議:此群 {n} 張、規模較大 → 建議加入白名單為新類別"
            if n >= 3
            else f"處置建議:此群僅 {n} 張、規模小 → 建議人工複查(可能為雜訊,或映射至最近已知類別)"
        )
    if clusters:
        DecisionLog(cfg.decision_log_path).append(
            "new_classes", decision=str(len(clusters)), extra={"clusters": len(clusters)}
        )
    log.info("new_classes: %d suspected new-class clusters", len(clusters))
    return clusters


def leakage(adapter, cfg, max_distance=0.05):  # U3
    leaks = cross_split_leakage(_image_items(adapter), max_distance)
    log.info("leakage: %d cross-split duplicate groups", len(leaks))
    return leaks


def harmful(adapter, cfg, top=50):  # U5
    gold = _image_items(adapter, want_tags=[Tag.GOLDEN])
    label_issue_imgs = {i.id.split(":")[0] for i in audit_labels(adapter, cfg)}
    dup_ids = {m for g in near_duplicate_groups(gold) for m in g}
    ranked = harmful_ranking(gold, label_issue_imgs, dup_ids, top=top)
    log.info("harmful: ranked %d golden images", len(gold))
    return ranked


def quality_trend(adapter, cfg, drop_threshold=0.15):  # U10
    res = class_quality_trend(_detection_items(adapter), drop_threshold=drop_threshold)
    if res["alerts"]:
        DecisionLog(cfg.decision_log_path).append(
            "quality_trend", decision="ALERT", extra={"alerts": res["alerts"]}
        )
    log.info("quality_trend: %d drop alerts", len(res["alerts"]))
    return res


def reviewer_audit(adapter, cfg, sim_threshold=0.9, class_filter=None, min_samples=10):  # U2 / Z7
    decisions = [
        {"reviewer_id": r.get("reviewer_id", ""), "id": r.get("vix_hash", ""), "decision": r.get("decision", "")}
        for r in DecisionLog(cfg.decision_log_path).read_all()
        if r.get("vix_hash") and r.get("event") in ("route", "review")
    ]
    result = reviewer_consistency(decisions, _image_items(adapter), sim_threshold, label_filter=class_filter)
    counts: dict[str, int] = {}
    for d in decisions:
        counts[d["reviewer_id"]] = counts.get(d["reviewer_id"], 0) + 1
    for rid, info in result.items():  # annotate so a rubber-stamper / low-sample reviewer is visible
        if isinstance(info, dict):
            info["n_decisions"] = counts.get(rid, 0)
            info["insufficient"] = counts.get(rid, 0) < min_samples  # too few to judge consistency
    log.info("reviewer_audit: %d reviewers analysed (class=%s)", len(result), class_filter or "all")
    return result


def pre_train_gate_stage(adapter, cfg, drift_triggered=None):  # U7
    # auto-wire guard -> gate: if a frozen reference exists, detect drift ourselves
    if drift_triggered is None:
        drift_triggered = False
        if cfg.anchor_ref_path.exists():
            try:
                ref = FrozenReference.load(cfg.anchor_ref_path)
                new = _emb_by_class(adapter, {Tag.REVIEW, Tag.PASS}) or _emb_by_class(adapter, set())
                drift_triggered = ref.guard(
                    new, cfg.drift_shift_threshold, cfg.consistency_drop_threshold, cfg.knn_k
                ).triggered
            except Exception as exc:  # noqa: BLE001 - drift detection is best-effort here
                log.warning("gate: drift auto-check skipped (%s)", exc)
    rows = list(adapter.samples())
    n_review = sum(1 for _h, _s, _d, t in rows if Tag.REVIEW in t)
    n_golden = sum(1 for _h, _s, _d, t in rows if Tag.GOLDEN in t)  # AF1: no golden -> NO-GO, not false GO
    eval_golden_overlap = sum(1 for _h, _s, _d, t in rows if Tag.EVAL in t and Tag.GOLDEN in t)  # AH2 leak
    gaps = coverage_gaps(_detection_items(adapter, want_tags=[Tag.GOLDEN]), k=min(5, cfg.knn_k))
    under = [c for c, v in gaps.items() if v["under_represented"]]

    # golden/train overlap = near-duplicate groups spanning golden vs train split
    items = []
    for h, _s, dets, tags in rows:
        embs = [np.asarray(d.embedding, float) for d in dets if d.embedding is not None]
        if not embs:
            continue
        split = "golden" if Tag.GOLDEN in tags else _parse_meta(tags)[1]
        items.append(EmbItem(h, dets[0].label if dets else "", np.mean(np.vstack(embs), axis=0), split=split))
    overlap = sum(
        1 for g in cross_split_leakage(items) if {"golden", "train"} <= set(g["splits"])
    )
    _dl = DecisionLog(cfg.decision_log_path)
    dlog_all = _dl.read_all()
    audit_ok = _dl.verify_chain() and not _dl.is_truncated()  # tampered OR tail-truncated ledger -> NO-GO
    backends = {r.get("extra", {}).get("embedding_backend") for r in dlog_all}
    backends.discard(None)
    backend_mixed = len(backends) > 1  # AI6: mixing pixel_fallback + DINOv2 makes thresholds/trends incomparable

    extra_reasons: list[str] = []
    extra_checks: dict = {}
    # encoder fingerprint: the encoder behind every PROXY number is now in the audit truth; if the data's
    # current encoder differs from the one used to calibrate (swapped weights / torch upgrade / CPU<->GPU /
    # re-pulled cache), the distance thresholds are invalid -> NO-GO. Fail-open when absent (legacy).
    cal_fp = ThresholdPolicy.load(cfg.thresholds_path).meta.get("encoder_fp") if cfg.thresholds_path.exists() else None
    try:
        live_fp = adapter.encoder_fingerprint().get("fp")
    except Exception:  # noqa: BLE001
        live_fp = None
    if cal_fp and live_fp and cal_fp != live_fp:
        extra_reasons.append("資料目前的編碼器指紋與 calibrate 時不一致(權重/前處理/行為已改),距離門檻不可比;請以同一編碼器重新 vix embed + calibrate")
        extra_checks["encoder_fp_mismatch"] = {"calibrated": cal_fp, "current": live_fp}

    # challenge-guard: opt-in mAP regression block (only when an eval result + a frozen baseline exist)
    if cfg.eval_results_path.exists() and cfg.eval_baseline_path.exists():
        cur = json.loads(cfg.eval_results_path.read_text(encoding="utf-8"))
        base = json.loads(cfg.eval_baseline_path.read_text(encoding="utf-8"))
        blocking, advisory = regression_check(
            cur.get("per_class_ap", {}), base.get("per_class_ap", {}),
            float(cur.get("mAP") or 0.0), float(base.get("mAP") or 0.0),
            map_drop_thr=float(base.get("map_drop_thr", 0.02)),
            protected=base.get("protected", {}),
            eval_support={k: int(v) for k, v in cur.get("n_gt", {}).items()},
            eval_set_changed=(cur.get("eval_set_hash") != base.get("eval_set_hash")),
        )
        extra_reasons += blocking
        if advisory:
            extra_checks["regression_advisory"] = advisory

    # consistency gate (opt-in): a SUPPORTED taxonomy/label_noise verdict on a PROTECTED class pair
    # blocks — a poisoned class definition must not be exported into an expensive external retrain.
    # representation_fixable pairs (a learned projection separates them) are NOT blocked: encoder
    # limit, not a definition dead-end. Needs the baseline's protected set + an eval (for confusion).
    if cfg.eval_baseline_path.exists() and cfg.eval_results_path.exists():
        protected = set(json.loads(cfg.eval_baseline_path.read_text(encoding="utf-8")).get("protected", {}))
        if protected:
            from .core.consistency import consistency_findings
            ev2 = json.loads(cfg.eval_results_path.read_text(encoding="utf-8"))
            for f in consistency_findings(_emb_by_class(adapter, {Tag.GOLDEN}),
                                          ev2.get("confusion"), ev2.get("n_gt"), adapt_rescued=_adapt_rescued(cfg)):
                if (f["verdict"] in ("taxonomy", "label_noise") and f["tier"] == "supported"
                        and not f.get("representation_fixable") and (set(f["pair"]) & protected)):
                    extra_reasons.append(
                        f"受保護類別對 {f['pair'][0]}↔{f['pair'][1]} 判定 {f['verdict']}(類別定義疑有問題)"
                        "— 先重新裁決/釐清定義再匯出/重訓")

    result = pre_train_gate(
        n_review_open=n_review, golden_train_overlap=overlap,
        under_represented=under, drift_triggered=drift_triggered,
        audit_chain_intact=audit_ok, n_golden=n_golden, eval_golden_overlap=eval_golden_overlap,
        backend_mixed=backend_mixed, extra_reasons=extra_reasons, extra_checks=extra_checks,
    )
    DecisionLog(cfg.decision_log_path).append(
        "pre_train_gate", decision=result.verdict, extra={"reasons": result.reasons}
    )
    log.info("pre_train_gate: %s (%s)", result.verdict, result.reasons)
    return result


def batch_gate(adapter, cfg, batch, max_distance=0.05, worklist=False):
    """The weekly 'can THIS batch go into training? what must I clean first?' verdict.

    HYGIENE + leakage-safety only — NOT a mAP-gain promise (VIX doesn't retrain). Two CAUSAL-harm
    BLOCK checks: (1) batch -> frozen eval/golden near-duplicate leakage (silently inflates the mAP
    the gate trusts) — the one genuinely-new check; (2) degenerate boxes in the batch. Plus an
    advisory clean-list (open review / suspected label errors / within-batch dups / confidently-wrong),
    which never blocks. Verdict BLOCK/PARTIAL/CLEAN/PASS (PARTIAL = no frozen eval/golden, so leakage
    is uncheckable — never a silent PASS). Reuses existing batch-scoped primitives; logs a batch_gate
    audit entry. No score, no dashboard."""
    from .core.analytics import cross_split_leakage, near_duplicate_groups, suspected_label_errors
    from .core.box_qa import audit_boxes
    from .core.gate import batch_gate_verdict

    btag = f"batch:{batch}"
    rows = list(adapter.samples())
    batch_hashes = {h for h, _s, _d, t in rows if btag in t}
    if not batch_hashes:
        raise ValueError(f"找不到 batch '{batch}'(tag {btag});請先 vix ingest --batch {batch}")

    # BLOCK 1 (NEW): leakage batch -> frozen eval/golden (contamination inflates the trusted mAP)
    items = []
    for h, _s, dets, tags in rows:
        embs = [np.asarray(d.embedding, float) for d in dets if d.embedding is not None]
        if not embs:
            continue
        if Tag.GOLDEN in tags or Tag.EVAL in tags:
            split = "frozen"
        elif btag in tags:
            split = "batch"
        else:
            continue  # only compare the batch against the frozen eval/golden set
        items.append(EmbItem(h, dets[0].label if dets else "", np.mean(np.vstack(embs), axis=0), split=split))
    eval_available = any(it.split == "frozen" for it in items)
    leak_ids = set()
    if eval_available:
        for g in cross_split_leakage(items, max_distance):
            if {"frozen", "batch"} <= set(g["splits"]):
                leak_ids |= {i for i in g["ids"] if i in batch_hashes}  # batch-side offenders

    # BLOCK 2: degenerate boxes in the batch (malformed training targets)
    recs = [{"id": h, "label": d.label, "bbox": d.bbox.as_tuple()}
            for h, _s, dets, t in rows if btag in t for d in dets]
    degenerate = sorted({i["id"] for i in audit_boxes(recs) if i["issue"] == "degenerate"})

    # advisory clean-list (never blocks)
    n_review = sum(1 for h, _s, _d, t in rows if btag in t and Tag.REVIEW in t)
    noise = [iss.id for iss in suspected_label_errors(_detection_items(adapter, want_tags=[btag]), cfg.knn_k)]
    within_dups = sum(len(g) - 1 for g in near_duplicate_groups(_image_items(adapter, want_tags=[btag]), max_distance))
    try:
        conf_wrong = [r["id"] for r in hardneg(adapter, cfg, mode="gt_free", batch=batch)["rows"]]
    except (ValueError, OSError):
        conf_wrong = []

    backend_ok = str(cfg.embedding_backend or "").startswith("dinov2")
    block = {"eval_leakage": sorted(leak_ids), "degenerate_boxes": degenerate}
    clean = {"open_review": n_review, "label_noise": noise,
             "within_batch_dups": within_dups, "confident_wrong": conf_wrong}
    if worklist:  # tag offenders vixq:batch:* so `vix app` surfaces them as clickable saved views
        for ids, tag in ((leak_ids, "vixq:batch:leakage"), (degenerate, "vixq:batch:degenerate"),
                         (noise, "vixq:batch:label_noise"), (conf_wrong, "vixq:batch:confident_wrong")):
            for h in ids:
                try:
                    adapter.apply_tags(h, [tag])
                except Exception:  # noqa: BLE001 - id may not be a live sample
                    pass
    verdict, reasons = batch_gate_verdict(block, clean, eval_available, backend_ok)
    DecisionLog(cfg.decision_log_path).append(
        "batch_gate", batch_id=str(batch), decision=verdict,
        extra={"n_batch": len(batch_hashes), "eval_available": eval_available, "backend": cfg.embedding_backend,
               "block": {k: len(v) for k, v in block.items()},
               "clean": {k: (len(v) if isinstance(v, list) else v) for k, v in clean.items()},
               "note": "hygiene + leakage safety, NOT a mAP-gain promise (no retraining in VIX)"})
    log.info("batch_gate %s: %s (%d imgs; %s)", batch, verdict, len(batch_hashes), reasons)
    return {"batch": batch, "verdict": verdict, "reasons": reasons, "n_batch": len(batch_hashes),
            "block": block, "clean": clean, "eval_available": eval_available}


def _training_pool_hash(adapter, cfg):
    """Content hash over the training pool (golden ∪ admitted) + thresholds meta — the checkpoint
    anchor for batch-admit (changes when a batch is admitted/un-admitted)."""
    from .core.snapshot import _content_hash
    pool_tags = {Tag.GOLDEN, Tag.ADMITTED}
    pool = sorted(h for h, _s, _d, t in adapter.samples() if pool_tags & set(t))
    thr_meta = ThresholdPolicy.load(cfg.thresholds_path).meta if cfg.thresholds_path.exists() else {}
    return _content_hash(pool, thr_meta, _box_digests(adapter, pool_tags))  # box content bound (audit hole)


def batch_admit(adapter, cfg, batch, force=False):
    """Formally admit a gated batch into the training pool — the governance keystone that makes the
    weekly decision DEFENSIBLE (why is w23 in training?), REVERSIBLE (batch-unadmit), and QUERYABLE
    (batch-ledger). Runs batch-gate first; a BLOCK verdict REFUSES admission unless force=True (the
    override is itself logged). Tags the batch `admitted`, records a hash-chained batch_admit event
    binding {verdict, pre/post training-pool content_hash, eval_set_hash, backend} to the batch.
    Hygiene-gated admission — NOT a claim the batch raises mAP."""
    gate = batch_gate(adapter, cfg, batch)
    btag = f"batch:{batch}"
    batch_hashes = sorted(h for h, _s, _d, t in adapter.samples() if btag in t)
    refused = gate["verdict"] == "BLOCK" and not force
    if refused:
        DecisionLog(cfg.decision_log_path).append(
            "batch_admit", batch_id=str(batch), decision="REFUSED",
            extra={"verdict": gate["verdict"], "reasons": gate["reasons"],
                   "block": {k: len(v) for k, v in gate["block"].items()},
                   "note": "admission refused: batch-gate BLOCK (use --force to override, logged)"})
        log.warning("batch_admit %s: REFUSED (gate BLOCK: %s)", batch, gate["reasons"])
        return {"admitted": False, "batch": batch, "verdict": gate["verdict"], "reasons": gate["reasons"]}

    pre_hash = _training_pool_hash(adapter, cfg)
    for h in batch_hashes:
        adapter.apply_tags(h, [Tag.ADMITTED])
    post_hash = _training_pool_hash(adapter, cfg)
    eval_set_hash = None
    if cfg.eval_results_path.exists():
        try:
            eval_set_hash = json.loads(cfg.eval_results_path.read_text(encoding="utf-8")).get("eval_set_hash")
        except (ValueError, OSError):
            pass
    forced = gate["verdict"] == "BLOCK" and force
    DecisionLog(cfg.decision_log_path).append(
        "batch_admit", batch_id=str(batch), decision=("FORCED" if forced else gate["verdict"]),
        extra={"verdict": gate["verdict"], "forced": forced, "n_admitted": len(batch_hashes),
               "pre_hash": pre_hash, "post_hash": post_hash, "eval_set_hash": eval_set_hash,
               "backend": cfg.embedding_backend, "block": {k: len(v) for k, v in gate["block"].items()},
               "note": "hygiene-gated admission record; reversible via batch-unadmit; NOT a mAP-gain claim"})
    log.info("batch_admit %s: %s (%d admitted, pool %s->%s)", batch,
             "FORCED" if forced else gate["verdict"], len(batch_hashes), pre_hash[:8], post_hash[:8])
    return {"admitted": True, "batch": batch, "verdict": gate["verdict"], "forced": forced,
            "n_admitted": len(batch_hashes), "pre_hash": pre_hash, "post_hash": post_hash}


def batch_unadmit(adapter, cfg, batch):
    """Reverse a batch admission: remove the `admitted` tag from the batch's samples and record a
    hash-chained batch_unadmit event (the training-pool hash reverts). The reversible half of the
    governance loop."""
    btag = f"batch:{batch}"
    admitted = [h for h, _s, _d, t in adapter.samples() if btag in t and Tag.ADMITTED in t]
    if not admitted:
        raise ValueError(f"batch '{batch}' 未被 admit(無 {Tag.ADMITTED} tag),無可回退")
    pre_hash = _training_pool_hash(adapter, cfg)
    try:
        for h in admitted:
            adapter.remove_tags(h, [Tag.ADMITTED])
    except NotImplementedError:
        raise ValueError("此 adapter 不支援移除 tag,無法 un-admit") from None
    post_hash = _training_pool_hash(adapter, cfg)
    DecisionLog(cfg.decision_log_path).append(
        "batch_unadmit", batch_id=str(batch), decision="UNADMITTED",
        extra={"n_unadmitted": len(admitted), "pre_hash": pre_hash, "post_hash": post_hash})
    log.info("batch_unadmit %s: %d un-admitted (pool %s->%s)", batch, len(admitted), pre_hash[:8], post_hash[:8])
    return {"unadmitted": len(admitted), "batch": batch, "pre_hash": pre_hash, "post_hash": post_hash}


def batch_ledger(cfg):
    """Which batches are currently admitted into the training pool, and the full admit/un-admit
    history — reconstructed from the hash-chained decision log ('why is w23 in the training set?')."""
    state: dict[str, str] = {}
    history = []
    for r in DecisionLog(cfg.decision_log_path).read_all():
        ev = r.get("event")
        if ev not in ("batch_admit", "batch_unadmit"):
            continue
        b = r.get("batch_id")
        history.append({"batch": b, "event": ev, "ts": r.get("ts_utc"),
                        "decision": r.get("decision"), "extra": r.get("extra", {})})
        if ev == "batch_admit" and r.get("decision") != "REFUSED":
            state[b] = "admitted"
        elif ev == "batch_unadmit":
            state[b] = "unadmitted"
    return {"admitted_batches": sorted(b for b, s in state.items() if s == "admitted"), "history": history}


def batch_trend(cfg):
    """Per-batch gate verdict + admit status across weekly drops (is batch quality drifting?),
    read from the hash-chained decision log."""
    from .core.trend import batch_trend as _batch_trend
    return _batch_trend(DecisionLog(cfg.decision_log_path).read_all())


def explain_one(adapter, cfg, vix_hash):  # U9
    policy = ThresholdPolicy.load(cfg.thresholds_path) if cfg.thresholds_path.exists() else None
    scorer = OutlierScorer(_emb_by_class(adapter, {Tag.GOLDEN}), k=cfg.knn_k)
    label_issue_imgs = {i.id.split(":")[0] for i in audit_labels(adapter, cfg)}
    for h, _s, dets, _t in adapter.samples():
        if h != vix_hash:
            continue
        scores = scorer.score_image(dets)
        label = dets[0].label if dets else ""
        ct = policy.thresholds.get(label) if policy else None
        return explain_image(
            label, scores.conf_max, scores.knn_dist,
            conf_thr=ct.conf_thr if ct else None,
            dist_thr=ct.dist_thr if ct else None,
            label_issue=vix_hash in label_issue_imgs,
        )
    return {"error": f"sample {vix_hash} not found"}


def verify_dataset(cfg, manifest_path, data_dir):  # U8
    res = verify_mod.verify_export(manifest_path, data_dir)
    log.info("verify_dataset: ok=%s checked=%d mismatched=%d missing=%d",
             res["ok"], res["n_checked"], len(res["mismatched"]), len(res["missing"]))
    return res


# --- keystone: close the data <-> model loop (eval ingestion + error mining) ---

def eval_ingest(adapter, cfg, results, iou_thr=0.5, strict_join=False):
    """Ingest a held-out val evaluation (GT + predictions) -> per-class AP, confusion,
    and per-image FP/FN; attach eval_fp/eval_fn fields and store eval_results.json.
    Turns VIX from model-blind (confidence+embedding proxies) into model-validated.

    ``results`` is a path (JSONL / JSON array, BOM-tolerant) OR an already-parsed list of
    {vix_hash, gt, pred} dicts (so eval-run / diagnose feed it without a temp file).

    ``strict_join`` (diagnose/eval-run path): every referenced vix_hash MUST exist in the
    adapter; otherwise raise. This catches the silent ingest<->eval key mismatch (predictions
    keyed by filename stem against a content-hash manifest) that would print a green mAP while
    attaching per-image FP/FN to nothing. The legacy bare `eval-ingest` verb keeps strict_join
    False (best-effort: an external eval may reference hashes not in this adapter view)."""
    from .core.eval_ingest import eval_set_hash, evaluate

    if isinstance(results, (str, Path)):
        raw = Path(results).read_text(encoding="utf-8-sig").strip()
        images = json.loads(raw) if raw.startswith("[") else [
            json.loads(line) for line in raw.splitlines() if line.strip()
        ]
    else:
        images = list(results)
    res = evaluate(images, iou_thr=iou_thr)
    res["eval_set_hash"] = eval_set_hash(images)  # binds this result to the exact eval SET (R6)
    if strict_join:  # KS3 guard: a stem-keyed JSONL against a content-hash manifest fails loudly here
        known = {h for h, _s, _d, _t in adapter.samples()}
        missing = [h for h in res["per_image"] if h not in known]
        if missing:
            raise ValueError(
                f"eval 結果引用了 {len(missing)} 個不在資料集裡的影像雜湊(例:{missing[0]});"
                "predictions 的 vix_hash 必須是內容雜湊(compute_hash),不是檔名 stem。"
            )
    for h, pi in res["per_image"].items():  # so review-queue / the App can sort by model failure
        try:
            adapter.attach_fields(h, {"eval_fp": pi["n_fp"], "eval_fn": pi["n_fn"]})
        except Exception:  # noqa: BLE001 - eval may reference hashes not in this adapter view
            if strict_join:
                raise
    cfg.eval_results_path.write_text(  # per_image stripped; fp_detail/fn_detail/eval_set_hash kept
        json.dumps({k: v for k, v in res.items() if k != "per_image"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    DecisionLog(cfg.decision_log_path).append(
        "eval_ingest", decision=str(res["mAP"]),
        extra={"mAP": res["mAP"], "per_class_ap": res["per_class_ap"],
               "loc_gap": res.get("loc_gap"), "eval_set_hash": res["eval_set_hash"],
               "n_fn_images": len(res["fn_hashes"])},
    )
    log.info("eval_ingest: mAP=%.3f, %d classes, %d FN / %d FP images",
             res["mAP"], len(res["per_class_ap"]), len(res["fn_hashes"]), len(res["fp_hashes"]))
    return res


def eval_run(adapter, cfg, weights, imgsz=640, conf=0.05, iou_thr=0.5,
             gt_tags=(Tag.PROVISIONAL, Tag.EVAL)):
    """Run the engineer's OWN model on the imported-GT images and feed predictions + GT into
    eval_ingest — the shipped `yolo val -> VIX` bridge (was trapped in dogfood_eval_yolo.py).

    GT comes from images tagged PROVISIONAL/EVAL (via import_labels). Predictions are keyed by
    the sample's CONTENT-HASH vix_hash (not filename stem), and eval_ingest runs with
    strict_join=True so a key mismatch fails loudly instead of attaching FP/FN to nothing."""
    try:
        from ultralytics import YOLO
    except Exception as exc:  # noqa: BLE001 - keep the verb importable without torch/ultralytics
        raise ValueError("eval-run 需要 ultralytics(pip install ultralytics);"
                         "或自行產生 predictions+GT 的 JSONL 後用 vix eval-ingest。") from exc
    model = YOLO(str(weights))
    want = set(gt_tags)
    images, n_pred = [], 0
    for h, src, dets, tags in adapter.samples():
        if not (want & set(tags)) or not dets:
            continue
        r = model.predict(src, imgsz=imgsz, conf=conf, verbose=False)[0]
        names = getattr(r, "names", {}) or {}
        preds = []
        for i in range(len(r.boxes)):
            cid = int(r.boxes.cls[i])
            preds.append({"label": names.get(cid, str(cid)),
                          "bbox": [round(float(v), 6) for v in r.boxes.xywhn[i].tolist()],
                          "conf": round(float(r.boxes.conf[i]), 4)})
        n_pred += len(preds)
        images.append({"vix_hash": h,  # content-hash key -> joins to the ingested image
                       "gt": [{"label": d.label, "bbox": [round(c, 6) for c in d.bbox.as_tuple()]}
                              for d in dets],
                       "pred": preds})
    if not images:
        raise ValueError("eval-run 找不到帶 GT 的影像;請先 vix import-labels / vix diagnose --labels 匯入標籤。")
    res = eval_ingest(adapter, cfg, images, iou_thr=iou_thr, strict_join=True)
    res["n_pred"] = n_pred
    log.info("eval_run: %d images, %d preds, mAP=%.3f", len(images), n_pred, res["mAP"])
    return res


def diagnose(adapter, cfg, folder, labels_fmt="auto", weights=None, audit=False, names=None,
             json_path=None, label_dir=None, out_path=None, batch="diagnose", iou_thr=0.5):
    """THE on-ramp: import the engineer's EXISTING labels + (optionally) run their model, and
    produce the weakness/attribution report — no golden/anchor/calibrate/route/gate worldview.

    Tier A (`weights`): typed per-class FP/FN + AP + confusion vs the imported labels — needs
    no DINOv2/FiftyOne. Tier B (`audit`): embedding label-audit + failure attribution (needs
    real embeddings). HONESTY: the imported labels are an UNVERIFIED reference (Tag.PROVISIONAL,
    never golden); the report is framed accordingly and the attribution firewall (no label_noise,
    no gate-block on an unverified reference) applies."""
    if not weights and not audit:
        raise ValueError("diagnose 需要 --weights(Tier A:模型弱點)或 --audit(Tier B:標籤稽核)其一")
    if labels_fmt in (None, "auto"):  # auto-detect so the caller can just give a folder
        from .core.label_import import detect_format
        det = detect_format(folder)
        labels_fmt = det["fmt"]
        json_path = json_path or det["json_path"]
        names = names or det["names"]
        if not labels_fmt:
            raise ValueError("無法自動判斷標籤格式:找不到 YOLO labels/(.txt)、VOC annotations/(.xml) 或 COCO json")
        if labels_fmt == "yolo" and names is None and weights:  # align numeric GT with the model's class names
            try:
                from ultralytics import YOLO
                names = YOLO(str(weights)).names
            except Exception:  # noqa: BLE001
                pass
    imp = import_labels(adapter, cfg, folder, labels_fmt, names=names, batch=batch,
                        as_="reference", json_path=json_path, label_dir=label_dir)
    if imp["n_boxes"] == 0:  # honesty guard: a wrong folder layout must not look like "all clean"
        hint = {"yolo": "找過 images/../labels/、sibling .txt、<root>/labels/;用 --label-dir 指定",
                "voc": "找過 annotations/、Annotations/;確認 xml 與影像 stem 對應",
                "coco": "用 --json 指定 instances.json,且其 file_name 對得上影像"}.get(labels_fmt, "")
        raise ValueError(
            f"找到 {imp['n_scanned']} 張影像但 0 個標籤配對成功({labels_fmt})。{hint}")
    out = {"import": imp, "tiers": []}
    if audit:  # Tier B: embed the imported-GT crops (real DINOv2 if the adapter has one)
        adapter.compute_embeddings(cfg.dinov2_model_key)
        out["tiers"].append("B")
    if weights:  # Tier A: run the user's model -> eval
        ev = eval_run(adapter, cfg, weights, iou_thr=iou_thr)
        out["eval"] = {"mAP": ev["mAP"], "loc_gap": ev.get("loc_gap"),
                       # weakest class first (matches the report table; raw evaluate() is alphabetical)
                       "per_class_ap": dict(sorted(ev["per_class_ap"].items(), key=lambda kv: kv[1]))}
        out["tiers"].append("A")
    wr = weakness_report(
        adapter, cfg, out_path=out_path, reference_unverified=True,
        consistency_tag=(Tag.PROVISIONAL if audit else Tag.GOLDEN),
        reference_trusted=False)  # imported labels are NOT human-confirmed -> firewall on
    out["summary"] = wr["data"].get("summary")
    out["report_md"] = wr.get("path")
    out["comparable"] = (wr["data"].get("provenance") or {}).get("comparable")  # for a state-aware next-step nudge
    return out


def _adapt_rescued(cfg):
    """{frozenset(pair): rescued} from a saved adapt-embedding report, else None."""
    if not cfg.adapt_report_path.exists():
        return None
    try:
        rep = json.loads(cfg.adapt_report_path.read_text(encoding="utf-8"))
        return {frozenset(p["pair"]): bool(p.get("rescued")) for p in rep.get("pairs", [])}
    except (ValueError, OSError):
        return None


def _active_projection(cfg):
    """Load the LDA projection IFF it exists, is gate-enabled (marker or env), and has dim>=2
    (1-d LDA breaks cosine ranking, so we only project multi-class spaces). Else None."""
    if not cfg.embed_projection_path.exists():
        return None
    if not (cfg.use_embed_projection or cfg.embed_projection_enabled_path.exists()):
        return None
    from .core.embed_adapt import load_projection
    proj = load_projection(cfg.embed_projection_path)
    if proj is None or proj.get("W") is None or np.asarray(proj["W"]).shape[1] < 2:
        return None
    return proj


def worklist_views(all_tags) -> dict:
    """FiftyOne saved-view specs for the weakness worklist (Tier 2: clickable worklist in the App).
    Each `vixq:*` tag (written by `weakness-report --worklist`) -> a named saved view, so the operator
    clears the worklist by clicking, not by hunting vix_hashes. Returns {view_name: tag}. Pure."""
    return {f"工作清單 {t[len('vixq:'):]}": t
            for t in sorted({x for x in all_tags if isinstance(x, str) and x.startswith("vixq:")})}


def _log_queue(cfg, queue, ids, predict):
    """Record a suggestion-queue emission so its hit-rate can be measured once ids get resolved.
    predict: 'wrong' (hardneg) | 'defect' (bank hard-positive) | 'label' (error-mine / weakness)."""
    ids = [i for i in (ids or [])]
    if not ids:
        return
    DecisionLog(cfg.decision_log_path).append(
        "queue_emit", decision=queue, extra={"queue": queue, "ids": ids, "predict": predict})


def queue_hit_rate(cfg, min_resolved=5):
    """Did VIX's suggestion queues turn out right? Joins logged `queue_emit` events with later human
    resolutions (review confirm/false_alarm, dismiss) -> per-queue precision / coverage / trend.
    Honest: only ids resolved AFTER emission count; only resolved ids score; insufficient flag."""
    from .core.queue_metrics import hit_rate
    recs = DecisionLog(cfg.decision_log_path).read_all()
    emissions, resolutions = [], []
    for seq, r in enumerate(recs):
        ev = r.get("event")
        if ev == "queue_emit":
            ex = r.get("extra", {})
            emissions.append({"queue": ex.get("queue"), "ids": ex.get("ids", []),
                              "predict": ex.get("predict", "label"), "seq": seq})
        elif ev == "review" and r.get("vix_hash"):
            outcome = "rejected" if r.get("decision") == "false_alarm" else "confirmed"
            resolutions.append({"id": r["vix_hash"], "outcome": outcome, "seq": seq})
        elif ev == "dismiss":
            for i in r.get("extra", {}).get("ids", []):
                resolutions.append({"id": i, "outcome": "rejected", "seq": seq})
    queues = hit_rate(emissions, resolutions, min_resolved=min_resolved)
    log.info("queue_hit_rate: %d emissions, %d resolutions, %d queues", len(emissions), len(resolutions), len(queues))
    return {"queues": queues, "n_emissions": len(emissions), "n_resolutions": len(resolutions)}


def _resolved_ids(cfg) -> set:
    """vix_hashes that already have a human resolution (review confirm/false_alarm, or dismiss).
    Used to mark worklist candidates done so a re-run doesn't ask the engineer to re-do them (L4 —
    NOT re-ranking: queue order is unchanged, resolved items are only flagged)."""
    out: set = set()
    for r in DecisionLog(cfg.decision_log_path).read_all():
        ev = r.get("event")
        if ev == "review" and r.get("vix_hash"):
            out.add(r["vix_hash"])
        elif ev == "dismiss":
            out.update(r.get("extra", {}).get("ids", []))
    return out


def _report_provenance(cfg, cur_eval_hash):
    """Provenance + eval-set comparability for a weakness report (L1/L3). Reads the hash-chained log
    for the previous eval_ingest's eval_set_hash/mAP and the previous weakness_report timestamp, so
    the report self-locates on its own trend and flags 'not comparable to last cycle' honestly
    (a +mAP on a silently-changed val set is the failure this prevents). Pure read, reuses the log."""
    recs = DecisionLog(cfg.decision_log_path).read_all()
    evs = [(r.get("extra", {}).get("eval_set_hash"), r.get("extra", {}).get("mAP"),
            r.get("extra", {}).get("per_class_ap"))
           for r in recs if r.get("event") == "eval_ingest"]
    prev_report_ts = next((r.get("ts_utc") for r in reversed(recs) if r.get("event") == "weakness_report"), None)
    prev_hash, prev_map, prev_pc = (evs[-2] if len(evs) >= 2 else (None, None, None))  # current = last eval_ingest
    comparable = None if (not cur_eval_hash or prev_hash is None) else (prev_hash == cur_eval_hash)
    return {"eval_set_hash": cur_eval_hash, "prev_report_ts": prev_report_ts, "prev_eval_set_hash": prev_hash,
            "comparable": comparable, "prev_mAP": (prev_map if comparable else None),
            # per-class AP of the previous COMPARABLE run -> "which class moved" (only when eval set is unchanged;
            # eval_set_hash includes GT, so a relabelled eval set is correctly NOT comparable)
            "prev_per_class_ap": (prev_pc if comparable else None)}


def report_trend(cfg, classes=None):
    """Per-class AP / mAP / health over time, read from the hash-chained decision log (Tier 2).
    Answers 'did my curation move this class's AP across rounds?' offline + auditable; AP deltas are
    flagged not-comparable if the eval set changed across the series."""
    from .core.trend import eval_trend
    return eval_trend(DecisionLog(cfg.decision_log_path).read_all(), classes)


def _match_box_emb(box, det_embs, thr):
    """Best stored-detection embedding whose box IoU>=thr with the (external) error box, else None.
    The eval JSON's pred/GT boxes carry no embedding and may come from a different run, so we
    match them back to a stored detection by geometry rather than assuming a 1:1 correspondence."""
    from .core.eval_ingest import iou
    best, best_iou = None, thr
    for d, e in det_embs:
        ov = iou(tuple(box), d.bbox.as_tuple())
        if ov >= best_iou:
            best_iou, best = ov, e
    return best


def error_mine(adapter, cfg, top=20, emb_match_iou=0.5, fn_match_iou=0.1, for_class=None, batch=None):
    """Rank unlabeled candidates by closeness to the model's val FP/FN error *regions*, so
    labeling effort lands on the model's demonstrated failures (not just novelty).

    Uses the typed fp_detail/fn_detail boxes (T1a): each error box is IoU-matched back to a
    stored detection's embedding (FP boxes precisely; FN boxes via an overlapping pred, down to
    the localization band). If an error image yields no matchable box (e.g. a pure `missed` FN,
    or external boxes that don't align), it falls back to that image's detection-mean embedding
    — strictly better than the old whole-error-set mean, and degrades cleanly on memory/pixel
    fallback adapters (model-loop-v2 R1).

    ``for_class`` (model-loop-v2 weakness-report): restrict the error regions to ONE class's
    boxes, so a weak class C yields "label these candidates nearest C's failures" rather than a
    class-blind global pool. In class mode the image-mean fallback is disabled (it would re-dilute
    class specificity); a class with too few matchable error boxes simply yields fewer candidates,
    which the caller can widen with coverage/uncertainty."""
    p = cfg.eval_results_path
    if not p.exists():
        raise ValueError("尚無評估結果;請先執行 vix eval-ingest <results.json>")
    ev = json.loads(p.read_text(encoding="utf-8"))
    fp_detail, fn_detail = ev.get("fp_detail", {}), ev.get("fn_detail", {})

    def _keep(box):
        return for_class is None or box.get("label") == for_class

    if for_class is None:
        error_hashes = set(ev.get("fn_hashes", [])) | set(ev.get("fp_hashes", []))
    else:  # only images that actually have an error box of this class
        error_hashes = {h for h, bs in fp_detail.items() if any(_keep(b) for b in bs)}
        error_hashes |= {h for h, bs in fn_detail.items() if any(_keep(b) for b in bs)}
    samples = {h: dets for h, _s, dets, _t in adapter.samples()}
    err_emb, n_box, n_fallback = [], 0, 0
    for h in error_hashes:
        det_embs = [(d, np.asarray(d.embedding, float)) for d in (samples.get(h) or []) if d.embedding is not None]
        matched = []
        for box in fp_detail.get(h, []):  # FP regions are predictions -> match tightly
            if _keep(box) and (e := _match_box_emb(box["bbox"], det_embs, emb_match_iou)) is not None:
                matched.append(e)
        for box in fn_detail.get(h, []):  # FN regions -> the overlapping pred (looser, localization band)
            if _keep(box) and (e := _match_box_emb(box["bbox"], det_embs, fn_match_iou)) is not None:
                matched.append(e)
        if matched:
            err_emb.extend(matched)
            n_box += len(matched)
        elif det_embs and for_class is None:  # global mode only: image-mean so the error image still counts
            err_emb.append(np.mean(np.vstack([e for _d, e in det_embs]), axis=0))
            n_fallback += 1
    if not err_emb:
        return []
    from .core.embed_adapt import transform as _proj_tx
    proj = _active_projection(cfg)  # domain-adapted embedding (gate-enabled, dim>=2) -> sharper ranking
    EM = np.vstack(err_emb)
    E = _l2norm(_proj_tx(proj, EM) if proj else EM)
    cands = _image_items(adapter, want_tags=([f"batch:{batch}"] if batch else None),  # batch scope: "label from THIS batch"
                         exclude_tags=[Tag.GOLDEN, Tag.ANCHOR, Tag.EVAL, Tag.REJECTED])
    tag = ",已套用 domain-adapted 投影" if proj else ""
    ranked = []
    for it in cands:
        raw = np.asarray(it.embedding, float)[None, :]
        v = _l2norm(_proj_tx(proj, raw) if proj else raw)[0]
        sim = float((E @ v).max())  # cosine to the nearest error region (in projected space if enabled)
        ranked.append({"id": it.id, "closeness": round(sim, 4),
                       "why": f"接近模型驗證集誤差區(cos {sim:.3f}{tag});標此張最可能補到模型實際失敗處"})
    ranked.sort(key=lambda r: -r["closeness"])
    log.info("error_mine: %d candidates vs %d error regions (%d box-matched, %d fallback, projection=%s)",
             len(cands), len(err_emb), n_box, n_fallback, bool(proj))
    return ranked[:top]


def set_eval_baseline(adapter, cfg, protected=None, map_drop_thr=0.02):  # challenge-guard baseline (T2)
    """Freeze the current eval_results.json as the regression baseline: mAP + per-class AP +
    the eval_set_hash. A later data change that drops overall mAP, or a *protected* class's AP,
    is then hard-blocked by `pre_train_gate`. ``protected`` = {class: max_allowed_AP_drop}."""
    if not cfg.eval_results_path.exists():
        raise ValueError("尚無評估結果;請先執行 vix eval-ingest <results.json>")
    cur = json.loads(cfg.eval_results_path.read_text(encoding="utf-8"))
    baseline = {
        "mAP": cur.get("mAP"),
        "per_class_ap": cur.get("per_class_ap", {}),
        "n_gt": cur.get("n_gt", {}),
        "eval_set_hash": cur.get("eval_set_hash"),
        "map_drop_thr": map_drop_thr,
        "protected": protected or {},
    }
    cfg.eval_baseline_path.write_text(json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8")
    DecisionLog(cfg.decision_log_path).append(
        "set_eval_baseline", decision=str(cur.get("mAP")),
        extra={"eval_set_hash": cur.get("eval_set_hash"), "mAP": cur.get("mAP"), "protected": protected or {}},
    )
    log.info("set_eval_baseline: mAP=%s frozen as challenge-guard baseline (protected=%s)",
             cur.get("mAP"), list((protected or {}).keys()))
    return baseline


def box_qa(adapter, cfg, top=50, min_support=8):  # T1c per-box geometry QA (read-only)
    """Static per-box quality audit of golden boxes (degenerate / edge-truncated / area &
    aspect outliers vs each class's own envelope). Read-only: returns a ranked issue list,
    writes no tags and no ledger entry (same posture as `harmful` without --remove)."""
    from .core.box_qa import audit_boxes

    records = [
        {"id": h, "label": d.label, "bbox": d.bbox.as_tuple()}
        for h, _s, dets, tags in adapter.samples() if Tag.GOLDEN in tags
        for d in dets
    ]
    issues = audit_boxes(records, min_support=min_support)
    log.info("box_qa: %d golden boxes audited, %d issues", len(records), len(issues))
    return issues[:top]


def box_tightness(adapter, cfg, model="mobile_sam.pt", limit=60, iou_thr=0.6, top=50):  # opt-in, needs SAM
    """Pixel-level GT box-tightness audit via a SAM mask — the one check box_qa structurally can't do
    (box_qa is geometry-only; loc_gap measures the MODEL's looseness, not the GT's). For each golden box
    (sampled to `limit` for cost), prompt SAM with the box, take the object mask's tight box, and flag
    boxes whose GT disagrees with the mask (IoU < iou_thr → loose/misaligned annotation). Opt-in: needs
    `ultralytics` + a one-time SAM weights download; CPU ~1s/box. PROXY (the mask is a model's guess) —
    returns a ranked suspect list, writes no tags/ledger, never auto-edits."""
    import random

    from PIL import Image

    from .core.box_tightness import tightness
    try:
        from ultralytics import SAM
    except ImportError as e:  # noqa: F841
        raise ValueError("box-tightness 需要 ultralytics(含 SAM):pip install ultralytics")
    sam = SAM(model)
    golden = [(h, src, dets) for h, src, dets, t in adapter.samples() if Tag.GOLDEN in t]
    if limit and len(golden) > limit:
        golden = random.Random(0).sample(golden, limit)
    out = []
    for h, src, dets in golden:
        try:
            W, H = Image.open(src).size
        except Exception:  # noqa: BLE001 - unreadable image -> skip
            continue
        gtb = [d.bbox.as_tuple() for d in dets]
        boxes_px = [[(cx - w / 2) * W, (cy - hh / 2) * H, (cx + w / 2) * W, (cy + hh / 2) * H]
                    for (cx, cy, w, hh) in gtb]
        if not boxes_px:
            continue
        res = sam(src, bboxes=boxes_px, verbose=False)[0]
        if res.masks is None:
            continue
        for i, (d, gb) in enumerate(zip(dets, gtb)):
            if i >= len(res.masks.data):
                break
            mk = res.masks.data[i].cpu().numpy().astype(bool)
            ys, xs = np.where(mk)
            if len(xs) == 0:
                continue
            mb = (float((xs.min() + xs.max()) / 2 / W), float((ys.min() + ys.max()) / 2 / H),
                  float((xs.max() - xs.min()) / W), float((ys.max() - ys.min()) / H))
            t = tightness(gb, mb, iou_thr)
            if t["loose"]:
                out.append({"id": h, "label": d.label, **t})
    out.sort(key=lambda r: r["iou"])
    log.info("box_tightness: %d golden images sampled, %d loose boxes (SAM=%s)", len(golden), len(out), model)
    return out[:top]


def hardneg(adapter, cfg, top=50, mode="auto", batch=None):
    """Confidently-wrong mining — the "YOLO most confident yet wrong" weakness lens (ported from SAFE).
    mode 'auto': GT (confirmed eval-FPs ranked by conf) if eval_results.json has conf-bearing fp_detail,
    else GT-free (high-conf detections the embedding overturns). 'gt'/'gt_free' force a mode.
    Returns {mode, rows}. Everything is offline (no training/inference) — wrongness is a PROXY."""
    from .core.hardneg import rank_eval_fps, rank_overturns

    if mode in ("auto", "gt") and cfg.eval_results_path.exists():
        ev = json.loads(cfg.eval_results_path.read_text(encoding="utf-8"))
        fp_detail = ev.get("fp_detail", {})
        if mode == "gt" or any("conf" in b for boxes in fp_detail.values() for b in boxes):
            rows = rank_eval_fps(fp_detail, top)
            log.info("hardneg(gt): %d confidently-wrong eval FPs", len(rows))
            return {"mode": "gt", "rows": rows}
    if mode == "gt":
        raise ValueError("GT 模式需要含 conf 的 eval_results.json;請先 vix eval-ingest")
    if not cfg.thresholds_path.exists():
        raise ValueError("GT-free hardneg 需已校準 thresholds(先 vix calibrate),或先 eval-ingest 走 GT 模式")
    policy = ThresholdPolicy.load(cfg.thresholds_path)
    scorer = OutlierScorer(_emb_by_class(adapter, {Tag.GOLDEN}), k=cfg.knn_k)
    dets = []
    for h, _s, ds, tags in adapter.samples():
        if set(tags) & {Tag.GOLDEN, Tag.ANCHOR, Tag.EVAL, Tag.REJECTED}:
            continue  # only unlabeled / incoming detections are candidates for "confidently wrong"
        if batch and f"batch:{batch}" not in tags:  # batch scope
            continue
        for d in ds:
            ct = policy.thresholds.get(d.label)
            if d.embedding is None or ct is None:
                continue
            kd, _low = scorer.score_detection(d.embedding, d.label)
            dets.append({"id": h, "pred_class": d.label, "conf": d.confidence,
                         "knn_dist": kd, "conf_thr": ct.conf_thr, "dist_thr": ct.dist_thr})
    rows = rank_overturns(dets, top)
    log.info("hardneg(gt_free): %d confident embedding-overturns from %d detections", len(rows), len(dets))
    return {"mode": "gt_free", "rows": rows}


def weakness_report(adapter, cfg, top_classes=5, queue_per_class=10, out_path=None, worklist=False, batch=None,
                    reference_unverified=False, consistency_tag=Tag.GOLDEN, reference_trusted=True):
    """Roll VIX's model-validated signals + hardneg + a per-weak-class label queue into ONE
    human-readable 'where YOLO is weak / go label these' Markdown report (model-loop-v2). Two-mode:
    GT block (per-class AP/confusion/loc_gap/FP-FN typing + confidently-wrong eval-FPs) when a val
    set was ingested; GT-free block (embedding overturns) when thresholds exist. Writes a .md, logs
    a proxy-stamped audit entry. Every 'go label these' ranking is a PROXY (no retraining)."""
    from collections import Counter

    from .core.weakness_report import (
        render_weakness_report, render_weakness_report_html, render_weakness_report_panel)

    data = {"mode": "gt_free", "mAP": None, "loc_gap": None, "per_class": [], "confusion": [],
            "confident_wrong": [], "overturns": [], "queue": {}, "consistency": [], "hit_rate": []}
    ev = None
    if cfg.eval_results_path.exists():
        ev = json.loads(cfg.eval_results_path.read_text(encoding="utf-8"))
        data["mode"] = "gt"
        data["mAP"], data["loc_gap"] = ev.get("mAP"), ev.get("loc_gap")
        data["map_by_iou"] = ev.get("map_by_iou")  # so the renderer can tell "0.0 (evaluated)" from "N/A"
        per_class_ap, n_gt = ev.get("per_class_ap", {}), ev.get("n_gt", {})
        confusion, fn_detail = ev.get("confusion", {}), ev.get("fn_detail", {})
        fn_types: dict[str, Counter] = {}  # dominant FN failure mode per class
        for boxes in fn_detail.values():
            for b in boxes:
                fn_types.setdefault(b["label"], Counter())[b["type"]] += 1
        partner: dict[str, tuple] = {}  # top confusion partner per truth class
        for pair, n in confusion.items():
            truth, pred = pair.split("->", 1)
            if n > partner.get(truth, ("", 0))[1]:
                partner[truth] = (pred, n)
        rows = []
        for c, ap in per_class_ap.items():
            dom = fn_types.get(c)
            cp = partner.get(c)
            rows.append({"cls": c, "ap": ap, "n_gt": n_gt.get(c, 0),
                         "dom_fn_type": (dom.most_common(1)[0][0] if dom else None),
                         "fn_types": (dict(dom) if dom else {}),  # full breakdown, not just the top one
                         "top_confusion": (f"{cp[0]} ({cp[1]})" if cp else None)})
        rows.sort(key=lambda r: r["ap"])  # weakest first
        data["per_class"] = rows
        data["confusion"] = list(confusion.items())[:10]
        data["confident_wrong"] = hardneg(adapter, cfg, top=15, mode="gt")["rows"]
        for r in rows[:top_classes]:  # per-weak-class label queue (class-aware error-mine)
            if r["ap"] >= 1.0:
                continue
            try:
                cands = error_mine(adapter, cfg, top=queue_per_class, for_class=r["cls"], batch=batch)
            except (ValueError, OSError):
                cands = []
            if cands:
                data["queue"][r["cls"]] = [{"id": x["id"], "closeness": x["closeness"]} for x in cands]
    data["batch"] = batch
    try:  # GT-free overturns: best-effort (needs calibrated thresholds + unlabeled detections)
        data["overturns"] = hardneg(adapter, cfg, top=15, mode="gt_free", batch=batch)["rows"]
    except (ValueError, OSError):
        pass

    # Attach a human-readable filename to each navigable row: the report otherwise shows a 64-char
    # vix_hash (unreadable AND un-clickable). The hash stays as `id` so the App panel can jump to the
    # image; `file` is display-only. Built once; never fail the report over a missing filename.
    hash_to_file: dict[str, str] = {}
    try:
        for h, fp, *_ in adapter.samples():
            hash_to_file[h] = Path(fp).name
    except Exception:  # noqa: BLE001 - filename is display sugar
        pass
    for r in data["confident_wrong"] + data["overturns"]:
        r["file"] = hash_to_file.get(r["id"], "")

    from .core.consistency import consistency_findings  # GT x embedding attribution (taxonomy/model/label)
    data["reference_unverified"] = reference_unverified  # honesty F1: imported labels are not human-confirmed
    data["consistency"] = consistency_findings(
        _emb_by_class(adapter, {consistency_tag}),
        (ev.get("confusion") if ev else None), (ev.get("n_gt") if ev else None),
        adapt_rescued=_adapt_rescued(cfg), reference_trusted=reference_trusted)

    # queue hit-rate: log THIS report's label queue, then surface every queue's measured precision/trend
    _log_queue(cfg, "weakness_queue", [c["id"] for cands in data["queue"].values() for c in cands], "label")
    data["hit_rate"] = queue_hit_rate(cfg)["queues"]

    resolved = _resolved_ids(cfg)  # L4: flag candidates a human already actioned (re-runs shouldn't re-ask)
    for cands in data["queue"].values():
        for c in cands:
            c["resolved"] = c["id"] in resolved
    data["provenance"] = _report_provenance(cfg, (ev or {}).get("eval_set_hash"))  # L1/L3
    prev_pc = data["provenance"].get("prev_per_class_ap")  # in-report before/after (Round 5): "which class moved"
    if prev_pc:  # only set when the eval set is unchanged (comparable) -> honest same-eval-set delta
        for r in data["per_class"]:
            if r["cls"] in prev_pc and prev_pc[r["cls"]] is not None:
                r["prev_ap"] = prev_pc[r["cls"]]
                r["delta_ap"] = round(r["ap"] - prev_pc[r["cls"]], 4)

    # TL;DR health verdict + "do this now" (Tier 1: scannability)
    worst = data["per_class"][0] if data["per_class"] else None
    bad_consist = [f for f in data["consistency"]
                   if f["verdict"] in ("taxonomy", "label_noise") and not f.get("representation_fixable")
                   and f["tier"] == "supported"]
    rep_fix = [f for f in data["consistency"] if f.get("representation_fixable")]
    audit_needed = [f for f in data["consistency"] if f["verdict"] == "label_audit_needed"]  # firewall (F2)
    n_open = sum(1 for v in data["queue"].values() for c in v if not c.get("resolved"))  # L4: open != emitted
    n_cw = len(data["confident_wrong"])
    if (worst and worst["ap"] < 0.5) or bad_consist:
        health = "RED"
    elif (worst and worst["ap"] < 0.8) or n_cw or audit_needed or any(
            f["verdict"] in ("taxonomy_watch", "model_watch") for f in data["consistency"]):
        health = "AMBER"
    else:
        health = "GREEN"
    todo = []
    if bad_consist:
        todo.append(f"重新檢視 {len(bad_consist)} 個類別對定義:" + ", ".join("↔".join(f["pair"]) for f in bad_consist))
    if rep_fix:  # A2: a representation problem -> adapt-embedding is the lever; labeling is the WRONG first move
        todo.append("套用 adapt-embedding:已 CV 驗證可分開 "
                    + ", ".join("↔".join(f["pair"]) for f in rep_fix) + "(表徵問題,標註不是這裡的槓桿)")
    if audit_needed:  # F2: unverified reference -> human-confirm before any label-noise claim
        todo.append("人工覆核這些匯入標籤(嵌入難分,確認後才能升級 golden):"
                    + ", ".join("↔".join(f["pair"]) for f in audit_needed))
    if n_open:
        todo.append(f"標 {n_open} 個候選(見佇列)")
    if n_cw:
        todo.append(f"覆核 {n_cw} 個自信誤報")
    data["summary"] = {"health": health, "todo": todo,
                       "weakest": (f"{worst['cls']} AP={worst['ap']} (n_gt={worst['n_gt']})" if worst else None)}

    out = Path(out_path or (cfg.workspace / "weakness_report.md"))

    # worklist export (Tier 1: turn the report into a clearable list, not just a read)
    import csv as _csv
    wl_rows = [{"queue": f"label:{cls}", "class": cls, "vix_hash": c["id"], "reason": "近該類失敗處"}
               for cls, cands in data["queue"].items() for c in cands]
    wl_rows += [{"queue": "confident_wrong", "class": cw.get("pred_class"), "vix_hash": cw["id"],
                 "reason": f"自信({cw['conf']}){cw.get('fp_type', '')} 誤報"} for cw in data["confident_wrong"]]
    wl_path = out.with_name("weakness_worklist.csv")
    with open(wl_path, "w", newline="", encoding="utf-8") as fc:
        w = _csv.DictWriter(fc, fieldnames=["queue", "class", "vix_hash", "reason"])
        w.writeheader()
        w.writerows(wl_rows)
    data["worklist_csv"] = str(wl_path)
    if worklist:  # opt-in: tag the worklist samples so the FiftyOne App can filter/build saved views
        for cls, cands in data["queue"].items():
            for c in cands:
                try:
                    adapter.apply_tags(c["id"], [f"vixq:label:{cls}"])
                except Exception:  # noqa: BLE001 - id may not be a live sample (e.g. external eval hash)
                    pass
        for cw in data["confident_wrong"]:
            try:
                adapter.apply_tags(cw["id"], ["vixq:confident_wrong"])
            except Exception:  # noqa: BLE001
                pass

    out.write_text(render_weakness_report(data), encoding="utf-8")
    html_out = out.with_suffix(".html")  # browsable surface (Playwright-verifiable)
    html_out.write_text(render_weakness_report_html(data), encoding="utf-8")
    out.with_name("weakness_report_panel.md").write_text(  # compact in-App panel layout
        render_weakness_report_panel(data), encoding="utf-8")
    # sidecar for the App panel's CLICKABLE tables: filename to read + hash to navigate to the image.
    panel_nav = {
        "confident_wrong": [{"file": r.get("file") or r["id"][:12], "hash": r["id"],
                             "pred_class": r.get("pred_class"), "conf": r.get("conf"),
                             "fp_type": r.get("fp_type") or "-"} for r in data["confident_wrong"]],
        "overturns": [{"file": r.get("file") or r["id"][:12], "hash": r["id"],
                       "pred_class": r.get("pred_class"), "conf": r.get("conf"),
                       "wrongness": round(r["wrongness"], 2)} for r in data["overturns"]],
    }
    out.with_name("weakness_report_panel.json").write_text(
        json.dumps(panel_nav, ensure_ascii=False), encoding="utf-8")
    DecisionLog(cfg.decision_log_path).append(
        "weakness_report", decision=data["summary"]["health"],
        extra={"mAP": data["mAP"], "health": data["summary"]["health"], "batch": batch,
               "eval_set_hash": data["provenance"].get("eval_set_hash"),       # L1: self-locating on the trend
               "prev_report_ts": data["provenance"].get("prev_report_ts"),
               "eval_comparable": data["provenance"].get("comparable"),         # L3: vs last cycle
               "weak_classes": [r["cls"] for r in data["per_class"][:top_classes]],
               "consistency": [f["verdict"] for f in data["consistency"]][:10],
               "worklist_n": len(wl_rows), "tagged": bool(worklist),
               "note": "proxy: no retraining; rankings are suspicion/priority, not measured mAP gain"})
    log.info("weakness_report: %s, mode=%s -> %s (+%s, worklist=%d)",
             data["summary"]["health"], data["mode"], out, html_out.name, len(wl_rows))
    return {"path": str(out), "html": str(html_out), "worklist_csv": str(wl_path), "data": data}


def consistency(adapter, cfg, max_pairs=20):
    """GT-powered consistency attribution: per class-pair separability (LOO-kNN in DINOv2 space) +
    confusion×embedding-overlap 2×2 verdict (taxonomy / model / label_noise), over the human-confirmed
    GOLDEN embeddings joined with the eval confusion matrix if present. Advisory, support-gated with
    CIs; separability is encoder-hedged ('in the current embedding space'). Offline; no training."""
    from .core.consistency import consistency_findings
    emb_by_class = _emb_by_class(adapter, {Tag.GOLDEN})
    confusion = n_gt = None
    if cfg.eval_results_path.exists():
        ev = json.loads(cfg.eval_results_path.read_text(encoding="utf-8"))
        confusion, n_gt = ev.get("confusion", {}), ev.get("n_gt", {})
    findings = consistency_findings(emb_by_class, confusion, n_gt, max_pairs=max_pairs,
                                    adapt_rescued=_adapt_rescued(cfg))
    DecisionLog(cfg.decision_log_path).append(
        "consistency", decision=str(len(findings)),
        extra={"n_classes": len(emb_by_class), "has_eval": confusion is not None,
               "verdicts": [f["verdict"] for f in findings][:10],
               "note": "advisory; separability encoder-hedged; small GT -> insufficient_support; never auto-merge"})
    log.info("consistency: %d classes, %d findings (has_eval=%s)", len(emb_by_class), len(findings), confusion is not None)
    return {"findings": findings, "n_classes": len(emb_by_class), "has_eval": confusion is not None}


def adapt_embedding(adapter, cfg, save=False, enable=False, max_pca=64, folds=5, min_gain=0.0, tol=0.02):
    """Learn a lightweight supervised projection of frozen DINOv2 (regularized LDA on golden GT) and
    report, per class-pair, whether a pair that's inseparable in frozen DINO becomes SEPARABLE after
    adaptation (a 'rescue' => representation problem, not a taxonomy dead-end). Before/after
    separability is k-fold CROSS-VALIDATED (refit on train folds only) to stay honest on small GT.
    A gate decides whether enabling the projection across the stack is safe (macro separability up,
    no per-pair regression). ``save`` persists embed_projection.npz + adapt_report.json; ``enable``
    writes the gate-validated enable marker ONLY if the gate says GO. Offline; NOT YOLO training."""
    from .core.embed_adapt import (cv_pair_separability, fit_projection, projection_gate, save_projection)

    emb_by_class = {c: np.atleast_2d(np.asarray(e, float))
                    for c, e in _emb_by_class(adapter, {Tag.GOLDEN}).items()}
    classes = sorted(c for c, e in emb_by_class.items() if e.shape[0] >= 2)
    if len(classes) < 2:
        raise ValueError("需要 ≥2 類、每類 ≥2 筆 golden 才能學投影")
    X = np.vstack([emb_by_class[c] for c in classes])
    y = np.concatenate([[c] * emb_by_class[c].shape[0] for c in classes])
    proj = fit_projection(X, y, max_pca=max_pca)
    out_dim = int(proj["W"].shape[1]) if proj.get("W") is not None else 0

    pairs = []
    for a in range(len(classes)):
        for b in range(a + 1, len(classes)):
            ci, cj = classes[a], classes[b]
            fr, ad, n_min = cv_pair_separability(emb_by_class[ci], emb_by_class[cj], folds=folds, max_pca=max_pca)
            pairs.append({"pair": [ci, cj], "frozen_sep_err": fr, "adapted_sep_err": ad, "n_min": n_min,
                          "rescued": bool(fr > 0.35 and ad <= 0.35), "delta": round(fr - ad, 4)})
    pairs.sort(key=lambda p: -p["delta"])
    n_rescued = sum(p["rescued"] for p in pairs)
    go, reasons, summary = projection_gate(pairs, min_gain=min_gain, tol=tol)

    saved = enabled = False
    if save:
        save_projection(cfg.embed_projection_path, proj)
        cfg.adapt_report_path.write_text(json.dumps(
            {"out_dim": out_dim, "pairs": pairs, "gate": {"go": go, "reasons": reasons, **summary}},
            ensure_ascii=False, indent=2), encoding="utf-8")
        saved = True
    if enable:  # gate-validated: only flip the switch if enabling is safe
        if go and cfg.embed_projection_path.exists():
            cfg.embed_projection_enabled_path.write_text("gate=GO\n", encoding="utf-8")
            enabled = True
        else:
            log.warning("adapt_embedding: --enable 但 gate=%s(%s)→ 不啟用(投影仍可作診斷)", "GO" if go else "NO-GO", reasons)
    DecisionLog(cfg.decision_log_path).append(
        "adapt_embedding", decision=("GO" if go else "NO-GO"),
        extra={"out_dim": out_dim, "n_pairs": len(pairs), "n_rescued": n_rescued, "saved": saved,
               "enabled": enabled, "gate": {"go": go, "reasons": reasons, **summary},
               "note": "supervised LDA projection of frozen DINOv2 from golden GT; CV'd; offline, no YOLO training"})
    log.info("adapt_embedding: %d classes -> %dd, %d/%d rescued, gate=%s, saved=%s, enabled=%s",
             len(classes), out_dim, n_rescued, len(pairs), "GO" if go else "NO-GO", saved, enabled)
    return {"classes": classes, "out_dim": out_dim, "pairs": pairs, "n_rescued": n_rescued,
            "gate": {"go": go, "reasons": reasons, **summary}, "saved": saved, "enabled": enabled}


def bank_audit(adapter, cfg, defect_tag=Tag.GOLDEN, reflection_tag=Tag.REJECTED, normal_tag=None,
               conf_lo=0.05, conf_hi=0.25, tau=0.10, k=None, novelty_radius=0.30,
               dedup_distance=0.05, top=50):
    """Multi-bank Top-K embedding audit of LOW-CONFIDENCE proposals (design of record:
    docs/discussion/bank-audit-design.md). Banks are built from tags (defect=golden,
    reflection=rejected, optional normal); each conf-band detection is voted across the
    calibrated banks -> defect_like/reflection_like/normal_like/unknown. Verdict is an
    ADVISORY field (never overrides routing); defect_like/unknown are staged as
    hard_positive (human-confirmed -> golden, never auto-promoted)."""
    import hashlib
    from collections import Counter

    from .core.bank_audit import bank_vote, build_bank_scales, loose_nms
    from .embedding.dinov2 import MODEL_KEY

    k = k or cfg.knn_k
    # 1. build banks from tags (pooled detection-crop embeddings)
    banks: dict[str, np.ndarray] = {}
    bank_label_map: dict[str, str] = {}
    specs = [(defect_tag, "defect_like"), (reflection_tag, "reflection_like")]
    if normal_tag:
        specs.append((normal_tag, "normal_like"))
        if normal_tag == Tag.PASS:
            log.warning("bank-audit: normal 銀行用 raw 'pass'(未驗證);建議用已驗證 normal tag")
    else:
        log.warning("bank-audit: 無 normal 銀行;退化為 defect/reflection 二銀行 + novelty radius")
    for tag, verdict in specs:
        emb = _emb_by_class(adapter, {tag})
        pooled = np.vstack(list(emb.values())) if emb else np.zeros((0, 1))
        if pooled.shape[0] == 0:
            log.warning("bank-audit: 銀行 '%s' 為空,略過", tag)
            continue
        banks[tag] = pooled
        bank_label_map[tag] = verdict
    if defect_tag not in banks:
        raise ValueError("bank-audit: defect 銀行(預設 golden)為空,無法審查")

    # 2. bank hygiene: warn on near-duplicate density that would bias the vote
    for tag in banks:
        dups = near_duplicate_groups(_image_items(adapter, want_tags=[tag]), dedup_distance)
        if dups:
            log.warning("bank-audit: 銀行 '%s' 有 %d 群近重複,會偏壓投票密度;建議先 dedup + audit-labels", tag, len(dups))

    # 3. per-bank calibration scales (build-time, once)
    scales = build_bank_scales(banks, k)
    bank_fp = hashlib.sha256(
        ("|".join(f"{t}:{banks[t].shape[0]}" for t in sorted(banks)) + MODEL_KEY).encode()
    ).hexdigest()[:12]

    # 4. collect low-conf proposals (conf band, non-reference samples), de-dup per image
    ref = {Tag.GOLDEN, Tag.ANCHOR, Tag.EVAL, Tag.REJECTED}
    results = []
    for h, _s, dets, tags in adapter.samples():
        if ref & set(tags):
            continue
        band = [d for d in dets if d.embedding is not None and conf_lo <= d.confidence < conf_hi]
        if not band:
            continue
        adapter.apply_tags(h, [Tag.PROPOSAL])  # isolate from golden routing/KPIs
        best = None
        sample_verdicts = []
        for d in loose_nms(band):
            v = bank_vote(np.asarray(d.embedding, float), banks, scales, bank_label_map, k, tau, novelty_radius)
            row = {"id": h, "conf": round(d.confidence, 3), "verdict": v.verdict,
                   "winning_bank": v.winning_bank, "margin": v.margin, "min_raw_dist": v.min_raw_dist,
                   "per_bank": v.per_bank, "topk_evidence": v.topk_evidence}
            results.append(row)
            if best is None:
                best = v  # representative (first = highest-conf after NMS) drives the advisory field
            sample_verdicts.append(v.verdict)
        if best is not None:
            adapter.attach_fields(h, {"bank_verdict": best.verdict,
                                      "bank_evidence": {"winning_bank": best.winning_bank,
                                                        "margin": best.margin, "min_raw_dist": best.min_raw_dist}})
            # stage if ANY proposal on the image is defect-like/novel (not only the representative box)
            if any(sv in ("defect_like", "unknown") for sv in sample_verdicts):
                adapter.apply_tags(h, [Tag.HARD_POSITIVE])

    rank = {"defect_like": 0, "unknown": 1, "reflection_like": 2, "normal_like": 3}
    results.sort(key=lambda r: (rank.get(r["verdict"], 9), -r["margin"]))
    counts = dict(Counter(r["verdict"] for r in results))
    DecisionLog(cfg.decision_log_path).append(
        "bank_audit", decision=str(len(results)),
        extra={"counts": counts, "bank_fingerprint": bank_fp,
               "banks": {t: int(banks[t].shape[0]) for t in banks},
               "embedding_backend": cfg.embedding_backend},
    )
    log.info("bank_audit: %d proposals -> %s (banks=%s fp=%s)",
             len(results), counts, {t: banks[t].shape[0] for t in banks}, bank_fp)
    return {"counts": counts, "n_proposals": len(results), "fingerprint": bank_fp,
            "banks": {t: int(banks[t].shape[0]) for t in banks},
            "results": results[:top] if top else results}


# --- round-4 additions (V1–V10) ------------------------------------------

def run_pipeline(adapter, cfg, input_folder=None, batch_id="run", weights=None,
                 classes=None, export_dst=None, ingest_tags=None):  # V1/X2 one-stop, fail-fast
    summary: dict = {"steps": []}

    def step(name, fn):
        try:
            r = fn()
            summary["steps"].append({"step": name, "ok": True})
            return r
        except Exception as exc:  # noqa: BLE001 - fail fast, record, abort
            summary["steps"].append({"step": name, "ok": False, "error": str(exc)})
            log.error("run: step '%s' failed: %s", name, exc)
            raise

    if input_folder:
        step("ingest", lambda: ingest(adapter, cfg, input_folder, batch_id, tags=ingest_tags or []))
    if weights:
        from .detect import run_yolo

        step("infer", lambda: run_yolo(adapter, cfg, weights))
    step("embed", lambda: adapter.compute_embeddings(cfg.dinov2_model_key))
    step("calibrate", lambda: calibrate(adapter, cfg))
    summary["route"] = step("route", lambda: route(adapter, cfg))
    summary["n_duplicate_groups"] = len(step("dedup", lambda: dedup(adapter, cfg)))
    summary["n_label_errors"] = len(step("audit_labels", lambda: audit_labels(adapter, cfg)))
    summary["n_leakage"] = len(step("leakage", lambda: leakage(adapter, cfg)))
    # auto-build the frozen reference if anchors exist but none is saved yet
    if _emb_by_class(adapter, {Tag.ANCHOR}) and not cfg.anchor_ref_path.exists():
        step("build_reference", lambda: build_reference(adapter, cfg))
    summary["gate"] = step("gate", lambda: pre_train_gate_stage(adapter, cfg)).verdict
    rep, _paths = step("report", lambda: health_report(adapter, cfg, cfg.workspace / "report"))
    summary["quality_score"] = rep["quality_score"]
    if export_dst:
        summary["export"] = step(
            "export",
            lambda: export(adapter, cfg, classes or sorted(rep["class_distribution"]), export_dst),
        )
    summary["audit_verified"] = DecisionLog(cfg.decision_log_path).verify_chain()
    DecisionLog(cfg.decision_log_path).append(
        "run", decision=summary["gate"],
        extra={"quality_score": summary["quality_score"], "audit_verified": summary["audit_verified"]},
    )
    log.info("run: complete gate=%s quality=%s audit_ok=%s",
             summary["gate"], summary["quality_score"], summary["audit_verified"])
    return summary


def history(cfg, vix_hash):  # V3 per-image submission history
    recs = [r for r in DecisionLog(cfg.decision_log_path).read_all() if r.get("vix_hash") == vix_hash]
    log.info("history: %d events for %s", len(recs), vix_hash)
    return recs


def routing_diff(cfg):  # V4 before/after routing diff
    prev = cfg.workspace / "routing_prev.json"
    cur = cfg.workspace / "routing_current.json"
    if not (prev.exists() and cur.exists()):
        return {"changed": [], "note": "need two routing runs to diff"}
    p = json.loads(prev.read_text(encoding="utf-8"))
    c = json.loads(cur.read_text(encoding="utf-8"))
    changed = [{"id": k, "from": p[k], "to": c[k]} for k in c if k in p and p[k] != c[k]]
    added = [k for k in c if k not in p]      # AI1: also surface samples that appeared/disappeared
    removed = [k for k in p if k not in c]
    log.info("routing_diff: %d changed, %d added, %d removed", len(changed), len(added), len(removed))
    return {"changed": changed, "n_changed": len(changed), "added": added, "removed": removed}


def _require_known(adapter, ids):  # B1: fail-closed — never act/log on an id the dataset lacks
    """Raise ValueError if any id is not a known vix_hash. The App shows FiftyOne *sample ids*
    (not vix_hash); a wrong/copied id must error loudly here, never silently no-op while still
    printing success AND writing a phantom record into the immutable decision log."""
    known = {h for h, *_ in adapter.samples()}
    missing = [i for i in ids if i not in known]
    if missing:
        raise ValueError(
            f"找不到 vix_hash {missing};App 顯示的是 sample id,"
            "請改用 vix_hash(見 vix history / vix explain),或用檔名"
        )


def dismiss(adapter, cfg, ids):  # V6 mark false alarms; excluded from future review queue
    _require_known(adapter, ids)  # B1: validate the whole batch before any tag/audit write
    for h in ids:
        adapter.apply_tags(h, [Tag.REJECTED])
    DecisionLog(cfg.decision_log_path).append("dismiss", decision=str(len(ids)), extra={"ids": list(ids)})
    log.info("dismiss: %d samples marked as false alarm", len(ids))
    return len(ids)


def restore_dismissed(adapter, cfg, ids):  # AL2/AL9 reverse a dismiss/harmful-remove (un-reject), audited
    _require_known(adapter, ids)  # B1
    try:
        for h in ids:
            adapter.remove_tags(h, [Tag.REJECTED])
    except NotImplementedError:  # surface as a clean user error, not a raw traceback past cli.py
        raise ValueError("此 adapter 不支援移除 tag,無法復原 dismiss") from None
    DecisionLog(cfg.decision_log_path).append("undismiss", decision=str(len(ids)), extra={"ids": list(ids)})
    log.info("restore_dismissed: un-rejected %d samples", len(ids))
    return len(ids)


def false_positive_rate(cfg):  # V6 FP tracking
    recs = DecisionLog(cfg.decision_log_path).read_all()
    reviewed = sum(1 for r in recs if r.get("event") == "route" and r.get("decision") == "review")
    dismissed = sum(len(r.get("extra", {}).get("ids", [])) for r in recs if r.get("event") == "dismiss")
    # also count false alarms recorded via the review-resolve path (operators / sync-reviews), not only `dismiss`
    dismissed += sum(1 for r in recs if r.get("event") == "review" and r.get("decision") == "false_alarm")
    fp = dismissed / reviewed if reviewed else 0.0
    return {"reviewed": reviewed, "dismissed_false_alarms": dismissed, "fp_rate": round(fp, 3)}


def set_threshold(adapter, cfg, class_name, conf_thr=None, dist_thr=None):  # AE6 per-class review policy
    """First-class, audited per-class threshold override on top of global calibration.
    Tighten a safety-critical class (higher conf bar / tighter distance) without re-flattening all."""
    if not cfg.thresholds_path.exists():
        raise ValueError("尚未校準;請先執行 vix calibrate")
    policy = ThresholdPolicy.load(cfg.thresholds_path)
    ct = policy.thresholds.get(class_name) or ClassThreshold(0.0, float("inf"), 0)
    new_conf = ct.conf_thr if conf_thr is None else float(conf_thr)
    new_dist = ct.dist_thr if dist_thr is None else float(dist_thr)
    policy.thresholds[class_name] = ClassThreshold(new_conf, new_dist, ct.n_support)
    policy.meta.setdefault("overrides", {})[class_name] = {"conf_thr": new_conf, "dist_thr": new_dist}
    policy.save(cfg.thresholds_path)
    DecisionLog(cfg.decision_log_path).append(
        "set_threshold", decision=class_name, extra={"conf_thr": new_conf, "dist_thr": new_dist}
    )
    log.info("set_threshold: %s conf<%.3f dist>%.3f (manual override, audited)", class_name, new_conf, new_dist)
    return {"class": class_name, "conf_thr": new_conf, "dist_thr": new_dist}


def reasons_breakdown(cfg):  # AE7 management summary: rejected/review grouped by plain-language reason
    recs = DecisionLog(cfg.decision_log_path).read_all()
    by_reason: dict[str, int] = {}
    n_review = 0
    for r in recs:
        if r.get("event") == "route" and r.get("decision") == "review":
            n_review += 1
            for reason in r.get("extra", {}).get("reasons", []):
                by_reason[reason] = by_reason.get(reason, 0) + 1
    rejected = sum(len(r.get("extra", {}).get("ids", [])) for r in recs if r.get("event") == "dismiss")
    rejected += sum(1 for r in recs if r.get("event") == "review" and r.get("decision") == "false_alarm")
    log.info("reasons_breakdown: %d review, %d rejected, %d reason types", n_review, rejected, len(by_reason))
    return {"n_review": n_review, "rejected": rejected, "by_reason": by_reason}


def throughput(cfg):  # AH8 review turnaround + rough effort estimate from the audit log (not an SLA promise)
    from datetime import datetime

    recs = DecisionLog(cfg.decision_log_path).read_all()
    routed_at: dict[str, str] = {}
    resolved_at: dict[str, str] = {}
    for r in recs:
        h, ev, dec, ts = r.get("vix_hash"), r.get("event"), r.get("decision"), r.get("ts_utc")
        if not h:
            continue
        if ev == "route" and dec == "review":
            routed_at.setdefault(h, ts)
        elif ev == "review":  # resolve_review closes a review item
            resolved_at[h] = ts
    durations = []
    for h, rt in routed_at.items():
        if h in resolved_at and rt and resolved_at[h]:
            try:
                d = (datetime.fromisoformat(resolved_at[h]) - datetime.fromisoformat(rt)).total_seconds() / 3600
                if d >= 0:
                    durations.append(d)
            except Exception:  # noqa: BLE001
                pass
    n_open = sum(1 for h in routed_at if h not in resolved_at)
    median = float(np.median(durations)) if durations else None
    p90 = float(np.percentile(durations, 90)) if durations else None
    est = (n_open * median) if median is not None else None
    log.info("throughput: %d resolved, median=%s h, %d open", len(durations), median, n_open)
    return {"n_resolved": len(durations), "median_hours": median, "p90_hours": p90,
            "n_open": n_open, "est_remaining_hours": est}


def capacity(cfg, volume=0):  # AI9 rough reviewer-hours plan from history x expected volume (estimate, not SLA)
    tp = throughput(cfg)
    recs = DecisionLog(cfg.decision_log_path).read_all()
    routed = sum(1 for r in recs if r.get("event") == "route")
    flagged = sum(1 for r in recs if r.get("event") == "route" and r.get("decision") == "review")
    flag_rate = flagged / routed if routed else 0.0
    median = tp["median_hours"]
    projected_review = int(volume * flag_rate)
    incoming_hours = (projected_review * median) if median is not None else None
    total = (tp["est_remaining_hours"] + incoming_hours) if (tp["est_remaining_hours"] is not None and incoming_hours is not None) else None
    return {"flag_rate": round(flag_rate, 3), "median_hours": median, "n_open": tp["n_open"],
            "backlog_hours": tp["est_remaining_hours"], "projected_review": projected_review,
            "incoming_hours": incoming_hours, "total_hours": total}


def relabel_rollback(adapter, cfg, change_log_path=None):  # V7 rollback via change log
    path = Path(change_log_path or (cfg.workspace / "relabel_changes.jsonl"))
    if not path.exists():
        return 0
    undo = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            c = json.loads(line)
            undo[c["id"]] = c["old"]
    n = 0
    for h, _s, dets, _t in adapter.samples():
        touched = False
        for i, det in enumerate(dets):
            rid = f"{h}:{i}"
            if rid in undo and det.label != undo[rid]:
                det.label = undo[rid]
                n += 1
                touched = True
        if touched:
            adapter.set_detections(h, dets)  # persist the rollback across CLI invocations
    DecisionLog(cfg.decision_log_path).append("relabel_rollback", decision=str(n))
    log.info("relabel_rollback: restored %d labels", n)
    return n


def resolve_review(adapter, cfg, vix_hash, decision, label=None, reviewer_id="reviewer"):
    """Apply a human review decision, closing the review -> golden loop (v1.0).

    decision: 'confirm' (-> golden; optional ``label`` relabels the detections)
              | 'false_alarm' (-> rejected, drops out of the review queue).
    Adding the golden/rejected tag is what removes the sample from the review
    queue (review_queue excludes those tags), so no tag-removal is needed.
    """
    _require_known(adapter, [vix_hash])  # B1: never log a phantom confirmation for an unknown id
    if decision == "confirm":
        if label:
            changes = []  # B2: make resolve --label reversible via the same log relabel uses
            for h, _src, dets, _t in adapter.samples():
                if h == vix_hash:
                    for i, d in enumerate(dets):
                        if d.label != label:
                            changes.append({"id": f"{h}:{i}", "old": d.label, "new": label})  # capture pre-overwrite
                            d.label = label
                    adapter.set_detections(h, dets)
                    break
            if changes:
                with open(cfg.workspace / "relabel_changes.jsonl", "a", encoding="utf-8") as f:
                    for c in changes:
                        f.write(json.dumps(c) + "\n")
        adapter.apply_tags(vix_hash, [Tag.GOLDEN])
        outcome = label or "confirmed"
    elif decision == "false_alarm":
        adapter.apply_tags(vix_hash, [Tag.REJECTED])
        outcome = "false_alarm"
    else:
        raise ValueError(f"unknown review decision: {decision}")
    DecisionLog(cfg.decision_log_path).append(
        "review", vix_hash=vix_hash, reviewer_id=reviewer_id, decision=outcome
    )
    log.info("resolve_review: %s -> %s (by %s)", vix_hash, outcome, reviewer_id)
    return outcome


def resolve_batch(adapter, cfg, decisions, reviewer_id="reviewer"):
    """Apply many review decisions, e.g. pulled from the App. Each item:
    {'vix_hash', 'decision', optional 'label'}."""
    for d in decisions:
        resolve_review(adapter, cfg, d["vix_hash"], d["decision"], d.get("label"), reviewer_id)
    log.info("resolve_batch: applied %d review decisions", len(decisions))
    return len(decisions)


def geometry_check(adapter, cfg, tag_a, tag_b, threshold=0.2):  # W3/W4 bbox geometry drift
    tag_a, tag_b = _resolve_tag(adapter, tag_a), _resolve_tag(adapter, tag_b)

    def dets_for(tag):
        out = []
        for _h, _s, dets, tags in adapter.samples():
            if tag in tags:
                out.extend(dets)
        return out

    res = geometry_drift(dets_for(tag_a), dets_for(tag_b), threshold)
    if res["alert"]:
        DecisionLog(cfg.decision_log_path).append(
            "geometry_drift", decision="ALERT", extra={"shifts": res["shifts"]}
        )
    log.info("geometry_check: alert=%s shifts=%s", res["alert"], res.get("shifts"))
    return res


def merge_preview(counts_a: dict, counts_b: dict, overrides: dict | None = None):  # W9
    merged = preview_merged_distribution(counts_a, counts_b, overrides)
    log.info("merge_preview: %d unified classes", len(merged))
    return merged


def merge_preview_tags(adapter, cfg, tag_a, tag_b, overrides=None):  # Z2 preview from live dataset
    from collections import Counter

    def counts(tag):
        c: Counter = Counter()
        for _h, _s, dets, tags in adapter.samples():
            if tag in tags:
                for d in dets:
                    c[d.label] += 1
        return dict(c)

    return merge_preview(counts(tag_a), counts(tag_b), overrides)


def merge_datasets(adapter, cfg, tag_a, tag_b, overrides=None):  # AA2: conflicts + preview in one
    from collections import Counter

    def counts(tag):
        c: Counter = Counter()
        for _h, _s, dets, tags in adapter.samples():
            if tag in tags:
                for d in dets:
                    c[d.label] += 1
        return dict(c)

    counts_a, counts_b = counts(tag_a), counts(tag_b)
    map_a = {i: n for i, n in enumerate(sorted(counts_a))}
    map_b = {i: n for i, n in enumerate(sorted(counts_b))}
    report = merge_class_maps(map_a, map_b, overrides)
    report["preview_distribution"] = preview_merged_distribution(counts_a, counts_b, overrides)
    log.info("merge_datasets: %d unified classes, %d need decision",
             len(report["unified_names"]), len(report["needs_decision"]))
    return report


# --- reference-list concepts (1–6) + review writeback loop (a) -----------

def calibrate_confidence(conf, correct, cfg):  # concept #1
    """Fit a temperature on (confidence, correct) pairs; report ECE before/after."""
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    t = fit_temperature(conf, correct)
    before = expected_calibration_error(conf, correct)
    after = expected_calibration_error(apply_temperature(conf, t), correct)
    payload = {"temperature": t, "ece_before": round(before, 4), "ece_after": round(after, 4)}
    cfg.calibration_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    log.info("calibrate_confidence: T=%.3f ECE %.4f -> %.4f", t, before, after)
    return payload


def label_noise(adapter, cfg, k=None, want_tags=None):  # concept #2 (confident learning on embedding-kNN pseudo-preds)
    items = _detection_items(adapter, want_tags=want_tags or [Tag.GOLDEN])
    if len(items) < 2:
        return {"issues": [], "noise_rates": {}}
    ids = [it.id for it in items]
    given = [it.label for it in items]
    M = _l2norm(np.vstack([np.asarray(it.embedding, float) for it in items]))
    sims = M @ M.T
    np.fill_diagonal(sims, -np.inf)
    kk = min(k or cfg.knn_k, len(items) - 1)
    pred = []
    conf = [float(it.confidence) for it in items]  # the detection's own confidence
    for i in range(len(items)):
        nn = np.argpartition(-sims[i], kk - 1)[:kk]
        labs = [given[j] for j in nn]
        vals, counts = np.unique(labs, return_counts=True)
        pred.append(str(vals[int(np.argmax(counts))]))  # embedding kNN-majority as the "model" label
    issues = find_label_issues(ids, given, pred, conf)
    C, classes, _thr = confident_joint(given, pred, conf)
    rates = noise_rates(C, classes)
    DecisionLog(cfg.decision_log_path).append(
        "label_noise", decision=str(len(issues)), extra={"noise_rates": rates}
    )
    log.info("label_noise: %d issues, %d class-pair noise rates", len(issues), len(rates))
    return {"issues": issues, "noise_rates": rates}


def compare(adapter, cfg, tag_a, tag_b, k=None, max_distance=0.05):  # AC4: side-by-side vendor/source comparison
    """One-shot side-by-side comparison of two tagged subsets (e.g. two annotation
    vendors): per-subset label-noise %, near-duplicate filler, plus cross-subset
    'recycling' (images in B that are near-duplicates of an image in A)."""
    tag_a, tag_b = _resolve_tag(adapter, tag_a), _resolve_tag(adapter, tag_b)
    out: dict = {"tags": [tag_a, tag_b], "per_tag": {}, "cross_recycled": 0}
    embs: dict[str, np.ndarray] = {}
    for tag in (tag_a, tag_b):
        det_items = _detection_items(adapter, want_tags=[tag])
        img_items = _image_items(adapter, want_tags=[tag])
        ln = label_noise(adapter, cfg, want_tags=[tag]) if len(det_items) >= 2 else {"issues": []}
        dups = near_duplicate_groups(img_items, max_distance) if img_items else []
        out["per_tag"][tag] = {
            "n_samples": len(img_items),
            "n_detections": len(det_items),
            "n_label_issues": len(ln["issues"]),
            "noise_pct": round(100 * len(ln["issues"]) / max(1, len(det_items)), 1),
            "dup_groups": len(dups),
            "redundant": sum(len(g) - 1 for g in dups),
        }
        embs[tag] = (
            _l2norm(np.vstack([np.asarray(it.embedding, float) for it in img_items]))
            if img_items
            else np.zeros((0, 1))
        )
    A, B = embs[tag_a], embs[tag_b]
    if A.size and B.size and A.shape[1] == B.shape[1]:
        nearest = (B @ A.T).max(axis=1)  # cosine sim of each B image to its closest A image
        out["cross_recycled"] = int((1.0 - nearest <= max_distance).sum())
    DecisionLog(cfg.decision_log_path).append(
        "compare", decision=f"{tag_a}|{tag_b}", extra={**out["per_tag"], "cross_recycled": out["cross_recycled"]}
    )
    log.info("compare %s vs %s -> %s (cross_recycled=%d)", tag_a, tag_b, out["per_tag"], out["cross_recycled"])
    return out


def drift_type(adapter, cfg, tag_a, tag_b):  # concept #3
    tag_a, tag_b = _resolve_tag(adapter, tag_a), _resolve_tag(adapter, tag_b)
    a = _detection_items(adapter, want_tags=[tag_a])
    b = _detection_items(adapter, want_tags=[tag_b])
    ref_emb = np.vstack([x.embedding for x in a]) if a else np.zeros((0, 1))
    new_emb = np.vstack([x.embedding for x in b]) if b else np.zeros((0, 1))
    res = diagnose_drift_type(ref_emb, new_emb, [x.label for x in a], [x.label for x in b],
                              cfg.drift_shift_threshold)
    DecisionLog(cfg.decision_log_path).append("drift_type", decision=res["verdict"], extra=res)
    log.info("drift_type: %s (cov=%.3f pred=%.3f)", res["verdict"],
             res["covariate_shift"], res["prediction_shift"])
    return res


def review_rate_series(adapter, cfg) -> list[float]:
    """Per-batch review ratio (for SPC monitoring)."""
    by_batch: dict[str, list[int]] = defaultdict(list)
    for _h, _s, _d, tags in adapter.samples():
        batch, _ = _parse_meta(tags)
        if batch:
            by_batch[batch].append(1 if Tag.REVIEW in tags else 0)
    return [sum(v) / len(v) for _b, v in sorted(by_batch.items()) if v]


def spc_monitor(series, target=None, sigma=None, lam=0.3, method="ewma"):  # concept #4
    series = list(series)
    if not series:
        return {"alarm": False, "alarm_index": None, "short_series": True}
    # Estimate the control limits from an IN-CONTROL baseline (the earliest ~25% of the
    # series), NOT the whole series — otherwise a slow monotonic drift gets absorbed into
    # target/sigma and only alarms at the last point, defeating the leading-indicator purpose (AG4).
    if target is None or sigma is None:
        m = max(3, len(series) // 4)
        base = series[:m]
        if target is None:
            target = float(np.median(base))
        if sigma is None:
            sigma = float(np.std(base)) or 1e-6
    res = (
        spc_mod.cusum_alarm(series, target, k=0.5 * sigma, h=4 * sigma)
        if method == "cusum"
        else spc_mod.ewma_alarm(series, target, sigma, lam)
    )
    res["short_series"] = len(series) < 8  # too few batches -> control limits not yet trustworthy
    return res


def parity(adapter, cfg, by="fab", lower_is_worse=True):  # concept #5
    prefix = f"{by}:"
    vals: dict[str, list[float]] = defaultdict(list)
    for _h, _s, dets, tags in adapter.samples():
        grp = next((t[len(prefix):] for t in tags if t.startswith(prefix)), None)
        if grp is not None:
            vals[grp].append(max((d.confidence for d in dets), default=0.0))
    group_means = {g: sum(v) / len(v) for g, v in vals.items() if v}
    group_counts = {g: len(v) for g, v in vals.items() if v}
    res = performance_parity(group_means, lower_is_worse=lower_is_worse, group_counts=group_counts)
    log.info("parity by %s: median=%.3f flagged=%s", by, res["median"], res["flagged"])
    return res


def cost_gate_eval(cfg, cr, fa, miss_cost, fa_cost, budget):  # concept #6
    res = cost_gate(1.0 - cr, fa, miss_cost, fa_cost, budget)
    DecisionLog(cfg.decision_log_path).append("cost_gate", decision=res["verdict"], extra=res)
    log.info("cost_gate: expected=%.3f budget=%.3f -> %s",
             res["expected_cost_per_unit"], budget, res["verdict"])
    return res


def sync_reviews(adapter, cfg):  # (a) close the App-review writeback loop
    decisions = adapter.pull_review_decisions()
    n = 0
    for d in decisions:
        dec = (d.decision or "").lower()
        if dec in ("false_alarm", "reject", "rejected"):
            resolve_review(adapter, cfg, d.vix_hash, "false_alarm", reviewer_id=d.reviewer_id)
        elif dec in ("", "confirm", "confirmed"):
            resolve_review(adapter, cfg, d.vix_hash, "confirm", reviewer_id=d.reviewer_id)
        else:  # a concrete class label -> confirm + relabel
            resolve_review(adapter, cfg, d.vix_hash, "confirm", label=d.decision, reviewer_id=d.reviewer_id)
        n += 1
    log.info("sync_reviews: applied %d pulled review decisions", n)
    return n


def restore_apply(adapter, cfg, path):  # X10 replay a snapshot's composition into the dataset
    snap = snap_mod.restore(path)
    man = Manifest.load(cfg.manifest_path)
    n = 0
    for row in snap["composition"]:
        entry = ManifestEntry(
            vix_hash=row["vix_hash"], src_path=row["src_path"], batch_id=row.get("batch_id", ""),
            ingested_at="", label_version=row.get("label_version", "v0"), tags=row.get("tags", []),
        )
        man.append(entry)
        n += 1
    adapter.sync(man.entries())
    DecisionLog(cfg.decision_log_path).append(
        "restore_apply", decision=snap["version"], extra={"n": n, "content_hash": snap["content_hash"]}
    )
    log.info("restore_apply: replayed %d samples from version %s", n, snap["version"])
    return {"version": snap["version"], "n_restored": n, "content_hash": snap["content_hash"]}


def harmful_remove(adapter, cfg, top=20, note=""):  # X9 one-step: rank harmful -> dismiss -> audit
    ids = [r["id"] for r in harmful(adapter, cfg, top=top)]
    for h in ids:
        adapter.apply_tags(h, [Tag.REJECTED])
    DecisionLog(cfg.decision_log_path).append(
        "harmful_remove", decision=str(len(ids)), extra={"ids": ids, "note": note}
    )
    log.info("harmful_remove: removed %d harmful samples (%s)", len(ids), note)
    return ids
