"""VIX command-line interface (v0.1).

    vix ingest <folder> --batch ID [--golden|--anchor]
    vix infer  --weights model.pt
    vix embed
    vix calibrate
    vix route
    vix guard  [--ack "reason"]
    vix export <dst> [--classes a,b] [--copy-images]
    vix app

Default backend is FiftyOne; use ``--adapter memory`` for a FiftyOne-free dry-run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import pipeline
from .config import Config
from .core.threshold import ThresholdPolicy
from .logging_setup import get_logger, setup_logging
from .types import Tag

log = get_logger("vix.cli")

_QUICKSTART = """\
VIX 快速上手(5 分鐘)
=====================
核心概念:
  golden  已確認、可進訓練的資料(事實基準)
  anchor  從 golden 凍結的一小份,永不訓練,用來偵測定義漂移
  review  被攔下待人工覆核的樣本    pass  自動通過
  rejected 經 dismiss 的誤報/有害樣本(排除於覆核佇列)

最短工作流(離線可跑:加 --adapter memory):
  1. vix ingest ./golden  --batch init --golden      # 匯入黃金集
     vix ingest ./anchor  --batch init --anchor      # 凍結錨點
     vix ingest ./incoming --batch w22               # 新批次
  2. vix infer --weights yolo.pt                      # YOLO 偵測
  3. vix embed                                        # DINOv2 + kNN 索引
  4. vix calibrate                                    # per-class 門檻
  5. vix route                                        # pass/review + 理由
  6. vix review-queue --top 40                        # 最高風險先看
  7. vix gate                                         # 能不能訓練? GO/NO-GO
  8. vix report ./out                                 # 品質分數+導覽報告
  9. vix export ./train_ready                         # 匯出 YOLOv8 + 逐檔hash

一鍵一條龍:  vix run --input ./incoming --batch w22 --weights yolo.pt --export ./train_ready
新人看現況:  vix report ./out   (自動對比上一份報告,含品質分數與下一步建議)
完整指令:    vix --help
"""


def _load_json_arg(s: str):
    """Accept either a path to a JSON file (BOM-tolerant) or an inline JSON string."""
    p = Path(s)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8-sig"))
    return json.loads(s)  # inline JSON (e.g. quoted from PowerShell)


def make_adapter(cfg: Config, kind: str):
    if kind == "auto":
        try:
            import fiftyone  # noqa: F401
        except ImportError:
            log.warning("FiftyOne not installed; falling back to --adapter memory (pixel embedder).")
            kind = "memory"
    if kind == "memory":
        from .adapters.memory import InMemoryAdapter
        from .embedding.simple import pixel_embedding

        cfg.embedding_backend = "pixel_fallback"  # mark offline fallback in audit/report
        # persist dry-run state so standalone commands (embed->route->explain…) work
        # across separate CLI invocations, not only inside a single `vix run`.
        return InMemoryAdapter(
            embedder=pixel_embedding, state_path=cfg.workspace / "memory_state.pkl"
        )
    from .adapters.fiftyone_adapter import FiftyOneAdapter

    return FiftyOneAdapter(cfg)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vix", description="Vision Integrity eXplainability - data gatekeeper")
    p.add_argument("--workspace", default=None, help="workspace dir (default ./vix_workspace or $VIX_WORKSPACE)")
    p.add_argument("--adapter", choices=["auto", "fiftyone", "memory"], default="auto")
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("ingest", help="import a folder of images into the dataset")
    sp.add_argument("folder")
    sp.add_argument("--batch", required=True)
    sp.add_argument("--golden", action="store_true", help="tag as golden set")
    sp.add_argument("--anchor", action="store_true", help="tag as frozen anchor set (drift reference)")
    sp.add_argument("--eval", dest="is_eval", action="store_true",
                    help="tag as held-out eval/regression set (never calibrated/routed/exported on)")

    si = sub.add_parser("infer", help="run YOLO -> detections (or --synthetic for an offline demo)")
    si.add_argument("--weights", default=None)
    si.add_argument("--synthetic", action="store_true", help="seed deterministic synthetic detections (offline demo/CI)")
    sub.add_parser("embed", help="DINOv2 embeddings + kNN index")
    sub.add_parser("calibrate", help="compute per-class percentile thresholds")
    sub.add_parser("route", help="route candidates to pass/review")

    sg = sub.add_parser("guard", help="frozen-reference drift self-gate")
    sg.add_argument("--ack", default=None, help="written acknowledgement to proceed past a triggered guard")
    sg.add_argument("--build", action="store_true", help="(re)build the frozen reference from anchors first")

    se = sub.add_parser("export", help="one-way export golden -> YOLO txt + data.yaml")
    se.add_argument("dst")
    se.add_argument("--classes", default=None, help="comma-separated class names (default: from thresholds.json)")
    se.add_argument("--copy-images", action="store_true")

    sub.add_parser("app", help="launch the FiftyOne review App")

    sa = sub.add_parser("audit-labels", help="find suspected label errors (kNN disagreement)")
    sa.add_argument("--k", type=int, default=None)
    sa.add_argument("--top", type=int, default=20)

    sd = sub.add_parser("dedup", help="find near-duplicate image groups")
    sd.add_argument("--max-distance", type=float, default=0.05)
    sd.add_argument("--full", action="store_true", help="list every member hash (default: count + first few)")

    scov = sub.add_parser("coverage", help="class distribution + coverage gaps (+need X more)")
    scov.add_argument("--target", type=int, default=None, help="absolute per-class target count")

    sv = sub.add_parser("value", help="how much new (non-golden) data covers novel regions")
    sv.add_argument("--radius", type=float, default=0.2)

    sal = sub.add_parser("active-learn", help="rank unlabeled candidates to label next")
    sal.add_argument("--budget", type=int, default=50)

    sdr = sub.add_parser("drift", help="cross-period class-definition drift")
    sdr.add_argument("--from", dest="from_tag", required=True)
    sdr.add_argument("--to", dest="to_tag", required=True)

    ss = sub.add_parser("snapshot", help="create an immutable dataset version snapshot")
    ss.add_argument("--version", required=True)

    sr = sub.add_parser("restore", help="restore a snapshot's composition + params")
    sr.add_argument("path")
    sr.add_argument("--apply", action="store_true", help="replay the composition back into the dataset")

    srep = sub.add_parser("report", help="one-click dataset health report")
    srep.add_argument("dst")
    srep.add_argument("--version", default="current")

    srq = sub.add_parser("review-queue", help="unified risk-ranked review queue (+ plain-language why)")
    srq.add_argument("--top", type=int, default=50)

    sau = sub.add_parser("audit", help="filter the append-only decision log")
    sau.add_argument("--since")
    sau.add_argument("--until")
    sau.add_argument("--event")
    sau.add_argument("--reviewer")

    sm = sub.add_parser("merge", help="reconcile two datasets' class maps + distribution preview (T2/AA2)")
    sm.add_argument("--map-a", help="JSON {id: name} (or use --tag-a/--tag-b)")
    sm.add_argument("--map-b", help="JSON {id: name}")
    sm.add_argument("--tag-a", help="dataset tag for subset A (one-command conflicts + preview)")
    sm.add_argument("--tag-b", help="dataset tag for subset B")
    sm.add_argument("--override", action="append", default=[], help="name=canonical (repeatable)")

    srl = sub.add_parser("relabel", help="rename/merge classes across the dataset, with rollback log")
    srl.add_argument("--map", action="append", help="old=new (repeatable)")
    srl.add_argument("--rollback", action="store_true", help="undo the last relabel via change log")

    sru = sub.add_parser("run", help="one-stop pipeline (ingest->infer->embed->route->guard->report->gate->export)")
    sru.add_argument("--input", default=None, help="folder to ingest first")
    sru.add_argument("--batch", default="run")
    sru.add_argument("--weights", default=None)
    sru.add_argument("--export", dest="export_dst", default=None)

    sh = sub.add_parser("history", help="per-image submission/decision history")
    sh.add_argument("hash")
    sub.add_parser("routing-diff", help="what changed between the last two routing runs")
    sdm = sub.add_parser("dismiss", help="mark samples as false alarms (excluded from review queue)")
    sdm.add_argument("ids", nargs="+")
    srd = sub.add_parser("restore-dismissed", help="reverse a dismiss/harmful-remove (un-reject), audited")
    srd.add_argument("ids", nargs="+")
    sub.add_parser("fp-rate", help="false-positive rate of routing vs dismissed")
    sgeo = sub.add_parser("geometry", help="bbox geometry drift between two tagged periods (W3/W4)")
    sgeo.add_argument("--from", dest="from_tag", required=True)
    sgeo.add_argument("--to", dest="to_tag", required=True)

    smp = sub.add_parser("merge-preview", help="preview merged class distribution before committing (W9)")
    smp.add_argument("--counts-a", help="JSON {class: count}")
    smp.add_argument("--counts-b", help="JSON {class: count}")
    smp.add_argument("--tag-a", help="dataset tag for subset A (alternative to --counts-a)")
    smp.add_argument("--tag-b", help="dataset tag for subset B")
    smp.add_argument("--override", action="append", default=[], help="name=canonical (repeatable)")

    sub.add_parser("quickstart", help="print the recommended new-user workflow + concept glossary")

    sres = sub.add_parser("resolve", help="apply a human review decision (close review->golden loop)")
    sres.add_argument("hash")
    sres.add_argument("--confirm", action="store_true")
    sres.add_argument("--false-alarm", dest="false_alarm", action="store_true")
    sres.add_argument("--label", default=None, help="optional corrected class when confirming")
    sres.add_argument("--reviewer", default="reviewer")

    sub.add_parser("sync-reviews", help="pull App review decisions and write them back (close the loop)")

    # --- reference-list concepts (1-6) ---
    scal = sub.add_parser("calibrate-confidence", help="temperature scaling on eval JSONL {conf,correct} (#1)")
    scal.add_argument("eval", help="JSONL with fields: conf (0-1), correct (0/1)")
    sub.add_parser("label-noise", help="confident-learning class-pair noise + label issues (#2)")
    sdt = sub.add_parser("drift-type", help="covariate vs concept drift between two tags (#3)")
    sdt.add_argument("--from", dest="from_tag", required=True)
    sdt.add_argument("--to", dest="to_tag", required=True)
    scmp = sub.add_parser("compare", help="side-by-side compare two tagged subsets (e.g. two annotation vendors) (AC4)")
    scmp.add_argument("--tag-a", dest="tag_a", required=True)
    scmp.add_argument("--tag-b", dest="tag_b", required=True)
    sst = sub.add_parser("set-threshold", help="override one class's routing threshold (per-class policy), audited")
    sst.add_argument("class_name")
    sst.add_argument("--conf", type=float, default=None, help="flag low_conf below this confidence")
    sst.add_argument("--dist", type=float, default=None, help="flag far_from_known above this distance")
    sub.add_parser("reasons", help="management summary: review/rejected grouped by plain-language reason")
    sub.add_parser("throughput", help="review turnaround (median/p90) + rough remaining effort from the audit log")
    scap = sub.add_parser("capacity", help="rough reviewer-hours plan: history x expected volume (estimate, not SLA)")
    scap.add_argument("--volume", type=int, default=0, help="expected incoming images next period")
    ssp = sub.add_parser("spc", help="SPC EWMA/CUSUM leading indicator on per-batch review-rate (#4)")
    ssp.add_argument("--method", choices=["ewma", "cusum"], default="ewma")
    ssp.add_argument("--target", type=float, default=None)
    ssp.add_argument("--sigma", type=float, default=None)
    spar = sub.add_parser("parity", help="cross-group performance parity, e.g. by fab (#5)")
    spar.add_argument("--by", default="fab")
    scg = sub.add_parser("cost-gate", help="asymmetric miss/false-alarm cost gate (#6)")
    scg.add_argument("--cr", type=float, required=True)
    scg.add_argument("--fa", type=float, required=True)
    scg.add_argument("--miss-cost", type=float, required=True)
    scg.add_argument("--fa-cost", type=float, default=1.0)
    scg.add_argument("--budget", type=float, required=True)

    sub.add_parser("verify-fiftyone", help="Tier-2 headless 驗證:FiftyOneAdapter 全鏈 + sync_reviews(需 fiftyone)")
    svg = sub.add_parser("verify-gui", help="Tier-2 GUI 驗證:Playwright 驅動 App + 執行 operator(需 fiftyone+playwright)")
    svg.add_argument("--no-execute", action="store_true", help="只截圖,不實際點 Execute")

    snc = sub.add_parser("new-classes", help="open-set: surface suspected new classes (U1)")
    snc.add_argument("--novelty-radius", type=float, default=0.3)
    snc.add_argument("--cluster-distance", type=float, default=0.2)
    sub.add_parser("leakage", help="train/val/test cross-split duplicate leakage (U3)")
    shf = sub.add_parser("harmful", help="rank the most harmful samples (U5)")
    shf.add_argument("--top", type=int, default=50)
    shf.add_argument("--remove", action="store_true", help="dismiss the ranked samples (audited)")
    shf.add_argument("--note", default="", help="reason recorded in the audit log when removing")
    sub.add_parser("trend", help="per-class confidence trend across batches + drop alerts (U10)")
    sra = sub.add_parser("reviewer-audit", help="per-reviewer self-consistency (U2)")
    sra.add_argument("--class", dest="class_filter", default=None, help="restrict to one class")
    sub.add_parser("gate", help="pre-training go/no-go gate (U7)")
    sex = sub.add_parser("explain", help="drill-down why one image was flagged (U9)")
    sex.add_argument("hash")
    sve = sub.add_parser("verify", help="verify a received dataset vs its export manifest (U8)")
    sve.add_argument("manifest")
    sve.add_argument("data_dir")
    sevi = sub.add_parser("eval-ingest", help="ingest a val eval (GT+pred) -> per-class AP / confusion / FP-FN (close the model loop)")
    sevi.add_argument("results", help="JSONL or JSON array of {vix_hash, gt:[{label,bbox}], pred:[{label,bbox,conf}]}")
    sevi.add_argument("--iou", type=float, default=0.5)
    sem = sub.add_parser("error-mine", help="rank unlabeled candidates nearest the model's eval FP/FN errors")
    sem.add_argument("--top", type=int, default=20)
    seb = sub.add_parser("set-eval-baseline", help="freeze current eval as the challenge-guard baseline (gate hard-blocks on mAP / protected-class AP regression)")
    seb.add_argument("--protect", action="append", default=[], metavar="CLASS", help="protected class (repeatable); its AP drop hard-blocks the gate, even at low support (fail-closed)")
    seb.add_argument("--protect-drop", type=float, default=0.05, help="max allowed AP drop for protected classes")
    seb.add_argument("--map-drop", type=float, default=0.02, help="max allowed overall mAP drop")
    sbq = sub.add_parser("box-qa", help="per-box geometry QA on golden boxes (degenerate/truncated/area/aspect outliers) — read-only")
    sbq.add_argument("--top", type=int, default=50)
    shn = sub.add_parser("hardneg", help="rank the detector's most confident-yet-wrong detections (GT eval-FP, or GT-free embedding overturn)")
    shn.add_argument("--top", type=int, default=50)
    shn.add_argument("--mode", choices=["auto", "gt", "gt_free"], default="auto")
    swr = sub.add_parser("weakness-report", help="roll per-class AP + confusion + FP/FN typing + loc_gap + hardneg + consistency into a human-readable report (writes .md + .html)")
    swr.add_argument("--top-classes", type=int, default=5)
    swr.add_argument("--queue-per-class", type=int, default=10)
    swr.add_argument("--out", default=None)
    scon = sub.add_parser("consistency", help="GT x embedding attribution: per class-pair separability + confusion-overlap 2x2 (taxonomy/model/label_noise) — advisory, CI-gated")
    scon.add_argument("--max-pairs", type=int, default=20)
    sae = sub.add_parser("adapt-embedding", help="learn a supervised LDA projection of frozen DINOv2 from golden GT; report which 'inseparable' pairs become separable (CV'd) — offline, not YOLO training")
    sae.add_argument("--save", action="store_true", help="persist embed_projection.npz + adapt_report.json")
    sae.add_argument("--enable", action="store_true", help="enable the projection for ranking (error-mine) — only takes effect if the gate says GO")
    sae.add_argument("--max-pca", type=int, default=64)
    sqhr = sub.add_parser("queue-hit-rate", help="did VIX's suggestion queues turn out right? join past emissions with human resolutions -> per-queue precision + trend (self-calibration)")
    sqhr.add_argument("--min-resolved", type=int, default=5)
    sba = sub.add_parser("bank-audit", help="multi-bank Top-K embedding audit of low-conf proposals -> defect/reflection/unknown (advisory)")
    sba.add_argument("--defect-tag", default="golden")
    sba.add_argument("--reflection-tag", default="rejected")
    sba.add_argument("--normal-tag", default=None)
    sba.add_argument("--conf-lo", type=float, default=0.05)
    sba.add_argument("--conf-hi", type=float, default=0.25)
    sba.add_argument("--tau", type=float, default=0.10, help="margin abstain knob (calibrated score)")
    sba.add_argument("--novelty-radius", type=float, default=0.30)
    sba.add_argument("--dedup-distance", type=float, default=0.05)
    sba.add_argument("--top", type=int, default=50)
    return p


def main(argv: list[str] | None = None) -> int:
    # CLI prints Chinese + ⚠️; the default Windows cp950 console raises UnicodeEncodeError
    # on those, so force UTF-8 (errors=replace) before any output.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    try:
        return _main(argv)
    except (ValueError, OSError) as e:  # expected user errors (missing/corrupt files, bad state) -> clean line
        print(f"錯誤:{e}", file=sys.stderr)
        print("(先確認前置步驟,或用 --log-level DEBUG 取得完整堆疊)", file=sys.stderr)
        return 2


def _main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = Config(workspace=args.workspace) if args.workspace else Config()
    cfg.ensure_dirs()
    # share one absolute workspace across CLI and any in-process App operators (vix app / verify-gui)
    os.environ["VIX_WORKSPACE"] = str(cfg.workspace.resolve())
    # auto-point FiftyOne at the bundled VIX plugins so `vix app` shows the review workstation
    # (must be set before fiftyone is first imported); user can override.
    os.environ.setdefault("FIFTYONE_PLUGINS_DIR", str(Path(__file__).resolve().parent / "plugins"))
    if args.adapter == "memory":
        cfg.embedding_backend = "pixel_fallback"  # mark offline fallback in audit/report
    setup_logging(args.log_level, log_file=cfg.log_path)
    adapter = make_adapter(cfg, args.adapter)

    if args.cmd == "ingest":
        tags = []
        if args.golden:
            tags.append(Tag.GOLDEN)
        if args.anchor:
            tags.append(Tag.ANCHOR)
        if args.is_eval:
            tags.append(Tag.EVAL)
        n_new, n_skipped = pipeline.ingest(adapter, cfg, args.folder, args.batch, tags=tags)
        print(f"ingested {n_new} new images, skipped {n_skipped} (already present)")
        if n_skipped:
            print("注:既有雜湊的影像本批未重新採用其標籤(以首次為準);如需更正用 resolve --confirm --label")

    elif args.cmd == "infer":
        if args.synthetic:
            n = pipeline.infer_synthetic(adapter, cfg)
            print(f"seeded synthetic detections on {n} images (offline demo; NOT real inference)")
        else:
            if not args.weights:
                raise ValueError("infer 需要 --weights(或用 --synthetic 做離線示範)")
            from .detect import run_yolo

            n = run_yolo(adapter, cfg, args.weights)
            print(f"inferred {n} images")

    elif args.cmd == "embed":
        adapter.compute_embeddings(cfg.dinov2_model_key)
        adapter.build_knn_index()
        print("embeddings + kNN index built")

    elif args.cmd == "calibrate":
        pol = pipeline.calibrate(adapter, cfg)
        print(f"calibrated {len(pol.thresholds)} classes -> {cfg.thresholds_path}")

    elif args.cmd == "route":
        counts = pipeline.route(adapter, cfg)
        print(f"routed: {counts['pass']} pass, {counts['review']} review")
        if counts.get("warning"):
            print(f"⚠️ {counts['warning']}")
        if counts.get("backend_mismatch"):
            print("⚠️ 校準與目前 embedding 後端不一致;距離門檻不可靠,請以同一後端重新 calibrate")

    elif args.cmd == "guard":
        if args.build:
            pipeline.build_reference(adapter, cfg)
        report = pipeline.guard(adapter, cfg, ack=args.ack)
        if report.triggered and not args.ack:
            print(f"GUARD TRIGGERED {report.reasons} (shift={report.max_shift:.3f}). "
                  f"Re-run with --ack '<reason>' to proceed.")
            return 2
        print(f"guard ok (shift={report.max_shift:.3f}, drop={report.consistency_drop:.3f})")

    elif args.cmd == "export":
        classes = (
            args.classes.split(",")
            if args.classes
            else sorted(ThresholdPolicy.load(cfg.thresholds_path).thresholds)
        )
        res = pipeline.export(adapter, cfg, classes, args.dst, copy_images=args.copy_images)
        print(f"exported {res['n_images']} images, {res['n_labels']} labels -> {res['data_yaml']}")

    elif args.cmd == "app":
        adapter.launch_app()

    elif args.cmd == "audit-labels":
        issues = pipeline.audit_labels(adapter, cfg, k=args.k)
        for i in issues[: args.top]:
            print(f"{i.id}: given={i.given_label} -> suggested={i.suggested_label} (disagree={i.disagreement:.2f})")
        print(f"{len(issues)} suspected label errors")

    elif args.cmd == "dedup":
        groups = pipeline.dedup(adapter, cfg, args.max_distance)
        for g in groups:
            if args.full:
                print(f"dup group ({len(g)}):", ", ".join(g))
            else:
                preview = ", ".join(g[:3]) + (f" …(+{len(g) - 3})" if len(g) > 3 else "")
                print(f"dup group ({len(g)}): {preview}")
        print(f"{len(groups)} near-duplicate groups ({sum(len(x) - 1 for x in groups)} redundant)")
        print("注:>~2000 自動切換 LSH(近似召回);--adapter memory 用像素特徵精度較低;--full 列出所有雜湊")

    elif args.cmd == "coverage":
        cov = pipeline.coverage(adapter, cfg, target=args.target)
        for c, n in sorted(cov["distribution"].items(), key=lambda kv: -kv[1]):
            g = cov["gaps"][c]
            mark = f"  (under-represented, 還需 {g['need']} 張)" if g["under_represented"] else ""
            print(f"{c}: {n}{mark}")

    elif args.cmd == "value":
        res = pipeline.coverage_value(adapter, cfg, args.radius)
        print(f"novel coverage: {res['novel_fraction']:.1%} ({len(res['novel_ids'])} images)")

    elif args.cmd == "active-learn":
        for r in pipeline.active_learn(adapter, cfg, args.budget):
            print(f"{r['id']}  score={r['score']}  (uncertainty={r['uncertainty']}, novelty={r['novelty']})")
            print(f"    why: {r['why']}")

    elif args.cmd == "drift":
        result = pipeline.drift_periods(adapter, cfg, args.from_tag, args.to_tag)
        for c, v in result.items():
            print(f"{c}: shift={v['shift']:.3f} alert={v['alert']}")

    elif args.cmd == "snapshot":
        snap, out = pipeline.snapshot(adapter, cfg, args.version)
        print(f"snapshot {args.version}: {snap['n_golden']} golden, {snap['n_excluded']} excluded -> {out}")

    elif args.cmd == "restore":
        if args.apply:
            r = pipeline.restore_apply(adapter, cfg, args.path)
            print(f"restored version {r['version']}: replayed {r['n_restored']} samples "
                  f"(content_hash={r['content_hash'][:12]}...)")
        else:
            r = pipeline.restore(cfg, args.path)
            print(f"version {r['version']}: {len(r['composition'])} golden, {len(r['excluded'])} excluded")

    elif args.cmd == "report":
        _report, paths = pipeline.health_report(adapter, cfg, args.dst, version=args.version)
        print(f"report -> {paths['md']}")

    elif args.cmd == "review-queue":
        for r in pipeline.review_queue(adapter, cfg, args.top):
            print(f"{r['id']}  risk={r['risk']:.3f}  {r['why']}")

    elif args.cmd == "audit":
        # friendly aliases: resolutions are logged as `review`; removals as `harmful_remove`
        event = {"resolve": "review", "remove": "harmful_remove"}.get(args.event, args.event)
        recs = pipeline.audit(cfg, args.since, args.until, event, args.reviewer)
        for r in recs:
            print(f"{r['ts_utc']}  {r['event']}  {r.get('vix_hash','')}  "
                  f"{r.get('decision','')}  by={r.get('reviewer_id')}")
        print(f"{len(recs)} matching records (時間以 UTC 比對;--since/--until 請用 UTC 或純日期)")

    elif args.cmd == "merge":
        overrides = dict(o.split("=", 1) for o in args.override)
        if args.tag_a and args.tag_b:
            res = pipeline.merge_datasets(adapter, cfg, args.tag_a, args.tag_b, overrides)
        elif args.map_a and args.map_b:
            map_a = {int(k): v for k, v in json.loads(Path(args.map_a).read_text(encoding="utf-8-sig")).items()}
            map_b = {int(k): v for k, v in json.loads(Path(args.map_b).read_text(encoding="utf-8-sig")).items()}
            res = pipeline.merge_maps(map_a, map_b, overrides)
        else:
            raise SystemExit("provide either --tag-a/--tag-b or --map-a/--map-b")
        print("unified:", res["unified_names"])
        print("needs decision:", res["needs_decision"])
        print("orphans:", res["orphans"])
        if "preview_distribution" in res:
            total = sum(res["preview_distribution"].values()) or 1
            print("merged distribution preview:")
            for c, n in sorted(res["preview_distribution"].items(), key=lambda kv: -kv[1]):
                flag = "  ⚠️ <5%" if n / total < 0.05 else ""
                print(f"  {c}: {n} ({n / total:.1%}){flag}")

    elif args.cmd == "relabel":
        if args.rollback:
            n = pipeline.relabel_rollback(adapter, cfg)
            print(f"rolled back {n} labels")
        else:
            mapping = dict(m.split("=", 1) for m in (args.map or []))
            diff = pipeline.relabel_dataset(adapter, cfg, mapping)
            print(f"relabel: {diff['total_changed']} changed | {diff['by_transition']}")
            print("注:relabel 僅保證機械式一致與可回滾(relabel --rollback),不保證新定義語意正確;請以 vix audit-labels 複查")

    elif args.cmd == "run":
        s = pipeline.run_pipeline(adapter, cfg, input_folder=args.input, batch_id=args.batch,
                                  weights=args.weights, export_dst=args.export_dst)
        print(f"run: gate={s['gate']} quality={s['quality_score']}/100 "
              f"(route={s['route']['pass']}p/{s['route']['review']}r, "
              f"{s['n_duplicate_groups']} dup groups, {s['n_label_errors']} label errors, "
              f"{s['n_leakage']} leakage, audit_verified={s['audit_verified']}, "
              f"backend={cfg.embedding_backend})")
        return 0 if s["gate"] == "GO" else 2

    elif args.cmd == "history":
        for r in pipeline.history(cfg, args.hash):
            print(f"{r['ts_utc']}  {r['event']}  {r.get('decision','')}  batch={r.get('batch_id','')}")

    elif args.cmd == "routing-diff":
        d = pipeline.routing_diff(cfg)
        for c in d["changed"]:
            print(f"{c['id']}: {c['from']} -> {c['to']}")
        print(d.get("note") or f"{d['n_changed']} decisions changed")
        if d.get("added") or d.get("removed"):
            print(f"  新增 {len(d.get('added', []))} 筆、消失 {len(d.get('removed', []))} 筆")

    elif args.cmd == "dismiss":
        n = pipeline.dismiss(adapter, cfg, args.ids)
        print(f"excluded {n} samples downstream (tag=rejected;可用 vix restore-dismissed 還原,已記稽核)")

    elif args.cmd == "restore-dismissed":
        n = pipeline.restore_dismissed(adapter, cfg, args.ids)
        print(f"restored {n} samples (rejected 標籤已移除,記為 undismiss)")

    elif args.cmd == "fp-rate":
        r = pipeline.false_positive_rate(cfg)
        print(f"reviewed={r['reviewed']} dismissed={r['dismissed_false_alarms']} fp_rate={r['fp_rate']:.1%}")

    elif args.cmd == "geometry":
        res = pipeline.geometry_check(adapter, cfg, args.from_tag, args.to_tag)
        print(f"geometry drift alert={res['alert']} shifts={res.get('shifts')}")

    elif args.cmd == "merge-preview":
        overrides = dict(o.split("=", 1) for o in args.override)
        if args.tag_a and args.tag_b:
            merged = pipeline.merge_preview_tags(adapter, cfg, args.tag_a, args.tag_b, overrides)
        elif args.counts_a and args.counts_b:
            counts_a = _load_json_arg(args.counts_a)
            counts_b = _load_json_arg(args.counts_b)
            merged = pipeline.merge_preview(counts_a, counts_b, overrides)
        else:
            raise SystemExit("provide either --tag-a/--tag-b or --counts-a/--counts-b")
        total = sum(merged.values()) or 1
        for c, n in sorted(merged.items(), key=lambda kv: -kv[1]):
            flag = "  ⚠️ <5%" if n / total < 0.05 else ""
            print(f"{c}: {n} ({n / total:.1%}){flag}")

    elif args.cmd == "quickstart":
        print(_QUICKSTART)

    elif args.cmd == "resolve":
        decision = "false_alarm" if args.false_alarm else "confirm"
        outcome = pipeline.resolve_review(adapter, cfg, args.hash, decision, args.label, args.reviewer)
        print(f"resolved {args.hash}: {outcome}")

    elif args.cmd == "sync-reviews":
        n = pipeline.sync_reviews(adapter, cfg)
        print(f"synced {n} review decisions from the App")

    elif args.cmd == "calibrate-confidence":
        rows = [json.loads(line) for line in Path(args.eval).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        res = pipeline.calibrate_confidence([r["conf"] for r in rows], [r["correct"] for r in rows], cfg)
        print(f"temperature={res['temperature']}  ECE {res['ece_before']} -> {res['ece_after']}")
        if len(rows) < 50:
            print(f"注:樣本過少({len(rows)}<50),溫度/ECE 估計不穩定,變化僅供參考")

    elif args.cmd == "label-noise":
        r = pipeline.label_noise(adapter, cfg)
        for k, v in sorted(r["noise_rates"].items(), key=lambda kv: -kv[1]):
            print(f"  noise {k}: {v:.1%}")
        for iss in r["issues"][:20]:
            print(f"  {iss.id}: given={iss.given_label} -> pred={iss.pred_label} (conf={iss.confidence:.2f})")
        print(f"{len(r['issues'])} label issues")
        print("注:這是 triage 啟發式(embedding kNN 多數票當偽標籤),非統計顯著的標籤錯誤率;用於排序優先複查")

    elif args.cmd == "drift-type":
        r = pipeline.drift_type(adapter, cfg, args.from_tag, args.to_tag)
        print(f"{r['verdict']}: cov={r['covariate_shift']} pred={r['prediction_shift']} | {r['recommended_action']}")
        print("注:以 DINOv2 embedding 為訊號;若僅解析度/檔案格式改變而視覺外觀一致可能不觸發 —"
              " 請搭配 `vix geometry` 與來源端 metadata 檢查。")

    elif args.cmd == "compare":
        r = pipeline.compare(adapter, cfg, args.tag_a, args.tag_b)
        for tag in r["tags"]:
            p = r["per_tag"][tag]
            print(
                f"[{tag}] 樣本={p['n_samples']} 偵測={p['n_detections']} "
                f"標籤雜訊={p['noise_pct']}% 疑似錯標={p['n_label_issues']} "
                f"近重複群={p['dup_groups']} 冗餘={p['redundant']}"
            )
        print(f"跨來源回收(tag-b 與 tag-a 近重複)= {r['cross_recycled']}")
        print("注:這是 triage 並排比較,非統計 A/B 檢定(無 p 值/檢定力);樣本太少時 reviewer-audit 會標示樣本不足")

    elif args.cmd == "set-threshold":
        r = pipeline.set_threshold(adapter, cfg, args.class_name, conf_thr=args.conf, dist_thr=args.dist)
        print(f"set-threshold {r['class']}: conf<{r['conf_thr']} dist>{r['dist_thr']} (手動覆寫,已記稽核)")

    elif args.cmd == "reasons":
        r = pipeline.reasons_breakdown(cfg)
        label = {"low_conf": "低信心", "far_from_known": "離已知太遠", "low_support": "支撐不足", "no_detection": "無偵測"}
        print(f"覆核總數={r['n_review']}  已排除(誤報/移除)={r['rejected']}")
        for reason, n in sorted(r["by_reason"].items(), key=lambda kv: -kv[1]):
            print(f"  {label.get(reason, reason)}: {n}")
        if "no_detection" in r["by_reason"]:
            print("注:「無偵測」一律送覆核;系統無法區分『真的沒有物件』與『模型漏報』,需人工確認")

    elif args.cmd == "throughput":
        r = pipeline.throughput(cfg)
        if r["median_hours"] is None:
            print(f"尚無已解決的覆核紀錄;目前待覆核 {r['n_open']} 筆")
        else:
            print(f"已解決 {r['n_resolved']} 筆:週轉中位數 {r['median_hours']:.1f}h、p90 {r['p90_hours']:.1f}h")
            print(f"待覆核 {r['n_open']} 筆 -> 估計約 {r['est_remaining_hours']:.1f} 人時(中位數×待辦,為估計非 SLA 保證)")

    elif args.cmd == "capacity":
        r = pipeline.capacity(cfg, volume=args.volume)
        m = f"{r['median_hours']:.1f}h" if r["median_hours"] is not None else "—"
        print(f"歷史 flag_rate={r['flag_rate']:.1%}  週轉中位數={m}  待覆核={r['n_open']}")
        if r["total_hours"] is not None:
            print(f"預估:積壓 {r['backlog_hours']:.1f}h + 新進 {args.volume}×{r['flag_rate']:.1%}={r['projected_review']} 筆 "
                  f"= 約 {r['total_hours']:.1f} 人時(估計,非 SLA 保證)")
        else:
            print("尚無足夠歷史(需已解決的覆核紀錄)來估算工時")

    elif args.cmd == "spc":
        series = pipeline.review_rate_series(adapter, cfg)
        r = pipeline.spc_monitor(series, args.target, args.sigma, method=args.method)
        print(f"series={[round(x, 3) for x in series]}")
        print(f"alarm={r['alarm']} at index {r['alarm_index']}")
        if r.get("short_series"):
            print("注:序列 < 8 批,控制界線僅供參考,需更多批次才能確認趨勢")

    elif args.cmd == "parity":
        r = pipeline.parity(adapter, cfg, by=args.by)
        for g, info in r["groups"].items():
            mark = "  ⚠️ 偏低" if info["worse"] else (" [樣本不足,僅供參考]" if info.get("low_confidence") else "")
            print(f"  {args.by}:{g} = {info['value']:.3f} (rel {info['rel_to_median']:+.1%}){mark}")
        print(f"median={r['median']:.3f} flagged={r['flagged']}")
        print("注:此處以平均信心作代理(非真實 CR/AP 顯著性檢定);新站點請先用該站自有 eval set 驗證")

    elif args.cmd == "cost-gate":
        r = pipeline.cost_gate_eval(cfg, args.cr, args.fa, args.miss_cost, args.fa_cost, args.budget)
        print(f"{r['verdict']}: expected_cost={r['expected_cost_per_unit']} "
              f"(miss {r['miss_component']} + fa {r['fa_component']}), budget={args.budget}")
        print("注:miss/fa 率需來自該站點有代表性的 eval set;沿用他站數字未經驗證")
        return 0 if r["verdict"] == "GO" else 2

    elif args.cmd == "verify-fiftyone":
        from .verification import run_headless

        return run_headless(cfg)

    elif args.cmd == "verify-gui":
        from .verification import run_gui

        return run_gui(cfg, execute=not args.no_execute)

    elif args.cmd == "new-classes":
        clusters = pipeline.new_classes(adapter, cfg, args.novelty_radius, args.cluster_distance)
        for c in clusters:
            print(f"{c['cluster']} ({c.get('size', len(c['ids']))} 張): {c['ids']}\n  {c['suggestion']}")
        print(f"{len(clusters)} suspected new-class clusters")

    elif args.cmd == "leakage":
        leaks = pipeline.leakage(adapter, cfg)
        for lk in leaks:
            print(f"leak across {lk['splits']}: {lk['ids']}")
        print(f"{len(leaks)} cross-split leakage groups")

    elif args.cmd == "harmful":
        if args.remove:
            ids = pipeline.harmful_remove(adapter, cfg, top=args.top, note=args.note)
            print(f"excluded {len(ids)} samples downstream (tag=rejected;可用 vix restore-dismissed 還原,已記稽核)")
        else:
            for r in pipeline.harmful(adapter, cfg, args.top):
                print(f"{r['id']}  harm={r['harm']:.3f}  {r['reasons']}")

    elif args.cmd == "trend":
        res = pipeline.quality_trend(adapter, cfg)
        for a in res["alerts"]:
            print(f"DROP {a['class']} @ {a['batch']} (-{a['drop']})")
        print(f"{len(res['alerts'])} drop alerts")

    elif args.cmd == "reviewer-audit":
        for rev, info in pipeline.reviewer_audit(adapter, cfg, class_filter=args.class_filter).items():
            flag = " [樣本不足,僅供參考]" if info.get("insufficient") else ""
            print(f"{rev}: consistency={info['intra_consistency']:.2f}, "
                  f"conflicts={len(info['conflicts'])}, n={info.get('n_decisions', 0)}{flag}")
        print("注:本指標僅量測『自我一致性』(同一人對近乎相同的樣本是否給相同決策),"
              "無法證明某次『確認』是否真的有人看過。")

    elif args.cmd == "gate":
        r = pipeline.pre_train_gate_stage(adapter, cfg)
        from .core.decision_log import DecisionLog

        dl = DecisionLog(cfg.decision_log_path)
        chain_ok, truncated = dl.verify_chain(), dl.is_truncated()
        print(f"{r.verdict}: {r.reasons or 'all checks passed'}")
        print(f"audit integrity: 鏈結={'OK' if chain_ok else 'FAIL'}  "
              f"尾端錨點={'FAIL(偵測到截斷)' if truncated else 'OK'}")
        return 0 if r.verdict == "GO" else 2

    elif args.cmd == "explain":
        print(json.dumps(pipeline.explain_one(adapter, cfg, args.hash), ensure_ascii=False, indent=2))

    elif args.cmd == "verify":
        res = pipeline.verify_dataset(cfg, args.manifest, args.data_dir)
        print(f"ok={res['ok']} checked={res['n_checked']} "
              f"mismatched={res['mismatched']} missing={res['missing']} "
              f"unexpected={res.get('unexpected', [])}")
        return 0 if res["ok"] else 2

    elif args.cmd == "eval-ingest":
        r = pipeline.eval_ingest(adapter, cfg, args.results, iou_thr=args.iou)
        print(f"mAP@{r['iou_thr']}={r['mAP']}")
        if r.get("loc_gap"):
            mbi = r.get("map_by_iou", {})
            print(f"  定位尾巴 loc_gap={r['loc_gap']}(mAP@0.5={mbi.get(0.5)} vs @0.75={mbi.get(0.75)};框越鬆此值越大)")
        for c, ap in sorted(r["per_class_ap"].items(), key=lambda kv: kv[1]):
            print(f"  AP {c}: {ap}")
        if r["confusion"]:
            print("混淆 (truth->pred):")
            for pair, n in list(r["confusion"].items())[:10]:
                print(f"  {pair}: {n}")
        print(f"{len(r['fn_hashes'])} 張漏報(FN)、{len(r['fp_hashes'])} 張誤報(FP);"
              "用 vix error-mine 反查最該標的候選")

    elif args.cmd == "error-mine":
        ranked = pipeline.error_mine(adapter, cfg, top=args.top)
        for r in ranked:
            print(f"{r['id']}  closeness={r['closeness']}  {r['why']}")
        if not ranked:
            print("無候選(需先 eval-ingest,且未標註候選需有 embedding)")
        pipeline._log_queue(cfg, "error_mine", [r["id"] for r in ranked], "label")  # for queue-hit-rate

    elif args.cmd == "set-eval-baseline":
        protected = {c: args.protect_drop for c in args.protect}
        b = pipeline.set_eval_baseline(adapter, cfg, protected=protected, map_drop_thr=args.map_drop)
        print(f"baseline mAP={b['mAP']} 已凍結(eval_set_hash={b['eval_set_hash']})")
        print(f"  保護類別={list(protected) or '(無)'};整體 mAP 掉> {args.map_drop} 或保護類別 AP 掉> {args.protect_drop} → 下次 gate NO-GO")

    elif args.cmd == "box-qa":
        issues = pipeline.box_qa(adapter, cfg, top=args.top)
        for it in issues:
            print(f"{it['id']}  [{it['issue']}] {it['label']}: {it['why']}")
        if not issues:
            print("無框品質問題(或 golden 框數不足以建立各類包絡)")

    elif args.cmd == "hardneg":
        r = pipeline.hardneg(adapter, cfg, top=args.top, mode=args.mode)
        print(f"hardneg 模式={r['mode']}({len(r['rows'])} 筆「自信卻錯」,高→低):")
        for row in r["rows"]:
            print(f"  {row['id']}  {row.get('pred_class')}  wrongness={row['wrongness']}  {row['why']}")
        if not r["rows"]:
            print("  無(GT 模式需先 eval-ingest;GT-free 需 calibrate 且有未標註偵測)")
        pipeline._log_queue(cfg, "hardneg", [row["id"] for row in r["rows"]], "wrong")  # for queue-hit-rate

    elif args.cmd == "weakness-report":
        r = pipeline.weakness_report(adapter, cfg, top_classes=args.top_classes,
                                     queue_per_class=args.queue_per_class, out_path=args.out)
        d = r["data"]
        print(f"YOLO 弱點報告({d['mode']} 模式)-> {r['path']}")
        if d.get("mAP") is not None:
            print(f"  mAP@0.5={d['mAP']}  loc_gap={d.get('loc_gap')}")
        for row in d["per_class"][:args.top_classes]:
            print(f"  弱類 {row['cls']}: AP={row['ap']} 漏報型態={row.get('dom_fn_type') or '-'} "
                  f"佇列={len(d['queue'].get(row['cls'], []))} 候選")
        print("  註:未重訓 → 佇列是 PROXY 優先排序,非實測 mAP 增益")

    elif args.cmd == "consistency":
        r = pipeline.consistency(adapter, cfg, max_pairs=args.max_pairs)
        print(f"一致性歸因({r['n_classes']} 類,has_eval={r['has_eval']},{len(r['findings'])} 個易混對):")
        for f in r["findings"]:
            c = f"{f['C_ij']}" if f.get("C_ij") is not None else "-"
            print(f"  {f['pair'][0]}↔{f['pair'][1]}  [{f['verdict']}] 可分={f['separable_in_embedding']} "
                  f"sep_err={f['sep_err']}{f.get('sep_ci')} O={f.get('O_ij')} C={c} ({f['tier']})")
            print(f"      → {f['action']}")
        if not r["findings"]:
            print("  無(需 ≥2 類 golden;接 eval-ingest 才能歸因 taxonomy/model/label)")
        print("  註:諮詢式;可分性綁定目前 embedding 空間;小樣本→insufficient_support,絕不自動 merge")

    elif args.cmd == "adapt-embedding":
        r = pipeline.adapt_embedding(adapter, cfg, save=args.save, enable=args.enable, max_pca=args.max_pca)
        g = r["gate"]
        print(f"領域自適應投影:{len(r['classes'])} 類 → {r['out_dim']}d;{r['n_rescued']}/{len(r['pairs'])} 對被「救回」(凍結不可分→投影後可分)")
        for p in r["pairs"]:
            tag = "✅救回(表徵問題,可修)" if p["rescued"] else ("↓改善" if p["delta"] > 0.05 else "—")
            print(f"  {p['pair'][0]}↔{p['pair'][1]}: 凍結 sep_err={p['frozen_sep_err']} → 投影 {p['adapted_sep_err']} (Δ{p['delta']}, n_min={p['n_min']})  {tag}")
        print(f"  Gate:{'GO ✅' if g['go'] else 'NO-GO ❌'}(macro {g.get('macro_frozen')}→{g.get('macro_adapted')},退步 {g.get('n_regressed')} 對){' '+';'.join(g['reasons']) if g['reasons'] else ''}")
        print(f"  {'已存投影+報告' if r['saved'] else '(未存;加 --save)'};啟用={r['enabled']}{'(gate NO-GO 故未啟用)' if (args.enable and not r['enabled']) else ''};CV 量測、離線、非訓練 YOLO")

    elif args.cmd == "bank-audit":
        r = pipeline.bank_audit(
            adapter, cfg, defect_tag=args.defect_tag, reflection_tag=args.reflection_tag,
            normal_tag=args.normal_tag, conf_lo=args.conf_lo, conf_hi=args.conf_hi,
            tau=args.tau, novelty_radius=args.novelty_radius,
            dedup_distance=args.dedup_distance, top=args.top,
        )
        print(f"banks={r['banks']} fingerprint={r['fingerprint']}")
        print(f"{r['n_proposals']} low-conf proposals -> {r['counts']}")
        for row in r["results"]:
            print(f"  {row['id']}  conf={row['conf']}  {row['verdict']}  "
                  f"(bank={row['winning_bank']}, margin={row['margin']})")
        print("注:bank_verdict 為諮詢欄位(不覆寫 route);defect_like/unknown 已標 hard_positive"
              "(人工 resolve→golden,不自動晉升)")
        pipeline._log_queue(cfg, "bank_hardpos",  # for queue-hit-rate (predict: these are defects)
                            [row["id"] for row in r["results"] if row.get("verdict") in ("defect_like", "unknown")], "defect")

    elif args.cmd == "queue-hit-rate":
        r = pipeline.queue_hit_rate(cfg, min_resolved=args.min_resolved)
        print(f"佇列命中率({r['n_emissions']} 次發出,{r['n_resolutions']} 次裁決):")
        for q in r["queues"]:
            prec = "-" if q["precision"] is None else q["precision"]
            note = " [樣本不足僅供參考]" if q["insufficient"] else ""
            print(f"  {q['queue']}(預測={q['predict']}): 命中率 {prec}  已解決 {q['resolved']}/{q['emitted']}  趨勢 {q['trend']}{note}")
        if not r["queues"]:
            print("  無資料(先跑 error-mine/hardneg/weakness-report 發出佇列,並有 resolve/dismiss 裁決)")
        print("  註:只算『已解決』的 id(誠實);label 佇列命中率=被採納率;趨勢上升=佇列越來越準")

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
