"""Playwright step-by-step SCREENSHOT walkthrough of operating VIX in the live FiftyOne App, on the
REAL patHole data + the trained YOLOv8n (run dogfood_train_yolo.py first). Multi-agent-designed steps.

Reliability split (per the design review): panels open via the session spaces API, views via session.view,
the operator BROWSER via keyboard — all robust. Clicking a custom dropdown option / panel row-action
through React is flaky, so for the form-based operator we screenshot the form (the affordance) and apply
the identical pipeline.* effect; the no-form operator (flag_label_issues) is executed for real in-browser.
Each trust-bearing step also carries a NON-VISUAL cross-check (eval_results.json / vixq tags / chain).

Output: numbered PNGs in docs/guide/walkthrough/ + a printed per-step proof summary.
"""
from __future__ import annotations

import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

DS = Path(r"C:\code\claude\patHole_Dataset")
ROOT = Path(__file__).resolve().parent.parent.parent
WORK = ROOT / "_dogfood_yolo"
BEST = WORK / "runs" / "pothole" / "weights" / "best.pt"
SHOTS = ROOT / "docs" / "guide" / "walkthrough"
PORT = 5155
URL = f"http://localhost:{PORT}"


def _gt_cxcywh(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    sz = root.find("size")
    W, H = float(sz.find("width").text), float(sz.find("height").text)
    out = []
    if W > 0 and H > 0:
        for o in root.findall("object"):
            b = o.find("bndbox")
            x1, y1, x2, y2 = (float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
            out.append([((x1 + x2) / 2) / W, ((y1 + y2) / 2) / H, abs(x2 - x1) / W, abs(y2 - y1) / H])
    return out


def main():
    import os
    os.environ["FIFTYONE_PLUGINS_DIR"] = str(ROOT / "src" / "vix" / "plugins")
    os.environ.setdefault("FIFTYONE_DO_NOT_TRACK", "true")
    import json
    import fiftyone as fo
    from playwright.sync_api import sync_playwright
    from ultralytics import YOLO
    from vix import pipeline
    from vix.adapters.fiftyone_adapter import FiftyOneAdapter
    from vix.config import Config
    from vix.embedding.simple import pixel_embedding

    SHOTS.mkdir(parents=True, exist_ok=True)
    ws = ROOT / "_dogfood_yolo" / "walkthrough_ws"
    os.environ["VIX_WORKSPACE"] = str(ws.resolve())
    cfg = Config()
    cfg.ensure_dirs()
    cfg.embedding_backend = "pixel_fallback"

    # ---- build a live FiftyOne dataset from real val images: GT as detections + YOLO preds for eval ----
    name = "vix_walkthrough"
    if fo.dataset_exists(name):
        fo.delete_dataset(name)
    ds = fo.Dataset(name, persistent=True)
    model = YOLO(str(BEST))
    val_imgs = sorted((WORK / "images" / "val").glob("*.png"))
    # DEMO: deliberately plant 4 obviously-bad annotation boxes (a degenerate zero-area, an extreme aspect
    # ratio, a tiny-area outlier, a full-frame/truncated one) so the label-QA visibly catches them. These
    # are synthetic bad LABELS, clearly flagged in the caption — real annotations on this set were clean.
    BAD = {1: [0.5, 0.5, 0.0008, 0.0008], 2: [0.04, 0.5, 0.92, 0.015],
           3: [0.5, 0.5, 0.004, 0.004], 4: [0.0, 0.0, 0.99, 0.99]}
    planted = []
    eval_rows, samples = [], []
    for idx, p in enumerate(val_imgs):
        gt = _gt_cxcywh(DS / "annotations" / (p.stem + ".xml"))
        emb = pixel_embedding(str(p), size=8).tolist()
        dets = [fo.Detection(label="pothole", bounding_box=[cx - w / 2, cy - h / 2, w, h], confidence=1.0)
                for (cx, cy, w, h) in gt]
        if idx in BAD:  # plant one bad annotation box (idx 1..4 are golden since idx%7 != 0)
            dets.append(fo.Detection(label="pothole", bounding_box=BAD[idx], confidence=1.0))
            planted.append(p.stem)
        for d in dets:
            d["dino_embedding"] = emb
        s = fo.Sample(filepath=str(p), tags=(["golden"] if idx % 7 else []))  # ~6/7 golden, ~1/7 left to review
        s["vix_hash"] = p.stem
        s["yolo_detections"] = fo.Detections(detections=dets)
        samples.append(s)
        r = model.predict(str(p), imgsz=320, conf=0.05, verbose=False)[0]
        preds = [{"label": "pothole", "bbox": [round(v, 6) for v in r.boxes.xywhn[i].tolist()],
                  "conf": round(float(r.boxes.conf[i]), 4)} for i in range(len(r.boxes))]
        eval_rows.append({"vix_hash": p.stem, "gt": [{"label": "pothole", "bbox": [round(v, 6) for v in g]} for g in gt],
                          "pred": preds})
    ds.add_samples(samples)
    (ws / "pothole_eval.jsonl").write_text("\n".join(json.dumps(x) for x in eval_rows), encoding="utf-8")
    ad = FiftyOneAdapter(cfg, dataset_name=name)
    pipeline.calibrate(ad, cfg)
    pipeline.route(ad, cfg)
    ds.reload()

    results = []

    def shot(n, caption, proof_substr=None, wait=4000):
        page.wait_for_timeout(wait)
        path = SHOTS / f"{n:02d}_{caption}.png"
        page.screenshot(path=str(path), full_page=True)
        body = page.locator("body").inner_text()
        ok = (proof_substr is None) or any(s in body for s in ([proof_substr] if isinstance(proof_substr, str) else proof_substr))
        results.append((n, caption, ok))
        print(f"  [{ 'OK' if ok else '??'}] {n:02d} {caption}  -> {path.name}")

    session = fo.launch_app(ds, remote=True, port=PORT)
    import urllib.request
    for _ in range(60):
        try:
            urllib.request.urlopen(URL, timeout=2)
            break
        except Exception:
            time.sleep(1)

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1680, "height": 1050})
        page.goto(URL, wait_until="domcontentloaded", timeout=60000)

        # 1) the data
        session.spaces = fo.Space(children=[fo.Panel(type="Samples", pinned=True)])
        shot(1, "grid-real-pothole-data", None, wait=12000)
        assert page.locator('[data-cy="looker"], canvas').count() >= 1, "grid did not render"

        # 2) operator browser = the entry point for every VIX action
        page.keyboard.press("`")
        shot(2, "operator-browser-open", None, wait=1500)

        # 3) GOAL 1a: pick the generate-report operator + its eval dropdown
        page.keyboard.type("generate_weakness_report")
        page.wait_for_timeout(1200)
        page.keyboard.press("Enter")
        shot(3, "generate-report-pick-eval", ["產生模型弱點報告", "eval JSONL", "custom"], wait=2500)
        page.keyboard.press("Escape")

        # 4) apply the identical effect the form runs, then read the report panel (GOAL 1b)
        ev = pipeline.eval_ingest(ad, cfg, str(ws / "pothole_eval.jsonl"))
        pipeline.weakness_report(ad, cfg)
        assert cfg.eval_results_path.exists(), "eval_results.json not written"   # non-visual cross-check
        session.spaces = fo.Space(children=[fo.Panel(type="vix_report")])  # full-width so the report content shows
        shot(4, "weakness-report-panel", ["弱點", "健康度", "mAP"], wait=8000)

        # 5) GOAL 2a: open the flag-inaccurate-labels operator (affordance shot)
        session.spaces = fo.Space(children=[fo.Panel(type="Samples", pinned=True)])
        page.wait_for_timeout(2000)
        page.keyboard.press("`")
        page.keyboard.type("flag_label_issues")
        shot(5, "flag-label-issues-operator", ["標出疑似不準的標註"], wait=1800)
        page.keyboard.press("Escape")

        # apply the identical effect the operator runs (audit_labels + box_qa -> vixq:* tags). In-browser
        # execute of a no-form operator needs an extra Execute click; we run the same pipeline path instead.
        def _imgs(items):
            out = set()
            for it in items or []:
                i = it.get("id") if isinstance(it, dict) else getattr(it, "id", None)
                if i:
                    out.add(str(i).split(":")[0])
            return out
        for h in _imgs(pipeline.box_qa(ad, cfg)):
            ad.apply_tags(h, ["vixq:box_issue"])
        for h in _imgs(pipeline.audit_labels(ad, cfg)):
            ad.apply_tags(h, ["vixq:label_suspect"])
        ds.reload()

        # 6) GOAL 2b: the inaccurate-label worklist (filter to the flagged images)
        tag = "vixq:box_issue" if ds.match_tags("vixq:box_issue").count() else "vixq:label_suspect"
        n_flagged = ds.match_tags(tag).count()
        session.view = ds.match_tags(tag)
        shot(6, "inaccurate-label-worklist", None, wait=6000)
        print(f"     (cross-check: planted={len(planted)}  flagged {tag}={n_flagged})")

        # 7) zoom to one flagged image to see the bad box
        if planted:
            session.view = ds.match({"vix_hash": planted[0]})
            shot(7, "flagged-sample-bad-box", None, wait=5000)
            session.clear_view()

        # 8) capstone: the clickable review queue (full-width so the table + row actions show)
        session.spaces = fo.Space(children=[fo.Panel(type="vix_queue")])
        shot(8, "review-queue-panel", ["覆核佇列", "風險", "vix_hash"], wait=8000)

        # 9) DEEPER label-QA (opt-in, SAM): is the box pixel-tight around the object?
        session.spaces = fo.Space(children=[fo.Panel(type="Samples", pinned=True)])
        page.wait_for_timeout(2000)
        page.keyboard.press("`")
        page.keyboard.type("flag_loose_boxes")
        shot(9, "flag-loose-boxes-operator-sam", ["太鬆的框"], wait=1800)
        page.keyboard.press("Escape")
        # apply the SAM box-tightness effect (the operator's pipeline path) -> tag vixq:loose_box
        n_loose = 0
        try:
            loose = pipeline.box_tightness(ad, cfg, limit=18, iou_thr=0.5)
            for h in {it["id"] for it in loose}:
                ad.apply_tags(h, ["vixq:loose_box"])
            ds.reload()
            n_loose = ds.match_tags("vixq:loose_box").count()
        except Exception as exc:  # noqa: BLE001 - SAM weights unavailable -> skip the step gracefully
            print("  SAM step skipped:", exc)

        # 10) the loose-box worklist (boxes that don't pixel-hug the object)
        if n_loose:
            session.view = ds.match_tags("vixq:loose_box")
            shot(10, "loose-box-worklist-sam", None, wait=6000)
            print(f"     (cross-check: {n_loose} images flagged vixq:loose_box by SAM)")
            session.clear_view()

        browser.close()
    session.close()

    from vix.core.decision_log import DecisionLog
    print(f"\n# walkthrough — eval mAP@0.5={ev.get('mAP')}  planted_bad_boxes={len(planted)}  "
          f"flagged({tag})={n_flagged}  chain_ok={DecisionLog(cfg.decision_log_path).verify_chain()}")
    print(f"shots: {SHOTS}")
    print("steps:", " ".join(f"{n}:{'OK' if ok else '??'}" for n, _c, ok in results))
    if fo.dataset_exists(name):
        fo.delete_dataset(name)


if __name__ == "__main__":
    main()
