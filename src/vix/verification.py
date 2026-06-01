"""Tier-2 自我驗證(供 `vix verify-fiftyone` / `vix verify-gui` 呼叫)。

fiftyone / playwright 都在函式內 lazy import —— 所以本模組可被打包,但只有實際執行
驗證指令時才需要 Tier-2 依賴;核心套件與 pytest(可能在無 fiftyone 的環境)不受影響。
"""

from __future__ import annotations

import os
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

from .config import Config
from .embedding.simple import pixel_embedding

DATASET = "vix_verify"
PORT = 5151
URL = f"http://localhost:{PORT}"


def _png(path: Path, kind: str, split: int) -> None:
    a = np.zeros((96, 96, 3), np.uint8)
    if kind == "vert":
        a[:, : split * 3] = 255
    else:
        a[: split * 3, :] = 255
    Image.fromarray(a).save(path)


def _build_dataset(fo):
    """Build a real FiftyOne dataset (no GPU/YOLO/DINOv2 needed; pixel-fallback embeddings)."""
    imgs = Path(tempfile.mkdtemp(prefix="vix_verify_"))
    if fo.dataset_exists(DATASET):
        fo.delete_dataset(DATASET)
    ds = fo.Dataset(DATASET, persistent=True)
    samples = []

    def make(name, kind, split, label, conf, tags):
        p = imgs / f"{name}.png"
        _png(p, kind, split)
        det = fo.Detection(label=label, bounding_box=[0.0, 0.0, 1.0, 1.0], confidence=conf)
        det["dino_embedding"] = pixel_embedding(str(p), size=8).tolist()
        s = fo.Sample(filepath=str(p), tags=list(tags))
        s["vix_hash"] = name
        s["yolo_detections"] = fo.Detections(detections=[det])
        samples.append(s)

    for i in range(8):
        make(f"vert{i}", "vert", 8 + i, "vert", 0.9, ["golden"] + (["anchor"] if i < 2 else []))
        make(f"horiz{i}", "horiz", 8 + i, "horiz", 0.9, ["golden"] + (["anchor"] if i < 2 else []))
    make("cand_low", "vert", 11, "vert", 0.05, ["review"])
    make("rev1", "vert", 11, "vert", 0.3, ["review"])
    make("rev2", "horiz", 28, "vert", 0.3, ["review"])
    ds.add_samples(samples)
    return ds


def _ready(timeout=60) -> bool:
    for _ in range(timeout):
        try:
            urllib.request.urlopen(URL, timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False


def run_headless(cfg: Config) -> int:
    """Headless Tier-2 驗證:FiftyOneAdapter 全鏈 + sync_reviews 回寫閉環。"""
    try:
        import fiftyone as fo
    except ImportError:
        print("verify-fiftyone 需要 FiftyOne:在 Python 3.10/3.11 執行 pip install -e \".[fiftyone]\"")
        return 1
    from . import pipeline
    from .adapters.fiftyone_adapter import FiftyOneAdapter
    from .core.decision_log import DecisionLog

    cfg.ensure_dirs()
    cfg.embedding_backend = "pixel_fallback"
    os.environ["VIX_WORKSPACE"] = str(cfg.workspace.resolve())  # App/CLI 共用 workspace
    print(f"# FiftyOne {fo.__version__} | workspace {cfg.workspace.resolve()}")
    ds = _build_dataset(fo)
    adapter = FiftyOneAdapter(cfg, dataset_name=DATASET)

    results: list[tuple[str, bool]] = []

    def chk(name, cond, detail=""):
        results.append((name, bool(cond)))
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}")

    rows = list(adapter.samples())
    chk("adapter.samples 讀回全部", len(rows) == 19, f"{len(rows)} samples")  # 16 golden + cand_low + rev1 + rev2
    chk("偵測 embedding 讀得回", all(d.embedding is not None for _h, _s, dets, _t in rows for d in dets))
    pol = pipeline.calibrate(adapter, cfg)
    chk("calibrate per-class 門檻", {"vert", "horiz"} <= set(pol.thresholds))
    counts = pipeline.route(adapter, cfg, pol)
    chk("route 寫回 pass/review", (counts["pass"] + counts["review"]) >= 2, str({k: counts[k] for k in ("pass", "review")}))
    chk("get_by_tag('golden')", len(list(adapter.get_by_tag("golden"))) == 16)
    rep, paths = pipeline.health_report(adapter, cfg, cfg.workspace / "report")
    chk("health_report + 品質分數", Path(paths["md"]).exists(), f"score={rep.get('quality_score')}")
    for h, dec in (("rev1", "vert"), ("rev2", "false_alarm")):
        s = adapter._sample(h)
        s["review_decision"] = dec
        s.save()
    pipeline.sync_reviews(adapter, cfg)
    chk("sync_reviews: rev1 -> golden", "golden" in adapter._sample("rev1").tags)
    chk("sync_reviews: rev2 -> rejected", "rejected" in adapter._sample("rev2").tags)
    chk("decision_log hash-chain", DecisionLog(cfg.decision_log_path).verify_chain())

    fo.delete_dataset(DATASET)
    ok = all(c for _n, c in results)
    print(f"=== {'全部 PASS' if ok else '有 FAIL'} ({sum(c for _n, c in results)}/{len(results)}) ===")
    return 0 if ok else 1


def run_gui(cfg: Config, execute: bool = True) -> int:
    """GUI Tier-2 驗證:Playwright 驅動 FiftyOne App,截圖並(可選)實際執行 confirm_golden operator。"""
    # 必須在 import fiftyone 之前設好,否則 fiftyone config 已快取、plugin 掃不到
    os.environ["FIFTYONE_PLUGINS_DIR"] = str(Path(__file__).resolve().parent / "plugins")
    os.environ.setdefault("FIFTYONE_DO_NOT_TRACK", "true")
    os.environ["VIX_WORKSPACE"] = str(cfg.workspace.resolve())  # operator 與 CLI 共用 workspace
    try:
        import fiftyone as fo
    except ImportError:
        print("verify-gui 需要 FiftyOne:pip install -e \".[fiftyone]\"")
        return 1
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("verify-gui 需要 Playwright:pip install playwright && playwright install chromium")
        return 1

    cfg.ensure_dirs()
    shots = cfg.workspace / "gui_shots"
    shots.mkdir(parents=True, exist_ok=True)

    ds = _build_dataset(fo)
    rev = ds.match({"vix_hash": "rev1"}).first()
    try:
        import fiftyone.plugins as fop

        print("plugins discovered:", [p.name for p in fop.list_plugins()] or "(none)")
    except Exception as exc:  # noqa: BLE001
        print("plugins listing skipped:", exc)

    session = fo.launch_app(ds, remote=True, port=PORT)
    ok = True
    try:
        if not _ready():
            print("FAIL: App server 未就緒")
            return 1
        import re

        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1600, "height": 1000})
            # FiftyOne App 有常駐 websocket,網路永不 idle ->「networkidle」會卡到 timeout(CI 必掛)。
            # 改等 DOM 載完即可,再用固定等待讓格狀算完(CI runner 較慢)。
            page.goto(URL, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(12000)  # CI runners are slower; give the grid time to render
            page.screenshot(path=str(shots / "app.png"), full_page=True)
            print(f"[OK] App 截圖 -> {shots / 'app.png'}")
            session.selected = [rev.id]  # client 已連上,程式化選取
            page.wait_for_timeout(2000)
            page.keyboard.press("`")
            page.wait_for_timeout(1500)
            page.keyboard.type("confirm_golden")
            page.wait_for_timeout(1500)
            page.screenshot(path=str(shots / "operators.png"))
            if execute:
                page.keyboard.press("Enter")
                page.wait_for_timeout(2500)
                btn = page.get_by_role("button", name=re.compile("execute|run|執行|送出", re.I))
                (btn.first.click() if btn.count() else page.keyboard.press("Enter"))
                page.wait_for_timeout(3500)
                page.screenshot(path=str(shots / "after_execute.png"))
            browser.close()
    finally:
        session.close()

    if execute:
        tags = []
        for _ in range(15):  # poll for the operator's effect (robust to CI timing)
            ds.reload()
            tags = ds.match({"vix_hash": "rev1"}).first().tags
            if "golden" in tags:
                break
            time.sleep(1)
        ok = "golden" in tags
        print(f"[{'PASS' if ok else 'FAIL'}] GUI 執行 confirm_golden -> rev1 tags={tags}")
    fo.delete_dataset(DATASET)
    print(f"=== {'GUI 驗證 PASS' if ok else 'GUI 驗證 FAIL'} (截圖在 {shots}) ===")
    return 0 if ok else 1
