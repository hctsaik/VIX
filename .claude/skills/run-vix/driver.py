#!/usr/bin/env python
"""VIX run-driver: restart the FiftyOne App server, drive it via Playwright, run the full E2E.

Why this exists: `vix app` launches ONE long-running FiftyOne server (`session.wait()`). FiftyOne
hot-reloads the plugin file but CACHES `import vix.adapters...` / `vix.pipeline` in that process, so
after you edit adapter/pipeline/core the running App keeps the OLD module → "object has no attribute
…" and other stale-state bugs. The only reliable fix is to RESTART the server process. This driver
makes that one command, plus a one-command full Playwright E2E so a GUI change is actually proven.

Run with the project venv from the repo root:
  .venv311/Scripts/python.exe .claude/skills/run-vix/driver.py restart [--dataset vix] [--port 5151]
  .venv311/Scripts/python.exe .claude/skills/run-vix/driver.py screenshot [--out shot.png] [--port 5199]
  .venv311/Scripts/python.exe .claude/skills/run-vix/driver.py e2e [-k EXPR]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]          # <repo>/.claude/skills/run-vix/driver.py -> <repo>
PY = REPO / ".venv311" / "Scripts" / "python.exe"   # the Tier-2 venv (FiftyOne + Playwright live here)
PLUGINS = REPO / "src" / "vix" / "plugins"


def _env() -> dict:
    e = os.environ.copy()
    e["FIFTYONE_PLUGINS_DIR"] = str(PLUGINS)         # so the @vix/review operators + panels load
    e.setdefault("FIFTYONE_DO_NOT_TRACK", "true")
    return e


def _pids_on_port(port: int) -> set[int]:
    import psutil
    pids: set[int] = set()
    for c in psutil.net_connections(kind="inet"):
        if c.laddr and c.laddr.port == port and c.status == psutil.CONN_LISTEN and c.pid:
            pids.add(c.pid)
    return pids


def _kill_port(port: int) -> list[int]:
    """Terminate whatever is LISTENing on `port` (the App server). Returns the PIDs killed."""
    import psutil
    killed = []
    for pid in _pids_on_port(port):
        try:
            p = psutil.Process(pid)
            p.terminate()
            killed.append(pid)
        except Exception:  # noqa: BLE001
            pass
    if killed:
        time.sleep(2)
        for pid in list(killed):
            try:
                if psutil.pid_exists(pid):
                    psutil.Process(pid).kill()       # SIGKILL the stragglers
            except Exception:  # noqa: BLE001
                pass
    return killed


def _wait_ready(port: int, timeout: int = 90) -> bool:
    url = f"http://localhost:{port}"
    for _ in range(timeout):
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:  # noqa: BLE001
            time.sleep(1)
    return False


def cmd_restart(args) -> int:
    """Kill the App server on the port, relaunch the named dataset on it, DETACHED (survives this exit)."""
    killed = _kill_port(args.port)
    print(f"killed {killed or 'nothing'} on :{args.port}")
    launcher = (
        "import fiftyone as fo;"
        f"ds=fo.load_dataset({args.dataset!r});"
        f"s=fo.launch_app(ds, port={args.port}, remote=True);"
        "s.wait()"
    )
    flags = 0
    if os.name == "nt":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | 0x00000008  # DETACHED_PROCESS
    subprocess.Popen([str(PY), "-c", launcher], cwd=str(REPO), env=_env(),
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     stdin=subprocess.DEVNULL, creationflags=flags, close_fds=True)
    ok = _wait_ready(args.port)
    print(f"restarted dataset={args.dataset!r} on http://localhost:{args.port}  ready={ok}")
    return 0 if ok else 1


def _build_verify_dataset():
    """The self-contained demo dataset the browser tests use (synthetic, pixel-fallback embeddings)."""
    import fiftyone as fo
    from vix import pipeline, verification as V
    from vix.adapters.fiftyone_adapter import FiftyOneAdapter
    from vix.config import Config
    cfg = Config()
    cfg.embedding_backend = "pixel_fallback"
    ds = V._build_dataset(fo)
    ad = FiftyOneAdapter(cfg, dataset_name=V.DATASET)
    pipeline.calibrate(ad, cfg)
    pipeline.route(ad, cfg)
    ds.reload()
    return ds


def cmd_screenshot(args) -> int:
    """Launch the App + drive it with Playwright + save a screenshot (the GUI-interaction proof)."""
    os.environ["FIFTYONE_PLUGINS_DIR"] = str(PLUGINS)
    os.environ.setdefault("FIFTYONE_DO_NOT_TRACK", "true")
    import fiftyone as fo
    from playwright.sync_api import sync_playwright

    ds = _build_verify_dataset()
    session = fo.launch_app(ds, remote=True, port=args.port)
    if not _wait_ready(args.port):
        print("app did not become ready", file=sys.stderr)
        return 1
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1600, "height": 1000})
        page.goto(f"http://localhost:{args.port}", wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(12000)                 # grid render (the FiftyOne websocket never idles)
        n_buttons = len(page.query_selector_all("img[src*='.svg']"))
        page.screenshot(path=str(out))
        browser.close()
    session.close()
    print(f"screenshot -> {out}  (vix toolbar svg imgs seen: {n_buttons})")
    return 0


def cmd_e2e(args) -> int:
    """The FULL Playwright E2E: real-browser suite + handler-anchored live suite. Exit code = pass/fail."""
    cmd = [str(PY), "-m", "pytest", "tests/test_gui_browser.py", "tests/test_gui_e2e.py",
           "-p", "no:cacheprovider", "-q"]
    if args.k:
        cmd += ["-k", args.k]
    print("running:", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser(description="VIX App run-driver (restart / screenshot / e2e)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("restart", help="kill the App server on the port and relaunch it (clears stale modules)")
    r.add_argument("--dataset", default="vix")
    r.add_argument("--port", type=int, default=5151)

    s = sub.add_parser("screenshot", help="launch the App + Playwright screenshot to disk (GUI proof)")
    s.add_argument("--out", default=str(REPO / "_artifacts" / "vix_app.png"))
    s.add_argument("--port", type=int, default=5199)

    e = sub.add_parser("e2e", help="run the full Playwright E2E suite (browser + handler-anchored)")
    e.add_argument("-k", default=None, help="pytest -k expression to run a subset")

    args = ap.parse_args()
    return {"restart": cmd_restart, "screenshot": cmd_screenshot, "e2e": cmd_e2e}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
