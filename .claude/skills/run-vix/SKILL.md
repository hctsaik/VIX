---
name: run-vix
description: Run, restart, screenshot, and Playwright-E2E-test the VIX FiftyOne review App. Use when editing VIX and you need to relaunch the App server so it picks up code changes (it caches imported modules), or to verify a GUI/plugin change end-to-end through a real browser. Triggers: run vix, restart the app server, launch the fiftyone app, screenshot the app, run the e2e / playwright tests, "按了沒反應", stale module / "object has no attribute".
---

# run-vix — drive the VIX FiftyOne App

VIX's GUI is a FiftyOne App (a local web server) plus the `@vix/review` plugin
(`src/vix/plugins/vix_review/`). `vix app` launches **one long-running server**
(`session.wait()` blocks). The driver is **`.claude/skills/run-vix/driver.py`** — it
restarts that server, screenshots the live App via Playwright, and runs the full E2E.

All paths below are relative to the repo root (`<unit>` = the VIX repo). The Tier-2
toolchain (FiftyOne + Playwright + Mongo) lives in **`.venv311`** — always use it.

## TWO RULES this skill exists to enforce

1. **After editing any non-plugin Python (`adapters/`, `pipeline.py`, `core/`), RESTART the
   App server.** FiftyOne hot-reloads the plugin file but **caches `import vix.adapters…` /
   `vix.pipeline`** in the running process, so the App keeps the OLD module → `'FiftyOneAdapter'
   object has no attribute …` and other stale-state bugs. Closing the browser tab does NOT fix
   it — the server *process* must restart.
2. **Verify GUI changes with the FULL Playwright E2E**, not a partial handler test. A handler
   test that pre-builds state proves the happy path, not what the user clicks.

## Prerequisites

The venv is already provisioned. Confirm the Tier-2 stack is importable:

```bash
.venv311/Scripts/python.exe -c "import psutil, playwright, fiftyone; print('ok', fiftyone.__version__)"
```

(Expect `ok 1.16.0`. Playwright's Chromium is already installed; FiftyOne bundles Mongo.)

## Run (agent path) — the driver

### Restart the App server (do this after every non-plugin edit)

```bash
.venv311/Scripts/python.exe .claude/skills/run-vix/driver.py restart
```

No dataset needed — datasets live in Mongo and the restart never touches them; the App comes up and
you pick a dataset in the UI (top-left dataset name). `restart` sweeps the FiftyOne port range
(`:5151`–`:5160` by default), kills **every** App server it finds, and relaunches one **detached**
(survives the driver exiting), then waits until it answers. Verified:

```
killed nothing (swept :5193-:5195)
restarted, no dataset bound (pick one in the App), on http://localhost:5193  ready=True
```

(and the kill side, relaunching over a running server: `killed [28600]` → `ready=True`.) Options:
`--port` (base port), `--scan` (how many ports to sweep), `--dataset NAME` (optional: reopen a
specific dataset instead of letting you pick).

### Screenshot the live App (GUI proof / eyeball a change)

```bash
.venv311/Scripts/python.exe .claude/skills/run-vix/driver.py screenshot --out _artifacts/vix_app.png
```

Builds the self-contained `vix_verify` demo dataset, launches the App on `:5199`, drives it with
Playwright, and writes the PNG. Verified:

```
screenshot -> C:\code\claude\VIX\_artifacts\vix_app.png  (vix toolbar svg imgs seen: 8)
```

The PNG shows the grid + the VIX toolbar buttons + sidebar fields (`vix_hash`, `yolo_detections`,
`knn_dist`, `routing_decision`). `_artifacts/` is gitignored. **Open the PNG and look** — a blank
grid or an error page means it didn't really render.

### Full Playwright E2E (verify a GUI/plugin change)

```bash
.venv311/Scripts/python.exe .claude/skills/run-vix/driver.py e2e
```

Runs `tests/test_gui_browser.py` (real-browser Playwright: launches a live App, clicks toolbar
buttons, asserts DOM toasts) + `tests/test_gui_e2e.py` (handler-anchored against live Mongo).
Full run this session: **browser 13 passed / 1 skipped, e2e 52 passed** (~4 min total). Subset:

```bash
.venv311/Scripts/python.exe .claude/skills/run-vix/driver.py e2e -k "b6_all_vix_toolbar_buttons_present"
```
```
1 passed, 65 deselected, 29 warnings in 36.34s
```

## Gotchas (battle scars — read before debugging the App)

- **Stale modules in the running App.** The #1 trap. The plugin reloads but the adapter/pipeline
  do not. Symptom: a method/behaviour you just added is "missing" or behaves like the old code.
  Fix: `driver.py restart`. (Defensive code should `getattr`-guard new adapter methods so a stale
  App degrades instead of crashing — but restart is the real fix.)
- **Closing the browser ≠ restarting the server.** The `vix app` Python process keeps running with
  cached modules. You must kill the *process* (the driver does).
- **Never run the GUI test suites while the user's App is open on `vix_verify`.** Both
  `test_gui_browser.py` and `test_gui_e2e.py` rebuild **and delete** the `vix_verify` dataset at
  teardown; a live App viewing it crashes (`estimatedSampleCount`). Restart the user's App onto a
  different dataset, or close it, before `driver.py e2e`.
- **prompt=False operators are silent on failure.** A toolbar Button with `prompt=False` shows no
  output modal, so a bare `return {"error": …}` is invisible ("按了沒反應"). Every path must
  `ctx.ops.notify(...)`. The browser test `b8b` asserts the toast actually appears in the DOM.
- **`find_similar` "Query IDs … do not exist in this index"** is NOT "no similar found" — the
  selected box isn't in the patch index (stale index, or a partially-embedded sample dropped by the
  all-or-nothing builder). The operator self-heals (rebuild + retry); rebuilding the index fixes it.
- **Playwright must use `wait_until="domcontentloaded"`**, never `networkidle` — the FiftyOne
  websocket never goes idle, so `networkidle` hangs. The grid needs a fixed `wait_for_timeout`
  (~12 s) after load; CI/first-run is slow.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `'FiftyOneAdapter' object has no attribute '…'` in the App | Stale module — `driver.py restart`. |
| Driver `restart` says `ready=False` | Port still held / Mongo slow. Re-run; or `_kill_port(port)` then retry. The dataset name must exist (`fo.load_dataset` raises otherwise). |
| `screenshot` PNG is blank / shows an error | App didn't render — increase the post-load wait, confirm `:port` is free, re-run. |
| E2E can't import fiftyone | You used the wrong interpreter. Use `.venv311/Scripts/python.exe`, not base Python. |
| Toolbar buttons missing in the App | `FIFTYONE_PLUGINS_DIR` not set to `src/vix/plugins`. The driver sets it; `vix app` relies on the installed plugin. |

## Human path

`vix app` (or `.venv311/Scripts/python.exe -m vix.cli app`) opens dataset `vix` on `:5151` and
blocks until Ctrl-C. Fine for eyeballing; useless headless and gives you no programmatic handle —
use the driver for anything you need to assert.
