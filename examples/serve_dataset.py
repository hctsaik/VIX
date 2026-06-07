"""Serve a SPECIFIC (persistent) FiftyOne dataset in the VIX App at http://localhost:5151.

Unlike serve_app.py (which builds the throwaway vix_verify demo), this opens an existing dataset by
name and forces persistent=True so it won't be auto-cleaned. Loads the @vix/review plugin so the
toolbar buttons (load / delete) appear.

    python examples/serve_dataset.py [dataset_name]      # default: patHole_Dataset
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("FIFTYONE_PLUGINS_DIR", str(ROOT / "src" / "vix" / "plugins"))
os.environ.setdefault("FIFTYONE_DO_NOT_TRACK", "true")
os.environ.setdefault("VIX_WORKSPACE", str(ROOT / "vix_workspace"))

import fiftyone as fo  # noqa: E402

PORT = 5151


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    name = sys.argv[1] if len(sys.argv) > 1 else "patHole_Dataset"
    if name not in fo.list_datasets():
        print(f"dataset {name!r} not found. available: {fo.list_datasets()}", flush=True)
        sys.exit(1)
    ds = fo.load_dataset(name)
    ds.persistent = True  # so external `fo` processes don't auto-clean it
    print(f"VIX App serving '{name}' ({len(ds)} samples) at http://localhost:{PORT}", flush=True)
    print("用瀏覽器開 http://localhost:5151;工具列有 📁 載入 / 🗑 刪除。Ctrl-C 結束。", flush=True)
    session = fo.launch_app(ds, remote=True, port=PORT)
    session.wait(-1)


if __name__ == "__main__":
    main()
