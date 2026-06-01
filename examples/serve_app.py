"""啟動一個常駐的 VIX FiftyOne App,供瀏覽器手動連入 http://localhost:5151。

建一個有內容的 dataset(19 樣本)→ calibrate + route(產生 pass/review 標籤與分數)→
launch App(自動載入 @vix/review plugin)→ 常駐到被 Ctrl-C 或關閉為止。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("FIFTYONE_PLUGINS_DIR", str(ROOT / "src" / "vix" / "plugins"))
os.environ.setdefault("FIFTYONE_DO_NOT_TRACK", "true")
os.environ.setdefault("VIX_WORKSPACE", str(ROOT / ".venv311" / "appws"))

import fiftyone as fo  # noqa: E402

from vix import pipeline  # noqa: E402
from vix.adapters.fiftyone_adapter import FiftyOneAdapter  # noqa: E402
from vix.config import Config  # noqa: E402
from vix.verification import DATASET, PORT, _build_dataset  # noqa: E402


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    cfg = Config(workspace=ROOT / ".venv311" / "appws")
    cfg.ensure_dirs()
    cfg.embedding_backend = "pixel_fallback"

    ds = _build_dataset(fo)  # vix_verify, 19 samples (golden/anchor/review), persistent
    adapter = FiftyOneAdapter(cfg, dataset_name=DATASET)
    pipeline.calibrate(adapter, cfg)
    pipeline.route(adapter, cfg)  # 產生 pass/review tag + 分數 + flag_reason

    print(f"VIX App serving '{DATASET}' ({len(ds)} samples) at http://localhost:{PORT}", flush=True)
    print("用瀏覽器開 http://localhost:5151;按 ` 叫出 @vix/review operators。Ctrl-C 結束。", flush=True)
    session = fo.launch_app(ds, remote=True, port=PORT)
    session.wait(-1)  # 常駐


if __name__ == "__main__":
    main()
