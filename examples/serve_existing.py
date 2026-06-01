"""只載入已建好的 vix_animals(含 feat_embedding + feat_umap)並啟 App,不重算。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import fiftyone as fo  # noqa: E402

ds = fo.load_dataset("vix_animals")
print(f"serving 'vix_animals' {len(ds)} samples -> http://localhost:5151", flush=True)
session = fo.launch_app(ds, remote=True, port=5151)
session.wait(-1)
