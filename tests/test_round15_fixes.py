"""Round 15 補強的回歸測試:
  - SPC 用前段 in-control 基線估參數 -> 緩慢漂移能「提早」警報(非只在最後一點)
  - 短序列旗標
  - 損毀影像在 embed 階段拋出「具名」的 VIX 錯誤(指出檔案)
"""

import numpy as np
import pytest

from vix import pipeline
from vix.adapters.memory import InMemoryAdapter
from vix.embedding.simple import pixel_embedding
from vix.types import BBox, Detection


def test_spc_monitor_baseline_detects_slow_drift_early():  # AG4
    rng = np.random.RandomState(0)
    flat = list(0.05 + 0.005 * rng.randn(6))     # 6 in-control batches
    rising = [0.05 + 0.03 * i for i in range(1, 7)]  # then a slow monotonic rise
    r = pipeline.spc_monitor(flat + rising, method="cusum")
    assert r["alarm"] is True
    assert r["alarm_index"] is not None and r["alarm_index"] < len(flat + rising) - 1  # not only the last point


def test_spc_monitor_flags_short_series():  # AG4
    assert pipeline.spc_monitor([0.1, 0.2, 0.3], method="ewma")["short_series"] is True
    assert pipeline.spc_monitor([0.1] * 10, target=0.1, sigma=0.01)["short_series"] is False


def test_embed_corrupt_image_names_file(tmp_path):  # AG6
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not really an image")
    ad = InMemoryAdapter(embedder=pixel_embedding)
    ad.seed("h", str(bad), [Detection("a", 0.9, BBox(0.5, 0.5, 0.2, 0.2))])
    with pytest.raises(ValueError) as ei:
        ad.compute_embeddings("k")
    assert "bad.png" in str(ei.value)  # the offending file is named, not a raw PIL traceback
