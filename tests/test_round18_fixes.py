"""Round 18 補強的回歸測試:
  - decision log 高水位錨點偵測尾端截斷(valid-but-shorter chain)
  - gate 對被截斷的帳本回 NO-GO
  - merge-preview 接受行內 JSON 字串(PowerShell 友善)
"""

import numpy as np

from vix import pipeline
from vix.adapters.memory import InMemoryAdapter
from vix.cli import main
from vix.config import Config
from vix.core.decision_log import DecisionLog
from vix.types import BBox, Detection, Tag


def _det(label, conf, emb):
    return Detection(label, conf, BBox(0.5, 0.5, 0.2, 0.2), embedding=np.array(emb, dtype=float))


def test_decision_log_detects_tail_truncation(tmp_path):
    dl = DecisionLog(tmp_path / "log.jsonl")
    dl.append("a")
    dl.append("b")
    dl.append("c")
    assert dl.is_truncated() is False

    lines = dl.path.read_text(encoding="utf-8").splitlines()
    dl.path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")  # drop the last record
    assert dl.verify_chain() is True   # the shorter chain is internally still valid...
    assert dl.is_truncated() is True   # ...but the high-watermark anchor catches the drop


def test_gate_nogo_on_truncated_audit(tmp_path):
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    ad = InMemoryAdapter()
    for i in range(6):
        ad.seed(f"g{i}", "g.png", [_det("a", 0.9, [1, 0])], tags=[Tag.GOLDEN])
    ad.seed("c0", "c.png", [_det("a", 0.05, [9, 9])], tags=[])
    pipeline.route(ad, cfg, pipeline.calibrate(ad, cfg))  # writes route records + hwm anchor

    p = cfg.decision_log_path
    lines = p.read_text(encoding="utf-8").splitlines()
    p.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")  # truncate the tail
    res = pipeline.pre_train_gate_stage(ad, cfg)
    assert res.verdict == "NO-GO"
    assert any("稽核鏈" in r for r in res.reasons)


def test_merge_preview_accepts_inline_json(tmp_path):
    rc = main(["--workspace", str(tmp_path / "ws"), "--adapter", "memory",
               "merge-preview", "--counts-a", '{"cat":10,"dog":5}', "--counts-b", '{"cat":8}'])
    assert rc == 0  # inline JSON string works (not only a file path)
