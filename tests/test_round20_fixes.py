"""Round 20 補強的回歸測試:
  - held-out EVAL 不再洩漏進 review_queue / active_learn 候選池
  - ingest 自動加 batch:<id> 樣本標籤
  - restore-dismissed 還原 rejected(remove_tags),並記稽核
"""

import numpy as np
from PIL import Image

from vix import pipeline
from vix.adapters.memory import InMemoryAdapter
from vix.config import Config
from vix.core.decision_log import DecisionLog
from vix.types import BBox, Detection, Tag


def _det(label, conf, emb):
    return Detection(label, conf, BBox(0.5, 0.5, 0.2, 0.2), embedding=np.array(emb, dtype=float))


def test_eval_excluded_from_candidate_pools(tmp_path):  # AL3/AL5/AL6
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    ad = InMemoryAdapter()
    for i in range(6):
        ad.seed(f"g{i}", "g.png", [_det("a", 0.9, [1, 0])], tags=[Tag.GOLDEN])
    ad.seed("e0", "e.png", [_det("a", 0.05, [9, 9])], tags=[Tag.EVAL])  # held-out
    ad.seed("c0", "c.png", [_det("a", 0.05, [9, 9])], tags=[])          # genuine candidate

    rq_ids = {r["id"].split(":")[0] for r in pipeline.review_queue(ad, cfg, top=50)}
    assert "e0" not in rq_ids  # eval never re-enters the human review loop
    al_ids = {r["id"].split(":")[0] for r in pipeline.active_learn(ad, cfg, budget=50)}
    assert "e0" not in al_ids


def test_ingest_applies_batch_tag(tmp_path):  # AL2/AL3
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    for i in range(3):
        Image.new("RGB", (8, 8), (i * 20, 0, 0)).save(imgs / f"{i}.png")
    ad = InMemoryAdapter()
    pipeline.ingest(ad, cfg, str(imgs), "w22")
    all_tags = [t for _h, _s, _d, tags in ad.samples() for t in tags]
    assert "batch:w22" in all_tags  # compare/drift-type/parity can now filter without manual tagging


def test_restore_dismissed_removes_rejected(tmp_path):  # AL2/AL9
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    ad = InMemoryAdapter()
    ad.seed("x0", "x.png", [_det("a", 0.9, [1, 0])], tags=[Tag.GOLDEN])
    pipeline.dismiss(ad, cfg, ["x0"])
    assert Tag.REJECTED in next(t for h, _s, _d, t in ad.samples() if h == "x0")

    pipeline.restore_dismissed(ad, cfg, ["x0"])
    assert Tag.REJECTED not in next(t for h, _s, _d, t in ad.samples() if h == "x0")
    assert any(e["event"] == "undismiss" for e in DecisionLog(cfg.decision_log_path).read_all())
