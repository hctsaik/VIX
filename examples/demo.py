"""VIX 離線 demo — 不需 FiftyOne / GPU / 網路。

從 repo 根目錄執行:
    python examples/demo.py

它會用「像素 fallback embedder」在臨時資料夾產生幾張合成影像,跑完整資料守門員流程
(calibrate -> route -> dedup -> coverage -> guard/gate -> report -> export)並印出結果。
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

# 讓腳本免安裝即可執行(把 src 加進 path)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vix import pipeline  # noqa: E402
from vix.adapters.memory import InMemoryAdapter  # noqa: E402
from vix.config import Config  # noqa: E402
from vix.embedding.simple import pixel_embedding  # noqa: E402
from vix.types import BBox, Detection, Tag  # noqa: E402


def make_img(path: Path, kind: str, split: int) -> None:
    a = np.zeros((32, 32, 3), np.uint8)
    if kind == "vert":
        a[:, :split] = 255
    else:
        a[:split, :] = 255
    Image.fromarray(a).save(path)


def det(label: str, conf: float) -> Detection:
    return Detection(label, conf, BBox(0.5, 0.5, 1.0, 1.0))  # 全圖框


def main() -> None:
    work = Path(tempfile.mkdtemp(prefix="vix_demo_"))
    imgs = work / "imgs"
    imgs.mkdir()
    cfg = Config(workspace=work / "ws")
    cfg.ensure_dirs()
    cfg.embedding_backend = "pixel_fallback"
    ad = InMemoryAdapter(embedder=lambda im: pixel_embedding(im, size=8))

    # golden set:兩個視覺上可分的類別;前 2 張同時當 anchor(凍結參照)
    for i in range(8):
        p = imgs / f"vert{i}.png"
        make_img(p, "vert", 8 + i)
        ad.seed(f"vert{i}", str(p), [det("vert", 0.9)], tags=[Tag.GOLDEN] + ([Tag.ANCHOR] if i < 2 else []))
        ph = imgs / f"horiz{i}.png"
        make_img(ph, "horiz", 8 + i)
        ad.seed(f"horiz{i}", str(ph), [det("horiz", 0.9)], tags=[Tag.GOLDEN] + ([Tag.ANCHOR] if i < 2 else []))

    # 一對完全相同的重複(進 golden,讓 dedup 抓出來)
    for name in ("dupA", "dupB"):
        p = imgs / f"{name}.png"
        make_img(p, "vert", 24)
        ad.seed(name, str(p), [det("vert", 0.9)], tags=[Tag.GOLDEN])

    # 待檢的新批次(非 golden)
    make_img(imgs / "ok.png", "vert", 11)
    ad.seed("ok", str(imgs / "ok.png"), [det("vert", 0.95)])          # 應 pass
    make_img(imgs / "low.png", "vert", 11)
    ad.seed("low", str(imgs / "low.png"), [det("vert", 0.02)])        # 信心過低 -> review
    make_img(imgs / "novel.png", "horiz", 28)
    ad.seed("novel", str(imgs / "novel.png"), [det("vert", 0.9)])     # 外觀離群 -> review

    print(f"# 工作區: {work}\n")
    ad.compute_embeddings("pixel")  # 用像素 embedder 從真實檔算 embedding

    policy = pipeline.calibrate(ad, cfg)
    print(f"calibrate: {len(policy.thresholds)} 類別門檻")

    counts = pipeline.route(ad, cfg, policy)
    print(f"route    : {counts['pass']} pass, {counts['review']} review (flag_rate={counts['flag_rate']})")
    for h in ("ok", "low", "novel"):
        f = ad.fields(h)
        print(f"   - {h:5s}: {f['routing_decision']:6s} reasons={f['flag_reason']}")

    groups = pipeline.dedup(ad, cfg)
    print(f"dedup    : {len(groups)} 近似重複群,例如 {groups[0] if groups else '無'}")

    cov = pipeline.coverage(ad, cfg)
    print(f"coverage : {cov['distribution']}")

    pipeline.build_reference(ad, cfg)
    gate = pipeline.pre_train_gate_stage(ad, cfg)
    print(f"gate     : {gate.verdict}  reasons={gate.reasons or '全部通過'}")

    rep, paths = pipeline.health_report(ad, cfg, work / "report")
    print(f"report   : 品質分數 {rep['quality_score']}/100  ->  {paths['md']}")

    res = pipeline.export(ad, cfg, ["vert", "horiz"], work / "train_ready")
    print(f"export   : {res['n_images']} 張 -> {res['data_yaml']}  (+逐檔 hash manifest)")

    from vix.core.decision_log import DecisionLog

    print(f"audit    : hash-chain 驗證 = {DecisionLog(cfg.decision_log_path).verify_chain()}")


if __name__ == "__main__":
    main()
