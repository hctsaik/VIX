"""Round 11 補強的回歸測試:
  - InMemoryAdapter 檔案持久化(跨指令乾跑)
  - `vix embed` 不再因漏傳 model_key 而崩潰
  - verify_export 偵測注入的額外檔 + 子目錄同名檔不再碰撞漏檢
  - `vix compare` 兩來源並排比較(noise / dup / 跨來源回收)
"""

import numpy as np
from PIL import Image

from vix import pipeline
from vix.adapters.memory import InMemoryAdapter
from vix.cli import main
from vix.config import Config
from vix.core.verify import verify_export, write_dir_manifest
from vix.types import BBox, Detection, Tag


def _det(label, conf, emb):
    return Detection(label, conf, BBox(0.5, 0.5, 0.2, 0.2), embedding=np.array(emb, dtype=float))


# --- memory persistence (AC1/AC2/AC9/AC10 的根因修補) ---
def test_memory_adapter_persists_across_instances(tmp_path):
    sp = tmp_path / "state.pkl"
    a = InMemoryAdapter(state_path=sp)
    a.seed("g0", "g0.png", [_det("a", 0.9, [1, 0])], tags=[Tag.GOLDEN])
    a.attach_fields("g0", {"routing_decision": "pass"})

    b = InMemoryAdapter(state_path=sp)  # simulate a fresh process
    rows = list(b.samples())
    assert len(rows) == 1 and rows[0][0] == "g0"
    assert b.fields("g0")["routing_decision"] == "pass"
    assert rows[0][2][0].embedding is not None  # detection embedding survived the round-trip


def test_memory_dryrun_persists_across_cli_invocations(tmp_path):
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    for i in range(6):
        Image.new("RGB", (16, 16), (i * 30, 0, 0)).save(imgs / f"{i}.png")
    ws = tmp_path / "ws"
    base = ["--workspace", str(ws), "--adapter", "memory"]

    assert main(base + ["ingest", str(imgs), "--batch", "init", "--golden"]) == 0
    assert main(base + ["embed"]) == 0  # <- would raise TypeError before the model_key fix

    fresh = InMemoryAdapter(state_path=ws / "memory_state.pkl")
    assert len(list(fresh.samples())) == 6  # state survived across separate CLI processes


# --- verify_export completeness + collision-safety (AC8) ---
def test_verify_export_detects_injected_extra_file(tmp_path):
    d = tmp_path / "export"
    (d / "images").mkdir(parents=True)
    (d / "labels").mkdir(parents=True)
    (d / "images" / "a.png").write_bytes(b"AAAA")
    (d / "labels" / "a.txt").write_bytes(b"0 0.5 0.5 1 1")
    man = write_dir_manifest(d)
    assert verify_export(man, d)["ok"] is True

    (d / "images" / "sneaky.png").write_bytes(b"ZZZZ")  # inject an unrecorded file
    res = verify_export(man, d)
    assert res["ok"] is False
    assert "images/sneaky.png" in res["unexpected"]


def test_verify_export_subdir_basename_collision_safe(tmp_path):
    d = tmp_path / "export"
    (d / "x").mkdir(parents=True)
    (d / "y").mkdir(parents=True)
    (d / "x" / "img.png").write_bytes(b"AAAA")
    (d / "y" / "img.png").write_bytes(b"BBBB")
    man = write_dir_manifest(d)
    assert verify_export(man, d)["n_checked"] == 2  # both recorded by relative path (no collision)

    (d / "y" / "img.png").write_bytes(b"CCCC")  # tamper only the y copy
    res = verify_export(man, d)
    assert res["ok"] is False and "y/img.png" in res["mismatched"]


# --- compare two vendors (AC4) ---
def test_compare_two_vendors(tmp_path):
    cfg = Config(workspace=tmp_path / "ws")
    cfg.ensure_dirs()
    ad = InMemoryAdapter()
    for i in range(8):  # vendor A: clean two-class
        ad.seed(f"a{i}", "a.png", [_det("a", 0.9, [1, 0])], tags=["vendor:A"])
        ad.seed(f"ab{i}", "ab.png", [_det("b", 0.9, [0, 1])], tags=["vendor:A"])
    for i in range(8):  # vendor B: same images recycled from A
        ad.seed(f"b{i}", "b.png", [_det("a", 0.9, [1, 0])], tags=["vendor:B"])
        ad.seed(f"bb{i}", "bb.png", [_det("b", 0.9, [0, 1])], tags=["vendor:B"])
    for i in range(3):  # vendor B: mislabeled (given 'a' but looks like 'b')
        ad.seed(f"bx{i}", "bx.png", [_det("a", 0.9, [0, 1])], tags=["vendor:B"])

    r = pipeline.compare(ad, cfg, "vendor:A", "vendor:B")
    assert set(r["tags"]) == {"vendor:A", "vendor:B"}
    assert r["per_tag"]["vendor:A"]["n_samples"] == 16
    assert r["per_tag"]["vendor:B"]["n_label_issues"] >= 1  # the mislabeled bx caught
    assert r["cross_recycled"] >= 1  # B images near-duplicate of an A image
