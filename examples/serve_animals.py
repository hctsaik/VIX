"""下載 CIFAR-10 子集 → 用 torchvision ResNet50(ImageNet 預訓練)算 embedding + UMAP
→ 啟 FiftyOne App 看分群。(改用 torchvision 直接抽特徵,繞過 fiftyone 壞掉的 torch zoo manifest)

瀏覽器開 http://localhost:5151,開 Embeddings 面板選 'feat_umap',Color by 選 ground_truth,
就會看到貓/狗/鳥/馬/汽車/船各自聚成一群。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import fiftyone as fo  # noqa: E402
import fiftyone.brain as fob  # noqa: E402
import fiftyone.zoo as foz  # noqa: E402
import torch  # noqa: E402
from fiftyone import ViewField as F  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision.models import ResNet50_Weights, resnet50  # noqa: E402

DATASET = "vix_animals"
PORT = 5151
CLASSES = ["cat", "dog", "bird", "horse", "automobile", "ship"]


def embed(ds) -> None:
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    model.fc = torch.nn.Identity()  # -> 2048-d penultimate features
    model.eval()
    tf = weights.transforms()  # resize/centre-crop/normalise (ImageNet)
    paths = ds.values("filepath")
    embs: list[list[float]] = []
    batch: list[torch.Tensor] = []
    with torch.no_grad():
        for i, fp in enumerate(paths, 1):
            batch.append(tf(Image.open(fp).convert("RGB")))
            if len(batch) == 32:
                embs.extend(model(torch.stack(batch)).numpy().tolist())
                batch = []
                print(f"    embedded {i}/{len(paths)}", flush=True)
        if batch:
            embs.extend(model(torch.stack(batch)).numpy().tolist())
    ds.set_values("feat_embedding", embs)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("[1/4] 載入 CIFAR-10(test,已快取)...", flush=True)
    src = foz.load_zoo_dataset("cifar10", split="test")

    print(f"[2/4] 取子集:{CLASSES} 約 600 張...", flush=True)
    if fo.dataset_exists(DATASET):
        fo.delete_dataset(DATASET)
    view = src.match(F("ground_truth.label").is_in(CLASSES)).take(600, seed=51)
    ds = view.clone(DATASET, persistent=True)

    print("[3/4] 算 ResNet50 embedding(CPU,1–2 分鐘)...", flush=True)
    embed(ds)

    print("[4/4] 算 UMAP 2D 降維...", flush=True)
    fob.compute_visualization(ds, embeddings="feat_embedding", method="umap", brain_key="feat_umap")

    print(f"READY: '{DATASET}' {len(ds)} 張 -> http://localhost:{PORT}", flush=True)
    print("App 內:開 Embeddings 面板選 'feat_umap';Color by 選 ground_truth。Ctrl-C 結束。", flush=True)
    session = fo.launch_app(ds, remote=True, port=PORT)
    session.wait(-1)


if __name__ == "__main__":
    main()
