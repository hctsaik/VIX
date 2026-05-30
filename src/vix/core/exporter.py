"""DatasetExporter — one-way export to YOLO txt + data.yaml.

Strictly read records -> write training files. It NEVER writes state back into
FiftyOne or the manifest (binding condition B#1: the adapter is one-directional).
Lets the user's existing YOLO training script consume golden data with zero change.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable

import yaml

from ..types import Detection


class DatasetExporter:
    def __init__(self, class_names: list[str]):
        self.class_names = list(class_names)
        self.index = {n: i for i, n in enumerate(self.class_names)}

    def _write_subset(self, recs, dst: Path, split: str, copy_images: bool) -> tuple[int, int, int]:
        img_dir = dst / "images" / split
        lbl_dir = dst / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        n_images = n_labels = n_skipped = 0
        for src_path, dets in recs:
            src = Path(src_path)
            n_images += 1
            if copy_images and src.exists():
                shutil.copy2(src, img_dir / src.name)
            lines: list[str] = []
            for d in dets:
                if d.label not in self.index:
                    n_skipped += 1
                    continue
                b = d.bbox
                lines.append(f"{self.index[d.label]} {b.cx:.6f} {b.cy:.6f} {b.w:.6f} {b.h:.6f}")
                n_labels += 1
            (lbl_dir / f"{src.stem}.txt").write_text(
                ("\n".join(lines) + "\n") if lines else "", encoding="utf-8"
            )
        return n_images, n_labels, n_skipped

    def export(
        self,
        records: Iterable[tuple[str, list[Detection]]],
        dst: str | Path,
        split: str = "train",
        copy_images: bool = False,
        val_split: float = 0.0,
    ) -> dict:
        dst = Path(dst)
        records = list(records)
        if 0.0 < val_split < 1.0:
            k = max(1, int(len(records) * val_split))
            subsets = {"val": records[:k], "train": records[k:]}
            val_dir = "images/val"
        else:
            subsets = {split: records}
            val_dir = f"images/{split}"

        n_images = n_labels = n_skipped = 0
        for sp, recs in subsets.items():
            ni, nl, ns = self._write_subset(recs, dst, sp, copy_images)
            n_images += ni
            n_labels += nl
            n_skipped += ns

        data_yaml = {
            "path": str(dst.resolve()),
            "train": "images/train",
            "val": val_dir,
            "names": {i: n for i, n in enumerate(self.class_names)},
        }
        yaml_path = dst / "data.yaml"
        yaml_path.write_text(
            yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        return {
            "n_images": n_images,
            "n_labels": n_labels,
            "n_skipped": n_skipped,
            "data_yaml": str(yaml_path),
        }
