"""FiftyOneAdapter — the real backend (requires the ``fiftyone`` extra).

FiftyOne is imported lazily inside methods so the rest of VIX imports cleanly
without it (and stays testable). This adapter is written against the FiftyOne
1.x API; it is validated on the real air-gapped deployment (it cannot run under
the dev environment here, which has no FiftyOne / MongoDB / GPU).

Air-gap notes (see docs/spec §1): set ``FIFTYONE_DO_NOT_TRACK=true``, pre-stage
the DINOv2 weights via ``FIFTYONE_MODEL_ZOO_DIR``, and keep ``database_uri`` local.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from ..config import Config
from ..core.manifest import ManifestEntry
from ..embedding.dinov2 import MODEL_KEY, crop_detection
from ..logging_setup import get_logger
from ..types import BBox, Decision, Detection
from .base import DatasetAdapter, SampleRow

log = get_logger("vix.adapters.fiftyone")

_DATASET = "vix"
_HASH_FIELD = "vix_hash"
_DET_FIELD = "yolo_detections"
_EMB_FIELD = "dino_embedding"
_REVIEW_FIELD = "review_decision"


def _to_fo_bbox(b: BBox) -> list[float]:
    # VIX uses centre (cx,cy,w,h); FiftyOne uses top-left [x,y,w,h], all normalised.
    return [b.cx - b.w / 2.0, b.cy - b.h / 2.0, b.w, b.h]


def _from_fo_bbox(xywh: list[float]) -> BBox:
    x, y, w, h = xywh
    return BBox(cx=x + w / 2.0, cy=y + h / 2.0, w=w, h=h)


class FiftyOneAdapter(DatasetAdapter):
    def __init__(self, cfg: Config | None = None, dataset_name: str = _DATASET):
        self.cfg = cfg or Config()
        self.dataset_name = dataset_name
        self._ds = None

    # --- dataset handle ---
    def _dataset(self):
        if self._ds is None:
            import fiftyone as fo

            if fo.dataset_exists(self.dataset_name):
                self._ds = fo.load_dataset(self.dataset_name)
            else:
                self._ds = fo.Dataset(self.dataset_name, persistent=True)
        return self._ds

    def _sample(self, vix_hash: str):
        import fiftyone as fo  # noqa: F401

        ds = self._dataset()
        view = ds.match({_HASH_FIELD: vix_hash})
        return view.first() if len(view) else None

    # --- DatasetAdapter ---
    def sync(self, entries: Iterable[ManifestEntry]) -> None:
        import fiftyone as fo

        ds = self._dataset()
        known = set(ds.values(_HASH_FIELD)) if len(ds) else set()
        new = []
        for e in entries:
            if e.vix_hash in known:
                continue
            s = fo.Sample(filepath=e.src_path, tags=list(e.tags))
            s[_HASH_FIELD] = e.vix_hash
            s["batch_id"] = e.batch_id
            new.append(s)
        if new:
            ds.add_samples(new)
            log.info("fiftyone.sync: added %d samples (total=%d)", len(new), len(ds))

    def set_detections(self, vix_hash: str, detections: list[Detection]) -> None:
        import fiftyone as fo

        s = self._sample(vix_hash)
        if s is None:
            return
        dets = []
        for d in detections:
            det = fo.Detection(label=d.label, bounding_box=_to_fo_bbox(d.bbox), confidence=d.confidence)
            dets.append(det)
        s[_DET_FIELD] = fo.Detections(detections=dets)
        s.save()

    def compute_embeddings(self, model_key: str = MODEL_KEY) -> None:
        import fiftyone.zoo as foz
        from PIL import Image

        ds = self._dataset()
        model = foz.load_zoo_model(model_key)
        with model:
            for s in ds.iter_samples(autosave=True, progress=True):
                if s[_DET_FIELD] is None:
                    continue
                img = Image.open(s.filepath).convert("RGB")
                vecs = []
                for det in s[_DET_FIELD].detections:
                    crop = crop_detection(img, _from_fo_bbox(det.bounding_box))
                    emb = np.asarray(model.embed(np.array(crop)), dtype=float).ravel()
                    det[_EMB_FIELD] = emb.tolist()
                    vecs.append(emb)
                if vecs:
                    s[_EMB_FIELD] = np.mean(np.vstack(vecs), axis=0).tolist()
        log.info("fiftyone.compute_embeddings: done (%s)", model_key)

    def build_knn_index(self, embeddings_field: str = _EMB_FIELD) -> str:
        import fiftyone.brain as fob

        ds = self._dataset()
        fob.compute_similarity(
            ds, embeddings=embeddings_field, backend=self.cfg.similarity_backend, brain_key="vix_sim"
        )
        log.info("fiftyone.build_knn_index: backend=%s", self.cfg.similarity_backend)
        return "vix_sim"

    def samples(self) -> Iterable[SampleRow]:
        ds = self._dataset()
        for s in ds.iter_samples():
            dets: list[Detection] = []
            field = s[_DET_FIELD]
            if field is not None:
                for det in field.detections:
                    emb = det.get_field(_EMB_FIELD) if det.has_field(_EMB_FIELD) else None
                    dets.append(
                        Detection(
                            label=det.label,
                            confidence=float(det.confidence or 0.0),
                            bbox=_from_fo_bbox(det.bounding_box),
                            embedding=np.asarray(emb, dtype=float) if emb is not None else None,
                        )
                    )
            yield s[_HASH_FIELD], s.filepath, dets, list(s.tags)

    def attach_fields(self, vix_hash: str, fields: dict) -> None:
        s = self._sample(vix_hash)
        if s is None:
            return
        for k, v in fields.items():
            s[k] = v
        s.save()

    def apply_tags(self, vix_hash: str, tags: list[str]) -> None:
        s = self._sample(vix_hash)
        if s is None:
            return
        for t in tags:
            if t not in s.tags:
                s.tags.append(t)
        s.save()

    def get_by_tag(self, tag: str) -> Iterable[tuple[str, str, list[Detection]]]:
        for h, src, dets, tags in self.samples():
            if tag in tags:
                yield h, src, dets

    def pull_review_decisions(self) -> list[Decision]:
        out: list[Decision] = []
        ds = self._dataset()
        for s in ds.iter_samples():
            val = s[_REVIEW_FIELD] if s.has_field(_REVIEW_FIELD) else None
            if val:
                out.append(Decision(vix_hash=s[_HASH_FIELD], decision=str(val)))
        return out

    def compute_visualization(self, embeddings_field: str = _EMB_FIELD) -> str:
        import fiftyone.brain as fob

        fob.compute_visualization(
            self._dataset(), embeddings=embeddings_field, method="umap", brain_key="vix_umap"
        )
        return "vix_umap"

    def launch_app(self, saved_views: dict | None = None) -> None:
        import fiftyone as fo

        ds = self._dataset()
        for name, tag in (saved_views or {"review_queue": "review", "passed": "pass"}).items():
            try:
                ds.save_view(name, ds.match_tags(tag), overwrite=True)
            except Exception as exc:  # noqa: BLE001 - non-fatal view setup
                log.warning("save_view %s failed: %s", name, exc)
        session = fo.launch_app(ds)
        session.wait()
