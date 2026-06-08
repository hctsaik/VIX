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
            if d.embedding is not None:  # preserve the DINOv2 crop embedding through relabel/rewrite —
                det[_EMB_FIELD] = np.asarray(d.embedding, dtype=float).tolist()  # else confirm→golden wipes it
            dets.append(det)
        s[_DET_FIELD] = fo.Detections(detections=dets)
        s.save()

    def _dino_model(self, model_key: str):
        """Real DINOv2: FiftyOne Model Zoo if it loads, else the built-in torch.hub embedder (offline-
        capable). Returns an object with `.embed(np_or_pil)` and a `with` context."""
        try:
            import fiftyone.zoo as foz
            m = foz.load_zoo_model(model_key)
            m._vix_tag = f"zoo:{model_key}"
            return m
        except Exception as e:  # noqa: BLE001 - zoo broken / model missing -> built-in DINOv2
            log.warning("FiftyOne zoo unavailable (%s); using built-in torch.hub DINOv2",
                        str(e).splitlines()[0][:100])
            from ..embedding.dinov2_torch import DinoV2Embedder
            m = DinoV2Embedder(model_key, hub_dir=getattr(self.cfg, "dinov2_hub_dir", None))
            m._vix_tag = f"torch.hub:{model_key}"
            return m

    def compute_embeddings(self, model_key: str = MODEL_KEY) -> None:
        import contextlib

        from PIL import Image

        if self.cfg.embedding_backend == "pixel_fallback":  # offline/no-GPU: cheap pixel embedding
            from ..embedding.simple import pixel_embedding
            model_cm: object = contextlib.nullcontext()
            tag = "pixel_fallback"

            def _emb(crop):
                return np.asarray(pixel_embedding(crop), dtype=float).ravel()
        else:  # real DINOv2 (zoo or built-in torch.hub)
            model = self._dino_model(model_key)
            model_cm = model
            tag = getattr(model, "_vix_tag", model_key)

            def _emb(crop):
                return np.asarray(model.embed(np.array(crop)), dtype=float).ravel()

        ds = self._dataset()
        with model_cm:
            for s in ds.iter_samples(autosave=True, progress=True):
                if s[_DET_FIELD] is None:
                    continue
                img = Image.open(s.filepath).convert("RGB")
                vecs = []
                for det in s[_DET_FIELD].detections:
                    emb = _emb(crop_detection(img, _from_fo_bbox(det.bounding_box)))
                    det[_EMB_FIELD] = emb.tolist()
                    vecs.append(emb)
                if vecs:
                    s[_EMB_FIELD] = np.mean(np.vstack(vecs), axis=0).tolist()
            # bind the encoder identity into the dataset (audit truth): a swapped/drifted encoder
            # (re-pulled weights, torch upgrade, CPU<->GPU, changed preprocessing) changes this fingerprint
            from ..core.encoder_fingerprint import encoder_fingerprint, probe_digest
            material = {"backend": self.cfg.embedding_backend, "vix_tag": tag, "probe_digest": probe_digest(_emb)}
            if self.cfg.embedding_backend != "pixel_fallback" and hasattr(model, "fingerprint_material"):
                material.update(model.fingerprint_material())
            fp = encoder_fingerprint(material)
            ds.info["vix_encoder_fp"] = fp["fp"]
            ds.info["vix_encoder_components"] = fp["components"]
            ds.save()
        log.info("fiftyone.compute_embeddings: done (%s, encoder_fp=%s)", tag, fp["fp"])

    def encoder_fingerprint(self) -> dict:
        """The encoder identity recorded at embed time (vix_encoder_fp in dataset.info), so calibrate/gate
        can bind to and detect a changed encoder without reloading the model."""
        info = self._dataset().info
        return {"fp": info.get("vix_encoder_fp"), "components": info.get("vix_encoder_components", {})}

    def build_knn_index(self, embeddings_field: str = _EMB_FIELD) -> str:
        import fiftyone.brain as fob

        ds = self._dataset()
        fob.compute_similarity(
            ds, embeddings=embeddings_field, backend=self.cfg.similarity_backend, brain_key="vix_sim"
        )
        log.info("fiftyone.build_knn_index: backend=%s", self.cfg.similarity_backend)
        return "vix_sim"

    def build_patch_similarity(self, patches_field: str = _DET_FIELD,
                               embeddings_field: str = _EMB_FIELD, backend: str = "sklearn") -> str:
        """Object-BOX (patch) similarity index over the per-detection DINOv2 crop embeddings, so the
        App's native sort-by-similarity (select a box → magnifying glass) ranks by how the OBJECT
        looks — not the whole scene. sklearn backend = exact NN, no extra deps (lancedb etc. optional).
        Idempotent: replaces any prior run so the operator can be clicked repeatedly.

        We hand brain an explicit {sample_id: (n_patches, dim)} array of OUR vectors rather than the
        field NAME: a field name doesn't register dynamic per-detection fields, so brain would treat the
        embeddings as absent and silently download a default zoo model (mobilenet) to recompute them —
        which is both wrong (not DINO) and fragile (fails if the zoo manifest is unavailable)."""
        import numpy as np
        import fiftyone.brain as fob

        ds = self._dataset()
        brain_key = "vix_patch_sim"
        if brain_key in ds.list_brain_runs():
            ds.delete_brain_run(brain_key)
        emb: dict = {}
        for s in ds.iter_samples():
            field = s[patches_field]
            if field is None or not field.detections:
                continue
            vecs = [d.get_field(embeddings_field) for d in field.detections
                    if d.has_field(embeddings_field) and d.get_field(embeddings_field) is not None]
            if vecs and len(vecs) == len(field.detections):  # all-or-nothing keeps array↔patch order aligned
                emb[s.id] = np.array(vecs, dtype=float)
        if not emb:
            raise ValueError("找不到偵測框的 DINO 嵌入;請先計算嵌入(vix embed / vix similarity 會自動計算)")
        fob.compute_similarity(
            ds, patches_field=patches_field, embeddings=emb, backend=backend, brain_key=brain_key,
        )
        log.info("fiftyone.build_patch_similarity: backend=%s patches=%s samples=%d", backend, patches_field, len(emb))
        return brain_key

    def has_embeddings(self, embeddings_field: str = _EMB_FIELD) -> bool:
        """True iff at least one detection already carries a crop embedding (so the operator can skip the
        expensive compute_embeddings when DINO vectors are already present)."""
        ds = self._dataset()
        for s in ds.iter_samples():
            field = s[_DET_FIELD]
            if field is not None:
                for det in field.detections:
                    if det.has_field(embeddings_field) and det.get_field(embeddings_field) is not None:
                        return True
        return False

    def has_full_embeddings(self, embeddings_field: str = _EMB_FIELD) -> bool:
        """True iff EVERY detection carries a crop embedding. build_patch_similarity is all-or-nothing per
        sample (a sample with ANY un-embedded box is dropped from the index, so its boxes become
        un-queryable -> FiftyOne 'Query IDs ... do not exist in this index'). So the similarity build must
        embed when coverage is PARTIAL, not only when it is zero — otherwise newly-added boxes silently
        fall out of the index and find-similar fails on them."""
        ds = self._dataset()
        for s in ds.iter_samples():
            field = s[_DET_FIELD]
            if field is None:
                continue
            for det in field.detections:
                if not (det.has_field(embeddings_field) and det.get_field(embeddings_field) is not None):
                    return False
        return True

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

    def remove_tags(self, vix_hash: str, tags: list[str]) -> None:
        s = self._sample(vix_hash)
        if s is None:
            return
        s.tags = [t for t in s.tags if t not in tags]
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

    def compute_visualization(self, embeddings_field: str = _EMB_FIELD, num_dims: int = 2,
                              refit: bool = False, seed: int = 42) -> str:
        """OBJECT-LEVEL, TRANSFORM-STABLE UMAP of the per-DETECTION DINOv2 crop embeddings -> the OSS
        Embeddings panel plots ONE POINT PER YOLO BOX (not per image); lasso selects OBJECTS, fully offline.

        STABILITY: the UMAP reducer is FIT ONCE and persisted (workspace/vix_umap_reducer.pkl), then later
        runs ``transform()`` new boxes into that FROZEN coordinate space — so adding data and recomputing
        keeps the same layout (no rotate/flip/reshuffle), unlike a re-fit-every-time UMAP. The reducer is
        bound to the encoder fingerprint: a re-embed (different encoder) or refit=True re-anchors the map.
        Points always come from ``transform()`` (even on the fit run) so the fit-run and later runs match.

        Object-level mirrors build_patch_similarity: hand brain an explicit {sample_id: (n_patches, dim)}
        POINTS dict with patches_field= (bypasses brain's own fit, never a zoo model)."""
        import pickle

        import fiftyone.brain as fob
        try:
            import umap
        except Exception as exc:  # noqa: BLE001
            raise ValueError("需要 umap-learn(Tier-2 已含;pip install umap-learn)") from exc

        ds = self._dataset()
        # FLAT (total_patches, dim) in collection->detection order, matching how brain enumerates patch
        # label_ids (samples.values(detections.id, unwind=True)) -> brain stores a flat points array the
        # Embeddings panel renders. Require FULL coverage (the operator/CLI embed first) so the row order
        # aligns with brain's full label enumeration; a missing box would misalign every later point.
        rows: list = []
        box_ids: list = []   # stable per-detection id -> pins each box's 2D point across rebuilds
        n_dets = 0
        for s in ds.iter_samples():
            field = s[_DET_FIELD]
            if field is None or not field.detections:
                continue
            for d in field.detections:
                n_dets += 1
                v = d.get_field(embeddings_field) if d.has_field(embeddings_field) else None
                if v is None:
                    raise ValueError("部分偵測框缺少 DINO 嵌入,無法做物件級視覺化;"
                                     "請先對全部框計算嵌入(vix embed / 建立相似索引會自動計算)")
                rows.append(v)
                box_ids.append(d.id)
        if n_dets == 0:
            raise ValueError("找不到偵測框,無法做物件級視覺化;請先跑偵測(vix infer)")
        if len(rows) < 5:  # UMAP spectral init degenerates on a handful of points (cryptic eigh/zero-size errors)
            raise ValueError(f"偵測框太少({len(rows)}),UMAP 視覺化需要至少 5 個框;請先累積更多標註框")
        X = np.array(rows, dtype=float)

        fp = (self.encoder_fingerprint() or {}).get("fp")  # freeze key: re-embed (new fp) re-anchors the map
        # dataset-SCOPED files so dataset B never reuses dataset A's frozen space (the workspace is shared).
        # reducer.pkl (the frozen UMAP) is written ONLY on (re)fit; points.pkl PINS each box's coordinate so
        # adding boxes never moves the existing ones (umap.transform jitters a re-projected batch; the cache
        # transforms ONLY new boxes and keeps old ones fixed) — the actual cross-rebuild consistency promise.
        rpath = self.cfg.workspace / f"vix_umap_reducer.{self.dataset_name}.pkl"
        ppath = self.cfg.workspace / f"vix_umap_points.{self.dataset_name}.pkl"
        reducer, cache = None, {}
        if not refit and rpath.exists():
            try:
                saved = pickle.loads(rpath.read_bytes())
                if saved.get("fp") == fp and saved.get("num_dims") == num_dims:
                    reducer = saved["reducer"]      # reuse the FROZEN space
            except Exception:  # noqa: BLE001 - corrupt/stale pickle -> just re-fit
                reducer = None
        fitted = reducer is None
        self.cfg.workspace.mkdir(parents=True, exist_ok=True)
        if fitted:
            # seed -> reproducible fit; clamp n_neighbors to the point count so small sets don't crash
            reducer = umap.UMAP(n_components=num_dims, n_neighbors=min(15, len(X) - 1), random_state=seed)
            reducer.fit(X)
            rpath.write_bytes(pickle.dumps({"fp": fp, "num_dims": num_dims, "reducer": reducer}))
        elif ppath.exists():
            try:
                pc = pickle.loads(ppath.read_bytes())
                if pc.get("fp") == fp:
                    cache = pc.get("points", {})    # pinned {box_id: [x, y]} from earlier runs
            except Exception:  # noqa: BLE001
                cache = {}
        new = [i for i, b in enumerate(box_ids) if b not in cache]  # transform ONLY un-pinned boxes
        if new:
            Pn = np.asarray(reducer.transform(X[new]), dtype=float)
            for j, i in enumerate(new):
                cache[box_ids[i]] = Pn[j].tolist()
        if fitted or new:                           # cache changed -> persist (else leave it untouched)
            ppath.write_bytes(pickle.dumps({"fp": fp, "points": cache}))
        P = np.asarray([cache[b] for b in box_ids], dtype=float)  # pinned points, in brain's label order

        if "vix_umap" in ds.list_brain_runs():
            ds.delete_brain_run("vix_umap")
        fob.compute_visualization(ds, patches_field=_DET_FIELD, points=P, brain_key="vix_umap")
        log.info("fiftyone.compute_visualization: vix_umap %s (%d boxes, %dD, fp=%s)",
                 "FIT" if fitted else "transform", len(X), num_dims, fp)
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
