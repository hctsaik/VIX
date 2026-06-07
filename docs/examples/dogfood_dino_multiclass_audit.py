"""Multi-class label-error detection with REAL DINOv2: `vix embed` (built-in DINO) -> `vix audit-labels`.

audit_labels flags samples whose DINO kNN-majority label disagrees with their given label — i.e. "this is
labelled X but its nearest neighbours are class Y" = a likely wrong-class label. This needs ≥2 classes
(unlike the single-class patHole outlier demo). We use SAFE's MVTec cutouts (capsule/hazelnut/metal_nut/
screw), deliberately MISLABEL 10, run real DINO + audit-labels, and check it catches them.

Faithful to the CLI flow: builds a FiftyOne dataset, runs FiftyOneAdapter.compute_embeddings (the built-in
torch.hub DINOv2 backend) then pipeline.audit_labels — the exact code `vix embed` + `vix audit-labels` run.
Set VIX_DINOV2_HUB_DIR to a hub cache for offline. Run: python docs/examples/dogfood_dino_multiclass_audit.py
"""
from __future__ import annotations

import glob
import os
import random
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

os.environ.setdefault("VIX_DINOV2_HUB_DIR", r"C:\code\claude\SAFE\.cache\torchhub")
os.environ.setdefault("FIFTYONE_DO_NOT_TRACK", "true")

CUT = Path(r"C:\code\claude\SAFE\Workspace\cutouts")
CLASSES = ["capsule", "hazelnut", "metal_nut", "screw"]
PER, N_MISLABEL = 30, 10


def main():
    import fiftyone as fo

    from vix import pipeline
    from vix.adapters.fiftyone_adapter import FiftyOneAdapter
    from vix.config import Config

    cfg = Config()
    cfg.embedding_backend = "dinov2-vits14-torch"
    name = "vix_mvtec_labelaudit"
    if fo.dataset_exists(name):
        fo.delete_dataset(name)
    ds = fo.Dataset(name)
    rng = random.Random(0)
    truth, samples = {}, []
    for cls in CLASSES:
        files = sorted(glob.glob(str(CUT / cls / "*.png")) + glob.glob(str(CUT / cls / "*.jpg")))
        for p in files[:PER]:
            h = f"{cls}_{Path(p).stem}"
            truth[h] = cls
            s = fo.Sample(filepath=p, tags=["golden"])
            s["vix_hash"] = h
            s["yolo_detections"] = fo.Detections(
                detections=[fo.Detection(label=cls, bounding_box=[0, 0, 1, 1], confidence=1.0)])
            samples.append(s)
    ds.add_samples(samples)

    # PLANT 10 wrong-class mislabels (set the detection label to a different class)
    hs = list(truth)
    rng.shuffle(hs)
    planted = {}
    for h in hs[:N_MISLABEL]:
        s = ds.match({"vix_hash": h}).first()
        wrong = rng.choice([c for c in CLASSES if c != truth[h]])
        s["yolo_detections"].detections[0].label = wrong
        s.save()
        planted[h] = (wrong, truth[h])  # (given/wrong, true)

    # the real flow: vix embed (built-in DINOv2) -> vix audit-labels
    ad = FiftyOneAdapter(cfg, dataset_name=name)
    ad.compute_embeddings("dinov2-vits14-torch")
    issues = pipeline.audit_labels(ad, cfg)

    flagged = {i.id.split(":")[0] for i in issues}
    caught = set(planted) & flagged
    extra = flagged - set(planted)
    print(f"\n# multi-class DINO label audit — classes={CLASSES}  n={len(truth)}  planted_mislabels={len(planted)}")
    print(f"audit-labels flagged {len(flagged)} images;  caught {len(caught)}/{len(planted)} planted  "
          f"(recall={len(caught)/len(planted):.2f}, precision={len(caught)/max(1,len(flagged)):.2f}, extra={len(extra)})")
    print("\nflagged (標成 X 但 DINO 鄰居多為 Y):")
    for i in issues[:16]:
        h = i.id.split(":")[0]
        mark = "PLANTED✓" if h in planted else "其他"
        print(f"  {h[:22]:22}  標成 {i.given_label:9}→ DINO 鄰居多為 {i.suggested_label:9} dis={i.disagreement:.2f}  [{mark}; 真類={truth.get(h,'?')}]")
    missed = set(planted) - caught
    if missed:
        print(f"\n漏掉的 planted: {[(h, planted[h]) for h in missed]}")
    fo.delete_dataset(name)


if __name__ == "__main__":
    main()
