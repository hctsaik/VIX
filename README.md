# VIX — Vision Integrity eXplainability

[![CI](https://github.com/hctsaik/VIX/actions/workflows/ci.yml/badge.svg)](https://github.com/hctsaik/VIX/actions/workflows/ci.yml)

A **Data-Centric AI "data gatekeeper"** built as a thin layer on top of
[FiftyOne](https://github.com/voxel51/fiftyone). It combines **YOLO confidence**
and **DINOv2 embedding distance** to surface edge cases and label-definition
drift, so you can answer two questions you currently can't:

1. *Did the data I just added make the model better or worse?* (data attribution)
2. *Is my new data quietly making "bubble" vs "reflection" inconsistent?* (concept drift)

> This is **v0.1 — "visibility"**: single-user, on-prem / air-gapped, $0, no API.
> See [docs/spec/v0.1-technical-spec.md](docs/spec/v0.1-technical-spec.md) for the
> full spec and [docs/discussion/](docs/discussion/) for how the design was reached.

## Architecture rule

`vix.core` has **zero FiftyOne dependency** and is fully unit-tested. FiftyOne is
touched only inside `vix.adapters.fiftyone_adapter`, behind the
`DatasetAdapter` seam — so the closed-loop logic stays ours and FiftyOne can be
swapped later by rewriting one file.

```
src/vix/
  core/       scorer · threshold · reference · decision_log · manifest · exporter   (pure, tested)
  adapters/   base (seam) · memory (test/dry-run) · fiftyone_adapter (real, lazy import)
  embedding/  dinov2 (ViT-B/14, crop)
  detect.py   YOLO inference -> detections
  pipeline.py orchestration
  cli.py      vix ingest|infer|embed|calibrate|route|guard|export|app
```

## Install

```bash
# core only (pure logic, runs anywhere)
pip install -e .
# with the real backend (FiftyOne + LanceDB) and YOLO
pip install -e ".[fiftyone,yolo,dev]"
```

Air-gapped install + DINOv2 offline weights + telemetry-off: see
[docs/spec/v0.1-technical-spec.md §1](docs/spec/v0.1-technical-spec.md).

## See the clustering for yourself (demo + step-by-step guide)

A CIFAR-10 animal demo (cat/dog/bird/horse/ship/automobile) you can run on any
machine to *see* the embedding clusters in the FiftyOne App:

- **Reproduce on another computer:** [docs/SETUP_OTHER_MACHINE.md](docs/SETUP_OTHER_MACHINE.md)
  (Python 3.11 → `scripts/setup_tier2.ps1` → `python examples/serve_animals.py` → open `http://localhost:5151`).
- **Illustrated walkthrough (0–7 steps, annotated screenshots):**
  [docs/guide/EMBEDDINGS_HOWTO.html](docs/guide/EMBEDDINGS_HOWTO.html)
  ([overview](docs/guide/README.md)) — switch dataset → open Embeddings panel →
  lasso a cluster → "only show selected" → inspect what those images are.

Tier-2 (FiftyOne GUI + Playwright) pinned deps: [requirements-tier2.txt](requirements-tier2.txt).

## Workflow

```bash
vix ingest ./golden  --batch init --golden       # import golden set
vix ingest ./anchor  --batch init --anchor       # frozen anchor (never trained on)
vix ingest ./incoming --batch 2026w22            # new batch to check
vix infer  --weights yolo.pt                      # YOLO -> detections
vix embed                                         # DINOv2 ViT-B/14 + LanceDB kNN index
vix calibrate                                     # per-class percentile thresholds
vix route                                         # tag pass / review (+ flag_reason)
vix guard  --build                                # frozen-reference drift self-gate
vix app                                           # review in the FiftyOne App
vix export ./train_ready                          # one-way -> YOLO txt + data.yaml
```

## Testing

```bash
pip install pytest && python -m pytest -q     # 25 tests, no FiftyOne needed
```

See [docs/spec/TESTING.md](docs/spec/TESTING.md) for what is verified here vs what
needs the real (FiftyOne) environment.

## License

Apache-2.0. Depends on FiftyOne + fiftyone-brain (Apache-2.0) and bundled MongoDB
(SSPL — fine for internal use; do not offer it as a hosted service).
