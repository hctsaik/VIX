# VIX — Vision Integrity eXplainability

[![CI](https://github.com/hctsaik/VIX/actions/workflows/ci.yml/badge.svg)](https://github.com/hctsaik/VIX/actions/workflows/ci.yml)

**`yolo val` tells you your mAP. VIX tells you *what to fix*** — point it at your
dataset and your `.pt` and get a per-class FP/FN + AP + confusion report naming
your weakest class and the exact images to look at. Offline, one command, no retrain,
no service.

```bash
vix diagnose ./dataset --labels yolo --weights best.pt --data-yaml data.yaml
#  → 匯入你的標籤 + 跑你的模型 → weakness_report.html
#    health: AMBER  weakest: pothole AP=0.73  | FN: 47 missed/18 loc  | 12 confident false alarms
#  --labels yolo|voc|coco   ·   只有 predictions+GT?  vix eval-ingest results.jsonl  (純離線)
```

This is **Tier A**: runs in your existing Python+YOLO env — **no FiftyOne, no DINOv2,
no golden/anchor/calibrate worldview**. **Tier B** (`--audit`, needs DINOv2) adds
embedding label-audit + failure attribution (taxonomy / model / label_noise).
*Honesty:* imported labels are an **unverified reference** (never treated as golden) —
a reported "false positive" may be a missing label in **your** GT; the report says so.

<details><summary>The full data-gatekeeper (golden/anchor curation loop)</summary>

A **Data-Centric AI "data gatekeeper"** built as a thin layer on top of
[FiftyOne](https://github.com/voxel51/fiftyone). It combines **YOLO confidence**
and **DINOv2 embedding distance** to surface edge cases and label-definition
drift, so you can answer two questions you currently can't:

1. *Did the data I just added make the model better or worse?* (data attribution)
2. *Is my new data quietly making "bubble" vs "reflection" inconsistent?* (concept drift)

> This is **v0.1 — "visibility"**: single-user, on-prem / air-gapped, $0, no API.
> See [docs/spec/v0.1-technical-spec.md](docs/spec/v0.1-technical-spec.md) for the
> full spec and [docs/discussion/](docs/discussion/) for how the design was reached.
> The landable on-ramp design + acceptance is in
> [docs/discussion/landable-system.md](docs/discussion/landable-system.md).

</details>

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

## Hand this to another engineer

- **Beginner docs site (start here):** [docs/guide/site/index.html](docs/guide/site/index.html)
  — a clear, multi-page site for a first-time engineer: 5-minute quickstart, install, the
  `vix diagnose` on-ramp, reading the weakness report (with screenshots), input formats,
  the honest "did my fix help?" loop, Tier-B label audit + the FiftyOne App, and an honesty page.
  Regenerate: `python docs/examples/gen_site.py` (report shots: `gen_beginner_report.py` + `shoot_beginner_report.py`).
- **Full operator handbook (SOP, HTML):** [docs/guide/VIX_SOP.html](docs/guide/VIX_SOP.html)
  — one page covering **both native FiftyOne and the VIX-custom features**: concepts,
  setup, a copy-paste "happy path", the `vix` CLI workflow, the in-App `@vix/review`
  operators, the audit ledger, and troubleshooting.

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

**On-ramp (have a labelled dataset + a model? start here):**

```bash
vix diagnose ./dataset --labels yolo --weights best.pt --data-yaml data.yaml
#   one command: import your GT → run your model → weakness_report.html (.md)
vix import-labels ./dataset --labels coco --json ann.json   # just load GT to inspect
vix eval-run --weights best.pt                              # the yolo-val → VIX bridge
```

**Full curation gatekeeper (golden/anchor loop):**

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
