# VIX Testing Strategy

The architecture (`core` has zero FiftyOne dependency, FiftyOne sits behind the
`DatasetAdapter` seam) is what makes VIX testable without a heavy stack. There
are two tiers.

## Tier 1 — runs anywhere (CI / dev / this repo)

`python -m pytest -q` → **102 tests, <1.5s**, needs only `numpy + pytest + PyYAML + pillow`.
No FiftyOne, MongoDB, GPU or network.

Dataset-management additions (scenario-validation driven): `test_labelmap.py` (merge / relabel / rollback), `test_errors.py` (IoU classification-vs-localization), `test_lsh_scale.py` (LSH ≈ brute-force), `test_triage_explain.py` (risk queue + plain-language), `test_open_set.py` (new-class clustering / split leakage / harmful ranking / drill-down), `test_quality_gate_verify.py` (reviewer consistency / quality trend / pre-train gate / export hash verify), `test_pipeline_extras.py` + `test_pipeline_round3.py` (CLI/pipeline wiring).

| Test file | Covers |
|-----------|--------|
| `test_scorer.py` | cosine kNN distance, leave-one-out intra-class distance, two-axis scoring, novel-class = inf |
| `test_threshold.py` | per-class percentile calibration, low_conf / far_from_known / low_support routing, save/load |
| `test_manifest.py` | sha256 hashing, append-only dedup, reload |
| `test_decision_log.py` | append-only JSONL, hash-chain linkage, **tamper detection** |
| `test_reference.py` | centroid shift, label consistency, **drift guard trigger**, npz save/load |
| `test_exporter.py` | YOLO txt + data.yaml, unknown-class skip |
| `test_analytics.py` | **label-error detection**, near-duplicate grouping, class distribution, coverage gaps, coverage delta, active-learning ranking, cross-period drift |
| `test_snapshot.py` | version snapshot composition + exclusions, deterministic content hash, restore |
| `test_report.py` | health report aggregation (dup rate / review ratio / under-represented), markdown render, diff vs prev |
| `test_pipeline_flow.py` | **end-to-end** ingest→calibrate→route→export + guard, via `InMemoryAdapter` |
| `test_e2e_realfiles.py` | **real PNG files** → pixel embedder → dedup / coverage / active-learn / health report (proves the embedding→analytics chain runs on disk) |
| `test_cli_smoke.py` | CLI argparse → pipeline → manifest + logging wiring |

This tier exercises **all the closed-loop / data-quality logic** — the part that
actually matters for correctness — because that logic lives in `core`.

## Tier 2 — needs the real environment (offline target machine)

These touch FiftyOne / LanceDB / DINOv2 / YOLO and therefore run on the
air-gapped deployment, not here (this dev box has Python 3.14 and no FiftyOne):

| Path | How to verify |
|------|---------------|
| `FiftyOneAdapter.sync / set_detections / samples / tags / fields` | `pip install -e ".[fiftyone]"`, ingest a small folder, assert round-trip via `vix` CLI |
| `compute_embeddings` (DINOv2 ViT-B/14 crops) | run `vix embed`, check `dino_embedding` populated; sanity-check throughput to fix batch size |
| `build_knn_index` (LanceDB backend) | `vix embed` then confirm `vix_sim` brain key + on-disk LanceDB |
| `detect.run_yolo` (ultralytics) | `vix infer --weights model.pt` on real images |
| `launch_app` + SavedViews + review pull | `vix app`, lasso-tag in the App, then `pull_review_decisions` |

### UI / E2E acceptance

No browser/UI-testing MCP is available in this environment, so the FiftyOne App
acceptance (lasso-select → tag → resolve loop) is done manually / via the
project `ux-test` skill on the target machine. The acceptance criteria are in
[v0.1-technical-spec.md §12](v0.1-technical-spec.md).

## Diagnosing failures (Log)

Every stage logs to `<workspace>/vix.log` (and stderr). Run with
`--log-level DEBUG` for verbose tracing. The append-only `decision_log.jsonl`
plus `DecisionLog.verify_chain()` reconstruct exactly what was routed/guarded and
prove the record was not tampered with.
