# VIX App — full E2E test plan (multi-agent, Phase 0)

Goal: every button/feature in the FiftyOne App is exercised by a real Playwright E2E (not just unit
tests) so untested features can't silently break (cf. the review-queue embedding-wipe bug). For
Enterprise-gated native features, VIX provides an offline DINO replacement.

## Launch / harness facts
- `examples/serve_dataset.py` → persistent dataset on **:5151**, `FIFTYONE_PLUGINS_DIR`, `VIX_WORKSPACE`.
- Open panels via the **spaces API** (`session.spaces = fo.Space(children=[fo.Panel(type=...)])`), NOT DOM tab clicks.
- Selection is server-side: `session.selected = [sample_id]`.
- `wait_until="domcontentloaded"` + fixed waits (never `networkidle` — FiftyOne ws never idles).
- Toolbar buttons: `img[src*='<icon>.svg']`; they collapse into an overflow `⌄` at narrow widths (use wide viewport).
- Operator form inputs are **portal-appended last** textboxes/checkboxes/dropdowns.
- **Stop the App before running pytest** (the suite churns the persistent `vix_verify`).

## Features (23) + how to verify
Toolbar buttons: load_dataset(folder.svg), delete_dataset(trash.svg), build_similarity(simindex.svg),
find_similar(similar.svg, prompt=False), confirm_golden(check.svg), dismiss_false_alarm(ban.svg),
compute_visualization(scatter.svg). Operator-browser only: explain_sample, generate_weakness_report,
flag_label_issues, audit_label_errors, flag_loose_boxes. Panels: vix_report (regen/worklist buttons +
clickable confident-wrong/overturn 看圖 tables), vix_queue (看圖/確認→golden/誤報排除 row actions +
🔄 重新整理佇列).

Verify with server-side truth where possible: sample `.tags`, `DecisionLog.read_all()` + `verify_chain()`,
`fo.list_datasets()`, `ds.list_brain_runs()`, files under `VIX_WORKSPACE`; DOM text/screenshots for toasts,
tables, modal opens. Every op fails CLOSED with `{"error":...}` / in-panel banner (assert those on the
negative paths), never crashes.

## Execution order (precondition-driven)
1. Mount: grid → vix_report → vix_queue (no preconditions).
2. Selection ops on demo (has golden): confirm_golden(rev1) → dismiss_false_alarm(rev2) → explain_sample.
3. Queue actions (need golden): 看圖 → 確認→golden → 誤報排除 → 🔄 refresh.
4. Eval→report: generate_weakness_report → regen → worklist → clickable 看圖 tables.
5. Audit/flag (need embeddings/≥2 classes): flag_label_issues, audit_label_errors; flag_loose_boxes only if SAM.
6. Similarity/embeddings (need embeddings): build_similarity → find_similar; compute_visualization → Embeddings panel.
7. Lifecycle: load_dataset (own scenario); delete_dataset **last**.

## Enterprise-gated → VIX DINO replacement
| Native App capability | Gated? | VIX replacement |
|---|---|---|
| Similarity Search panel | Enterprise (in-app index build) | **done** — build_similarity + find_similar (vix_patch_sim, sklearn over DINO crops) |
| Embeddings "Create Embeddings" | Enterprise (compute button); VIEW is OSS | **done** — compute_visualization (vix_umap, UMAP over DINO) → native Embeddings panel plots it |
| Model Evaluation | Enterprise compute | covered differently — eval-ingest + weakness-report + VixReportPanel |
| Data Quality / dedup | Enterprise convenience | covered — dedup / near-dup-labels / audit-labels / box-qa |

Note: `Config.similarity_backend` defaults to lancedb (not installed) — VIX ops pass `backend="sklearn"`.
