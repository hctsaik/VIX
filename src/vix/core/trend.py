"""Report trend over time (Tier 2: "did my curation actually move bubble's AP over the rounds?").

The decision log already records every `eval_ingest` (mAP / per-class AP / eval_set_hash) and
`weakness_report` (health) event, append-only and hash-chained. So the trend is ALREADY a
tamper-evident, queryable record — this just reads and presents it. No retraining, no snapshot bloat;
the AP numbers come from the engineer's periodic external eval-ingest (offline). Pure: operates on
already-read records.

HONESTY: AP deltas are only comparable when the eval SET didn't change (eval_set_hash constant). A
changed eval set across the series is flagged — otherwise a "+0.1 AP" could just be an easier val set.
"""

from __future__ import annotations


def eval_trend(records: list[dict], classes: list[str] | None = None) -> dict:
    """Build mAP / per-class AP / health series from decision-log records (chronological)."""
    points = []
    for r in records:
        if r.get("event") != "eval_ingest":
            continue
        ex = r.get("extra", {}) or {}
        points.append({"ts": r.get("ts_utc"), "mAP": ex.get("mAP"),
                       "per_class_ap": ex.get("per_class_ap", {}) or {},
                       "eval_set_hash": ex.get("eval_set_hash"), "loc_gap": ex.get("loc_gap")})

    hashes = [p["eval_set_hash"] for p in points if p["eval_set_hash"]]
    eval_set_changed = len(set(hashes)) > 1

    all_cls = classes or sorted({c for p in points for c in p["per_class_ap"]})
    per_class = {c: [(p["ts"], p["per_class_ap"].get(c)) for p in points] for c in all_cls}
    per_class_delta = {}
    for c, series in per_class.items():
        vals = [v for _t, v in series if v is not None]
        if len(vals) >= 2:
            per_class_delta[c] = round(vals[-1] - vals[0], 4)  # first -> last (only meaningful if eval set fixed)

    health_series = [(r.get("ts_utc"), (r.get("extra", {}) or {}).get("health"))
                     for r in records if r.get("event") == "weakness_report"]
    return {
        "n_evals": len(points),
        "mAP_series": [(p["ts"], p["mAP"]) for p in points],
        "per_class": per_class,
        "per_class_delta": per_class_delta,
        "health_series": health_series,
        "eval_set_changed": eval_set_changed,
        "note": ("⚠ eval set 在期間內變過 → 各類 AP delta 不可直接比較(可能只是 val 變簡單)"
                 if eval_set_changed else "單一 eval set → AP delta 可比較"),
    }
