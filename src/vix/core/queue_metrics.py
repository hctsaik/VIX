"""Queue hit-rate — did VIX's "label/inspect these" suggestions turn out right? (gt-consistency #1).

VIX emits several suggestion queues (error-mine, hardneg, the weakness-report queue, bank-audit
hard-positives). Each is a PROXY today — ranked by suspicion, never scored. Once humans resolve some
of those ids (confirm -> golden / false_alarm / dismiss), we can finally measure each queue's
PRECISION: of the suggestions you acted on, how many did the queue get right? That turns proxy
queues into a self-tuning, trustable signal and lets the engineer see the trend cycle over cycle.

Honest by construction:
- Only ids resolved AFTER an emission count (a resolution before the suggestion isn't its result).
- Only RESOLVED ids count toward precision (we cannot know about ids nobody acted on) — `coverage`
  reports how much of a queue was acted on; `insufficient` flags too-few-resolved to trust.
- "hit" depends on what the queue PREDICTED: a 'wrong' queue (hardneg) is right when the id was
  rejected; a 'defect' queue (bank hard-positive) is right when confirmed; a 'label' queue is
  scored as acted-on (any resolution), so its precision == coverage.

Pure / stdlib-only.
"""

from __future__ import annotations


def _is_hit(predict: str, outcome: str) -> bool:
    if predict == "wrong":
        return outcome == "rejected"
    if predict == "defect":
        return outcome == "confirmed"
    return True  # 'label' queue: any resolution = the human found it worth acting on


def hit_rate(emissions: list[dict], resolutions: list[dict], min_resolved: int = 5) -> list[dict]:
    """emissions: [{queue, ids, predict, seq}]; resolutions: [{id, outcome in {confirmed,rejected}, seq}].
    Returns one record per queue: emitted / resolved / hits / precision / coverage / per-emission
    trend / insufficient."""
    by_id: dict[str, list] = {}
    for r in resolutions:
        by_id.setdefault(r["id"], []).append((r["seq"], r["outcome"]))
    for v in by_id.values():
        v.sort()

    agg: dict[str, dict] = {}
    for e in emissions:
        q = e.get("queue", "?")
        predict = e.get("predict", "label")
        a = agg.setdefault(q, {"queue": q, "predict": predict, "emitted": 0, "resolved": 0, "hits": 0, "trend": []})
        eh = er = 0
        for i in e.get("ids", []):
            a["emitted"] += 1
            outcome = next((o for (s, o) in by_id.get(i, ()) if s > e["seq"]), None)  # resolved AFTER emission
            if outcome is None:
                continue
            a["resolved"] += 1
            er += 1
            hit = _is_hit(predict, outcome)
            a["hits"] += int(hit)
            eh += int(hit)
        if er:
            a["trend"].append(round(eh / er, 3))  # this emission's precision (trend over cycles)

    out = []
    for a in agg.values():
        a["precision"] = round(a["hits"] / a["resolved"], 3) if a["resolved"] else None
        a["coverage"] = round(a["resolved"] / a["emitted"], 3) if a["emitted"] else None
        a["insufficient"] = a["resolved"] < min_resolved
        out.append(a)
    out.sort(key=lambda r: r["queue"])
    return out
