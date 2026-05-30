"""Pure VIX logic — no FiftyOne, no MongoDB, no GPU. Fully unit-testable.

Modules:
    manifest      structured, DVC-trackable record of every ingested image
    scorer        OutlierScorer (two-axis: YOLO confidence + DINOv2 kNN distance)
    threshold     ThresholdPolicy (per-class percentile routing)
    reference     FrozenReference (anchor centroid drift + label consistency guard)
    decision_log  append-only JSONL audit log with optional hash-chain
    exporter      one-way export to YOLO txt + data.yaml
"""
