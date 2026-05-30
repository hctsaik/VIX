"""One-click dataset health report (S10) — pure aggregation + markdown render.

Turns the analytics outputs into a single artifact a CV engineer / PM can read in
under a minute: class distribution, duplicate rate, review ratio, suspected label
errors, coverage gaps, and a diff vs the previous report.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def build_report(
    *,
    version: str,
    total: int,
    class_dist: dict[str, int],
    pass_count: int = 0,
    review_count: int = 0,
    duplicate_groups: list[list[str]] | None = None,
    label_issues: list | None = None,
    coverage: dict[str, dict] | None = None,
    novel_fraction: float | None = None,
    prev: dict | None = None,
    n_batches: int | None = None,
    suggestions: list[str] | None = None,
    gate_verdict: str | None = None,
    embedding_backend: str | None = None,
) -> dict:
    duplicate_groups = duplicate_groups or []
    label_issues = label_issues or []
    coverage = coverage or {}

    redundant = sum(len(g) - 1 for g in duplicate_groups)
    routed = pass_count + review_count
    dup_rate = (redundant / total) if total else 0.0
    review_ratio = (review_count / routed) if routed else 0.0
    label_rate = (len(label_issues) / total) if total else 0.0
    under = [c for c, v in coverage.items() if v.get("under_represented")]
    # Single 0-100 quality score: start at 100, subtract weighted penalties.
    quality_score = max(
        0.0,
        round(100.0 - dup_rate * 30 - label_rate * 30 - review_ratio * 20 - len(under) * 5, 1),
    )
    report = {
        "version": version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_images": total,
        "class_distribution": class_dist,
        "duplicate_rate": dup_rate,
        "n_duplicate_groups": len(duplicate_groups),
        "review_ratio": review_ratio,
        "n_suspected_label_errors": len(label_issues),
        "under_represented_classes": under,
        "novel_fraction": novel_fraction,
        "quality_score": quality_score,
    }
    if gate_verdict is not None:
        report["gate_verdict"] = gate_verdict
    if embedding_backend is not None:
        report["embedding_backend"] = embedding_backend
    if n_batches is not None:
        report["n_batches"] = n_batches
    if suggestions:
        report["suggested_next_steps"] = suggestions
    if prev:
        prev_dist = prev.get("class_distribution", {})
        report["diff"] = {
            "total_delta": total - prev.get("total_images", 0),
            "new_classes": sorted(set(class_dist) - set(prev_dist)),
            "removed_classes": sorted(set(prev_dist) - set(class_dist)),
        }
    return report


def render_markdown(report: dict) -> str:
    lines = [
        f"# VIX 資料集健檢報告 — {report.get('version', '')}",
        f"_generated: {report['generated_at']}_",
        "",
        f"## 資料集品質分數: **{report.get('quality_score', 0)} / 100**"
        + (f"  |  放行建議: **{report['gate_verdict']}**" if report.get("gate_verdict") else ""),
        "",
        f"- 影像總數: **{report['total_images']}**",
        f"- 重複率: **{report['duplicate_rate']:.1%}** ({report['n_duplicate_groups']} 群近似重複)",
        f"- 待覆核比例: **{report['review_ratio']:.1%}**",
        f"- 疑似標錯: **{report['n_suspected_label_errors']}** 筆",
    ]
    if report.get("novel_fraction") is not None:
        lines.append(f"- 新批次新增覆蓋率: **{report['novel_fraction']:.1%}**")
    lines += ["", "## 類別分布"]
    for c, n in sorted(report["class_distribution"].items(), key=lambda kv: -kv[1]):
        flag = "  ⚠️ 樣本不足" if c in report["under_represented_classes"] else ""
        lines.append(f"- `{c}`: {n}{flag}")
    if report.get("n_batches") is not None:
        lines.append(f"\n- 批次數: {report['n_batches']}")
    if report.get("embedding_backend"):
        lines.append(f"- Embedding 後端: `{report['embedding_backend']}`")
    if report.get("diff"):
        d = report["diff"]
        lines += [
            "",
            "## 與上一版差異",
            f"- 總數變化: {d['total_delta']:+d}",
            f"- 新增類別: {d['new_classes'] or '無'}",
            f"- 消失類別: {d['removed_classes'] or '無'}",
        ]
    if report.get("suggested_next_steps"):
        lines += ["", "## 建議的第一步"] + [f"- {s}" for s in report["suggested_next_steps"]]
    return "\n".join(lines) + "\n"


def write_report(report: dict, out_dir: str | Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "health_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "health_report.md").write_text(render_markdown(report), encoding="utf-8")
    return {"json": str(out_dir / "health_report.json"), "md": str(out_dir / "health_report.md")}
