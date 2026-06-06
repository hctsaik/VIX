"""Pre-training go/no-go gate (U7).

Integrates the otherwise-scattered checks into one auditable verdict before a
retrain: review backlog cleared, no golden/train leakage, class distribution
healthy, no unresolved drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GateResult:
    verdict: str  # "GO" | "NO-GO"
    reasons: list[str] = field(default_factory=list)
    checks: dict = field(default_factory=dict)


def regression_check(
    current_ap: dict,
    baseline_ap: dict,
    current_map: float,
    baseline_map: float,
    *,
    map_drop_thr: float = 0.02,
    protected: dict[str, float] | None = None,
    eval_support: dict[str, int] | None = None,
    min_support: int = 20,
    eval_set_changed: bool = False,
) -> tuple[list[str], list[str]]:
    """Challenge-guard core (model-loop-v2 T2/R6/R7). Returns (blocking_reasons, advisory).

    Fail-closed on protected classes: a protected class is ALWAYS evaluated for blocking and
    a protected class with < min_support eval GTs is itself a block ("can't certify"), never a
    silent pass. Small-support is downgraded to advisory ONLY for non-protected classes. A
    changed eval set yields advisory-only (an honest "comparison invalid", never a fake block
    or a fake pass)."""
    protected = protected or {}
    eval_support = eval_support or {}
    blocking: list[str] = []
    advisory: list[str] = []

    if eval_set_changed:
        advisory.append("eval set 已變(內容雜湊不符 baseline);本次不套用回歸阻擋,mAP 比較無效")
        return blocking, advisory

    drop = round(baseline_map - current_map, 4)
    if drop > map_drop_thr:
        blocking.append(
            f"整體 mAP 退步 {drop:.3f}(baseline {baseline_map:.3f}→{current_map:.3f},門檻 {map_drop_thr})"
        )

    for c, thr in protected.items():  # fail-closed: protected classes bypass the small-N advisory
        sup = eval_support.get(c, 0)
        if sup < min_support:
            blocking.append(f"受保護類別 {c} 的 eval 覆蓋不足(n={sup}<{min_support}),無法認證(fail-closed)")
            continue
        cdrop = round(baseline_ap.get(c, 0.0) - current_ap.get(c, 0.0), 4)
        if cdrop > thr:
            blocking.append(f"受保護類別 {c} AP 退步 {cdrop:.3f}(門檻 {thr})")

    for c in sorted(set(baseline_ap) | set(current_ap)):  # non-protected: small N -> advisory only
        if c in protected:
            continue
        if eval_support.get(c, 0) < min_support:
            advisory.append(f"類別 {c} eval 樣本少(n={eval_support.get(c, 0)}),AP delta 不穩,僅供參考")
    return blocking, advisory


def pre_train_gate(
    *,
    n_review_open: int = 0,
    golden_train_overlap: int = 0,
    under_represented: list[str] | None = None,
    drift_triggered: bool = False,
    audit_chain_intact: bool = True,
    n_golden: int | None = None,
    eval_golden_overlap: int = 0,
    backend_mixed: bool = False,
    extra_reasons: list[str] | None = None,   # opt-in: challenge-guard blocking reasons (no-op when None)
    extra_checks: dict | None = None,
) -> GateResult:
    under_represented = under_represented or []
    reasons: list[str] = []
    if n_golden is not None and n_golden == 0:
        reasons.append("尚無已確認 golden 資料,無法訓練(先 vix ingest --golden 並 calibrate)")
    if eval_golden_overlap > 0:
        reasons.append(f"eval 與 golden 重疊 {eval_golden_overlap} 筆(held-out 評估集洩漏進訓練)")
    if backend_mixed:
        reasons.append("偵測到混用 embedding 後端(pixel_fallback 與 DINOv2),距離門檻/趨勢不可比")
    if n_review_open > 0:
        reasons.append(f"{n_review_open} 筆仍待覆核未回寫")
    if golden_train_overlap > 0:
        reasons.append(f"golden/train 重疊 {golden_train_overlap} 筆(評估洩漏)")
    if under_represented:
        reasons.append(f"類別樣本不足: {under_represented}")
    if drift_triggered:
        reasons.append("偵測到未解決的定義漂移")
    if not audit_chain_intact:
        reasons.append("稽核鏈損毀(decision log 不可信,無法保證來源可追溯)")
    if extra_reasons:  # challenge-guard regression blocks (only present when a baseline exists)
        reasons.extend(extra_reasons)
    return GateResult(
        verdict="GO" if not reasons else "NO-GO",
        reasons=reasons,
        checks={
            "review_open": n_review_open,
            "golden_train_overlap": golden_train_overlap,
            "under_represented": under_represented,
            "drift_triggered": drift_triggered,
            "audit_chain_intact": audit_chain_intact,
            "n_golden": n_golden,
            **(extra_checks or {}),
        },
    )


def cost_gate(
    miss_rate: float,
    fa_rate: float,
    miss_cost: float,
    fa_cost: float,
    budget_per_unit: float,
) -> dict:
    """Asymmetric cost gate (concept #6).

    In a fab a miss (漏報, e.g. defect flows to exposure) costs orders of
    magnitude more than a false alarm (誤報, an extra manual re-check). A
    symmetric CR/FA gate ignores this; here the verdict is driven by expected
    cost per unit = miss_rate·miss_cost + fa_rate·fa_cost vs a budget.
    """
    expected = miss_rate * miss_cost + fa_rate * fa_cost
    return {
        "expected_cost_per_unit": round(expected, 4),
        "budget_per_unit": budget_per_unit,
        "miss_component": round(miss_rate * miss_cost, 4),
        "fa_component": round(fa_rate * fa_cost, 4),
        "verdict": "GO" if expected <= budget_per_unit else "NO-GO",
    }
