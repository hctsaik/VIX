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


def pre_train_gate(
    *,
    n_review_open: int = 0,
    golden_train_overlap: int = 0,
    under_represented: list[str] | None = None,
    drift_triggered: bool = False,
    audit_chain_intact: bool = True,
    n_golden: int | None = None,
    eval_golden_overlap: int = 0,
) -> GateResult:
    under_represented = under_represented or []
    reasons: list[str] = []
    if n_golden is not None and n_golden == 0:
        reasons.append("尚無已確認 golden 資料,無法訓練(先 vix ingest --golden 並 calibrate)")
    if eval_golden_overlap > 0:
        reasons.append(f"eval 與 golden 重疊 {eval_golden_overlap} 筆(held-out 評估集洩漏進訓練)")
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
