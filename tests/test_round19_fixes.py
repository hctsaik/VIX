"""Round 19 補強的回歸測試:
  - coverage --target 對「均衡但低於絕對目標」的類別也報缺額(AK4 真 bug)
  - decision log .hwm fail-closed:非空帳本但錨點缺失視為可疑
"""

import numpy as np

from vix.core.analytics import EmbItem, coverage_gaps
from vix.core.decision_log import DecisionLog


def _it(id, label):
    return EmbItem(id, label, np.array([1.0, 0.0]), 1.0)


def test_coverage_target_reports_shortfall_for_balanced_classes():  # AK4
    items = [_it(f"a{i}", "a") for i in range(3)] + [_it(f"b{i}", "b") for i in range(3)]
    gaps = coverage_gaps(items, target=10)  # balanced (3/3) but both below an absolute target
    assert gaps["a"]["under_represented"] and gaps["a"]["need"] == 7
    assert gaps["b"]["under_represented"] and gaps["b"]["need"] == 7

    gaps_default = coverage_gaps(items)  # no target -> relative-imbalance heuristic, balanced not flagged
    assert not gaps_default["a"]["under_represented"]


def test_hwm_fail_closed(tmp_path):  # AK10
    dl = DecisionLog(tmp_path / "log.jsonl")
    dl.append("a")
    dl.append("b")
    assert dl.is_truncated() is False

    dl.path.with_suffix(dl.path.suffix + ".hwm").unlink()  # delete the anchor on a non-empty ledger
    assert dl.is_truncated() is True  # fail-closed -> suspicious

    empty = DecisionLog(tmp_path / "empty.jsonl")
    assert empty.is_truncated() is False  # empty ledger with no anchor is fine
