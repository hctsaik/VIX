"""Round 14 補強的回歸測試:
  - 並行 append 不丟資料(修 R13 引入的 Windows 鎖競態 PermissionError)
  - gate 對零 golden 直接 NO-GO(不再假 GO)
  - 未校準類別 route 進 review(不再靜默 PASS)
  - CLI 對預期錯誤回乾淨訊息 + exit 2(不噴 traceback)
"""

import threading

from vix.cli import main
from vix.core.decision_log import DecisionLog
from vix.core.gate import pre_train_gate
from vix.core.threshold import ClassThreshold, ThresholdPolicy
from vix.types import Routing


def test_decision_log_concurrent_appends_no_loss(tmp_path):  # AF2 (the R13-introduced regression)
    dl = DecisionLog(tmp_path / "log.jsonl")
    n_threads, per = 6, 20

    def worker(tid):
        for i in range(per):
            dl.append("route", vix_hash=f"t{tid}_{i}", reviewer_id=f"r{tid}", decision="review")

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    recs = dl.read_all()
    assert len(recs) == n_threads * per  # every append landed; none silently lost
    assert dl.verify_chain() is True  # and the chain is still intact


def test_gate_blocks_on_zero_golden():  # AF1
    assert pre_train_gate(n_golden=0).verdict == "NO-GO"
    assert pre_train_gate(n_golden=5).verdict == "GO"
    assert pre_train_gate().verdict == "GO"  # None == not provided -> no check (backward compat)


def test_uncalibrated_class_routes_to_review():  # AF1
    pol = ThresholdPolicy({"x": ClassThreshold(0.0, float("inf"), 0)})  # golden-tagged but no evidence
    r = pol.route("x", conf=0.99, knn_dist=0.0)
    assert r.decision == Routing.REVIEW  # fail-safe to review, not a silent PASS


def test_cli_clean_error_no_traceback(tmp_path):  # usability
    rc = main(["--workspace", str(tmp_path / "ws"), "--adapter", "memory", "route"])
    assert rc == 2  # route before calibrate -> one clean line + exit 2, not a Python traceback
