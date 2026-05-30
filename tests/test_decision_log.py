from vix.core.decision_log import DecisionLog


def test_append_and_chain(tmp_path):
    log = DecisionLog(tmp_path / "d.jsonl")
    log.append("route", vix_hash="h1", decision="review", scores={"conf": 0.5})
    log.append("review", vix_hash="h1", decision="bubble", reviewer_id="u1")
    recs = log.read_all()
    assert len(recs) == 2
    assert recs[0]["prev_hash"] == ""
    assert recs[1]["prev_hash"] == recs[0]["entry_hash"]
    assert log.verify_chain() is True


def test_tamper_is_detected(tmp_path):
    p = tmp_path / "d.jsonl"
    log = DecisionLog(p)
    log.append("route", vix_hash="h1")
    log.append("route", vix_hash="h2")
    assert log.verify_chain() is True

    lines = p.read_text(encoding="utf-8").splitlines()
    lines[0] = lines[0].replace("h1", "hX")  # edit a committed record
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert log.verify_chain() is False
