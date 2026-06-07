"""Two-tier --help: core verbs shown, long-tail hidden, ALL verbs still registered/dispatchable."""

import argparse
import contextlib
import io

from vix.cli import _build_parser


def _help_text():
    p = _build_parser()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        with contextlib.suppress(SystemExit):
            p.parse_args(["--help"])
    return buf.getvalue()


def _subparsers():
    p = _build_parser()
    return next(a for a in p._actions if isinstance(a, argparse._SubParsersAction))


def test_default_help_shows_core_hides_longtail():
    h = _help_text()
    for v in ("diagnose", "import-labels", "eval-run", "ingest", "gate", "weakness-report", "export"):
        assert v in h, f"core verb {v} missing from default help"
    for v in ("spc", "parity", "cost-gate", "bank-audit", "capacity", "adapt-embedding", "batch-ledger"):
        assert v not in h, f"long-tail verb {v} leaked into default help"


def test_all_verbs_still_registered_and_dispatchable():
    choices = _subparsers().choices
    for v in ("spc", "parity", "cost-gate", "bank-audit", "adapt-embedding", "eval-ingest",
              "diagnose", "import-labels", "eval-run"):
        assert v in choices, f"{v} not registered (would break dispatch)"
    assert len(choices) >= 70  # zero deletions: full surface intact


def test_new_onramp_verbs_present():
    choices = _subparsers().choices
    assert {"diagnose", "import-labels", "eval-run"} <= set(choices)
