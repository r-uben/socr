"""GH-658a: a default install has no scanned-table witness, silently.

``pytesseract`` was declared nowhere in ``pyproject.toml`` -- not a core
dependency, not an extra -- so a fresh install always lands on
``WITNESS_PACKAGE_MISSING`` (``source_evidence.py``) and a scanned table page
silently has no evidence source. This ticket is packaging + visibility only:

1. declare the witness as an optional extra so an operator has something to
   install;
2. emit a loud (WARNING, not the pre-existing DEBUG), once-per-run notice
   naming which of the two distinct gaps (package vs binary) is missing.

No behaviour change to the fail-closed verdict itself -- pinned here by
comparing the ``SourceEvidenceResult`` produced with and without the warning
having already fired.
"""

from __future__ import annotations

import logging
import tomllib
from pathlib import Path

from socr.tables.source_evidence import (
    SourceEvidenceBundle,
    WITNESS_BINARY_MISSING,
    WITNESS_PACKAGE_MISSING,
    WITNESS_STATE_MESSAGES,
    _WARNED_WITNESS_STATES,
    verify_table_tokens,
)
from socr.tables.source_evidence import TableTokens as _TableTokens

_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _tokens() -> _TableTokens:
    return _TableTokens(has_numeric=True)


def _bundle(state: str) -> SourceEvidenceBundle:
    return SourceEvidenceBundle(witness_state=state)


# --------------------------------------------------------------------------
# AC1: the extra is declared; a plain install stays untouched.
# --------------------------------------------------------------------------


def test_scanned_extra_declares_pytesseract() -> None:
    data = tomllib.loads(_PYPROJECT.read_text())
    extras = data["project"]["optional-dependencies"]
    assert "scanned" in extras, "pyproject.toml declares no 'scanned' extra"
    assert any(dep.startswith("pytesseract") for dep in extras["scanned"]), (
        f"the 'scanned' extra does not declare pytesseract: {extras['scanned']}"
    )


def test_core_dependencies_do_not_pull_in_pytesseract() -> None:
    """A plain install must be unchanged in behaviour (AC1) -- so the witness
    stays opt-in, not a transitive core dependency."""
    data = tomllib.loads(_PYPROJECT.read_text())
    core_deps = data["project"]["dependencies"]
    assert not any("pytesseract" in dep for dep in core_deps), (
        "pytesseract leaked into the core dependency list; the extra is meant "
        "to be the only way to install it"
    )


# --------------------------------------------------------------------------
# AC2/AC3: the warning fires once per run, and names the two gaps distinctly.
# --------------------------------------------------------------------------


def test_missing_package_warns_exactly_once_across_many_pages(caplog) -> None:
    _WARNED_WITNESS_STATES.discard(WITNESS_PACKAGE_MISSING)
    try:
        with caplog.at_level(logging.WARNING, logger="socr.tables.source_evidence"):
            for _ in range(5):  # stands in for a 5-page scan, all unwitnessed
                verify_table_tokens(_bundle(WITNESS_PACKAGE_MISSING), _tokens())
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_records) == 1, (
            f"expected exactly one WARNING across 5 pages, got {len(warning_records)}: "
            f"{[r.message for r in warning_records]}"
        )
        assert WITNESS_PACKAGE_MISSING in _WARNED_WITNESS_STATES  # state really recorded
    finally:
        _WARNED_WITNESS_STATES.discard(WITNESS_PACKAGE_MISSING)


def test_missing_package_and_missing_binary_each_get_their_own_first_warning(caplog) -> None:
    """AC3: the two gaps need two different operator actions, so a run that
    hits both must not have the second one silenced by the first."""
    _WARNED_WITNESS_STATES.discard(WITNESS_PACKAGE_MISSING)
    _WARNED_WITNESS_STATES.discard(WITNESS_BINARY_MISSING)
    try:
        with caplog.at_level(logging.WARNING, logger="socr.tables.source_evidence"):
            verify_table_tokens(_bundle(WITNESS_PACKAGE_MISSING), _tokens())
            verify_table_tokens(_bundle(WITNESS_BINARY_MISSING), _tokens())
        messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert len(messages) == 2, f"expected one warning per distinct gap, got: {messages}"
        assert WITNESS_STATE_MESSAGES[WITNESS_PACKAGE_MISSING] in messages[0]
        assert WITNESS_STATE_MESSAGES[WITNESS_BINARY_MISSING] in messages[1]
        assert messages[0] != messages[1], (
            "the package-missing and binary-missing warnings read identically; "
            "an operator cannot tell which of the two different fixes applies"
        )
    finally:
        _WARNED_WITNESS_STATES.discard(WITNESS_PACKAGE_MISSING)
        _WARNED_WITNESS_STATES.discard(WITNESS_BINARY_MISSING)


def test_warning_names_the_package_and_the_extra() -> None:
    """The message an operator actually reads must say what to install."""
    assert "pytesseract" in WITNESS_STATE_MESSAGES[WITNESS_PACKAGE_MISSING]
    assert "tesseract" in WITNESS_STATE_MESSAGES[WITNESS_BINARY_MISSING]


# --------------------------------------------------------------------------
# AC4: the warning is purely additive -- the fail-closed verdict is unchanged.
# --------------------------------------------------------------------------


def test_the_warning_does_not_change_the_verdict() -> None:
    """Same bundle, same tokens; the only difference is whether this state was
    already warned about in this process. The verdict must not move."""
    _WARNED_WITNESS_STATES.discard(WITNESS_PACKAGE_MISSING)
    try:
        first = verify_table_tokens(_bundle(WITNESS_PACKAGE_MISSING), _tokens())
        # By now the state has been recorded as warned; a second call takes
        # the "already warned" branch instead.
        assert WITNESS_PACKAGE_MISSING in _WARNED_WITNESS_STATES
        second = verify_table_tokens(_bundle(WITNESS_PACKAGE_MISSING), _tokens())
        assert first.verifiable == second.verifiable
        assert first.passed == second.passed
        assert first.reason == second.reason
        assert first.cause == second.cause
    finally:
        _WARNED_WITNESS_STATES.discard(WITNESS_PACKAGE_MISSING)
