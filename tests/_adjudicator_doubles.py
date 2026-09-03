"""Blind-cell adjudicator doubles, shared by the P1 ladder-terminal tests.

The adjudicator's TRANSPORT is exercised in ``test_table_rung_ollama.py``
against the module's own ``_post_chat`` seam. What every OTHER file needs is a
callable of the guard's shape whose ANSWER is controlled, so a test can pin
which terminal the ruled chain reaches for a given reader verdict.

Why these exist at all (GH-575, cold review round 1, finding 1): after the
ruling, a reader rejection only reaches ``TABLE_WITHHELD`` when a blind third
reader actually looked at the crop and read a DIFFERENT token. With no
adjudicator the same rejection ends ``TABLE_UNVERIFIED``. So a test that means
"the readers rejected this table and it was withheld" must supply an
adjudicator that mismatches -- otherwise it is pinning the no-adjudicator
path and says nothing about rejection at all.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from socr.judge.table_rung_ollama import BlindCellResult, adjudicator_rung_id

#: Identity these doubles report. A NON-default model on purpose: it is the
#: only way a test can tell the metered/journalled identity apart from the
#: configured default (cold review round 1, finding 5).
DOUBLE_MODEL = "double-adjudicator:cloud"
DOUBLE_RUNG_ID = adjudicator_rung_id(DOUBLE_MODEL)


def _tag(fn):
    fn.rung_kind = "adjudicator"
    fn.rung_id = DOUBLE_RUNG_ID
    fn.executing = DOUBLE_MODEL
    return fn


def mismatching_adjudicator(token: str = "∅ not this value"):
    """Reads a different token for every requested cell: active disagreement.

    This is the ONLY answer that may withhold a rejected table's bytes.
    """
    calls: list[list[str]] = []

    @_tag
    def _adj(crop_path: Path | None, cell_refs: Sequence[str]) -> BlindCellResult:
        calls.append([str(r) for r in cell_refs])
        return BlindCellResult(
            rung=DOUBLE_RUNG_ID, ok=True, tokens={str(r): token for r in cell_refs}
        )

    _adj.calls = calls
    return _adj


def agreeing_adjudicator(tokens: dict[str, str]):
    """Reads exactly ``tokens``: clears the table when they all agree."""
    calls: list[list[str]] = []

    @_tag
    def _adj(crop_path: Path | None, cell_refs: Sequence[str]) -> BlindCellResult:
        calls.append([str(r) for r in cell_refs])
        return BlindCellResult(
            rung=DOUBLE_RUNG_ID,
            ok=True,
            tokens={str(r): tokens.get(str(r), "") for r in cell_refs},
        )

    _adj.calls = calls
    return _adj


def unavailable_adjudicator(*, refusal: bool = False, error: str = "simulated outage"):
    """An OUTAGE (or an external REFUSAL): latches, never withholds."""
    calls: list[list[str]] = []

    @_tag
    def _adj(crop_path: Path | None, cell_refs: Sequence[str]) -> BlindCellResult:
        calls.append([str(r) for r in cell_refs])
        return BlindCellResult(
            rung=DOUBLE_RUNG_ID, ok=False, error=error, unavailable=True, refusal=refusal
        )

    _adj.calls = calls
    return _adj


def defective_adjudicator(error: str = "unusable answer"):
    """A DETERMINISTIC defect: never latches, never withholds."""
    calls: list[list[str]] = []

    @_tag
    def _adj(crop_path: Path | None, cell_refs: Sequence[str]) -> BlindCellResult:
        calls.append([str(r) for r in cell_refs])
        return BlindCellResult(rung=DOUBLE_RUNG_ID, ok=False, error=error, unavailable=False)

    _adj.calls = calls
    return _adj
