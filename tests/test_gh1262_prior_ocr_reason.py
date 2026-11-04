"""#1262: a scan with an earlier OCR layer says WHICH reading shipped, without changing its status.

The #1156 clause marks every such page UNKNOWN and never CHECKED: one reading of a scan is
one reading, whoever produced it. Before #1262 the reason always said "earlier OCR, not
re-read", including when a model HAD re-read the page from its image and that reading
shipped (71 of 103 marked pages in task 0 of the Fed post-2000 Rorqual run). Now:

* the prior layer shipped        -> "earlier OCR, not re-read" (unchanged);
* a model reading of the image   -> "read once from the image, no second reading agreed".

The status, the failure mode and the figure-extraction skip are pinned identical for both,
so only the explanation moves.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import socr
from socr.core.manifest import (
    PRIOR_OCR_MARK_REASON,
    PRIOR_OCR_MODEL_MARK_REASON,
    _select_and_finalize_page,
    page_ships_its_own_page_image,
)
from socr.core.result import FailureMode, PageOutput, PageStatus
from socr.core.state import DocumentHandle, DocumentState, PageState
from socr.core.verdict import PRIOR_OCR_MODEL_REASON, PRIOR_OCR_NATIVE_REASON


def test_loaded_source_is_this_checkout() -> None:
    assert Path(socr.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1] / "src")


def _scan_with_prior_layer(*, model_reading: bool) -> DocumentState:
    """A scanned page whose text layer is an earlier OCR run; optionally an accepted
    model reading of its image is the winner."""
    state = DocumentState(handle=DocumentHandle(path=Path("x.pdf"), page_count=1))
    p = PageState(page_num=1)
    p.is_born_digital = True
    p.native_text = "Raised discount rates from 3 to 3 Y, per cent and $400 million"
    p.prior_ocr_layer = True
    p.prior_ocr_layer_suspect = True
    p.needs_ocr_enhancement = True
    if model_reading:
        accepted = PageOutput(
            page_num=1,
            text="A MODEL READING OF THE PAGE IMAGE",
            status=PageStatus.SUCCESS,
            engine="qwen",
            audit_passed=True,
            failure_mode=FailureMode.NONE,
        )
        p.attempts = [accepted]
        p.best_output = accepted
    state.pages[1] = p
    return state


@pytest.mark.parametrize("model_reading", [False, True], ids=["prior-layer", "model-reading"])
def test_the_reason_names_what_shipped(model_reading: bool) -> None:
    state = _scan_with_prior_layer(model_reading=model_reading)
    text = _select_and_finalize_page(state, 1).output.text
    expected, other = (
        (PRIOR_OCR_MODEL_REASON, PRIOR_OCR_NATIVE_REASON)
        if model_reading
        else (PRIOR_OCR_NATIVE_REASON, PRIOR_OCR_MODEL_REASON)
    )
    assert expected in text
    assert other not in text


def test_the_status_is_the_same_whichever_reading_shipped() -> None:
    """The change is the explanation only: both endings stay demoted, never CHECKED."""
    outcomes = []
    for model_reading in (False, True):
        out = _select_and_finalize_page(
            _scan_with_prior_layer(model_reading=model_reading), 1
        ).output
        assert out.status is not PageStatus.SUCCESS
        outcomes.append(out.status)
    assert outcomes[0] is outcomes[1]
    model_out = _select_and_finalize_page(_scan_with_prior_layer(model_reading=True), 1).output
    assert model_out.failure_mode is FailureMode.PRIOR_OCR_LAYER_UNVERIFIED
    assert "unknown=UNKNOWN" not in model_out.text


@pytest.mark.parametrize("model_reading", [False, True], ids=["prior-layer", "model-reading"])
def test_figure_extraction_still_skips_the_scan(model_reading: bool) -> None:
    """The figure-extraction skip is keyed on the page's own mark. A reworded mark that it
    did not recognise would let the extractor caption the scan's raster as a figure."""
    state = _scan_with_prior_layer(model_reading=model_reading)
    text = _select_and_finalize_page(state, 1).output.text
    assert page_ships_its_own_page_image(state.pages[1], text)


def test_the_marks_are_what_the_page_actually_prints() -> None:
    """The mark renderer replaces ``;`` inside a reason, so a reason containing one would
    never match its own shipped form. Both reasons must survive rendering byte-for-byte."""
    for model_reading, mark in (
        (False, PRIOR_OCR_MARK_REASON),
        (True, PRIOR_OCR_MODEL_MARK_REASON),
    ):
        text = _select_and_finalize_page(
            _scan_with_prior_layer(model_reading=model_reading), 1
        ).output.text
        assert mark in text
    assert ";" not in PRIOR_OCR_MODEL_REASON


@pytest.mark.parametrize(
    ("engine", "reason"),
    [("native", PRIOR_OCR_NATIVE_REASON), ("qwen", PRIOR_OCR_MODEL_REASON)],
)
def test_the_mode_table_names_what_shipped(engine: str, reason: str) -> None:
    """When only the failure MODE carries the demotion (the flag that raises the clause is
    not on the page state, as on a rewrite or resume pass), the mode table's row speaks for
    the page. It must name the same reading the clause would."""
    from socr.core.verdict import derive_clauses

    out = PageOutput(
        page_num=1,
        text="a reading",
        status=PageStatus.WARNING,
        engine=engine,
        audit_passed=engine != "native",
        failure_mode=FailureMode.PRIOR_OCR_LAYER_UNVERIFIED,
    )
    reasons = [
        c.reason
        for c in derive_clauses(out, PageState(page_num=1))
        if c.check.value == "prior_ocr_layer"
    ]
    assert reasons == [reason]
