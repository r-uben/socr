"""GH-671: a truncated ``best_output`` must not win by the PASSING_BEST_OUTPUT
short-circuit.

``_select_page_output_tagged`` returns early on any non-native ``best_output``
with ``audit_passed=True``, gated only on ``native_distrusted`` /
``native_text_shredded`` -- S1's own truncation filtering
(``_truncated_grid_reading_ids`` / ``structure_class_truncated_engines`` /
``_strict_grid_authored_pool``'s drop) runs LATER, under
``_reaches_structure_class_branch``, and is never consulted in the
short-circuit region. So a judge-cleared TRUNCATED model that happens to be
``best_output`` shipped unchallenged, and a COMPLETE alternative reading
elsewhere in the wider pool was never considered -- a silent truncation win
over a complete reading (the live #645 bulletin p2 shape, here with the
truncated candidate promoted to ``best_output`` itself rather than merely
present in ``p.attempts``).

#665's own pin (``test_cross_pool_truncated_strict_loses_to_complete_wide_pool_only``
in ``tests/tables/test_structure_check_truncated.py``) deliberately keeps
``best_output.audit_passed=False`` so S1 is reachable at all -- it structurally
cannot exercise this short-circuit. This file targets the real caller
(``_select_page_output_tagged``), not a lower helper, per the ticket's own
"hermetic pin at the real caller" requirement.

Hermetic: hand-built ``PageState``/``DocumentState`` objects, no provider
ladder, no ``_phase_agentic``, no ollama -- same seam
``tests/test_a1c_header_binding_unverified_surfacing.py`` and
``tests/tables/test_structure_check_truncated.py`` already use for this exact
selector.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from socr.core.document import DocumentHandle
from socr.core.manifest import (
    SelectionProvenance,
    _reaches_structure_class_branch,
    _select_page_output_tagged,
)
from socr.core.result import FailureMode, PageOutput, PageStatus
from socr.core.state import DocumentState, PageState

NATIVE_PROSE = "Table 1 below reports quarterly balances for 2018-2020."
REGION = (0.0, 0.0, 200.0, 100.0)

# Mixed style: a fully-bordered body row, then a final row missing its
# trailing pipe -- ``table_truncated``'s term (a), same shape
# ``tests/tables/test_structure_check_truncated.py`` uses for its own
# MIXED_STYLE_MD fixture.
TRUNCATED_MD = (
    "| Year | A | B |\n| :--- | :--- | :--- |\n| 2018 | 100.0 | 200.0 |\n| 2019 | 110.0 | 21"
)

COMPLETE_MD = (
    "| Year | A | B |\n"
    "| :--- | :--- | :--- |\n"
    "| 2018 | 100.0 | 200.0 |\n"
    "| 2019 | 110.0 | 210.0 |\n"
    "| 2020 | 120.0 | 220.0 |\n"
)


def _row_words(y: float, tokens: list[str]) -> list[tuple]:
    words = []
    x = 0.0
    for tok in tokens:
        words.append((x, y, x + 8.0, y + 10.0, tok))
        x += 12.0
    return words


NATIVE_WORDS: list[tuple] = (
    _row_words(10.0, ["2018", "100.0", "200.0"])
    + _row_words(30.0, ["2019", "110.0", "210.0"])
    + _row_words(50.0, ["2020", "120.0", "220.0"])
)


def _truncated_best_output(*, page_num: int = 1) -> PageOutput:
    """Judge-cleared (``audit_passed=True``) as the winning ``best_output`` --
    exactly the shape the ticket names: "truncated = judge-cleared model as
    best_output".
    """
    return PageOutput(
        page_num=page_num,
        text=TRUNCATED_MD,
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
        confidence=0.9,
        failure_mode=FailureMode.NONE,
    )


def _complete_wide_pool_attempt(*, page_num: int = 1) -> PageOutput:
    """Wider-pool-only: never judge-cleared (``audit_passed=False``), so it
    never qualifies for the strict grid-authored pool on its own -- only the
    ragged ``_grid_reading_attempt`` pool the corroboration fallback scores
    against.
    """
    return PageOutput(
        page_num=page_num,
        text=COMPLETE_MD,
        status=PageStatus.SUCCESS,
        engine="gemini",
        audit_passed=False,
        confidence=0.5,
        failure_mode=FailureMode.NONE,
    )


def _make_pdf(tmp_path: Path) -> Path:
    fitz = pytest.importorskip("fitz")
    path = tmp_path / "doc.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Table 1 prose")
    doc.save(str(path))
    doc.close()
    return path


def _state_with_page(tmp_path: Path, p: PageState) -> DocumentState:
    pdf_path = _make_pdf(tmp_path)
    state = DocumentState(handle=DocumentHandle.from_path(pdf_path))
    state.pages[1] = p
    return state


def _page(*, best_output: PageOutput, attempts: list[PageOutput]) -> PageState:
    p = PageState(page_num=1)
    p.is_born_digital = True
    p.native_text = NATIVE_PROSE
    p.has_tables = True
    p.attempts = attempts
    p.best_output = best_output
    p.native_words = NATIVE_WORDS
    p.detected_table_bboxes = [REGION]
    return p


def test_truncated_best_output_does_not_win_by_short_circuit(tmp_path: Path) -> None:
    """The ticket's own three assertions, at the real caller."""
    truncated = _truncated_best_output()
    complete = _complete_wide_pool_attempt()
    p = _page(best_output=truncated, attempts=[truncated, complete])
    state = _state_with_page(tmp_path, p)

    output, provenance = _select_page_output_tagged(state, 1)

    # The winner is the complete reading, not the truncated best_output.
    assert output.engine == "gemini"
    assert "120.0" in (output.text or "")  # 2020 row only COMPLETE_MD carries
    # Provenance is NOT a bare short-circuit that ignored truncation.
    assert provenance != SelectionProvenance.PASSING_BEST_OUTPUT
    # candidate_truncated fired for the truncated engine.
    truncated_events = [e for e in state.events if e.kind == "candidate_truncated"]
    assert any(e.engine == "qwen" for e in truncated_events)


def test_untruncated_passing_best_output_still_takes_the_short_circuit(
    tmp_path: Path,
) -> None:
    """Both directions matter (ticket requirement #2): an ordinary clean,
    non-truncated ``best_output`` must still take PASSING_BEST_OUTPUT. A fix
    that routes every clean model page down the long S1 path instead would be
    a different defect, invisible except as changed dispositions elsewhere.
    """
    clean = PageOutput(
        page_num=1,
        text=COMPLETE_MD,
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
        confidence=0.9,
        failure_mode=FailureMode.NONE,
    )
    p = _page(best_output=clean, attempts=[clean])
    state = _state_with_page(tmp_path, p)

    output, provenance = _select_page_output_tagged(state, 1)

    assert output is clean
    assert provenance == SelectionProvenance.PASSING_BEST_OUTPUT
    assert not [e for e in state.events if e.kind == "candidate_truncated"]


def test_the_two_mirrored_gates_agree_on_the_truncated_case(tmp_path: Path) -> None:
    """``_select_page_output_tagged``'s own short-circuit and
    ``_reaches_structure_class_branch`` duplicate the SAME three-condition
    gate (see both docstrings) and must never disagree on when the S1 branch
    is reachable -- ``_reaches_structure_class_branch``'s own docstring
    records what happened the one time they drifted (#269 BLOCKING 2): a
    page's real winner shipped via one branch while a document-level bucket,
    reading only the OTHER function, believed a different branch had fired.
    Pins the agreement directly rather than trusting the two edits stayed in
    sync by inspection.
    """
    truncated = _truncated_best_output()
    complete = _complete_wide_pool_attempt()
    p = _page(best_output=truncated, attempts=[truncated, complete])
    state = _state_with_page(tmp_path, p)

    # Selector side: does NOT take the short-circuit.
    _output, provenance = _select_page_output_tagged(state, 1)
    assert provenance != SelectionProvenance.PASSING_BEST_OUTPUT

    # Bucket side: agrees the S1 branch IS reachable for this same page.
    assert _reaches_structure_class_branch(p) is True
