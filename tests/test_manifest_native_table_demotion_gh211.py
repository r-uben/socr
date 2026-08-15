"""GH-211 MAJOR-1 regression: demoting a native table's trust must not

silently revert the page's TEXT to the pre-extraction ``native_text``
snapshot. Content appended to ``PageOutput.text`` after native capture --
notably GH-36b's equation LaTeX sidecar, spliced in place -- must survive
the demotion that ``--native-only`` (or the ordinary native-table-structure
gate) applies when a table region fails verification.
"""

from __future__ import annotations

import pytest

fitz = pytest.importorskip("fitz")  # PyMuPDF; skip whole module if unavailable

from socr.core.document import DocumentHandle  # noqa: E402
from socr.core.manifest import _winning_page_output  # noqa: E402
from socr.core.result import FailureMode, PageOutput, PageStatus  # noqa: E402
from socr.core.state import DocumentState  # noqa: E402


def _make_pdf(path):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "native table page")
    doc.save(str(path))
    doc.close()
    return path


def _state_with_demoted_native_table(tmp_path, *, sidecar_appended: bool):
    pdf_path = _make_pdf(tmp_path / "doc.pdf")
    handle = DocumentHandle.from_path(pdf_path)
    state = DocumentState(handle=handle)

    native_text = "| A | B |\n| --- | --- |\n| 1 | 2 |"
    live_text = native_text
    if sidecar_appended:
        # Mirrors _attach_equation_latex_sidecars' 1C splice: append, never
        # replace, directly onto the live PageOutput object (orchestrator.py
        # ``po.text = po.text + "\n\n" + result.sidecar_block``).
        live_text = native_text + "\n\n$$E = mc^2$$"

    from socr.core.state import PageState

    ps = PageState(page_num=1)
    ps.is_born_digital = True
    ps.native_text = native_text
    ps.native_table_unverifiable = True  # GH-211: --native-only distrust

    # The native PageOutput is demoted at CREATION time (audit_passed=False)
    # by both the agentic and non-agentic --native-only paths -- this is not
    # a later in-place demotion. Its .text may already carry appended content.
    demoted = PageOutput(
        page_num=1,
        text=live_text,
        status=PageStatus.WARNING,
        engine="native",
        audit_passed=False,
        failure_mode=FailureMode.NATIVE_TABLE_STRUCTURE_FAILED,
    )
    ps.attempts.append(demoted)
    # NOTE: ``best_output`` is deliberately NOT set here. ``apply_result`` only
    # promotes an ``audit_passed`` output, and ``_score_per_page`` clears it
    # outright when it demotes under --native-only, so a demoted native page
    # genuinely has ``best_output is None`` on the deterministic path. An
    # earlier revision of this fixture planted ``ps.best_output = demoted``,
    # which modelled the agentic path only and made the test pass against code
    # that still dropped the sidecar.
    state.pages[1] = ps
    return state


def test_demoted_native_table_preserves_appended_equation_sidecar(tmp_path) -> None:
    state = _state_with_demoted_native_table(tmp_path, sidecar_appended=True)

    winning = _winning_page_output(state, 1)

    assert "$$E = mc^2$$" in winning.text, (
        "demoting the native table's trust discarded content appended after "
        "native capture (GH-36b equation sidecar) -- silent content loss"
    )
    assert winning.text.startswith("| A | B |")
    assert winning.status is PageStatus.WARNING
    assert winning.audit_passed is False
    assert winning.failure_mode is FailureMode.NATIVE_TABLE_STRUCTURE_FAILED


def test_demoted_native_table_without_sidecar_ships_native_text(tmp_path) -> None:
    state = _state_with_demoted_native_table(tmp_path, sidecar_appended=False)

    winning = _winning_page_output(state, 1)

    assert winning.text == "| A | B |\n| --- | --- |\n| 1 | 2 |"
    assert winning.status is PageStatus.WARNING
    assert winning.audit_passed is False


def test_sidecar_survives_the_real_apply_result_lifecycle(tmp_path) -> None:
    """Build the page through ``apply_result``, not by planting ``best_output``.

    ``apply_result`` promotes only ``audit_passed`` outputs (state.py), so a
    demoted native table leaves ``best_output is None`` and the appended sidecar
    lives ONLY in ``attempts``. Reading the fallback from ``best_output`` drops
    it here while every hand-planted fixture stays green.
    """
    pdf_path = _make_pdf(tmp_path / "doc.pdf")
    handle = DocumentHandle.from_path(pdf_path)
    state = DocumentState(handle=handle)

    from socr.core.result import EngineResult
    from socr.core.state import PageState

    native_text = "| A | B |\n| --- | --- |\n| 1 | 2 |"

    ps = PageState(page_num=1)
    ps.is_born_digital = True
    ps.native_text = native_text
    ps.native_table_unverifiable = True
    state.pages[1] = ps

    demoted = PageOutput(
        page_num=1,
        text=native_text + "\n\n$$E = mc^2$$",
        status=PageStatus.WARNING,
        engine="native",
        audit_passed=False,
        failure_mode=FailureMode.NATIVE_TABLE_STRUCTURE_FAILED,
    )
    state.apply_result(EngineResult(document_path=pdf_path, engine="native", pages=[demoted]))

    # The precondition that made the original fix ineffective.
    assert ps.best_output is None, "apply_result must not promote a demoted output"
    assert ps.attempts and "$$E = mc^2$$" in ps.attempts[-1].text

    winning = _winning_page_output(state, 1)

    assert "$$E = mc^2$$" in winning.text, (
        "the appended equation sidecar was dropped on the deterministic "
        "--native-only path, where best_output is never promoted"
    )
    assert winning.status is PageStatus.WARNING
    assert winning.audit_passed is False


def test_sidecar_survives_best_output_cleared_by_scoring(tmp_path) -> None:
    """``_score_per_page`` sets ``best_output = None`` when it demotes in place.

    So even a page that DID have a promoted ``best_output`` loses that pointer
    before assemble. The sidecar must still ship from ``attempts``.
    """
    state = _state_with_demoted_native_table(tmp_path, sidecar_appended=True)
    ps = state.pages[1]

    # Mirror the agentic shape (promoted), then the scoring demotion that
    # clears the pointer -- orchestrator._score_per_page under --native-only.
    ps.best_output = ps.attempts[-1]
    ps.best_output = None

    winning = _winning_page_output(state, 1)

    assert "$$E = mc^2$$" in winning.text, (
        "clearing best_output during scoring reverted the page to its "
        "pre-sidecar native_text snapshot"
    )


def test_ocr_attempt_text_never_substitutes_for_native(tmp_path) -> None:
    """The recovery must only accept a NATIVE attempt that EXTENDS the snapshot.

    A longer non-native attempt, or a native attempt that rewrote rather than
    appended, must not be spliced in as the native reading.
    """
    state = _state_with_demoted_native_table(tmp_path, sidecar_appended=False)
    ps = state.pages[1]
    native_text = ps.native_text

    ps.attempts.append(
        PageOutput(
            page_num=1,
            text=native_text + "\n\nOCR EXTRA",
            status=PageStatus.SUCCESS,
            engine="qwen-local",  # not native -> ineligible
            audit_passed=True,
        )
    )
    ps.attempts.append(
        PageOutput(
            page_num=1,
            text="COMPLETELY DIFFERENT AND LONGER TEXT " * 3,
            status=PageStatus.SUCCESS,
            engine="native",  # native but does not extend the snapshot
            audit_passed=False,
        )
    )

    winning = _winning_page_output(state, 1)

    assert winning.text == native_text
    assert "OCR EXTRA" not in winning.text
    assert "COMPLETELY DIFFERENT" not in winning.text
