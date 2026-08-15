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
    ps.best_output = demoted
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
