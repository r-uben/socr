"""Shared fixtures for the native-first table tests (the ship gate and the rotated lane).

One forecast grid, the pipeline config these ``process()`` tests need, a fake
``route_page`` decision, the synthetic PDFs, and the flush/restore step of a resume
test. Importable by name (``tests/`` is on ``sys.path``), like ``p6_corpus_fixture``.
"""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from socr.core.config import EngineType, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.result import PageOutput, PageStatus
from socr.core.state import DocumentState
from socr.pipeline.agentic import PageDecision, ProviderAttempt
from socr.tables import ship_gate
from socr.tables.ship_gate import LineDirections

COL_XS = [90.0, 180.0, 270.0, 360.0, 450.0]
CHAR_W = 5.0
PITCH = 14.0
Y0 = 100.0
#: Height of a synthetic word box, in points.
WORD_H = 9.0

HEADER = ["Variable", "b", "s", "h", "q"]
ROWS = [
    ["GDP", "0.253", "0.179", "0.211", "0.301"],
    ["CPI", "0.144", "0.135", "0.290", "0.188"],
    ["IP", "0.041", "0.050", "0.154", "0.099"],
    ["UR", "0.082", "0.321", "0.144", "0.211"],
    ["CB", "0.180", "0.171", "0.365", "0.244"],
    ["TR", "0.310", "0.220", "0.410", "0.188"],
]

UNCHECKED = LineDirections.unchecked_for_tests()


@pytest.fixture
def no_prose_in_header(monkeypatch):
    """Switch ``prose_in_header`` (GH-936) off for a grid fixture that has no prose on the page.

    These synthetic pages put a whole row on one text line, so the column pitch is the only gap
    ``_median_word_gap`` can measure and every header row reads as one run. A real page's word
    space comes from its body text. The predicate has its own file (``test_gh936_prose_in_header``).
    """
    monkeypatch.setattr(ship_gate, "prose_in_header_faults", lambda *a, **k: [])


def native_first_config() -> PipelineConfig:
    """A hermetic agentic native-first config for tests that drive ``process()``.

    ``primary_engine``, ``local_engine`` and ``enabled_engines`` are all pinned: an
    ``AUTO`` primary engine makes ``process()`` probe the engines this machine happens to
    have pulled (CLAUDE.md, #841).
    """
    return PipelineConfig(
        agentic=True,
        native_first=True,
        native_only=False,
        primary_engine=EngineType.QWEN,
        local_engine=EngineType.QWEN,
        enabled_engines=[EngineType.QWEN],
        tiered=False,
        dual_pass_tables=False,
        detect_equations=False,
        save_figures=False,
        quiet=True,
        table_judge_ladder=False,
    )


def routed_decision(
    page_num: int,
    ladder: list,
    *,
    text: str = "model table",
    status: PageStatus = PageStatus.SUCCESS,
    accepted: bool = True,
) -> PageDecision:
    """What a fake ``route_page`` returns: one attempt on the first rung of *ladder*."""
    out = PageOutput(
        page_num=page_num,
        text=text,
        status=status,
        engine="qwen",
        audit_passed=accepted,
    )
    prof = ladder[0]
    attempt = ProviderAttempt(
        engine=prof.engine,
        output=out,
        cost_usd=0.0,
        accepted=accepted,
        reason="test",
        provider_id=prof.id,
        model=prof.model,
        backend=prof.backend,
    )
    return PageDecision(page_num=page_num, final_output=out, attempts=[attempt], accepted=accepted)


def flush_and_restore(pipeline, state: DocumentState, pdf: Path, out_dir: Path) -> DocumentState:
    """Flush page 1 of *state* as terminal, then restore it into a fresh state (a resume)."""
    assert pipeline._flush_page_sidecar(state, 1, out_dir, terminal=True) is not None
    resumed = DocumentState(handle=DocumentHandle(path=pdf, page_count=1))
    resumed.pages[1] = state.pages[1]
    restored = PageOutput(
        page_num=1,
        text="model table",
        status=PageStatus.SUCCESS,
        engine="qwen",
        audit_passed=True,
    )
    pipeline._restore_terminal_page_state(resumed, 1, restored, out_dir)
    return resumed


def place(u: float, v: float, rotation: int, width: float, height: float) -> tuple[float, float]:
    """Map an upright-frame point (u right, v down) onto a page drawn at *rotation*.

    GH-902: fitz ``rotate=90`` text reads bottom-to-top, so the upright top edge
    lands on the LEFT of the page and the upright left edge at the BOTTOM;
    ``rotate=270`` is the mirror image. The earlier fixture laid the grid out
    180 degrees off this mapping, which is exactly what the wrong rowizer sign
    undid, so the fixture and the bug agreed and nothing failed.
    """
    if rotation == 0:
        return u, v
    if rotation == 90:
        return v, height - u
    if rotation == 270:
        return width - v, u
    raise ValueError(rotation)


def forecast_pdf(path: Path, rotation: int = 90) -> None:
    """The PP-6 grid with a ruled table signal, drawn at *rotation* (0, 90, 270)."""
    doc = fitz.open()
    width, height = 612, 792
    page = doc.new_page(width=width, height=height)

    def text(u: float, v: float, s: str, size: float) -> None:
        page.insert_text(
            place(u, v, rotation, width, height),
            s,
            fontsize=size,
            fontname="helv",
            rotate=rotation,
        )

    text(72, 50, "Table 1. GDP growth forecasts across baseline and shock scenarios.", 10)
    text(72, 400, "* Forecasts are annualized percent changes.", 9)
    for ci, hdr in enumerate(HEADER):
        text(COL_XS[ci], 80, hdr, 9)
    for ri, row in enumerate(ROWS):
        for ci, cell in enumerate(row):
            text(COL_XS[ci], 100 + ri * 22, cell, 9)
    x0, y0, tw, th = 70, 70, 400, 180
    for r in range(9):
        page.draw_line(
            place(x0, y0 + r * 20, rotation, width, height),
            place(x0 + tw, y0 + r * 20, rotation, width, height),
        )
    for c in range(6):
        page.draw_line(
            place(x0 + c * 70, y0, rotation, width, height),
            place(x0 + c * 70, y0 + th, rotation, width, height),
        )
    # GH-902: pin the fixture to real PDF geometry, not to the code
    # under test. The pre-fix fixtures drew 90/270 swapped, and the wrong sign
    # undid it, so the tests passed against the bug.
    from socr.core.born_digital import upright_rotation_for

    assert upright_rotation_for(page) == rotation, (upright_rotation_for(page), rotation)
    doc.save(str(path))
    doc.close()


def rotated_forecast_pdf(path: Path) -> None:
    forecast_pdf(path, 90)


def dense_pdf(path: Path) -> None:
    """The forecast grid, upright and unruled, under a one-line caption."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 50), "Table 1. GDP growth forecasts.", fontsize=10, fontname="helv")
    for ci, hdr in enumerate(HEADER):
        page.insert_text((COL_XS[ci], 80), hdr, fontsize=9, fontname="helv")
    for ri, row in enumerate(ROWS):
        for ci, cell in enumerate(row):
            page.insert_text((COL_XS[ci], 100 + ri * 22), cell, fontsize=9, fontname="helv")
    doc.save(str(path))
    doc.close()
