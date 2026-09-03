"""GH-519: the chart-asset lane's debt, stated as its own disposition.

The lane preserves a figure page's native prose and its page PNG, and the
existing ``chart_asset_page`` event says in prose that data values are not
transcribed. A figure carrying in-image text -- axis labels, a legend, an
embedded table -- ships that text nowhere in the markdown, and nothing counted
how often that happened.

``docs/log/2026-09-02_p4-structure-lane-design.md`` section 7 (Q3) ruled that
figures are done for preservation, not machine-readable extraction, and that
the debt must be VISIBLE rather than buried. So it gets a kind of its own --
a consumer counting the debt should not have to parse a sentence -- surfaced
in the page sidecar, the document note and the CLI summary like every other
lane disposition.

Reading in-image text is explicitly NOT in scope, and nothing here reads any.

The resume pin is the GH-563 lesson applied before it can bite again: a record
that lives only in the running process tells the truth once and then tells the
resumed operator nothing at all. Measured through the real
``_restore_terminal_page_state``, not by re-asserting a flag.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

fitz = pytest.importorskip("fitz")

from socr.core.audit_log import VISUAL_VALUES_NOT_TRANSCRIBED_KIND  # noqa: E402
from socr.core.document import DocumentHandle  # noqa: E402
from socr.core.result import PageOutput, PageStatus  # noqa: E402
from socr.core.state import DocumentState  # noqa: E402
from socr.pipeline.orchestrator import UnifiedPipeline  # noqa: E402


# The canonical chart-lane fixtures and harness. Imported rather than rewritten:
# a hand-rolled "chart" page that does not actually reach the lane makes every
# assertion below vacuous, which is exactly what the first attempt at this file
# did (the bars cleared no cluster test and the page went to the OCR ladder).
from test_chart_lane import (  # noqa: E402
    _make_agentic_pipeline,
    _make_decorated_prose_pdf,
    _make_state_with_page,
    _make_vector_chart_pdf,
)


def _prose_with_chart_pdf(tmp_path: Path) -> Path:
    """Born-digital by the REAL detector, and carrying real chart marks.

    ``_make_vector_chart_pdf`` has one line of text, so the live detector does
    not call it born-digital and ``trigger_rates`` never reaches the free lane
    for it -- fine for the agentic tests above, which supply their own
    assessment, useless for a measurement that runs the detector itself. Prose
    for the first condition, the same coloured bars and thick series line for
    the second.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    pdf = tmp_path / "prose_chart.pdf"
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    for i in range(20):
        page.insert_text(
            (72, 80 + i * 18),
            "This is a regular paragraph of born-digital prose text. " * 3,
            fontsize=10,
        )
    red, blue, green = (0.9, 0.1, 0.1), (0.1, 0.1, 0.9), (0.1, 0.8, 0.1)
    for i, (col, x) in enumerate(zip([red, blue, green, red, blue], [100, 180, 260, 340, 420])):
        page.draw_rect(fitz.Rect(x, 640 - i * 20, x + 60, 760), color=col, fill=col, width=1)
    page.draw_line(fitz.Point(100, 470), fitz.Point(480, 470), color=(0.8, 0.2, 0.0), width=3)
    doc.save(str(pdf))
    doc.close()
    return pdf


def _dir(tmp_path: Path, name: str) -> Path:
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _run_agentic(pdf: Path, tmp_path: Path):
    pipeline = _make_agentic_pipeline()
    state = _make_state_with_page(pdf)
    pipeline._last_assessment = state._last_assessment
    with patch("socr.pipeline.orchestrator.route_page") as route:
        pipeline._phase_agentic(state, tmp_path)
        route.assert_not_called()
    return pipeline, state


def _debt_events(state: DocumentState) -> list:
    return [e for e in state.events if getattr(e, "kind", "") == VISUAL_VALUES_NOT_TRANSCRIBED_KIND]


def test_a_chart_asset_page_records_the_debt(tmp_path: Path) -> None:
    """The disposition itself: a kind, not a sentence to be parsed."""
    pipeline, state = _run_agentic(_make_vector_chart_pdf(_dir(tmp_path, "c")), tmp_path / "o")

    assert state.pages[1].best_output is not None
    assert state.pages[1].best_output.engine == "chart_asset", (
        "the page did not take the chart-asset lane, so this test is not "
        f"measuring the lane's debt: engine={state.pages[1].best_output.engine}"
    )

    events = _debt_events(state)
    assert len(events) == 1, f"expected exactly one debt event, got {events}"
    assert events[0].page_num == 1
    assert "not in the markdown" in events[0].detail

    note = pipeline._visual_values_not_transcribed_note(state)
    assert note is not None, "the debt reached no document note"
    assert "visual values not transcribed" in note
    assert "page(s) 1" in note


def test_a_page_that_takes_no_chart_lane_records_nothing(tmp_path: Path) -> None:
    """Difference control. Without it, an implementation that stamped every page
    -- or a note that was never conditional -- would satisfy the test above, and
    the count in `trigger_rates` would describe nothing."""
    pipeline, state = _run_agentic(_make_decorated_prose_pdf(_dir(tmp_path, "p")), tmp_path / "op")

    assert state.pages[1].best_output is None or (
        state.pages[1].best_output.engine != "chart_asset"
    ), "the prose control routed to the chart lane; it is not a control"
    assert _debt_events(state) == []
    assert pipeline._visual_values_not_transcribed_note(state) is None, (
        "a run with no chart-asset page still produced a debt note"
    )


def test_the_debt_survives_a_resume(tmp_path: Path) -> None:
    """GH-563's lesson, applied before it can bite again.

    The debt is a standing property of the page, not of the run that noticed it.
    An event kind missing from ``resume_restore_kinds`` is written to the
    sidecar and then vanishes from ``audit_log.json`` and from the note the
    moment the page resumes -- silently retiring a debt nobody paid.
    """
    pipeline, state = _run_agentic(_make_vector_chart_pdf(_dir(tmp_path, "rc")), tmp_path / "orc")
    assert _debt_events(state), "nothing to resume"

    out_dir = tmp_path / "resume_out"
    pipeline._scan_root = state.handle.path.parent
    pipeline._flush_page_sidecar(state, 1, out_dir)
    sidecar = next(out_dir.rglob("pages/00001.json"))
    kinds = [ev.get("kind") for ev in json.loads(sidecar.read_text()).get("audit_events", [])]
    assert VISUAL_VALUES_NOT_TRANSCRIBED_KIND in kinds, (
        f"the debt never reached the sidecar, so the restore below is empty: {kinds}"
    )

    resumed = DocumentState(handle=DocumentHandle.from_path(state.handle.path))
    assert not resumed.events
    page_out = PageOutput(
        page_num=1, text="body", status=PageStatus.SUCCESS, engine="chart_asset", audit_passed=True
    )
    pipeline._restore_terminal_page_state(resumed, 1, page_out, out_dir)

    assert _debt_events(resumed), (
        "the debt did not survive the resume; the resumed run's audit log and "
        "document note both silently drop it"
    )
    assert pipeline._visual_values_not_transcribed_note(resumed) is not None


def test_the_corpus_count_uses_the_production_predicate(tmp_path: Path) -> None:
    """GH-519's second half. ``has_figures`` is not the routing gate --
    ``has_chart_marks`` is, and it applies the GH-167/#510 placed-area gate.
    Counting the first would report a debt the pipeline does not incur, and a
    locally re-derived gate would drift the first time the real one moved."""
    from socr.benchmark import trigger_rates

    chart = _prose_with_chart_pdf(tmp_path / "tc")
    prose = _make_decorated_prose_pdf(_dir(tmp_path, "tp"))

    tally = trigger_rates.measure([chart, prose])
    by_doc = {(r.doc, r.page): r for r in tally.rows}
    assert (chart.name, 1) in by_doc, "the chart page never reached the free lane"
    assert (prose.name, 1) in by_doc, "the prose page never reached the free lane"

    assert by_doc[(chart.name, 1)].chart_marks is True
    assert by_doc[(prose.name, 1)].chart_marks is False, (
        "the prose control counted as chart-sized, so the count measures nothing"
    )
    assert tally.chart_marks == 1

    report = trigger_rates.report(tally)
    assert "visual values not transcribed" in report, (
        "the debt count is tallied but not reported, which is how the last one went unnoticed"
    )


def test_a_render_failure_does_not_claim_the_image_preserved_it(tmp_path: Path) -> None:
    """cubic P2 on #565. The debt event fires on the render-failure path too,
    where no page PNG was saved at all.

    "Preserved in the page image only" is then false comfort, and worse than
    the debt it describes: an operator reading it believes the figure survived
    somewhere. The page is already WARNING with a `chart_asset_render_failed`
    event; the note must not quietly upgrade it.
    """
    pdf = _make_vector_chart_pdf(_dir(tmp_path, "rf"))
    pipeline = _make_agentic_pipeline()
    state = _make_state_with_page(pdf)
    pipeline._last_assessment = state._last_assessment

    with (
        patch("socr.pipeline.orchestrator.route_page"),
        patch.object(
            UnifiedPipeline,
            "_render_chart_page_png",
            side_effect=RuntimeError("render died"),
        ),
    ):
        pipeline._phase_agentic(state, tmp_path / "orf")

    events = _debt_events(state)
    assert len(events) == 1, f"expected the debt event on the failure path too: {events}"
    assert events[0].data.get("png_saved") is False, (
        "the render did not actually fail, so this test measures the happy path"
    )
    assert "NOWHERE" in events[0].detail

    note = pipeline._visual_values_not_transcribed_note(state)
    assert note is not None
    assert "preserved nowhere" in note, f"the note lost the render failure: {note}"
    assert "preserved in the page image" not in note, (
        f"the note claims the image preserved a figure that was never saved: {note}"
    )
