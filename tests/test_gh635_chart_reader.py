"""#635 Stage 1: the geometric bar reader, on charts whose counts are known.

Every chart here is DRAWN by the test, so the answer is not an oracle taken
from a document someone read -- it is the number the drawing was made from.
The corpus fixture gets one test of its own at the end, skipped when the file
is absent, and its counts live in ``docs/plans/chart-data/GOLDENS.md``.

The pins are differences, never absolutes measured on one machine: the same
chart at a different page scale, the same reading run twice, the same counts
with and without an acceptance hook.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import fitz
import pytest

from socr.figures.chart_reader import (
    ACCEPT,
    INTEGER,
    NO_OPINION,
    PRESENT,
    PRESENCE_UNRESOLVED,
    REJECT,
    REJECTED,
    UNRESOLVED,
    UNRESOLVED_MARKER,
    UNVERIFIED,
    VERIFIED,
    crop_digest,
    panel_block,
    read_chart_page,
    verify_panel,
)

DOTPLOT_PDF = Path.home() / "Data/socr/fixtures/dotplot/dotplot-p20.pdf"


# ---------------------------------------------------------------------------
# A chart builder: the drawing, from the counts
# ---------------------------------------------------------------------------

SOLID = "September"
DASHED = "June"


def build_chart(
    path: Path,
    solid: list[int] | None,
    dashed: list[int] | None,
    *,
    scale: float = 1.0,
    bins: list[str] | None = None,
    extra_bars: list[tuple[float, float, int]] = (),
    ticks: tuple[int, ...] = (2, 4, 6, 8, 10),
    heading: str = "PANEL A",
) -> tuple[fitz.Document, list]:
    """Draw a two-series histogram and return ``(doc, [full-page bbox])``.

    The y scale is two units per tick, so an odd count sits exactly between two
    printed ticks -- the case the reader must resolve from the fit rather than
    from a tick it can see. ``extra_bars`` adds raw ``(x0, x1, count)`` bars for
    the overlap cases.
    """
    labels = bins or ["B1", "B2", "B3", "B4", "B5"]
    n = len(labels)
    x0, x1 = 100.0 * scale, 400.0 * scale
    base = 400.0 * scale
    unit = 10.0 * scale  # points per participant; ticks are every 2 units
    bin_w = 50.0 * scale
    bin_x0 = 110.0 * scale
    centres = [bin_x0 + bin_w * (i + 0.5) for i in range(n)]

    doc = fitz.open()
    page = doc.new_page(width=612 * scale, height=792 * scale)

    # Frame: the axis the bars rest on, plus a tick ladder attached to it.
    page.draw_line(fitz.Point(x0, base), fitz.Point(x1, base), width=0.4 * scale)
    for value in ticks:
        y = base - value * unit
        page.draw_line(fitz.Point(x0, y), fitz.Point(x0 + 10 * scale, y), width=0.4 * scale)
        page.insert_text(fitz.Point(x1 + 5 * scale, y + 2 * scale), str(value), fontsize=5 * scale)

    # x bin labels, printed below the axis and centred on their bins.
    for centre, label in zip(centres, labels, strict=True):
        width = len(label) * 1.4 * scale
        page.insert_text(fitz.Point(centre - width, base + 10 * scale), label, fontsize=5 * scale)

    # Panel heading, drawn inside the plot and above the highest tick.
    page.insert_text(fitz.Point(x0 + 15 * scale, base - 104 * scale), heading, fontsize=6 * scale)

    # Legend: a filled swatch and a dashed swatch, each named by the text set
    # beside it. Both sit clear of every bin's printed label centre.
    sw_x0, sw_x1 = x0 + 195 * scale, x0 + 203 * scale
    sy = base - 100 * scale
    page.draw_rect(
        fitz.Rect(sw_x0, sy - 2 * scale, sw_x1, sy + 2 * scale),
        color=(0, 0, 0),
        fill=(0.4, 0.6, 0.8),
        width=0.3 * scale,
    )
    page.insert_text(fitz.Point(sw_x1 + 4 * scale, sy + 1.5 * scale), SOLID, fontsize=5 * scale)
    dy = base - 90 * scale
    page.draw_line(
        fitz.Point(sw_x0, dy),
        fitz.Point(sw_x1, dy),
        width=3 * scale,
        dashes="[2 2] 0",
        color=(0, 0.4, 0.7),
    )
    page.insert_text(fitz.Point(sw_x1 + 4 * scale, dy + 1.5 * scale), DASHED, fontsize=5 * scale)

    if solid is not None:
        for i, count in enumerate(solid):
            if count <= 0:
                continue
            left = centres[i] - bin_w * 0.4
            right = centres[i] + bin_w * 0.4
            page.draw_rect(
                fitz.Rect(left, base - count * unit, right, base),
                color=(0, 0, 0),
                fill=(0.4, 0.6, 0.8),
                width=0.3 * scale,
            )
    for bx0, bx1, count in extra_bars:
        page.draw_rect(
            fitz.Rect(bx0 * scale, base - count * unit, bx1 * scale, base),
            color=(0, 0, 0),
            fill=(0.4, 0.6, 0.8),
            width=0.3 * scale,
        )

    if dashed is not None:
        edges = [bin_x0 + bin_w * i for i in range(n + 1)]
        levels = [base - c * unit for c in dashed]
        prev = base
        dash = {"width": 1.5 * scale, "dashes": "[2 2] 0", "color": (0, 0.4, 0.7)}
        for i, level in enumerate(levels):
            if level != prev:
                page.draw_line(
                    fitz.Point(edges[i], min(level, prev)),
                    fitz.Point(edges[i], max(level, prev)),
                    **dash,
                )
            if level != base:
                page.draw_line(fitz.Point(edges[i], level), fitz.Point(edges[i + 1], level), **dash)
            prev = level
        if prev != base:
            page.draw_line(
                fitz.Point(edges[n], min(prev, base)), fitz.Point(edges[n], max(prev, base)), **dash
            )

    doc.save(str(path))
    reopened = fitz.open(str(path))
    return reopened, [reopened[0].rect]


def counts(panel, name: str) -> list[str]:
    for s in panel.series:
        if s.name == name:
            return [c.rendered for c in s.cells]
    raise AssertionError(f"no series {name!r} in {[s.name for s in panel.series]}")


def read_one(tmp_path: Path, solid, dashed, **kw):
    tmp_path.mkdir(parents=True, exist_ok=True)
    doc, bboxes = build_chart(tmp_path / "c.pdf", solid, dashed, **kw)
    reading = read_chart_page(doc[0], bboxes, page_num=1)
    assert 1 in reading.panels, reading.refusals.get(1)
    return reading.panels[1]


# ---------------------------------------------------------------------------
# The counts themselves
# ---------------------------------------------------------------------------


def test_two_series_read_back_the_counts_they_were_drawn_from(tmp_path: Path) -> None:
    panel = read_one(tmp_path, [0, 4, 2, 0, 1], [1, 3, 3, 2, 0])
    assert counts(panel, SOLID) == ["0", "4", "2", "0", "1"]
    assert counts(panel, DASHED) == ["1", "3", "3", "2", "0"]
    assert panel.label == "PANEL A"
    assert [b.label for b in panel.bins] == ["B1", "B2", "B3", "B4", "B5"]


def test_odd_counts_between_printed_ticks_resolve(tmp_path: Path) -> None:
    """Every tick is even; an odd bar's top falls between two of them."""
    panel = read_one(tmp_path, [1, 3, 5, 7, 9], [9, 7, 5, 3, 1])
    assert counts(panel, SOLID) == ["1", "3", "5", "7", "9"]
    assert counts(panel, DASHED) == ["9", "7", "5", "3", "1"]


def test_an_empty_bin_is_zero_and_says_what_observed_it(tmp_path: Path) -> None:
    panel = read_one(tmp_path, [3, 0, 3, 0, 3], [0, 2, 0, 2, 0])
    assert counts(panel, SOLID) == ["3", "0", "3", "0", "3"]
    assert counts(panel, DASHED) == ["0", "2", "0", "2", "0"]
    zeros = [c for s in panel.series for c in s.cells if c.status == INTEGER and c.count == 0]
    assert zeros, "no zero was read"
    assert all(c.empty_bin_observed for c in zeros), "a zero was emitted without an observation"


def test_a_series_with_no_marks_is_unresolved_never_a_row_of_zeros(tmp_path: Path) -> None:
    """The 2021 case. Absent and all-zero are not distinguishable here."""
    panel = read_one(tmp_path, [1, 2, 3, 2, 1], None)
    named = {s.name: s for s in panel.series}
    assert named[SOLID].presence == PRESENT
    assert named[DASHED].presence == PRESENCE_UNRESOLVED
    assert named[DASHED].cells == (), "an undrawn series was given cells"
    body = panel_block(verify_panel(panel, "k", None))
    assert DASHED not in body.split("| Series |")[1], "an undrawn series got a table row"
    assert "unresolved" in body


def test_two_bars_in_one_bin_are_unresolved_not_summed(tmp_path: Path) -> None:
    panel = read_one(
        tmp_path,
        [0, 4, 0, 0, 0],
        None,
        extra_bars=[(165.0, 195.0, 3)],
    )
    assert counts(panel, SOLID)[1] == UNRESOLVED_MARKER
    cell = [c for c in panel.series[0].cells if c.bin_label == "B2"][0]
    assert cell.status == UNRESOLVED and cell.count is None


def test_a_bar_owning_no_bin_makes_the_bins_it_touches_unresolved(tmp_path: Path) -> None:
    """A bar straddling a boundary covers no label centre; nothing is guessed."""
    panel = read_one(tmp_path, [0, 0, 0, 0, 0], None, extra_bars=[(205.0, 215.0, 5)])
    rendered = counts(panel, SOLID)
    assert UNRESOLVED_MARKER in rendered, rendered
    assert "5" not in rendered, "an unplaceable bar was still assigned a bin"


# ---------------------------------------------------------------------------
# The reading does not depend on how it is rendered
# ---------------------------------------------------------------------------


def test_page_scale_does_not_move_a_count(tmp_path: Path) -> None:
    small = read_one(tmp_path / "a", [1, 4, 0, 7, 2], [2, 2, 5, 0, 1])
    big = read_one(tmp_path / "b", [1, 4, 0, 7, 2], [2, 2, 5, 0, 1], scale=1.7)
    assert counts(small, SOLID) == counts(big, SOLID)
    assert counts(small, DASHED) == counts(big, DASHED)
    assert small.calibration.points_per_unit != big.calibration.points_per_unit


def test_crop_dpi_changes_provenance_and_not_one_count(tmp_path: Path) -> None:
    doc, bboxes = build_chart(tmp_path / "c.pdf", [2, 0, 6, 1, 0], [1, 1, 4, 0, 0])
    page = doc[0]
    low = {1: crop_digest(page, bboxes[0], 72)}
    high = {1: crop_digest(page, bboxes[0], 216)}
    a = read_chart_page(page, bboxes, page_num=1, crop_digests=low).panels[1]
    b = read_chart_page(page, bboxes, page_num=1, crop_digests=high).panels[1]
    assert counts(a, SOLID) == counts(b, SOLID) == ["2", "0", "6", "1", "0"]
    assert counts(a, DASHED) == counts(b, DASHED) == ["1", "1", "4", "0", "0"]
    assert a.crop_dpi == 72 and b.crop_dpi == 216
    assert a.crop_sha256 and b.crop_sha256 and a.crop_sha256 != b.crop_sha256


def test_reading_the_same_page_twice_is_byte_identical(tmp_path: Path) -> None:
    doc, bboxes = build_chart(tmp_path / "c.pdf", [3, 0, 1, 5, 0], [0, 2, 2, 4, 1])
    first = panel_block(
        verify_panel(read_chart_page(doc[0], bboxes, page_num=1).panels[1], "k", None)
    )
    second = panel_block(
        verify_panel(read_chart_page(doc[0], bboxes, page_num=1).panels[1], "k", None)
    )
    assert first == second
    assert hashlib.sha256(first.encode()).hexdigest() == hashlib.sha256(second.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Acceptance: the totals live in the caller
# ---------------------------------------------------------------------------


def test_without_a_hook_the_derivation_is_published_unverified(tmp_path: Path) -> None:
    panel = verify_panel(read_one(tmp_path, [1, 2, 3, 0, 0], None), "survey", None)
    assert panel.verification == UNVERIFIED
    body = panel_block(panel)
    assert "UNVERIFIED" in body
    assert "| 1 | 2 | 3 | 0 | 0 |" in body


def test_an_accepting_hook_marks_the_derivation_verified(tmp_path: Path) -> None:
    seen: list[tuple] = []

    def hook(survey: str, horizon: str, payload: dict) -> str:
        seen.append((survey, horizon, payload))
        return ACCEPT if sum(payload[SOLID].values()) == 6 else REJECT

    panel = verify_panel(read_one(tmp_path, [1, 2, 3, 0, 0], None), "survey-2018", hook)
    assert panel.verification == VERIFIED
    assert seen == [
        ("survey-2018", "PANEL A", {SOLID: {"B1": 1, "B2": 2, "B3": 3, "B4": 0, "B5": 0}})
    ]
    assert "| 1 | 2 | 3 | 0 | 0 |" in panel_block(panel)


def test_a_rejecting_hook_withholds_the_counts_and_keeps_the_image(tmp_path: Path) -> None:
    panel = verify_panel(read_one(tmp_path, [1, 2, 3, 0, 0], None), "s", lambda *_a: REJECT)
    assert panel.verification == REJECTED
    body = panel_block(panel)
    assert "REJECTED" in body
    assert "| Series |" not in body, "a rejected derivation still published its counts"
    assert "preserved" in body, "rejecting the derivation rejected the image"


def test_a_hook_with_no_opinion_leaves_the_derivation_unverified(tmp_path: Path) -> None:
    panel = verify_panel(read_one(tmp_path, [1, 0, 0, 0, 0], None), "s", lambda *_a: NO_OPINION)
    assert panel.verification == UNVERIFIED


def test_a_hook_that_raises_never_takes_the_reading_with_it(tmp_path: Path) -> None:
    def boom(*_a):
        raise RuntimeError("caller bug")

    panel = verify_panel(read_one(tmp_path, [1, 0, 0, 0, 0], None), "s", boom)
    assert panel.verification == UNVERIFIED
    assert "RuntimeError" in panel.verification_detail
    assert "| 1 | 0 | 0 | 0 | 0 |" in panel_block(panel)


def test_a_wrong_total_hook_cannot_change_a_single_count(tmp_path: Path) -> None:
    """Pin the DIFFERENCE: the same reading, three verdicts, one set of counts."""
    plain = read_one(tmp_path, [2, 2, 2, 0, 0], None)
    verdicts = [
        verify_panel(plain, "s", None),
        verify_panel(plain, "s", lambda *_a: ACCEPT),
        verify_panel(plain, "s", lambda *_a: REJECT),
    ]
    assert {v.verification for v in verdicts} == {UNVERIFIED, VERIFIED, REJECTED}
    assert {tuple(counts(v, SOLID)) for v in verdicts} == {("2", "2", "2", "0", "0")}


# ---------------------------------------------------------------------------
# Refusals are outcomes, not errors
# ---------------------------------------------------------------------------


def test_a_page_with_no_drawings_is_refused_not_read(tmp_path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(fitz.Point(100, 100), "a scan would look like nothing to this reader")
    reading = read_chart_page(page, [page.rect], page_num=1)
    assert reading.panels == {}
    assert "raster" in reading.refusals[1]


def test_a_chart_with_no_legend_binds_no_series(tmp_path: Path) -> None:
    doc, bboxes = build_chart(tmp_path / "c.pdf", [1, 2, 3, 0, 0], None)
    page = doc[0]
    stripped = fitz.open()
    out = stripped.new_page(width=page.rect.width, height=page.rect.height)
    out.show_pdf_page(out.rect, doc, 0, clip=fitz.Rect(0, 320, page.rect.x1, page.rect.y1))
    reading = read_chart_page(stripped[0], [stripped[0].rect], page_num=1)
    assert reading.panels == {}, "a legend-less chart was read anyway"
    assert "legend" in reading.refusals[1], reading.refusals


# ---------------------------------------------------------------------------
# The corpus page
# ---------------------------------------------------------------------------

DOTPLOT_EXPECTED = {
    "2018": {
        "September projections": [0, 4, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "June projections": [2, 5, 7, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    },
    "2019": {
        "September projections": [0, 1, 1, 1, 4, 4, 4, 1, 0, 0, 0, 0, 0],
        "June projections": [1, 1, 0, 1, 4, 4, 3, 1, 0, 0, 0, 0, 0],
    },
    "2020": {
        "September projections": [0, 1, 0, 1, 1, 4, 2, 6, 1, 0, 0, 0, 0],
        "June projections": [1, 0, 0, 2, 0, 2, 5, 3, 0, 2, 0, 0, 0],
    },
    "2021": {"September projections": [0, 1, 0, 1, 4, 1, 5, 2, 1, 1, 0, 0, 0]},
    "Longer run": {
        "September projections": [0, 0, 3, 4, 6, 1, 1, 0, 0, 0, 0, 0, 0],
        "June projections": [0, 1, 1, 5, 5, 1, 1, 0, 0, 0, 0, 0, 0],
    },
}


@pytest.mark.skipif(not DOTPLOT_PDF.exists(), reason="dotplot corpus fixture is not present")
def test_dotplot_fixture_five_panels() -> None:
    """The corpus page, against docs/plans/chart-data/GOLDENS.md.

    Machine-read and awaiting human annotation; the goldens file writes each
    bar's top out against the printed ticks so the annotation can be made
    without this code.
    """
    from socr.tables.reconstruct import chart_region_bboxes

    doc = fitz.open(str(DOTPLOT_PDF))
    page = doc[0]
    bboxes = chart_region_bboxes(page)
    reading = read_chart_page(page, bboxes, page_num=20)
    assert reading.refusals == {}, reading.refusals
    assert len(reading.panels) == 5

    got = {}
    for idx in sorted(reading.panels):
        panel = reading.panels[idx]
        assert [b.label for b in panel.bins][0] == "1.88-2.12"
        assert len(panel.bins) == 13
        got[panel.label] = {
            s.name: [c.count for c in s.cells] for s in panel.series if s.presence == PRESENT
        }
    assert got == DOTPLOT_EXPECTED

    absent = [
        s for p in reading.panels.values() for s in p.series if s.presence == PRESENCE_UNRESOLVED
    ]
    assert len(absent) == 1 and absent[0].name == "June projections"
    assert reading.panels[4].label == "2021"


@pytest.mark.skipif(not DOTPLOT_PDF.exists(), reason="dotplot corpus fixture is not present")
def test_dotplot_calibration_supports_every_integer_uniquely() -> None:
    from socr.tables.reconstruct import chart_region_bboxes

    doc = fitz.open(str(DOTPLOT_PDF))
    page = doc[0]
    reading = read_chart_page(page, chart_region_bboxes(page), page_num=20)
    for panel in reading.panels.values():
        cal = panel.calibration
        assert cal.checked_ticks == 9
        assert cal.residual < cal.half_count_points
        for s in panel.series:
            for c in s.cells:
                if c.status != INTEGER or c.top_y is None:
                    continue
                lo, hi = c.interval
                assert lo <= c.count <= hi
                assert hi - lo < 1.0, "an interval wide enough for two integers was resolved"


# ---------------------------------------------------------------------------
# The pipeline: what a reader of the document actually gets
# ---------------------------------------------------------------------------


def _pipeline_helpers():
    import test_gh635_chart_table_skeletons as stage0

    return stage0


def _readable_chart_pdf(tmp_path: Path) -> Path:
    pdf = tmp_path / "chart.pdf"
    doc, _b = build_chart(pdf, [0, 4, 2, 0, 1], [1, 3, 3, 2, 0])
    doc.close()
    return pdf


CANDIDATE = (
    "Preamble sentence unique alpha\n\n"
    "### PANEL A\n\n"
    "| Percent range | B1 | B2 | B3 | B4 | B5 |\n"
    "| :--- | :---: | :---: | :---: | :---: | :---: |\n"
    "| **Participants** |  |  |  |  |  |\n"
)


def test_the_pipeline_ships_the_counts_where_the_empty_grid_stood(tmp_path: Path) -> None:
    stage0 = _pipeline_helpers()
    pdf = _readable_chart_pdf(tmp_path)
    _pipe, state, result = stage0._run(pdf, tmp_path / "out", CANDIDATE)
    body = result.pages[0].text or ""

    assert "| September | 0 | 4 | 2 | 0 | 1 |" in body, body
    assert "| June | 1 | 3 | 3 | 2 | 0 |" in body, body
    assert "**Participants**" not in body, "the empty grid still ships"
    assert "counts not extracted" not in body, "Stage 0's note shipped over a real reading"
    assert "chart_region_p1_1.png" in body, "the crop was dropped"

    from socr.figures.chart_reader import CHART_DERIVATION

    events = [e for e in state.events if e.kind == CHART_DERIVATION]
    assert len(events) == 1
    data = events[0].data
    for key in (
        "reader_version",
        "crop_sha256",
        "crop_dpi",
        "crop_clip",
        "calibration",
        "bins",
        "series",
        "verification",
        "source_checksum",
    ):
        assert key in data, key
    assert data["calibration"]["tick_pairs"], "no calibration tick pair was persisted"
    cell = data["series"][0]["cells"][1]
    assert cell["interval"] and cell["bar_bbox"] and cell["top_y"] is not None
    assert state.pages[1].chart_derivations == 1


def test_the_derivation_is_surfaced_on_the_page_and_the_cli(tmp_path: Path) -> None:
    import json as _json
    from unittest.mock import MagicMock, patch

    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.pipeline import orchestrator as orch
    from socr.pipeline.orchestrator import UnifiedPipeline

    stage0 = _pipeline_helpers()
    pdf = _readable_chart_pdf(tmp_path)
    out_dir = tmp_path / "out"
    pipeline = stage0._make_pipeline()
    pipeline.config.quiet = False
    state = stage0._make_state(pdf, "Preamble sentence unique alpha")
    pipeline._last_assessment = state._last_assessment
    printed = MagicMock()
    with (
        patch.object(orch, "console", printed),
        patch(
            "socr.pipeline.orchestrator.route_page",
            return_value=stage0._accepted_decision(CANDIDATE),
        ),
        patch.object(
            UnifiedPipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]
        ),
        patch.object(UnifiedPipeline, "_resolve_judge_model", return_value=""),
    ):
        pipeline._phase_agentic(state, out_dir)
        pipeline._phase_assemble(state, out_dir)

    sidecar = _json.loads(next(out_dir.rglob("pages/00001.json")).read_text())
    notes = " ".join((sidecar.get("winning_output") or {}).get("audit_notes") or [])
    assert "READ from the page's own vector geometry" in notes

    said = [line for line in stage0._cli_lines(printed) if "chart derivation(s)" in line]
    assert said, stage0._cli_lines(printed)
    assert "1 chart derivation(s)" in said[0]
    assert "0 verified / 1 unverified / 0 rejected" in said[0]
    assert "UNRESOLVED" in said[0]


def test_a_second_assembly_reproduces_the_same_bytes(tmp_path: Path) -> None:
    stage0 = _pipeline_helpers()
    pdf = _readable_chart_pdf(tmp_path)
    _p1, _s1, first = stage0._run(pdf, tmp_path / "a", CANDIDATE)
    _p2, _s2, second = stage0._run(pdf, tmp_path / "b", CANDIDATE)
    assert (first.pages[0].text or "") == (second.pages[0].text or "")


def test_the_derivation_provenance_survives_a_resume(tmp_path: Path) -> None:
    import json

    from socr.core.result import PageOutput
    from socr.figures.chart_reader import CHART_DERIVATION

    stage0 = _pipeline_helpers()
    pdf = _readable_chart_pdf(tmp_path)
    out_dir = tmp_path / "out"
    pipeline, state, _result = stage0._run(pdf, out_dir, CANDIDATE)
    before = [e for e in state.events if e.kind == CHART_DERIVATION]
    assert len(before) == 1

    meta = json.loads(next(out_dir.rglob("pages/00001.json")).read_text())
    restored = stage0._make_state(pdf, "Preamble sentence unique alpha")
    page_out = PageOutput.from_dict(meta["winning_output"])
    pipeline._restore_terminal_page_state(restored, 1, page_out, out_dir)

    after = [e for e in restored.events if e.kind == CHART_DERIVATION]
    assert len(after) == 1
    assert after[0].data["series"] == before[0].data["series"]
    assert after[0].data["calibration"] == before[0].data["calibration"]
    assert restored.pages[1].chart_derivations == 1


def test_a_rejecting_hook_publishes_no_counts_through_the_pipeline(tmp_path: Path) -> None:
    stage0 = _pipeline_helpers()
    pdf = _readable_chart_pdf(tmp_path)
    plain = stage0._run(pdf, tmp_path / "a", CANDIDATE)[2].pages[0].text or ""

    pipeline = stage0._make_pipeline()
    pipeline.config.chart_constraint_hook = lambda *_a: REJECT
    state = stage0._make_state(pdf, "Preamble sentence unique alpha")
    pipeline._last_assessment = state._last_assessment
    from unittest.mock import patch

    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.pipeline.orchestrator import UnifiedPipeline

    with (
        patch(
            "socr.pipeline.orchestrator.route_page",
            return_value=stage0._accepted_decision(CANDIDATE),
        ),
        patch.object(
            UnifiedPipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]
        ),
        patch.object(UnifiedPipeline, "_resolve_judge_model", return_value=""),
    ):
        pipeline._phase_agentic(state, tmp_path / "b")
        rejected = (pipeline._phase_assemble(state, tmp_path / "b").pages[0].text) or ""

    assert "| September | 0 | 4 | 2 | 0 | 1 |" in plain
    assert "| September |" not in rejected, "a rejected derivation published its counts"
    assert "REJECTED" in rejected
    assert "chart_region_p1_1.png" in rejected, "rejecting the derivation dropped the image"


def test_the_recorded_crop_digest_names_the_file_the_pipeline_writes(tmp_path: Path) -> None:
    """Provenance must name the crop that ships, not one it rendered privately."""
    import hashlib as _hashlib

    from socr.figures.extractor import RENDER_DPI
    from socr.tables.reconstruct import chart_region_bboxes

    stage0 = _pipeline_helpers()
    pdf = _readable_chart_pdf(tmp_path)
    doc = fitz.open(str(pdf))
    bboxes = chart_region_bboxes(doc[0])
    assert bboxes, "the synthetic chart was not detected as a region"
    recorded, dpi, _clip = crop_digest(doc[0], bboxes[0], RENDER_DPI)
    assert recorded and dpi == RENDER_DPI

    pipeline = stage0._make_pipeline()
    assets = pipeline._render_chart_region_crops(pdf, 1, bboxes, tmp_path / "figures")
    assert assets and assets[0].rendered, assets
    on_disk = (tmp_path / "figures" / assets[0].filename).read_bytes()
    assert _hashlib.sha256(on_disk).hexdigest() == recorded
