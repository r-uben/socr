"""#1393: a chart page never ships SUCCESS with only SOME of its charts pictured and no mark.

Fed p000 has six chart panels. The chart-region lane's inventory (``chart_region_bboxes``)
holds ONE box, over the bar chart, because its mark gate keeps a thin-stroke chart out; the lane
crops that box and the page shipped SUCCESS with four charts in no picture and no mark. The
lane's ``crops`` claim also excused the #1389 tick mark, so nothing else spoke either.

Hermetic: no provider, no model, no ollama. Differences, not absolutes (CLAUDE.md, #257): every
test runs the same page twice and changes the one thing under test, and the real chart-region
inventory runs on a real PDF (nothing here pins it except where a test says so).
"""

from __future__ import annotations

import json
from pathlib import Path

import fitz
import pytest
from test_gh1187_unshipped_figure_pages import _out, _run_dir

from socr.core.audit_log import AuditEvent
from socr.core.document import DocumentHandle
from socr.core.result import PageStatus
from socr.core.state import DocumentState, PageState
from socr.figures import unshipped_figures as uf
from socr.figures.unshipped_figures import (
    CHART_TABLE_ARBITRATION_KIND,
    UNCAPTIONED_CROP_KIND,
    chart_region_lane,
    decide,
    plots_a_series,
    uncovered_charts,
)
from socr.pipeline import refinalize_unshipped_figures as tool
from socr.pipeline.orchestrator import UnifiedPipeline

#: A reading that names no figure and carries no tick list the #1225 / #1389 rules read.
BODY = "## Corporate earnings\n\nSource. Compustat."
_BAR_TOP, _LINE_TOP = 100.0, 460.0
_PAGE_BOX = [(0.0, 0.0, 612.0, 792.0)]


def _pdf(
    path: Path,
    *,
    series: bool = True,
    ticks: bool = True,
    rotate: int = 0,
    line_chart: bool = True,
) -> Path:
    """One page, two charts. The upper one is coloured bars (the chart inventory finds it, as
    on p000's bar panel). The lower one is a thin-stroke line chart (the inventory's mark gate
    drops it). ``series`` / ``ticks`` / ``line_chart`` each remove one thing from the lower
    chart and nothing else, which is how every arm below is built."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    shape = page.new_shape()
    shape.draw_rect(fitz.Rect(100, _BAR_TOP, 500, _BAR_TOP + 300 - 100))
    shape.finish(color=(0, 0, 0), width=1.5)
    for i in range(8):
        x = 120 + i * 45
        shape.draw_rect(fitz.Rect(x, _BAR_TOP + 190 - (i + 1) * 15, x + 30, _BAR_TOP + 195))
        shape.finish(color=(0.8, 0.1, 0.1), fill=(0.8, 0.1, 0.1), width=1.5)
    shape.commit()
    if line_chart:
        shape = page.new_shape()
        top, bottom, left, right = _LINE_TOP + 60, _LINE_TOP + 200, 120.0, 500.0
        shape.draw_rect(fitz.Rect(left, top, right, bottom))
        shape.finish(color=(0, 0, 0), width=0.5)
        if series:
            points = [(left + 10 + k * 36, bottom - 20 - ((k * 37) % 90)) for k in range(10)]
            shape.draw_polyline([fitz.Point(*p) for p in points])
            shape.finish(color=(0, 0, 0), width=0.5, closePath=False)
        shape.commit()
        if ticks:
            for k, label in enumerate(("40", "30", "20", "10", "0")):
                page.insert_text((80, top + 15 + k * 28), label, fontname="helv", fontsize=8)
            for k, label in enumerate(("2001", "2003", "2005", "2007", "2009")):
                page.insert_text(
                    (left + 10 + k * 80, bottom + 12), label, fontname="helv", fontsize=8
                )
    if rotate:
        page.set_rotation(rotate)
    doc.save(path)
    doc.close()
    return path


def _page(pdf: Path):
    return fitz.open(pdf)[0]


def test_loaded_source_is_this_checkout() -> None:
    import socr

    assert Path(socr.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1] / "src")


# ---------------------------------------------------------------------------
# The evidence: a scale AND a series, outside every crop
# ---------------------------------------------------------------------------


def test_the_inventory_holds_the_bar_chart_and_not_the_line_chart(tmp_path: Path) -> None:
    """The premise, measured on the fixture: the chart inventory is the bar chart alone."""
    from socr.tables.reconstruct import chart_region_bboxes, sized_drawing_clusters

    page = _page(_pdf(tmp_path / "p.pdf"))
    inventory = chart_region_bboxes(page)
    assert len(inventory) == 1 and inventory[0].y1 < _LINE_TOP + 40
    assert len(sized_drawing_clusters(page)) == 2, "the unfiltered clusters hold both charts"


def test_the_chart_the_inventory_misses_is_found_and_each_half_of_the_evidence_is_needed(
    tmp_path: Path,
) -> None:
    arms = {
        "both": _pdf(tmp_path / "both.pdf"),
        "no-series": _pdf(tmp_path / "noseries.pdf", series=False),
        "no-ticks": _pdf(tmp_path / "noticks.pdf", ticks=False),
        "no-line-chart": _pdf(tmp_path / "nochart.pdf", line_chart=False),
    }
    from socr.tables.reconstruct import chart_region_bboxes

    found = {}
    for name, pdf in arms.items():
        page = _page(pdf)
        found[name] = len(uncovered_charts(page, chart_region_bboxes(page)))
    assert found == {"both": 1, "no-series": 0, "no-ticks": 0, "no-line-chart": 0}


def test_a_chart_inside_a_crop_box_is_covered(tmp_path: Path) -> None:
    """Covered means contained. The same page, the same chart: only the crop boxes differ."""
    from socr.tables.reconstruct import chart_region_bboxes

    page = _page(_pdf(tmp_path / "p.pdf"))
    own = chart_region_bboxes(page)
    covering = [*own, fitz.Rect(0, _LINE_TOP, 612, 792)]
    clipping = [*own, fitz.Rect(0, _LINE_TOP + 100, 612, 792)]
    assert len(uncovered_charts(page, own)) == 1
    assert uncovered_charts(page, covering) == []
    assert len(uncovered_charts(page, clipping)) == 1, (
        "a crop that clips the chart does not cover it"
    )


def test_a_rotated_page_is_not_decided(tmp_path: Path) -> None:
    """Drawing clusters and table boxes of a rotated page are in different coordinate spaces."""
    from socr.tables.reconstruct import chart_region_bboxes

    seen = {}
    for rotate in (0, 90):
        page = _page(_pdf(tmp_path / f"r{rotate}.pdf", rotate=rotate))
        seen[rotate] = len(uncovered_charts(page, chart_region_bboxes(page)))
    assert seen == {0: 1, 90: 0}


def test_a_series_is_a_run_of_slanted_segments_not_a_frame(tmp_path: Path) -> None:
    both = _page(_pdf(tmp_path / "a.pdf"))
    frame = _page(_pdf(tmp_path / "b.pdf", series=False))
    core = (100.0, _LINE_TOP + 60, 500.0, _LINE_TOP + 200)
    assert plots_a_series(both, core)
    assert not plots_a_series(frame, core)


def _polyline_page(segments: int, *, slanted: bool = True):
    """One page with one path of *segments* consecutive segments, slanted or axis-aligned."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    shape = page.new_shape()
    points = []
    for k in range(segments + 1):
        if slanted:
            points.append(fitz.Point(100 + k * 40, 200 + (k % 2) * 40))
        else:  # a staircase: every segment is horizontal or vertical
            points.append(fitz.Point(100 + (k // 2 + k % 2) * 40, 200 + (k // 2) * 40))
    shape.draw_polyline(points)
    shape.finish(color=(0, 0, 0), width=0.5, closePath=False)
    shape.commit()
    return page, (90.0, 190.0, 100 + (segments + 1) * 40.0, 200 + (segments + 1) * 40.0)


@pytest.mark.parametrize(("segments", "expected"), [(2, False), (3, True), (6, True)])
def test_a_series_needs_three_slanted_segments_in_a_row(segments: int, expected: bool) -> None:
    """The definition pinned as literals (a corner is two segments), not read back from the
    constant, so changing the constant is a failing test and not a silent redefinition."""
    page, core = _polyline_page(segments)
    assert plots_a_series(page, core) is expected
    assert uf.MIN_SERIES_RUN == 3


def test_a_smoothed_line_drawn_as_curves_is_a_series() -> None:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    shape = page.new_shape()
    for k in range(uf.MIN_SERIES_RUN):
        x = 100 + k * 60
        shape.draw_bezier(
            fitz.Point(x, 300),
            fitz.Point(x + 20, 240),
            fitz.Point(x + 40, 360),
            fitz.Point(x + 60, 300 - 20 * (k + 1)),
        )
    shape.finish(color=(0, 0, 0), width=0.5, closePath=False)
    shape.commit()
    assert plots_a_series(page, (90.0, 200.0, 100 + uf.MIN_SERIES_RUN * 60 + 10.0, 400.0))
    doc.close()


def test_axis_aligned_segments_are_a_rule_or_a_table_not_a_series() -> None:
    """A staircase of horizontal and vertical segments (a ruled grid drawn as one path) is not
    slanted however long it is."""
    page, core = _polyline_page(8, slanted=False)
    assert not plots_a_series(page, core)


# ---------------------------------------------------------------------------
# The lane
# ---------------------------------------------------------------------------


def test_a_lane_that_crops_only_some_charts_does_not_own_the_page(tmp_path: Path) -> None:
    with_chart = _pdf(tmp_path / "with.pdf")
    without = _pdf(tmp_path / "without.pdf", line_chart=False)
    got = {}
    for name, pdf in (("partly", with_chart), ("whole", without)):
        for arbitrated in (True, False):
            owns, crops, uncovered = chart_region_lane(
                pdf, 1, table_signal=True, arbitrated=arbitrated
            )
            got[(name, arbitrated)] = (owns, crops, len(uncovered))
    assert got == {
        ("partly", True): (False, False, 1),
        ("partly", False): (False, False, 1),
        ("whole", True): (True, True, 0),
        ("whole", False): (True, True, 0),
    }


def test_a_page_with_no_table_signal_and_no_arbitration_is_not_this_lane(tmp_path: Path) -> None:
    pdf = _pdf(tmp_path / "p.pdf")
    assert chart_region_lane(pdf, 1, table_signal=False, arbitrated=False) == (False, False, [])


# ---------------------------------------------------------------------------
# decide(): the picture is cut from the page, and the evidence stands without the locator
# ---------------------------------------------------------------------------


def test_decide_ships_the_page_when_the_lane_left_charts_and_the_locator_finds_none() -> None:
    uncovered = [fitz.Rect(0, 500, 512, 700)]
    no_region = lambda: []  # noqa: E731
    render = lambda boxes: ["figures/unshipped_figure_p1_1.png"]  # noqa: E731
    got = decide(
        BODY, no_region, render, chart_region_uncovered=uncovered, page_box=lambda: _PAGE_BOX
    )
    none = decide(BODY, no_region, render, page_box=lambda: _PAGE_BOX)
    assert none is None, "without the lane's evidence nothing is owed on this page"
    assert got is not None and got.kind == UNCAPTIONED_CROP_KIND and got.marks
    assert got.refs == ("figures/unshipped_figure_p1_1.png",)
    assert "outside every crop of the chart-region lane" in got.detail


def test_decide_marks_the_page_when_the_picture_cannot_be_saved() -> None:
    got = decide(
        BODY,
        lambda: [],
        lambda boxes: [],
        chart_region_uncovered=[fitz.Rect(0, 500, 512, 700)],
        page_box=lambda: _PAGE_BOX,
    )
    assert got is not None and got.kind == uf.UNCAPTIONED_CROP_LOST_KIND and got.marks


# ---------------------------------------------------------------------------
# Live path and tool, fed DIFFERENT table signals
# ---------------------------------------------------------------------------


def _live(
    tmp_path: Path, pdf: Path, *, has_tables: bool, arbitrated: bool, save_figures: bool = False
) -> PageState:
    state = DocumentState(handle=DocumentHandle(path=pdf, page_count=1))
    page = PageState(page_num=1)
    page.has_tables = has_tables
    state.pages[1] = page
    if arbitrated:
        state.events.append(
            AuditEvent(page_num=1, kind=CHART_TABLE_ARBITRATION_KIND, engine="", detail="")
        )
    pipe = UnifiedPipeline.__new__(UnifiedPipeline)
    pipe.config = type("C", (), {"figures_max_per_page": 3, "save_figures": save_figures})()
    pipe._agentic_unshipped_figure_page(state, 1, page, _out(BODY), tmp_path / "figs")
    return page


def _tooled(tmp_path: Path, pdf: Path, *, table_count: int, events: list[str]):
    _, doc_dir = _run_dir(tmp_path, pdf, BODY)
    sidecar = json.loads((doc_dir / "pages" / "00001.json").read_text())
    sidecar["native_table_region_count"] = table_count
    sidecar["audit_events"] = [{"page_num": 1, "kind": k} for k in events]
    return tool.refinalize_page(sidecar, pdf, tmp_path / "figs", 3, render=False)


@pytest.mark.parametrize(
    ("live_tables", "tool_count"),
    [(True, 0), (True, 1), (False, 1), (False, 0)],
    ids=["live-yes/tool-no", "both-yes", "live-no/tool-yes", "both-no"],
)
def test_an_arbitrated_partly_pictured_page_gets_the_same_answer_whatever_the_table_signal(
    tmp_path: Path, live_tables: bool, tool_count: int
) -> None:
    """The live gate reads ``_page_has_tables`` and the tool reads the sidecar's table region
    count; they differ on 34 of the 100 Fed pages (#1393 item 5). Arbitration is the evidence
    both have, so an arbitrated page is decided the same under every pairing of the two."""
    pdf = _pdf(tmp_path / "p.pdf")
    live = _live(tmp_path, pdf, has_tables=live_tables, arbitrated=True)
    tooled = _tooled(tmp_path, pdf, table_count=tool_count, events=[CHART_TABLE_ARBITRATION_KIND])
    assert live.unshipped_figure_kind == UNCAPTIONED_CROP_KIND
    assert tooled is not None and tooled[2].kind == live.unshipped_figure_kind
    assert live.unshipped_figure_refs and tooled[2].refs


def test_the_page_the_lane_preserved_a_crop_for_is_decided_like_any_other(
    tmp_path: Path,
) -> None:
    """``chart_region_preserved`` is the table signal the sidecar keeps for a page whose count
    is 0 (p000 itself: arbitrated, count 0). Same page, live and tool, no arbitration."""
    pdf = _pdf(tmp_path / "p.pdf")
    live = _live(tmp_path, pdf, has_tables=True, arbitrated=False)
    tooled = _tooled(tmp_path, pdf, table_count=0, events=["chart_region_preserved"])
    assert live.unshipped_figure_kind == UNCAPTIONED_CROP_KIND
    assert tooled is not None and tooled[2].kind == UNCAPTIONED_CROP_KIND


def test_live_difference_the_line_chart_is_the_only_thing_that_changes_the_answer(
    tmp_path: Path,
) -> None:
    """Pinned as a DIFFERENCE (CLAUDE.md, #257): the same body, the same signals, the same
    inventory (the bar chart). Only the second chart differs."""
    kinds = {}
    for name, line_chart in (("whole", False), ("partly", True)):
        pdf = _pdf(tmp_path / f"{name}.pdf", line_chart=line_chart)
        kinds[name] = _live(
            tmp_path / name, pdf, has_tables=True, arbitrated=True
        ).unshipped_figure_kind
    assert kinds == {"whole": "", "partly": UNCAPTIONED_CROP_KIND}


def test_the_picture_never_turns_a_warned_page_clean(tmp_path: Path) -> None:
    """The mark adds a clause and moves no status up: a warned page stays warned, and the page
    that was SUCCESS is not SUCCESS with a figure clause missing."""
    pdf = _pdf(tmp_path / "p.pdf")
    _, doc_dir = _run_dir(tmp_path, pdf, BODY)
    base = json.loads((doc_dir / "pages" / "00001.json").read_text())
    base["native_table_region_count"] = 1
    base["audit_events"] = [{"page_num": 1, "kind": CHART_TABLE_ARBITRATION_KIND}]
    warned = json.loads(json.dumps(base))
    warned["status"] = "warning"
    warned["winning_output"]["status"] = "warning"
    got = {}
    for name, side in (("success", base), ("warning", warned)):
        result = tool.refinalize_page(side, pdf, tmp_path / "figs", 3, render=False)
        assert result is not None
        got[name] = result[0].status
    assert got["warning"] is PageStatus.WARNING
    assert got["success"] is PageStatus.WARNING, "the mark moves a clean page DOWN, never up"


@pytest.fixture
def locator_finds_nothing(monkeypatch):
    """The lane's own evidence must stand without the locator (the p002 shape: it finds 0)."""
    monkeypatch.setattr(uf, "locate_figure_regions", lambda *a, **k: [])
    monkeypatch.setattr(tool, "locate_figure_regions", lambda *a, **k: [])


def test_live_and_tool_ship_the_page_when_the_locator_finds_nothing(
    tmp_path: Path, locator_finds_nothing
) -> None:
    pdf = _pdf(tmp_path / "p.pdf")
    live = _live(tmp_path, pdf, has_tables=True, arbitrated=True)
    tooled = _tooled(tmp_path, pdf, table_count=0, events=[CHART_TABLE_ARBITRATION_KIND])
    assert live.unshipped_figure_kind == UNCAPTIONED_CROP_KIND
    assert tooled is not None and tooled[2].kind == UNCAPTIONED_CROP_KIND


def test_the_preserved_crop_event_is_the_table_signal_the_tool_keeps(
    tmp_path: Path, locator_finds_nothing
) -> None:
    """Not arbitrated, table count 0, the locator silent: the only thing that tells the tool the
    chart-region lane was here is ``chart_region_preserved``. Same page without the event."""
    pdf = _pdf(tmp_path / "p.pdf")
    with_event = _tooled(tmp_path / "a", pdf, table_count=0, events=["chart_region_preserved"])
    without = _tooled(tmp_path / "b", pdf, table_count=0, events=[])
    assert with_event is not None and with_event[2].kind == UNCAPTIONED_CROP_KIND
    assert without is None, "no signal that the lane is involved: the tool leaves the page alone"


def test_with_save_figures_the_ordinary_figure_phase_owns_the_page(tmp_path: Path) -> None:
    """``--save-figures`` ships the page's pictures through the figure phase, so this lane adds
    none (unchanged by #1393; the flag also switches the #1389 fix off, see #1393 item 2)."""
    pdf = _pdf(tmp_path / "p.pdf")
    kinds = {
        flag: _live(
            tmp_path / str(flag), pdf, has_tables=True, arbitrated=True, save_figures=flag
        ).unshipped_figure_kind
        for flag in (False, True)
    }
    assert kinds == {False: UNCAPTIONED_CROP_KIND, True: ""}
