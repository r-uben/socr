"""#1389: a chart page never ships SUCCESS with no picture and no mark.

Fed p074 (three line panels) and p002 (six chart panels) shipped SUCCESS with neither. Two
causes, measured on the real pages:

* p074 is arbitrated (chart marks and a table signal), so the #1385 lane abstained because the
  chart-region lane "owns" the page; that lane's own inventory (``chart_region_bboxes``) is
  EMPTY on the page, so it skipped it without a word. Both lanes silent.
* p002 has no region the locator finds and no own-line ticks, but the reading wrote its axes as
  ``- **Left Axis (Percent):** 1.0, 1.5, 2.0``, which the #1225 rule did not read.

Hermetic: no provider, no model. Differences, not absolutes (CLAUDE.md, #257): each test runs the
same page twice and changes one thing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_gh1187_unshipped_figure_pages import _out, _pdf_with_a_figure, _run_dir

from socr.core.audit_log import AuditEvent
from socr.core.document import DocumentHandle
from socr.core.state import DocumentState, PageState
from socr.figures.unshipped_figures import (
    CHART_SKELETON_KIND,
    CHART_TABLE_ARBITRATION_KIND,
    UNCAPTIONED_CROP_KIND,
    chart_region_lane,
    decide,
    figure_captions,
    tick_run,
)
from socr.pipeline import refinalize_unshipped_figures as tool
from socr.pipeline.orchestrator import UnifiedPipeline

#: The shape of the saved p074 reading, ticks included.
P074_BODY = (
    "## Net Percentage of Domestic Respondents Tightening Standards\n"
    "- **Y-axis label:** Percent\n"
    "- **Y-axis ticks:** 80, 60, 40, 20, 0, -20\n"
    "- **X-axis ticks:** 1990, 1991, 1992, 1993, 1994, 1995\n"
)
#: The same page with the ticks the reading did not write: titles and units only.
TICKLESS_BODY = "## Net Percentage of Domestic Respondents Tightening Standards\n\nPercent"
#: The shape of the saved p002 reading.
P002_BODY = (
    "## Unemployment Insurance\n\n"
    "- **Left Axis (Percent):** 1.0, 1.5, 2.0, 2.5, 3.0, 3.5\n"
    "- **X-Axis:** 1996, 1998, 2000, 2002, 2004\n"
)
_REGION = [(100.0, 120.0, 500.0, 420.0)]
_PAGE = [(0.0, 0.0, 612.0, 792.0)]


def _box():
    return _PAGE


@pytest.fixture(autouse=True)
def _no_words_check(monkeypatch):
    from socr.core import verdict

    monkeypatch.setattr(verdict, "page_missing_words", lambda output, p: None)


def _inventory(monkeypatch, boxes: list) -> None:
    """Pin the chart-region lane's own inventory, the one thing that differs between arms."""
    from socr.tables import reconstruct

    monkeypatch.setattr(reconstruct, "chart_region_bboxes", lambda page: list(boxes))


def test_loaded_source_is_this_checkout() -> None:
    import socr

    assert Path(socr.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1] / "src")


# ---------------------------------------------------------------------------
# The inline tick list
# ---------------------------------------------------------------------------


def test_the_fixtures_have_the_shapes_the_issue_describes() -> None:
    for body in (P074_BODY, P002_BODY, TICKLESS_BODY):
        assert not figure_captions(body)
        assert tick_run(body) < 3, "the #1225 own-line rule returns nothing on the list form"
    assert tick_run(P074_BODY, inline=True) >= 3 and tick_run(P002_BODY, inline=True) >= 3
    assert tick_run(TICKLESS_BODY, inline=True) < 3


@pytest.mark.parametrize(
    "line",
    [
        "| Total: 1 | 2 | 3 |",  # a table row
        "Total: 1, 5, 11",  # not evenly spaced
        "Revenue: 1,000, 2,000",  # two thousands-grouped numbers, not four items
        "Years covered: 2001, 2002",  # two numbers: any two are evenly spaced
        "Series: 3, 6, nine, 12",  # a non-number item
        "Codes: 1e3, 2e3, 3e3",  # parses as a float, is not a tick
        "| Item | Ticks: 10, 20, 30",  # a table row that happens to end in a list
        "Pairs: 10,20, 30,40",  # comma-grouped pairs, not four items
        "In 2001, 2002, 2003 we grew.",  # prose, no label colon
    ],
)
def test_a_line_that_is_not_an_axis_list_scores_nothing(line: str) -> None:
    assert tick_run(line, inline=True) < 3


# ---------------------------------------------------------------------------
# p002: no locator region, ticks as an inline list
# ---------------------------------------------------------------------------


def test_decide_marks_an_inline_tick_page_the_locator_cannot_picture() -> None:
    no_region = lambda: []  # noqa: E731
    ticks = decide(P002_BODY, no_region, lambda b: [], page_box=_box)
    bare = decide(TICKLESS_BODY, no_region, lambda b: [], page_box=_box)
    assert bare is None, "no ticks and no region: nothing is owed"
    assert ticks is not None and ticks.kind == CHART_SKELETON_KIND and ticks.marks
    assert ticks.refs == (), "a mark, never a picture the locator could not support"


def test_a_picture_beats_the_inline_tick_mark_when_the_locator_finds_a_region() -> None:
    """The picture branch is asked FIRST, so inline ticks never downgrade a page that gets one
    (Fed p017, p053, p093 did, in the first draft)."""
    got = decide(P074_BODY, lambda: _REGION, lambda b: ["f.png"], page_box=_box)
    assert got is not None and got.kind == UNCAPTIONED_CROP_KIND and got.refs == ("f.png",)


# ---------------------------------------------------------------------------
# p074: a claimed chart the chart-region lane cannot crop
# ---------------------------------------------------------------------------


def test_the_chart_region_lane_owns_a_page_only_when_its_inventory_has_a_region(
    tmp_path: Path, monkeypatch
) -> None:
    pdf = _pdf_with_a_figure(tmp_path / "p.pdf", draw_figure=True)

    def lane(arbitrated: bool, table: bool = True) -> tuple[bool, bool]:
        return chart_region_lane(pdf, 1, table_signal=table, arbitrated=arbitrated)

    _inventory(monkeypatch, [])
    empty = (lane(True), lane(False))
    _inventory(monkeypatch, _REGION)
    full = (lane(True), lane(False))
    # (owns, crops): an arbitrated page whose inventory is empty is not the lane's; a plain
    # table page stays the lane's but the lane crops nothing on it.
    assert empty == ((False, False), (True, False))
    assert full == ((True, True), (True, True))
    assert lane(False, table=False) == (False, False)


def _live_kind(tmp_path: Path, body: str, *, arbitrated: bool, has_tables: bool) -> str:
    pdf = _pdf_with_a_figure(tmp_path / "live.pdf", draw_figure=True)
    state = DocumentState(handle=DocumentHandle(path=pdf, page_count=1))
    page = PageState(page_num=1)
    page.has_tables = has_tables
    state.pages[1] = page
    if arbitrated:
        state.events.append(
            AuditEvent(page_num=1, kind=CHART_TABLE_ARBITRATION_KIND, engine="", detail="")
        )
    pipe = UnifiedPipeline.__new__(UnifiedPipeline)
    pipe.config = type("C", (), {"figures_max_per_page": 3, "save_figures": False})()
    pipe._agentic_unshipped_figure_page(state, 1, page, _out(body), tmp_path / "figs")
    return page.unshipped_figure_kind


@pytest.mark.parametrize("inventory", [[], _REGION], ids=["lane-empty", "lane-has-region"])
def test_live_gate_difference_arbitrated_page_by_lane_inventory(
    tmp_path: Path, monkeypatch, inventory: list
) -> None:
    """Both lanes abstaining is impossible: an arbitrated page is acted on exactly when the
    chart-region lane has nothing to crop."""
    _inventory(monkeypatch, inventory)
    kind = _live_kind(tmp_path, TICKLESS_BODY, arbitrated=True, has_tables=True)
    assert kind == ("" if inventory else UNCAPTIONED_CROP_KIND)


def test_a_table_page_with_no_arbitration_is_unchanged_whatever_the_inventory(
    tmp_path: Path, monkeypatch
) -> None:
    """No chart evidence, so no whole-page picture on a table page the locator calls a figure."""
    kinds = []
    for inventory in ([], _REGION):
        _inventory(monkeypatch, inventory)
        kinds.append(_live_kind(tmp_path, TICKLESS_BODY, arbitrated=False, has_tables=True))
    assert kinds == ["", ""]


@pytest.mark.parametrize("body", [TICKLESS_BODY, P002_BODY], ids=["tickless", "inline-ticks"])
def test_the_tool_and_the_live_lane_decide_identically(
    tmp_path: Path, monkeypatch, body: str
) -> None:
    pdf = _pdf_with_a_figure(tmp_path / "paper.pdf", draw_figure=True)
    _, doc_dir = _run_dir(tmp_path, pdf, body)
    sidecar = json.loads((doc_dir / "pages" / "00001.json").read_text())
    seen = set()
    for inventory in ([], _REGION):
        _inventory(monkeypatch, inventory)
        for arbitrated in (False, True):
            side = dict(sidecar)
            side["native_table_region_count"] = 1
            side["audit_events"] = (
                [{"page_num": 1, "kind": CHART_TABLE_ARBITRATION_KIND}] if arbitrated else []
            )
            tooled = tool.refinalize_page(side, pdf, tmp_path / "figs", 3, render=False)
            live = _live_kind(tmp_path, body, arbitrated=arbitrated, has_tables=True)
            assert (tooled[2].kind if tooled else "") == live, (inventory, arbitrated)
            seen.add(live)
    assert len(seen) > 1, "the matrix must exercise more than one answer"


def test_a_page_the_chart_region_lane_will_crop_is_not_marked_for_its_inline_ticks(
    tmp_path: Path, monkeypatch
) -> None:
    """Fed p024: the lane's pictures ship at assembly, after the page loop, so the body shows
    none yet. Marking it for ticks gave a pictured page a false mark (live replay)."""
    kinds = []
    for inventory in ([], _REGION):
        _inventory(monkeypatch, inventory)
        kinds.append(_live_kind(tmp_path, P002_BODY, arbitrated=False, has_tables=True))
    assert kinds == [CHART_SKELETON_KIND, ""]
