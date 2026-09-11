"""#635 Stage 0: an EMPTY grid derived from a chart is withheld; the crop is kept.

The defect: a model reading a chart page emits, per panel, a markdown grid whose
header row is the chart's own axis bins and whose body row is empty in every
cell.  Nothing was read, and the reader is handed a table shaped exactly like an
extraction that succeeded.

Every pin here is a DIFFERENCE, run twice in one process with only the new pass
switched on or off, because the absolute page status of a chart page depends on
machinery (the D3 floor, ``native_fallback``) that does not fire in CI.  The
provider ladder and the judge are patched out for the same reason.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

fitz = pytest.importorskip("fitz")

from socr.figures.chart_data import (  # noqa: E402
    SKELETON_SUPPRESSED,
    SKELETON_UNBOUND,
    find_empty_skeletons,
    suppress_chart_table_skeletons,
)

_RED, _BLUE, _GREEN = (0.9, 0.1, 0.1), (0.1, 0.1, 0.9), (0.1, 0.8, 0.1)

#: The real corpus page this ticket is about.
DOTPLOT_PDF = Path.home() / "Data/socr/fixtures/dotplot/dotplot-p20.pdf"
DOTPLOT_PAGE = Path.home() / "Data/socr/fixtures/dotplot/out-2026-09-09/pages/00001.md"

EMPTY_GRID = "| Bin | B1 | B2 | B3 |\n| :--- | :---: | :---: | :---: |\n| Count | | | |"
ZERO_GRID = "| Bin | B1 | B2 | B3 |\n| :--- | :---: | :---: | :---: |\n| Count | 0 | 0 | 0 |"
TEXT_GRID = "| Bin | B1 | B2 | B3 |\n| :--- | :---: | :---: | :---: |\n| Count | n/a | -- | ? |"
FOREIGN_GRID = "| Bin | Z7 | Z8 | Z9 |\n| :--- | :---: | :---: | :---: |\n| Count | | | |"


def _winner(
    first: str = EMPTY_GRID, second: str = EMPTY_GRID, *, reversed_order: bool = False
) -> str:
    top, bottom = ("BRAVO", "ALPHA") if reversed_order else ("ALPHA", "BRAVO")
    return (
        "Preamble sentence unique alpha\n\n"
        "Second preamble line unique echo\n\n"
        f"### {top}\n\n{first}\n\n"
        "Middle prose unique bravo\n\n"
        "Second middle line unique foxtrot\n\n"
        f"### {bottom}\n\n{second}\n\n"
        "Final line unique delta\n\n"
        "Trailing line unique golf"
    )


# ---------------------------------------------------------------------------
# Synthetic source: two chart panels, each with one label only it draws
# ---------------------------------------------------------------------------


def _panel(page, y_top: float, label: str) -> None:
    """A qualifying vector cluster with an in-region panel label and axis keys."""
    page.insert_text((100, y_top + 13), label, fontsize=8)
    for i, (col, x) in enumerate(
        zip([_RED, _BLUE, _GREEN, _RED, _BLUE], [100, 180, 260, 340, 420])
    ):
        page.draw_rect(
            fitz.Rect(x, y_top + 18 + i * 6, x + 60, y_top + 95), color=col, fill=col, width=1
        )
    for value in (8, 6, 4, 2):
        page.insert_text((88, y_top + 30 + (8 - value) * 8), str(value), fontsize=7)
    for j, key in enumerate(("B1", "B2", "B3")):
        page.insert_text((110 + j * 120, y_top + 104), key, fontsize=7)


def _write_prose(page) -> None:
    page.insert_text((72, 40), "Preamble sentence unique alpha", fontsize=9)
    page.insert_text((72, 52), "Second preamble line unique echo", fontsize=9)
    page.insert_text((72, 210), "Middle prose unique bravo", fontsize=9)
    page.insert_text((72, 222), "Second middle line unique foxtrot", fontsize=9)
    page.insert_text((72, 390), "Final line unique delta", fontsize=9)
    page.insert_text((72, 402), "Trailing line unique golf", fontsize=9)


def _make_two_panel_pdf(tmp_path: Path) -> Path:
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    _write_prose(page)
    _panel(page, 80, "ALPHA")
    _panel(page, 250, "BRAVO")
    out = tmp_path / "two_panel_chart.pdf"
    doc.save(str(out))
    doc.close()
    return out


def _make_chart_free_pdf(tmp_path: Path) -> Path:
    """The SAME prose with every vector mark and panel label removed."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    _write_prose(page)
    page.insert_text((100, 93), "ALPHA", fontsize=8)
    page.insert_text((100, 263), "BRAVO", fontsize=8)
    out = tmp_path / "chart_free.pdf"
    doc.save(str(out))
    doc.close()
    return out


# ---------------------------------------------------------------------------
# Pipeline harness (mirrors tests/test_gh189_mixed_chart_preservation.py)
# ---------------------------------------------------------------------------


def _make_pipeline():
    from socr.core.config import EngineType, PipelineConfig
    from socr.pipeline.orchestrator import UnifiedPipeline

    return UnifiedPipeline(
        PipelineConfig(
            primary_engine=EngineType.DEEPSEEK,
            enabled_engines=list(EngineType),
            agentic=True,
            quiet=True,
            native_first=True,
            save_figures=False,
            describe_figures=False,
            table_judge_ladder=False,
        )
    )


def _make_state(pdf_path: Path, native_text: str):
    from socr.core.born_digital import DocumentAssessment, PageAssessment
    from socr.core.document import DocumentHandle
    from socr.core.state import DocumentState, PageState

    with patch.object(DocumentHandle, "__post_init__", lambda self: None):
        handle = DocumentHandle(path=pdf_path, page_count=1)
    state = DocumentState(handle=handle)
    state.pages[1] = PageState(
        page_num=1,
        is_born_digital=True,
        native_text=native_text,
        needs_ocr_enhancement=False,
        has_tables=True,
    )
    state._last_assessment = DocumentAssessment(
        path=pdf_path,
        pages=[
            PageAssessment(
                page_num=1,
                is_born_digital=True,
                native_text=native_text,
                confidence=0.9,
                needs_ocr_enhancement=False,
                has_tables=True,
            )
        ],
    )
    return state


def _accepted_decision(text: str):
    from socr.core.result import PageOutput, PageStatus

    decision = MagicMock()
    decision.accepted = True
    decision.attempts = []
    decision.final_output = PageOutput(
        page_num=1, text=text, status=PageStatus.SUCCESS, engine="deepseek", audit_passed=True
    )
    decision.winning_engine = "deepseek"
    decision.total_cost_usd = 0.002
    return decision


def _run(pdf: Path, out_dir: Path, winner: str, *, suppress: bool = True, state=None):
    """Route (accepted) then assemble, with no provider and no judge reachable.

    ``suppress=False`` is the control arm: the SAME run with only #635's pass
    turned off, which is the only comparison that is stable across a machine
    with a provider and a CI box without one.
    """
    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.pipeline.orchestrator import UnifiedPipeline

    pipeline = _make_pipeline()
    state = state or _make_state(pdf, "Preamble sentence unique alpha")
    pipeline._last_assessment = state._last_assessment
    stack = [
        patch("socr.pipeline.orchestrator.route_page", return_value=_accepted_decision(winner)),
        patch.object(
            UnifiedPipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]
        ),
        patch.object(UnifiedPipeline, "_resolve_judge_model", return_value=""),
    ]
    if not suppress:
        stack.append(
            patch.object(UnifiedPipeline, "_suppress_chart_table_skeletons", return_value=0)
        )
    for ctx in stack:
        ctx.start()
    try:
        pipeline._phase_agentic(state, out_dir)
        result = pipeline._phase_assemble(state, out_dir)
    finally:
        for ctx in reversed(stack):
            ctx.stop()
    return pipeline, state, result


def _cli_lines(printed) -> list[str]:
    return [str(c.args[0]) for c in printed.print.call_args_list if c.args]


def _body(result) -> str:
    return result.pages[0].text or ""


def _table_lines(text: str) -> list[str]:
    from socr.tables.reconcile import table_syntax_line_indices

    lines = text.split("\n")
    return [lines[i] for i in sorted(table_syntax_line_indices(lines))]


def _events(state, kind: str) -> list:
    return [e for e in state.events if getattr(e, "kind", "") == kind]


def _crops(out_dir: Path) -> list[Path]:
    return sorted(out_dir.rglob("chart_region_p1_*.png"))


# ---------------------------------------------------------------------------
# The structural check, on its own
# ---------------------------------------------------------------------------


def test_empty_data_cells_are_a_skeleton_and_zero_or_text_are_not() -> None:
    """The one structural rule: a data cell that says nothing at all is empty."""
    assert [s.table_index for s in find_empty_skeletons(EMPTY_GRID)] == [1]
    assert find_empty_skeletons(ZERO_GRID) == [], "a literal 0 is an observation"
    assert find_empty_skeletons(TEXT_GRID) == [], "text and unresolved tokens are observations"


def test_one_empty_cell_among_values_is_not_a_skeleton() -> None:
    partial = "| Bin | B1 | B2 |\n| --- | --- |\n| Count | 4 | |"
    assert find_empty_skeletons(partial) == []


def test_numeric_bin_headers_are_never_counted_as_observations() -> None:
    """The header carries the numbers; a grid is still empty."""
    grid = "| Percent range | 1.88-2.12 | 2.13-2.37 |\n| :--- | :---: | :---: |\n| P | | |"
    found = find_empty_skeletons(grid)
    assert len(found) == 1
    assert found[0].data_headers == ["1.88-2.12", "2.13-2.37"]


def test_a_header_only_grid_has_no_body_and_is_not_a_skeleton() -> None:
    assert find_empty_skeletons("| A | B |\n| --- | --- |") == []


def test_no_chart_region_leaves_an_empty_grid_byte_identical() -> None:
    text = _winner()
    out, sup, ref = suppress_chart_table_skeletons(text, page_num=1, interiors={}, crop_names={})
    assert out == text
    assert sup == []
    assert [r.table_index for r in ref] == [1, 2]


# ---------------------------------------------------------------------------
# Binding, end to end on a synthetic source
# ---------------------------------------------------------------------------


def test_bound_empty_grids_are_withheld_and_both_crops_survive(tmp_path: Path) -> None:
    """The difference pin: same page, same winner, only #635's pass switched."""
    pdf = _make_two_panel_pdf(tmp_path)
    winner = _winner()

    _p, off_state, off = _run(pdf, tmp_path / "off", winner, suppress=False)
    _p, on_state, on = _run(pdf, tmp_path / "on", winner, suppress=True)

    # Control arm: the empty grids really do ship on this input.
    assert _table_lines(_body(off)), "control arm shipped no table at all; the pin proves nothing"
    assert "| Count | | | |" in _body(off)

    # Treatment arm: they do not, and the notes say why.
    assert _table_lines(_body(on)) == [], f"an empty grid still ships: {_body(on)!r}"
    assert _body(on).count("counts not extracted") == 2
    assert "(ALPHA)" in _body(on) and "(BRAVO)" in _body(on)

    # The evidence is untouched by the difference.
    assert len(_crops(tmp_path / "off")) == len(_crops(tmp_path / "on")) == 2
    for ref in ("chart_region_p1_1.png", "chart_region_p1_2.png"):
        assert ref in _body(on), f"{ref} left the document"

    # And the page's own prose is intact -- only the grids went.
    for line in ("unique alpha", "unique echo", "unique bravo", "unique foxtrot", "unique golf"):
        assert line in _body(on)

    assert len(_events(on_state, SKELETON_SUPPRESSED)) == 2
    assert _events(off_state, SKELETON_SUPPRESSED) == []


def test_suppression_keeps_the_original_bytes_and_hash_in_provenance(tmp_path: Path) -> None:
    import hashlib

    pdf = _make_two_panel_pdf(tmp_path)
    _p, state, _r = _run(pdf, tmp_path / "out", _winner())
    events = _events(state, SKELETON_SUPPRESSED)
    assert len(events) == 2
    for event in events:
        data = event.data
        assert data["original_text"] == EMPTY_GRID
        assert data["sha256"] == hashlib.sha256(EMPTY_GRID.encode()).hexdigest()
        assert data["crop"] == f"chart_region_p1_{data['region_index']}.png"
        assert data["table_index"] in (1, 2)


def test_the_page_sidecar_carries_the_suppression_note(tmp_path: Path) -> None:
    pdf = _make_two_panel_pdf(tmp_path)
    out_dir = tmp_path / "out"
    _run(pdf, out_dir, _winner())
    sidecars = list(out_dir.rglob("pages/00001.json"))
    assert sidecars, "no page sidecar was written"
    sidecar = json.loads(sidecars[0].read_text())
    notes = (sidecar.get("winning_output") or {}).get("audit_notes") or []
    assert any("#635" in n and "NOT extracted" in n for n in notes), notes
    kinds = [e.get("kind") for e in sidecar.get("audit_events") or []]
    assert kinds.count(SKELETON_SUPPRESSED) == 2, kinds


def test_the_cli_reports_the_count_and_says_the_crops_were_kept(tmp_path: Path) -> None:
    """The third surface. A page note and an audit event a human never opens is
    not a report; the run itself has to say it withheld something."""
    from socr.pipeline import orchestrator as orch

    pdf = _make_two_panel_pdf(tmp_path)
    pipeline = _make_pipeline()
    pipeline.config.quiet = False
    state = _make_state(pdf, "Preamble sentence unique alpha")
    pipeline._last_assessment = state._last_assessment
    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.pipeline.orchestrator import UnifiedPipeline

    printed = MagicMock()
    with (
        patch.object(orch, "console", printed),
        patch("socr.pipeline.orchestrator.route_page", return_value=_accepted_decision(_winner())),
        patch.object(
            UnifiedPipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]
        ),
        patch.object(UnifiedPipeline, "_resolve_judge_model", return_value=""),
    ):
        pipeline._phase_agentic(state, tmp_path / "out")
        pipeline._phase_assemble(state, tmp_path / "out")
    said = [line for line in _cli_lines(printed) if "skeleton" in line]
    assert said, _cli_lines(printed)
    assert "2 chart-table skeleton(s) suppressed" in said[0]
    assert "crops kept" in said[0]


def test_a_grid_carrying_a_literal_zero_is_left_alone(tmp_path: Path) -> None:
    pdf = _make_two_panel_pdf(tmp_path)
    winner = _winner(first=ZERO_GRID, second=ZERO_GRID)
    _p, off_state, off = _run(pdf, tmp_path / "off", winner, suppress=False)
    _p, on_state, on = _run(pdf, tmp_path / "on", winner, suppress=True)
    assert _body(on) == _body(off), "a grid of zeros is data and must not move"
    assert _events(on_state, SKELETON_SUPPRESSED) == []


def test_a_grid_of_text_and_unresolved_tokens_is_left_alone(tmp_path: Path) -> None:
    pdf = _make_two_panel_pdf(tmp_path)
    winner = _winner(first=TEXT_GRID, second=TEXT_GRID)
    _p, _s, off = _run(pdf, tmp_path / "off", winner, suppress=False)
    _p, on_state, on = _run(pdf, tmp_path / "on", winner, suppress=True)
    assert _body(on) == _body(off)
    assert _events(on_state, SKELETON_SUPPRESSED) == []


def test_an_empty_grid_on_a_chart_free_page_is_never_deleted(tmp_path: Path) -> None:
    """No chart region binds it, so the empty form is the page's own content."""
    pdf = _make_chart_free_pdf(tmp_path)
    winner = _winner()
    _p, _s, off = _run(pdf, tmp_path / "off", winner, suppress=False)
    _p, on_state, on = _run(pdf, tmp_path / "on", winner, suppress=True)
    assert _body(on) == _body(off)
    assert "| Count | | | |" in _body(on)
    assert _events(on_state, SKELETON_SUPPRESSED) == []


def test_a_grid_whose_keys_the_chart_never_drew_is_quarantined(tmp_path: Path) -> None:
    """Axis attestation: Z7/Z8/Z9 are on no panel, so nothing is proven."""
    pdf = _make_two_panel_pdf(tmp_path)
    winner = _winner(first=FOREIGN_GRID, second=FOREIGN_GRID)
    _p, _s, off = _run(pdf, tmp_path / "off", winner, suppress=False)
    _p, on_state, on = _run(pdf, tmp_path / "on", winner, suppress=True)
    assert _body(on) == _body(off)
    assert _events(on_state, SKELETON_SUPPRESSED) == []
    reasons = [e.detail for e in _events(on_state, SKELETON_UNBOUND)]
    assert any("does not draw the grid's column keys in full" in r for r in reasons), reasons


def test_an_ambiguous_panel_label_quarantines_rather_than_guesses(tmp_path: Path) -> None:
    """The label names two lines, so it names no position: refuse, keep the grid."""
    pdf = _make_two_panel_pdf(tmp_path)
    ambiguous = _winner().replace(
        "Final line unique delta", "ALPHA\n\nBRAVO\n\nFinal line unique delta"
    )
    _p, _s, off = _run(pdf, tmp_path / "off", ambiguous, suppress=False)
    _p, on_state, on = _run(pdf, tmp_path / "on", ambiguous, suppress=True)
    assert _body(on) == _body(off)
    assert _events(on_state, SKELETON_SUPPRESSED) == []
    assert _events(on_state, SKELETON_UNBOUND)


def test_a_candidate_whose_panels_run_backwards_refuses_the_whole_page(tmp_path: Path) -> None:
    pdf = _make_two_panel_pdf(tmp_path)
    winner = _winner(reversed_order=True)
    _p, _s, off = _run(pdf, tmp_path / "off", winner, suppress=False)
    _p, on_state, on = _run(pdf, tmp_path / "on", winner, suppress=True)
    assert _body(on) == _body(off), "a contradicted layout must withhold no grid at all"
    assert _events(on_state, SKELETON_SUPPRESSED) == []
    reasons = [e.detail for e in _events(on_state, SKELETON_UNBOUND)]
    assert any("backwards" in r for r in reasons), reasons


# ---------------------------------------------------------------------------
# Stability: the mutation happens once and survives re-assembly and resume
# ---------------------------------------------------------------------------


def test_repeated_assembly_is_byte_identical(tmp_path: Path) -> None:
    pdf = _make_two_panel_pdf(tmp_path)
    pipeline, state, first = _run(pdf, tmp_path / "out", _winner())
    second = pipeline._phase_assemble(state, tmp_path / "out")
    assert _body(second) == _body(first)
    third = pipeline._phase_assemble(state, tmp_path / "out")
    assert _body(third) == _body(first)


def test_a_second_run_over_the_same_output_is_byte_identical(tmp_path: Path) -> None:
    """Resume: the withheld grid is already gone, so the pass is a no-op."""
    pdf = _make_two_panel_pdf(tmp_path)
    out_dir = tmp_path / "out"
    _p, _s, first = _run(pdf, out_dir, _winner())
    _p, _s2, second = _run(pdf, out_dir, _winner())
    assert _body(second) == _body(first)


def test_the_pass_is_idempotent_on_its_own_output(tmp_path: Path) -> None:
    from socr.core.pdf import open_pdf
    from socr.figures.chart_data import region_interior_rows
    from socr.figures.chart_regions import chart_region_filename
    from socr.tables.reconstruct import chart_region_bboxes

    pdf = _make_two_panel_pdf(tmp_path)
    with open_pdf(str(pdf)) as doc:
        page = doc[0]
        bboxes = chart_region_bboxes(page)
        interiors = region_interior_rows(page, bboxes)
    crops = {i: chart_region_filename(1, i) for i in range(1, len(bboxes) + 1)}
    once, sup, _ref = suppress_chart_table_skeletons(
        _winner(), page_num=1, interiors=interiors, crop_names=crops
    )
    assert len(sup) == 2
    twice, sup2, _r2 = suppress_chart_table_skeletons(
        once, page_num=1, interiors=interiors, crop_names=crops
    )
    assert twice == once and sup2 == []


# ---------------------------------------------------------------------------
# The corpus page this ticket is about
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not DOTPLOT_PDF.exists() or not DOTPLOT_PAGE.exists(),
    reason="dotplot corpus fixture is not present",
)
def test_dotplot_page_ships_five_crops_and_no_empty_grid(tmp_path: Path) -> None:
    """The FOMC SEP dot-plot page: five panels, five empty grids, five crops."""
    winner = DOTPLOT_PAGE.read_text()
    assert len(find_empty_skeletons(winner)) == 5, "the recorded page is not the five-grid shape"

    _p, off_state, off = _run(DOTPLOT_PDF, tmp_path / "off", winner, suppress=False)
    _p, on_state, on = _run(DOTPLOT_PDF, tmp_path / "on", winner, suppress=True)

    assert len(_table_lines(_body(off))) == 15, "control arm: five 3-line grids"
    assert _table_lines(_body(on)) == [], f"an empty grid still ships: {_body(on)!r}"

    assert len(_crops(tmp_path / "on")) == 5
    for index in range(1, 6):
        assert f"chart_region_p1_{index}.png" in _body(on)

    events = _events(on_state, SKELETON_SUPPRESSED)
    assert len(events) == 5
    assert [e.data["region_index"] for e in events] == [1, 2, 3, 4, 5]
    assert [e.data["label"] for e in events] == ["2018", "2019", "2020", "2021", "Longer run"]
    assert _events(off_state, SKELETON_SUPPRESSED) == []

    # The page's own prose survives.
    assert "Federal Open Market Committee" in _body(on)
    assert "notes to table 1" in _body(on)


# ---------------------------------------------------------------------------
# Round 2 (Astra): the four findings, each pinned where it failed
# ---------------------------------------------------------------------------


def test_partial_key_overlap_is_not_axis_attestation() -> None:
    """P1. ``1.88-9.99`` shares a token with the axis; the axis never drew 9.99.

    An unrelated empty form standing under a heading the chart also draws must
    survive, which is the case the design exists to protect.
    """
    text = (
        "### ALPHA\n\n| Plan | 1.88-9.99 | 2.13-unrelated |\n| --- | --- | --- |\n| Entry | | |\n"
    )
    result, events, refusals = suppress_chart_table_skeletons(
        text,
        page_num=1,
        interiors={1: ["ALPHA", "1.88 2.13"]},
        crop_names={1: "chart_region_p1_1.png"},
    )
    assert result == text
    assert events == []
    assert any("in full" in r.reason for r in refusals), [r.reason for r in refusals]


def test_complete_two_line_tick_label_does_attest() -> None:
    """The same keys, drawn in full: lower bounds on one row, upper bounds below."""
    text = "### ALPHA\n\n| Plan | 1.88-2.12 | 2.13-2.37 |\n| --- | --- | --- |\n| Entry | | |\n"
    result, events, _r = suppress_chart_table_skeletons(
        text,
        page_num=1,
        interiors={1: ["ALPHA", "1.88 2.13", "2.12 2.37"]},
        crop_names={1: "chart_region_p1_1.png"},
    )
    assert [e.region_index for e in events] == [1]
    assert "counts not extracted" in result


def test_upper_bounds_drawn_above_the_axis_line_do_not_attest() -> None:
    """Order matters: a two-line tick label's second line is drawn BELOW the first."""
    _result, events, _r = suppress_chart_table_skeletons(
        "### ALPHA\n\n| Plan | 1.88-2.12 |\n| --- | --- |\n| Entry | |\n",
        page_num=1,
        interiors={1: ["2.12", "ALPHA", "1.88"]},
        crop_names={1: "a.png"},
    )
    assert events == []


def test_a_fenced_example_is_never_rewritten() -> None:
    """P2. A grid inside a code fence is a sample, and the note must not land in it."""
    text = "### ALPHA\n\n```\n| Bin | 1.88 | 2.13 |\n| --- | --- | --- |\n| Entry | | |\n```\n"
    result, events, refusals = suppress_chart_table_skeletons(
        text,
        page_num=1,
        interiors={1: ["ALPHA", "1.88 2.13"]},
        crop_names={1: "chart_region_p1_1.png"},
    )
    assert result == text
    assert (events, refusals) == ([], [])
    assert find_empty_skeletons(text) == []


def test_a_heading_inside_a_fence_cannot_anchor_a_region() -> None:
    """The same masking on the LABEL side: a fenced heading is not the page's heading."""
    text = "```\n### ALPHA\n```\n\n| Bin | 1.88 | 2.13 |\n| --- | --- | --- |\n| Entry | | |\n"
    result, events, _r = suppress_chart_table_skeletons(
        text,
        page_num=1,
        interiors={1: ["ALPHA", "1.88 2.13"]},
        crop_names={1: "a.png"},
    )
    assert (result, events) == (text, [])


def test_suppression_provenance_survives_a_resume(tmp_path: Path) -> None:
    """P2. The withheld grid's bytes live ONLY in the event; the body cannot rebuild them."""
    from socr.core.result import PageOutput

    pdf = _make_two_panel_pdf(tmp_path)
    out_dir = tmp_path / "out"
    pipeline, state, _result = _run(pdf, out_dir, _winner())
    first = _events(state, SKELETON_SUPPRESSED)
    assert len(first) == 2
    assert state.pages[1].chart_table_skeletons_suppressed == 2

    meta = json.loads(next(out_dir.rglob("pages/00001.json")).read_text())
    restored_state = _make_state(pdf, "Preamble sentence unique alpha")
    page_out = PageOutput.from_dict(meta["winning_output"])
    pipeline._restore_terminal_page_state(restored_state, 1, page_out, out_dir)

    replayed = _events(restored_state, SKELETON_SUPPRESSED)
    assert len(replayed) == 2
    assert [e.data["original_text"] for e in replayed] == [e.data["original_text"] for e in first]
    assert [e.data["sha256"] for e in replayed] == [e.data["sha256"] for e in first]
    assert restored_state.pages[1].chart_table_skeletons_suppressed == 2

    # Replaying the same sidecar again restores the provenance without doubling
    # the count the CLI line reports.
    pipeline._restore_terminal_page_state(restored_state, 1, page_out, out_dir)
    assert len(_events(restored_state, SKELETON_SUPPRESSED)) == 2
    assert restored_state.pages[1].chart_table_skeletons_suppressed == 2


def test_the_page_judge_assesses_the_suppressed_candidate(tmp_path: Path) -> None:
    """P2. The withholding happens at candidate ingestion, BEFORE the page judge.

    ``route_page`` runs for real here -- only the provider and the judge are
    stand-ins -- so the ordering under test is the pipeline's own, not a mock's.
    """
    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.core.result import PageOutput, PageStatus
    from socr.pipeline.agentic import AcceptDecision
    from socr.pipeline.orchestrator import UnifiedPipeline

    pdf = _make_two_panel_pdf(tmp_path)
    winner = _winner()
    judged: list[str] = []

    class _RecordingJudge:
        def assess(self, output, provider) -> AcceptDecision:
            judged.append(output.text or "")
            return AcceptDecision(accept=True, reason="stand-in")

    def _fake_engine(self, state, pages, *args, **kwargs):
        return [
            PageOutput(
                page_num=pages[0],
                text=winner,
                status=PageStatus.SUCCESS,
                engine="qwen_local",
            )
        ]

    pipeline = _make_pipeline()
    state = _make_state(pdf, "Preamble sentence unique alpha")
    pipeline._last_assessment = state._last_assessment
    stack = [
        patch.object(
            UnifiedPipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]
        ),
        patch.object(UnifiedPipeline, "_run_engine_on_pages", _fake_engine),
        patch.object(UnifiedPipeline, "_build_page_judge", return_value=_RecordingJudge()),
    ]
    for ctx in stack:
        ctx.start()
    try:
        pipeline._phase_agentic(state, tmp_path / "out")
    finally:
        for ctx in reversed(stack):
            ctx.stop()

    assert judged, "the judge was never reached; this test would prove nothing"
    assert _table_lines(judged[0]) == [], f"the judge assessed an empty grid: {judged[0]!r}"
    assert "counts not extracted" in judged[0]
    assert len(_events(state, SKELETON_SUPPRESSED)) == 2


def test_a_later_candidate_cannot_reintroduce_the_grid(tmp_path: Path) -> None:
    """P2. Crop reread and escalation replace ``bo.text``; they cross the same seam."""
    pdf = _make_two_panel_pdf(tmp_path)
    pipeline, state, _result = _run(pdf, tmp_path / "out", _winner())
    assert len(_events(state, SKELETON_SUPPRESSED)) == 2

    bo = state.pages[1].best_output
    bo.text = _winner()  # a later reading puts both empty grids back
    withheld = pipeline._suppress_chart_table_skeletons(state, 1, bo)

    assert withheld == 2
    assert _table_lines(bo.text) == []
    # Same grids, same bytes: one finding, not two.
    assert len(_events(state, SKELETON_SUPPRESSED)) == 2
    assert state.pages[1].chart_table_skeletons_suppressed == 2
