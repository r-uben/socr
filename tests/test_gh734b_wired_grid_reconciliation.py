"""#734 Stage B: the filled-grid reconciler, wired into the pipeline.

Stage A built the pure reconciler and nothing called it: no page was checked by
it and no shipped number had ever been compared against geometry. This file
pins the wiring -- the binding that decides WHICH panel a grid describes, the
withholding of a contradicted cell, and the five surfaces the finding reaches.

Every pipeline pin here is a DIFFERENCE, run twice in one process with only the
thing under test changed, because the absolute status of a chart page depends on
machinery (the D3 floor, ``native_fallback``) that does not fire in CI. The
provider ladder and the judge are patched out for the same reason.

The policy being pinned, and none of it is this file's invention:

* a CONTRADICTED cell publishes NOTHING -- not the model's number and not
  geometry's, because nothing here adjudicates between them;
* every other cell keeps the model's number. Geometry with no opinion is not a
  contradiction, and refusing those cells would delete 424 readings on this
  corpus on the strength of a documented reader limit (#739);
* an unknown or refused cell is NEVER reported as checked;
* totals never force agreement -- there is no arithmetic in the lane at all;
* ``audit_passed`` is never touched. It is the winner-SELECTION flag (#252) and
  flipping it makes assemble DISCARD the page's text, turning a flagged cell
  into a lost page. Demotion is by page STATUS.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

fitz = pytest.importorskip("fitz")

from test_gh735_sep_reader import draw_panel  # noqa: E402

from socr.figures.chart_data import bind_filled_grids, find_filled_grids  # noqa: E402
from socr.figures.chart_reconcile import (  # noqa: E402
    CONTRADICTED_MARKER,
    GRID_CONTRADICTED,
    GRID_RECONCILE_REFUSED,
    GRID_RECONCILED,
    reconcile_grid,
    reconciliation_note,
    withhold_contradicted,
)

SERIES = "September projections"
#: What the one-panel fixture draws, bin by bin. ``draw_panel`` renders bars at
#: these heights and the reader derives them back off the vector geometry, so a
#: grid carrying these numbers AGREES and a grid carrying anything else does not.
DRAWN = {"1.0": 3, "2.0": 5, "3.0": 0, "4.0": 0}


def grid_text(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def one_grid(text: str):
    found = find_filled_grids(text)
    assert len(found) == 1, f"expected one filled grid, got {len(found)}"
    return found[0]


def counts_grid(**overrides: int) -> str:
    """The fixture's own counts as a transposed grid, with cells overridden.

    Transposed -- bins down the label column, the series across the header --
    because that is how every one of the 37 grids in the SEP corpus is written.
    """
    values = dict(DRAWN) | overrides
    return grid_text(["Percent range", SERIES], [[b, str(values[b])] for b in DRAWN])


def page_text(grid: str, anchor: str = SERIES) -> str:
    return f"Preamble unique alpha\n\n{anchor}\n\n{grid}\n\nTrailing unique delta\n"


# ---------------------------------------------------------------------------
# The binding: which panel does this grid describe?
# ---------------------------------------------------------------------------
#
# Pure, so these need no PDF. ``interiors`` is what the caller reads off the
# page's chart regions; the rules under test are about the CANDIDATE's layout.

ALPHA_GRID = grid_text(["Bin", "Count"], [["1.0", "3"]])
BRAVO_GRID = grid_text(["Bin", "Count"], [["2.0", "7"]])


def two_region_text(first: str, second: str, *, reversed_order: bool = False) -> str:
    top, bottom = ("BRAVO", "ALPHA") if reversed_order else ("ALPHA", "BRAVO")
    return (
        f"Preamble unique alpha\n\n{top}\n\n{first}\n\n"
        f"Middle prose unique bravo\n\n{bottom}\n\n{second}\n\nTrailing unique delta\n"
    )


INTERIORS = {1: ["ALPHA", "shared row"], 2: ["BRAVO", "shared row"]}


def test_binding_is_by_label_and_a_permutation_is_refused_not_bound_positionally() -> None:
    """Binding follows the page's own labels, never the grid's position.

    The DIFFERENCE: the same two grids in the same order, with only the two
    panel labels swapped. A binder that went by position could not tell the two
    arms apart and would return ``{1: 1, 2: 2}`` for both. This one reads the
    labels, finds that the candidate pairs region 2 with the FIRST grid, and
    refuses the page's whole set rather than keeping a plausible half -- so the
    swapped arm binds nothing at all. That asymmetry is the proof that position
    did not decide the forward arm either.
    """
    forward, refused = bind_filled_grids(
        two_region_text(ALPHA_GRID, BRAVO_GRID), page_num=1, interiors=INTERIORS
    )
    swapped, swapped_refusals = bind_filled_grids(
        two_region_text(ALPHA_GRID, BRAVO_GRID, reversed_order=True),
        page_num=1,
        interiors=INTERIORS,
    )
    assert refused == []
    assert {r: g.table_index for r, g in forward.items()} == {1: 1, 2: 2}
    assert swapped == {}
    assert swapped_refusals, "a permuted candidate must be refused, never bound positionally"


def test_another_panels_label_between_them_refuses_rather_than_guesses() -> None:
    """A grid separated from its anchor by another panel's label describes THAT
    panel, so binding it to the first would judge the model against the wrong
    geometry -- and a wrong binding manufactures false contradictions, which
    withhold correct numbers. The refusal is recorded, never silent."""
    text = f"Preamble unique alpha\n\nALPHA\n\nBRAVO\n\n{ALPHA_GRID}\n\nTrailing unique delta\n"
    bound, refused = bind_filled_grids(text, page_num=1, interiors=INTERIORS)

    assert 1 not in bound
    assert any("separates region 1" in r.reason for r in refused)


def test_a_candidate_whose_panels_run_backwards_refuses_the_whole_page() -> None:
    """Regions are keyed in source order, so bindings that run backwards mean
    the candidate's layout contradicts the page. No binding on it is evidence
    of anything, so the page's whole set goes -- not the ones that happen to
    fit, which is the shape that would keep a plausible half."""
    text = (
        f"Preamble unique alpha\n\nBRAVO\n\n{ALPHA_GRID}\n\n"
        f"Middle prose unique bravo\n\nALPHA\n\n{BRAVO_GRID}\n\nTrailing unique delta\n"
    )
    bound, refused = bind_filled_grids(text, page_num=1, interiors=INTERIORS)

    assert bound == {}
    assert any("backwards against the source order" in r.reason for r in refused)


def test_a_region_that_resolves_no_anchor_refuses_the_page(tmp_path: Path) -> None:
    """P5 route A: the destructive one, and the guard it needed.

    A region resolving no label of its own is simply ABSENT from ``anchors``, so
    the intervening-label guard has nothing to fire on and the grid shifts
    silently onto the wrong panel — bound, no refusal, and reconciled against
    geometry it does not describe. Identity cannot catch it: panels of one
    figure draw the same bins, so the grid matches the wrong panel's bins just
    as well. Stage B is the lane that DELETES, so a wrong binding replaces
    correct numbers with the marker and files a successful reconciliation.

    The DIFFERENCE is whether region 2 draws a label only it draws; everything
    else is held fixed. Before the pairing rule the test arm returned
    ``{1: grid 1}`` with no refusal at all.
    """
    text = f"Preamble unique alpha\n\nALPHA\n\nBRAVO\n\n{ALPHA_GRID}\n\nTrailing unique delta\n"
    control, control_refusals = bind_filled_grids(text, page_num=1, interiors=INTERIORS)
    blind, blind_refusals = bind_filled_grids(
        text, page_num=1, interiors={1: ["ALPHA", "shared"], 2: ["shared"]}
    )

    # Control: region 2 owns the grid, and region 1 is refused for the right reason.
    assert {r: g.table_index for r, g in control.items()} == {2: 1}
    assert any("separates region 1" in r.reason for r in control_refusals)

    # Test: nothing binds, and the page says why rather than shifting.
    assert blind == {}
    assert any("do not pair 1:1" in r.reason for r in blind_refusals)


def test_a_caption_below_its_figure_refuses_rather_than_shifting(tmp_path: Path) -> None:
    """P5 route B: no unresolved label needed, and the order check cannot see it.

    Same two grids, same two labels, same panel order — only the label/grid
    order within each panel flipped, which is what a caption drawn BELOW its
    figure produces. Every pairing shifts by one and the bindings stay
    ASCENDING, so the backwards-order rule is silent. Before the pairing rule
    this bound ``{1: grid 2}`` — a mis-binding — while the only refusal named
    the OTHER grid.
    """
    above = (
        f"Preamble unique alpha\n\nALPHA\n\n{ALPHA_GRID}\n\n"
        f"Middle unique bravo\n\nBRAVO\n\n{BRAVO_GRID}\n\nTrailing unique delta\n"
    )
    below = (
        f"Preamble unique alpha\n\n{ALPHA_GRID}\n\nALPHA\n\n"
        f"Middle unique bravo\n\n{BRAVO_GRID}\n\nBRAVO\n\nTrailing unique delta\n"
    )
    up, _up_ref = bind_filled_grids(above, page_num=1, interiors=INTERIORS)
    down, down_ref = bind_filled_grids(below, page_num=1, interiors=INTERIORS)

    assert {r: g.table_index for r, g in up.items()} == {1: 1, 2: 2}
    assert down == {}, "an off-by-one shift must refuse, never bind"
    assert any("do not pair 1:1" in r.reason for r in down_ref)


def test_a_page_that_fails_the_pairing_withholds_nothing(tmp_path: Path) -> None:
    """P5, end to end: the consequence, not just the binding.

    The two binding guards stop at ``bind_filled_grids`` and assert it returns
    nothing. This one carries it to the only thing that matters -- a page whose
    regions and grids do not pair 1:1 must reach the body unchanged, with no
    cell withheld. A binder that refuses while the caller still withholds would
    satisfy both of those guards and delete numbers anyway.

    The DIFFERENCE is one unrelated filled table appended to a contradicting
    page: without it the grid binds, reconciles and withholds; with it the page
    carries two filled grids against one chart region, fails the pairing, and
    must ship byte-identical.
    """
    contradicting = counts_grid(**{"1.0": 9})
    alone = page_text(contradicting)
    with_table = alone.rstrip("\n") + "\n\n" + grid_text(["Year", "Value"], [["2019", "42"]]) + "\n"

    _p, bound_state, bound_bo = _reconcile(tmp_path, "pair_ok", alone)
    _p2, unpaired_state, unpaired_bo = _reconcile(tmp_path, "pair_no", with_table)

    # Control: it binds, contradicts and withholds.
    assert CONTRADICTED_MARKER in bound_bo.text
    assert bound_state.pages[1].chart_grid_cells_contradicted == 1

    # Unpaired: nothing is checked, and NOTHING is removed from the body.
    assert unpaired_bo.text == with_table
    assert CONTRADICTED_MARKER not in unpaired_bo.text
    assert unpaired_state.pages[1].chart_grid_cells_contradicted == 0
    assert _kinds(unpaired_state, GRID_RECONCILED) == []
    # ...and the page says so rather than going quiet about it.
    assert _kinds(unpaired_state, GRID_RECONCILE_REFUSED)


def test_a_grid_no_region_binds_is_refused_and_not_silently_kept() -> None:
    """The finding this lane exists for: a filled grid on a chart page whose
    numbers were compared against nothing. Silence would report it as checked."""
    text = f"Preamble unique alpha\n\n{ALPHA_GRID}\n\nTrailing unique delta\n"
    bound, refused = bind_filled_grids(text, page_num=1, interiors=INTERIORS)

    assert bound == {}
    assert [r.table_index for r in refused] == [1]
    assert "compared" in refused[0].reason


def test_an_ordinary_table_page_binds_nothing_and_refuses_nothing() -> None:
    """No chart region means no finding. An ordinary table is the page's own
    content, and reporting a refusal for every table in the corpus would bury
    the real ones."""
    assert bind_filled_grids(page_text(ALPHA_GRID), page_num=1, interiors={}) == ({}, [])


# ---------------------------------------------------------------------------
# Withholding: what a contradiction removes, and what it must not
# ---------------------------------------------------------------------------


def _panel(tmp_path: Path, name: str):
    from socr.figures.chart_reader import read_chart_page
    from socr.tables.reconstruct import chart_region_bboxes

    doc, _full = draw_panel(tmp_path / f"{name}.pdf", bars={"1.0": 3, "2.0": 5})
    page = doc[0]
    return read_chart_page(page, chart_region_bboxes(page), page_num=1).panels[1]


def test_only_the_contradicted_cell_is_withheld(tmp_path: Path) -> None:
    """The DIFFERENCE: one grid against one panel, changing only one cell.

    Geometry reads 3 in bin 1.0. The agreeing arm publishes it; the
    contradicting arm withholds that cell and NOTHING else -- every other row
    is byte-identical between the two.
    """
    panel = _panel(tmp_path, "only")
    agreed = withhold_contradicted(
        one_grid(counts_grid()), reconcile_grid(one_grid(counts_grid()), panel)
    )
    wrong = counts_grid(**{"1.0": 9})
    contra = withhold_contradicted(one_grid(wrong), reconcile_grid(one_grid(wrong), panel))

    assert agreed is None, "an agreeing grid must be left byte-identical"
    assert contra is not None
    changed = [(a, b) for a, b in zip(wrong.split("\n"), contra.split("\n"), strict=True) if a != b]
    assert len(changed) == 1, changed
    assert CONTRADICTED_MARKER in changed[0][1]


def test_the_withheld_cell_publishes_neither_number(tmp_path: Path) -> None:
    """Not the model's value, not geometry's, and above all not a value derived
    from the two. Nothing here adjudicates between them."""
    panel = _panel(tmp_path, "neither")
    wrong = counts_grid(**{"1.0": 9})
    grid = one_grid(wrong)
    result = reconcile_grid(grid, panel)
    body = withhold_contradicted(grid, result)

    row = next(line for line in body.split("\n") if line.startswith("| 1.0 "))
    assert "9" not in row and "3" not in row
    assert CONTRADICTED_MARKER in row
    # Both readings survive on the verdict, for the audit record.
    cell = next(c for c in result.cells if c.bin_label == "1.0")
    assert (cell.model_count, cell.reader_count) == (9, 3)
    assert cell.published is None


def test_a_cell_geometry_could_not_check_keeps_its_number(tmp_path: Path) -> None:
    """The rule that separates this lane from content loss.

    A bin geometry never read is UNKNOWN, not contradicted, and the model's
    number survives. On the SEP corpus 424 cells are in exactly this state --
    every one of them a documented abstention (#739) -- so withholding them
    would drop 424 readings on the strength of a refusal to read.
    """
    panel = _panel(tmp_path, "unknown")
    text = grid_text(["Percent range", SERIES], [["1.0", "3"], ["9.9", "42"]])
    grid = one_grid(text)
    result = reconcile_grid(grid, panel)

    assert result.unknown == 1
    assert withhold_contradicted(grid, result) is None
    assert "42" in text


def test_totals_are_never_used_to_force_agreement(tmp_path: Path) -> None:
    """A wrong reader and a wrong model that happen to sum alike must not
    certify each other. The grid below sums to geometry's own total and still
    contradicts it cell by cell."""
    panel = _panel(tmp_path, "totals")
    swapped = counts_grid(**{"1.0": 5, "2.0": 3})
    result = reconcile_grid(one_grid(swapped), panel)

    assert sum(DRAWN.values()) == 5 + 3 + 0 + 0
    assert result.contradicted == 2
    assert all(c.published is None for c in result.cells if c.status == "contradicted")


# ---------------------------------------------------------------------------
# Disclosure: what the note is allowed to say
# ---------------------------------------------------------------------------


def _note(tmp_path: Path, name: str, text: str) -> str:
    panel = _panel(tmp_path, name)
    return reconciliation_note(1, [reconcile_grid(one_grid(text), panel)])


def test_the_note_reports_unchecked_cells_and_never_calls_them_checked(
    tmp_path: Path,
) -> None:
    note = _note(
        tmp_path, "unchecked", grid_text(["Percent range", SERIES], [["1.0", "3"], ["9.9", "42"]])
    )

    assert "1 cell(s) carry a number geometry did not check" in note
    assert "UNVERIFIED" in note


def test_the_note_never_sums_unknown_and_uncovered(tmp_path: Path) -> None:
    """A cell geometry could not resolve is BOTH unknown and uncovered -- the
    model wrote a number nobody could check, AND a reading existed that nothing
    compared. They answer different questions; adding them double-counts one
    cell. Pinned as a DIFFERENCE: the two figures appear on separate lines and
    no line carries their sum."""
    note = _note(
        tmp_path, "sums", grid_text(["Percent range", SERIES], [["1.0", "3"], ["9.9", "42"]])
    )

    result = reconcile_grid(
        one_grid(grid_text(["Percent range", SERIES], [["1.0", "3"], ["9.9", "42"]])),
        _panel(tmp_path, "sums2"),
    )
    total = result.unknown + result.uncovered_count
    assert result.unknown and result.uncovered_count
    assert f"{total} cell(s)" not in note
    assert f"{total} reading(s)" not in note


def test_coverage_is_reported_per_series_not_only_per_page(tmp_path: Path) -> None:
    """ "Geometry never saw the December column" is what a citation-corpus
    reader needs. "An opinion on a fifth of this page" hides which half was
    checked."""
    note = _note(
        tmp_path, "series", grid_text(["Percent range", SERIES], [["1.0", "3"], ["9.9", "42"]])
    )

    assert f"series “{SERIES}”" in note


def test_geometry_abstains_it_does_not_fail_or_disagree(tmp_path: Path) -> None:
    """On this corpus all 90 zero-resolving series refuse explicitly, every one
    ``dashed_stroke`` naming the segment it cannot decompose (#739). Calling
    that a failure or a disagreement states something false about the page."""
    note = _note(tmp_path, "abstain", counts_grid(**{"1.0": 9}))

    assert "ABSTAINED" in note or "abstain" not in note.lower()
    assert "geometry failed" not in note.lower()
    assert "geometry disagreed" not in note.lower()


def test_recall_loss_and_coverage_beside_published_are_reported_apart() -> None:
    """Ranked by co-occurrence, never by volume. A grid that published nothing
    can leave a whole chart uncovered and still ship no number -- that is recall
    loss. The dangerous shape is the small one: geometry never consulted BESIDE
    cells that did agree and publish, a table that reads as checked and is half
    a chart."""
    from socr.figures.chart_reconcile import GridReconciliation, UncoveredReading

    from socr.figures.chart_reconcile import AGREED, CellVerdict

    uncovered = (UncoveredReading(series_name="Dec", bin_label="1.0", reader_count=7),)
    published = GridReconciliation(
        page_num=1,
        region_index=1,
        table_index=1,
        orientation="bins_in_header",
        cells=(
            CellVerdict(
                bin_label="2.0",
                series_name="Sep",
                status=AGREED,
                model_text="4",
                model_count=4,
                reader_count=4,
            ),
        ),
        uncovered=uncovered,
    )
    silent = GridReconciliation(
        page_num=1,
        region_index=1,
        table_index=1,
        orientation="bins_in_header",
        uncovered=uncovered,
    )

    assert published.uncovered_beside_published == 1
    assert silent.uncovered_beside_published == 0
    assert "reads as checked" in reconciliation_note(1, [published])
    assert "reads as checked" not in reconciliation_note(1, [silent])


# ---------------------------------------------------------------------------
# Wired: the pipeline seam
# ---------------------------------------------------------------------------


def _pipeline():
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


def _state(pdf: Path):
    from socr.core.born_digital import DocumentAssessment, PageAssessment
    from socr.core.document import DocumentHandle
    from socr.core.state import DocumentState, PageState

    with patch.object(DocumentHandle, "__post_init__", lambda self: None):
        handle = DocumentHandle(path=pdf, page_count=1)
    state = DocumentState(handle=handle)
    state.pages[1] = PageState(
        page_num=1,
        is_born_digital=True,
        native_text="Preamble unique alpha",
        needs_ocr_enhancement=False,
        has_tables=True,
    )
    state._last_assessment = DocumentAssessment(
        path=pdf,
        pages=[
            PageAssessment(
                page_num=1,
                is_born_digital=True,
                native_text="Preamble unique alpha",
                confidence=0.9,
                needs_ocr_enhancement=False,
                has_tables=True,
            )
        ],
    )
    return state


def _candidate(text: str):
    from socr.core.result import PageOutput, PageStatus

    return PageOutput(
        page_num=1, text=text, status=PageStatus.SUCCESS, engine="deepseek", audit_passed=True
    )


def _reconcile(tmp_path: Path, name: str, text: str):
    """One crossing of the candidate boundary, on a freshly drawn fixture."""
    pdf = tmp_path / f"{name}.pdf"
    draw_panel(pdf, bars={"1.0": 3, "2.0": 5})
    pipeline, state, bo = _pipeline(), _state(pdf), _candidate(text)
    pipeline._reconcile_chart_table_grids(state, 1, bo)
    return pipeline, state, bo


def _kinds(state, kind: str) -> list:
    return [e for e in state.events if getattr(e, "kind", "") == kind]


def test_a_contradicted_cell_is_withheld_from_the_shipped_body(tmp_path: Path) -> None:
    """The whole ticket, as a difference: the same page, the same panel, one
    cell changed. The agreeing arm ships byte-identical; the contradicting arm
    ships the marker and records both readings."""
    agree_text = page_text(counts_grid())
    wrong_text = page_text(counts_grid(**{"1.0": 9}))
    _p, agree_state, agree_bo = _reconcile(tmp_path, "agree", agree_text)
    _p2, wrong_state, wrong_bo = _reconcile(tmp_path, "wrong", wrong_text)

    assert agree_bo.text == agree_text
    assert _kinds(agree_state, GRID_CONTRADICTED) == []
    assert agree_state.pages[1].chart_grid_cells_contradicted == 0

    assert wrong_bo.text != wrong_text
    assert CONTRADICTED_MARKER in wrong_bo.text
    assert wrong_state.pages[1].chart_grid_cells_contradicted == 1
    event = _kinds(wrong_state, GRID_CONTRADICTED)[0]
    assert (event.data["cell"]["model_count"], event.data["cell"]["reader_count"]) == (9, 3)


def test_audit_passed_is_never_touched_by_the_reconciler(tmp_path: Path) -> None:
    """The #252 trap. ``audit_passed`` selects the winner, so flipping it to
    flag a page makes assemble DISCARD that page's text -- a withheld cell
    would become a lost page."""
    _p, _s, bo = _reconcile(tmp_path, "flag", page_text(counts_grid(**{"1.0": 9})))

    assert bo.audit_passed is True
    assert CONTRADICTED_MARKER in bo.text


def test_socrs_own_published_derivation_is_not_reconciled_against_itself(
    tmp_path: Path,
) -> None:
    """#635 Stage 0 substitutes a REAL table of counts for a bound empty grid,
    and it is a filled grid on every later crossing. Reconciling geometry
    against itself corroborates nothing while reporting a check that happened.
    The DIFFERENCE is the head line socr stamps, and nothing else."""
    from socr.figures.chart_reader import derivation_prefix

    body = counts_grid()
    plain = page_text(body)
    head = f"{derivation_prefix(1, 1)} — read from the source"
    socr_authored = (
        f"Preamble unique alpha\n\n{SERIES}\n\n{head}\n\n{body}\n\nTrailing unique delta\n"
    )

    _p, plain_state, _b = _reconcile(tmp_path, "plain", plain)
    _p2, own_state, _b2 = _reconcile(tmp_path, "own", socr_authored)

    assert len(_kinds(plain_state, GRID_RECONCILED)) == 1
    assert _kinds(own_state, GRID_RECONCILED) == []


def test_a_second_crossing_of_the_withheld_bytes_files_no_second_verdict(
    tmp_path: Path,
) -> None:
    """The candidate boundary is crossed four times per page. Withholding
    CHANGES the bytes by design, so the grid's digest is a different digest and
    no dedup key built from it can match -- and the second pass reaches
    ``not_a_count`` on the withheld cell, filing a CLEAN reconciliation over the
    top of a contradiction. A consumer reading the latest event for that grid
    would conclude nothing was ever disputed.

    Measured on ``sep-20201216-p09`` before the fix: grid 5 contradicts once,
    and a second crossing filed ``contradicted: 0`` above it.
    """
    pipeline, state, bo = _reconcile(tmp_path, "twice", page_text(counts_grid(**{"1.0": 9})))
    after_first = [(e.kind, e.detail) for e in state.events]

    pipeline._reconcile_chart_table_grids(state, 1, bo)

    assert [(e.kind, e.detail) for e in state.events] == after_first
    assert [e.data["contradicted"] for e in _kinds(state, GRID_RECONCILED)] == [1]


def test_an_unbound_grid_on_a_chart_page_is_recorded_as_unchecked(tmp_path: Path) -> None:
    """Its numbers ship, and the finding is that nothing checked them."""
    _p, state, bo = _reconcile(
        tmp_path, "unbound", f"Preamble unique alpha\n\n{counts_grid()}\n\nTrailing unique delta\n"
    )

    assert len(_kinds(state, GRID_RECONCILE_REFUSED)) == 1
    assert _kinds(state, GRID_RECONCILED) == []


def test_the_page_note_states_what_was_checked_and_what_was_not(tmp_path: Path) -> None:
    _p, _s, bo = _reconcile(tmp_path, "note", page_text(counts_grid(**{"1.0": 9})))

    note = next(n for n in bo.audit_notes if "Chart grids on page 1" in n)
    assert "3 cell(s) agreed" in note
    assert "1 cell(s) CONTRADICTED" in note
    assert CONTRADICTED_MARKER in note


# ---------------------------------------------------------------------------
# The document surfaces: sidecar and CLI
# ---------------------------------------------------------------------------


def _accepted(text: str):
    decision = MagicMock()
    decision.accepted = True
    decision.attempts = []
    decision.final_output = _candidate(text)
    decision.winning_engine = "deepseek"
    decision.total_cost_usd = 0.0
    return decision


def _run(tmp_path: Path, name: str, text: str, out: Path):
    from socr.core.providers import PROFILE_QWEN_LOCAL
    from socr.pipeline import orchestrator as orch
    from socr.pipeline.orchestrator import UnifiedPipeline

    pdf = tmp_path / f"{name}.pdf"
    draw_panel(pdf, bars={"1.0": 3, "2.0": 5})
    pipeline, state = _pipeline(), _state(pdf)
    pipeline.config.quiet = False
    pipeline._last_assessment = state._last_assessment
    printed = MagicMock()
    with (
        patch.object(orch, "console", printed),
        patch("socr.pipeline.orchestrator.route_page", return_value=_accepted(text)),
        patch.object(
            UnifiedPipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]
        ),
        patch.object(UnifiedPipeline, "_resolve_judge_model", return_value=""),
    ):
        pipeline._phase_agentic(state, out)
        pipeline._phase_assemble(state, out)
    lines = [str(c.args[0]) for c in printed.print.call_args_list if c.args]
    return state, lines


def test_the_sidecar_and_the_cli_both_report_the_withheld_cell(tmp_path: Path) -> None:
    """A finding that reaches only the page object is not surfaced. The
    DIFFERENCE: the same run, one cell changed."""
    agree_state, agree_cli = _run(tmp_path, "cli_ok", page_text(counts_grid()), tmp_path / "a")
    wrong_state, wrong_cli = _run(
        tmp_path, "cli_bad", page_text(counts_grid(**{"1.0": 9})), tmp_path / "b"
    )

    said = [line for line in wrong_cli if "chart grid(s) checked" in line]
    assert said, wrong_cli
    assert "1 CONTRADICTED and withheld" in said[0]
    assert "0 CONTRADICTED and withheld" in next(
        line for line in agree_cli if "chart grid(s) checked" in line
    )

    sidecar = json.loads(next((tmp_path / "b").rglob("pages/00001.json")).read_text())
    kinds = [e.get("kind") for e in sidecar.get("audit_events") or []]
    assert GRID_CONTRADICTED in kinds
    notes = (sidecar.get("winning_output") or {}).get("audit_notes") or []
    assert any("CONTRADICTED" in n for n in notes), notes


def test_the_page_is_demoted_by_status_and_keeps_its_text(tmp_path: Path) -> None:
    """Demotion is by STATUS, never ``audit_passed`` -- and the text survives.

    Pinned as a difference plus an invariant rather than as an absolute: the
    agreeing arm's own status depends on machinery CI does not run, so what is
    asserted is that the contradicting arm is never SUCCESS while the agreeing
    arm records no contradiction at all.
    """
    from socr.core.result import PageStatus

    agree_state, _a = _run(tmp_path, "dem_ok", page_text(counts_grid()), tmp_path / "c")
    wrong_state, _w = _run(
        tmp_path, "dem_bad", page_text(counts_grid(**{"1.0": 9})), tmp_path / "d"
    )

    assert agree_state.pages[1].chart_grid_cells_contradicted == 0
    assert wrong_state.pages[1].chart_grid_cells_contradicted == 1

    winner = wrong_state.pages[1].best_output
    assert winner is not None
    assert winner.status is not PageStatus.SUCCESS
    # The whole point of demoting by status: the page's content is still here.
    assert CONTRADICTED_MARKER in winner.text
    assert "Trailing unique delta" in winner.text


def test_the_contradicted_page_names_its_failure_mode(tmp_path: Path) -> None:
    """The demotion sets a MODE as well as a status, and the mode is its own line.

    Separated from the status guard above deliberately, and the separation is
    the point. That test does discriminate on status -- removing the status
    demotion turns the contradicting arm back to SUCCESS and it goes red -- but
    it was ALSO going red under a mutant that broke ``withhold_contradicted``,
    because withholding fails upstream of the demotion and nothing downstream
    can run. A guard that only fails once a DIFFERENT failure has already
    occurred has not been shown to guard its own subject; measured at e1a0340,
    deleting the ``failure_mode`` assignment alone left all 23 tests green while
    the page reported ``FailureMode.NONE``.

    ``failure_mode`` is one of the five surfaces this ticket exists to populate
    -- it is what the sidecar and the manifest read -- so an unguarded line here
    would let the metadata surface be hollowed out with no signal at all.

    Pinned as a DIFFERENCE between the two arms rather than as an absolute: the
    agreeing arm's own mode depends on machinery CI does not run, and the
    demotion only fills a mode in when none was set. If both arms report the
    same mode, the assignment did nothing.
    """
    from socr.core.result import FailureMode

    agree_state, _a = _run(tmp_path, "fm_ok", page_text(counts_grid()), tmp_path / "e")
    wrong_state, _w = _run(tmp_path, "fm_bad", page_text(counts_grid(**{"1.0": 9})), tmp_path / "f")
    agree = agree_state.pages[1].best_output
    wrong = wrong_state.pages[1].best_output
    assert agree is not None and wrong is not None

    # The contradiction happened in exactly one arm, so the modes must differ.
    assert wrong.failure_mode is not agree.failure_mode
    # And where the agreeing arm carried no mode, the contradicting arm names
    # THIS one -- never a reused HALLUCINATION, which would publish a verdict
    # about which side is wrong that this lane has not reached.
    if agree.failure_mode is FailureMode.NONE:
        assert wrong.failure_mode is FailureMode.CHART_GRID_CONTRADICTED


def test_every_contradicted_cell_of_a_grid_gets_its_own_record(tmp_path: Path) -> None:
    """One record per withheld NUMBER, not one per grid.

    ``GRID_CONTRADICTED`` is recorded per cell, so a dedup key that names only
    the grid collides on the second cell and discards it and every one after.
    The failure is silent and the surfaces then disagree with each other: the
    body withholds three values and the note says three, while the event list
    and the page counter say one. On the corpus that is 10 withheld numbers in
    5 grids reported as 5, and the CLI prints the length of that event list.

    Pinned as a DIFFERENCE across cell COUNT rather than as an absolute: one
    contradicted cell and three, with everything else held fixed, and the four
    surfaces required to agree with the reconciler in both arms. A guard using
    a single bad cell cannot see this at all -- which is why the nine pipeline
    guards that preceded it did not.
    """
    for expected, overrides in ((1, {"1.0": 9}), (3, {"1.0": 9, "2.0": 9, "3.0": 9})):
        _p, state, bo = _reconcile(
            tmp_path, f"multi{expected}", page_text(counts_grid(**overrides))
        )

        reconciled = _kinds(state, GRID_RECONCILED)[0]
        assert reconciled.data["contradicted"] == expected
        assert len(_kinds(state, GRID_CONTRADICTED)) == expected
        assert state.pages[1].chart_grid_cells_contradicted == expected
        assert bo.text.count(CONTRADICTED_MARKER) == expected


def test_a_model_that_writes_the_marker_is_still_checked(tmp_path: Path) -> None:
    """socr's own withheld output is known by DIGEST, never by its contents.

    A substring test for the marker is a silent opt-out, and it was the only
    unrecorded route in this lane: a grid carrying ``WITHHELD`` anywhere
    produced no verdict, no refusal, no counter and an unchanged body. A model
    is entitled to write it -- central-bank releases redact values, and a model
    re-transcribing an earlier socr output carries the marker straight back in.

    It is also the rule ``_grid_authored_by_socr`` already follows for Stage 1's
    block, and for the same stated reason: a model is entitled to write a table
    that looks like socr's, so socr's own output is recognised by what socr
    stamped and never by what the text happens to contain.

    The DIFFERENCE: two arms identical but for one UNRELATED cell carrying the
    marker, both contradicting geometry at the same bin. They must reconcile
    alike.
    """
    plain = page_text(counts_grid(**{"1.0": 9}))
    marked = page_text(counts_grid(**{"1.0": 9}).replace("| 4.0 | 0 |", "| 4.0 | WITHHELD |"))
    assert CONTRADICTED_MARKER in marked and CONTRADICTED_MARKER not in plain

    _p, plain_state, plain_bo = _reconcile(tmp_path, "mk_plain", plain)
    _p2, marked_state, marked_bo = _reconcile(tmp_path, "mk_marked", marked)

    for state, bo in ((plain_state, plain_bo), (marked_state, marked_bo)):
        assert len(_kinds(state, GRID_RECONCILED)) == 1
        assert len(_kinds(state, GRID_CONTRADICTED)) == 1
        assert state.pages[1].chart_grid_cells_contradicted == 1
    # The contradiction found is the same contradiction in both arms.
    assert [e.data["cell"]["bin_label"] for e in _kinds(marked_state, GRID_CONTRADICTED)] == [
        e.data["cell"]["bin_label"] for e in _kinds(plain_state, GRID_CONTRADICTED)
    ]
    assert marked_bo.text != marked


def _assemble_cli(pipeline, state, out: Path) -> list[str]:
    """Re-run assembly with the console captured, and return what it printed."""
    from socr.pipeline import orchestrator as orch

    printed = MagicMock()
    with patch.object(orch, "console", printed):
        pipeline.config.quiet = False
        pipeline._phase_assemble(state, out)
    return [str(c.args[0]) for c in printed.print.call_args_list if c.args]


def test_a_resumed_page_still_reports_what_it_withheld(tmp_path: Path) -> None:
    """The worst shape this lane can produce, and the one a resume creates.

    A resumed page is terminal and never re-processed, so the reconciler does
    not run -- while the restored BODY still carries ``WITHHELD`` where numbers
    were removed. If the events are not replayed and the counters not rebuilt,
    the document ships with content removed and NO surface saying why: the CLI
    prints nothing, the contradicted counter reads zero, the sidecar carries no
    reconciliation.

    Deliberately a RESUMED RUN rather than a membership test over
    ``resume_restore_kinds``. A membership test passes the moment the kinds are
    added and stays green if the counters never come back -- which is the second
    half of the defect and the half it cannot reach. This repo has already been
    bitten there: that set's own docstring records a guard which rebuilt the
    union from the same sources and so could not see its own subject.
    """
    from socr.core.result import PageOutput

    out = tmp_path / "resume"
    state, _cli = _run(tmp_path, "res", page_text(counts_grid(**{"1.0": 9, "2.0": 9})), out)
    assert state.pages[1].chart_grid_cells_contradicted == 2

    meta = json.loads(next(out.rglob("pages/00001.json")).read_text())
    winner = PageOutput.from_dict(meta["winning_output"])
    assert CONTRADICTED_MARKER in (winner.text or ""), "the restored BODY is missing numbers"

    pipeline = _pipeline()
    restored = _state(tmp_path / "res.pdf")
    pipeline._restore_terminal_page_state(restored, 1, winner, out)

    # The record comes back...
    assert len(_kinds(restored, GRID_RECONCILED)) == 1
    assert len(_kinds(restored, GRID_CONTRADICTED)) == 2
    # ...and so do BOTH counters, which the kind set alone would not restore.
    assert restored.pages[1].chart_grids_reconciled == 1
    assert restored.pages[1].chart_grid_cells_contradicted == 2

    # Replaying the same sidecar again must not double what the CLI reports.
    pipeline._restore_terminal_page_state(restored, 1, winner, out)
    assert restored.pages[1].chart_grid_cells_contradicted == 2


def test_the_cli_counts_grids_and_cells_by_identity_not_by_event(tmp_path: Path) -> None:
    """The CLI must agree with the body, and with the counters, after a re-emit.

    Three crossings of one page: contradict, re-emit the SAME bytes (which must
    add nothing), then re-emit the same cell with a DIFFERENT wrong number --
    fresh bytes, correctly re-judged and re-withheld. Counting raw events then
    reported "2 filled chart grid(s) checked" for ONE grid, because it was the
    only surface reading neither the deduped counters nor the reconciler.

    The DIFFERENCE is the third crossing; the invariant is that the CLI figures
    equal the summed PageState counters in both arms.
    """
    out = tmp_path / "cli"
    state, _first = _run(tmp_path, "cli3", page_text(counts_grid(**{"1.0": 9})), out)
    pipeline = _pipeline()

    # Rung 2: the same bytes again. Rung 3: a different wrong number.
    bo = state.pages[1].best_output
    pipeline._reconcile_chart_table_grids(state, 1, _candidate(bo.text or ""))
    pipeline._reconcile_chart_table_grids(
        state, 1, _candidate(page_text(counts_grid(**{"1.0": 7})))
    )

    said = next(
        line for line in _assemble_cli(pipeline, state, out) if "chart grid(s) checked" in line
    )
    assert "1 filled chart grid(s) checked" in said, said
    assert f"{state.pages[1].chart_grid_cells_contradicted} CONTRADICTED and withheld" in said


def test_a_cell_a_later_rung_gets_right_is_retired_from_the_count(tmp_path: Path) -> None:
    """A contradiction belongs to one READING of a grid, not to the page.

    The page is crossed at several rungs. When a later rung gets the same cell
    right, the body no longer withholds anything -- so a count that unions every
    contradiction event ever filed keeps a retired one and reports a number the
    body does not contain. Measured before the fix: the latest reading said
    ``agreed 4, contradicted 0`` while the withheld count still said 1, so the
    CLI accounted for FIVE cells on a four-cell grid.

    This is the across-rung half of the same count P1 fixed within a rung, and
    the error changed sign: P1 was an undercount, this is an overcount. Pinned
    as a DIFFERENCE over whether the second rung corrects the cell.
    """
    corrected = _reconcile(tmp_path, "retire_a", page_text(counts_grid(**{"1.0": 9})))[1]
    still_wrong = _reconcile(tmp_path, "retire_b", page_text(counts_grid(**{"1.0": 9})))[1]
    pipeline = _pipeline()

    # Arm A: the second rung gets it right. Arm B: still wrong, different bytes.
    pipeline._reconcile_chart_table_grids(corrected, 1, _candidate(page_text(counts_grid())))
    pipeline._reconcile_chart_table_grids(
        still_wrong, 1, _candidate(page_text(counts_grid(**{"1.0": 8})))
    )

    assert corrected.pages[1].chart_grid_cells_contradicted == 0
    assert still_wrong.pages[1].chart_grid_cells_contradicted == 1

    # The surfaces agree with the body in both arms: the corrected page carries
    # no marker, the still-wrong page carries one.
    from socr.figures.chart_reconcile import (
        latest_grid_reconciliations,
        withheld_cell_identities,
    )

    for state, expected in ((corrected, 0), (still_wrong, 1)):
        events = [(e.page_num, e.kind, e.data or {}) for e in state.events]
        latest = list(latest_grid_reconciliations(events).values())
        assert len(latest) == 1, "one grid, however many rungs crossed it"
        assert len(withheld_cell_identities(events)) == expected
        # agreed + withheld must never exceed the grid's own cell count.
        assert latest[0]["agreed"] + expected <= len(DRAWN)


def test_one_unbound_grid_re_emitted_is_one_unchecked_grid(tmp_path: Path) -> None:
    """The refusal figure counts GRIDS, not events.

    An unbound grid re-emitted with different bytes files a refusal per
    candidate, correctly -- each is a real record of a real crossing. But the
    CLI line says how many grids went unchecked, and a raw event count turned
    one such grid into three. It was the only one of the three figures on that
    line still counted by event after P4 deduplicated the other two.
    """
    from socr.figures.chart_reconcile import unreconciled_grid_identities

    pipeline, state, _bo = _reconcile(
        tmp_path,
        "unb",
        f"Preamble unique alpha\n\n{counts_grid()}\n\nTrailing unique delta\n",
    )
    for value in (1, 2):
        pipeline._reconcile_chart_table_grids(
            state,
            1,
            _candidate(
                f"Preamble unique alpha\n\n{counts_grid(**{'4.0': value})}"
                "\n\nTrailing unique delta\n"
            ),
        )

    events = [(e.page_num, e.kind, e.data or {}) for e in state.events]
    # Every crossing is still recorded -- nothing is lost from the audit trail.
    assert len(_kinds(state, GRID_RECONCILE_REFUSED)) == 3
    # ...but the page carries ONE unchecked grid, and that is what is reported.
    assert len(unreconciled_grid_identities(events)) == 1


def test_an_unreadable_page_records_that_its_grids_went_unchecked(tmp_path: Path) -> None:
    """The last silent route in the lane, closed on Stage 0's precedent.

    Three routes leave a chart page unchecked: unreadable geometry, a
    model-written marker, and a resumed run. The other two are recorded; this
    one returned zero and said nothing at all. Stage 0 records its equivalent
    (``SKELETON_UNBOUND``) when the page's geometry raises, and the operator's
    question is the same however the checking failed.

    The DIFFERENCE: the same page and the same grid, with only the geometry
    probe failing.
    """
    from socr.pipeline.orchestrator import UnifiedPipeline

    text = page_text(counts_grid(**{"1.0": 9}))
    _p, ok_state, ok_bo = _reconcile(tmp_path, "geo_ok", text)

    pdf = tmp_path / "geo_bad.pdf"
    draw_panel(pdf, bars={"1.0": 3, "2.0": 5})
    pipeline, bad_state, bad_bo = _pipeline(), _state(pdf), _candidate(text)
    with patch.object(UnifiedPipeline, "_chart_page_geometry", return_value=None):
        pipeline._reconcile_chart_table_grids(bad_state, 1, bad_bo)

    assert len(_kinds(ok_state, GRID_RECONCILED)) == 1
    assert CONTRADICTED_MARKER in ok_bo.text

    # Nothing could be checked -- and the page says so rather than staying mute.
    assert _kinds(bad_state, GRID_RECONCILED) == []
    refusals = _kinds(bad_state, GRID_RECONCILE_REFUSED)
    assert len(refusals) == 1
    assert "geometry could not be read" in refusals[0].detail
    assert bad_bo.text == text


# ---------------------------------------------------------------------------
# The corpus this ticket was measured on
# ---------------------------------------------------------------------------

SEP_IN = Path.home() / "Data/socr/sep-dotplots/in"
SEP_OUT = Path.home() / "Data/socr/sep-dotplots/out"


@pytest.mark.skipif(
    not (SEP_IN.exists() and SEP_OUT.exists()), reason="SEP dot-plot corpus is not present"
)
def test_the_sep_corpus_reproduces_the_measured_shape() -> None:
    """The eight pages that shipped a filled chart grid, end to end.

    Not a golden of every number -- those are the team lead's and are recorded
    in the log. What is pinned here is the SHAPE the design rests on, and each
    line of it would have to change for a conclusion in the log to be wrong:
    every grid binds, geometry contradicts a handful of cells, the great
    majority are unchecked rather than disputed, and NOTHING is verified.

    Zero verified is the CORRECT answer and must never be loosened to move it:
    on every panel the prior meeting is a dashed staircase the reader refuses
    with a stated reason (#739), so a grid naming both series cannot reach
    complete coverage whatever this lane does.
    """
    from socr.figures.chart_data import region_interior_rows
    from socr.figures.chart_reader import read_chart_page
    from socr.tables.reconstruct import chart_region_bboxes

    bound = reconciled = agreed = contradicted = unknown = verified = 0
    for md in sorted(SEP_OUT.rglob("pages/00001.md")):
        pdf = SEP_IN / f"{md.parent.parent.name}.pdf"
        if not pdf.exists():
            continue
        text = md.read_text()
        if not find_filled_grids(text):
            continue
        page = fitz.open(str(pdf))[0]
        bboxes = chart_region_bboxes(page)
        bindings, _refused = bind_filled_grids(
            text, page_num=1, interiors=region_interior_rows(page, bboxes)
        )
        panels = read_chart_page(page, bboxes, page_num=1).panels
        bound += len(bindings)
        for region, grid in bindings.items():
            if region not in panels:
                continue
            result = reconcile_grid(grid, panels[region])
            if result.refusal:
                continue
            reconciled += 1
            agreed += result.agreed
            contradicted += result.contradicted
            unknown += result.unknown
            verified += result.verified

    assert bound == 37, "every filled grid on the corpus binds to a panel"
    assert reconciled == 33 and agreed == 106 and contradicted == 10 and unknown == 424
    assert verified == 0, "zero verified is correct here and is gated by #739, not by this lane"
    assert unknown > agreed, (
        "most cells are UNCHECKED, not disputed -- never report them as checked"
    )
