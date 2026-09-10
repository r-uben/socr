"""#696 round 2 — the three findings the cold review reproduced on ee8277a.

Every probe here started life in the reviewer's own file and is kept in the
reviewer's shape, including the controls that already passed, so a regression in
any of them fails as the review would have.

The three findings were: the header rewrite deleted a printed label-only body
row; a three-level parent heading reached only one of the child groups it
covers; and a judge's ``RnCm`` does not land on the same printed cell before and
after the flatten on a candidate that already parses.

The third has no fix that preserves the reference, and this module says so out
loud rather than asserting an invariant that the owner's FLATTEN ruling makes
impossible — see
``TestCoordinateContract.test_flatten_shifts_body_refs_by_the_bands_it_folds``.
"""

from __future__ import annotations

import fitz
import pytest
from test_gh696_spanning_header_flatten import (
    _DATA_ROWS,
    _FONT_SIZE,
    _padded_markdown,
    _single_level_markdown,
    _survey_page,
)

from socr.judge.table_verdict import resolve_cell_refs
from socr.pipeline.agentic import NativeTableVerifierJudge
from socr.tables.header_repair import (
    _candidate_header_depth,
    _claim_lanes,
    _fold_header_bands_into_lanes,
    repair_table_headers_in_text,
)
from socr.tables.reconcile import find_table_blocks


def _word(x0: float, x1: float, text: str, line: int) -> tuple:
    """One PyMuPDF word tuple, each on its own (block, line) so it is its own run."""
    return (x0, line * 10, x1, line * 10 + 8, text, 0, line, 0)


# --------------------------------------------------------------------------
# Finding 1 — the header rewrite deleted a printed body row
# --------------------------------------------------------------------------


class TestBodyRowsSurviveTheRewrite:
    def test_printed_panel_row_is_not_deleted(self):
        """The reviewer's reproducer: the panel row is on the PAGE and in the candidate.

        A label-only row between the leaf headings and the first value is body.
        The old boundary — "everything above the first sufficiently numeric row
        is header" — folded it into the header and it left the document.
        """
        page = _survey_page()
        page.insert_text((58.5, 211), "IMPORTANT PANEL", fontsize=_FONT_SIZE)
        lines = _padded_markdown().splitlines()
        lines.insert(3, "| IMPORTANT PANEL |" + " |" * 10)

        after, _count = repair_table_headers_in_text(page.get_text("words"), "\n".join(lines))

        assert "IMPORTANT PANEL" in after

    def test_label_only_body_row_before_numeric_rows_is_not_deleted(self):
        """The candidate-only variant: nothing on the page corroborates the row.

        Uncorroborated is not a licence to delete — this module rewrites header
        TEXT and owes the body back verbatim either way.
        """
        page = _survey_page()
        lines = _padded_markdown().splitlines()
        lines.insert(3, "| IMPORTANT PANEL |" + " |" * 10)

        after, _count = repair_table_headers_in_text(page.get_text("words"), "\n".join(lines))

        assert "IMPORTANT PANEL" in after

    def test_the_panel_row_keeps_its_place_above_the_first_value(self):
        """Not merely present: still the first body row, still ahead of the data."""
        page = _survey_page()
        page.insert_text((58.5, 211), "IMPORTANT PANEL", fontsize=_FONT_SIZE)
        lines = _padded_markdown().splitlines()
        lines.insert(3, "| IMPORTANT PANEL |" + " |" * 10)

        after, count = repair_table_headers_in_text(page.get_text("words"), "\n".join(lines))

        assert count == 1
        body = find_table_blocks(after)[0].grid[1:]
        assert body[0][0] == "IMPORTANT PANEL"
        assert body[1:] == [[label] + values for label, values in _DATA_ROWS]

    def test_extra_body_cell_is_not_silently_deleted(self):
        page = _survey_page()
        lines = _padded_markdown().splitlines()
        lines[-1] = lines[-1].rstrip()[:-1] + " EXTRA_VALUE |"
        before = "\n".join(lines)

        after, _count = repair_table_headers_in_text(page.get_text("words"), before)

        assert "EXTRA_VALUE" in before
        assert "EXTRA_VALUE" in after

    def test_header_depth_is_the_band_the_page_prints(self):
        """The boundary itself, read directly: one leaf row, then body."""
        page = _survey_page()
        grid = find_table_blocks(_padded_markdown())[0].grid
        from socr.tables.header_repair import _spanning_header_bands, _table_geometry

        geom = _table_geometry(grid, page.get_text("words"))
        assert geom is not None
        spanning = _spanning_header_bands(geom)
        assert spanning is not None

        assert _candidate_header_depth(grid, spanning[1]) == 2


# --------------------------------------------------------------------------
# Finding 2 — a three-level parent reached only one child group
# --------------------------------------------------------------------------


class TestParentHeadingsReachEveryChild:
    def test_three_level_parent_reaches_both_subgroups(self):
        """``Domestic`` spans two of the three subgroups and heads all four columns.

        Its overlap with the two child extents is 40 against 35: under greatest
        overlap it went to the first child alone and vanished from columns 3
        and 4, with nonempty leaf labels hiding the loss from the faithfulness
        check.
        """
        bands = [(100 + i * 20, 120 + i * 20) for i in range(6)]
        rows = [
            [_word(100, 175, "Domestic", 0), _word(185, 215, "Foreign", 1)],
            [
                _word(105, 135, "Group A", 2),
                _word(145, 175, "Group B", 3),
                _word(185, 215, "Group C", 4),
            ],
            [_word(105 + i * 20, 115 + i * 20, f"Leaf{i}", 5 + i) for i in range(6)],
        ]

        folded = _fold_header_bands_into_lanes(rows, bands)

        assert folded is not None
        assert all("Domestic" in folded[i] for i in range(1, 5))
        assert folded[1:] == [
            "Domestic Group A Leaf0",
            "Domestic Group A Leaf1",
            "Domestic Group B Leaf2",
            "Domestic Group B Leaf3",
            "Foreign Group C Leaf4",
            "Foreign Group C Leaf5",
        ]

    def test_parent_containing_leaf_keeps_both_levels(self):
        bands = [(100 + i * 20, 120 + i * 20) for i in range(4)]
        rows = [
            [_word(105, 135, "Apr Summary", 0), _word(145, 175, "Other", 1)],
            [
                _word(105 + i * 20, 115 + i * 20, text, i + 2)
                for i, text in enumerate(["Apr", "Jul", "Apr", "Jul"])
            ],
        ]

        folded = _fold_header_bands_into_lanes(rows, bands)

        assert folded[1] == "Apr Summary Apr"

    def test_odd_column_group_and_tied_overlap(self):
        bands = [(100 + i * 20, 120 + i * 20) for i in range(5)]
        rows = [
            [_word(105, 155, "Odd group", 0), _word(165, 195, "Pair", 1)],
            [_word(105 + i * 20, 115 + i * 20, f"Leaf{i}", i + 2) for i in range(5)],
        ]

        folded = _fold_header_bands_into_lanes(rows, bands)

        assert folded == [
            "",
            "Odd group Leaf0",
            "Odd group Leaf1",
            "Odd group Leaf2",
            "Pair Leaf3",
            "Pair Leaf4",
        ]
        assert _claim_lanes([(105, 115, "A"), (105, 115, "B")], bands, {0}) is None

    def test_single_level_wrapped_per_column_abstains(self):
        bands = [(100 + i * 20, 120 + i * 20) for i in range(4)]
        rows = [
            [_word(105 + i * 20, 115 + i * 20, f"Upper{i}", i) for i in range(4)],
            [_word(105 + i * 20, 115 + i * 20, f"Lower{i}", i + 4) for i in range(4)],
        ]

        assert _fold_header_bands_into_lanes(rows, bands) is None


# --------------------------------------------------------------------------
# Finding 3 — the coordinate contract on a candidate that already parses
# --------------------------------------------------------------------------


class _RecordingInnerJudge:
    """An inner judge that remembers the bytes it was asked to assess."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def assess(self, output, provider):
        from socr.core.result import AcceptDecision

        self.seen.append(output.text)
        return AcceptDecision(accept=True, reason="stub", confidence=1.0)


class TestCoordinateContract:
    def test_flatten_shifts_body_refs_by_the_bands_it_folds(self):
        """The honest statement of finding 3, pinned rather than wished away.

        On ``_padded_markdown`` the candidate parses before the repair: its
        markdown header is the padded group row and ``R1`` is the leaf band.
        The owner's ruling folds that band INTO the header, so the leaf row
        stops being a row and every body reference moves up by exactly the
        number of bands folded away minus one. There is no flattening that both
        obeys the ruling and leaves ``R1C2`` on ``Apr 18``: after the fold that
        cell is a header cell, not a body one.

        What the ruling's "coordinates must not shift" constraint therefore
        protects is that no reference is ever resolved across the rewrite —
        pinned by ``test_repair_precedes_every_rncm_emitter`` below — plus the
        column contract, which does hold exactly: same column count, same
        column for every value.
        """
        page = _survey_page()
        before = _padded_markdown()
        grid_before = find_table_blocks(before)[0].grid

        after, count = repair_table_headers_in_text(page.get_text("words"), before)
        assert count == 1

        from socr.tables.header_repair import _spanning_header_bands, _table_geometry

        geom = _table_geometry(grid_before, page.get_text("words"))
        folded_bands = _candidate_header_depth(grid_before, _spanning_header_bands(geom)[1])
        shift = folded_bands - 1
        assert shift == 1

        body_rows = len(find_table_blocks(after)[0].grid) - 1
        for row in range(1, body_rows + 1):
            for col in range(1, 12):
                moved = resolve_cell_refs(after, [f"R{row}C{col}"])
                original = resolve_cell_refs(before, [f"R{row + shift}C{col}"])
                assert list(moved.values()) == list(original.values())

    def test_flatten_changes_no_column_coordinate(self):
        """The half of the constraint that does hold: values never change column."""
        page = _survey_page()
        before = _padded_markdown()
        after, _count = repair_table_headers_in_text(page.get_text("words"), before)

        refs = [f"R{row}C{col}" for row in range(1, len(_DATA_ROWS) + 1) for col in range(1, 12)]
        resolved = resolve_cell_refs(after, refs)

        assert resolved is not None
        expected = {}
        for row_index, (label, values) in enumerate(_DATA_ROWS, start=1):
            for col_index, cell in enumerate([label] + values, start=1):
                expected[f"R{row_index}C{col_index}"] = cell
        assert {str(ref): text for ref, text in resolved.items()} == expected

    @pytest.mark.parametrize("emitted", [_padded_markdown, _single_level_markdown])
    def test_the_judge_leaves_only_repaired_bytes_behind(self, emitted):
        """``assess`` rewrites ``output.text`` in place before it returns.

        The table ladder — the only machinery in socr that emits ``RnCm`` —
        runs later in the page loop and builds its witness from
        ``best_output.text``, i.e. from the object this mutated. So the row
        shift above is never observable across the rewrite: by the time any
        reference exists, the old indices are gone.
        """
        from socr.core.providers import PROFILE_QWEN_LOCAL
        from socr.core.result import PageOutput, PageStatus

        page = _survey_page()
        before = emitted()
        expected_text, count = repair_table_headers_in_text(page.get_text("words"), before)

        inner = _RecordingInnerJudge()
        judge = NativeTableVerifierJudge(
            inner=inner,
            get_fitz_page=lambda _pn: page,
            is_table_page=lambda _pn: True,
            record_event=lambda _e: None,
        )
        output = PageOutput(
            page_num=1,
            text=before,
            status=PageStatus.SUCCESS,
            engine="qwen",
            audit_passed=True,
        )

        judge.assess(output, PROFILE_QWEN_LOCAL)

        assert output.text == expected_text
        assert (output.text != before) is bool(count)
        assert all(seen == expected_text for seen in inner.seen)

    def test_the_repair_runs_before_the_inner_judge_is_consulted(self):
        """Source-order pin, because the fixture above ships on EXACT_PASS.

        On the survey page the deterministic verifier passes and the inner
        judge is never reached, so an assertion over what the inner judge saw
        would be vacuous there. This reads ``assess`` itself: the repair call
        must precede every ``self._inner.assess`` on the table-page path. If
        someone moves the repair below a delegation, an ``RnCm`` minted by a
        judge on that path would name a row the shipped bytes no longer have.
        """
        import inspect

        source = inspect.getsource(NativeTableVerifierJudge.assess)
        verify_at = source.index("vr = verify_native_table(")
        repair_at = source.index("_maybe_repair_collapsed_headers")
        delegations = [
            index for index in range(len(source)) if source.startswith("self._inner.assess(", index)
        ]
        on_the_geometry_path = [index for index in delegations if index > verify_at]

        assert delegations, "assess no longer delegates to the inner judge"
        assert verify_at < repair_at, "the repair no longer follows the verifier it repairs for"
        assert on_the_geometry_path, "no delegation left on the born-digital table path"
        assert all(index > repair_at for index in on_the_geometry_path), (
            "a delegation on the born-digital table path now precedes the header repair"
        )

    def test_the_ladder_witness_is_built_from_the_repaired_output(self):
        """The one ``resolve_cell_refs`` consumer reads ``best_output.text``.

        ``_run_table_judge_gate`` hands ``prepare_table_witnesses`` the page's
        ``best_output`` text — the same object ``assess`` rewrote — and
        ``_resolve_table_guard_chain`` then resolves the doubted refs against
        ``witness.markdown``. Pinned by source so that sourcing the witness
        from a cached or pre-repair copy fails here instead of silently
        resolving references against bytes that never shipped.
        """
        import inspect

        from socr.pipeline.orchestrator import UnifiedPipeline

        gate = inspect.getsource(UnifiedPipeline._run_table_judge_gate)
        assert "prepare_table_witnesses(state.handle.path, page_num, bo.text)" in gate

        guard = inspect.getsource(UnifiedPipeline._resolve_table_guard_chain)
        assert "resolve_cell_refs(witness.markdown, refs)" in guard


# --------------------------------------------------------------------------
# Controls the review ran and that must keep passing
# --------------------------------------------------------------------------


def test_two_blocks_different_header_depths_preserve_flat_one():
    page = _survey_page()
    flat = _single_level_markdown()
    text = _padded_markdown() + "\nBetween tables.\n\n" + flat

    out, count = repair_table_headers_in_text(page.get_text("words"), text)

    assert count == 1
    assert find_table_blocks(out)[1].grid == find_table_blocks(flat)[0].grid
    assert "Between tables." in out


def test_fixture_page_is_a_real_pdf_page():
    """Grounding canary: the geometry under every probe is a rendered page."""
    assert isinstance(_survey_page(), fitz.Page)
