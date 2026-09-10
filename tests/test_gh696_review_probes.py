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

import json
from pathlib import Path

import fitz
import pytest
from test_gh696_spanning_header_flatten import (
    _DATA_ROWS,
    _DATA_XS,
    _FONT_SIZE,
    _LEAF_BAND,
    _padded_markdown,
    _single_level_markdown,
    _survey_page,
)

from socr.judge.table_verdict import resolve_cell_refs
from socr.pipeline.agentic import AcceptDecision, NativeTableVerifierJudge
from socr.tables.binding import bind, classify_binding_evidence
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

    @pytest.mark.parametrize("value", ["", "18"])
    def test_body_tokens_repeated_in_header_do_not_erase_body(self, value):
        """A body row whose words also occur in the header is still a body row.

        The survey's own ``Overall`` printed again as a body label matched the
        group heading's vocabulary, and a vocabulary test read that as header
        material and deleted the row. With ``18`` in its first value cell the
        printed value went with it, because ``18`` occurs in ``Apr 18`` --
        native corroboration on both the row and the number, and both gone.
        The boundary now asks where the page prints THIS row, not whether its
        words appear in the header somewhere.
        """
        page = _survey_page()
        page.insert_text((58.5, 211), "Overall", fontsize=_FONT_SIZE)
        if value:
            page.insert_text((211.8, 211), value, fontsize=_FONT_SIZE)
        row = ["Overall", value] + [""] * 9
        lines = _padded_markdown().splitlines()
        lines.insert(3, "| " + " | ".join(row) + " |")

        after, count = repair_table_headers_in_text(page.get_text("words"), "\n".join(lines))

        assert count == 1
        assert find_table_blocks(after)[0].grid[1] == row

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

        assert _candidate_header_depth(grid, spanning[1], geom) == 2


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
        pinned by ``test_inner_judge_observes_repaired_coordinates`` below —
        plus the
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
        folded_bands = _candidate_header_depth(grid_before, _spanning_header_bands(geom)[1], geom)
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

    def test_inner_judge_observes_repaired_coordinates(self):
        """The behavioural ordering pin, on a page that really does delegate.

        The survey fixture ships on the verifier's EXACT_PASS, so the inner
        judge is never reached there and an assertion over what it saw would be
        vacuous. Forcing the verifier onto its AMBIGUOUS branch — and only
        that; the repair, the page and the markdown are all real — makes the
        delegation happen. The inner judge is called exactly once, sees exactly
        the repaired bytes, and an ``R1C2`` minted at that moment resolves to
        the first printed data value rather than to the leaf heading it would
        have named before the fold.
        """
        from unittest.mock import patch

        from socr.core.providers import PROFILE_QWEN_LOCAL
        from socr.core.result import PageOutput, PageStatus
        from socr.tables.native_verifier import VerifierResult

        page = _survey_page()
        before = _padded_markdown()
        expected_text, _count = repair_table_headers_in_text(page.get_text("words"), before)

        seen: list[str] = []
        resolved_at_delegation: list[list[str]] = []

        class _Inner:
            def assess(self, output, provider):
                seen.append(output.text)
                resolved_at_delegation.append(
                    list(resolve_cell_refs(output.text, ["R1C2"]).values())
                )
                return AcceptDecision(accept=True, reason="stub", confidence=1.0)

        judge = NativeTableVerifierJudge(
            inner=_Inner(),
            get_fitz_page=lambda _pn: page,
            is_table_page=lambda _pn: True,
        )
        output = PageOutput(
            page_num=1,
            text=before,
            status=PageStatus.SUCCESS,
            engine="qwen",
        )

        with patch(
            "socr.tables.native_verifier.verify_native_table",
            return_value=VerifierResult(warn=True, output_col_count=11),
        ):
            judge.assess(output, PROFILE_QWEN_LOCAL)

        assert seen == [expected_text]
        assert resolved_at_delegation == [["0"]]

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


# --------------------------------------------------------------------------
# Round-3 controls
# --------------------------------------------------------------------------


def test_three_candidate_header_rows_shift_two_and_reversed_order():
    """Three emitted header rows fold to one, so the body shifts by two.

    Both orderings of the printed header vocabulary are accepted: nothing on
    the page is printed below the band to contradict them, so they stay header
    and the arithmetic follows the count of bands folded, not their order.
    """
    page = _survey_page()
    base = _padded_markdown().splitlines()
    groups, separator, leaves = base[0], base[1], base[2]

    for headers in ([groups, groups, leaves], [leaves, groups, groups]):
        before = "\n".join([headers[0], separator, *headers[1:], *base[3:]])

        after, count = repair_table_headers_in_text(page.get_text("words"), before)

        assert count == 1
        for row in range(1, len(_DATA_ROWS) + 1):
            for col in range(1, 12):
                moved = resolve_cell_refs(after, [f"R{row}C{col}"])
                original = resolve_cell_refs(before, [f"R{row + 2}C{col}"])
                assert list(moved.values()) == list(original.values())


def test_already_flattened_header_is_byte_identical():
    """Repairing the repaired output is a no-op: the fold is idempotent."""
    page = _survey_page()
    once, first_count = repair_table_headers_in_text(page.get_text("words"), _padded_markdown())
    twice, second_count = repair_table_headers_in_text(page.get_text("words"), once)

    assert first_count == 1
    assert second_count == 0
    assert twice == once


# --------------------------------------------------------------------------
# Finding 4 — an ambiguous boundary row authorised a partial fold
# --------------------------------------------------------------------------


def test_footnote_vocabulary_cannot_stop_the_walk_half_way():
    """The reviewer's round-4 reproducer: a footnote makes the leaf band ambiguous.

    ``Apr Jul 18`` printed below the table accounts for every word of the
    printed leaf band. Under a body-first vocabulary test that row was called
    body -- but the repeated group row above it had already advanced the depth
    to two, so the repair folded the group bands and kept the leaf row as a
    sixth body row above five printed data rows, with ``R1C2`` naming
    ``Apr 18`` instead of the first printed value.

    Either outcome is correct: fold the three header rows completely, or leave
    the table alone. Shipping a flattened header with a header row still in
    the body is neither.
    """
    page = _survey_page()
    page.insert_text((72, 400), "Apr Jul 18", fontsize=_FONT_SIZE)
    base = _padded_markdown().splitlines()
    before = "\n".join([base[0], base[1], base[0], *base[2:]])

    after, count = repair_table_headers_in_text(page.get_text("words"), before)

    if count:
        assert find_table_blocks(after)[0].grid[1:] == [
            [label, *values] for label, values in _DATA_ROWS
        ]
    else:
        assert after == before


def test_a_footnote_no_longer_withholds_the_two_row_repair():
    """The same footnote on the ordinary two-band candidate: a complete flatten.

    This control used to pin ``count == 0``. That abstention was a safe
    alternative to an incorrect rewrite, never a requirement: the footnote is
    not part of this table, so its words never had standing to make the
    table's own leaf band ambiguous. Now that body evidence is scoped to the
    table in both dimensions, the leaf band is accounted for by the header
    alone and the repair the page justifies goes through.
    """
    page = _survey_page()
    page.insert_text((72, 400), "Apr Jul 18", fontsize=_FONT_SIZE)
    before = _padded_markdown()

    after, count = repair_table_headers_in_text(page.get_text("words"), before)

    assert count == 1
    assert find_table_blocks(after)[0].grid[1:] == [
        [label, *values] for label, values in _DATA_ROWS
    ]


def test_a_row_combining_two_printed_bands_still_folds():
    """One candidate row carrying both bands' words is header, and folds.

    No footnote, so no word of that row is printed below the band: the row is
    accounted for by the header alone and the ambiguity rule never fires.
    """
    page = _survey_page()
    base = _padded_markdown().splitlines()
    cells = find_table_blocks(_padded_markdown())[0].grid
    combined = [f"{upper} {lower}".strip() for upper, lower in zip(cells[0], cells[1])]
    before = "\n".join([base[0], base[1], "| " + " | ".join(combined) + " |", *base[3:]])

    after, count = repair_table_headers_in_text(page.get_text("words"), before)

    assert count == 1
    assert find_table_blocks(after)[0].grid[1:] == [
        [label, *values] for label, values in _DATA_ROWS
    ]


# --------------------------------------------------------------------------
# Finding 5 — a second table's heading row posed as this table's body
# --------------------------------------------------------------------------


def test_a_lower_tables_heading_is_not_this_tables_body():
    """The reviewer's round-5 reproducer: the witness has to belong to THIS table.

    A second table lower on the same page repeats the ten leaf headings. The
    upper candidate's leaf row matches that row's words exactly, so the
    round-4 escape read it as a located body occurrence, stopped the walk at
    depth two and folded — shipping the leaf headings twice, once in the
    reconstructed header and once as a sixth body row over five printed data
    rows. A row below the header floor is only a body witness if it lies
    inside the extent the geometry chain already attributed to this table.
    """
    page = _survey_page()
    page.insert_text((58.5, 330), "Second table", fontsize=_FONT_SIZE)
    for x, text in _LEAF_BAND:
        page.insert_text((x, 350), text, fontsize=_FONT_SIZE)
    page.insert_text((58.5, 365), "Second observation", fontsize=_FONT_SIZE)
    for x, value in zip(_DATA_XS, [str(i) for i in range(1, 11)]):
        page.insert_text((x, 365), value, fontsize=_FONT_SIZE)
    base = _padded_markdown().splitlines()
    before = "\n".join([base[0], base[1], base[0], *base[2:]])

    after, count = repair_table_headers_in_text(page.get_text("words"), before)

    if count:
        assert find_table_blocks(after)[0].grid[1:] == [
            [label, *values] for label, values in _DATA_ROWS
        ]
    else:
        assert after == before


def test_the_body_escape_is_not_a_cell_verification_credential():
    """Locating a row as body says where it belongs, not that its cells are right.

    A candidate whose two printed ``18``s sit one column right of where the
    page prints them still satisfies the escape, because the escape compares
    an unordered multiset. The repair leaves the row exactly as emitted — it
    neither relocates nor drops a value — and binding is what reports the
    shift, so the two mechanisms stay separate.
    """
    page = _survey_page()
    page.insert_text((58.5, 211), "Overall", fontsize=_FONT_SIZE)
    for x in _DATA_XS[:2]:
        page.insert_text((x, 211), "18", fontsize=_FONT_SIZE)
    shifted = ["Overall", "", "18", "18"] + [""] * 7
    lines = _padded_markdown().splitlines()
    lines.insert(3, "| " + " | ".join(shifted) + " |")

    after, count = repair_table_headers_in_text(page.get_text("words"), "\n".join(lines))

    assert count == 1
    assert find_table_blocks(after)[0].grid[1] == shifted

    result = bind(page.get_text("words"), after)
    assert classify_binding_evidence(result) is not None
    assert not result.fully_checked
    assert (
        result.contradicted_cells
        or result.model_unbound
        or result.native_unbound
        or result.column_binding_unverifiable
    )


_CENSUS_ROOT = Path.home() / "Data/socr/census-ecb-2026-09-06"
_CENSUS_SLUG = "ecb-surveys-2018-ecb.blssurvey2018q2.en-p37-39"


@pytest.mark.skipif(
    not (_CENSUS_ROOT / "in" / f"{_CENSUS_SLUG}.pdf").exists(),
    reason="local census corpus only — the cached candidates are not in the repo",
)
def test_the_motivating_census_page_still_repairs():
    """The page #696 was filed from: the bad candidate repairs, the clean one is left.

    This is the end the boundary rules exist for, and every tightening round
    has to keep clearing it: the emitted candidate whose ``grid_shape`` defect
    the flatten removes still has it removed, and the candidate that arrived
    clean is still untouched.
    """
    from socr.tables.locate import _horizontal_rules
    from socr.tables.structure_check import table_output_defect

    with fitz.open(_CENSUS_ROOT / "in" / f"{_CENSUS_SLUG}.pdf") as doc:
        page = doc[0]
        words = page.get_text("words")
        rules = _horizontal_rules(page)

    outcomes = []
    for path in sorted((_CENSUS_ROOT / "out" / _CENSUS_SLUG / "cache").glob("*/*.json")):
        data = json.loads(path.read_text())
        if data.get("page_num") != 1 or not data.get("text"):
            continue
        after, count = repair_table_headers_in_text(words, data["text"])
        outcomes.append(
            (
                count,
                table_output_defect(data["text"], words, rules),
                table_output_defect(after, words, rules),
            )
        )

    assert outcomes
    assert any(count and before and not after for count, before, after in outcomes)


# --------------------------------------------------------------------------
# Finding 6 — a table beside this one still posed as its body
# --------------------------------------------------------------------------


def test_a_side_by_side_tables_heading_is_not_this_tables_body():
    """The reviewer's round-6 reproducer: a y-interval is not a table.

    The neighbouring table sits inside the first table's vertical range and
    500pt to its right, and repeats the same ten leaf headings. Scoping the
    body witness by y alone still let that heading settle the first table's
    ambiguous leaf row as body, and the fold duplicated the leaf labels into
    the header and the first body row at once. Ownership now has to hold in
    both dimensions, so the neighbour is silent about this table and the leaf
    band folds where it belongs.
    """
    page = _survey_page()
    page.set_mediabox(fitz.Rect(0, 0, 1100, 500))
    page.insert_text((600, 230), "Second table", fontsize=_FONT_SIZE)
    for x, text in _LEAF_BAND:
        page.insert_text((x + 500, 252), text, fontsize=_FONT_SIZE)
    page.insert_text((600, 266), "Settlement dates", fontsize=_FONT_SIZE)
    page.insert_text((704.1, 266), "12/04/89", fontsize=_FONT_SIZE)
    base = _padded_markdown().splitlines()
    before = "\n".join([base[0], base[1], base[0], *base[2:]])

    after, count = repair_table_headers_in_text(page.get_text("words"), before)

    if count:
        assert find_table_blocks(after)[0].grid[1:] == [
            [label, *values] for label, values in _DATA_ROWS
        ]
    else:
        assert after == before


def test_a_trailing_source_line_survives_the_fold():
    """A text-only row below the last numeric row is not swept up by the fold.

    It sits outside the body witness's bottom edge, so nothing certifies it —
    but the walk has already stopped at the first established body row, and
    everything under that ships verbatim.
    """
    page = _survey_page()
    page.insert_text((58.5, 289), "Source: survey respondents", fontsize=_FONT_SIZE)
    before = _padded_markdown() + "| Source: survey respondents |" + " |" * 10 + "\n"

    after, _count = repair_table_headers_in_text(page.get_text("words"), before)

    assert find_table_blocks(after)[0].grid[-1][0] == "Source: survey respondents"


@pytest.mark.skipif(
    not (_CENSUS_ROOT / "in" / f"{_CENSUS_SLUG}.pdf").exists(),
    reason="local census corpus only — the cached candidates are not in the repo",
)
def test_a_footnote_changes_nothing_on_the_census_page():
    """Scoping is measured on the real page, not only on the fixture.

    Printing an unrelated ``Apr Jul 18`` at the foot of the actual 2018 page
    leaves both cached candidates' repair counts and defects exactly as they
    were, which is the property the token scoping was for.
    """
    from socr.tables.locate import _horizontal_rules
    from socr.tables.structure_check import table_output_defect

    with fitz.open(_CENSUS_ROOT / "in" / f"{_CENSUS_SLUG}.pdf") as doc:
        page = doc[0]
        original = page.get_text("words")
        rules = _horizontal_rules(page)
        page.insert_text((20, page.rect.height - 10), "Apr Jul 18", fontsize=6)
        with_footnote = page.get_text("words")

    seen = []
    for path in sorted((_CENSUS_ROOT / "out" / _CENSUS_SLUG / "cache").glob("*/*.json")):
        data = json.loads(path.read_text())
        if data.get("page_num") != 1 or not data.get("text"):
            continue
        outcomes = []
        for words in (original, with_footnote):
            after, count = repair_table_headers_in_text(words, data["text"])
            outcomes.append((count, table_output_defect(after, words, rules)))
        assert outcomes[0] == outcomes[1]
        seen.append(outcomes[0])

    assert (1, "") in seen
