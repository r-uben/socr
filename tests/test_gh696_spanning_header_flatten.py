"""#696 — a spanning group heading folds into each column beneath it.

Measured 2026-09-10 by hand-judging the ECB census corpus: 9 of 30 pages shipped
``[page N failed: unverifiable table — see image]`` instead of their table, and
every lost page carried a two-level column header. The model emits the two
levels as two markdown rows (or one padded with blanks), Markdown has no
spanning cell, and the header ends up narrower than — or misaligned with — its
own body. Owner ruling on the issue: FLATTEN. The 2018 BLS survey Q6 table ships
ten data columns headed ``Overall Apr 18``, ``Overall Jul 18``, ``Loans to small
and medium-sized enterprises Apr 18``, and so on, in one header row.

The fixture is the real geometry of
``ecb-surveys-2018-ecb.blssurvey2018q2.en-p37-39`` page 1 — every word at the x
it is printed at, rendered at the size that reproduces the source page's word
extents to a tenth of a point — with the survey's own labels and values, so the
lane pitch, the wrapped group headings and the ``Apr 18``/``Jul 18`` leaf band
are the ones that produced the defect. No PDF ships with the tests.
"""

from __future__ import annotations

import fitz
import pytest

from socr.judge.table_verdict import resolve_cell_refs
from socr.tables.header_repair import (
    native_header_row,
    repair_table_headers_in_text,
)
from socr.tables.reconcile import find_table_blocks

# Rendering at this size reproduces the source page's word extents; the runs
# PyMuPDF then groups match the ones the real page yields word for word.
_FONT_SIZE = 6.05

# (x, y, word) for the wrapped group-heading band, exactly as page 1 prints it.
_GROUP_BAND = [
    (262.3, 169.0, "Loans"),
    (281.8, 169.0, "to"),
    (289.2, 169.0, "small"),
    (257.2, 178.0, "and"),
    (269.5, 178.0, "medium-sized"),
    (318.6, 178.0, "Loans"),
    (338.1, 178.0, "to"),
    (345.5, 178.0, "large"),
    (217.3, 186.0, "Overall"),
    (267.3, 186.0, "enterprises"),
    (323.1, 186.0, "enterprises"),
    (371.2, 186.0, "Short-term"),
    (403.5, 186.0, "loans"),
    (427.6, 186.0, "Long-term"),
    (459.0, 186.0, "loans"),
]

# The leaf band: five Apr/Jul pairs, one under each group heading.
_LEAF_BAND = [
    (204.1, "Apr"),
    (216.0, "18"),
    (232.9, "Jul"),
    (243.3, "18"),
    (260.1, "Apr"),
    (271.9, "18"),
    (288.9, "Jul"),
    (299.2, "18"),
    (316.0, "Apr"),
    (327.9, "18"),
    (344.8, "Jul"),
    (355.1, "18"),
    (371.9, "Apr"),
    (383.8, "18"),
    (400.7, "Jul"),
    (411.0, "18"),
    (427.8, "Apr"),
    (439.7, "18"),
    (456.6, "Jul"),
    (466.9, "18"),
]

_DATA_XS = [211.8, 239.8, 267.7, 295.7, 323.7, 351.6, 379.6, 407.5, 435.5, 463.5]

_DATA_ROWS = [
    ("Decreased considerably", ["0", "0", "0", "0", "0", "0", "0", "0", "0", "0"]),
    ("Decreased somewhat", ["7", "10", "9", "9", "7", "13", "6", "7", "5", "11"]),
    (
        "Remained basically unchanged",
        ["72", "64", "68", "64", "74", "62", "78", "77", "71", "58"],
    ),
    ("Increased somewhat", ["22", "26", "21", "25", "18", "23", "15", "16", "23", "30"]),
    (
        "Mean",
        ["3.15", "3.16", "3.13", "3.17", "3.12", "3.12", "3.09", "3.09", "3.18", "3.19"],
    ),
]

#: The ten column names the owner's ruling names, in order. Same list appears as
#: the worked example in ``prompts/table_extract.md``, which is how the model is
#: told to produce the form the native reconstruction below derives.
EXPECTED_HEADER = [
    "",
    "Overall Apr 18",
    "Overall Jul 18",
    "Loans to small and medium-sized enterprises Apr 18",
    "Loans to small and medium-sized enterprises Jul 18",
    "Loans to large enterprises Apr 18",
    "Loans to large enterprises Jul 18",
    "Short-term loans Apr 18",
    "Short-term loans Jul 18",
    "Long-term loans Apr 18",
    "Long-term loans Jul 18",
]


def _survey_page() -> fitz.Page:
    """The 2018 BLS survey Q6 page, at its printed geometry."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=500)
    for x, y, word in _GROUP_BAND:
        page.insert_text((x, y + 6.0), word, fontsize=_FONT_SIZE)
    for x, word in _LEAF_BAND:
        page.insert_text((x, 206.0), word, fontsize=_FONT_SIZE)
    for index, (label, values) in enumerate(_DATA_ROWS):
        y = 217.0 + index * 14.0
        page.insert_text((58.5, y), label, fontsize=_FONT_SIZE)
        for x, value in zip(_DATA_XS, values):
            page.insert_text((x, y), value, fontsize=_FONT_SIZE)
    return page


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _stacked_markdown() -> str:
    """What the model emits today: two header rows, the upper one narrower.

    Verbatim shape of the refused candidate for this page (cached under the
    2026-09-06 census run): six cells of group names over eleven of periods.
    """
    lines = [
        _row(
            [
                "",
                "Overall",
                "Loans to small and medium-sized enterprises",
                "Loans to large enterprises",
                "Short-term loans",
                "Long-term loans",
            ]
        ),
        _row([":---"] * 11),
        _row([""] + ["Apr 18", "Jul 18"] * 5),
    ]
    lines += [_row([label] + values) for label, values in _DATA_ROWS]
    return "\n".join(lines) + "\n"


def _padded_markdown() -> str:
    """The other emitted shape: full width, group names padded with blanks."""
    groups = [
        "",
        "Overall",
        "",
        "Loans to small and medium-sized enterprises",
        "",
        "Loans to large enterprises",
        "",
        "Short-term loans",
        "",
        "Long-term loans",
        "",
    ]
    lines = [_row(groups), _row([":---"] * 11), _row([""] + ["Apr 18", "Jul 18"] * 5)]
    lines += [_row([label] + values) for label, values in _DATA_ROWS]
    return "\n".join(lines) + "\n"


def _single_level_markdown() -> str:
    """The same body under one header row — the untouched case."""
    lines = [
        _row([""] + [f"{period} {index}" for index, period in enumerate(["Apr", "Jul"] * 5)]),
        _row([":---"] * 11),
    ]
    lines += [_row([label] + values) for label, values in _DATA_ROWS]
    return "\n".join(lines) + "\n"


class TestFlattenSpanningHeader:
    def test_native_reconstruction_yields_the_ten_ruled_column_names(self):
        """Done-when 4: the 2018 p1 shape, five groups over two periods each."""
        page = _survey_page()
        grid = find_table_blocks(_stacked_markdown())[0].grid

        assert native_header_row(grid, page.get_text("words")) == EXPECTED_HEADER

    def test_stacked_header_is_repaired_to_one_row_of_flattened_names(self):
        page = _survey_page()
        repaired, count = repair_table_headers_in_text(page.get_text("words"), _stacked_markdown())

        assert count == 1
        grid = find_table_blocks(repaired)[0].grid
        assert grid[0] == EXPECTED_HEADER

    def test_padded_full_width_header_is_repaired_too(self):
        """The p2/p3 shape: group names padded to the body's width with blanks.

        ``detect_header_column_collapse`` sees nothing wrong with it, so before
        #696 nothing touched it and the leaf labels stayed a row away from the
        values they head.
        """
        page = _survey_page()
        repaired, count = repair_table_headers_in_text(page.get_text("words"), _padded_markdown())

        assert count == 1
        grid = find_table_blocks(repaired)[0].grid
        assert grid[0] == EXPECTED_HEADER

    @pytest.mark.parametrize("emitted", [_stacked_markdown, _padded_markdown])
    def test_flattening_changes_the_header_and_nothing_else(self, emitted):
        """Header TEXT only: same column count, same values, same row order."""
        page = _survey_page()
        before = emitted()
        after, _count = repair_table_headers_in_text(page.get_text("words"), before)

        body_after = find_table_blocks(after)[0].grid[1:]
        assert body_after == [[label] + values for label, values in _DATA_ROWS]
        assert all(len(row) == 11 for row in body_after)

    def test_single_level_header_is_byte_identical(self):
        """Done-when 2. The fold must not fire when there is nothing spanning."""
        page = _survey_page()
        before = _single_level_markdown()

        after, count = repair_table_headers_in_text(page.get_text("words"), before)

        assert count == 0
        assert after == before

    def test_resolve_cell_refs_lands_on_the_printed_data_cell(self):
        """Done-when 3, first half.

        A judge's ``RnCm`` resolves against the physical rows of the markdown
        that shipped, and the flattening runs before any judge sees the page
        (``NativeTableVerifierJudge.assess`` repairs headers at its top, before
        the tri-state dispatch and before the structural gate). So the pin is on
        the shipped text: every body reference must land on the value actually
        printed in that row and column, which is exactly what a rewrite that
        shifted a coordinate would break.

        Note the pre-flatten markdown does not resolve at all — its header row
        is narrower than its body, so ``parse_grid`` refuses the whole table.
        That is the defect, not a baseline to compare against.
        """
        page = _survey_page()
        before = _stacked_markdown()
        after, _count = repair_table_headers_in_text(page.get_text("words"), before)

        assert resolve_cell_refs(before, ["R1C1"]) is None

        refs = [f"R{row}C{col}" for row in range(1, len(_DATA_ROWS) + 1) for col in range(1, 12)]
        resolved = resolve_cell_refs(after, refs)

        assert resolved is not None
        expected = {}
        for row_index, (label, values) in enumerate(_DATA_ROWS, start=1):
            for col_index, cell in enumerate([label] + values, start=1):
                expected[f"R{row_index}C{col_index}"] = cell
        assert {str(ref): text for ref, text in resolved.items()} == expected

    def test_single_level_table_resolves_identically_before_and_after(self):
        """Done-when 3, second half: the untouched case really is untouched."""
        page = _survey_page()
        before = _single_level_markdown()
        after, _count = repair_table_headers_in_text(page.get_text("words"), before)

        refs = [f"R{row}C1" for row in range(1, len(_DATA_ROWS) + 1)]

        assert resolve_cell_refs(after, refs) == resolve_cell_refs(before, refs)

    def test_prompt_worked_example_matches_the_native_reconstruction(self):
        """The two sides must agree on the SAME flattened form.

        The extraction prompt tells the model that a group ``Overall`` above an
        ``Apr 18``/``Jul 18`` pair becomes ``Overall Apr 18`` and ``Overall Jul
        18``; this asserts the geometry chain derives exactly those strings, so
        the model candidate and the native grid cannot disagree over the
        rendering of a spanning header.
        """
        from pathlib import Path

        import socr

        prompt = (Path(socr.__file__).parent / "prompts" / "table_extract.md").read_text()
        page = _survey_page()
        grid = find_table_blocks(_stacked_markdown())[0].grid
        derived = native_header_row(grid, page.get_text("words"))

        assert derived is not None
        for example in ("Overall Apr 18", "Overall Jul 18"):
            assert f"`{example}`" in prompt
            assert example in derived
