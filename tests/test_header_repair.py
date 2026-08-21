"""Tests for deterministic collapsed-header repair (GH-56).

Hermetic: uses synthetic fitz pages + markdown grids; no ollama/GPU.
"""

from __future__ import annotations

import fitz

from socr.tables.header_repair import (
    detect_header_column_collapse,
    repair_collapsed_header,
    repair_table_headers_in_text,
    repair_table_headers_on_page,
)
from socr.tables.native_verifier import verify_native_table_region

# Gap between data column x-positions (must exceed rendered token width).
_COL_GAP = 55.0

_LABEL_X = 40.0
_DATA_XS = [_LABEL_X + 110.0 + i * _COL_GAP for i in range(7)]
_NARROW_HEADER_DATA_XS = [150.0, 215.0, 280.0, 345.0]

# Geometry derived from CE 202401.pdf page 3 exchange-rate table (coordinates
# only — invented labels/values for license cleanliness).
_CE_EXCHANGE_DATA_XS = [160.1, 197.3, 232.3, 270.0, 307.2, 346.8, 384.0]
_CE_EXCHANGE_HDR1 = [
    ("-23%", 153.1),
    ("-14%", 190.1),
    ("-5%", 229.9),
    ("+5%", 303.4),
    ("+14%", 338.6),
    ("+23%", 374.9),
]
_CE_EXCHANGE_HDR2 = [
    ("or", 148.6),
    ("more", 158.2),
    ("to", 185.0),
    ("-22%", 194.6),
    ("to", 222.7),
    ("-13%", 232.3),
    ("+/-4%", 263.0),
    ("to", 296.4),
    ("+13%", 306.0),
    ("to", 333.4),
    ("+22%", 343.0),
    ("or", 371.8),
    ("more", 381.4),
]


def _md_table(header: list[str], rows: list[list[str]]) -> str:
    sep = "| " + " | ".join(["---"] * len(header)) + " |"
    lines = ["| " + " | ".join(header) + " |", sep]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _make_probability_table_page() -> fitz.Page:
    """Synthetic CE-style probability-bin table (license-clean)."""
    doc = fitz.open()
    page = doc.new_page(width=700, height=400)

    # Two-line multi-band header (7 data columns).
    hdr1 = ["-23%", "-14%", "-5%", "+5%", "+14%", "+23%", "+30%"]
    hdr2 = ["or more", "to", "to", "+/-4%", "to", "to", "or more"]
    for x, tok in zip(_DATA_XS, hdr1):
        page.insert_text((x, 100), tok, fontsize=9)
    for x, tok in zip(_DATA_XS, hdr2):
        page.insert_text((x, 118), tok, fontsize=9)

    page.insert_text((_LABEL_X, 140), "Euro1", fontsize=9)
    for x, val in zip(_DATA_XS, ["1", "2", "16", "49", "23", "8", "1"]):
        page.insert_text((x, 140), val, fontsize=9)

    page.insert_text((_LABEL_X, 170), "Yen1", fontsize=9)
    for x, val in zip(_DATA_XS, ["0", "3", "15", "40", "29", "9", "4"]):
        page.insert_text((x, 170), val, fontsize=9)

    return page


def _make_exchange_rate_table_page() -> fitz.Page:
    """Three-line split-range header matching CE 202401 p.3 geometry."""
    doc = fitz.open()
    page = doc.new_page(width=700, height=400)

    # Top band: percentages only (middle +/-4% bin absent on this line).
    for tok, x in _CE_EXCHANGE_HDR1:
        page.insert_text((x, 120), tok, fontsize=9)

    # Prose subtitle between header bands and data (not lane-aligned).
    page.insert_text(
        (200.0, 110.0),
        "(between survey date and end-Jan. 2025)",
        fontsize=8,
    )

    # Continuation band: qualifiers at offset x-positions vs line 1.
    for tok, x in _CE_EXCHANGE_HDR2:
        page.insert_text((x, 131), tok, fontsize=9)

    page.insert_text((_LABEL_X, 156), "Euro1", fontsize=9)
    for x, val in zip(_CE_EXCHANGE_DATA_XS, ["1", "2", "16", "49", "23", "8", "1"]):
        page.insert_text((x, 158), val, fontsize=9)

    page.insert_text((_LABEL_X, 186), "Yen1", fontsize=9)
    for x, val in zip(_CE_EXCHANGE_DATA_XS, ["0", "3", "15", "40", "29", "9", "4"]):
        page.insert_text((x, 188), val, fontsize=9)

    return page


def _make_narrow_spanning_header_page(far_label_x: float = 315.0) -> fitz.Page:
    """Synthetic regression table with a right-edge secondary header label."""
    doc = fitz.open()
    page = doc.new_page(width=600, height=300)
    page.insert_text((150.0, 80.0), "Near outcome", fontsize=9)
    page.insert_text((far_label_x, 80.0), "Far outcome", fontsize=9)

    for x, ordinal in zip(_NARROW_HEADER_DATA_XS, ["(1)", "(2)", "(3)", "(4)"]):
        page.insert_text((x, 110.0), ordinal, fontsize=9)
    for x, value in zip(_NARROW_HEADER_DATA_XS, ["-4.8", None, "-4.1", "-0.2"]):
        if value is not None:
            page.insert_text((x, 140.0), value, fontsize=9)
    for x, value in zip(_NARROW_HEADER_DATA_XS, ["0.1", "0.2", "0.3", "0.4"]):
        page.insert_text((x, 170.0), value, fontsize=9)
    return page


def _collapsed_exchange_rate_md() -> str:
    return _md_table(
        ["Currency", "-23% to -14% to -5% to +5% to +14% to +23% or more", "Appreciation"],
        [
            ["(between survey date and end-Jan. 2025)", "", ""],
            ["-23% to -14% to -5% +5% +14% +23%", "", ""],
            ["or more to -22% to -13% +/-4% to +13% to +22% or more", "", ""],
            ["Euro1", "1", "2", "16", "49", "23", "8", "1"],
            ["Yen1", "0", "3", "15", "40", "29", "9", "4"],
        ],
    )


class TestDetectHeaderColumnCollapse:
    def test_detects_collapsed_header(self):
        grid = [
            ["Currency", "-23% to -14% ... +23% or more", "Appreciation"],
            ["Euro1", "1", "2", "16", "49", "23", "8", "1"],
        ]
        collapsed, hdr_cols, expected = detect_header_column_collapse(grid)
        assert collapsed is True
        assert hdr_cols == 3
        assert expected == 8

    def test_no_false_positive_on_aligned_table(self):
        grid = [
            ["", "A", "B", "C"],
            ["Row1", "1", "2", "3"],
            ["Row2", "4", "5", "6"],
        ]
        collapsed, _, _ = detect_header_column_collapse(grid)
        assert collapsed is False


class TestRepairCollapsedHeader:
    def test_repair_expands_header_to_match_data_columns(self):
        page = _make_probability_table_page()
        words = page.get_text("words")

        collapsed_md = _md_table(
            ["Currency", "-23% to -14% to -5% to +5% to +14% to +23% or more", "Side"],
            [
                ["Euro1", "1", "2", "16", "49", "23", "8", "1"],
                ["Yen1", "0", "3", "15", "40", "29", "9", "4"],
            ],
        )
        from socr.tables.reconcile import find_table_blocks

        grid = find_table_blocks(collapsed_md)[0].grid
        assert detect_header_column_collapse(grid)[0] is True

        repaired = repair_collapsed_header(grid, words)
        assert repaired is not None
        assert len(repaired[0]) == 8, f"header still collapsed: {repaired[0]}"
        # Each probability bin should land in its own column (merged multi-line header).
        assert "-23% or more" in repaired[0][1]
        assert "+/-4%" in repaired[0][4]
        assert "+23%" in repaired[0][6] or "+30%" in repaired[0][6]
        # Data values unchanged.
        assert repaired[1] == ["Euro1", "1", "2", "16", "49", "23", "8", "1"]

    def test_three_line_split_range_header_real_geometry(self):
        page = _make_exchange_rate_table_page()
        words = page.get_text("words")
        from socr.tables.reconcile import find_table_blocks

        grid = find_table_blocks(_collapsed_exchange_rate_md())[0].grid
        repaired = repair_collapsed_header(grid, words)
        assert repaired is not None
        header = repaired[0]
        assert len(header) == 8
        assert header[1] == "-23% or more"
        assert header[2] == "-14% to -22%"
        assert header[3] == "-5% to -13%"
        assert header[4] == "+/-4%"
        assert header[5] == "+5% to +13%"
        assert header[6] == "+14% to +22%"
        assert header[7] == "+23% or more"
        assert len(repaired) == 3
        assert repaired[1][0] == "Euro1"

    def test_strips_junk_header_rows_from_body(self):
        page = _make_exchange_rate_table_page()
        from socr.tables.reconcile import find_table_blocks

        grid = find_table_blocks(_collapsed_exchange_rate_md())[0].grid
        repaired = repair_collapsed_header(grid, page.get_text("words"))
        assert repaired is not None
        body_col0 = [row[0] for row in repaired[1:]]
        assert "(between survey date" not in " ".join(body_col0)
        assert "or more to -22%" not in " ".join(body_col0)

    def test_declines_when_middle_header_lane_unrecoverable(self):
        """Faithfulness guard: do not emit blank-filled headers."""
        doc = fitz.open()
        page = doc.new_page(width=700, height=400)
        # Top band only — lane 4 (+/-4%) exists only on the continuation line.
        for tok, x in _CE_EXCHANGE_HDR1:
            page.insert_text((x, 120), tok, fontsize=9)
        page.insert_text((_LABEL_X, 156), "Euro1", fontsize=9)
        for x, val in zip(_CE_EXCHANGE_DATA_XS, ["1", "2", "16", "49", "23", "8", "1"]):
            page.insert_text((x, 158), val, fontsize=9)

        from socr.tables.reconcile import find_table_blocks

        grid = find_table_blocks(_collapsed_exchange_rate_md())[0].grid
        assert repair_collapsed_header(grid, page.get_text("words")) is None

    def test_repair_improves_per_region_verifier(self):
        page = _make_probability_table_page()
        region = fitz.Rect(0, 80, 700, 200)

        collapsed_md = _md_table(
            ["Currency", "-23% to -14% to -5% to +5% to +14% to +23% or more", "Side"],
            [["Euro1", "1", "2", "16", "49", "23", "8", "1"]],
        )
        before = verify_native_table_region(page, collapsed_md, region)
        assert before.output_col_count == 3

        repaired_md, count = repair_table_headers_on_page(page, collapsed_md)
        assert count == 1
        after = verify_native_table_region(page, repaired_md, region)
        assert after.output_col_count == 8
        assert after.output_col_count > before.output_col_count

    def test_idempotent_on_already_correct_table(self):
        page = _make_probability_table_page()
        good_md = _md_table(
            ["", "A", "B", "C", "D", "E", "F", "G"],
            [["Euro1", "1", "2", "16", "49", "23", "8", "1"]],
        )
        out, count = repair_table_headers_on_page(page, good_md)
        assert count == 0
        assert out == good_md


class TestRepairTooNarrowSpanningHeader:
    def test_repairs_width_and_right_edge_header_binding_from_geometry(self):
        malformed = _md_table(
            ["", "Dependent variable:", "", ""],
            [
                ["", "Near outcome", "", "Far outcome"],
                ["", "(1)", "(2)", "(3)", "(4)"],
                ["Signal", "-4.8", "", "-4.1", "-0.2"],
                ["Control", "0.1", "0.2", "0.3", "0.4"],
            ],
        )

        page = _make_narrow_spanning_header_page()
        repaired_md, count = repair_table_headers_on_page(page, malformed)

        from socr.tables.reconcile import find_table_blocks

        repaired = find_table_blocks(repaired_md)[0].grid
        assert count == 1
        assert {len(row) for row in repaired} == {5}
        assert repaired[1] == [
            "",
            "Near outcome",
            "",
            "",
            "Far outcome",
        ]
        assert repaired[3] == ["Signal", "-4.8", "", "-4.1", "-0.2"]

        second_pass, second_count = repair_table_headers_on_page(page, repaired_md)
        assert second_count == 0
        assert second_pass == repaired_md

    def test_geometry_can_keep_final_label_in_place_and_append_blank(self):
        malformed = _md_table(
            ["", "Dependent variable:", "", ""],
            [
                ["", "Near outcome", "", "Far outcome"],
                ["", "(1)", "(2)", "(3)", "(4)"],
                ["Signal", "-4.8", "", "-4.1", "-0.2"],
                ["Control", "0.1", "0.2", "0.3", "0.4"],
            ],
        )
        page = _make_narrow_spanning_header_page(far_label_x=250.0)

        repaired_md, count = repair_table_headers_on_page(page, malformed)

        from socr.tables.reconcile import find_table_blocks

        repaired = find_table_blocks(repaired_md)[0].grid
        assert count == 1
        assert repaired[1] == ["", "Near outcome", "", "Far outcome", ""]

    def test_abstains_when_right_edge_binding_has_no_geometry(self):
        malformed = _md_table(
            ["", "Dependent variable:", "", ""],
            [
                ["", "Near outcome", "", "Far outcome"],
                ["", "(1)", "(2)", "(3)", "(4)"],
                ["Signal", "-4.8", "", "-4.1", "-0.2"],
                ["Control", "0.1", "0.2", "0.3", "0.4"],
            ],
        )

        repaired_md, count = repair_table_headers_in_text([], malformed)

        assert count == 0
        assert repaired_md == malformed

    def test_abstains_for_a_trailing_blank_secondary_band(self):
        malformed = _md_table(
            ["", "Dependent variable:", "", ""],
            [
                ["", "Near outcome", "Far outcome", ""],
                ["", "(1)", "(2)", "(3)", "(4)"],
                ["Signal", "-4.8", "", "-4.1", "-0.2"],
                ["Control", "0.1", "0.2", "0.3", "0.4"],
            ],
        )

        repaired_md, count = repair_table_headers_on_page(
            _make_narrow_spanning_header_page(), malformed
        )

        assert count == 0
        assert repaired_md == malformed

    def test_does_not_widen_for_one_anomalously_wide_body_row(self):
        malformed = _md_table(
            ["Measure", "A", "B", "C"],
            [
                ["First", "1", "2", "3"],
                ["Second", "4", "5", "6"],
                ["Outlier", "7", "8", "9", "10"],
            ],
        )

        repaired_md, count = repair_table_headers_in_text([], malformed)

        assert count == 0
        assert repaired_md == malformed

    def test_does_not_reclassify_a_short_numeric_body_row_as_header(self):
        malformed = _md_table(
            ["Measure", "A", "B", "C"],
            [
                ["First", "1", "2", ""],
                ["Second", "3", "4", "5", "6"],
                ["Third", "7", "8", "9", "10"],
                ["Fourth", "11", "12", "13", "14"],
            ],
        )

        repaired_md, count = repair_table_headers_in_text([], malformed)

        assert count == 0
        assert repaired_md == malformed

    def test_abstains_when_the_wider_suffix_is_still_ragged(self):
        malformed = _md_table(
            ["", "Dependent variable:", "", ""],
            [
                ["", "Near outcome", "", "Far outcome"],
                ["", "(1)", "(2)", "(3)", "(4)"],
                ["First", "1", "2", "3", "4"],
                ["Second", "5", "6", "7", "8"],
                ["Ragged", "9", "10", "11", "12", "13"],
            ],
        )

        repaired_md, count = repair_table_headers_in_text([], malformed)

        assert count == 0
        assert repaired_md == malformed
