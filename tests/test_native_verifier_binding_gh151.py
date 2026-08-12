"""Tests for the GH-151 A2 column-binding check (BindingReport).

Self-contained: does not import from ``tests/test_native_table_verifier.py``.
Additive, model-free, report-only — nothing here drives ``_phase_agentic``
or ``process()``, so no ``_available_engines_for_agentic`` patch is needed.

Plan reference: c87af1a:docs/plans/gh151-structural-gate/TICKETS.md
(TICKET-A2).
"""

from __future__ import annotations

import fitz

from socr.tables.native_verifier import (
    BindingReport,
    verify_column_binding,
    verify_column_binding_region,
)

# Physical column gap used in synthetic fitz pages.
#
# Must be large enough that PyMuPDF inserts separate word tokens in
# get_text("words"). Empirically, insert_text calls at the same y are merged
# into one "word" run unless the x-distance exceeds the rendered width of the
# preceding token. For 9pt font, "0.043" is ~45pt wide, so a gap >= 60pt
# reliably creates separate tokens. Also comfortably exceeds the verifier's
# _LANE_X_TOL_PT (6pt) lane-clustering tolerance.
_PHYS_COL_GAP: float = 60.0


def _make_fitz_page_with_words(rows: list[list[tuple[float, str]]]) -> fitz.Page:
    """Build a fitz page where each row is a list of (x, word) pairs.

    Words are placed at y = 100 + row_index * 30.
    """
    doc = fitz.open()
    page = doc.new_page(width=700, height=900)
    for row_idx, cells in enumerate(rows):
        y = 100.0 + row_idx * 30
        for x, word in cells:
            page.insert_text((x, y), word, fontsize=9)
    return page


def _make_empty_page() -> fitz.Page:
    """A fitz page with NO inserted text (simulates a scanned page)."""
    doc = fitz.open()
    return doc.new_page(width=500, height=700)


def _md_table(header: list[str], rows: list[list[str]]) -> str:
    """Build a minimal GitHub-markdown table string."""
    sep = "| " + " | ".join(["---"] * len(header)) + " |"
    lines = [
        "| " + " | ".join(header) + " |",
        sep,
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


class TestColumnBindingClean:
    def test_case1_clean_three_lanes_three_rows(self):
        """3 rows x 3 well-separated native lanes, page-unique values,
        markdown cells in native order -> clean report."""
        native_rows = [
            [
                (100.0, "1.1"),
                (100.0 + _PHYS_COL_GAP, "1.2"),
                (100.0 + 2 * _PHYS_COL_GAP, "1.3"),
            ],
            [
                (100.0, "2.1"),
                (100.0 + _PHYS_COL_GAP, "2.2"),
                (100.0 + 2 * _PHYS_COL_GAP, "2.3"),
            ],
            [
                (100.0, "3.1"),
                (100.0 + _PHYS_COL_GAP, "3.2"),
                (100.0 + 2 * _PHYS_COL_GAP, "3.3"),
            ],
        ]
        page = _make_fitz_page_with_words(native_rows)
        output_text = _md_table(
            ["label", "c1", "c2", "c3"],
            [
                ["A", "1.1", "1.2", "1.3"],
                ["B", "2.1", "2.2", "2.3"],
                ["C", "3.1", "3.2", "3.3"],
            ],
        )
        report = verify_column_binding(page, output_text)
        assert isinstance(report, BindingReport)
        assert report.failed is False
        assert report.misbound == []
        assert report.checked > 0
        assert report.certain > 0
        assert report.unresolved == 0
        assert report.lane_shared_across_columns is False
        assert report.column_order_violation is False


class TestColumnBindingMisbound:
    def test_case2_shifted_page_unique_value(self):
        """Native geometry identical to case 1; one row's c2/c3 values are
        swapped in the markdown only (native x untouched) -> misbound."""
        native_rows = [
            [
                (100.0, "1.1"),
                (100.0 + _PHYS_COL_GAP, "1.2"),
                (100.0 + 2 * _PHYS_COL_GAP, "1.3"),
            ],
            [
                (100.0, "2.1"),
                (100.0 + _PHYS_COL_GAP, "2.2"),
                (100.0 + 2 * _PHYS_COL_GAP, "2.3"),
            ],
            [
                (100.0, "3.1"),
                (100.0 + _PHYS_COL_GAP, "3.2"),
                (100.0 + 2 * _PHYS_COL_GAP, "3.3"),
            ],
        ]
        page = _make_fitz_page_with_words(native_rows)
        output_text = _md_table(
            ["label", "c1", "c2", "c3"],
            [
                ["A", "1.1", "1.2", "1.3"],
                ["B", "2.1", "2.3", "2.2"],  # c2/c3 swapped, markdown only
                ["C", "3.1", "3.2", "3.3"],
            ],
        )
        report = verify_column_binding(page, output_text)
        assert report.failed is True
        assert report.misbound != []
        row1_entries = [m for m in report.misbound if m["row_idx"] == 1]
        assert row1_entries, f"expected a misbound entry for row 1, got {report.misbound}"
        for entry in row1_entries:
            assert entry["predicate"] in ("modal_disagreement", "ordinal_mismatch")
            assert "output_column" in entry
            assert "expected_signature" in entry
            assert "observed_signature" in entry or "observed_lane" in entry

    def test_case4_uniform_column_swap(self):
        """V == L single-lane grid; two value columns swapped in EVERY row
        (>= 3 rows). Modal signature IS the swap (no modal_disagreement) so
        the ordinal predicate must fire instead."""
        native_rows = [
            [
                (100.0, "1.1"),
                (100.0 + _PHYS_COL_GAP, "1.2"),
                (100.0 + 2 * _PHYS_COL_GAP, "1.3"),
            ],
            [
                (100.0, "2.1"),
                (100.0 + _PHYS_COL_GAP, "2.2"),
                (100.0 + 2 * _PHYS_COL_GAP, "2.3"),
            ],
            [
                (100.0, "3.1"),
                (100.0 + _PHYS_COL_GAP, "3.2"),
                (100.0 + 2 * _PHYS_COL_GAP, "3.3"),
            ],
        ]
        page = _make_fitz_page_with_words(native_rows)
        # c2 and c3 swapped in every row, consistently.
        output_text = _md_table(
            ["label", "c1", "c2", "c3"],
            [
                ["R0", "1.1", "1.3", "1.2"],
                ["R1", "2.1", "2.3", "2.2"],
                ["R2", "3.1", "3.3", "3.2"],
            ],
        )
        report = verify_column_binding(page, output_text)
        assert report.failed is True
        assert report.misbound != []
        ordinal_entries = [m for m in report.misbound if m["predicate"] == "ordinal_mismatch"]
        assert ordinal_entries, f"expected ordinal_mismatch entries, got {report.misbound}"
        modal_entries = [m for m in report.misbound if m["predicate"] == "modal_disagreement"]
        assert modal_entries == [], (
            "swap is uniform across all rows -> every row agrees with its column's "
            f"own modal signature, so modal_disagreement must not fire; got {modal_entries}"
        )
        assert report.column_order_violation is True or report.lane_shared_across_columns is True

    def test_case8_single_row_ordinal_acceptance(self):
        """Exactly one data row, V == L, one value in the wrong markdown
        column, page-unique values -> ordinal_mismatch fires despite no
        cross-row modal support (ruling amendment 1)."""
        native_rows = [
            [
                (100.0, "5.1"),
                (100.0 + _PHYS_COL_GAP, "5.2"),
                (100.0 + 2 * _PHYS_COL_GAP, "5.3"),
            ],
        ]
        page = _make_fitz_page_with_words(native_rows)
        output_text = _md_table(
            ["label", "c1", "c2", "c3"],
            [["Z", "5.1", "5.3", "5.2"]],  # c2/c3 swapped
        )
        report = verify_column_binding(page, output_text)
        assert report.failed is True
        ordinal_entries = [m for m in report.misbound if m["predicate"] == "ordinal_mismatch"]
        assert ordinal_entries, f"expected ordinal_mismatch entries, got {report.misbound}"


class TestColumnBindingPairedYearGuard:
    def test_case3_paired_year_no_ordinal_misfire(self):
        """Native has more x-lanes than markdown value cells (two year values
        pack into one cell). V < L is structurally exempt from the ordinal
        predicate -> clean report, no ordinal_mismatch entries."""
        native_rows = [
            [(100.0, "1.2"), (160.0, "1.3"), (220.0, "2.5")],
            [(100.0, "2.2"), (160.0, "2.3"), (220.0, "3.5")],
            [(100.0, "4.2"), (160.0, "4.3"), (220.0, "5.5")],
        ]
        page = _make_fitz_page_with_words(native_rows)
        output_text = _md_table(
            ["label", "years", "total"],
            [
                ["A", "1.2 1.3", "2.5"],
                ["B", "2.2 2.3", "3.5"],
                ["C", "4.2 4.3", "5.5"],
            ],
        )
        report = verify_column_binding(page, output_text)
        assert report.failed is False
        assert report.misbound == []
        ordinal_entries = [m for m in report.misbound if m["predicate"] == "ordinal_mismatch"]
        assert ordinal_entries == [], (
            "V (2) < L (3) must structurally prevent the ordinal predicate from firing"
        )


class TestColumnBindingAmbiguity:
    def test_case5_repeated_value_goes_to_unresolved_not_misbound(self):
        """The same N2 value occurs twice in one paired native row; the
        output places one occurrence -> unresolved, never misbound (no
        greedy leftmost match)."""
        native_rows = [
            [(100.0, "1.2"), (100.0 + _PHYS_COL_GAP, "1.2"), (100.0 + 2 * _PHYS_COL_GAP, "3.3")],
        ]
        page = _make_fitz_page_with_words(native_rows)
        output_text = _md_table(
            ["label", "c1", "c2", "c3"],
            [["A", "1.2", "", "3.3"]],
        )
        report = verify_column_binding(page, output_text)
        assert report.failed is False
        assert report.misbound == []
        assert report.unresolved > 0
        assert report.certain > 0


class TestColumnBindingBypass:
    def test_case6_scan_page_bypass(self):
        page = _make_empty_page()
        output_text = _md_table(
            ["label", "c1", "c2"],
            [["A", "1.1", "1.2"]],
        )
        report = verify_column_binding(page, output_text)
        assert report.checked == 0
        assert report.certain == 0
        assert report.unresolved == 0
        assert report.misbound == []
        assert report.failed is False

    def test_case7_no_table_output_bypass(self):
        """Real words page but prose output with no markdown table
        -> bypass, distinguishable from a clean report via checked == 0."""
        native_rows = [[(100.0, "1.2"), (100.0 + _PHYS_COL_GAP, "2.3")]]
        page = _make_fitz_page_with_words(native_rows)
        output_text = "This page discusses the forecast in prose, no table here."
        report = verify_column_binding(page, output_text)
        assert report.checked == 0
        assert report.misbound == []
        assert report.failed is False


class TestColumnBindingRegion:
    def test_case9_region_matches_page_scope_on_clean_fixture(self):
        native_rows = [
            [
                (100.0, "1.1"),
                (100.0 + _PHYS_COL_GAP, "1.2"),
                (100.0 + 2 * _PHYS_COL_GAP, "1.3"),
            ],
            [
                (100.0, "2.1"),
                (100.0 + _PHYS_COL_GAP, "2.2"),
                (100.0 + 2 * _PHYS_COL_GAP, "2.3"),
            ],
            [
                (100.0, "3.1"),
                (100.0 + _PHYS_COL_GAP, "3.2"),
                (100.0 + 2 * _PHYS_COL_GAP, "3.3"),
            ],
        ]
        page = _make_fitz_page_with_words(native_rows)
        output_text = _md_table(
            ["label", "c1", "c2", "c3"],
            [
                ["A", "1.1", "1.2", "1.3"],
                ["B", "2.1", "2.2", "2.3"],
                ["C", "3.1", "3.2", "3.3"],
            ],
        )
        page_report = verify_column_binding(page, output_text)

        covering_bbox = fitz.Rect(0, 90, 700, 200)
        region_report = verify_column_binding_region(page, output_text, covering_bbox)
        assert region_report.failed == page_report.failed
        assert region_report.checked == page_report.checked
        assert region_report.certain == page_report.certain
        assert region_report.unresolved == page_report.unresolved
        assert region_report.misbound == page_report.misbound

        disjoint_bbox = fitz.Rect(0, 800, 700, 890)
        empty_report = verify_column_binding_region(page, output_text, disjoint_bbox)
        assert empty_report.checked == 0
        assert empty_report.misbound == []
        assert empty_report.failed is False
