"""Negative tests for table-shaped native row completeness hard-fail."""

from __future__ import annotations

import fitz
import pytest

import test_native_table_verifier as ntv

from socr.tables.native_verifier import (
    COMPLETENESS_MISSING_TABLE_SHAPED_ROWS,
    verify_native_table,
)

_PHYS_COL_GAP = ntv._PHYS_COL_GAP
_make_fitz_page_explicit = ntv._make_fitz_page_explicit
_make_fitz_page_with_words = ntv._make_fitz_page_with_words
_md_table = ntv._md_table

FIXTURE_PDF = (
    pytest.importorskip("pathlib").Path(__file__).parent / "fixtures" / "table_repair" / "ce_like_p4.pdf"
)


class TestCompletenessDoesNotFireOnNoise:
    def test_footnote_marker_row_does_not_hard_fail(self) -> None:
        """A footnote (1) near a correct table must not trip completeness."""
        tokens: list[tuple[float, float, str]] = []
        for row_i, (v1, v2, v3) in enumerate(
            [("2.1", "1.9", "3.4"), ("1.5", "2.3", "2.8"), ("3.2", "2.8", "3.1")]
        ):
            y = 100.0 + row_i * 30
            tokens.extend(
                [
                    (y, 150.0, v1),
                    (y, 150.0 + _PHYS_COL_GAP, v2),
                    (y, 150.0 + 2 * _PHYS_COL_GAP, v3),
                ]
            )
        tokens.append((250.0, 150.0, "(1)"))
        page = _make_fitz_page_explicit(tokens)
        output_text = _md_table(
            ["Forecaster", "2024", "2025", "CPI"],
            [
                ["Ashford", "2.1", "1.9", "3.4"],
                ["Brightwater", "1.5", "2.3", "2.8"],
                ["Clearview", "3.2", "2.8", "3.1"],
            ],
        )
        result = verify_native_table(page, output_text)
        assert result.hard_fail is False or "value_guard_completeness" not in result.reason

    def test_lone_page_number_does_not_hard_fail(self) -> None:
        """A page-number token outside the table must not hard-fail completeness."""
        table_rows = [
            [(100.0 + i * _PHYS_COL_GAP, str(10 + i)) for i in range(3)] for _ in range(4)
        ]
        table_rows.append([(100.0, "22")])
        page = _make_fitz_page_with_words(table_rows)
        output_text = _md_table(
            ["Row", "a", "b", "c"],
            [[f"R{i}", *[str(10 + j) for j in range(3)]] for i in range(4)],
        )
        result = verify_native_table(page, output_text)
        assert "value_guard_completeness" not in (result.reason or "")

    def test_ce_main_table_reading_without_second_table(self) -> None:
        """Reading only the main forecaster grid on CE p4 must not completeness-fail."""
        from socr.core.born_digital import BornDigitalDetector

        doc = fitz.open(str(FIXTURE_PDF))
        page = doc[0]
        structured = BornDigitalDetector().extract_structured(page)
        doc.close()
        tables = [b for b in structured.split("\n\n") if b.strip().startswith("|")]
        assert len(tables) >= 1
        main_table_md = tables[0]
        doc = fitz.open(str(FIXTURE_PDF))
        result = verify_native_table(doc[0], main_table_md)
        doc.close()
        if result.hard_fail:
            assert "value_guard_completeness" not in result.reason, result.reason
        missing_predicate_rows = [
            r for r in result.drifted_rows if r.get("predicate") == "missing_table_shaped_row"
        ]
        assert len(missing_predicate_rows) < COMPLETENESS_MISSING_TABLE_SHAPED_ROWS
