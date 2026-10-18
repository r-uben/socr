"""RESF PLF 2025 fixtures: default-on native per-cell binding check."""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from socr.tables.locate import locate_tables
from socr.tables.native_cell_check import (
    NativeCellOutcome,
    check_table_cell_binding,
)
from socr.tables.native_verifier import (
    _canonical_binding_numeric,
    _normalize_binding_numeric_token,
    binding_numeric_tokens_equivalent,
)
from socr.core.born_digital import flatten_page_spans

FIXTURES = Path(__file__).parent / "fixtures" / "resf_native_cell"


@pytest.fixture(scope="module")
def p59_table():
    doc = fitz.open(FIXTURES / "resf_p59.pdf")
    page = doc[0]
    boxes = locate_tables(page)
    region = fitz.Rect(boxes[1].bbox)
    md = (FIXTURES / "p59_tableau1.md").read_text(encoding="utf-8")
    words = page.get_text("words")
    spans = flatten_page_spans(page)
    doc.close()
    return words, md, region, spans


@pytest.fixture(scope="module")
def p67_table():
    doc = fitz.open(FIXTURES / "resf_p67.pdf")
    page = doc[0]
    boxes = locate_tables(page)
    region = fitz.Rect(boxes[0].bbox)
    md = (FIXTURES / "p67_tableau8.md").read_text(encoding="utf-8")
    words = page.get_text("words")
    spans = flatten_page_spans(page)
    doc.close()
    return words, md, region, spans


def test_locale_decimal_comma_not_thousands():
    assert _normalize_binding_numeric_token("1,5") == "1.5"
    assert _normalize_binding_numeric_token("15") == "15"
    assert _normalize_binding_numeric_token("1,5") != _normalize_binding_numeric_token("15")
    assert _normalize_binding_numeric_token("-5,5") == "-5.5"
    assert _normalize_binding_numeric_token("1 234,5") == "1234.5"
    assert _canonical_binding_numeric("1,204")[1] is True
    assert binding_numeric_tokens_equivalent("1,204", "1204") is None
    assert binding_numeric_tokens_equivalent("45%", "45") is False
    assert _normalize_binding_numeric_token("(0,4)") == "-0.4"
    assert _normalize_binding_numeric_token("1,5 M€") == "1.5"


class TestP59Tableau1:
    def test_correct_table_verified(self, p59_table):
        words, md, region, spans = p59_table
        r = check_table_cell_binding(
            table_id="t0", words=words, markdown=md, region_bbox=region, spans=spans
        )
        assert r.outcome is NativeCellOutcome.VERIFIED

    @pytest.mark.parametrize(
        "mutator,label",
        [
            (
                lambda m: m.replace("Solde public | -5,5 | -6,1", "Solde public | -6,1 | -5,5"),
                "swapped_year_columns",
            ),
            (
                lambda m: m.replace("Solde public | -5,5 |", "Solde public | 15 |"),
                "wrong_value_15_vs_-5,5",
            ),
            (lambda m: m.replace("Solde public | -5,5 |", "Solde public | 5,5 |"), "flipped_sign"),
            (
                lambda m: m.replace(
                    "Organismes divers d'administration centrale",
                    "Administrations publiques locales",
                ).replace(
                    "Administrations publiques locales | -0,4",
                    "Organismes divers d'administration centrale | -0,4",
                    1,
                ),
                "swapped_row_labels",
            ),
        ],
    )
    def test_mutations_contradicted(self, p59_table, mutator, label):
        words, md, region, spans = p59_table
        r = check_table_cell_binding(
            table_id="t0",
            words=words,
            markdown=mutator(md),
            region_bbox=region,
            spans=spans,
        )
        assert r.outcome is NativeCellOutcome.CONTRADICTED, label


def _p67_swap_row_labels(md: str) -> str:
    lines = md.splitlines()
    lines[2], lines[4] = lines[4], lines[2]
    return "\n".join(lines)


def _p67_remove_gaz_row(md: str) -> str:
    lines = md.splitlines()
    del lines[2]
    return "\n".join(lines)


def _p67_blank_one_cell(md: str) -> str:
    return md.replace("| 0,4 | 4,5 | 2,0 | 0,5 | - |", "| 0,4 | | 2,0 | 0,5 | - |", 1)


class TestP67Tableau8:
    def test_correct_table_verified(self, p67_table):
        words, md, region, spans = p67_table
        r = check_table_cell_binding(
            table_id="t0", words=words, markdown=md, region_bbox=region, spans=spans
        )
        assert r.outcome is NativeCellOutcome.VERIFIED

    def test_removed_data_row_not_verified(self, p67_table) -> None:
        """Lane-mismatch table: dropping a native data row must not verify."""
        words, md, region, spans = p67_table
        r = check_table_cell_binding(
            table_id="t0",
            words=words,
            markdown=_p67_remove_gaz_row(md),
            region_bbox=region,
            spans=spans,
        )
        assert r.outcome is not NativeCellOutcome.VERIFIED

    def test_blanked_cell_not_verified(self, p67_table) -> None:
        words, md, region, spans = p67_table
        r = check_table_cell_binding(
            table_id="t0",
            words=words,
            markdown=_p67_blank_one_cell(md),
            region_bbox=region,
            spans=spans,
        )
        assert r.outcome is not NativeCellOutcome.VERIFIED

    @pytest.mark.parametrize(
        "mutator,label",
        [
            (
                lambda m: m.replace("| 2022 | 2023 |", "| 2023 | 2022 |").replace(
                    "| 0,4 | 4,5 | 2,0 |", "| 0,4 | 2,0 | 4,5 |", 1
                ),
                "swapped_year_columns",
            ),
            (
                lambda m: m.replace("| 0,4 | 4,5 | 2,0 |", "| 15 | 4,5 | 2,0 |", 1),
                "wrong_value_15_vs_0,4",
            ),
            (
                lambda m: m.replace("| 0,4 | 4,5 | 2,0 |", "| -0,4 | 4,5 | 2,0 |", 1),
                "flipped_sign",
            ),
            (_p67_swap_row_labels, "swapped_row_labels"),
            (
                lambda m: m.replace("| 0,4 | 4,5 | 2,0 |", "| 4,5 | 0,4 | 2,0 |", 1),
                "swapped_numeric_columns",
            ),
        ],
    )
    def test_mutations_contradicted(self, p67_table, mutator, label):
        words, md, region, spans = p67_table
        r = check_table_cell_binding(
            table_id="t0",
            words=words,
            markdown=mutator(md),
            region_bbox=region,
            spans=spans,
        )
        assert r.outcome is NativeCellOutcome.CONTRADICTED, label
