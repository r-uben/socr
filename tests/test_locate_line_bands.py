"""TICKET-C2a: row-band / column-edge / ordinal-origin helpers.

Design: docs/plans/verifier-independence/logs/2026-09-05_C1-design.md, §(a).
No table on the frozen corpus has per-row rules or a single vertical rule
(C1 §0), so the helpers below address printed rows from PDF text-line
geometry inside the witness region, with an ordinal origin from the region's
own horizontal rules — never from the binder's rows or the candidate's
columns.

Hermetic — synthetic ``fitz`` pages built with ``insert_text`` (separate
calls per cell, so PyMuPDF's own line clustering yields one PDF "line" per
cell, exactly the geometry ``label_column_edge`` reasons about) plus real
drawn rules. The frozen-corpus check at the bottom runs only when the
corpus directory exists — CI has no corpus, so it is skipped there, never
faked.
"""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from socr.tables.locate import (
    RowBand,
    band_index_for,
    label_column_edge,
    ordinal_origin,
    row_bands,
    row_bands_from_lines,
    row_bands_from_rules,
)

_REGION_X0, _REGION_X1 = 30.0, 260.0
_LABEL_X, _COL1_X, _COL2_X = 40.0, 150.0, 210.0
_FONTSIZE = 10.0
_ROW_BASELINES = [140.0, 160.0, 180.0]
_ROWS = [
    ("Alpha", "0.12", "0.34"),
    ("Beta", "1.20", "3.40"),
    ("Gamma", "2.10", "4.30"),
]


def _new_page():
    doc = fitz.open()
    return doc, doc.new_page()


def _insert_row(page, baseline: float, label: str, val1: str, val2: str) -> None:
    page.insert_text((_LABEL_X, baseline), label, fontsize=_FONTSIZE)
    page.insert_text((_COL1_X, baseline), val1, fontsize=_FONTSIZE)
    page.insert_text((_COL2_X, baseline), val2, fontsize=_FONTSIZE)


def _draw_rule(page, y: float, width: float = 1.0) -> None:
    page.draw_line((35.0, y), (250.0, y), width=width)


def test_ruled_fixture_one_band_per_row_equal_to_rules():
    """Per-row rules: every printed row gets exactly one band, and
    ``row_bands`` returns exactly ``row_bands_from_rules``'s bands."""
    _, page = _new_page()
    for baseline, (label, v1, v2) in zip(_ROW_BASELINES, _ROWS):
        _insert_row(page, baseline, label, v1, v2)
    rule_ys = [125.0, 146.0, 166.0, 186.0]  # bounds each row's text tightly
    for y in rule_ys:
        _draw_rule(page, y)

    region = (_REGION_X0, 120.0, _REGION_X1, 190.0)
    from socr.tables.locate import _horizontal_rules

    rules = _horizontal_rules(page)
    expected = row_bands_from_rules(rules, region)
    bands = row_bands(page, region)

    assert len(bands) == 3 == len(expected)
    assert bands == expected
    assert all(b.source == "rule" for b in bands)


def test_booktabs_fixture_one_band_per_row_from_lines_and_origin_is_midrule():
    """Booktabs: only border rules exist. ``row_bands`` falls back to
    text-line bands (one per printed row) and ``ordinal_origin`` is the
    second rule group — here, the midrule right after the header."""
    _, page = _new_page()
    # Doubled \toprule: two rules closer than their own drawn thickness.
    _draw_rule(page, 88.0, width=1.0)
    _draw_rule(page, 88.8, width=1.0)
    # \midrule, right after the (unrendered) header.
    _draw_rule(page, 115.0, width=1.0)
    for baseline, (label, v1, v2) in zip(_ROW_BASELINES, _ROWS):
        _insert_row(page, baseline, label, v1, v2)
    # \bottomrule.
    _draw_rule(page, 200.0, width=1.0)

    region = (_REGION_X0, 80.0, _REGION_X1, 205.0)
    bands = row_bands(page, region)
    expected = row_bands_from_lines(page, region)

    assert len(bands) == 3 == len(expected)
    assert bands == expected
    assert all(b.source == "line" for b in bands)

    origin = ordinal_origin(page, region)
    assert origin == pytest.approx(115.0, abs=0.01)


def test_no_rules_ordinal_origin_is_none_not_a_guess():
    _, page = _new_page()
    for baseline, (label, v1, v2) in zip(_ROW_BASELINES, _ROWS):
        _insert_row(page, baseline, label, v1, v2)
    region = (_REGION_X0, 120.0, _REGION_X1, 190.0)

    assert ordinal_origin(page, region) is None
    # No rules at all: row_bands must still address every row from text.
    assert len(row_bands(page, region)) == 3


def test_label_column_edge_finds_the_edge_and_is_none_on_one_column():
    _, page = _new_page()
    for baseline, (label, v1, v2) in zip(_ROW_BASELINES, _ROWS):
        _insert_row(page, baseline, label, v1, v2)
    region = (_REGION_X0, 120.0, _REGION_X1, 190.0)

    edge = label_column_edge(page, region)
    assert edge == pytest.approx(_COL1_X, abs=0.01)

    doc2, page2 = _new_page()
    for baseline, (label, _v1, _v2) in zip(_ROW_BASELINES, _ROWS):
        page2.insert_text((_LABEL_X, baseline), label, fontsize=_FONTSIZE)
    region2 = (_REGION_X0, 120.0, _REGION_X1, 190.0)
    assert label_column_edge(page2, region2) is None
    doc2.close()


def test_band_index_for():
    bands = [RowBand(0.0, 10.0, "line"), RowBand(10.0, 20.0, "line"), RowBand(20.0, 30.0, "line")]
    assert band_index_for(bands, 5.0) == 0
    assert band_index_for(bands, 25.0) == 2
    assert band_index_for(bands, 100.0) is None


# ---------------------------------------------------------------------------
# Frozen-corpus check (C1's measured origins). Skipped in CI — no corpus.
# ---------------------------------------------------------------------------

_CORPUS_DIR = Path.home() / "Data" / "socr" / "ladder-run2-2026-09-04"

# doc01 116.3, doc02 p3/p4 123.9, doc03 241.5, doc05/doc07 121.0, doc04 None
# (C1 §(a), measured on the frozen corpus with the shipped locate.py/binding.py).
_EXPECTED_ORIGIN_BY_SLUG = {
    "doc01": 116.3,
    "doc02": 123.9,
    "doc03": 241.5,
    "doc04": None,
    "doc05": 121.0,
    "doc07": 121.0,
}


@pytest.mark.skipif(not _CORPUS_DIR.exists(), reason="frozen ladder corpus not present")
def test_corpus_origins_match_c1_measurement():
    from socr.benchmark.replay_binding import _select_candidate_for_table, discover_pages
    from socr.core.pdf import open_pdf
    from socr.tables.witness import WitnessStatus, prepare_table_witnesses

    seen = {}
    for record in discover_pages(_CORPUS_DIR):
        for table_id in sorted(record.binding_adjudication):
            candidate_markdown, _note = _select_candidate_for_table(record, table_id)
            if candidate_markdown is None:
                continue
            with open_pdf(record.pdf_path) as doc:
                page = doc[record.page_num - 1]
                with prepare_table_witnesses(
                    record.pdf_path, record.page_num, candidate_markdown
                ) as witnesses:
                    witness = next((w for w in witnesses if w.table_id == table_id), None)
                    if witness is None or witness.status is not WitnessStatus.LOCATED:
                        continue
                    if witness.box is None:
                        continue
                    region = witness.box.bbox
                    origin = ordinal_origin(page, region)
                    bands = row_bands(page, region)
            seen[(record.doc_slug, record.page_num, table_id)] = (origin, len(bands))

    assert len(seen) == 7, f"expected 7 adjudicated tables, found {sorted(seen)}"
    for (slug, page_num, table_id), (origin, band_count) in sorted(seen.items()):
        expected = _EXPECTED_ORIGIN_BY_SLUG[slug]
        if expected is None:
            assert origin is None, (
                f"{slug} p{page_num} {table_id}: expected no origin (no vector rules), got {origin}"
            )
        else:
            assert origin is not None, (
                f"{slug} p{page_num} {table_id}: expected origin {expected}, got None"
            )
            assert round(origin, 1) == pytest.approx(expected, abs=0.05), (
                f"{slug} p{page_num} {table_id}: origin {origin} != C1's {expected}"
            )
        assert band_count > 0, f"{slug} p{page_num} {table_id}: no row bands addressed at all"
