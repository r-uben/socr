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
    _group_lines_by_baseline,
    _group_lines_by_baseline_with_reason,
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
    # Shared edge is in two bands — abstain, do not pick the first.
    assert band_index_for(bands, 10.0) is None


def test_band_index_for_overlapping_bands_is_none():
    bands = [RowBand(0.0, 15.0, "line"), RowBand(10.0, 25.0, "line")]
    assert band_index_for(bands, 12.0) is None
    assert band_index_for(bands, 5.0) == 0
    assert band_index_for(bands, 20.0) == 1


def test_ordinal_origin_merges_doubled_rule_on_small_type():
    """6 pt table, 4 pt doubled-rule gap: the pair is one border.

    Half the body font (3 pt) would split them and land the origin on the
    second hairline. The region's own rule-gap natural break merges the
    pair, so the origin is the header rule.
    """
    _, page = _new_page()
    _draw_rule(page, 80.0, width=0.5)
    _draw_rule(page, 84.0, width=0.5)  # 4 pt doubled pair
    page.insert_text((_LABEL_X, 95.0), "Hdr", fontsize=6)
    page.insert_text((_COL1_X, 95.0), "c1", fontsize=6)
    _draw_rule(page, 110.0, width=0.5)  # header / mid rule — the origin
    page.insert_text((_LABEL_X, 125.0), "A", fontsize=6)
    page.insert_text((_COL1_X, 125.0), "1", fontsize=6)
    page.insert_text((_LABEL_X, 140.0), "B", fontsize=6)
    page.insert_text((_COL1_X, 140.0), "2", fontsize=6)
    region = (_REGION_X0, 70.0, _REGION_X1, 160.0)

    origin = ordinal_origin(page, region)
    assert origin == pytest.approx(110.0, abs=0.01)


def test_ordinal_origin_does_not_merge_distinct_rules_on_large_type():
    """24 pt table, two distinct rules 8 pt apart: they stay two groups.

    A baseline sits between them, so they are not one drawn border (the
    gap-below-line-height test never fires). Origin is the second rule.
    """
    _, page = _new_page()
    _draw_rule(page, 80.0, width=1.0)
    page.insert_text((_LABEL_X, 84.0), "Hdr", fontsize=24)
    page.insert_text((_COL1_X, 84.0), "c", fontsize=24)
    _draw_rule(page, 88.0, width=1.0)  # 8 pt, but a baseline lies between
    region = (_REGION_X0, 70.0, _REGION_X1, 160.0)

    origin = ordinal_origin(page, region)
    assert origin == pytest.approx(88.0, abs=0.01)


def test_row_bands_uniform_pitch_one_band_per_printed_row():
    """6 printed rows at a uniform 9.5 pt pitch on 10 pt type → 6 bands.

    Baseline-distance clustering (even vs the band's first line) still
    admits the neighbour (9.5 < 10) and yields 3 bands. Vertical-extent
    overlap vs the adjacent-row overlap on this page keeps one band per
    printed row.
    """
    _, page = _new_page()
    fontsize = 10.0
    step = 9.5
    for i in range(6):
        y = 120.0 + i * step
        page.insert_text((_LABEL_X, y), f"Row{i}", fontsize=fontsize)
        page.insert_text((_COL1_X, y), f"{i}.0", fontsize=fontsize)
    region = (_REGION_X0, 100.0, _REGION_X1, 200.0)

    bands = row_bands_from_lines(page, region)
    assert len(bands) == 6


def test_row_bands_subscript_joins_its_label():
    """A subscript under a label is the same printed row (doc04 shape).

    Four printed rows so the row pitch occurs more than once and is
    certified; a two-row fixture has unique gaps and would over-split.
    """
    _, page = _new_page()
    page.insert_text((_LABEL_X, 140.0), "GDP", fontsize=10)
    page.insert_text((_LABEL_X + 12.0, 144.0), "t", fontsize=7)
    page.insert_text((_COL1_X, 140.0), "1.0", fontsize=10)
    for i, y in enumerate((160.0, 180.0, 200.0)):
        page.insert_text((_LABEL_X, y), f"R{i}", fontsize=10)
        page.insert_text((_COL1_X, y), f"{i}.0", fontsize=10)
    region = (_REGION_X0, 120.0, _REGION_X1, 220.0)

    bands = row_bands_from_lines(page, region)
    assert len(bands) == 4
    first = bands[0]
    assert first.y0 <= 140.0 <= first.y1
    assert first.y0 <= 144.0 <= first.y1


def test_row_bands_jittered_pitch_keeps_subscript_fold():
    """Reviewer construction: 8 printed rows at a 12 pt pitch with ±0.1 pt
    row-to-row jitter (no two gaps repeat bit-for-bit) plus a subscript under
    row 3 (doc04 shape). Exact-float gap equality called this "no gap repeats"
    and over-split to 9 bands, losing the fold. Clustering by the region's
    smallest gap certifies the pitch: 8 bands, subscript inside row 3's band,
    ambiguity None. Pinned as a difference from the over-split.
    """
    _, page = _new_page()
    jitter = (0.0, 0.1, -0.1, 0.05, -0.05, 0.1, -0.1, 0.0)
    ys = [140.0 + 12.0 * i + jitter[i] for i in range(8)]
    for i, y in enumerate(ys):
        page.insert_text((_LABEL_X, y), f"R{i}", fontsize=10)
        page.insert_text((_COL1_X, y), f"{i}.0", fontsize=10)
    page.insert_text((_LABEL_X + 14.0, ys[2] + 4.0), "t", fontsize=7)  # subscript under row 3
    region = (_REGION_X0, 120.0, _REGION_X1, 250.0)

    bands = row_bands_from_lines(page, region)
    assert len(bands) == 8
    assert all(b.ambiguity is None and b.source == "line" for b in bands)
    third = bands[2]
    assert third.y0 <= ys[2] <= third.y1
    assert third.y0 <= ys[2] + 4.0 <= third.y1


def test_row_bands_one_stray_close_pair_does_not_poison_the_region():
    """Reviewer construction: 8 rows at a 12 pt pitch with ±0.2 pt jitter plus
    ONE stray pair of line entries 0.3 pt apart on the same visual row (a split
    same-line span, or a tight subscript). Under gap clustering the stray gap
    set the bound for the whole region and every row over-split ("tied pitch
    clusters"). Under row-capable pairs the stray pair is sub-size and folds
    into its row; the eleven 12 pt pairs are row-capable and certify the
    rows: 8 bands, ambiguity None.
    """
    _, page = _new_page()
    jitter = (0.0, 0.2, -0.2, 0.1, -0.1, 0.2, -0.2, 0.0)
    ys = [140.0 + 12.0 * i + jitter[i] for i in range(8)]
    for i, y in enumerate(ys):
        page.insert_text((_LABEL_X, y), f"R{i}", fontsize=10)
        page.insert_text((_COL1_X, y), f"{i}.0", fontsize=10)
    page.insert_text((_COL1_X + 30.0, ys[4] + 0.3), "b", fontsize=10)  # stray 0.3 pt pair
    region = (_REGION_X0, 120.0, _REGION_X1, 250.0)

    bands = row_bands_from_lines(page, region)
    assert len(bands) == 8
    assert all(b.ambiguity is None for b in bands)
    fifth = bands[4]
    assert fifth.y0 <= ys[4] <= fifth.y1 and fifth.y0 <= ys[4] + 0.3 <= fifth.y1


def _box_line(baseline: float, y0: float, y1: float) -> dict:
    """A synthetic PDF line with an exact ``[y0, y1]`` box."""
    return {
        "baseline": baseline,
        "size": y1 - y0,
        "bbox": (40.0, y0, 80.0, y1),
        "x0": 40.0,
        "x1": 80.0,
    }


def test_row_bands_nonuniform_pitch_does_not_merge():
    """Baselines 0 / 9.5 / 19.1, 10 pt-high boxes → 3 bands.

    Picking the larger gap (9.6) as pitch sets the overlap threshold to
    0.4 and merges the first two rows (overlap 0.5). No gap value occurs
    more than once, so no merge is certified.
    """
    lines = [
        _box_line(0.0, 0.0, 10.0),
        _box_line(9.5, 9.5, 19.5),
        _box_line(19.1, 19.1, 29.1),
    ]
    groups, ambiguity, _flagged = _group_lines_by_baseline_with_reason(lines)
    assert len(groups) == 3
    assert [ln["baseline"] for g in groups for ln in g] == [0.0, 9.5, 19.1]
    # Every gap (9.5, 9.6) is below the 10 pt type: no pair can be certified as
    # two rows, so nothing merges and the region says so — over-split, surfaced.
    assert ambiguity is not None


def test_row_bands_tied_modal_pitch_does_not_merge():
    """Tied modal gaps 9.5 / 9.6 (two each), 10 pt-high boxes → 5 bands.

    Picking 9.6 sets the overlap threshold to 0.4 and merges every 9.5 pt
    pair (overlap 0.5). A tie does not certify a pitch.
    """
    lines = [
        _box_line(0.0, 0.0, 10.0),
        _box_line(9.5, 9.5, 19.5),
        _box_line(19.1, 19.1, 29.1),
        _box_line(28.6, 28.6, 38.6),
        _box_line(38.2, 38.2, 48.2),
    ]
    groups, ambiguity, _flagged = _group_lines_by_baseline_with_reason(lines)
    assert len(groups) == 5
    assert ambiguity is not None  # 9.5/9.6 gaps under 10 pt type: uncertifiable


def test_row_bands_alternating_gaps_certified_by_row_capable_pairs():
    """Alternating 10 / 20 pt gaps on 10 pt type: every pair is row-capable
    (gap ≥ size), the pairs overlap by 0, so a merge would need overlap > 0
    and none has it — 5 rows, certified, no ambiguity. A majority pitch is
    not consulted; each observed row-capable pair is evidence."""
    lines = [
        _box_line(0.0, 0.0, 10.0),
        _box_line(10.0, 10.0, 20.0),
        _box_line(30.0, 30.0, 40.0),
        _box_line(40.0, 40.0, 50.0),
        _box_line(60.0, 60.0, 70.0),
    ]
    groups, ambiguity, _flagged = _group_lines_by_baseline_with_reason(lines)
    assert len(groups) == 5
    assert ambiguity is None


def test_row_bands_from_lines_marks_ambiguous_pitch_on_every_band(tmp_path):
    """The ambiguity reaches the returned geometry: every band from an
    uncertifiable pitch carries ``source == "line-ambiguous"`` and a reason,
    while a certified pitch yields plain ``"line"`` bands with ``ambiguity``
    None. Pinned as a difference between the two fixtures."""
    import fitz

    def _page(baselines):
        doc = fitz.open()
        page = doc.new_page(width=300, height=300)
        for i, y in enumerate(baselines):
            page.insert_text((20, 40 + y), f"row{i}", fontsize=10)
        return doc, page

    certified, page = _page([0, 12, 24, 36, 48, 60])
    bands = row_bands_from_lines(page, (0, 0, 300, 300))
    assert bands and all(b.source == "line" and b.ambiguity is None for b in bands)
    certified.close()

    ambiguous, page = _page([0, 8, 16, 24, 32])  # every gap below the 10 pt type
    bands = row_bands_from_lines(page, (0, 0, 300, 300))
    assert bands and all(b.source == "line-ambiguous" and b.ambiguity for b in bands)
    ambiguous.close()


def test_ordinal_origin_three_rule_plain_table_origin_is_midrule():
    """3-rule toprule/midrule/bottomrule, gaps 15/45: origin is the midrule.

    A ratio heuristic on two gaps always declares a class break and would
    merge top+mid (origin = bottomrule at 140). No text-between + gap <
    line height does not: 15 pt holds a header line.
    """
    _, page = _new_page()
    _draw_rule(page, 80.0)
    page.insert_text((_LABEL_X, 90.0), "Hdr", fontsize=10)
    page.insert_text((_COL1_X, 90.0), "c1", fontsize=10)
    _draw_rule(page, 95.0)
    page.insert_text((_LABEL_X, 110.0), "A", fontsize=10)
    page.insert_text((_COL1_X, 110.0), "1", fontsize=10)
    page.insert_text((_LABEL_X, 125.0), "B", fontsize=10)
    page.insert_text((_COL1_X, 125.0), "2", fontsize=10)
    _draw_rule(page, 140.0)
    region = (_REGION_X0, 70.0, _REGION_X1, 155.0)

    origin = ordinal_origin(page, region)
    assert origin == pytest.approx(95.0, abs=0.01)


def test_ordinal_origin_doubled_then_ordinary_second_group():
    """Doubled 2.5 pt pair then an ordinary rule 20 pt below: origin is
    the second group (the ordinary rule)."""
    _, page = _new_page()
    _draw_rule(page, 80.0, width=0.5)
    _draw_rule(page, 82.5, width=0.5)
    page.insert_text((_LABEL_X, 92.0), "Hdr", fontsize=10)
    page.insert_text((_COL1_X, 92.0), "c1", fontsize=10)
    _draw_rule(page, 102.5)
    page.insert_text((_LABEL_X, 120.0), "A", fontsize=10)
    page.insert_text((_COL1_X, 120.0), "1", fontsize=10)
    region = (_REGION_X0, 70.0, _REGION_X1, 140.0)

    origin = ordinal_origin(page, region)
    assert origin == pytest.approx(102.5, abs=0.01)


def test_ordinal_origin_two_doubled_borders_only_is_none():
    """Gaps 2.5/3.0 only: two doubled borders, no ordinary rule → None."""
    _, page = _new_page()
    _draw_rule(page, 80.0, width=0.5)
    _draw_rule(page, 82.5, width=0.5)
    _draw_rule(page, 85.5, width=0.5)
    page.insert_text((_LABEL_X, 120.0), "A", fontsize=10)
    page.insert_text((_COL1_X, 120.0), "1", fontsize=10)
    region = (_REGION_X0, 70.0, _REGION_X1, 140.0)

    assert ordinal_origin(page, region) is None


def test_label_column_edge_equal_size_wrap_is_an_ambiguous_band_and_r_is_the_data_column():
    """An equal-size wrap 6 pt under a 9 pt label is geometrically a tight
    row or a continuation — no evidence either way — so its boundary is
    ambiguous: the wrap is its own band, flagged, and the column edge is the
    real data column (150), not a collapse. C2b abstains on that row only.
    """
    _, page = _new_page()
    page.insert_text((_LABEL_X, 140.0), "Central government net", fontsize=9)
    page.insert_text((_LABEL_X, 146.0), "debt", fontsize=9)
    page.insert_text((_COL1_X, 146.0), "1.0", fontsize=9)
    page.insert_text((_COL2_X, 146.0), "2.0", fontsize=9)
    for y, a, b, lab in (
        (170.0, "4.0", "5.0", "Other row"),
        (200.0, "6.0", "7.0", "Third"),
        (230.0, "8.0", "9.0", "Fourth"),
    ):
        page.insert_text((_LABEL_X, y), lab, fontsize=9)
        page.insert_text((_COL1_X, y), a, fontsize=9)
        page.insert_text((_COL2_X, y), b, fontsize=9)
    region = (_REGION_X0, 120.0, _REGION_X1, 250.0)
    bands = row_bands_from_lines(page, region)
    assert sum(1 for b in bands if b.ambiguity) == 2  # both sides of the ambiguous boundary
    assert label_column_edge(page, region) == _COL1_X


def test_label_column_edge_none_when_wrapped_label_collapses_r():
    """A wrapped-label row must not report the label's own x0 as R.

    The wrap (smaller type, inside the label's x-span) merges into the first
    label line's band, so its x0 is a non-leftmost candidate and R collapses
    onto the label column — the same degeneracy as R == region.x0. Return None.
    """
    _, page = _new_page()
    page.insert_text((_LABEL_X, 140.0), "Central government net", fontsize=9)
    # smaller-type wrap inside the label span, alone on its baseline: merges
    page.insert_text((_LABEL_X, 146.0), "debt", fontsize=7)
    page.insert_text((_COL1_X, 140.0), "1.0", fontsize=9)
    page.insert_text((_COL2_X, 140.0), "2.0", fontsize=9)
    page.insert_text((_LABEL_X, 170.0), "Other row", fontsize=9)
    page.insert_text((_COL1_X, 170.0), "4.0", fontsize=9)
    page.insert_text((_COL2_X, 170.0), "5.0", fontsize=9)
    page.insert_text((_LABEL_X, 200.0), "Third", fontsize=9)
    page.insert_text((_COL1_X, 200.0), "6.0", fontsize=9)
    page.insert_text((_COL2_X, 200.0), "7.0", fontsize=9)
    page.insert_text((_LABEL_X, 230.0), "Fourth", fontsize=9)
    page.insert_text((_COL1_X, 230.0), "8.0", fontsize=9)
    page.insert_text((_COL2_X, 230.0), "9.0", fontsize=9)
    region = (_REGION_X0, 120.0, _REGION_X1, 250.0)

    assert label_column_edge(page, region) is None


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
        if slug == "doc04":
            assert band_count == 12, f"doc04 p{page_num}: bands {band_count} != 12"
        if slug == "doc05":
            assert band_count == 17, f"doc05 p{page_num}: bands {band_count} != 17"
