"""GH-96: hierarchy-aware table cell exactness.

The metric exists because ``BenchmarkScorer.score_table_cells`` cannot see the
failure mode #96 is about. On a hierarchical table the VLM emits a *permutation* -
every digit present, in the right column, on the wrong row - and the positional
scorer's shape-mismatch fallback is multiset recall, which scores a permutation
100%. Measured on OBR EFO November 2022 page 13, ``score_table_cells`` returns
100.0% both for output that is 38.4% correct by label and for a perfect
transcription of the same page.

Fixtures are generated with ``fitz`` at test time rather than committed, so no
third-party document content lives in the repo. #96 accepts "OBR-style ... or
redacted minimal".
"""

from __future__ import annotations

import fitz
import pytest

from socr.benchmark.scorer import BenchmarkScorer
from socr.benchmark.table_exactness import (
    ExactnessReport,
    LabeledRow,
    _superscript_tokens,
    markdown_rows,
    native_rows_from_page,
    normalize_label,
    score_page,
    score_rows,
)

# An OBR-shaped hierarchical table: section parents that carry their own values,
# "of which:" markers, indented children, and - the case that broke an earlier
# scratch scorer - the SAME child label under two different parents.
_ROWS = [
    ("Total effect of decisions", 0, ["42.8", "30.5", "2.6"]),
    ("September energy package", 0, ["43.2", "26.8", "3.7"]),
    ("of which:", 1, None),
    ("Energy price guarantee", 1, ["24.8", "26.8", "3.7"]),
    ("Other measures", 1, ["18.4", "0.0", "0.0"]),
    ("Autumn Statement", 0, ["3.8", "0.3", "-19.3"]),
    ("of which:", 1, None),
    ("Tax measures", 1, ["-2.2", "-7.3", "-20.1"]),
    ("Other measures", 1, ["0.1", "-5.4", "-3.7"]),
]

_LABEL_X = 60.0
_INDENT_X = 78.0
_VALUE_XS = (300.0, 360.0, 420.0)


@pytest.fixture(scope="module")
def hierarchical_page(tmp_path_factory):
    """A born-digital page carrying the table above, drawn with fitz."""
    path = tmp_path_factory.mktemp("gh96") / "hierarchical.pdf"
    doc = fitz.open()
    page = doc.new_page()

    y = 200.0
    for label, depth, values in _ROWS:
        page.insert_text((_LABEL_X if depth == 0 else _INDENT_X, y), label, fontsize=9)
        if values:
            for x, value in zip(_VALUE_XS, values):
                page.insert_text((x, y), value, fontsize=9)
        y += 18.0

    # A ruled box so locate_tables finds the region.
    page.draw_line(fitz.Point(50, 190), fitz.Point(470, 190))
    page.draw_line(fitz.Point(50, y), fitz.Point(470, y))
    doc.save(path)
    doc.close()

    opened = fitz.open(path)
    yield opened[0]
    opened.close()


# ----------------------------------------------------------------------
# Ground truth extraction
# ----------------------------------------------------------------------


def test_native_ground_truth_recovers_every_data_row(hierarchical_page):
    rows = native_rows_from_page(hierarchical_page)

    labels = [r.label for r in rows]
    assert "Energy price guarantee" in labels
    assert labels.count("Other measures") == 2
    # "of which:" is a hierarchy marker, never a data row.
    assert all(r.label != "of which:" for r in rows)
    assert all(len(r.values) == 3 for r in rows)


def test_duplicate_child_label_is_disambiguated_by_parent(hierarchical_page):
    """The collision that made an earlier scorer understate a perfect page."""
    rows = native_rows_from_page(hierarchical_page)
    others = [r for r in rows if r.label == "Other measures"]

    assert len(others) == 2
    assert others[0].values != others[1].values
    # Each is attributed to its own parent, so a report names the block.
    assert others[0].path != others[1].path


def test_a_numeral_inside_a_label_is_not_read_as_a_value(tmp_path):
    """ "Growth Plan after 17 October reversals" must keep its 17."""
    path = tmp_path / "label_numeral.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((60.0, 200.0), "Growth Plan after 17 October reversals", fontsize=9)
    for x, value in zip(_VALUE_XS, ["8.1", "18.8", "19.3"]):
        page.insert_text((x, 200.0), value, fontsize=9)
    page.draw_line(fitz.Point(50, 190), fitz.Point(470, 190))
    page.draw_line(fitz.Point(50, 215), fitz.Point(470, 215))
    doc.save(path)
    doc.close()

    opened = fitz.open(path)
    rows = native_rows_from_page(opened[0])
    opened.close()

    assert len(rows) == 1
    assert rows[0].label == "Growth Plan after 17 October reversals"
    assert list(rows[0].values) == ["8.1", "18.8", "19.3"]


# ----------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------


def _row(path, values):
    return LabeledRow(path=tuple(path), values=tuple(values))


def test_a_perfect_transcription_scores_100():
    gt = [_row(("A",), ("1.0", "2.0")), _row(("A", "B"), ("3.0", "4.0"))]

    report = score_rows(gt, list(gt))

    assert report.pct == 100.0
    assert report.rows_not_found == 0


def test_the_hierarchical_shift_is_caught():
    """The #96 failure: every value present, each on the row above its own."""
    gt = [
        _row(("September energy package",), ("43.2",)),
        _row(("September energy package", "Energy price guarantee"), ("24.8",)),
        _row(("September energy package", "Energy bill relief scheme"), ("18.4",)),
    ]
    shifted = [
        _row(("Energy price guarantee",), ("43.2",)),
        _row(("Energy bill relief scheme",), ("24.8",)),
    ]

    report = score_rows(gt, shifted)

    assert report.exact == 0
    assert report.pct == 0.0
    # And the misses name the rows, so a reader knows which block moved.
    assert any("Energy price guarantee" in m.path for m in report.misses)


def test_a_permutation_is_not_credited_as_recall():
    """Contrast with the positional scorer's multiset fallback, which scores 100%."""
    gt = [_row(("A",), ("1.0",)), _row(("B",), ("2.0",))]
    swapped = [_row(("A",), ("2.0",)), _row(("B",), ("1.0",))]

    assert score_rows(gt, swapped).pct == 0.0


def test_duplicate_labels_match_in_document_order():
    gt = [
        _row(("P1", "Other measures"), ("1.0",)),
        _row(("P2", "Other measures"), ("2.0",)),
    ]
    predicted = [
        _row(("Other measures",), ("1.0",)),
        _row(("Other measures",), ("2.0",)),
    ]

    report = score_rows(gt, predicted)

    assert report.pct == 100.0, "the n-th occurrence must match the n-th source row"


def test_a_missing_row_is_reported_not_silently_skipped():
    gt = [_row(("A",), ("1.0",)), _row(("B",), ("2.0",))]

    report = score_rows(gt, [_row(("A",), ("1.0",))])

    assert report.rows_not_found == 1
    assert report.cells == 2
    assert report.exact == 1
    assert any(m.got is None for m in report.misses)


# ----------------------------------------------------------------------
# The ceiling: a tie must be distinguishable from a shared failure
# ----------------------------------------------------------------------


def test_degenerate_ground_truth_is_flagged_not_scored_as_zero():
    report = score_rows([_row(("A",), ("1.0",))], [])

    assert report.scorable is False
    assert "not scorable" in report.ceiling_note


def test_unmatched_ground_truth_flags_the_parser_not_the_engine():
    gt = [_row(("A",), ("1.0",)), _row(("B",), ("2.0",))]

    report = score_rows(gt, [_row(("Z",), ("9.9",))])

    assert report.pct == 0.0
    assert report.scorable is False
    assert "parser" in report.ceiling_note


def test_empty_report_has_no_percentage():
    assert ExactnessReport().pct is None


# ----------------------------------------------------------------------
# Markdown parsing
# ----------------------------------------------------------------------


def test_markdown_children_are_bound_to_the_preceding_parent():
    md = "\n".join(
        [
            "| | 2022-23 |",
            "| --- | --- |",
            "| **September energy package** | 43.2 |",
            "| of which: | |",
            "| Energy price guarantee | 24.8 |",
        ]
    )

    rows, _diag = markdown_rows(md)

    by_label = {r.label: r.path for r in rows}
    assert by_label["Energy price guarantee"] == (
        "September energy package",
        "Energy price guarantee",
    )


def test_markdown_counts_orphan_and_empty_rows():
    md = "\n".join(
        [
            "| | 2022-23 |",
            "| --- | --- |",
            "| **September energy package** | |",  # labelled, no values
            "| | 18.4 |",  # values, no label
            "| Tax measures | -2.2 |",
        ]
    )

    _rows, diag = markdown_rows(md)

    assert diag.labelled_but_empty == 1
    assert diag.orphan_rows == 1


def test_label_normalization_folds_emphasis_and_footnotes():
    assert normalize_label("**Income tax1**") == normalize_label("Income tax")
    assert normalize_label("NICs/health and social care levy") == normalize_label(
        "NICs / health and social care levy"
    )


# ----------------------------------------------------------------------
# End to end on the generated fixture
# ----------------------------------------------------------------------


def test_end_to_end_perfect_markdown_scores_100(hierarchical_page):
    md_lines = ["| | c1 | c2 | c3 |", "| --- | --- | --- | --- |"]
    for label, _depth, values in _ROWS:
        md_lines.append(
            f"| {label} | " + " | ".join(values) + " |" if values else f"| {label} | | | |"
        )
    report = score_page(hierarchical_page, "\n".join(md_lines))

    assert report.scorable
    assert report.pct == 100.0


def test_end_to_end_shifted_markdown_scores_poorly(hierarchical_page):
    """Same digits, slid one row down inside each block."""
    md_lines = [
        "| | c1 | c2 | c3 |",
        "| --- | --- | --- | --- |",
        "| Total effect of decisions | 42.8 | 30.5 | 2.6 |",
        "| September energy package | | | |",
        "| of which: | | | |",
        "| Energy price guarantee | 43.2 | 26.8 | 3.7 |",
        "| Other measures | 24.8 | 26.8 | 3.7 |",
        "| | 18.4 | 0.0 | 0.0 |",
    ]
    report = score_page(hierarchical_page, "\n".join(md_lines))

    assert report.scorable
    assert report.pct is not None and report.pct < 50.0
    assert report.orphan_rows == 1


def test_the_positional_scorer_cannot_see_what_this_one_measures(hierarchical_page):
    """Pins the justification for this module existing.

    The shifted table below is a pure permutation, so the positional scorer's
    shape-mismatch fallback (multiset recall) credits it fully.
    """
    gt_md = "\n".join(
        ["| | c1 |", "| --- | --- |"]
        + [f"| {label} | {values[0]} |" for label, _d, values in _ROWS if values]
    )
    shifted_md = "\n".join(
        [
            "| | c1 |",
            "| --- | --- |",
            "| Total effect of decisions | 42.8 |",
            "| September energy package | |",
            "| Energy price guarantee | 43.2 |",
            "| Other measures | 24.8 |",
            "| Autumn Statement | 18.4 |",
            "| Tax measures | 3.8 |",
            "| Other measures | -2.2 |",
            "| | 0.1 |",
        ]
    )

    positional, structure_ok = BenchmarkScorer.score_table_cells(shifted_md, gt_md)
    hierarchical = score_page(hierarchical_page, shifted_md)

    assert structure_ok is False
    assert positional == 1.0, "the existing scorer sees a permutation as perfect"
    assert hierarchical.pct is not None and hierarchical.pct < 50.0


def test_an_engine_that_emitted_nothing_is_a_real_zero_not_a_ceiling():
    """Pages 46/48/55 of the reference doc emitted no table at all.

    Flagging those as unscorable would hide the largest measurable wins, so an
    empty prediction is a genuine 0%, not a parser ceiling.
    """
    gt = [_row(("A",), ("1.0",)), _row(("B",), ("2.0",))]

    report = score_rows(gt, [])

    assert report.pct == 0.0
    assert report.scorable is True, "empty output is an engine failure, not a ceiling"


def _one_row_pdf(path, rows):
    """rows = [(label, label_size, [(value, size)])] laid out as a ruled table."""
    doc = fitz.open()
    page = doc.new_page()
    y = 200.0
    for label, label_size, values in rows:
        page.insert_text((60.0, y), label, fontsize=label_size)
        x = 60.0 + len(label) * label_size * 0.5 + 4
        for value, size in values:
            page.insert_text((x, y), value, fontsize=size)
            x = max(x + 40, _VALUE_XS[0])
        y += 18.0
    page.draw_line(fitz.Point(50, 190), fitz.Point(500, 190))
    page.draw_line(fitz.Point(50, y), fitz.Point(500, y))
    doc.save(path)
    doc.close()
    return fitz.open(path)


class _StubPage:
    """Minimal page exposing only the get_text("dict") shape the helper reads."""

    def __init__(self, spans):
        self._spans = spans

    def get_text(self, kind):
        assert kind == "dict"
        return {
            "blocks": [
                {
                    "lines": [
                        {
                            "spans": [
                                {"size": size, "bbox": (x, 200.0, x + 10, 209.0), "text": text}
                                for size, x, text in self._spans
                            ]
                        }
                    ]
                }
            ]
        }


def test_a_footnote_superscript_is_not_a_value():
    """ "Memo: UK oil and gas revenues5" must not contribute a 5 to ground truth.

    Built as a stub rather than a generated PDF: fitz puts each insert_text call
    on its own line, so a synthetic page cannot place a label, its superscript and
    its values in one line the way a real table does.
    """
    page = _StubPage(
        [
            (9.0, 60.0, "Memo: UK oil and gas revenues"),
            (6.0, 210.0, "5"),
            (9.0, 300.0, "2.6"),
            (9.0, 360.0, "14.9"),
        ]
    )

    markers = _superscript_tokens(page, (0.0, 0.0, 500.0, 500.0))

    assert (210, "5") in markers
    assert not any(t in ("2.6", "14.9") for _x, t in markers)


def test_region_wide_sizing_would_delete_a_small_rows_data():
    """Why the comparison is per-row, not per-region."""
    page = _StubPage([(7.0, 60.0, "Memo: smaller row"), (7.0, 300.0, "5.0"), (7.0, 360.0, "6.0")])

    markers = _superscript_tokens(page, (0.0, 0.0, 500.0, 500.0))

    assert markers == set(), "a uniformly small row has no superscript in it"


def test_a_wholly_small_row_keeps_all_its_values(tmp_path):
    """Regression: comparing font size region-wide deleted a small row's real data.

    A "Memo:" row is often set smaller than the table body. Sizing against the
    region's modal size flagged every one of its values as a superscript; sizing
    within the row does not.
    """
    path = tmp_path / "small_row.pdf"
    opened = _one_row_pdf(
        path,
        [
            ("Body row", 9.0, [("1.0", 9.0), ("2.0", 9.0)]),
            ("Body row two", 9.0, [("3.0", 9.0), ("4.0", 9.0)]),
            ("Memo: smaller row", 7.0, [("5.0", 7.0), ("6.0", 7.0)]),
        ],
    )
    rows = native_rows_from_page(opened[0])
    opened.close()

    memo = [r for r in rows if "Memo" in r.label]
    assert memo, "the small row was dropped entirely"
    assert list(memo[0].values) == ["5.0", "6.0"]


def test_bold_value_cells_are_scored_not_discarded(hierarchical_page):
    """Third instance of the GH-103 defect, in the metric's own tokenizer.

    Engines emit section-total rows in bold. ``BenchmarkScorer._is_numeric_cell``
    anchors its regex without stripping markdown, so ``**43.2**`` read as
    non-numeric, the row looked value-less and was discarded. It under-scored a
    real engine by 39 points and would have argued for changing engines.
    """
    plain = "\n".join(
        ["| | c1 | c2 | c3 |", "| --- | --- | --- | --- |"]
        + [f"| {lab} | " + " | ".join(v) + " |" for lab, _d, v in _ROWS if v]
    )
    bold = "\n".join(
        ["| | c1 | c2 | c3 |", "| --- | --- | --- | --- |"]
        + [
            f"| **{lab}** | " + " | ".join(f"**{x}**" for x in v) + " |"
            for lab, _d, v in _ROWS
            if v
        ]
    )

    assert score_page(hierarchical_page, plain).pct == score_page(hierarchical_page, bold).pct
    assert score_page(hierarchical_page, bold).pct == 100.0


def test_every_footnote_spelling_folds_to_one_key():
    """Five spellings were found in the wild before this became one generic rule.

    The native layer writes a bare trailing digit; one engine writes LaTeX, another
    HTML, and multi-note markers appear as "1,2". Handling them one at a time left
    identical rows one character apart so they never matched: a PERFECT page scored
    85.1%, and one engine's aggregate was understated by 15 points, which reversed
    the apparent ranking of two engines. It reached past reporting - the escalation
    accept rule runs on this metric, so an engine was penalised for its footnote
    syntax.
    """
    bare = normalize_label("Nominal GDP1")
    assert bare == normalize_label("Nominal GDP<sup>1</sup>")
    assert bare == normalize_label("Nominal GDP$^1$")
    assert bare == normalize_label("Nominal GDP^{1}")
    assert bare == normalize_label("Nominal GDP\u00b9")
    # multi-note markers, both spellings
    assert normalize_label("Nominal GDP (£ billion)1,2") == normalize_label(
        "Nominal GDP (£ billion)<sup>1,2</sup>"
    )


def test_latex_and_bare_footnote_markers_fold_together():
    """The native layer writes "Underlying differences1"; engines write "$^1$".

    Handling only the bare-digit form left the two keys one character apart, so the
    rows never matched and a PERFECT page scored 85.1% instead of 100%. That reached
    past reporting: the escalation accept rule runs on this metric, so a candidate
    writing footnotes in LaTeX was penalised against an incumbent that did not.
    """
    assert normalize_label("Underlying differences1") == normalize_label(
        "Underlying differences$^1$"
    )
    assert normalize_label("Energy and cost-of-living support2") == normalize_label(
        "Energy and cost-of-living support$^2$"
    )
    assert normalize_label("Income tax1") == normalize_label("Income tax^{1}")


def test_a_trailing_number_that_is_not_a_footnote_still_distinguishes():
    """The strip is anchored to a preceding letter, so these stay distinct."""
    assert normalize_label("Panel 1") != normalize_label("Panel 2")
    assert normalize_label("Table 3") != normalize_label("Table 4")


def test_a_wrapped_label_is_not_merged_with_the_line_above(tmp_path):
    """Wrapped labels overlap the line above by a descender at tight leading.

    Merging on ANY y-overlap then interleaves the two lines' words by x-sort:
    "Central government net / debt" became "Central net government debt", the row
    never matched its ground truth, and five pages of a real document were reported
    at 86-95% when they were 100%.
    """
    path = tmp_path / "wrapped.pdf"
    doc = fitz.open()
    page = doc.new_page()
    # two label lines at tight leading, then a data row
    page.insert_text((60.0, 200.0), "Central government net", fontsize=9)
    page.insert_text((60.0, 209.0), "debt", fontsize=9)
    for x, v in zip(_VALUE_XS, ["1.0", "2.0", "3.0"]):
        page.insert_text((x, 209.0), v, fontsize=9)
    page.insert_text((60.0, 230.0), "Other row", fontsize=9)
    for x, v in zip(_VALUE_XS, ["4.0", "5.0", "6.0"]):
        page.insert_text((x, 230.0), v, fontsize=9)
    page.draw_line(fitz.Point(50, 190), fitz.Point(470, 190))
    page.draw_line(fitz.Point(50, 245), fitz.Point(470, 245))
    doc.save(path)
    doc.close()

    opened = fitz.open(path)
    rows = native_rows_from_page(opened[0])
    opened.close()

    labels = [r.label for r in rows]
    assert not any("net government" in lbl for lbl in labels), (
        f"words from two lines were interleaved: {labels}"
    )
