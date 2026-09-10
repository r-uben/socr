"""GH-703: A2's row-shortfall term must not floor a complete TEXT table.

A2's term (b) (``structure_check._truncated_row_shortfall``) reconciles the
candidate's numeric body-row count against the native table-shaped row count,
with ``row_shape_min`` derived from the candidate itself. On a text table --
a comparison box whose cells are sentences carrying zero or one number each --
that minimum collapses to 1, so every prose line on the page that mentions a
figure counts as a native "table row", and a complete, ladder-accepted
candidate reads as a massive shortfall and is discarded.

The gate added here (``structure_check._numeric_dominant``) restricts term (b)
to candidates whose own body rows are numeric-dominant: a strict majority
carrying at least two genuine numeric tokens. Term (a) (the style break) and
the ladder verdict are untouched.

Difference pins below monkeypatch ``_numeric_dominant`` to a constant ``True``,
which restores the pre-#703 behaviour exactly (the gate is the only change to
term (b)), and assert the one outcome that flips.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from socr.tables import structure_check
from socr.tables.reconcile import raw_table_block_lines
from socr.tables.row_corroboration import numeric_body_rows, table_blocks
from socr.tables.structure_check import (
    DEFECT_TABLE_TRUNCATED,
    _final_row_truncated,
    _numeric_dominant,
    table_output_defect,
    table_truncated,
)


def _row_words(y: float, tokens: list[str]) -> list[tuple]:
    """Native words shaped like ``page.get_text("words")``, one baseline band."""
    words = []
    x = 0.0
    for tok in tokens:
        words.append((x, y, x + 8.0, y + 10.0, tok))
        x += 12.0
    return words


# ---------------------------------------------------------------------------
# The dominance rule itself
# ---------------------------------------------------------------------------


def test_all_single_numeric_rows_are_not_numeric_dominant() -> None:
    """The #703 shape: prose cells, one number each."""
    assert _numeric_dominant([("4%",), ("32.",)]) is False


def test_wide_numeric_rows_are_numeric_dominant() -> None:
    assert _numeric_dominant([("2018", "1.0", "2.0"), ("2019", "1.1", "2.1")]) is True


def test_one_stray_single_numeric_row_does_not_disable_the_guard() -> None:
    """Why a strict majority and not ``min(len(row)) >= 2``: a lone total or
    footnote-marker row inside an otherwise numeric table would take the
    minimum to 1 and switch term (b) off for the whole candidate.
    """
    rows = [("2018", "1.0", "2.0"), ("2019", "1.1", "2.1"), ("7",)]
    assert min(len(row) for row in rows) == 1
    assert _numeric_dominant(rows) is True


def test_empty_candidate_rows_are_not_dominant() -> None:
    assert _numeric_dominant([]) is False


# ---------------------------------------------------------------------------
# Hermetic synthetic text table
# ---------------------------------------------------------------------------

# A two-column comparison box, cells are sentences. Two of the eight body rows
# carry exactly one number; the rest carry none. This is the BoE Table 3.B
# shape, reduced.
TEXT_TABLE_MD = (
    "| Developments anticipated in February | Developments now anticipated |\n"
    "| :--- | :--- |\n"
    "| **Unemployment** | **Revised down slightly** |\n"
    "| Unemployment rate to remain around four percent. | Unemployment rate to fall to 4% "
    "by the end of the year. |\n"
    "| **Participation** | **Broadly unchanged** |\n"
    "| Participation rate to remain just above the recent average. | Participation rate to "
    "remain just under that average. |\n"
    "| **Average hours** | **Broadly unchanged** |\n"
    "| Average weekly hours worked to remain around 32. | Average weekly hours worked to "
    "fall slightly. |\n"
    "| **Productivity** | **Broadly unchanged** |\n"
    "| Quarterly hourly labour productivity growth to average just over a quarter of a "
    "percent. | Unchanged from February. |\n"
)

# The page around that box: prose lines, most of which mention exactly one
# figure. At row_shape_min == 1 every one of these counts as a native
# "table-shaped row" -- which is the whole defect.
TEXT_TABLE_WORDS: list[tuple] = []
for _i, _line in enumerate(
    [
        ["Inflation", "Report", "May", "2018", "Section", "3"],
        ["Chart", "3.7", "Productivity", "growth", "remains", "subdued"],
        ["Percentage", "changes", "on", "a", "year", "earlier"],
        ["Output", "per", "hour", "since", "2002"],
        ["Sources:", "ONS", "and", "Bank", "calculations", "2018"],
        ["Unemployment", "rate", "to", "fall", "to", "4%"],
        ["Average", "weekly", "hours", "worked", "around", "32"],
        ["For", "more", "details", "see", "Tenreyro", "2018"],
        ["De-globalisation", "and", "inflation", "Carney", "2017"],
        ["Productivity", "growth", "in", "the", "year", "2016"],
        ["Business", "investment", "growth", "in", "2015"],
        ["Total", "factor", "productivity", "since", "2014"],
    ]
):
    TEXT_TABLE_WORDS += _row_words(10.0 + _i * 20.0, _line)


def test_text_table_is_not_truncated() -> None:
    """A complete text table must survive the shipping gate."""
    candidate_rows = [
        row for rows in table_blocks(TEXT_TABLE_MD) for row in numeric_body_rows(rows) if row
    ]
    assert candidate_rows, "fixture must have numeric body rows for term (b) to be reachable"
    assert all(len(row) == 1 for row in candidate_rows), "fixture must be the text-table shape"

    assert table_truncated(TEXT_TABLE_MD, TEXT_TABLE_WORDS) is False
    assert table_output_defect(TEXT_TABLE_MD, TEXT_TABLE_WORDS) != DEFECT_TABLE_TRUNCATED


def test_text_table_difference_pin_term_b_ungated_vs_gated(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same page, same words, same term (a): the ONLY thing that changes is
    whether term (b)'s dominance gate is consulted.
    """
    gated = table_truncated(TEXT_TABLE_MD, TEXT_TABLE_WORDS)

    monkeypatch.setattr(structure_check, "_numeric_dominant", lambda rows: True)
    ungated = table_truncated(TEXT_TABLE_MD, TEXT_TABLE_WORDS)

    assert (ungated, gated) == (True, False)
    # and term (a) is not what changed -- it never fired on this page
    assert not any(_final_row_truncated(block) for block in raw_table_block_lines(TEXT_TABLE_MD))


def test_numeric_table_shortfall_is_unchanged_by_the_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other side of the difference pin: on a numeric-dominant candidate
    the gate is inert -- gated and ungated verdicts are identical, truncated
    and complete alike.
    """
    rows = [(2000 + i, float(100 + i), float(200 + i)) for i in range(20)]
    header = "| Year | A | B |\n|---|---|---|\n"
    complete_md = header + "".join(f"| {y} | {a} | {b} |\n" for y, a, b in rows)
    truncated_md = header + "".join(f"| {y} | {a} | {b} |\n" for y, a, b in rows[:-2])
    words: list[tuple] = []
    for i, (year, a, b) in enumerate(rows):
        words += _row_words(10.0 + i * 20.0, [str(year), str(a), str(b)])

    gated = (table_truncated(complete_md, words), table_truncated(truncated_md, words))
    monkeypatch.setattr(structure_check, "_numeric_dominant", lambda candidate_rows: True)
    ungated = (table_truncated(complete_md, words), table_truncated(truncated_md, words))

    assert gated == ungated == (False, True)


# ---------------------------------------------------------------------------
# The real BoE fixture (corpus-skipped)
# ---------------------------------------------------------------------------

_CENSUS = Path.home() / "Data/socr/census-boe-2026-09-10"
BOE_2018_PDF = _CENSUS / "in/boe-meetings-2018-scan-p28-30.pdf"
# The cached qwen candidate for p1 shipped in the census run: 23/23 of the
# page's numbers, 0 extras, 10 pipe rows, table_ladder_accepted -- and floored
# to 0/23 by candidate_truncated. See docs/log/2026-09-10_third-institution-census.md.
BOE_2018_P1_QWEN = (
    _CENSUS
    / "out/boe-meetings-2018-scan-p28-30/cache/c6"
    / "c6270478f9d5cd36d245a5663cd19285c35e1f104e0dc133a6d8e5d3ec90a098.json"
)


def _boe_p1() -> tuple[str, list]:
    import pymupdf

    markdown = json.loads(BOE_2018_P1_QWEN.read_text())["text"]
    with pymupdf.open(BOE_2018_PDF) as doc:
        words = doc[0].get_text("words")
    return markdown, list(words)


@pytest.mark.skipif(
    not (BOE_2018_PDF.exists() and BOE_2018_P1_QWEN.exists()),
    reason="BoE census corpus not present on this machine",
)
def test_real_boe_p1_text_table_ships() -> None:
    """The ticket's own page, on the real PDF and the real cached attempt."""
    markdown, words = _boe_p1()

    # grounding: this IS the complete candidate the ladder accepted
    assert "Table 3.B Monitoring the MPC's key judgements" in markdown
    assert "Unemployment rate to fall to 4% by the end of the year." in markdown

    candidate_rows = [
        row for rows in table_blocks(markdown) for row in numeric_body_rows(rows) if row
    ]
    assert candidate_rows == [("4%",), ("32.",)]
    assert _numeric_dominant(candidate_rows) is False

    assert table_truncated(markdown, words) is False
    assert table_output_defect(markdown, words) != DEFECT_TABLE_TRUNCATED


@pytest.mark.skipif(
    not (BOE_2018_PDF.exists() and BOE_2018_P1_QWEN.exists()),
    reason="BoE census corpus not present on this machine",
)
def test_real_boe_p1_difference_pin(monkeypatch: pytest.MonkeyPatch) -> None:
    """The regression, pinned as a difference on the real page: ungated term
    (b) truncates it (the census outcome, 0/23 shipped); gated, it does not.
    """
    markdown, words = _boe_p1()

    gated = table_truncated(markdown, words)
    monkeypatch.setattr(structure_check, "_numeric_dominant", lambda candidate_rows: True)
    ungated = table_truncated(markdown, words)

    assert (ungated, gated) == (True, False)
