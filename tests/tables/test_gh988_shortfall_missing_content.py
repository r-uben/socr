"""#988: ``table_truncated``'s row-shortfall term is confirmed by content.

The term compares the candidate's numeric body-row count with the count of
every table-shaped band on the whole page. Year captions, footers, footnote
lines and ``<br>`` second lines are bands with numbers in them but never body
rows, so complete readings were refused as truncated (Coca-Cola 2018-2021
sustainability reports: 32 of 40 ``table_truncated`` refusals were complete and
correct). The term now fires only when the count of bands the candidate
ACCOUNTS for (its numbers written anywhere, each written number accounting for
one band) falls short as well.

Every test pins a difference in the same process: the complete reading passes
and a cut of it is refused, or a reading the row count accepts is not refused.
"""

from __future__ import annotations

import json
from pathlib import Path

from socr.tables.structure_check import (
    DEFECT_TABLE_TRUNCATED,
    table_output_defect,
    table_truncated,
)

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "gh988_coke_2021_p74"


def _row_words(y: float, tokens: list[str]) -> list[tuple]:
    words = []
    x = 0.0
    for tok in tokens:
        words.append((x, y, x + 8.0, y + 10.0, tok))
        x += 12.0
    return words


def _page_words(rows: list[list[str]]) -> list[tuple]:
    words: list[tuple] = []
    for i, tokens in enumerate(rows):
        words += _row_words(10.0 + i * 20.0, tokens)
    return words


def _table(rows: list[list[str]]) -> str:
    width = len(rows[0])
    md = "| Item | " + " | ".join(f"C{i}" for i in range(1, width)) + " |\n"
    md += "|" + "---|" * width + "\n"
    md += "".join("| " + " | ".join(row) + " |\n" for row in rows)
    return md


# --------------------------------------------------------------------------
# The real page: Coca-Cola 2021 Business & ESG Report p74 (GHG emissions)
# --------------------------------------------------------------------------


def _p74() -> tuple[str, list[tuple]]:
    words = [tuple(w) for w in json.loads((FIXTURE / "words.json").read_text())]
    return (FIXTURE / "gemini.txt").read_text(), words


def test_coke_p74_complete_reading_passes_and_its_cut_is_refused() -> None:
    """Gemini's reading of p74 matches every number on the text layer, but it
    has 17 numeric body rows against 24 table-shaped native bands: the year
    caption, a lone superscript, two ``<br>`` second lines, two footnote lines
    and the footer. All of those numbers are in the reading, so nothing is
    unaccounted. Cut after the "Emissions Ratio" row, the energy rows' numbers
    are gone and the term fires."""
    md, words = _p74()
    lines = md.splitlines()
    cut_at = next(i for i, ln in enumerate(lines) if ln.startswith("| Emissions Ratio"))
    cut = "\n".join(lines[: cut_at + 1])

    assert table_truncated(md, words) is False
    assert table_output_defect(md, words) != DEFECT_TABLE_TRUNCATED
    assert table_truncated(cut, words) is True


# --------------------------------------------------------------------------
# Synthetic pages, one mechanism each
# --------------------------------------------------------------------------

DATA_ROWS = [[f"R{i}", f"{10 + i}.1", f"{20 + i}.2", f"{30 + i}.3"] for i in range(12)]


def _native_values(rows: list[list[str]]) -> list[list[str]]:
    return [row[1:] for row in rows]


def test_footer_and_footnote_bands_written_as_text_are_accounted_for() -> None:
    """A one-value row lowers ``row_shape_min`` to 1, so the footer and two
    footnote lines become table-shaped native bands. The reading writes them
    as text below the table: complete. Cut after row 8 (footnotes and footer
    gone with it): refused."""
    rows = DATA_ROWS + [["Total", "99.9", "", ""]]
    native = _native_values(DATA_ROWS) + [["99.9"], ["1", "Restated", "in", "2019"], ["74"]]
    words = _page_words(native)
    complete = _table(rows) + "\n1 Restated in 2019\n\nAnnual Report | 74\n"
    cut = _table(rows[:8])

    assert table_truncated(complete, words) is False
    assert table_truncated(cut, words) is True


def test_br_second_lines_are_accounted_for() -> None:
    """A two-line cell (``a<br>b``) is printed as two native lines; the second
    line's numbers are in the reading, so the band is accounted for."""
    rows = [list(r) for r in DATA_ROWS]
    native = _native_values(DATA_ROWS)
    for i in (3, 7, 10):
        rows[i] = [rows[i][0]] + [f"{v}<br>{float(v) + 0.5:.1f}" for v in rows[i][1:]]
    for i in sorted((3, 7, 10), reverse=True):
        native.insert(i + 1, [f"{float(v) + 0.5:.1f}" for v in DATA_ROWS[i][1:]])
    words = _page_words(native)

    assert table_truncated(_table(rows), words) is False
    assert table_truncated(_table(rows[:-3]), words) is True


def test_a_value_written_once_accounts_for_one_band_only() -> None:
    """The page prints the same six rows of values in two panels. A reading
    that writes both panels passes; one that writes only the first panel is
    refused, because each written number accounts for one band and the
    second panel's bands find nothing left to draw on."""
    panel = DATA_ROWS[:6]
    second = [[f"S{i}", *row[1:]] for i, row in enumerate(panel)]
    words = _page_words(_native_values(panel) * 2)

    assert table_truncated(_table(panel + second), words) is False
    assert table_truncated(_table(panel), words) is True


def test_glued_superscript_marker_is_accounted_for_in_every_written_form() -> None:
    """PyMuPDF glues a printed superscript to the value before it (``0.32`` +
    ``3`` -> ``0.323``). The reading's marker is glued the same way, whether
    written as ``³``, ``$^3$`` or ``<sup>3</sup>``. The footer and footnote
    bands make the row count fall short, so the content count decides."""
    native = _native_values(DATA_ROWS)
    for i in (2, 5, 9):
        native[i] = [native[i][0] + "3", native[i][1] + "3", native[i][2]]
    native += [["99.9"], ["1", "Restated", "in", "2019"], ["74"]]
    words = _page_words(native)
    furniture = "\n1 Restated in 2019\n\nAnnual Report | 74\n"
    for marker in ("³", "$^3$", "<sup>3</sup>"):
        rows = [list(r) for r in DATA_ROWS] + [["Total", "99.9", "", ""]]
        for i in (2, 5, 9):
            rows[i] = [rows[i][0], rows[i][1] + marker, rows[i][2] + marker, rows[i][3]]
        assert table_truncated(_table(rows) + furniture, words) is False, marker
        assert table_truncated(_table(rows[:-3]), words) is True, marker


def test_never_refuses_a_reading_the_row_count_accepts() -> None:
    """A complete table whose markers the model wrote in a form the content
    count cannot read (``^3``, a space before ``³``, or no marker at all):
    the row count matches the page, so the term does not fire, as before
    #988. Only a page the row count already doubts is checked by content."""
    native = [[a + "3", b, c] for a, b, c in _native_values(DATA_ROWS)]
    words = _page_words(native)
    for marker in ("^3", " ³", ""):
        rows = [[r[0], r[1] + marker, r[2], r[3]] for r in DATA_ROWS]
        assert table_truncated(_table(rows), words) is False, marker


def test_bold_chart_labels_are_accounted_for() -> None:
    """Chart labels written as a list (``- **18**: 0%``) are read through the
    emphasis asterisks."""
    rows = DATA_ROWS + [["Share", "45%", "", ""]]
    native = _native_values(DATA_ROWS) + [["45%"], ["18", "0%"], ["19", "3%"], ["20", "5%"]]
    words = _page_words(native)
    labels = "\n- **18**: 0%\n- **19**: 3%\n- **20**: 5%\n"

    assert table_truncated(_table(rows) + labels, words) is False
    assert table_truncated(_table(rows[:-4]) + labels, words) is True


def test_rows_of_years_count_like_any_other_row() -> None:
    """A table whose values are years (start and end of a programme) loses
    half its rows: refused. Review of the first #988 draft found that skipping
    year-only bands let exactly this cut through."""
    rows = [[f"P{i}", str(1950 + i), str(1980 + i)] for i in range(24)]
    words = _page_words([r[1:] for r in rows])
    header = "| Programme | Start | End |\n|---|---|---|\n"
    full = header + "".join(f"| {a} | {b} | {c} |\n" for a, b, c in rows)
    cut = header + "".join(f"| {a} | {b} | {c} |\n" for a, b, c in rows[:12])

    assert table_truncated(full, words) is False
    assert table_truncated(cut, words) is True


def test_known_limit_a_dropped_row_restated_in_prose_is_credited() -> None:
    """Disclosed limit (#988): the candidate's numbers are counted wherever
    they are written, so a row dropped from the table but restated in prose
    reads as accounted for. Pinned so a change to it is a decision."""
    words = _page_words(_native_values(DATA_ROWS))
    dropped = DATA_ROWS[-3:]
    prose = "\n".join(f"{r[0]} values {r[1]} {r[2]} {r[3]}" for r in dropped)

    assert table_truncated(_table(DATA_ROWS[:-3]), words) is True
    assert table_truncated(_table(DATA_ROWS[:-3]) + "\n" + prose + "\n", words) is False
