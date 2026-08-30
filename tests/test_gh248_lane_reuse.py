"""GH-248: a corrupt text layer must not read as a borderless table.

``has_numeric_columns`` is the second-pass gate: it runs when ``find_tables()``
returns nothing, and asks whether numeric tokens form a grid. It counted rows that
populate several x-lanes at once, with no requirement that those lanes RECUR.

A text layer OCR'd upside down manufactures numeric-looking tokens out of ordinary
prose (``LoIic6`` for Police), and they scatter across many nearly-unique lanes. The
co-occupancy count read that scatter as a row x column grid, so pure prose pages were
routed to the table path and inflated every corpus count of "table pages".

The fix tightens "lane" to "column": a lane counts only if it appears on at least
``_MIN_TABLE_ROWS`` rows — a real borderless table reuses the same x positions row
after row, and noise does not. No new constant is introduced; the gate reuses the two
counts it already required.

Measured before adoption, as the issue asked, in both directions:

- Glaeser/Sacerdote/Scheinkman pp. 6, 17, 36, 38 (mirrored OCR, ``find_tables() == 0``)
  went True -> False. 43-66 spurious tokens across 22-30 lanes, ~2 per lane.
- All 15 table pages of the 9-paper manifest kept their classification exactly.

These tests use synthetic geometry so they are hermetic; nothing reads a PDF from
disk and no provider runs.
"""

from __future__ import annotations

from socr.tables.reconstruct import (
    _MIN_LANES_PER_ROW,
    _MIN_TABLE_ROWS,
    has_numeric_columns,
)


class _Word(tuple):
    """A ``get_text("words")`` tuple: (x0, y0, x1, y1, text, block, line, word)."""


def _w(x: float, y: float, text: str) -> tuple:
    return (x, y, x + 18.0, y + 8.0, text, 0, 0, 0)


class _FakePage:
    def __init__(self, words):
        self._words = words

    def get_text(self, kind):
        assert kind == "words"
        return self._words


def _real_table(rows: int = 4, cols: int = 4) -> _FakePage:
    """A borderless table: the same lanes, reused on every data row."""
    words = []
    for r in range(rows):
        y = 100.0 + r * 14.0
        for c in range(cols):
            words.append(_w(80.0 + c * 60.0, y, f"{r}.{c}2"))
    return _FakePage(words)


def _scatter(rows: int = 10, per_row: int = 4) -> _FakePage:
    """Corrupt-layer noise: as many tokens, but each in its own lane.

    Deliberately gives the page MORE numeric tokens and MORE co-occupied rows than
    the real table above, so the difference cannot be explained by volume.
    """
    words = []
    x = 60.0
    for r in range(rows):
        y = 100.0 + r * 14.0
        for _ in range(per_row):
            words.append(_w(x, y, "911"))
            x += 23.0  # every token lands in a fresh lane, never revisited
    return _FakePage(words)


def test_a_real_borderless_table_is_still_detected():
    """The case the gate exists for must survive the fix."""
    assert has_numeric_columns(_real_table()) is True


def test_scattered_noise_is_not_a_table():
    """The regression: lane co-occupancy alone accepted this."""
    assert has_numeric_columns(_scatter()) is False


def test_the_two_are_distinguished_despite_noise_having_more_tokens():
    """Pinned as a DIFFERENCE, and not one volume could explain.

    The noise page carries 40 numeric tokens over 10 rows; the table carries 16 over
    4. Before the fix both passed. The discriminator is lane REUSE, not quantity.
    """
    table, noise = _real_table(), _scatter()
    assert len(noise.get_text("words")) > len(table.get_text("words"))
    assert has_numeric_columns(table) != has_numeric_columns(noise)


def test_lanes_must_recur_not_merely_co_occur():
    """The precise mechanism: same row count, same lanes-per-row, different reuse.

    Both pages put `_MIN_LANES_PER_ROW` tokens on each of `_MIN_TABLE_ROWS` rows. The
    only difference is whether those rows land in the SAME lanes.
    """
    shared, fresh = [], []
    x = 60.0
    for r in range(_MIN_TABLE_ROWS):
        y = 100.0 + r * 14.0
        for c in range(_MIN_LANES_PER_ROW):
            shared.append(_w(60.0 + c * 60.0, y, "1.23"))  # same lanes every row
            fresh.append(_w(x, y, "1.23"))  # a new lane every single token
            x += 40.0

    assert has_numeric_columns(_FakePage(shared)) is True
    assert has_numeric_columns(_FakePage(fresh)) is False


def test_a_page_with_no_numeric_tokens_is_unchanged():
    """The early exit is untouched."""
    assert has_numeric_columns(_FakePage([_w(60.0, 100.0, "Police")])) is False
