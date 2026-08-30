"""GH-331: a bracketed value split across words must be rejoined.

``get_text("words")`` splits on whitespace, so a confidence interval printed as
``[0.01, 0.35]`` arrives as two words — ``[0.01,`` and ``0.35]``. Lane assignment
then places them by their own x positions and they land in different columns, so
the interval ships torn in half:

    | Small T | (0.15) | (0.34) | [0.01, |  | 0.35] |

A reader taking ``[0.01,`` as the value gets a truncated number with no signal that
anything is missing.

Hermetic: synthetic word geometry, no PDF read and no provider.
"""

from __future__ import annotations

from socr.tables.reconstruct import _merge_unclosed_bracket_words, rowize_from_word_list


def _w(x: float, y: float, text: str) -> tuple:
    return (x, y, x + 20.0, y + 8.0, text, 0, 0, 0)


def test_a_split_interval_is_rejoined():
    """The regression: two words, two lanes, one torn value."""
    row = [_w(100.0, 50.0, "[0.01,"), _w(140.0, 50.0, "0.35]")]
    merged = _merge_unclosed_bracket_words(row)
    assert len(merged) == 1
    assert merged[0][4] == "[0.01, 0.35]"


def test_a_prose_parenthetical_is_left_alone():
    """The bound, and the reason for it.

    An unclosed bracket in prose — a table note reading "(as in Gertler and
    Karadi, 2015)." — would otherwise be swallowed into one token and lost. That is
    silent content loss, strictly worse than the split being repaired. Measured: an
    unbounded version dropped words from a real corpus page.
    """
    row = [
        _w(100.0, 50.0, "(as"),
        _w(120.0, 50.0, "in"),
        _w(140.0, 50.0, "Gertler"),
        _w(180.0, 50.0, "2015)."),
    ]
    assert _merge_unclosed_bracket_words(row) == row


def test_an_unclosed_bracket_that_never_closes_is_untouched():
    """A runaway merge would consume the rest of the row."""
    row = [_w(100.0, 50.0, "[0.01,"), _w(140.0, 50.0, "0.35")]
    assert _merge_unclosed_bracket_words(row) == row


def test_no_token_is_lost_by_merging():
    """The house rule, pinned: rejoining may reshape cells, never drop content."""
    row = [
        _w(60.0, 50.0, "Small"),
        _w(90.0, 50.0, "T"),
        _w(140.0, 50.0, "[0.01,"),
        _w(180.0, 50.0, "0.35]"),
        _w(220.0, 50.0, "(0.15)"),
    ]
    before = " ".join(w[4] for w in row)
    after = " ".join(w[4] for w in _merge_unclosed_bracket_words(row))
    assert sorted(before.split()) == sorted(after.split())


def test_the_interval_reaches_one_cell_end_to_end():
    """Pinned as a DIFFERENCE through the rowizer, not just the helper."""

    def grid(split: bool):
        words: list = []
        y = 100.0
        for _ in range(4):
            words.append(_w(60.0, y, "Row"))
            words.append(_w(140.0, y, "1.11"))
            if split:
                words.append(_w(220.0, y, "[0.01,"))
                words.append(_w(260.0, y, "0.35]"))
            else:
                words.append(_w(220.0, y, "[0.01,0.35]"))
            y += 16.0
        regions = rowize_from_word_list(words)
        return [
            [c.strip() for c in line.strip().strip("|").split("|")]
            for line in regions[0][1].splitlines()
            if line.lstrip().startswith("|") and "---" not in line
        ]

    # Assert on the CELL, never on the joined row: joining with spaces turns two
    # adjacent torn cells back into the substring "[0.01, 0.35]", so a substring
    # test passes whether or not the merge happened. That is how the first version
    # of this test managed to be vacuous.
    rows = [r for r in grid(split=True) if any("0.01" in c for c in r)]
    assert rows, "fixture produced no row carrying the interval"
    for row in rows:
        assert any(c.strip() == "[0.01, 0.35]" for c in row), (
            f"interval is not in a single cell: {row}"
        )
