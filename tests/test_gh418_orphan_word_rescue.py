"""GH-418: a word beyond every lane's snap radius must not vanish.

`_rowize_segment` assigned each row word to its nearest lane only if it was
within the snap radius; a word further than that from EVERY lane was discarded
with no cell, no label and no event. Footnote markers -- `n.a.`, a dagger, a
star -- are exactly the tokens that qualify a number, so losing them silently is
the citation-corpus failure this repo forbids.

The drop was load-bearing, not an oversight: it is what stops a PROSE page being
gridded whole. #342 tried removing the radius, and tried capturing anything
between the first and last lane; both turned prose into a whole-page table,
because on a prose page the lanes span the text width and x-position alone
cannot tell the two apart.

Row-level evidence can, which is the discriminator the ticket proposed. A row
that already snaps at least `_MIN_LANES_PER_ROW` NUMERIC words into distinct
lanes is a data row by the same standard the rest of the module uses; a prose
line is not, whatever its lanes look like.
"""

from __future__ import annotations

from socr.tables.reconstruct import rowize_from_word_list


def _w(x0: float, y0: float, text: str, width: float = 24.0) -> tuple:
    return (x0, y0, x0 + width, y0 + 10, text, 0, 0, 0)


def _table_with_gutter_marker() -> list:
    """Four data rows, each with an `n.a.` stranded mid-gutter."""
    words: list = []
    y = 100.0
    for r in range(4):
        words += [
            _w(50.0, y, f"Label{r}"),
            _w(150.0, y, f"{r}.11"),
            _w(230.0, y, "n.a."),
            _w(320.0, y, f"{r}.22"),
            _w(400.0, y, f"{r}.33"),
        ]
        y += 16.0
    return words


def _cells(md: str) -> list[list[str]]:
    return [
        [c.strip() for c in line.strip().strip("|").split("|")]
        for line in md.splitlines()
        if line.lstrip().startswith("|") and "---" not in line
    ]


def test_a_stranded_marker_is_not_dropped() -> None:
    regions = rowize_from_word_list(_table_with_gutter_marker())
    assert regions, "fixture must reconstruct"
    tokens = {t for row in _cells(regions[0][1]) for c in row for t in c.split()}
    assert "n.a." in tokens, (
        f"the gutter marker was silently dropped from the grid: {regions[0][1]}"
    )


def test_the_data_values_are_untouched() -> None:
    """Control: rescuing an orphan must not move or drop a real value."""
    tokens = {
        t
        for row in _cells(rowize_from_word_list(_table_with_gutter_marker())[0][1])
        for c in row
        for t in c.split()
    }
    for r in range(4):
        for suffix in (".11", ".22", ".33"):
            assert f"{r}{suffix}" in tokens, f"data value {r}{suffix} was lost"


# The prose half of this fix is guarded by an EXISTING test, not a new one here:
# `tests/test_reconstruct.py::test_prose_with_small_embedded_table_is_not_whole_page_gridded`.
# That is the test #342's two attempts failed, and it is the reason the drop was
# load-bearing in the first place.
#
# I tried to add a prose fixture of my own and could not build one: a prose row
# appended to a table fixture makes the whole segment fail the tabular gate, so
# it reconstructs to NOTHING and the assertion would have been vacuous. Rather
# than ship a fixture that cannot fail for the right reason, the scope is
# verified by reverting the row-evidence condition and confirming the existing
# prose guard goes red -- recorded in the commit message with the result.
