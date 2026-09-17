"""#818: ``_is_data_row`` tested RAW model-grid cells, missing the same
decode/dash-strip step already applied on the sibling markdown-table paths
(``collect_table_tokens`` #679, ``binding._candidate_row_multiset`` #690).

    PATH_WITNESS plain : ('41.3',        True,  '41.3', True)
    PATH_WITNESS dash  : ('41.3--',      False, '41.3', True)
    PATH_WITNESS emdash: ('41.3&mdash;', False, '41.3', True)

A cell carrying a trailing dash-rule run or an HTML dash entity failed
``is_numeric_token`` on the raw text even though ``_normalize_cell`` already
strips exactly that decoration -- so the row was missed as data and
``_grid_to_markdown`` emitted a different table shape (an invented empty
header, one fewer body row) than for the plain value.

Hermetic: pure unit tests against ``reconstruct.py``'s module-private
functions. No ollama, no provider ladder, no fitz page I/O.
"""

from __future__ import annotations

import pytest

from socr.tables.native_verifier import _normalize_cell, is_numeric_token
from socr.tables.reconstruct import _grid_to_markdown, _is_data_row

# --------------------------------------------------------------------------
# The family's closing criterion (GH-773 lesson: pin a PROPERTY, not a grep).
#
#     is_numeric_token(_normalize_cell(wrap(x))) == is_numeric_token(x)
#
# i.e. wrapping a value in trailing decoration and then normalizing it must
# not change whether it reads as numeric, for both numeric and non-numeric x.
# --------------------------------------------------------------------------


def _dash_wrap(x: str) -> str:
    return f"{x}--"


def _mdash_entity_wrap(x: str) -> str:
    return f"{x}&mdash;"


@pytest.mark.parametrize("wrap", [_dash_wrap, _mdash_entity_wrap])
@pytest.mark.parametrize(
    "x",
    [
        "41.3",
        "-1.5",
        "(0.14)",
        "1,204",
        "45%",
        "Firm",  # genuinely non-numeric: property must hold in both directions
        "Nominal",
    ],
)
def test_normalize_then_numeric_matches_the_unwrapped_value(wrap, x: str) -> None:
    assert is_numeric_token(_normalize_cell(wrap(x))) == is_numeric_token(x)


# --------------------------------------------------------------------------
# (a) a decorated numeric cell now makes the row a data row.
# --------------------------------------------------------------------------


def test_trailing_dash_decorated_value_is_a_data_row():
    assert _is_data_row(["Reserves", "41.3--", ""])


def test_mdash_entity_decorated_value_is_a_data_row():
    assert _is_data_row(["Reserves", "41.3&mdash;", ""])


def test_unicode_dash_glyph_decorated_value_is_a_data_row():
    """The literal EM DASH glyph (what ``&mdash;`` decodes to) -- the form a
    native PDF's own typeset dash-rule decoration would carry directly,
    without ever passing through an HTML entity."""
    assert _is_data_row(["Reserves", "41.3\N{EM DASH}", ""])


# --------------------------------------------------------------------------
# (b) both directions: a genuinely non-numeric decorated cell must still NOT
#     make the row a data row. A fix that classifies everything as numeric
#     is as wrong as the bug it closes.
# --------------------------------------------------------------------------


def test_decorated_word_is_still_not_a_data_row():
    assert not _is_data_row(["Firm", "Nominal--", "Real&mdash;"])


def test_bare_dash_run_is_still_not_a_data_row():
    """A cell that is nothing but decoration (no digits underneath) must not
    normalize into an empty string that somehow reads as numeric."""
    assert not _is_data_row(["Note", "--", ""])


# --------------------------------------------------------------------------
# (c) the leading-sign control: `-1.5` must stay numeric AND negative through
#     the new normalize-then-check path. A fix that turned `-1.5` into `1.5`
#     would silently flip a sign on a citation corpus.
# --------------------------------------------------------------------------


def test_leading_minus_sign_survives_normalization_and_stays_a_data_row():
    assert _normalize_cell("-1.5") == "-1.5"
    assert _is_data_row(["Delta", "-1.5", ""])


def test_leading_minus_sign_with_trailing_decoration_also_stays_negative():
    assert _normalize_cell("-1.5--") == "-1.5"
    assert _is_data_row(["Delta", "-1.5--", ""])


def test_leading_unicode_minus_sign_is_not_stripped_as_decoration():
    """U+2212 MINUS SIGN at the front is a sign, not decoration -- only a
    TRAILING occurrence is presentation."""
    assert _normalize_cell("\N{MINUS SIGN}5.2") == "\N{MINUS SIGN}5.2"


# --------------------------------------------------------------------------
# (d) pin the DIFFERENCE at the _grid_to_markdown level, not an absolute
#     markdown blob: the same grid shape with a plain value and with a
#     decorated value must render identically (row 0 stays in the body,
#     since it names an entity and carries a value either way).
# --------------------------------------------------------------------------


def _rows(md: str) -> list[str]:
    return md.splitlines()


def test_plain_and_decorated_grids_produce_the_same_markdown_shape():
    plain = [["Reserves", "41.3", ""], ["Loans", "12.0", ""]]
    decorated = [["Reserves", "41.3--", ""], ["Loans", "12.0", ""]]

    md_plain = _grid_to_markdown(plain)
    md_decorated = _grid_to_markdown(decorated)

    # Same number of rendered lines (header + separator + N body rows) --
    # before the fix, the decorated grid's row 0 read as non-data, so an
    # invented empty header was prepended and row 0 stayed the only "header"
    # while `_is_data_row` disagreed with the plain grid's classification.
    assert len(_rows(md_plain)) == len(_rows(md_decorated))
    # Both keep row 0 in the body (empty header emitted), since row 0 names
    # an entity ("Reserves") and carries a value in both the plain and the
    # decorated grid.
    assert _rows(md_plain)[0] == "|  |  |  |"
    assert _rows(md_decorated)[0] == "|  |  |  |"


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
