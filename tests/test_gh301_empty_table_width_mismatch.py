"""GH-301: a width mismatch must not excuse an empty table from the content term.

#299 closed the GH-190 fixture, but `table_content_defect` still required
`len(delimiter) == len(header)` before it looked at the body. Two spellings
walked through that gate and shipped SUCCESS with no rows:

- blank header, narrower delimiter, empty body
- populated header, narrower delimiter, empty body matching the delimiter

Nothing downstream caught them either: emission skips a blank header and a
ragged `content_widths`, and `_parse_grid` drops the blank body, so
`table_output_defect` never saw either one.

A width mismatch is a SHAPE defect and keeps its existing owner. It is not a
reason to stop asking whether the table has any content at all -- body width
was already ignored here for exactly that reason.
"""

from __future__ import annotations

import pytest

from socr.tables.reconcile import TABLE_CONTENT_EMPTY, table_content_defect

BLANK_HEADER_NARROW_DELIM = "|  |  |  |\n| --- | --- |\n|  |  |  |\n"
POPULATED_HEADER_NARROW_DELIM = "| A | B | C |\n| --- | --- |\n|  |  |\n"


@pytest.mark.parametrize(
    ("name", "markdown"),
    [
        ("blank header", BLANK_HEADER_NARROW_DELIM),
        ("populated header", POPULATED_HEADER_NARROW_DELIM),
    ],
)
def test_an_empty_table_is_caught_despite_a_narrower_delimiter(name: str, markdown: str) -> None:
    assert table_content_defect(markdown) == TABLE_CONTENT_EMPTY, (
        f"{name}: an empty table escaped the content term on a width mismatch"
    )


def test_a_mismatched_width_table_WITH_content_is_still_clean() -> None:
    """The control that stops this becoming a width rule.

    Dropping the width gate must widen only the EMPTY case. A table whose
    delimiter is narrower but whose body carries real values is a shape defect
    for its existing owner to report -- this term must stay silent on it, or
    the fix has turned the content grammar into a width check.
    """
    assert table_content_defect("| A | B | C |\n| --- | --- |\n| 1 | 2 |\n") == ""


def test_the_equal_width_empty_case_still_works() -> None:
    """Regression guard: the case #299 already caught must not be lost."""
    assert table_content_defect("| A | B |\n| --- | --- |\n|  |  |\n") == TABLE_CONTENT_EMPTY
