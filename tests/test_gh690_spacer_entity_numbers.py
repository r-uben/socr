"""#690: the #601 spacer filter drops unlabeled rows whose values are
entity-encoded numbers.

``_candidate_row_multiset`` ran ``is_numeric_token`` on the RAW cell.
``_normalize_candidate_rows`` only HTML-unescapes the LABEL cell
(``_normalize_label_cell``, col 0). So an unlabeled row whose only value is
entity-encoded (``&minus;1.5``) or ``&nbsp;``-prefixed (``&nbsp;62.5``) never
enters the numeric multiset, ``_is_spacer_row`` classifies it as a #601
layout spacer, and ``bind()`` silently drops it -- real values lost, not
even surfaced as a contradiction.

Fixed by decoding/dash-stripping value cells with ``_normalize_cell``
(``native_verifier.py``, promoted from ``source_evidence.py``'s #679 fix)
before the numeric check in ``_candidate_row_multiset`` -- the SAME function
``_is_spacer_row`` calls for classification and ``_bind_rows`` calls for the
row-anchoring multiset compare, so one fix closes both.

Pinned at ``bind()`` -- the real filter (``candidate_spacer_rows_dropped``,
``grid.spacer_row_indices``) -- not at an isolated unescape helper, per the
ticket's explicit instruction. Hermetic: synthetic ``page.get_text("words")``
tuples and literal markdown, no PDFs/corpus/provider.
"""

from __future__ import annotations

from collections import Counter

from socr.tables.binding import _candidate_row_multiset, _is_spacer_row, bind, parse_grid

# --------------------------------------------------------------------------
# (a) entity-encoded value rows are KEPT, not classified as #601 spacers --
#     the ticket's core hole, measured at bind()'s real filter.
# --------------------------------------------------------------------------


def test_gh690_html_minus_entity_value_row_is_kept_not_dropped():
    """``|  | &minus;1.5 |`` between two labelled rows must survive:
    ``grid.spacer_row_indices`` must not name it, ``spacer_rows_dropped``
    must be 0, and ``bind()``'s own drop counter
    (``candidate_spacer_rows_dropped``) must agree -- this is bind()'s own
    working copy of the filter, not merely ``parse_grid``'s report."""
    markdown = (
        "| Item | A |\n| --- | --- |\n| Yield | 1.5 |\n|  | &minus;1.5 |\n| Forward | 3.5 |\n"
    )
    grid = parse_grid(markdown)
    assert grid is not None
    assert grid.rows == (("Yield", "1.5"), ("", "&minus;1.5"), ("Forward", "3.5"))
    assert grid.spacer_row_indices == frozenset()
    assert grid.spacer_rows_dropped == 0

    result = bind(
        [
            (100, 60, 140, 70, "1.5", 0, 0, 0),
            (100, 140, 140, 150, "3.5", 0, 0, 0),
        ],
        markdown,
    )
    assert result.candidate_spacer_rows_dropped == 0


def test_gh690_nbsp_prefixed_value_row_is_kept_not_dropped():
    """The wider case the ticket names beyond ``&minus;``: an unlabeled row
    whose value is ``&nbsp;``-decorated (the exact #679 decoration, one gate
    over) drops by the identical mechanism and must be fixed the same way."""
    markdown = "| Item | A |\n| --- | --- |\n|  | &nbsp;62.5 |\n| Total | 62.5 |\n"
    grid = parse_grid(markdown)
    assert grid is not None
    assert grid.spacer_row_indices == frozenset()
    assert grid.spacer_rows_dropped == 0

    result = bind([(100, 140, 140, 150, "62.5", 0, 0, 0)], markdown)
    assert result.candidate_spacer_rows_dropped == 0


def test_gh690_candidate_row_multiset_decodes_entity_before_numeric_check():
    """Unit-level pin on the exact function both ``_is_spacer_row`` (#601
    classification) and ``_bind_rows`` (row-anchoring multiset compare) call
    -- one fix, two call sites, both closed by construction."""
    assert _candidate_row_multiset(("", "&minus;1.5")) == Counter({"-1.5": 1})
    assert _candidate_row_multiset(("", "&nbsp;62.5")) == Counter({"62.5": 1})
    assert not _is_spacer_row(("", "&minus;1.5"))
    assert not _is_spacer_row(("", "&nbsp;62.5"))


# --------------------------------------------------------------------------
# (b) a GENUINE layout spacer must still classify as a spacer -- the
#     guarantee this fix must not break, measured in both directions.
# --------------------------------------------------------------------------


def test_gh690_control_wholly_empty_row_still_a_spacer():
    assert _is_spacer_row(("", "")) is True
    assert _candidate_row_multiset(("", "")) == Counter()


def test_gh690_control_markdown_rule_row_still_a_spacer():
    """``_normalize_cell`` strips a trailing dash RUN, so a printed rule
    cell (``'---'``) normalises to the empty string -- the intended outcome,
    pinned explicitly rather than left as an accident of the fix."""
    assert _is_spacer_row(("", "---")) is True
    assert _candidate_row_multiset(("", "---")) == Counter()


def test_gh690_control_punctuation_only_row_still_a_spacer():
    assert _is_spacer_row(("", "***")) is True
    assert _candidate_row_multiset(("", "***")) == Counter()


def test_gh690_control_genuine_spacer_row_still_dropped_by_bind():
    markdown = "| Item | A |\n| --- | --- |\n| Yield | 1.5 |\n|  |  |\n| Forward | 3.5 |\n"
    grid = parse_grid(markdown)
    assert grid is not None
    assert grid.spacer_row_indices == frozenset({1})
    assert grid.spacer_rows_dropped == 1

    result = bind(
        [
            (100, 60, 140, 70, "1.5", 0, 0, 0),
            (100, 140, 140, 150, "3.5", 0, 0, 0),
        ],
        markdown,
    )
    assert result.candidate_spacer_rows_dropped == 1
