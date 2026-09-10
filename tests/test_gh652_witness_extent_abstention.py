"""#652 round 5: where a table's extent cannot be measured, the witness abstains.

Two reproduced findings (Astra review at ff5ed74), both in
``tables/row_corroboration.corroboration_witness_words`` and both the same
mistake -- converting an UNPROVEN gap into positive prose attribution:

* the average step between neighbouring anchors is not an upper bound on the
  individual steps inside one table, so an unevenly spaced table's own row
  label was stopped out of the walk and promoted to evidence;
* ``_is_genuine_numeric`` rejects maturity dates, so a names+dates table has
  no anchor at all and the no-anchor branch declared the whole page
  unambiguously prose.

The reviewer's reproducers are kept verbatim in behaviour; the controls around
them pin what must NOT change. Abstaining costs no page text: since #649 the
native prose ships flagged whether or not a model attempt corroborates.
"""

from __future__ import annotations

import pytest

from socr.tables.reconcile import table_syntax_line_indices
from socr.tables.row_corroboration import corroboration_witness_words

from test_gh649_scanned_prose_recovery import MARKER, _page, _ship

LABEL = "Austrian National Bank German Federal Bank Swiss National Bank"
FABRICATION = "ratified quarterly dividends."


def words(line: str, y: float, x: float = 0, height: float = 8) -> list[tuple]:
    """Native words on one printed baseline, one entry per whitespace token."""
    return [
        (x + i * 14, y, x + i * 14 + 12, y + height, tok, 0, 0, 0)
        for i, tok in enumerate(line.split())
    ]


def _attempt(ps) -> str:
    """Ship the page with an attempt whose prose is fabricated FROM the table's
    own labels and whose table half is faithful."""
    ps.best_output.text = (
        LABEL + " " + FABRICATION + "\n\n| Bank | Value |\n| --- | --- |\n| Bank | 250.0 |\n"
    )
    return _ship(ps).text


def test_irregular_table_pitch_does_not_admit_its_label() -> None:
    """A units caption printed tight under its row label makes the label's own
    step (18pt) exceed the pitch its rows average (12pt). The walk stops there,
    but a larger-than-average local step does not establish that the label is
    prose -- and the label is a lone band with no block of its own to measure."""
    ps = _page()
    ps.native_words = []
    for text, y in [
        (LABEL, 0),
        ("Maturity schedule", 18),
        ("250.0", 24),
        ("Reference total", 36),
        ("Annual schedule", 48),
        ("300.0", 60),
    ]:
        ps.native_words += words(text, y)
    for i in range(5):
        ps.native_words += words(
            "Additional explanatory remarks concerning the original document "
            "and its historical context",
            100 + 12 * i,
        )

    assert FABRICATION not in _attempt(ps)


def test_date_only_table_is_not_unambiguously_prose() -> None:
    """No recognised numeric ROW is not no table: the dates are withheld by the
    shipping partition, so the page has unrecognised numeric bands and its
    extent is unmeasured."""
    ps = _page(
        table_rows=[
            LABEL,
            "Maturity schedule",
            "12/04/89",
            "Reference institution",
            "12/05/90",
        ]
    )

    assert FABRICATION not in _attempt(ps)


def test_far_anchor_pair_keeps_intervening_paragraph_without_label_evidence() -> None:
    """The control the abstention must not swallow: a real paragraph printed
    clear of both tables is a run of its own, separated by more than its own
    widest internal step, so it stays evidence -- while the table label above
    the first row does not."""
    ps = _page()
    native = words(LABEL, 0) + words("250.0", 12)
    for i in range(4):
        native += words("The committee discussed monetary policy between the tables", 60 + i * 12)
    native += words("Reference total", 140) + words("300.0", 152)
    ps.native_words = native

    witness, _unresolved = corroboration_witness_words(native)
    tokens = {w[4] for w in witness}

    assert "committee" in tokens
    assert "Austrian" not in tokens
    assert FABRICATION not in _attempt(ps)


@pytest.mark.parametrize(
    "lines,expected",
    [
        (
            ["| A | B |", "| --- | --- |", "| a | 1 |", "| C | D |", "| --- | --- |", "| c | 2 |"],
            set(range(6)),
        ),
        (["| --- | --- |", "| a | 1 |"], {0, 1}),
        (["Before", "| A | B |", "| --x | --- |", "After"], set()),
        (["A | B", "--- | ---", "Tail"], {0, 1}),
    ],
)
def test_table_syntax_boundaries(lines: list[str], expected: set[int]) -> None:
    """#649's accepted syntax bound, re-pinned: a run is a table only when its
    separator says so, and it ends where the separator's block ends."""
    assert table_syntax_line_indices(lines) == expected


def test_two_column_shared_digits_withheld() -> None:
    """The disclosed two-column limitation stays in the safe direction: every
    printed amount is withheld and marked."""
    # Imported inside the test: at module scope pytest would re-collect the
    # imported class and run #649's own suite a second time.
    from test_gh649_scanned_prose_recovery import TestTwoColumnPagesFailSafe

    output = _ship(TestTwoColumnPagesFailSafe()._page())

    assert MARKER in output.text
    for value in ("250.0", "3000.0", "2000.0"):
        assert value not in output.text, value
