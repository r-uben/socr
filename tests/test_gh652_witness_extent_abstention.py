"""#652 rounds 5-9: where a block's role is unproven, the witness abstains.

Two reproduced findings (Astra review at ff5ed74), both in
``tables/row_corroboration.corroboration_witness_words`` and both the same
mistake -- converting an UNPROVEN gap into positive prose attribution:

* the average step between neighbouring anchors is not an upper bound on the
  individual steps inside one table, so an unevenly spaced table's own row
  label was stopped out of the walk and promoted to evidence;
* ``_is_genuine_numeric`` rejects maturity dates, so a names+dates table has
  no anchor at all and the no-anchor branch declared the whole page
  unambiguously prose.

Round 6 (re-review at 341ee68) reproduced the same mistake one level up:
separation proves a BLOCK exists, not that the block is prose. A table label
wrapped over two lines, and a whole date table printed between two recognised
numeric rows, were both admitted as "separated blocks" -- and the date table's
own dates landed in neither the witness nor the unresolved list, so nothing
subtracted them either.

Round 7 (re-review at 0c67d2d) reproduced it once more with the block at the
page edge: a recognised row on ONE side of a block says a table is nearby, not
that the block is prose rather than that table's wrapped label. Requiring BOTH
sides then narrowed the accepted layouts without identifying prose either -- a
wrapped header between two numeric sections passed it, and shipped the same
fabricated sentence.

Round 8 ends the sequence on Astra's ruling: band-gap geometry says where a
block BREAKS and can never say what a block IS, so model-prose salvage is
DISABLED on any scanned page whose shipping partition withholds a numeric band.
Every test below therefore reads as a measurement of refusal and of what #649
ships in its place. The one page that still yields a witness is the pure-prose
scan with no printed numeral anywhere; it is pinned in
``tests/pipeline/test_page_failed_marker_scope.py`` alongside the floor it
exercises.

Round 9 moves the same rule from the helper into the caller, where it was
being asked of the wrong population: ``_prose_corroboration_ok`` filtered the
detected table bboxes away and then partitioned the remainder, so an
incomplete bbox could delete a page's numerals before the check that asks
whether the page has any. A filtered region with no numerals is not a page
with no numerals.

The reviewer's reproducers are kept verbatim in behaviour; the controls around
them pin what must NOT change. Abstaining costs no page text: since #649 the
native prose ships flagged whether or not a model attempt corroborates.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from socr.core.manifest import _prose_corroboration_ok
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


def test_far_anchor_pair_paragraph_is_refused_but_still_ships() -> None:
    """RE-PINNED in round 7, and standing for a wider reason in round 8.

    This shape -- a genuine paragraph printed clear of the tables above and
    below it -- was round 4's positive control: the paragraph entered the
    witness while the table label above the first row did not. Round 7 refused
    it because the nearest attributed band below the paragraph is "Reference
    total", a zero-digit label the walk absorbed rather than a measured row.
    Round 8 refuses it for the reason that outlived every geometric variant:
    the page has withheld numeric bands, so its table's extent is in question
    and no block on it is evidence about the page's prose.

    What the re-pin must show is that abstention is not content loss: the
    paragraph still ships, flagged, from #649's native path, both printed
    amounts stay withheld, and the fabrication built from the table's labels
    is still refused."""
    ps = _page()
    native = words(LABEL, 0) + words("250.0", 12)
    for i in range(4):
        native += words("The committee discussed monetary policy between the tables", 60 + i * 12)
    native += words("Reference total", 140) + words("300.0", 152)
    ps.native_words = native

    witness, unresolved = corroboration_witness_words(native)
    assert witness == []
    assert len(unresolved) == len(native)

    shipped = _attempt(ps)
    assert FABRICATION not in shipped
    assert "The committee discussed monetary policy between the tables" in shipped
    assert MARKER in shipped
    for amount in ("250.0", "300.0"):
        assert amount not in shipped, amount


def test_two_line_label_directly_above_anchor_does_not_vouch() -> None:
    """Round 7 (Astra, re-review at 0c67d2d). Round 6 admitted a run when the
    nearest attributed band on EITHER side was an anchor. Two bank-name bands
    printed at the page edge, 6pt apart, directly above an amount 24pt below
    them, satisfied that: one side had no neighbour at all and the other was a
    real row. The fabricated sentence built from those bank names shipped.

    A numeric row on one side proves a table is nearby, not that the text
    beside it is prose rather than that table's wrapped label -- the words and
    bboxes cannot tell those two readings apart. Round 7 asked for both sides;
    round 8 stopped asking geometry at all. Either way this page refuses, and
    the reproducer keeps its value as the case that showed one side is not
    evidence."""
    ps = _page()
    ps.native_words = []
    for text, y in (
        ("Austrian National Bank", 0),
        ("German Federal Bank Swiss National Bank", 6),
        ("250.0", 30),
        ("Reference total", 42),
        ("Annual schedule", 54),
        ("300.0", 66),
    ):
        ps.native_words += words(text, y, height=4)

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


def test_two_line_table_label_does_not_become_prose() -> None:
    """Round 6, form 1. The stranded label is wrapped over TWO lines, so it has
    an internal step (6pt) and a larger gap below it (18pt) -- everything the
    round-5 helper asked for. But its lines are set closer together than the
    page sets its own (12pt), which is stacked table material, not body prose."""
    ps = _page()
    ps.native_words = []
    for text, y in [
        ("Austrian National Bank", 0),
        ("German Federal Bank Swiss National Bank", 6),
        ("Maturity schedule", 24),
        ("250.0", 30),
        ("Reference total", 42),
        ("Annual schedule", 54),
        ("300.0", 66),
    ]:
        ps.native_words += words(text, y, height=4)

    assert FABRICATION not in _attempt(ps)


def test_date_table_between_numeric_anchors_is_not_prose() -> None:
    """Round 6, form 2. Two recognised anchors sit far above and below, so the
    no-anchor branch never runs; the date table between them is an isolated
    block whose dates the shipping partition withholds but the row matcher does
    not recognise. Its labels must not become evidence -- and its dates must
    land in the unresolved list, not in neither."""
    ps = _page()
    ps.native_words = []
    for text, y in [
        ("250.0", 0),
        ("Austrian National Bank", 40),
        ("12/04/89", 46),
        ("German Federal Bank Swiss National Bank", 52),
        ("12/05/90", 58),
        ("300.0", 120),
    ]:
        ps.native_words += words(text, y, height=4)

    witness, unresolved = corroboration_witness_words(ps.native_words)
    assert not {w[4] for w in witness}
    # The hole this closes: every band is in exactly one list, so the dates are
    # subtracted from the witness rather than silently belonging to neither.
    for date in ("12/04/89", "12/05/90"):
        assert date in {w[4] for w in unresolved}, date

    assert FABRICATION not in _attempt(ps)


def test_subtracted_shared_words_still_ship_from_native() -> None:
    """The disclosed conservative refusal, pinned as a difference rather than
    argued: a genuine attempt whose every word also appears in the withheld
    table rows is refused, and #649 ships that same text from the native layer
    anyway. Abstention costs no page content."""
    ps = _page()
    ps.native_words = []
    for line, y in [
        ("Austrian National Bank 250.0", 0),
        ("German Federal Bank 300.0", 12),
        ("Austrian National Bank", 100),
        ("German Federal Bank", 112),
    ]:
        ps.native_words += words(line, y)
    genuine = "Austrian National Bank\nGerman Federal Bank"
    ps.best_output.text = genuine + "\n\n| Bank | Amount |\n| --- | --- |\n| Bank | 250.0 |\n"

    assert not _prose_corroboration_ok(ps, ps.best_output.text)
    assert genuine in _ship(ps).text


FED_1989_P3 = Path.home() / "Data/socr/fed-sample-2026-09-05/in/fed-1989-11-14-minutes.pdf"
FED_1989_P3_NOUGAT = (
    Path.home()
    / "Data/socr/census-591-recheck/out/fed-1989-11-14-minutes/cache/ef"
    / "ef6b822de4eb1e8546c0fa1d51be70b25e5f0200b4701462000c3c8773ca9a65.json"
)


@pytest.mark.skipif(
    not (FED_1989_P3.exists() and FED_1989_P3_NOUGAT.exists()),
    reason="real fixture not present on this machine",
)
def test_real_fixture_abstains_in_full_without_losing_its_prose() -> None:
    """The ticket's own page, on the real PDF and the real cached attempt.

    RE-PINNED in round 7 and unchanged by round 8. Rounds 5-6 kept 185 of this
    page's 295 words in the witness and the genuine nougat attempt
    corroborated. Round 7's both-sides rule emptied the witness because the
    directive runs to the page bottom; round 8 empties it because the page
    prints numeric bands at all. The witness is empty, every word is
    unresolved, and corroboration is refused.

    That refusal costs this page NOTHING, which is the point of the re-pin and
    was measured, not assumed: the page had already failed its table check, so
    what ships is #649's native recovery either way. The shipped ``PageOutput``
    is byte-identical to the one round 6 produced -- the three directive
    paragraphs flagged, every printed amount withheld -- and only the
    corroboration verdict moved."""
    fitz = pytest.importorskip("fitz")

    with fitz.open(str(FED_1989_P3)) as doc:
        native_words = doc[2].get_text("words")
    nougat_text = json.loads(FED_1989_P3_NOUGAT.read_text())["text"]

    witness, unresolved = corroboration_witness_words(native_words)
    assert witness == []
    assert len(unresolved) == len(native_words)

    ps = _page()
    ps.native_words = native_words
    assert _prose_corroboration_ok(ps, nougat_text) is False

    ps.best_output.text = nougat_text
    shipped = _ship(ps).text
    assert "domestic policy directive:" in shipped
    assert "civilian unemployment rate" in shipped
    assert MARKER in shipped
    for amount in ("1,000.0", "6,000.0", "1,250.0"):
        assert amount not in shipped, amount


def test_a_table_bbox_cannot_hide_the_pages_numerals_from_the_gate() -> None:
    """Round 9 (Astra re-review at b9f45f4), reproduced in the caller.

    The refusal round 8 installed is a claim about a PAGE: no withheld numeric
    band anywhere. ``_prose_corroboration_ok`` used to evaluate it on what
    survived the detected-table bbox filter, so a bbox drawn over this scan's
    two amount bands but not over its bank-name bands deleted every printed
    digit BEFORE the check -- the labels became a full witness and the sentence
    fabricated from them shipped. The gate now reads the whole-page partition,
    the same one #649 ships from.

    Pinned as a difference: the bbox is the only thing that changes between the
    two runs, and the outcome must not."""
    ps = _page()
    ps.native_words = []
    for text, y in [
        ("Austrian National Bank", 0),
        ("German Federal Bank Swiss National Bank", 6),
        ("250.0", 30),
        ("300.0", 66),
    ]:
        ps.native_words += words(text, y, height=4)

    without_bbox = _attempt(ps)
    assert _prose_corroboration_ok(ps, ps.best_output.text) is False
    assert FABRICATION not in without_bbox

    ps.detected_table_count = 1
    ps.detected_table_bboxes = [(-5, 25, 300, 80)]
    with_bbox = _attempt(ps)
    assert _prose_corroboration_ok(ps, ps.best_output.text) is False
    assert FABRICATION not in with_bbox
    assert with_bbox == without_bbox


def test_the_bbox_filter_still_excludes_a_wordless_tables_vocabulary() -> None:
    """Why the bbox exclusion is kept AFTER the gate rather than deleted as
    dead code. A page clears the gate when no band bears a printed numeral --
    which a detected table of purely TEXTUAL cells also satisfies. Its bbox is
    then the only evidence on the page that those bank names are a table, and
    without the filter they are a full witness for a sentence invented from
    them.

    The difference under test is the bbox alone."""
    ps = _page()
    ps.native_words = []
    for text, y in [
        ("Austrian National Bank", 0),
        ("German Federal Bank Swiss National Bank", 6),
    ]:
        ps.native_words += words(text, y, height=4)
    ps.best_output.text = LABEL + " " + FABRICATION

    assert _prose_corroboration_ok(ps, ps.best_output.text) is True

    ps.detected_table_count = 1
    ps.detected_table_bboxes = [(-5, -5, 600, 20)]
    assert _prose_corroboration_ok(ps, ps.best_output.text) is False


def test_a_page_whose_only_numeral_is_its_page_number_keeps_its_prose() -> None:
    """The refusal's blast radius, measured on the page shape most likely to
    trip it by accident. A folio number is a numeric band like any other, so
    the model attempt is refused -- and #649 still ships both genuine
    paragraphs under the unverified-scan banner, with the attempt's invented
    table value withheld. Refusal collapses no page to a bare marker."""
    from socr.core.manifest import SCANNED_PROSE_RECOVERED_FLAG

    lines = [
        "The committee discussed monetary policy and reviewed economic conditions.",
        "Members agreed to continue monitoring developments across financial markets.",
    ]
    ps = _page(prose_above=lines, table_rows=["3"], prose_below=[])
    ps.best_output.text = (
        "\n".join(lines) + "\n\n| Item | Value |\n| --- | --- |\n| Invented | 999 |\n"
    )

    assert not _prose_corroboration_ok(ps, ps.best_output.text)
    shipped = _ship(ps)
    assert shipped.text.startswith(SCANNED_PROSE_RECOVERED_FLAG.format(page_num=1))
    for line in lines:
        assert line in shipped.text
    assert shipped.scanned_prose_recovered
    assert "999" not in shipped.text
