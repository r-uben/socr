"""GH-714: A1b's row-shape reconciliation must not refuse a complete TEXT table.

``manifest._row_shape_reconciliation_ok`` (TICKET-A1b, #640) is the twin of
A2's term (b) at a different call site: it derives ``row_shape_min`` from the
candidate's own numeric body rows, so on a text table -- a comparison box whose
cells are sentences carrying zero or one number each -- that minimum collapses
to 1, every native prose band mentioning a figure counts as a table row, and a
complete candidate reads as a massive row shortfall and is dropped from the
corroboration pool.

#714 applies the SAME eligibility rule #703 settled for term (b), reused from
``structure_check._native_page_has_column_lanes`` rather than re-derived: the
reconciliation is meaningful only where the native page shows recurring numeric
column lanes. Where it does not, A1b abstains.

**Abstain maps to True.** A1b's contract is a veto, not a vote -- its sole
caller does ``if not _row_shape_reconciliation_ok(...): continue`` and nothing
reports the outcome separately -- so "no evidence either way" is "do not veto",
which is how the predicate's two pre-existing abstentions already behave. See
its docstring.

Every pin below is a DIFFERENCE: the same page, the same candidate, the same
words, run twice in one process with the gate real and with it monkeypatched
open (which restores the pre-#714 behaviour exactly, since the gate is the only
change). Fixtures are #703's, reused rather than re-invented.
"""

from __future__ import annotations

import pytest

from socr.core.manifest import _row_shape_reconciliation_ok
from socr.tables import structure_check

from test_gh703_text_table_dominance import (  # noqa: I001  (pytest rootdir import)
    BOE_2018_P1_QWEN,
    BOE_2018_PDF,
    TEXT_TABLE_MD,
    TEXT_TABLE_WORDS,
    _boe_p1,
    _sparse_prefix_fixture,
)
from test_structure_check_truncated import (  # noqa: I001  (pytest rootdir import)
    BULLETIN_P2_COMPLETE,
    BULLETIN_P2_TRUNCATED,
    BULLETIN_P3_COMPLETE,
    BULLETIN_P3_TRUNCATED,
    _fixture_words,
)


def _gated_vs_open(
    monkeypatch: pytest.MonkeyPatch, words: list, markdown: str
) -> tuple[bool, bool]:
    """``(gate_open, gate_real)`` for one candidate, measured in one process.

    ``gate_open`` forces ``_native_page_has_column_lanes`` to True, which is
    exactly the pre-#714 predicate.
    """
    real = _row_shape_reconciliation_ok(words, markdown)
    with monkeypatch.context() as m:
        m.setattr(structure_check, "_native_page_has_column_lanes", lambda words: True)
        forced = _row_shape_reconciliation_ok(words, markdown)
    return forced, real


# ---------------------------------------------------------------------------
# Text tables: the defect
# ---------------------------------------------------------------------------


def test_hermetic_text_table_difference_pin(monkeypatch: pytest.MonkeyPatch) -> None:
    """#703's synthetic BoE-shaped comparison box, at A1b's call site.

    Ungated (the shipped behaviour before #714) A1b refuses the complete
    candidate; gated it abstains. Nothing else about the call changed.
    """
    assert structure_check._native_page_has_column_lanes(TEXT_TABLE_WORDS) is False
    assert _gated_vs_open(monkeypatch, TEXT_TABLE_WORDS, TEXT_TABLE_MD) == (False, True)


# ---------------------------------------------------------------------------
# Numeric pages: every existing refusal is preserved
# ---------------------------------------------------------------------------


def test_sparse_prefix_truncation_is_still_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """Astra's #703 counterexample at A1b's call site: a numeric table
    truncated to two legitimately sparse rows. The native page has lanes, so
    the gate is inert and the refusal stands -- gated and ungated alike.
    """
    complete, truncated, words = _sparse_prefix_fixture()
    assert structure_check._native_page_has_column_lanes(words) is True

    assert _gated_vs_open(monkeypatch, words, truncated) == (False, False)
    assert _gated_vs_open(monkeypatch, words, complete) == (True, True)


# The row values below are the ones the two existing ECB fixture tests in
# ``test_structure_check_truncated`` build their native words from, verbatim --
# the real values the COMPLETE candidate's table contains.
_P2_ROWS = [
    [
        "2018",
        "4,404.9",
        "4,489.0",
        "991.4",
        "844.2",
        "2,569.4",
        "5,741.9",
        "6,024.9",
        "682.6",
        "4,356.4",
        "702.9",
    ],
    [
        "2019",
        "4,475.8",
        "4,577.9",
        "967.4",
        "878.0",
        "2,630.4",
        "5,931.1",
        "6,224.0",
        "720.1",
        "4,524.6",
        "686.4",
    ],
    [
        "2020",
        "4,723.6",
        "4,841.3",
        "898.9",
        "1,012.0",
        "2,812.7",
        "6,119.9",
        "6,390.1",
        "700.2",
        "4,725.1",
        "694.6",
    ],
]
_P3_ROWS = [
    [
        "2018",
        "389.2",
        "6,817.4",
        "1,940.0",
        "56.1",
        "2,099.7",
        "2,721.6",
        "1,030.0",
        "460.2",
        "187.0",
        "194.9",
    ],
    [
        "2019",
        "364.2",
        "7,058.9",
        "1,946.1",
        "50.1",
        "2,156.5",
        "2,906.1",
        "1,455.5",
        "452.3",
        "178.9",
        "187.2",
    ],
    [
        "2020",
        "749.0",
        "6,967.4",
        "1,916.7",
        "42.1",
        "1,994.9",
        "3,013.7",
        "1,432.7",
        "539.6",
        "130.1",
        "139.2",
    ],
]


@pytest.mark.parametrize(
    ("truncated_md", "complete_md", "rows"),
    [
        (BULLETIN_P2_TRUNCATED, BULLETIN_P2_COMPLETE, _P2_ROWS),
        (BULLETIN_P3_TRUNCATED, BULLETIN_P3_COMPLETE, _P3_ROWS),
    ],
    ids=["bulletin_p2", "bulletin_p3"],
)
def test_real_ecb_truncation_fixtures_keep_their_a1b_outcome(
    monkeypatch: pytest.MonkeyPatch,
    truncated_md: str,
    complete_md: str,
    rows: list[list[str]],
) -> None:
    """The two real ECB bulletin truncation fixtures: aligned numeric columns,
    so the gate opens and A1b's verdicts are identical to the pre-#714 ones --
    the truncated reading refused, the complete one admitted.
    """
    words = _fixture_words(rows)
    assert structure_check._native_page_has_column_lanes(words) is True

    assert _gated_vs_open(monkeypatch, words, truncated_md) == (False, False)
    assert _gated_vs_open(monkeypatch, words, complete_md) == (True, True)


def test_deleted_row_refusals_are_unchanged_by_the_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """The A1b selection suite's own reproducers (a 20-row grid with two rows
    deleted from either end) reach A1b through an aligned native page, so the
    gate is inert on both edges.
    """
    rows = [(2000 + i, float(100 + i), float(200 + i)) for i in range(20)]

    def _md(subset: list[tuple[int, float, float]]) -> str:
        lines = ["| Year | A | B |", "|---|---|---|"]
        lines += [f"| {y} | {a} | {b} |" for y, a, b in subset]
        return "\n".join(lines) + "\n"

    words: list[tuple] = []
    for i, (year, a, b) in enumerate(rows):
        x = 0.0
        for tok in (str(year), str(a), str(b)):
            words.append((x, 10.0 + i * 20.0, x + 8.0, 20.0 + i * 20.0, tok))
            x += 12.0

    assert structure_check._native_page_has_column_lanes(words) is True
    assert _gated_vs_open(monkeypatch, words, _md(rows)) == (True, True)
    assert _gated_vs_open(monkeypatch, words, _md(rows[:-2])) == (False, False)
    assert _gated_vs_open(monkeypatch, words, _md(rows[2:])) == (False, False)


# ---------------------------------------------------------------------------
# The real BoE page (corpus-skipped)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (BOE_2018_PDF.exists() and BOE_2018_P1_QWEN.exists()),
    reason="BoE census corpus not present on this machine",
)
def test_real_boe_p1_difference_pin(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ticket's page, on the real PDF and the real cached qwen attempt.

    Measured 2026-09-10 and reproduced here: A1b returned False on a candidate
    holding 23/23 of the page's numbers, because its two numeric body rows
    (``('4%',)``, ``('32.',)``) set ``row_shape_min = 1``, at which the native
    page shows 19 "table-shaped rows".
    """
    markdown, words = _boe_p1()

    # grounding: this IS the complete, ladder-accepted candidate
    assert "Table 3.B Monitoring the MPC's key judgements" in markdown
    assert "Unemployment rate to fall to 4% by the end of the year." in markdown
    assert structure_check._native_page_has_column_lanes(words) is False

    assert _gated_vs_open(monkeypatch, words, markdown) == (False, True)
