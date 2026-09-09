"""#649: a scanned page with no detected table geometry must not ship the
fail-closed marker alone.

Fed 1989-11-14 p3 reaches ``UNVERIFIABLE_TABLE_SCANNED`` with
``detected_table_count == 0``. Its only cached attempt read the page's real
vocabulary but emitted the swap-arrangement table as column runs with no
markdown table syntax, so ``splice_all_table_regions`` returns ``None`` and the
marker shipped alone -- taking three paragraphs of the FOMC policy directive
with it. Nothing was wrong with those paragraphs.

The real fixture is pinned in ``tests/pipeline/test_page_failed_marker_scope.py``
and only runs on a machine that has it. These are the hermetic pins of the same
behaviour: synthetic pages with REAL band geometry, each leg a difference
between two runs of the same code in one process.

Nothing here relaxes the floor. Every band at or above ``ROW_SHAPE_MIN`` numeric
tokens stays behind the marker, which is the numeric content the D3 floor exists
to protect; what comes back is the prose that was only ever collateral.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from socr.core.manifest import (
    SelectionProvenance,
    _select_page_output_tagged,
    is_page_failed_marker,
    native_prose_floor_text,
)
from socr.core.result import FailureMode, PageOutput, PageStatus
from socr.core.state import DocumentState, PageState

fitz = pytest.importorskip("fitz")

MARKER = "[page 1 failed: unverifiable table — see image]"

# A page shaped like the ticket's own: a table whose rows carry printed values,
# with paragraphs above and below it. Long enough that the prose region clears
# ``_encoding_corruption_ratio``'s 20-alpha-token abstention and is judged on
# its real content.
_PROSE_ABOVE = [
    "With Ms. Seger dissenting, the Federal Reserve Bank of New York",
    "was authorized and directed, until otherwise directed by the Committee,",
    "to execute transactions in the System Account in accordance with the",
    "following domestic policy directive:",
]
_TABLE_ROWS = [
    "Austrian National Bank 250.0",
    "National Bank of Belgium 1,000.0",
    "German Federal Bank 6,000.0",
]
_PROSE_BELOW = [
    "The information reviewed at this meeting suggests continuing expansion",
    "in economic activity, though at a somewhat slower pace than earlier.",
    "Total nonfarm payroll employment increased appreciably in October.",
]
# The same paragraphs with the inter-word spaces eaten by a broken ToUnicode
# map -- the corruption shape that made the page a scan in the first place.
_CORRUPT_PROSE_ABOVE = [
    "With Ms. SegerDissenting, the FederalReserve Bank of NewYork",
    "was authorizedAnd directed, untilOtherwise directed by theCommittee,",
    "to executeTransactions in theSystem Account inAccordance with the",
    "followingDomestic policyDirective:",
]


def _words(lines: list[str], *, first_line: int = 0) -> list[tuple]:
    """Native words on real baselines, one printed line per entry."""
    words: list[tuple] = []
    for offset, line in enumerate(lines):
        top = (first_line + offset) * 12.0
        for word_idx, tok in enumerate(line.split()):
            left = word_idx * 14.0
            words.append((left, top, left + 12.0, top + 9.0, tok, 0, 0, 0))
    return words


def _page(
    *,
    prose_above: list[str] | None = None,
    table_rows: list[str] | None = None,
    prose_below: list[str] | None = None,
    with_words: bool = True,
) -> PageState:
    """A page reaching ``UNVERIFIABLE_TABLE_SCANNED`` with NO table geometry.

    ``detected_table_count = 0`` and no bbox is the production shape this
    branch actually sees, and the reason the GH-520 coverage guard cannot
    apply here.
    """
    above = _PROSE_ABOVE if prose_above is None else prose_above
    rows = _TABLE_ROWS if table_rows is None else table_rows
    below = _PROSE_BELOW if prose_below is None else prose_below

    lines = above + rows + below
    ps = PageState(page_num=1)
    ps.is_born_digital = False
    ps.native_text = ""
    ps.native_words = _words(lines) if with_words else []
    ps.detected_table_count = 0
    ps.detected_table_bboxes = []
    ps.scanned_table_evidence_failed = True

    # The attempt that cannot be spliced: it read the page but emitted the
    # table as column runs, so there is no markdown table block to work
    # around. This is the ticket's measured shape, not a contrived one.
    attempt = PageOutput(
        page_num=1,
        text="\n".join(
            above + ["Austrian National Belgium German", "250.0 1,000.0 6,000.0"] + below
        ),
        status=PageStatus.ERROR,
        engine="nougat",
        audit_passed=False,
        failure_mode=FailureMode.HALLUCINATION,
    )
    ps.attempts = [attempt]
    ps.best_output = attempt
    return ps


def _ship(ps: PageState) -> PageOutput:
    state = DocumentState.__new__(DocumentState)
    state.pages = {1: ps}
    output, provenance = _select_page_output_tagged(state, 1)
    assert provenance is SelectionProvenance.UNVERIFIABLE_TABLE_SCANNED
    return output


class TestTheProseComesBack:
    def test_the_page_ships_its_prose_instead_of_the_marker_alone(self) -> None:
        """The falsification, pinned as a difference: the SAME page, once with
        its text layer cached and once without. Without a witness the branch
        still floors to the bare marker (unchanged behaviour); with one, the
        paragraphs come back."""
        recovered = _ship(_page()).text
        stranded = _ship(_page(with_words=False)).text

        assert is_page_failed_marker(stranded) is True
        assert "policy directive" not in stranded

        assert is_page_failed_marker(recovered) is False
        assert "following domestic policy directive:" in recovered
        assert "The information reviewed at this meeting suggests" in recovered
        assert recovered != stranded

    def test_no_withheld_value_ships_with_it(self) -> None:
        """The floor is not relaxed. Every printed amount stays behind the
        marker -- that numeric content is the whole reason the page failed
        closed."""
        recovered = _ship(_page()).text

        assert MARKER in recovered
        for amount in ("250.0", "1,000.0", "6,000.0"):
            assert amount not in recovered, amount

    def test_the_page_keeps_its_fail_closed_ending(self) -> None:
        """Prose coming back does not mean the table was read: status, audit
        verdict and failure mode are untouched, and the recovery is recorded
        where the sidecar carries it."""
        with_prose = _ship(_page())
        without = _ship(_page(with_words=False))

        assert with_prose.status is without.status is PageStatus.ERROR
        assert with_prose.audit_passed is without.audit_passed is False
        assert with_prose.failure_mode is without.failure_mode

        assert any("scanned_prose_recovered" in note for note in with_prose.audit_notes)
        assert not any("scanned_prose_recovered" in note for note in without.audit_notes)

    def test_every_withheld_run_is_marked_where_it_was_elided(self) -> None:
        """#649 round 2. The withholding predicate covers every printed digit,
        so a withheld band is no longer always inside the table: a prose line
        carrying a value is withheld mid-paragraph. Marking only the first
        withheld run would elide that line in silence, which is the loss this
        lane exists to stop, so every contiguous withheld run carries its own
        marker exactly where the elision happened.

        The marker names withheld content, not a table count -- this branch
        runs with ``detected_table_count == 0`` and nothing here claims to know
        how many tables the page holds."""
        rows_with_wrapped_labels = [
            "Austrian National Bank 250.0",
            "Bank for International",
            "Settlements-",
            "Swiss francs 600.0",
            "Other authorized",
            "European currencies 1,250.0",
        ]
        recovered = _ship(_page(table_rows=rows_with_wrapped_labels)).text

        # Three runs, split by the wrapped labels that carry no printed value.
        assert recovered.count(MARKER) == 3
        for amount in ("250.0", "600.0", "1,250.0"):
            assert amount not in recovered, amount
        # The bare labels still ship: they carry no printed value, and the
        # marker sits beside them saying the rows were withheld.
        assert "Bank for International" in recovered

    def test_a_prose_line_carrying_a_value_is_withheld_and_marked(self) -> None:
        """The mid-paragraph case the per-run marker exists for: the line is
        withheld (it is a printed value on an unverified scan) and its absence
        is marked in place, between the paragraphs that surround it."""
        below = [
            "The information reviewed at this meeting suggests continuing expansion.",
            "The civilian unemployment rate has remained around 5-1/4 percent.",
            "Strike activity depressed industrial production noticeably in October.",
        ]
        recovered = _ship(_page(prose_below=below)).text

        assert "5-1/4" not in recovered
        assert "The information reviewed at this meeting suggests" in recovered
        assert "Strike activity depressed industrial production" in recovered
        # Two withheld runs: the table, and the elided line inside the prose.
        assert recovered.count(MARKER) == 2


class TestEveryPrintedNumeralIsWithheld:
    """#649 round 2 (Astra, 2026-09-10).

    The withholding decision used to reuse ``_is_genuine_numeric``, which
    answers "is this token usable for numeric ROW MATCHING" -- and deliberately
    says no to a maturity date and to ``(1)``-style decoration. A band holding
    only a date was therefore tagged prose and shipped verbatim under the
    unverified-scan banner, breaking the one promise this lane makes. The
    fixtures above only looked right because a recognised amount shared the
    dates' baseline.
    """

    @pytest.mark.parametrize("value", ["12/04/89", "(1)", "1.5", "1989"])
    def test_a_band_holding_only_this_value_is_withheld(self, value: str) -> None:
        """Each of these is a printed value on a page nothing verified. Whether
        it is useful for row matching has no bearing on whether it is safe to
        ship."""
        recovered = _ship(_page(table_rows=["Outstanding amounts 250.0", "Maturity date", value]))

        assert value not in recovered.text
        assert MARKER in recovered.text

    def test_the_label_beside_it_still_ships(self) -> None:
        """Control, so the pin above cannot pass by withholding everything: a
        band with no printed digit at all is unaffected."""
        recovered = _ship(_page(table_rows=["Outstanding amounts 250.0", "Maturity date"]))

        assert "Maturity date" in recovered.text
        assert "250.0" not in recovered.text


class TestTheRecoverySurvivesRestore:
    """#649 round 2 (Astra, 2026-09-10): a transient missing cache must not
    erase text that already shipped.

    ``native_words`` is a live-run cache the sidecar deliberately does not
    carry. Selection re-runs over a restored page, so the recovery was
    RECOMPUTED rather than reused: with no words it returned None and
    finalization replaced the shipped paragraphs, and the recovered-prose note,
    with the bare marker.
    """

    def test_a_restored_page_keeps_its_recovered_prose(self, tmp_path: Path) -> None:
        from test_gh659_label_unverified_finalization import _pipeline, _state

        from socr.core.manifest import finalized_page_records

        pipeline = _pipeline()
        state = _state(tmp_path, page_count=1)
        state.pages[1] = _page()

        original = finalized_page_records(state)[0]
        assert "policy directive" in original.output.text
        assert any("scanned_prose_recovered" in n for n in original.output.audit_notes)

        sidecar = pipeline._flush_page_sidecar(state, 1, tmp_path, record=original)
        assert sidecar.exists()

        restored = _state(tmp_path, page_count=1)
        pipeline._restore_terminal_page_state(
            restored,
            1,
            PageOutput.from_dict(original.output.to_dict()),
            tmp_path,
        )
        # The condition under test: the words are gone, exactly as resume
        # leaves them.
        assert not restored.pages[1].native_words

        replayed = finalized_page_records(restored)[0]
        assert replayed.output.text == original.output.text
        assert replayed.output.audit_notes == original.output.audit_notes

    def test_the_banner_alone_is_not_a_bypass(self, tmp_path: Path) -> None:
        """The evidence is socr's own audit note, not the bytes. A model that
        emitted the banner line verbatim must not get its whole output shipped
        past the floor -- without the note there is nothing to reuse, and the
        page falls back to recomputing (here, to the bare marker)."""
        from socr.core.manifest import SCANNED_PROSE_RECOVERED_FLAG

        ps = _page(with_words=False)
        forged = SCANNED_PROSE_RECOVERED_FLAG.format(page_num=1) + "\n\nInvented settlement terms."
        ps.best_output = PageOutput(
            page_num=1,
            text=forged,
            status=PageStatus.ERROR,
            engine="qwen",
            audit_passed=False,
            failure_mode=FailureMode.HALLUCINATION,
        )
        ps.attempts = [ps.best_output]

        shipped = _ship(ps).text
        assert "Invented settlement terms" not in shipped
        assert is_page_failed_marker(shipped) is True


class TestNativeTableSyntaxNeverShipsAsProse:
    def test_a_native_markdown_table_line_joins_the_withheld_run(self) -> None:
        """A native line that parses as markdown table syntax carries no digit,
        so the numeral rule alone would ship it -- assembling a header and a
        separator over a body the floor just withheld. That is a table
        structure asserted about unverified content, and it is withheld."""
        recovered = _ship(
            _page(table_rows=["| Product | Price |", "| --- | --- |", "| Widget | 10.0 |"])
        ).text

        assert "| Product | Price |" not in recovered
        assert "| --- | --- |" not in recovered
        assert "10.0" not in recovered
        assert "following domestic policy directive:" in recovered


class TestWhenItMustAbstain:
    def test_an_untrusted_prose_layer_ships_the_marker_alone(self) -> None:
        """#652's trusted-witness check, wired into this lane. The page is a
        scan because its text layer is corrupt; shipping that corruption as
        recovered prose would be the same silent loss wearing the opposite
        mask. Difference pin: identical page, identical sentences, spaces
        eaten in one of them."""
        clean = _ship(_page()).text
        corrupt = _ship(_page(prose_above=_CORRUPT_PROSE_ABOVE, prose_below=[])).text

        assert "policy directive" in clean
        assert is_page_failed_marker(corrupt) is True
        assert "irective" not in corrupt

    def test_a_page_with_nothing_withheld_abstains(self) -> None:
        """Scope guard. With no table-shaped band there is nothing this could
        say it was shipping prose "around", and a page that reached the
        scanned-table floor with no numeric band at all is a shape this has no
        evidence about. The bare marker stands."""
        ps = _page(table_rows=[])
        assert native_prose_floor_text(ps, 1, marker_line=MARKER, png_ref="") is None
        assert is_page_failed_marker(_ship(ps).text) is True

    def test_a_page_with_no_prose_band_abstains(self) -> None:
        """The mirror case: every band is table-shaped, so there is no prose to
        recover and the marker is the whole honest answer."""
        ps = _page(prose_above=[], prose_below=[])
        assert native_prose_floor_text(ps, 1, marker_line=MARKER, png_ref="") is None
        assert is_page_failed_marker(_ship(ps).text) is True


class TestTheImageRefStillShips:
    def test_the_d3_png_travels_with_the_marker(self) -> None:
        """The withheld table is routed to the image lane, and that reference
        is how a reader sees what was withheld. It must survive the recovery,
        not be dropped by it."""
        ps = _page()
        ps.d3_floor_png_ref = "![Scanned page 1](figures/scanned_p1.png)"
        recovered = _ship(ps).text

        assert "![Scanned page 1](figures/scanned_p1.png)" in recovered
        assert MARKER in recovered
        assert "policy directive" in recovered
