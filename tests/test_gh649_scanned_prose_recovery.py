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

import json
import shutil
import subprocess

from pathlib import Path

import pytest

from socr.core.manifest import (
    SCANNED_NATIVE_TEXT_FLAG,
    SCANNED_PROSE_RECOVERED_FLAG,
    SelectionProvenance,
    _escaped_native_line,
    _select_page_output_tagged,
    is_page_failed_marker,
    native_prose_floor_text,
)
from socr.core.result import FailureMode, PageOutput, PageStatus
from socr.core.state import DocumentState, PageState

fitz = pytest.importorskip("fitz")

#: Resolved at import, deliberately. ``tests/conftest.py`` patches
#: ``shutil.which`` on the module object for every test, so asking inside a
#: test body answers ``None`` on a machine that has node and silently skips.
_NODE = shutil.which("node")

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

    def test_a_model_note_quoting_the_credential_cannot_forge_a_recovery(
        self, tmp_path: Path
    ) -> None:
        """#649 round 3 (Astra), and the reason the credential is a TYPED field
        rather than a string in ``audit_notes``.

        ``orchestrator`` appends ``dual-pass {action}: {summary}`` for every
        reconciliation disagreement, and that summary quotes the page and crop
        cell values verbatim -- model-controlled text. An attempt carrying the
        banner at the top and the recovery note inside a table cell therefore
        put its own credential into ``audit_notes`` through a real code path,
        and the substring check accepted it: the invented sentence and its
        invented numbers shipped as an already-finalized recovery.

        Everything here is real except the unrelated PDF-backed crop-repair
        eligibility check: real reconciliation, the real summary, the real note
        append, real selection. No note is injected by the test."""
        from types import SimpleNamespace
        from unittest.mock import patch

        from test_gh659_label_unverified_finalization import _pipeline, _state

        from socr.core.manifest import (
            SCANNED_PROSE_RECOVERED_FLAG,
            SCANNED_PROSE_RECOVERED_NOTE,
        )

        ps = _page(with_words=False)
        ps.best_output.text = (
            SCANNED_PROSE_RECOVERED_FLAG.format(page_num=1)
            + "\n\nInvented settlement terms.\n\n| Label | Value |\n| --- | --- |\n| "
            + SCANNED_PROSE_RECOVERED_NOTE
            + " | 10.0 |\n"
        )
        state = _state(tmp_path, page_count=1)
        state.pages[1] = ps

        pipeline = _pipeline()
        pipeline.config.auto_patch_tables = False
        crop = SimpleNamespace(
            markdown="| Label | Value |\n| --- | --- |\n| Different label | 10.0 |\n",
            source="ruled",
        )
        with patch("socr.tables.crop_repair.page_needs_crop_repair_fallback", return_value=False):
            pipeline._reread_page_tables(state, 1, [crop], extractor=object())

        # The fixture premise: the model's own text really did reach a note.
        assert any(SCANNED_PROSE_RECOVERED_NOTE in n for n in ps.best_output.audit_notes)
        # And it buys nothing.
        assert ps.best_output.scanned_prose_recovered is False
        shipped = _ship(ps).text
        assert "Invented settlement terms." not in shipped
        assert "10.0" not in shipped

    def test_the_banner_alone_is_not_a_bypass(self, tmp_path: Path) -> None:
        """The second factor on its own. A model that emitted the banner line
        verbatim must not get its whole output shipped past the floor -- with
        no credential there is nothing to reuse, and the page falls back to
        recomputing (here, to the bare marker)."""
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

    def test_an_ordinary_sentence_holding_a_pipe_still_ships(self) -> None:
        """#649 round 3 (Astra). Round 2 asked ``_is_table_line``, whose regex
        accepts any line containing a pipe, so this numeral-free sentence was
        withheld -- avoidable prose loss, not safe over-exclusion. A lone pipe
        is not a table."""
        sentence = "The symbol | separates the alternatives in this paragraph."
        below = [
            "The committee reviewed the schedule in detail.",
            sentence,
            "No dissenting votes were recorded at the meeting.",
        ]
        recovered = _ship(_page(prose_below=below)).text

        # #712: the pipe ships escaped, so it stays a printed character instead
        # of a cell boundary. The point of the test is that the LINE is not
        # withheld, which is what the escaped form still shows.
        assert _escaped_native_line(sentence) in recovered
        # Only the table's own run is marked; nothing in the prose was elided.
        assert recovered.count(MARKER) == 1

    @pytest.mark.parametrize("side", ["before", "after"])
    def test_a_pipe_sentence_beside_a_real_table_survives(self, side: str) -> None:
        """#649 round 4 (Astra). Round 3 asked ``find_table_blocks``, which
        groups consecutive pipe-bearing lines but knows nothing about where the
        table inside that run BEGINS, so the same sentence still vanished when
        it sat directly before or after a genuine header/separator/body block.
        The boundaries come from the table's own structure now."""
        sentence = "The symbol | separates alternatives in this paragraph."
        table = ["| Header | Amount |", "| --- | --- |", "| Bank | 250.0 |"]
        rows = [sentence, *table] if side == "before" else [*table, sentence]

        recovered = _ship(_page(table_rows=rows)).text

        assert _escaped_native_line(sentence) in recovered
        assert "| Header | Amount |" not in recovered
        assert "250.0" not in recovered

    def test_a_table_between_two_ordinary_lines_keeps_both(self) -> None:
        """Control: the neighbours survive and the table does not, with one
        marker where it went."""
        recovered = _ship(
            _page(
                table_rows=[
                    "Introductory words",
                    "| Header | Amount |",
                    "| --- | --- |",
                    "| Bank | 250.0 |",
                    "Concluding words",
                ]
            )
        ).text

        assert "Introductory words" in recovered
        assert "Concluding words" in recovered
        assert "| Header" not in recovered
        assert recovered.count(MARKER) == 1

    def test_pipe_lines_with_no_separator_are_not_a_table(self) -> None:
        """The separator is the one element that cannot be mistaken for prose,
        so a run without one stays prose whatever its pipes suggest."""
        rows = [
            "Outstanding amounts 250.0",
            "Options were listed as accept | defer in the minutes.",
            "The chair noted accept | defer had been discussed before.",
        ]
        recovered = _ship(_page(table_rows=rows)).text

        assert _escaped_native_line("accept | defer in the minutes.") in recovered
        assert "had been discussed before." in recovered

    def test_a_printed_dash_rule_still_ships(self) -> None:
        """Control from the same probe: a printed rule carries no pipe and no
        digit, and is ordinary page furniture."""
        assert "--------" in _ship(_page(table_rows=["amount 10.0", "--------"])).text


class TestTwoColumnPagesFailSafe:
    """Bands are clustered by y across the full page width, so on a two-column
    page a left-column prose line and a right-column table row share a band.

    Raised as an open question in review rather than a finding, and measured
    here rather than argued: the failure is entirely in the safe direction. The
    shared band carries the right column's digits, so it is withheld and
    marked; no printed value reaches the page, and the witness treats the same
    band as table-attributed so it cannot vouch for a fabrication either. What
    it costs is the left column's prose, withheld behind the marker instead of
    shipped. Column-aware banding would recover that text; nothing here leaks
    without it, which is why this is a limitation and not a hole.
    """

    _LEFT = [
        "The committee reviewed the swap arrangements at length",
        "and authorized their renewal for a further twelve months",
        "with no dissenting votes recorded in the minutes today",
    ]
    _RIGHT = [
        "Austrian National Bank 250.0",
        "Bank of England 3000.0",
        "Bank of France 2000.0",
    ]

    def _page(self) -> PageState:
        words: list[tuple] = []
        for idx, (left, right) in enumerate(zip(self._LEFT, self._RIGHT)):
            top = idx * 12.0
            for word_idx, tok in enumerate(left.split()):
                x = word_idx * 14.0
                words.append((x, top, x + 12.0, top + 8.0, tok, 0, 0, 0))
            for word_idx, tok in enumerate(right.split()):
                x = 400.0 + word_idx * 14.0
                words.append((x, top, x + 12.0, top + 8.0, tok, 0, 0, 0))
        ps = _page(with_words=False)
        ps.native_words = words
        return ps

    def test_no_printed_value_reaches_the_page(self) -> None:
        recovered = _ship(self._page()).text
        for value in ("250.0", "3000.0", "2000.0"):
            assert value not in recovered, value
        assert MARKER in recovered

    def test_a_shared_band_cannot_vouch_for_a_fabrication(self) -> None:
        """Re-scoped in #652 round 10 to what ships. The corroboration guard
        this used to call is gone -- no attempt's prose leaves this branch at
        all -- so the pin is on the bytes, where it was always the point."""
        ps = self._page()
        ps.best_output.text = (
            "The committee reviewed the swap arrangements ratified quarterly dividends."
        )
        assert "ratified quarterly dividends" not in _ship(ps).text


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

    def test_a_page_with_nothing_withheld_ships_its_native_text(self) -> None:
        """#652 round 11 (Astra's ruling, reproducer prose10). A scan whose
        native layer prints no numeral has no band to withhold, and until this
        round that abstained -- so two clean policy paragraphs collapsed to the
        bare marker with the page's own trusted text sitting unread beside it.
        Round 10 removed the model-prose route from this branch, which is what
        turned the abstention into a loss.

        What ships is the page's own lines, nothing else: the marker and the
        image stay, once, because the table that failed here was never
        verified."""
        ps = _page(table_rows=[])
        shipped = _ship(ps)

        assert shipped.status is PageStatus.ERROR
        assert shipped.audit_passed is False
        assert shipped.scanned_prose_recovered is True
        assert is_page_failed_marker(shipped.text) is False
        for line in _PROSE_ABOVE + _PROSE_BELOW:
            assert line in shipped.text, line
        # The notice survives, once, and claims nothing about a verified table.
        assert shipped.text.count(MARKER) == 1
        assert shipped.text.startswith(SCANNED_NATIVE_TEXT_FLAG.format(page_num=1))

    def test_the_pages_own_characters_never_become_structure(self) -> None:
        """#652 round 12 (Astra's prose11 reproduction), through the installed
        renderer rather than through a claim about escaping.

        Round 11 escaped only the pipe, and two losses came straight back: a
        native ``<!--`` line turned the sentence after it into an HTML comment
        that a reading consumer never sees, and ``# Literal heading marker``
        became an ``<h1>``. The page authored neither construct. Both lines --
        and every other active character on the page -- must reach a reader as
        the characters that were printed."""
        markdown_it = pytest.importorskip("markdown_it")
        actives = [
            "<!--",
            "The committee retained the original mandate.",
            "# Literal heading marker",
            "> quoted",
            "- bulleted",
            "*emphasis* and `code` and [link](x) and A & B",
            "~~struck~~",
        ]
        shipped = _ship(_page(table_rows=[], prose_below=actives)).text
        rendered = markdown_it.MarkdownIt().render(shipped)

        assert "<!--" not in rendered
        assert "The committee retained the original mandate." in rendered
        for tag in ("<h1>", "<blockquote>", "<li>", "<em>", "<code>", "<a href", "<s>"):
            assert tag not in rendered, tag
        # The page's own characters, as characters: what a reader sees is the
        # line that was printed, with the renderer's entity forms for < and &.
        assert "# Literal heading marker" in rendered
        assert "*emphasis* and `code` and [link](x) and A &amp; B" in rendered

    def test_socrs_own_banner_and_notice_stay_outside_the_literal_body(self) -> None:
        """The escaping covers the PAGE's characters. socr's banner, the
        table-unverified notice and the image reference are socr's own
        markdown and must not be mangled into literal text -- an escaped image
        reference would stop being an image."""
        ps = _page(table_rows=[])
        ps.d3_floor_png_ref = "![Scanned page 1](figures/scanned_p1.png)"
        shipped = _ship(ps).text

        assert shipped.startswith(SCANNED_NATIVE_TEXT_FLAG.format(page_num=1))
        assert MARKER in shipped
        assert "![Scanned page 1](figures/scanned_p1.png)" in shipped
        assert "\\!" not in shipped

    def test_a_text_only_table_ships_as_lines_not_as_a_grid(self) -> None:
        """No grid is reconstructed and no cell inferred. A text-only table's
        rows are literal baseline lines here, and a printed pipe is escaped so
        the recovered page cannot assemble into a markdown table nothing on it
        verified."""
        ps = _page(
            table_rows=[],
            prose_above=_PROSE_ABOVE,
            prose_below=["| Austrian National Bank | Member |"],
        )
        shipped = _ship(ps).text

        assert "Austrian National Bank" in shipped
        assert "| Austrian National Bank | Member |" not in shipped
        assert "\\| Austrian National Bank \\| Member \\|" in shipped

    def test_an_untrusted_layer_with_nothing_withheld_still_floors(self) -> None:
        """The trust check is the same one, applied to the same layer. A
        corrupt no-numeral scan is not published; the bare marker stands.
        Difference pin: identical page, spaces eaten in one of them."""
        clean = _ship(_page(table_rows=[])).text
        corrupt = _ship(_page(table_rows=[], prose_above=_CORRUPT_PROSE_ABOVE, prose_below=[])).text

        assert is_page_failed_marker(clean) is False
        assert is_page_failed_marker(corrupt) is True
        assert "irective" not in corrupt

    def test_no_native_words_with_nothing_withheld_still_floors(self) -> None:
        """No text layer, no recovery -- the other unchanged abstention."""
        ps = _page(table_rows=[], with_words=False)
        assert native_prose_floor_text(ps, 1, marker_line=MARKER, png_ref="") is None
        assert is_page_failed_marker(_ship(ps).text) is True

    def test_a_text_only_table_cannot_put_the_models_wording_on_the_page(self) -> None:
        """The complementary control from the same reproducer. Recovering the
        page's own lines is not a route back for the attempt's: the invented
        sentence the deleted corroboration guard used to authorise is still
        refused, because nothing here consults the attempt at all."""
        ps = _page(
            table_rows=[],
            prose_above=["Austrian National Bank Member"],
            prose_below=["German Federal Bank Swiss National Bank Member"],
        )
        ps.best_output.text = (
            "Austrian National Bank German Federal Bank Swiss National Bank "
            "ratified quarterly dividends.\n\n| Bank | Status |\n| --- | --- |\n"
            "| Austrian National Bank | Member |\n"
        )
        assert "ratified quarterly dividends" not in _ship(ps).text

    def test_a_page_with_no_prose_band_abstains(self) -> None:
        """The mirror case: every band is table-shaped, so there is no prose to
        recover and the marker is the whole honest answer."""
        ps = _page(prose_above=[], prose_below=[])
        assert native_prose_floor_text(ps, 1, marker_line=MARKER, png_ref="") is None
        assert is_page_failed_marker(_ship(ps).text) is True


def _finalized_text(ps: PageState, tmp_path: Path) -> str:
    """What the page's record carries after finalization, not just selection."""
    from unittest.mock import patch

    from socr.core.manifest import finalized_page_records
    from socr.core.document import DocumentHandle

    with patch.object(DocumentHandle, "__post_init__", lambda self: None):
        handle = DocumentHandle(path=tmp_path / "doc.pdf", page_count=1)
    state = DocumentState(handle=handle)
    state.pages[1] = ps
    return finalized_page_records(state)[0].output.text


@pytest.mark.parametrize(
    "line", ["\\&", "---", "***", "===", "```", "&nbsp;", "*emphasis*", "# literal", "<!--"]
)
def test_an_escaped_native_line_renders_back_to_itself(line: str) -> None:
    """#652 round 13 (Astra's prose12 controls). Round-trip, not absence of
    tags: what a reader sees must be the line that was printed, character for
    character -- including a backslash the page itself printed, which must
    survive as one backslash and not be eaten as an escape."""
    from html.parser import HTMLParser

    markdown_it = pytest.importorskip("markdown_it")

    class _Visible(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.parts: list[str] = []

        def handle_data(self, data: str) -> None:
            self.parts.append(data)

    from socr.core.manifest import _escaped_native_line

    parser = _Visible()
    parser.feed(markdown_it.MarkdownIt().render(_escaped_native_line(line)))
    assert "".join(parser.parts).strip() == line


def test_the_trust_verdict_does_not_depend_on_which_reconstruction_is_judged() -> None:
    """#652 round 12 (Astra's prose11 control), closing a round-11 residual.

    The no-numeral lane judges the lines it is about to ship; the withholding
    lane judges ``native_region_text`` over the prose words. I flagged the
    divergence as a residual; it is not one. Both disqualifiers are token-local
    or count-based, so the verdict is invariant to word order -- pinned by
    reversing the word list, which changes both reconstructions and neither
    verdict, on a clean page and a corrupt one."""
    from socr.core.born_digital import text_layer_trusted
    from socr.core.manifest import _band_line, _page_prose_partition, native_region_text

    for ps in (
        _page(table_rows=[]),
        _page(table_rows=[], prose_above=_CORRUPT_PROSE_ABOVE, prose_below=[]),
    ):
        ps.native_words = list(reversed(ps.native_words))
        lines = "\n".join(_band_line(band) for _is_prose, band in _page_prose_partition(ps))
        assert text_layer_trusted(lines) == text_layer_trusted(native_region_text(ps.native_words))


class TestFinalizationKeepsTheLiteralBody:
    def test_the_escaped_body_survives_finalization_byte_for_byte(self, tmp_path: Path) -> None:
        """#652 round 12. Escaping is only worth anything if the escaped bytes
        are the bytes that ship: a finalization step that unescaped, re-wrapped
        or re-parsed this body would hand the page's characters their
        structural meaning back on the way out."""
        ps = _page(
            table_rows=[],
            prose_below=["| Institution | Role |", "| --- | --- |", "| Bank | Member |"],
        )
        selected = _ship(ps).text
        finalized = _finalized_text(
            _page(
                table_rows=[],
                prose_below=["| Institution | Role |", "| --- | --- |", "| Bank | Member |"],
            ),
            tmp_path,
        )

        assert finalized == selected
        assert "Bank" in finalized
        assert "following domestic policy directive:" in finalized

    def test_a_resumed_page_finalizes_to_the_same_bytes(self, tmp_path: Path) -> None:
        """The round trip that actually happens in production: the page ships,
        its sidecar is read back on the next run with ``native_words`` gone,
        and finalization must produce what it produced the first time."""
        first = _finalized_text(_page(table_rows=[]), tmp_path)

        saved = PageOutput.from_dict(_ship(_page(table_rows=[])).to_dict())
        resumed = _page(table_rows=[])
        resumed.native_words = []
        resumed.attempts = [saved]
        resumed.best_output = saved

        assert _finalized_text(resumed, tmp_path) == first


class TestResumeKeepsWhatShipped:
    def test_an_old_banner_without_the_credential_grants_nothing(self) -> None:
        """#652 round 12 (Astra's prose11 control). The banner is bytes a model
        could echo; the typed credential is not. A sidecar carrying the banner
        with no credential and no native words to rebuild from is not reused --
        the bare marker ships instead of unproven text."""
        ps = _page(table_rows=[], with_words=False)
        data = PageOutput(
            page_num=1,
            text=SCANNED_PROSE_RECOVERED_FLAG.format(page_num=1) + "\n\nOld unproven prose",
            status=PageStatus.ERROR,
            engine="nougat",
            audit_passed=False,
        ).to_dict()
        data.pop("scanned_prose_recovered", None)
        ps.best_output = PageOutput.from_dict(data)
        ps.attempts = [ps.best_output]

        shipped = _ship(ps).text
        assert is_page_failed_marker(shipped) is True
        assert "Old unproven prose" not in shipped

    def test_a_no_numeral_recovery_survives_its_own_sidecar(self) -> None:
        """#652 round 11. The new lane has its OWN banner, and the restore
        path recognises the recovery by banner plus typed credential. A restore
        that knew only the withholding banner would replace these lines with
        the bare marker on the next run -- the resume loss #649 round 2 closed
        for the other shape, reopened for this one.

        Pinned the way that loss is actually reached: the frozen bytes come
        back, ``native_words`` does not."""
        first = _ship(_page(table_rows=[]))
        saved = PageOutput.from_dict(first.to_dict())

        resumed = _page(table_rows=[])
        resumed.native_words = []
        resumed.attempts = [saved]
        resumed.best_output = saved

        assert _ship(resumed).text == first.text


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


#: The constructs #712 names, as a scan might print them, plus the sentence
#: whose disappearance is the actual loss: an unclosed ``<!--`` swallows it.
_WITHHOLDING_ACTIVES = [
    "<!--",
    "The committee retained the original mandate.",
    "# Literal heading marker",
    "> quoted directive",
    "- bulleted item",
    "*emphasis* and `code` and [link](x) and A & B",
]


class TestTheWithholdingLaneShipsLiteralCharactersToo:
    """#712. ``_escaped_native_line`` reached only the no-numeral lane.

    The withholding lane -- the one that runs whenever a page HAS a numeric
    band to hold back, which is the ticket's own Fed 1989-11-14 p3 and every
    page shaped like it -- appended its prose lines raw. Both lanes promise the
    page's literal characters, and one of them was emitting active markdown:
    a native ``<!--`` line hid the sentence beneath it and ``# ...`` became an
    ``<h1>`` in every CommonMark consumer of the shipped ``.md``, not only in
    the review viewer.
    """

    @staticmethod
    def _shipped() -> str:
        """A withholding-shaped page: a real numeric table to hold back, and a
        prose band made of the characters that must stay characters."""
        shipped = _ship(_page(prose_below=_WITHHOLDING_ACTIVES)).text
        # The lane under test is the withholding one, not the no-numeral one.
        assert shipped.startswith(SCANNED_PROSE_RECOVERED_FLAG.format(page_num=1))
        assert MARKER in shipped
        return shipped

    def test_the_pages_own_characters_never_become_structure(self) -> None:
        """The no-numeral lane's pin, mirrored onto the lane that was missing
        it, through the installed renderer rather than a claim about escaping."""
        markdown_it = pytest.importorskip("markdown_it")

        rendered = markdown_it.MarkdownIt().render(self._shipped())

        assert "<!--" not in rendered
        assert "The committee retained the original mandate." in rendered
        for tag in ("<h1>", "<blockquote>", "<li>", "<em>", "<code>", "<a href"):
            assert tag not in rendered, tag
        assert "# Literal heading marker" in rendered
        assert "&gt; quoted directive" in rendered
        assert "*emphasis* and `code` and [link](x) and A &amp; B" in rendered

    def test_the_review_viewer_shows_the_same_characters(self) -> None:
        """The second consumer, which has its own regex renderer rather than
        markdown-it. Run as the JavaScript socr actually ships, under Node."""
        if _NODE is None:
            pytest.skip("node not installed")

        from socr.review.html import _TEMPLATE

        renderer = _TEMPLATE[_TEMPLATE.index("function esc(") : _TEMPLATE.index("function head(")]
        driver = (
            '\nconst fs = require("fs");'
            '\nprocess.stdout.write(renderMd(JSON.parse(fs.readFileSync(0, "utf8"))));'
        )
        rendered = subprocess.run(
            [_NODE, "-e", renderer + driver],
            input=json.dumps(self._shipped()),
            text=True,
            capture_output=True,
            check=True,
            timeout=30,
        ).stdout

        for tag in ("<h1>", "<blockquote>", "<li>", "<i>", "<code>", "<a "):
            assert tag not in rendered, tag
        assert "<!--" not in rendered
        assert "The committee retained the original mandate." in rendered

    def test_the_escaping_survives_finalization(self, tmp_path: Path) -> None:
        """Selection is not what a reader opens. The literal body has to reach
        the assembled record too, unchanged."""
        ps = _page(prose_below=_WITHHOLDING_ACTIVES)
        shipped = _ship(ps).text

        assert _finalized_text(ps, tmp_path) == shipped

    def test_the_escaping_survives_resume(self) -> None:
        """And the next run. The frozen bytes come back through the typed
        credential with ``native_words`` gone, which is how the resume loss is
        actually reached."""
        first = _ship(_page(prose_below=_WITHHOLDING_ACTIVES))
        saved = PageOutput.from_dict(first.to_dict())

        resumed = _page(prose_below=_WITHHOLDING_ACTIVES)
        resumed.native_words = []
        resumed.attempts = [saved]
        resumed.best_output = saved

        assert _ship(resumed).text == first.text
        assert "\\# Literal heading marker" in _ship(resumed).text

    def test_the_withheld_table_is_still_withheld(self) -> None:
        """Escaping changes what the prose looks like, not what ships. The
        numeric band stays behind the marker."""
        shipped = self._shipped()

        assert "250.0" not in shipped
        assert shipped.count(MARKER) == 1
