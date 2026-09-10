"""#652: B1's prose witness must not trust a corrupt, table-contaminated, or
resume-empty native layer.

Three legs, each pinned as a DIFFERENCE between two runs of the same code in
the same process that vary only the one thing under test -- never as an
absolute tuple, which CI's providerless environment can legitimately change
(see CLAUDE.md).

P1  the witness must be a layer we trust. A page is classified SCANNED
    precisely when its embedded layer is too corrupt to route on, so
    corroborating an OCR attempt against that same layer is circular: an
    attempt echoing the corruption clears the guard the corruption caused.

P2a missing evidence is not a pass. ``native_words`` is a live-run cache the
    sidecar never carried, so a resumed page re-ran the bbox sanity check with
    bboxes and no words and the check returned "no objection", letting resume
    stamp a floor outcome the live run did not reach.

P2b the witness must be PROSE. With no detected bbox (the production shape --
    Fed 1989-11-14 p3 measures ``detected_table_count == 0``) every native
    token counted, including the withheld table's own headers and row labels,
    so an attempt that copies the table faithfully and fabricates the prose
    corroborated itself on the half that was never in doubt.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from socr.core.config import PipelineConfig
from socr.core.manifest import (
    SCANNED_PROSE_RECOVERED_FLAG,
    _select_page_output_tagged,
    _table_bbox_sane,
    table_floor_text_for_source,
)
from socr.core.result import FailureMode, PageOutput, PageStatus
from socr.core.state import DocumentHandle, DocumentState, PageState
from socr.pipeline.orchestrator import UnifiedPipeline

fitz = pytest.importorskip("fitz")


def _words(lines: list[str], *, y0: float = 0.0, x0: float = 0.0) -> list[tuple]:
    """Native words laid out one printed line per entry of *lines*.

    Real geometry matters to every check under test (the bbox sanity check,
    the prose/table band partition, and since round 3 the witness's own walk
    outward from a table row), so these fixtures place words on real baselines
    rather than stacking them on one placeholder box.

    An EMPTY entry emits no words but still consumes a line, which is how
    these fixtures print a block break -- the blank line between a table and
    the paragraph beneath it. Without one, a page is set solid and the witness
    walk correctly finds nothing separating the table's labels from its prose.
    """
    words: list[tuple] = []
    for line_idx, line in enumerate(lines):
        top = y0 + line_idx * 10.0
        for word_idx, tok in enumerate(line.split()):
            left = x0 + word_idx * 12.0
            words.append((left, top, left + 10.0, top + 8.0, tok, 0, 0, 0))
    return words


# A prose layer with 6 of 24 alpha tokens showing font/ToUnicode corruption --
# dropped inter-word spaces read back as mid-word capitals, the shape
# ``BornDigitalDetector._encoding_corruption_ratio`` scores. 25% is far above
# MAX_ENCODING_CORRUPTION (5%), the same threshold that routed the page to OCR
# in the first place. Measured on the real fixture for scale: Fed 1989-11-14 p3
# reads 6.6% over the whole page and 33.3% over its numeric bands.
_CORRUPT_PROSE = [
    "The informationReviewed at this meetingSuggests continuing",
    "expansion in economicActivity though at a somewhatSlower",
    "pace than earlier in the year TotalNonfarm payroll employment",
    "increased appreciably in October butOn balance its growth",
]
# The same sentences with the spaces intact: a clean layer, same vocabulary.
_CLEAN_PROSE = [
    "The information reviewed at this meeting suggests continuing",
    "expansion in economic activity though at a somewhat slower",
    "pace than earlier in the year total nonfarm payroll employment",
    "increased appreciably in October but on balance its growth",
]


def _attempt_echoing(lines: list[str]) -> str:
    """An OCR attempt whose vocabulary is exactly the layer's own.

    This is the hallucination the guard exists to catch when the layer is not
    trustworthy: perfect token overlap proves the attempt agrees with the text
    layer, and nothing more.
    """
    return "\n".join(lines)


def _scanned_page(words: list[tuple]) -> PageState:
    ps = PageState(page_num=1)
    ps.is_born_digital = False
    ps.native_text = ""
    ps.native_words = words
    ps.detected_table_count = 0
    ps.detected_table_bboxes = []
    ps.scanned_table_evidence_failed = True
    return ps


MARKER = "[page 1 failed: unverifiable table — see image]"


def _shipped(ps: PageState, attempt_text: str) -> str:
    """What the page ACTUALLY ships with *attempt_text* as its winning attempt.

    #652 round 10 re-scoped this whole suite. Every check below used to call
    ``manifest._prose_corroboration_ok``, the vocabulary-overlap guard on the
    scanned-table-failure branch; the guard is gone, because nine rounds of
    review showed no page-local evidence separates a withheld table's
    vocabulary from its prose's. The findings each test pinned are unchanged
    facts about layout, so they are re-asserted where they always mattered --
    in the bytes the branch emits through real selection.
    """
    ps.best_output = PageOutput(
        page_num=1,
        text=attempt_text,
        status=PageStatus.ERROR,
        engine="nougat",
        audit_passed=False,
        failure_mode=FailureMode.HALLUCINATION,
    )
    ps.attempts = [ps.best_output]
    state = DocumentState.__new__(DocumentState)
    state.pages = {1: ps}
    output, _provenance = _select_page_output_tagged(state, 1)
    return output.text


def _is_native_body(text: str) -> bool:
    """Whether a shipped body came from the PAGE rather than from the model.

    Only two bodies can leave this branch now: #649's native recovery, which
    always opens with its banner, or the bare marker where even that cannot be
    proven. A spliced attempt would open with the model's own first line, so
    this predicate is what "the attempt was refused" means in bytes.
    """
    return text.startswith(SCANNED_PROSE_RECOVERED_FLAG.format(page_num=1)) or text == MARKER


class TestP1TheWitnessMustBeATrustedLayer:
    """RE-SCOPED in round 10 to the caller that still reads the layer.

    The corrupt-layer check was written for the corroboration guard: a page is
    classified SCANNED precisely because its embedded layer is too corrupt to
    route on, so scoring an OCR attempt against that same layer was circular.
    The guard is gone; ``text_layer_trusted`` is not, because #649 SHIPS that
    layer as the page's body. The circularity became a retention question with
    the same answer -- a layer we do not trust is not published as the page's
    own text either -- and that caller is reachable, which is why these two
    checks stay.

    Both fixtures print one numeric row beneath the paragraph. Without it the
    page has no withheld band, #649's recovery declines, and both bodies
    collapse to the bare marker for a reason that has nothing to do with
    corruption -- the round-10 residual recorded in the branch log.
    """

    _ROW = ["Nonfarm payroll index 118.4"]

    def test_a_corrupt_layer_is_not_published_as_the_pages_own_text(self) -> None:
        """The falsification, re-pinned in bytes. Both pages carry the SAME
        sentences and the attempt echoes them word for word; the only
        difference is whether the layer's own text is corrupt. The clean page
        ships its paragraphs from its own layer, flagged; the corrupt page
        ships the marker alone rather than publish the corruption that made it
        a scan in the first place."""
        corrupt = _scanned_page(_words(_CORRUPT_PROSE + self._ROW))
        clean = _scanned_page(_words(_CLEAN_PROSE + self._ROW))

        corrupt_body = _shipped(corrupt, _attempt_echoing(_CORRUPT_PROSE))
        clean_body = _shipped(clean, _attempt_echoing(_CLEAN_PROSE))

        assert "employment" in clean_body, (
            "control: a clean layer must still ship its prose, or the pin "
            "below passes for the wrong reason (everything refused)"
        )
        assert corrupt_body == MARKER
        assert corrupt_body != clean_body

    def test_neither_layer_lets_the_model_author_the_body(self) -> None:
        """The other half, which round 10 made unconditional: whatever the
        layer's state, what ships is the page's own text or nothing. The
        attempt here carries a parseable grid, so before #652 this branch
        spliced it unchecked."""
        for lines in (_CORRUPT_PROSE, _CLEAN_PROSE):
            ps = _scanned_page(_words(lines + self._ROW))
            body = _shipped(
                ps, _attempt_echoing(lines) + "\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n"
            )
            assert _is_native_body(body)
            assert "| A | B |" not in body


class TestP2bTheWitnessMustBeProse:
    """A faithful table beside fabricated prose, with no detected bbox.

    The adversarial shape #652 asks for: today's fixtures only refuse when
    fabricated prose is paired with fabricated LABELS, so the guard could be
    passing on table vocabulary alone.
    """

    _TABLE_LINES = [
        "Austrian National Bank 250.0 12 mos.",
        "National Bank of Belgium 1,000.0 12 mos.",
        "German Federal Bank 6,000.0 12 mos.",
        "Swiss National Bank 4,000.0 12 mos.",
        "Netherlands Bank 500.0 12 mos.",
        "Bank of England 3,000.0 12 mos.",
    ]
    _PROSE_LINES = [
        "authorized and directed until otherwise directed by the Committee",
        "to execute transactions in the System Account in accordance",
    ]

    def _page(self) -> PageState:
        return _scanned_page(_words(self._TABLE_LINES + [""] + self._PROSE_LINES))

    _FABRICATED_ATTEMPT = (
        "| Foreign Bank | Amount |\n| --- | --- |\n"
        + "".join(f"| {line} |\n" for line in _TABLE_LINES)
        + "\nQuarterly dividends were ratified.\n"
    )
    _GENUINE_ATTEMPT = (
        "| Foreign Bank | Amount |\n| --- | --- |\n"
        + "".join(f"| {line} |\n" for line in _TABLE_LINES)
        + "\n"
        + "\n".join(_PROSE_LINES)
        + "\n"
    )

    def test_a_faithful_table_cannot_vouch_for_fabricated_prose(self) -> None:
        """Every prose word in the attempt is invented; every table word is
        copied exactly. The invented sentence must not reach the page."""
        body = _shipped(self._page(), self._FABRICATED_ATTEMPT)

        assert "Quarterly dividends were ratified" not in body
        assert _is_native_body(body)

    def test_the_same_attempt_with_genuine_prose_ships_from_the_page_instead(self) -> None:
        """RE-PINNED three times, and the shape of the whole ticket.

        This was the difference pin: identical table half, real prose,
        accepted. Round 7 refused it (a recognised row on one side only),
        round 8 refused it (the page has a withheld numeric band at all), and
        round 10 stopped asking -- no attempt authors a body in this branch.

        What the re-pin has to show is that refusing the MODEL is not losing
        the PAGE, and here it is not: the paragraph still ships, from the
        page's own layer, under #649's banner, while the attempt's table --
        the half nothing verified -- does not."""
        body = _shipped(self._page(), self._GENUINE_ATTEMPT)

        assert _is_native_body(body)
        assert "authorized and directed" in body
        assert "| Foreign Bank |" not in body
        for withheld in ("250.0", "6,000.0"):
            assert withheld not in body, withheld

    def test_flanking_a_paragraph_with_rows_changes_nothing(self) -> None:
        """RE-PINNED in rounds 8 and 10. Round 7 added this layout as proof the
        guard was not vacuous: with a recognised row on both sides of the
        paragraph, the genuine attempt cleared and the fabricated one did not.

        Astra's ruling rejected the inference -- a flanked block is not thereby
        prose, and the same layout was reproduced as a fabrication path -- so
        what survives is the measurement. Both attempts are refused, the page's
        own paragraph still ships, no printed value does."""
        page = _scanned_page(
            _words(self._TABLE_LINES[:3] + [""] + self._PROSE_LINES + [""] + self._TABLE_LINES[3:])
        )

        fabricated_body = _shipped(page, self._FABRICATED_ATTEMPT)
        genuine_body = _shipped(
            _scanned_page(
                _words(
                    self._TABLE_LINES[:3] + [""] + self._PROSE_LINES + [""] + self._TABLE_LINES[3:]
                )
            ),
            self._GENUINE_ATTEMPT,
        )

        assert "Quarterly dividends were ratified" not in fabricated_body
        for body in (fabricated_body, genuine_body):
            assert _is_native_body(body)
            assert "authorized and directed" in body
            assert "| Foreign Bank |" not in body


class TestWrappedLabelsAreNotEvidence:
    """#652 round 2 (Astra, 2026-09-10): the witness is not the shipping
    partition.

    #649 ships a zero-numeral band -- a wrapped row label -- flagged rather
    than lose it, because it carries no printed value. Reusing that same
    partition as the corroboration witness quietly promoted those bands from
    "safe to show" to "trustworthy evidence about prose". A table whose labels
    sat on baselines separate from their amounts therefore put its ENTIRE
    vocabulary back into the witness, and the identical fabricated attempt that
    was refused with labels inline was accepted with them split.
    """

    _LABELS = [
        "Austrian National Bank",
        "National Bank of Belgium",
        "German Federal Bank",
        "Swiss National Bank",
        "Netherlands Bank",
        "Bank of England",
    ]
    # Invented prose, built entirely from the table's own row labels.
    _FABRICATED = (
        "Austrian National Bank German Federal Bank Swiss National Bank "
        "ratified quarterly dividends."
    )
    _ATTEMPT = (
        _FABRICATED + "\n\n| Label | Amount |\n| --- | --- |\n| Austrian National Bank | 250.0 |\n"
    )

    # Ordinary layout that puts a row label TWO bands from its value: a units
    # caption between them. Spacer lines and a value wrapped below a caption do
    # the same thing.
    _CAPTION = "in millions of dollars unless noted"

    def _page(self, *, split: bool) -> PageState:
        """The same table twice. ``split`` puts each label on its own baseline,
        the way a printed table wraps a long row label; otherwise each label
        shares its amount's baseline."""
        if split:
            table = [line for label in self._LABELS for line in (label, "250.0")]
        else:
            table = [f"{label} 250.0" for label in self._LABELS]
        prose = [
            "authorized and directed until otherwise directed by the Committee",
            "to execute transactions in the System Account in accordance",
        ]
        return _scanned_page(_words(table + [""] + prose))

    @pytest.mark.parametrize("split", [False, True])
    def test_the_invented_sentence_never_ships(self, split: bool) -> None:
        """The falsification, through real selection. Before #652, splitting
        each row label onto its own baseline was enough for this attempt's
        invented sentence to ship: baseline layout must not decide what a page
        publishes."""
        body = _shipped(self._page(split=split), self._ATTEMPT)

        assert "ratified quarterly dividends" not in body
        assert _is_native_body(body)

    # Astra's round-3 reproducer, verbatim in shape: ONE row whose label sits
    # two bands from its value, and a fabrication built from that label.
    _CAPTIONED_TABLE = ["Austrian National Bank", _CAPTION, "250.0"]
    _CAPTIONED_FABRICATION = "Austrian National Bank ratified quarterly dividends."
    _CAPTIONED_ATTEMPT = (
        _CAPTIONED_FABRICATION
        + "\n\n| Label | Amount |\n| --- | --- |\n| Austrian National Bank | 250.0 |\n"
    )

    _CAPTIONED_PROSE = [
        "authorized and directed until otherwise directed by the Committee",
        "to execute transactions in the System Account in accordance",
    ]

    def _captioned_page(self, *, rows: int = 2) -> PageState:
        """Label, caption, value -- the label is at distance 2 from the band
        that is actually withheld, with a block break before the paragraph.

        Two rows by default: with only one numeric row the page has no
        measurable row pitch and the witness abstains outright (see
        ``test_a_single_row_table_abstains_rather_than_guess``), which would
        make the fabrication pins below pass for a reason that has nothing to
        do with distance.
        """
        table = [
            line
            for row in range(rows)
            for line in (self._LABELS[row], self._CAPTION, f"{250 + row * 50}.0")
        ]
        return _scanned_page(_words(table + [""] + self._CAPTIONED_PROSE))

    def test_a_single_row_table_abstains_rather_than_guess(self) -> None:
        """#652 round 4 (Astra). One numeric row gives no row pitch, so no step
        on the page can be shown to be a block break rather than the table's
        own advance. The witness abstains rather than admit the label on its
        distance alone -- and abstaining costs no page text, because #649 ships
        the native prose either way."""
        fabricated_body = _shipped(self._captioned_page(rows=1), self._CAPTIONED_ATTEMPT)
        genuine_body = _shipped(self._captioned_page(rows=1), " ".join(self._CAPTIONED_PROSE))

        assert "ratified quarterly dividends" not in fabricated_body
        for body in (fabricated_body, genuine_body):
            assert _is_native_body(body)
            assert "250.0" not in body

    def test_a_label_two_bands_from_its_value_is_not_evidence_either(self) -> None:
        """#652 round 3 (Astra). The first fix walked one hop, so a single
        intervening zero-digit band -- a units caption, ordinary layout -- put
        the label straight back into the witness and this returned True: the
        same fabrication class, one line away."""
        body = _shipped(self._captioned_page(), self._CAPTIONED_ATTEMPT)

        assert "ratified quarterly dividends" not in body
        assert _is_native_body(body)

    def test_a_paragraph_across_a_real_gap_needs_a_row_on_both_sides(self) -> None:
        """RE-PINNED in round 7. This control kept the round-3 fix from being
        "refuse everything": a block break between the table and the paragraph
        was enough for a genuine attempt to clear.

        Round 7 requires a recognised numeric row across a measured gap on BOTH
        sides of the block, so the same paragraph, printed last on the page,
        is refused. Adding a second captioned row block BENEATH it does not
        rescue it either, and that is the sharper half of the cost: the band
        this layout puts against the paragraph is the caption, a zero-digit
        line the walk absorbed, not the row itself. Where a table's outer band
        is a wrapped label or a units caption, the page cannot supply the
        evidence round 7 asks for on that side at all.

        What is refused is the WITNESS, not the page: since #649 the native
        layer's own prose ships flagged either way, and the fabrication built
        from the table's label is still refused on both layouts."""
        genuine = " ".join(self._CAPTIONED_PROSE) + "."
        assert _is_native_body(_shipped(self._captioned_page(), genuine))

        sandwiched = _scanned_page(
            _words(
                self._CAPTIONED_TABLE
                + [""]
                + self._CAPTIONED_PROSE
                + [""]
                + ["German Federal Bank", self._CAPTION, "6,000.0"]
            )
        )
        assert _is_native_body(_shipped(sandwiched, genuine))
        fabricated_body = _shipped(sandwiched, self._CAPTIONED_ATTEMPT)
        assert "ratified quarterly dividends" not in fabricated_body
        assert _is_native_body(fabricated_body)

    @pytest.mark.parametrize("footnote_pitch", [12.0, 6.0])
    def test_text_elsewhere_on_the_page_cannot_redraw_the_table(
        self, footnote_pitch: float
    ) -> None:
        """#652 round 4 (Astra), the finding itself. The stopping criterion was
        the page-wide MEDIAN step, so tightening an unrelated footnote block to
        6pt pulled the median down, the table's own unchanged 12pt step was
        reclassified as a block break, its label entered the witness and the
        fabrication shipped. A table's extent is a fact about the table; text
        elsewhere on the page must not be able to redraw it.

        Same table at both pitches, only the footnote block differs."""
        table: list[tuple] = []
        for row in range(2):
            base = row * 36.0
            table += _words([self._LABELS[row]], y0=base)
            table += _words([self._CAPTION], y0=base + 12.0)
            table += _words([f"{250 + row * 50}.0"], y0=base + 24.0)

        footnotes: list[tuple] = []
        for idx in range(10):
            footnotes += _words(
                ["Additional explanatory remarks concerning the original document"],
                y0=120.0 + idx * footnote_pitch,
            )

        body = _shipped(_scanned_page(table + footnotes), self._CAPTIONED_ATTEMPT)

        assert "ratified quarterly dividends" not in body
        assert _is_native_body(body)

    def test_genuine_prose_on_the_split_layout_needs_a_row_on_both_sides(self) -> None:
        """RE-PINNED in round 7, same reason as the captioned control above:
        the page's own paragraph is refused where it is the last block on the
        page, and refused again when a second split-label block is printed
        beneath it, because the band that block puts against the paragraph is
        a wrapped label rather than the numeric row round 7 asks for.

        Split labels are still not what decides either verdict, which is what
        this control exists to say: the inline-label layout of
        ``TestP2bTheWitnessMustBeProse`` accepts the same shape of paragraph
        when rows flank it, so the refusal here is about the evidence a layout
        can supply, not about where a label was printed."""
        genuine = (
            "authorized and directed until otherwise directed by the Committee "
            "to execute transactions in the System Account in accordance."
        )
        assert _is_native_body(_shipped(self._page(split=True), genuine))

        sandwiched = _scanned_page(
            _words(
                [line for label in self._LABELS[:3] for line in (label, "250.0")]
                + [""]
                + [
                    "authorized and directed until otherwise directed by the Committee",
                    "to execute transactions in the System Account in accordance",
                ]
                + [""]
                + [line for label in self._LABELS[3:] for line in (label, "250.0")]
            )
        )
        assert _is_native_body(_shipped(sandwiched, genuine))
        fabricated_body = _shipped(sandwiched, self._ATTEMPT)
        assert "ratified quarterly dividends" not in fabricated_body
        assert _is_native_body(fabricated_body)


class TestP2aMissingEvidenceIsNotSanity:
    """``_table_bbox_sane`` and the floor it gates, live versus resumed."""

    # A bbox that swallowed the paragraph beneath the table: within it, bands
    # carrying no numeric token outnumber the bands that do. That is the "too
    # large" objection, and it must survive resume.
    _INSANE_LINES = [
        "Widget 10.0",
        "Gadget 20.0",
        "the committee reviewed the arrangement at length",
        "and authorized its renewal for a further twelve months",
        "with no dissenting votes recorded in the minutes",
    ]
    _SANE_LINES = [
        "Widget 10.0",
        "Gadget 20.0",
        "Sprocket 30.0",
        "units in millions",
    ]

    def _page(self, lines: list[str]) -> PageState:
        ps = PageState(page_num=1)
        ps.is_born_digital = True
        ps.native_words = _words(lines)
        ps.detected_table_count = 1
        ps.detected_table_bboxes = [(-5.0, -5.0, 500.0, 5.0 + len(lines) * 10.0)]
        ps.native_table_region_count = 1
        return ps

    _SOURCE = "Header\n\n| Item | Value |\n| --- | --- |\n| Widget | 10.0 |\n\nFooter"

    def test_live_and_resumed_reach_the_same_floor_outcome(self) -> None:
        """The difference pin. Same page twice in one process: once live (words
        present), once as resume restores it (words gone, verdict restored).
        The two bodies must be identical -- whatever the live run decided."""
        for lines in (self._INSANE_LINES, self._SANE_LINES):
            live = self._page(lines)
            live_body = table_floor_text_for_source(live, 1, self._SOURCE)

            # What the orchestrator records, and what the sidecar carries.
            verdict = _table_bbox_sane(live)

            resumed = self._page(lines)
            resumed.native_words = []  # never persisted; gone on resume
            resumed.table_bbox_sane = verdict
            resumed_body = table_floor_text_for_source(resumed, 1, self._SOURCE)

            assert resumed_body == live_body, (
                "a resumed page re-ran the floor over restored bboxes and no "
                "words, and stamped a different outcome than the run that "
                "wrote the bytes"
            )

    def test_the_two_pages_do_reach_different_outcomes(self) -> None:
        """Guards the pin above against passing because everything agrees:
        the insane bbox must floor where the sane one splices."""
        insane = table_floor_text_for_source(self._page(self._INSANE_LINES), 1, self._SOURCE)
        sane = table_floor_text_for_source(self._page(self._SANE_LINES), 1, self._SOURCE)
        assert "Header" in sane and "Header" not in insane
        assert insane != sane

    def test_no_words_and_no_recorded_verdict_fails_closed(self) -> None:
        """The falsification proper: with a bbox claimed and NOTHING able to
        check it, the old code returned "no objection" and spliced."""
        stranded = self._page(self._SANE_LINES)
        stranded.native_words = []
        stranded.table_bbox_sane = None

        assert _table_bbox_sane(stranded) is False
        body = table_floor_text_for_source(stranded, 1, self._SOURCE)
        assert "Header" not in body, "absence of evidence is not sanity"

    def test_a_page_with_no_bbox_at_all_is_still_no_objection(self) -> None:
        """Scope guard: the fail-closed turn is about a CLAIMED bbox that
        cannot be checked, not about pages that claim nothing. Those are
        already floored by the four coverage conditions."""
        ps = PageState(page_num=1)
        ps.native_words = []
        ps.detected_table_bboxes = []
        assert _table_bbox_sane(ps) is True


class TestTheVerdictRoundTripsThroughTheSidecar:
    """P2a's persistence leg: a verdict that never reaches the sidecar cannot
    be restored, and the restored page fails closed on a page the live run
    spliced."""

    def _state(self) -> DocumentState:
        with patch.object(DocumentHandle, "__post_init__", lambda self: None):
            handle = DocumentHandle(path=Path("/tmp/gh652.pdf"), page_count=1)
        return DocumentState(handle=handle)

    @pytest.mark.parametrize("verdict", [True, False])
    def test_write_then_restore_preserves_the_verdict(self, tmp_path: Path, verdict: bool) -> None:
        pipeline = UnifiedPipeline(PipelineConfig(quiet=True))
        pipeline._scan_root = tmp_path
        state = self._state()
        state.pages[1].table_bbox_sane = verdict

        sidecar = pipeline._flush_page_sidecar(state, 1, tmp_path, terminal=False)
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        assert meta.get("table_bbox_sane") is verdict

        restored = self._state()
        pipeline._restore_terminal_page_state(
            restored,
            1,
            PageOutput(
                page_num=1,
                text="body",
                status=PageStatus.SUCCESS,
                engine="qwen",
                audit_passed=True,
            ),
            tmp_path,
        )
        assert restored.pages[1].table_bbox_sane is verdict

    def test_an_older_sidecar_restores_doubt_not_sanity(self, tmp_path: Path) -> None:
        """A sidecar written before this key existed must not read back as a
        pass: ``None`` is "never evaluated", and that is doubt."""
        pipeline = UnifiedPipeline(PipelineConfig(quiet=True))
        pipeline._scan_root = tmp_path
        state = self._state()
        state.pages[1].table_bbox_sane = True

        sidecar = pipeline._flush_page_sidecar(state, 1, tmp_path, terminal=False)
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        meta.pop("table_bbox_sane")
        sidecar.write_text(json.dumps(meta), encoding="utf-8")

        restored = self._state()
        pipeline._restore_terminal_page_state(
            restored,
            1,
            PageOutput(
                page_num=1,
                text="body",
                status=PageStatus.SUCCESS,
                engine="qwen",
                audit_passed=True,
            ),
            tmp_path,
        )
        assert restored.pages[1].table_bbox_sane is None
