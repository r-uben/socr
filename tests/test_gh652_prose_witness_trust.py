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
    PROSE_CORROBORATION_MIN,
    _prose_corroboration_ok,
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

    Real geometry matters to every check under test (both the bbox sanity
    check and the prose/table band partition read it), so these fixtures place
    words on real baselines rather than stacking them on one placeholder box.
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


class TestP1TheWitnessMustBeATrustedLayer:
    def test_a_corrupt_layer_refuses_the_attempt_that_echoes_it(self) -> None:
        """The falsification. Both pages carry the SAME vocabulary and the
        attempt echoes it word for word, so token overlap is 1.0 on both; the
        only difference is whether the layer's own text is corrupt. Before the
        fix both returned True -- the corrupt layer vouched for an attempt that
        may have been read straight off it."""
        corrupt = _scanned_page(_words(_CORRUPT_PROSE))
        clean = _scanned_page(_words(_CLEAN_PROSE))

        corrupt_verdict = _prose_corroboration_ok(corrupt, _attempt_echoing(_CORRUPT_PROSE))
        clean_verdict = _prose_corroboration_ok(clean, _attempt_echoing(_CLEAN_PROSE))

        assert clean_verdict is True, (
            "control: a clean layer must still corroborate, or the pin below "
            "passes for the wrong reason (everything refused)"
        )
        assert corrupt_verdict is False
        assert corrupt_verdict != clean_verdict

    def test_the_refusal_reaches_the_page_that_ships(self) -> None:
        """Wired through the production call site, not only the predicate: the
        same two pages selected end to end must ship different bodies."""
        from socr.core.manifest import _select_page_output_tagged

        def _ship(lines: list[str]) -> str:
            ps = _scanned_page(_words(lines))
            attempt = PageOutput(
                page_num=1,
                # A parseable grid, so ``splice_all_table_regions`` has
                # something to splice and the guard is the only thing that can
                # refuse: without it this branch ships the prose unchecked.
                text=_attempt_echoing(lines) + "\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n",
                status=PageStatus.ERROR,
                engine="nougat",
                audit_passed=False,
            )
            ps.attempts = [attempt]
            ps.best_output = attempt
            state = DocumentState.__new__(DocumentState)
            state.pages = {1: ps}
            output, _provenance = _select_page_output_tagged(state, 1)
            return output.text

        corrupt_body = _ship(_CORRUPT_PROSE)
        clean_body = _ship(_CLEAN_PROSE)

        assert "payroll" in clean_body, "control: the clean page still ships its prose"
        assert "payroll" not in corrupt_body, (
            "the corrupt page must fail closed to the marker, not splice prose "
            "corroborated against the corruption that made it a scan"
        )
        assert corrupt_body != clean_body


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
        return _scanned_page(_words(self._TABLE_LINES + self._PROSE_LINES))

    def test_a_faithful_table_cannot_vouch_for_fabricated_prose(self) -> None:
        # Every prose word is invented; every table word is copied exactly.
        fabricated = (
            "| Foreign Bank | Amount |\n| --- | --- |\n"
            + "".join(f"| {line} |\n" for line in self._TABLE_LINES)
            + "\nQuarterly dividends were ratified.\n"
        )
        assert _prose_corroboration_ok(self._page(), fabricated) is False

    def test_the_same_attempt_with_genuine_prose_is_accepted(self) -> None:
        """Control, and the difference pin: identical table half, real prose."""
        genuine = (
            "| Foreign Bank | Amount |\n| --- | --- |\n"
            + "".join(f"| {line} |\n" for line in self._TABLE_LINES)
            + "\n"
            + "\n".join(self._PROSE_LINES)
            + "\n"
        )
        assert _prose_corroboration_ok(self._page(), genuine) is True

    def test_the_table_half_alone_would_have_cleared_the_floor(self) -> None:
        """Proves the refusal above is not vacuous. Scored the way the code
        scored before this ticket -- every native token as witness, the whole
        attempt as subject -- the fabricated attempt clears
        ``PROSE_CORROBORATION_MIN`` on table vocabulary alone."""
        import re

        fabricated_attempt = " ".join(self._TABLE_LINES) + " Quarterly dividends were ratified."
        tokens = lambda text: set(re.findall(r"[a-z]{4,}", text.lower()))  # noqa: E731

        attempt_tokens = tokens(fabricated_attempt)
        whole_page_witness = tokens(" ".join(self._TABLE_LINES + self._PROSE_LINES))
        old_overlap = len(attempt_tokens & whole_page_witness) / len(attempt_tokens)

        assert old_overlap >= PROSE_CORROBORATION_MIN, (
            "the fixture must be one the old whole-page witness accepted, or "
            "the refusal above pins nothing"
        )


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
        return _scanned_page(_words(table + prose))

    @pytest.mark.parametrize("split", [False, True])
    def test_the_same_fabrication_is_refused_either_way(self, split: bool) -> None:
        """The falsification. Before this, ``split=True`` returned True."""
        assert _prose_corroboration_ok(self._page(split=split), self._ATTEMPT) is False

    @pytest.mark.parametrize("split", [False, True])
    def test_the_invented_sentence_never_ships(self, split: bool) -> None:
        """Through real selection, not the predicate alone: baseline layout
        must not decide what a page ships."""
        ps = self._page(split=split)
        ps.best_output = PageOutput(
            page_num=1,
            text=self._ATTEMPT,
            status=PageStatus.ERROR,
            engine="nougat",
            audit_passed=False,
            failure_mode=FailureMode.HALLUCINATION,
        )
        ps.attempts = [ps.best_output]

        state = DocumentState.__new__(DocumentState)
        state.pages = {1: ps}
        output, _provenance = _select_page_output_tagged(state, 1)

        assert "ratified quarterly dividends" not in output.text

    def test_genuine_prose_still_corroborates_on_the_split_layout(self) -> None:
        """Control. The stricter witness must not refuse everything: an attempt
        whose prose really is the page's still clears, with the labels on their
        own baselines."""
        genuine = (
            "authorized and directed until otherwise directed by the Committee "
            "to execute transactions in the System Account in accordance."
        )
        assert _prose_corroboration_ok(self._page(split=True), genuine) is True


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
