"""#1270 (B4, class 2) under the RELEASE RULE: a page with no number on either side.

Before: 49 pages of run 51270eb5 carried ``text_layer_numbers=UNKNOWN`` ("the reading ships no
number") because they have none: blank and divider pages, contents pages of words. The owner's
ruling stands (CHECKED requires ``compared > 0``). What may remove the mark is independent
page-scope evidence on the text the page ships: ``independent_page_evidence`` over the trace the
page would carry (the #1294 words check PASSED, no FAIL or UNKNOWN entry anywhere), on top of the
guards of the original class: the #1050 ink check PASSED, no embedded raster, a clean layer, and
every token of the layer shipped (the words check ignores words under four letters, so a dropped
``not`` passes it: Astra, re-check of PR #1325).

A blank page whose layer the detector refused as a scan has no words PASS, so it keeps its mark,
which is true: it ships no number. Each guard has a control arm here and a mutant.
"""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import fitz
import pytest

from socr.core.config import EngineType, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.providers import PROFILE_QWEN_LOCAL
from socr.core.result import PageOutput, PageStatus
from socr.core.state import DocumentState
from socr.core.verdict import (
    NO_NUMBER_EITHER_SIDE_REASON,
    NO_NUMBER_LAYER_HAS_SOME_REASON,
    NO_NUMBER_REASON,
    Check,
    CheckResult,
    TraceResult,
    numbers_clause_released,
    silent_success_clauses,
    text_layer_numbers_clause,
    text_layer_numbers_outcome,
    text_layer_words_outcome,
)
from socr.pipeline.agentic import PageDecision
from socr.pipeline.orchestrator import UnifiedPipeline

LAYER = "Federal Reserve policy does not permit unrestricted lending."
BLANK = "(This page intentionally blank.)"


def _p(**over) -> SimpleNamespace:
    base = dict(
        own_layer_text=LAYER,
        own_layer_word_reference=LAYER,
        own_layer_figure_words="",
        is_born_digital=True,
        prior_ocr_layer_suspect=False,
        has_unrecovered_symbol_glyphs=False,
        table_ladder_disposition=None,
        text_coverage_verdict="passed",
        has_figures=False,
        layer_glyphs_undamaged=True,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _out(text: str = LAYER) -> PageOutput:
    return PageOutput(page_num=1, text=text, status=PageStatus.SUCCESS, engine="qwen")


def test_a_page_with_independent_evidence_is_released() -> None:
    assert text_layer_numbers_clause(_out(), _p()) is None
    outcome = text_layer_numbers_outcome(_out(), _p())
    assert outcome.entry.result is TraceResult.NOT_APPLICABLE
    assert outcome.entry.reason == NO_NUMBER_EITHER_SIDE_REASON
    assert numbers_clause_released(_out(), _p()) is True
    assert silent_success_clauses(_out(), _p()) == ()


def test_astra_a_dropped_not_stays_marked_although_the_words_check_passes() -> None:
    """Astra's breaking input. The words check PASSES on it (it ignores short words), so without
    the every-token guard the release rule would lift the mark."""
    dropped = "Federal Reserve policy does permit unrestricted lending."
    assert text_layer_words_outcome(_out(dropped), _p()).entry.result is TraceResult.PASS, (
        "premise: the independent evidence the rule asks for is present"
    )
    clause = text_layer_numbers_clause(_out(dropped), _p())
    assert clause is not None and clause.result is CheckResult.UNKNOWN
    assert clause.reason == NO_NUMBER_REASON
    assert [c.check for c in silent_success_clauses(_out(dropped), _p())] == [
        Check.TEXT_LAYER_NUMBERS
    ]
    # difference arm: the same page with the word back is released
    assert text_layer_numbers_clause(_out(LAYER), _p()) is None


@pytest.mark.parametrize(
    "guard",
    [
        pytest.param({"text_coverage_verdict": "flagged"}, id="ink-unread"),
        pytest.param({"text_coverage_verdict": "unknown"}, id="coverage-unknown"),
        pytest.param({"text_coverage_verdict": ""}, id="coverage-not-run"),
        pytest.param({"has_figures": True}, id="raster-image"),
        pytest.param({"layer_glyphs_undamaged": False}, id="glyph-scans-not-shown-clean"),
        pytest.param({"has_unmapped_math_glyphs": True}, id="unmapped-math-glyphs"),
        pytest.param({"has_corrupt_math": True}, id="corrupt-math"),
        pytest.param({"has_unrecovered_symbol_glyphs": True}, id="unrecovered-symbol-glyphs"),
        pytest.param({"control_byte_digit_hits": 1}, id="control-byte-digit"),
        pytest.param({"minus_as_digit_hits": 1}, id="minus-as-digit"),
    ],
)
def test_each_missing_piece_of_evidence_keeps_the_unknown(guard) -> None:
    """The control arms: the SAME page, one guard removed at a time."""
    assert text_layer_numbers_clause(_out(), _p()) is None, "premise"
    clause = text_layer_numbers_clause(_out(), _p(**guard))
    assert clause is not None
    assert clause.check is Check.TEXT_LAYER_NUMBERS and clause.result is CheckResult.UNKNOWN
    assert clause.reason == NO_NUMBER_REASON


def test_without_a_words_pass_there_is_no_independent_evidence() -> None:
    """The release rule itself, isolated: every class-2 guard holds, but the words check recorded
    nothing (no word reference), so :func:`independent_page_evidence` is False."""
    page = _p(own_layer_word_reference="")
    assert text_layer_words_outcome(_out(), page).entry.result is not TraceResult.PASS
    clause = text_layer_numbers_clause(_out(), page)
    assert clause is not None and clause.reason == NO_NUMBER_REASON


def test_a_blank_page_refused_as_a_scan_keeps_its_true_mark() -> None:
    """The blank and divider pages of run 51270eb5: refused for shape, so no words PASS."""
    page = _p(
        own_layer_text=BLANK,
        own_layer_word_reference=BLANK,
        is_born_digital=False,
        layer_refused_for_shape_alone=True,
    )
    clause = text_layer_numbers_clause(_out(BLANK), page)
    assert clause is not None and clause.reason == NO_NUMBER_REASON


def test_a_reading_that_dropped_the_layers_words_is_not_released() -> None:
    """II-22 shape: the layer prints words, the reading ships a bare image comment."""
    clause = text_layer_numbers_clause(_out("<!-- Image -->"), _p())
    assert clause is not None and clause.result is CheckResult.UNKNOWN


def test_a_page_with_a_failing_check_elsewhere_is_not_released() -> None:
    """Any FAIL or UNKNOWN entry anywhere removes the evidence: a control byte in the text."""
    clause = text_layer_numbers_clause(_out(LAYER + "\x03"), _p())
    assert clause is not None and clause.reason == NO_NUMBER_REASON


def test_a_layer_that_prints_numbers_keeps_its_own_reason() -> None:
    clause = text_layer_numbers_clause(
        _out("Table of contents"),
        _p(own_layer_text="Contents 12 34", own_layer_word_reference="Contents"),
    )
    assert clause is not None and clause.result is CheckResult.UNKNOWN
    assert clause.reason == NO_NUMBER_LAYER_HAS_SOME_REASON.format(count=2)


def test_a_page_with_no_layer_of_its_own_stays_unknown() -> None:
    clause = text_layer_numbers_clause(_out(), _p(own_layer_text="", own_layer_word_reference=""))
    assert clause is not None and clause.result is CheckResult.UNKNOWN


def test_a_page_where_socr_withheld_a_table_is_not_released() -> None:
    from socr.core.result import FailureMode

    out = PageOutput(
        page_num=1,
        text=LAYER,
        status=PageStatus.SUCCESS,
        engine="qwen",
        failure_mode=FailureMode.TABLE_WITHHELD,
    )
    clause = text_layer_numbers_clause(out, _p())
    assert clause is not None and clause.result is CheckResult.UNKNOWN


def test_a_prior_ocr_layer_gets_no_clause_here_as_before() -> None:
    assert text_layer_numbers_clause(_out(), _p(prior_ocr_layer_suspect=True)) is None
    assert numbers_clause_released(_out(), _p(prior_ocr_layer_suspect=True)) is False


# -------------------------------------------------- through the pipeline, both provider states


def _prose_pdf(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), LAYER, fontsize=11, fontname="helv")
    doc.save(path)
    doc.close()
    return path


def _run(
    tmp_path: Path,
    *,
    check_coverage: bool,
    providers: list,
    resumed: bool = False,
    reading=LAYER,
    layer=LAYER,
    want_state: bool = False,
):
    from socr.core.manifest import finalized_page_record

    pdf = tmp_path / "doc.pdf"
    if not pdf.exists():
        _prose_pdf(pdf)
    pipeline = UnifiedPipeline(
        PipelineConfig(
            primary_engine=EngineType.QWEN,
            local_engine=EngineType.QWEN,
            enabled_engines=[EngineType.QWEN],
            agentic=True,
            judge_backend="heuristic",
            quiet=True,
            save_figures=False,
            write_manifest=False,
            native_first=False,
            # the #1050 ink-vs-layer check is part of the evidence; off, it never runs
            check_text_coverage=check_coverage,
        )
    )
    pipeline._scan_root = pdf.parent
    state = DocumentState(handle=DocumentHandle.from_path(pdf))
    ps = state.pages[1]
    if not resumed:  # a resumed page's layer facts are NOT recomputed: only its sidecar speaks
        ps.is_born_digital = True
        ps.layer_glyphs_undamaged = True
        ps.own_layer_text = layer
        ps.own_layer_word_reference = layer

    def _route(page_num, *a, **kw):
        assert not resumed, "a resumed terminal page must not be re-read"
        out = PageOutput(
            page_num=page_num,
            text=reading,
            status=PageStatus.SUCCESS,
            engine="qwen",
            audit_passed=True,
        )
        return PageDecision(page_num=page_num, final_output=out, attempts=[], accepted=True)

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(pipeline, "_available_engines_for_agentic", return_value=providers)
        )
        stack.enter_context(patch("socr.pipeline.orchestrator.route_page", side_effect=_route))
        stack.enter_context(
            patch("socr.pipeline.orchestrator.probe_ollama_idle", return_value=True)
        )
        stack.enter_context(patch.object(pipeline, "_resolve_judge_model", return_value=""))
        pipeline._phase_agentic(state, tmp_path / "out")
        pipeline._phase_assemble(state, tmp_path / "out")  # writes the terminal sidecar
    rec = finalized_page_record(state, 1)
    clauses = [(c.check.value, c.result.value) for c in rec.verdict.clauses]
    if want_state:
        return state, (rec.output.status, rec.output.text, clauses)
    return rec.output.status, rec.output.text, clauses


@pytest.mark.parametrize("providers", [[PROFILE_QWEN_LOCAL], []], ids=["provider", "no-provider"])
def test_the_shipped_page_differs_only_by_the_evidence(tmp_path, providers):
    with_evidence = _run(tmp_path / "a", check_coverage=True, providers=providers)
    without = _run(tmp_path / "b", check_coverage=False, providers=providers)
    if providers:
        assert ("text_layer_numbers", "UNKNOWN") in without[2], "premise: the arm is marked"
        assert with_evidence[2] == []
        assert with_evidence[0] is PageStatus.SUCCESS
        assert without[0] is PageStatus.WARNING
        assert LAYER in with_evidence[1] and LAYER in without[1]
    else:
        # no provider: the page is never read, so the evidence changes nothing
        assert with_evidence == without


@pytest.mark.parametrize("providers", [[PROFILE_QWEN_LOCAL], []], ids=["provider", "no-provider"])
def test_astra_the_dropped_not_ships_marked_through_the_pipeline(tmp_path, providers):
    dropped = "Federal Reserve policy does permit unrestricted lending."
    whole = _run(tmp_path / "a", check_coverage=True, providers=providers)
    cut = _run(tmp_path / "b", check_coverage=True, providers=providers, reading=dropped)
    if providers:
        assert whole[2] == [] and whole[0] is PageStatus.SUCCESS
        assert cut[0] is PageStatus.WARNING
        assert ("text_layer_numbers", "UNKNOWN") in cut[2]
    else:
        assert cut[2] == whole[2]


@pytest.mark.parametrize("providers", [[PROFILE_QWEN_LOCAL], []], ids=["provider", "no-provider"])
def test_the_released_page_stays_released_on_resume(tmp_path, providers):
    """The page is SUCCESS, so run 2 skips it from its terminal sidecar; the restored page has
    no layer facts and no coverage verdict, and must not gain the clause back."""
    first = _run(tmp_path, check_coverage=True, providers=providers)
    if not providers:
        pytest.skip("no provider: the page is never read, so there is nothing to resume")
    again = _run(tmp_path, check_coverage=True, providers=providers, resumed=True)
    assert first[0] is PageStatus.SUCCESS and first[2] == []
    assert again == first


# ---- detector, on real PDFs: the stamp says what the scans found


def _pdf_with(tmp_path: Path, text: str, *, fontname: str = "helv") -> fitz.Page:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=11, fontname=fontname)
    return page


@pytest.mark.parametrize(
    ("text", "undamaged"),
    [
        pytest.param("Federal Reserve forecast heading", True, id="clean"),
        pytest.param("Euler \u00f0", False, id="kimi-euler-eth"),
    ],
)
def test_the_detector_stamps_the_glyph_scans(tmp_path, text, undamaged):
    from socr.core.born_digital import BornDigitalDetector

    page = _pdf_with(tmp_path, text)
    pa = BornDigitalDetector()._assess_page(page, 1)
    assert pa.layer_glyphs_undamaged is undamaged


def test_a_scan_that_raises_is_not_shown_clean(tmp_path):
    from socr.core.born_digital import BornDigitalDetector

    page = _pdf_with(tmp_path, "Federal Reserve forecast heading")
    with patch("socr.core.born_digital.detect_garbled_math", side_effect=RuntimeError("x")):
        assert BornDigitalDetector()._assess_page(page, 1).layer_glyphs_undamaged is False


def test_a_replacement_character_in_the_layer_is_not_shown_clean(tmp_path):
    """Helvetica cannot draw U+FFFD into a test PDF, so the helper is called with the text."""
    from socr.core.born_digital import BornDigitalDetector

    page = _pdf_with(tmp_path, "Federal Reserve forecast heading")
    det = BornDigitalDetector()
    assert det._layer_glyphs_are_undamaged(page, "Federal Reserve forecast heading") is True
    assert det._layer_glyphs_are_undamaged(page, "Federal \ufffd.5 forecast") is False
