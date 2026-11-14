"""#1270 (B4, class 2): a page with no number on either side is not "unchecked".

Before: 24 pages in run 51270eb5 carried ``text_layer_numbers=UNKNOWN`` ("the reading ships
no number") only because they have none: 21 "(This page intentionally blank.)" and two
divider pages. Their layers hold 32-33 characters, under ``MIN_CHARS_FOR_TEXT_LAYER``, so the
detector refuses them as scans and a model reads them.

The owner's ruling stands (CHECKED requires ``compared > 0``). What is added is evidence that
there is nothing to check: the #1294 words check compared the layer's words and none is missing,
the #1050 ink-vs-layer check passed, and the page holds no embedded raster image. Each is a
guard, each has a control arm and a mutant. Without them Greenbook II-22 (a raster chart whose
only layer text is a running header, shipped as ``<!-- Image -->``) would be cleared.
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
    NO_NUMBER_LAYER_HAS_SOME_REASON,
    NO_NUMBER_REASON,
    Check,
    CheckResult,
    text_layer_numbers_clause,
)
from socr.pipeline.agentic import PageDecision
from socr.pipeline.orchestrator import UnifiedPipeline

BLANK = "(This page intentionally blank.)"
DIVIDER = "# International Developments"


def _p(**over) -> SimpleNamespace:
    base = dict(
        own_layer_text=BLANK,
        own_layer_word_reference=BLANK,
        own_layer_figure_words="",
        is_born_digital=False,  # refused as a scan: under the 50-character floor
        prior_ocr_layer_suspect=False,
        has_unrecovered_symbol_glyphs=False,
        table_ladder_disposition=None,
        text_coverage_verdict="passed",
        has_figures=False,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _out(text: str = BLANK) -> PageOutput:
    return PageOutput(page_num=1, text=text, status=PageStatus.SUCCESS, engine="qwen")


@pytest.mark.parametrize(
    ("text", "layer"), [(BLANK, BLANK), (DIVIDER, "International Developments")]
)
def test_a_blank_or_divider_page_with_independent_evidence_carries_no_clause(text, layer):
    assert (
        text_layer_numbers_clause(
            _out(text), _p(own_layer_text=layer, own_layer_word_reference=layer)
        )
        is None
    )


@pytest.mark.parametrize(
    "guard",
    [
        pytest.param({"text_coverage_verdict": "flagged"}, id="ink-unread"),
        pytest.param({"text_coverage_verdict": "unknown"}, id="coverage-unknown"),
        pytest.param({"text_coverage_verdict": ""}, id="coverage-not-run"),
        pytest.param({"has_figures": True}, id="raster-image"),
    ],
)
def test_each_missing_piece_of_evidence_keeps_the_unknown(guard):
    """The control arms: the SAME page, one guard removed at a time."""
    clause = text_layer_numbers_clause(_out(), _p(**guard))
    assert clause is not None
    assert clause.check is Check.TEXT_LAYER_NUMBERS and clause.result is CheckResult.UNKNOWN
    assert clause.reason == NO_NUMBER_REASON


def test_a_reading_that_dropped_the_layers_words_is_not_cleared():
    """II-22 shape: the layer prints words, the reading ships a bare image comment."""
    clause = text_layer_numbers_clause(
        _out("<!-- Image -->"), _p(own_layer_word_reference="Greenbook Part II Domestic")
    )
    assert clause is not None and clause.result is CheckResult.UNKNOWN


def test_a_raster_chart_page_is_not_cleared_even_when_the_words_match():
    """The words check alone cannot clear it: a running header matches while the chart is
    unread. Both the ink check and the image guard stand in the way."""
    page = _p(
        own_layer_text="Greenbook",
        own_layer_word_reference="Greenbook",
        has_figures=True,
        text_coverage_verdict="passed",
    )
    assert text_layer_numbers_clause(_out("Greenbook\n\n<!-- Image -->"), page) is not None
    page.has_figures = False
    page.text_coverage_verdict = "flagged"
    assert text_layer_numbers_clause(_out("Greenbook\n\n<!-- Image -->"), page) is not None


def test_a_layer_that_prints_numbers_keeps_its_own_reason():
    clause = text_layer_numbers_clause(
        _out("Table of contents"),
        _p(own_layer_text="Contents 12 34", own_layer_word_reference="Contents"),
    )
    assert clause is not None and clause.result is CheckResult.UNKNOWN
    assert clause.reason == NO_NUMBER_LAYER_HAS_SOME_REASON.format(count=2)


def test_a_page_with_no_layer_of_its_own_stays_unknown():
    clause = text_layer_numbers_clause(_out(), _p(own_layer_text=""))
    assert clause is not None and clause.result is CheckResult.UNKNOWN


# -------------------------------------------------- through the pipeline, both provider states


def _blank_pdf(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), BLANK, fontsize=11, fontname="helv")
    doc.save(path)
    doc.close()
    return path


def _run(tmp_path: Path, *, check_coverage: bool, providers: list, resumed: bool = False):
    from socr.core.manifest import finalized_page_record

    pdf = tmp_path / "doc.pdf"
    if not pdf.exists():
        _blank_pdf(pdf)
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
            # the #1050 ink-vs-layer check is the independent evidence; off, it never runs
            check_text_coverage=check_coverage,
        )
    )
    pipeline._scan_root = pdf.parent
    state = DocumentState(handle=DocumentHandle.from_path(pdf))
    ps = state.pages[1]
    if not resumed:  # a resumed page's layer facts are NOT recomputed: only its sidecar speaks
        ps.is_born_digital = False
        ps.own_layer_text = BLANK
        ps.own_layer_word_reference = BLANK

    def _route(page_num, *a, **kw):
        assert not resumed, "a resumed terminal page must not be re-read"
        out = PageOutput(
            page_num=page_num,
            text=BLANK,
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
    return rec.output.status, rec.output.text, clauses


@pytest.mark.parametrize("providers", [[PROFILE_QWEN_LOCAL], []], ids=["provider", "no-provider"])
def test_the_shipped_blank_page_differs_only_by_the_clause(tmp_path, providers):
    with_evidence = _run(tmp_path / "a", check_coverage=True, providers=providers)
    without = _run(tmp_path / "b", check_coverage=False, providers=providers)
    if providers:
        assert ("text_layer_numbers", "UNKNOWN") in without[2], "premise: the arm is marked"
        assert ("text_layer_numbers", "UNKNOWN") not in with_evidence[2]
        assert with_evidence[0] is PageStatus.SUCCESS
        assert without[0] is PageStatus.WARNING
        # the BYTES of the page body are the same either way (the marked arm adds a mark)
        assert BLANK in with_evidence[1] and BLANK in without[1]
    else:
        # no provider: the page is never read, so the evidence changes nothing
        assert with_evidence == without


@pytest.mark.parametrize("providers", [[PROFILE_QWEN_LOCAL], []], ids=["provider", "no-provider"])
def test_the_cleared_blank_page_stays_cleared_on_resume(tmp_path, providers):
    """The page is SUCCESS, so run 2 skips it from its terminal sidecar; the restored page has
    no layer facts and no coverage verdict, and must not gain the clause back."""
    first = _run(tmp_path, check_coverage=True, providers=providers)
    if not providers:
        pytest.skip("no provider: the page is never read, so there is nothing to resume")
    again = _run(tmp_path, check_coverage=True, providers=providers, resumed=True)
    assert first[0] is PageStatus.SUCCESS and first[2] == []
    assert again == first


def test_a_blank_page_the_words_check_recorded_nothing_on_is_cleared_by_the_whole_layer():
    """#1294's page-edge excusal empties the reference of a blank page (16 of the 24 blank and
    divider pages of run 51270eb5). The whole layer is then held to the page, with no excusal."""
    page = _p(own_layer_word_reference="")
    assert text_layer_numbers_clause(_out(), page) is None


@pytest.mark.parametrize(
    "over",
    [
        pytest.param(
            {"own_layer_text": "Greenbook Part II Domestic"}, id="reading-drops-layer-words"
        ),
        pytest.param({"text_coverage_verdict": "flagged"}, id="ink-unread"),
        pytest.param({"has_figures": True}, id="raster-image"),
        pytest.param({"prior_ocr_layer_suspect": True}, id="prior-ocr"),
    ],
)
def test_the_whole_layer_route_keeps_every_guard(over):
    page = _p(own_layer_word_reference="", **over)
    clause = text_layer_numbers_clause(
        _out("<!-- Image -->" if "own_layer_text" in over else BLANK), page
    )
    if over.get("prior_ocr_layer_suspect"):
        assert clause is None  # the numbers clause defers to #1156's own mark, as it always did
    else:
        assert clause is not None and clause.result is CheckResult.UNKNOWN


def test_a_layer_with_no_compared_word_clears_nothing():
    """Agreement over nothing is vacuous: a layer of one-letter tokens compares zero words."""
    page = _p(own_layer_text="a b", own_layer_word_reference="")
    clause = text_layer_numbers_clause(_out("a b"), page)
    assert clause is not None and clause.result is CheckResult.UNKNOWN


def test_a_page_where_socr_withheld_a_table_is_not_cleared_by_the_whole_layer():
    """The layer's table words are missing BY DESIGN there; a words match proves nothing."""
    from socr.core.result import FailureMode

    out = PageOutput(
        page_num=1,
        text=BLANK,
        status=PageStatus.SUCCESS,
        engine="qwen",
        failure_mode=FailureMode.TABLE_WITHHELD,
    )
    clause = text_layer_numbers_clause(out, _p(own_layer_word_reference=""))
    assert clause is not None and clause.result is CheckResult.UNKNOWN
