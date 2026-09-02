"""P5 (GH-513 follow-up, docs/log/2026-09-01_conceptual-revision.md): dual-pass
crop reread becomes escalate-on-signal.

Before this ticket, the PP-3 in-loop table reread
(``UnifiedPipeline._reread_page_tables``, wired at orchestrator.py
~3804-3836) ran on EVERY accepted OCR table page whenever
``dual_pass_tables`` was True -- an unconditional crop-read-and-reconcile
pass, paid even on a page the judge accepted cleanly with no disagreement.
``dual_pass_tables`` defaulted True, so this was the trunk, not the tail.

After the fix, the crop reread is an ESCALATION TOOL: it only fires when the
same signal that drives the existing GH-96 escalation lane
(``_table_page_needs_escalation``, or a real judge CERTAIN_FAIL / structural
rejection) actually fires for that page. A clean accept -- no disagreement
with the native layer -- must cost zero crop-reader calls, with the flag on
or off. ``dual_pass_tables`` also defaults False now (see
tests/test_dual_pass_tables.py::test_phase_disabled_flag_default and
tests/test_r174b_config_schema.py).

Every case here pins a DIFFERENCE (call count under a controlled condition),
never an absolute status/failure-mode tuple -- CI has no provider, and the
provider-dependent floor legitimately differs there (CLAUDE.md's "no-provider
trap", PR #253/#257).

Hermetic: ``_available_engines_for_agentic`` is patched to a fixed profile
list per scenario; the OCR call (``_run_engine_on_pages``) is stubbed; the
real crop reader (``TableCropExtractor.extract``) is replaced with a counting
mock so a signal-fired scenario never actually reaches Ollama/network; the
crop VLM model is resolved through a patched deterministic judge-model resolver
rather than through any live probe.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

fitz = pytest.importorskip("fitz")

from socr.core.config import EngineType, PipelineConfig  # noqa: E402
from socr.core.providers import PROFILE_GEMINI, PROFILE_QWEN_LOCAL  # noqa: E402
from socr.core.result import PageOutput, PageStatus  # noqa: E402
from socr.pipeline.agentic import route_page as _real_route_page  # noqa: E402
from socr.pipeline.orchestrator import UnifiedPipeline  # noqa: E402
from socr.tables import locate_tables as _real_locate_tables  # noqa: E402
from socr.tables.extract import TableCropExtractor  # noqa: E402

# ---------------------------------------------------------------------------
# Fixture: a real, well-formed 4-row/3-column ruled table (mirrors
# tests/test_gh96_escalation_lane.py, whose ``_table_page_needs_escalation``
# coverage this reuses the same shape for). ``_PERFECT`` matches the native
# layer exactly -> no signal. ``_SHIFTED`` drops a whole row's values -> a
# REAL, calibrated disagreement signal via the same code path GH-96 drives.
# ---------------------------------------------------------------------------

_ROWS = [
    ("Total effect of decisions", ["42.8", "30.5", "2.6"]),
    ("September energy package", ["43.2", "26.8", "3.7"]),
    ("Energy price guarantee", ["24.8", "26.8", "3.7"]),
    ("Energy bill relief scheme", ["18.4", "9.1", "5.5"]),
]


def _md(rows: list[tuple[str, list[str]]]) -> str:
    out = ["| | c1 | c2 | c3 |", "| --- | --- | --- | --- |"]
    for label, values in rows:
        out.append(f"| {label} | " + " | ".join(values) + " |")
    return "\n".join(out)


_PERFECT = _md(_ROWS)
_SHIFTED = _md(
    [
        ("Total effect of decisions", ["42.8", "30.5", "2.6"]),
        ("September energy package", ["", "", ""]),
        ("Energy price guarantee", ["43.2", "26.8", "3.7"]),
        ("Energy bill relief scheme", ["24.8", "26.8", "3.7"]),
    ]
)
_CERTAIN_FAIL = _md(
    [
        ("Total effect of decisions", ["42.8", "30.5", "2.6"]),
        ("September energy package", ["43.2", "26.8", "3.7"]),
        ("Energy price guarantee", ["24.8", "26.8", "9.9"]),
        ("Energy bill relief scheme", ["18.4", "9.1", "5.5"]),
    ]
)


def _build_fixture_pdf(tmp_path: Path) -> Path:
    doc = fitz.open()
    pg = doc.new_page()
    y = 200.0
    for label, values in _ROWS:
        pg.insert_text((60.0, y), label, fontsize=9)
        for x, v in zip((300.0, 360.0, 420.0), values):
            pg.insert_text((x, y), v, fontsize=9)
        y += 18.0
    pg.draw_line(fitz.Point(50, 190), fitz.Point(470, 190))
    pg.draw_line(fitz.Point(50, y), fitz.Point(470, y))
    pdf_path = tmp_path / "doc.pdf"
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


def _run_pipeline(
    tmp_path: Path,
    *,
    candidate_text: str,
    dual_pass_tables: bool,
    available_profiles: list,
    escalate_ambiguous_tables: bool = True,
    capture_routing: bool = False,
):
    """Run the real agentic pipeline end to end and report how many times the
    crop reader and its model-resolution step were reached.

    Returns (result, extract_call_count, resolve_crop_model_call_count), or that
    tuple plus routing call/event details when ``capture_routing`` is true.
    """
    pdf_path = _build_fixture_pdf(tmp_path)
    out_dir = tmp_path / "out"

    config = PipelineConfig(
        primary_engine=EngineType.QWEN,
        agentic=True,
        judge_backend="heuristic",
        # The resolver is patched below, so no live provider probe is possible;
        # the crop path must still resolve this deterministic model lazily.
        judge_model=None,
        enabled_engines=[EngineType.QWEN, EngineType.GEMINI],
        quiet=True,
        save_figures=False,
        write_manifest=False,
        native_first=True,
        dual_pass_tables=dual_pass_tables,
        escalate_ambiguous_tables=escalate_ambiguous_tables,
        table_judge_ladder=False,
    )
    pipeline = UnifiedPipeline(config)

    def _stub_run_engine_on_pages(
        state, page_nums, enhancement_pages, engine_type, label, profile=None
    ):
        return [
            PageOutput(
                page_num=page_nums[0],
                text=candidate_text,
                status=PageStatus.SUCCESS,
                engine="qwen",
                confidence=0.9,
            )
        ]

    extract_mock = MagicMock(return_value=[])
    locate_mock = MagicMock(side_effect=_real_locate_tables)
    reread_calls: list[tuple] = []
    decisions: list = []
    phase_states: list = []
    extractor_init_calls: list[int] = []
    resolve_calls: list[int] = []
    original_resolve = UnifiedPipeline._resolve_crop_vlm_model
    original_reread = UnifiedPipeline._reread_page_tables
    original_assemble = UnifiedPipeline._phase_assemble
    original_extractor_init = TableCropExtractor.__init__

    def _spy_resolve(self):
        resolve_calls.append(1)
        return original_resolve(self)

    def _spy_reread(self, *args, **kwargs):
        reread_calls.append(args)
        return original_reread(self, *args, **kwargs)

    def _spy_route_page(*args, **kwargs):
        decision = _real_route_page(*args, **kwargs)
        decisions.append(decision)
        return decision

    def _spy_assemble(self, state, output_dir):
        phase_states.append(state)
        return original_assemble(self, state, output_dir)

    def _spy_extractor_init(self, *args, **kwargs):
        extractor_init_calls.append(1)
        original_extractor_init(self, *args, **kwargs)

    with (
        patch.object(pipeline, "_available_engines_for_agentic", return_value=available_profiles),
        patch.object(pipeline, "_resolve_judge_model", return_value="fake-vlm-model"),
        patch.object(pipeline, "_is_agentic_trusted_native", return_value=False),
        patch.object(
            UnifiedPipeline,
            "_run_engine_on_pages",
            autospec=True,
            side_effect=lambda self, *a, **k: _stub_run_engine_on_pages(*a, **k),
        ),
        patch.object(TableCropExtractor, "__init__", _spy_extractor_init),
        patch.object(TableCropExtractor, "extract", extract_mock),
        patch("socr.tables.locate_tables", locate_mock),
        patch.object(UnifiedPipeline, "_reread_page_tables", _spy_reread),
        patch("socr.pipeline.orchestrator.route_page", side_effect=_spy_route_page),
        patch.object(UnifiedPipeline, "_phase_assemble", _spy_assemble),
        patch.object(UnifiedPipeline, "_resolve_crop_vlm_model", _spy_resolve),
    ):
        result = pipeline.process(pdf_path, out_dir)

    basic = (result, extract_mock.call_count, len(resolve_calls))
    if not capture_routing:
        return basic
    return basic, {
        "locate_calls": locate_mock.call_count,
        "reread_calls": len(reread_calls),
        "extractor_init_calls": len(extractor_init_calls),
        "decisions": decisions,
        "events": phase_states[0].events if phase_states else [],
    }


_BOTH_TIERS = [PROFILE_QWEN_LOCAL, PROFILE_GEMINI]
_LOCAL_ONLY = [PROFILE_QWEN_LOCAL]


class TestCropRereadIsEscalationGated:
    """Difference pins: the crop reader's call count under (flag, signal)."""

    @pytest.mark.parametrize("dual_pass_tables", [False, True])
    def test_clean_acceptance_never_enters_crop_signal_path(
        self, tmp_path: Path, dual_pass_tables: bool
    ) -> None:
        """A clean page costs no locate, extract, or reread calls in either state."""
        (_result, extract_calls, _resolve_calls), capture = _run_pipeline(
            tmp_path,
            candidate_text=_PERFECT,
            dual_pass_tables=dual_pass_tables,
            available_profiles=_BOTH_TIERS,
            capture_routing=True,
        )

        assert capture["locate_calls"] == 0
        assert extract_calls == 0
        assert capture["reread_calls"] == 0
        assert capture["extractor_init_calls"] == 0

    def test_clean_accept_flag_off_no_crop_calls(self, tmp_path: Path) -> None:
        _result, extract_calls, _resolve_calls = _run_pipeline(
            tmp_path,
            candidate_text=_PERFECT,
            dual_pass_tables=False,
            available_profiles=_BOTH_TIERS,
        )
        assert extract_calls == 0

    def test_clean_accept_flag_on_no_crop_calls(self, tmp_path: Path) -> None:
        """The key P5 regression: with no disagreement signal, turning the
        flag ON must NOT make every accepted table page pay for a crop
        reread. Pre-fix this fires unconditionally whenever the flag is on."""
        _result, extract_calls, _resolve_calls = _run_pipeline(
            tmp_path,
            candidate_text=_PERFECT,
            dual_pass_tables=True,
            available_profiles=_BOTH_TIERS,
        )
        assert extract_calls == 0, (
            "a clean, agreeing table page must cost zero crop-reader calls "
            "even with dual_pass_tables=True -- the reread is an escalation "
            "tool, not a trunk pass over every accepted table page"
        )

    def test_fired_signal_flag_on_triggers_exactly_one_crop_call(self, tmp_path: Path) -> None:
        """A REAL disagreement signal -- the same
        ``_table_page_needs_escalation`` predicate GH-96's escalation lane
        already uses -- with the flag on must trigger the crop reread."""
        _result, extract_calls, resolve_calls = _run_pipeline(
            tmp_path,
            candidate_text=_SHIFTED,
            dual_pass_tables=True,
            available_profiles=_BOTH_TIERS,
        )
        assert extract_calls == 1
        assert resolve_calls >= 1, "the crop model must be resolved once a signal fires"

    def test_fired_signal_flag_off_no_crop_calls(self, tmp_path: Path) -> None:
        """The flag is the master switch: even a fired signal must not reach
        the crop reader while ``dual_pass_tables`` is off."""
        _result, extract_calls, _resolve_calls = _run_pipeline(
            tmp_path,
            candidate_text=_SHIFTED,
            dual_pass_tables=False,
            available_profiles=_BOTH_TIERS,
        )
        assert extract_calls == 0

    def test_fired_signal_without_escalation_provider_still_triggers_crop_reread(
        self, tmp_path: Path
    ) -> None:
        """The crop reread is selected by the signal + flag, not by whether a
        non-local escalation provider happens to be available -- it must
        still fire in --strict-local-shaped runs (only the local profile
        available)."""
        (_result, extract_calls, _resolve_calls), capture = _run_pipeline(
            tmp_path,
            candidate_text=_SHIFTED,
            dual_pass_tables=True,
            available_profiles=_LOCAL_ONLY,
            capture_routing=True,
        )
        assert extract_calls == 1
        assert capture["locate_calls"] == 1
        assert capture["reread_calls"] == 1
        assert capture["extractor_init_calls"] == 1

    def test_crop_model_not_resolved_when_flag_off(self, tmp_path: Path) -> None:
        """Lazy construction: with the flag off, the crop VLM model must
        never be resolved at all -- not even once per document."""
        _result, _extract_calls, resolve_calls = _run_pipeline(
            tmp_path,
            candidate_text=_SHIFTED,
            dual_pass_tables=False,
            available_profiles=_BOTH_TIERS,
        )
        assert resolve_calls == 0

    def test_crop_model_not_resolved_on_a_clean_page_even_with_flag_on(
        self, tmp_path: Path
    ) -> None:
        """Lazy construction: with the flag on but no signal ever firing on
        this document, the crop VLM model must never be resolved either --
        resolution happens lazily on the first eligible fired signal, not
        eagerly at document setup."""
        _result, _extract_calls, resolve_calls = _run_pipeline(
            tmp_path,
            candidate_text=_PERFECT,
            dual_pass_tables=True,
            available_profiles=_BOTH_TIERS,
        )
        assert resolve_calls == 0, (
            "pre-fix the extractor (and its model resolution) is built eagerly "
            "at document scope whenever dual_pass_tables is True, regardless "
            "of whether any page ever needs it"
        )

    def test_fired_signal_is_real_native_rejection_and_multi_attempt_route(
        self, tmp_path: Path
    ) -> None:
        """The fired case must come from real routing facts.

        The shifted candidate reaches the real native verifier, which rejects
        it and makes the real route_page continue from the local rung to the
        cloud rung. The crop lane is then selected exactly once for that page.
        """
        (_result, extract_calls, _resolve_calls), capture = _run_pipeline(
            tmp_path,
            candidate_text=_CERTAIN_FAIL,
            dual_pass_tables=True,
            available_profiles=_BOTH_TIERS,
            capture_routing=True,
        )

        assert extract_calls == 1
        assert capture["locate_calls"] == 1
        assert capture["reread_calls"] == 1
        assert any(e.kind == "native_table_verifier_hard_fail" for e in capture["events"])
        assert any(len(decision.attempts) >= 2 for decision in capture["decisions"])
