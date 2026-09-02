"""Cold review round 2 on the P3+P5 branch (`fix/p3-p5-judged-bytes-ship`).

Round 1 made the crop reread's patched text go back through the judge. Round 2
found that the re-judge was not wired into the page's state, its events, or its
metering, and that the P3 invariant was still overclaimed:

1. **BLOCKING** — an ACCEPTING re-judge copied only the text. The page kept
   `audit_passed=False`, `native_table_structure_failed=True` and the
   fail-closed floor PNG that ladder exhaustion had stamped, so
   `_grid_authored_attempt` still refused it and the page shipped the
   structure-class floor instead of the bytes the judge had just accepted. An
   accepting re-judge must promote through the same state transitions
   `route_page`'s acceptance uses.
2. **BLOCKING** — "judged bytes are shipped bytes" is scoped by ruling: no
   post-verdict step may ADD or ALTER content. Two SUBTRACTIVE sanitizers are
   enumerated exceptions and stay; the equation sidecar is an ADDITIVE step and
   must be guarded.
3. **BLOCKING** — the composed judge closes over `state.events`, so a REFUSED
   re-judge left its `native_table_verifier_hard_fail` behind and
   `tables_trust.json` marked the shipped (accepted, clean) bytes untrusted.
4. **SHOULD-FIX** — the extra judge call was neither metered nor journalled.
5. **SHOULD-FIX** — `docs/ARCHITECTURE.md` described an ordering the code does
   not implement (docs only; no test).

Every case pins a DIFFERENCE between two runs of the real pipeline in one
process (CLAUDE.md / #257), never an absolute status tuple. Hermetic: provider
ladder, judge model resolver, backend probes, the OCR call and the crop reader
are all patched.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

fitz = pytest.importorskip("fitz")

from ocr_output_contract import split_native_pages  # noqa: E402

from socr.core.config import EngineType, PipelineConfig  # noqa: E402
from socr.core.providers import PROFILE_GEMINI, PROFILE_QWEN_LOCAL  # noqa: E402
from socr.core.result import PageOutput, PageStatus  # noqa: E402
from socr.pipeline.orchestrator import UnifiedPipeline  # noqa: E402
from socr.tables import locate_tables as _real_locate_tables  # noqa: E402
from socr.tables.extract import TableCropExtractor  # noqa: E402

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

#: One digit the native value guard proves wrong -> CERTAIN_FAIL on every rung.
_CERTAIN_FAIL = _md(
    [
        ("Total effect of decisions", ["42.8", "30.5", "2.6"]),
        ("September energy package", ["43.2", "26.8", "3.7"]),
        ("Energy price guarantee", ["24.8", "26.8", "9.9"]),
        ("Energy bill relief scheme", ["18.4", "9.1", "5.5"]),
    ]
)


def _build_fixture_pdf(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
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
    candidate_text: str = _PERFECT,
    dual_pass_tables: bool = False,
    force_score_signal: bool | None = None,
    crop_patch_text: str | None = None,
    escalate_ambiguous_tables: bool = False,
):
    """One real agentic run over a one-page table document.

    ``crop_patch_text`` stubs ``_reread_page_tables`` with exactly the mutation
    the real one performs (``ps.best_output.text = <patched>``), so the crop
    lane is exercised without a live crop VLM.
    """
    pdf_path = _build_fixture_pdf(tmp_path)
    out_dir = tmp_path / "out"

    config = PipelineConfig(
        primary_engine=EngineType.QWEN,
        agentic=True,
        judge_backend="heuristic",
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

    original_scoring = UnifiedPipeline._surface_table_scoring

    def _spy_scoring(self, state, page_num, ps, bo):
        real = original_scoring(self, state, page_num, ps, bo)
        return real if force_score_signal is None else force_score_signal

    original_reread = UnifiedPipeline._reread_page_tables

    def _stub_reread(self, state, page_num, raw_crops, extractor):
        if crop_patch_text is None:
            return original_reread(self, state, page_num, raw_crops, extractor)
        state.pages[page_num].best_output.text = crop_patch_text
        return 1, 0

    phase_states: list = []
    original_assemble = UnifiedPipeline._phase_assemble

    def _spy_assemble(self, state, output_dir):
        phase_states.append(state)
        return original_assemble(self, state, output_dir)

    with (
        patch.object(
            pipeline,
            "_available_engines_for_agentic",
            return_value=[PROFILE_QWEN_LOCAL, PROFILE_GEMINI],
        ),
        patch.object(pipeline, "_resolve_judge_model", return_value=""),
        patch.object(pipeline, "_is_agentic_trusted_native", return_value=False),
        patch.object(pipeline, "_probe_backend_idle", return_value=True),
        patch("socr.pipeline.orchestrator.probe_ollama_idle", return_value=True),
        patch("socr.pipeline.orchestrator.probe_openai_server_idle", return_value=True),
        patch.object(
            UnifiedPipeline,
            "_run_engine_on_pages",
            autospec=True,
            side_effect=lambda self, *a, **k: _stub_run_engine_on_pages(*a, **k),
        ),
        patch.object(UnifiedPipeline, "_surface_table_scoring", _spy_scoring),
        patch.object(UnifiedPipeline, "_reread_page_tables", _stub_reread),
        patch.object(UnifiedPipeline, "_phase_assemble", _spy_assemble),
        patch.object(UnifiedPipeline, "_resolve_crop_vlm_model", lambda self: "qwen-test-crop"),
        patch("socr.tables.extract.make_table_reader", MagicMock(return_value=MagicMock())),
        patch.object(TableCropExtractor, "extract", MagicMock(return_value=[MagicMock()])),
        patch("socr.tables.locate_tables", MagicMock(side_effect=_real_locate_tables)),
    ):
        result = pipeline.process(pdf_path, out_dir)

    state = phase_states[0]
    # ``config`` and ``pdf_path`` are returned so a later round can resume the
    # page this run left on disk with the SAME run fingerprint. Additive.
    return {
        "result": result,
        "out_dir": out_dir,
        "state": state,
        "ps": state.pages[1],
        "config": config,
        "pdf_path": pdf_path,
    }


def _final_page_one(out_dir: Path) -> str:
    md = [p for p in out_dir.rglob("*.md") if "pages" not in p.parts and p.name != "README.md"]
    assert md, f"no final assembled markdown under {out_dir}"
    pages = split_native_pages(md[0].read_text(encoding="utf-8"))
    assert pages
    return pages[0]


def _page_shape(ps) -> tuple:
    """The SUCCESS-shaped page state a first-time acceptance leaves behind."""
    bo = ps.best_output
    return (
        bo.audit_passed,
        bo.status,
        bo.failure_mode,
        bool(getattr(ps, "native_table_structure_failed", False)),
        bool(getattr(ps, "d3_floor_png_ref", "")),
        bool(getattr(ps, "scanned_table_evidence_failed", False)),
    )


def _page_event_kinds(state) -> list[str]:
    return [e.kind for e in state.events if getattr(e, "page_num", None) == 1]


# ---------------------------------------------------------------------------
# Finding 1 — an accepting re-judge must promote like a first-time acceptance
# ---------------------------------------------------------------------------


class TestAcceptedRejudgeIsPromoted:
    def test_exhausted_ladder_recovered_by_the_crop_matches_a_direct_acceptance(
        self, tmp_path: Path
    ) -> None:
        """Both rungs are CERTAIN_FAIL-rejected, the ladder exhausts, the crop
        repairs the grid and the same judge accepts the repaired candidate.

        The page must then ship exactly what a direct acceptance of those bytes
        ships, in exactly the same state -- otherwise the fail-closed floor
        wins and the recovery is invisible."""
        recovered = _run_pipeline(
            tmp_path / "recovered",
            candidate_text=_CERTAIN_FAIL,
            dual_pass_tables=True,
            crop_patch_text=_PERFECT,
        )
        direct = _run_pipeline(tmp_path / "direct", candidate_text=_PERFECT)

        assert _final_page_one(direct["out_dir"]) == _PERFECT, (
            "control: a direct acceptance of these bytes must ship them"
        )
        assert _final_page_one(recovered["out_dir"]) == _PERFECT
        assert _page_shape(recovered["ps"]) == _page_shape(direct["ps"])


# ---------------------------------------------------------------------------
# Finding 2 — post-verdict helpers: subtractive exceptions, guarded additions
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z]+|\d+(?:[.,]\d+)?")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text or ""))


class TestPostVerdictHelpersAreSubtractive:
    """Enumerated exceptions to "judged bytes are shipped bytes": a post-verdict
    step may REMOVE content, never add or alter it."""

    def test_image_ref_sanitizer_only_removes_tokens(self, tmp_path: Path) -> None:
        from socr.core.state import DocumentState

        pipeline = UnifiedPipeline(PipelineConfig(quiet=True, save_figures=False))
        state = MagicMock(spec=DocumentState)
        state.events = []
        state.pages = {}
        before = (
            "Revenue rose to 42.8 in 2024.\n\n"
            "![chart](https://i.imgur.com/fabricated9999.png)\n\n"
            "![missing](image-url)\n\n"
            "| a | b |\n| --- | --- |\n| 1.5 | 2.5 |\n"
        )
        out = PageOutput(page_num=1, text=before, status=PageStatus.SUCCESS, engine="qwen")
        pipeline._sanitize_agentic_page_image_refs(state, 1, out, tmp_path)
        assert _tokens(out.text) <= _tokens(before), (
            "the image-ref sanitizer must be subtractive: no token may appear in "
            "its output that was not in its input"
        )

    def test_repetition_guard_only_removes_tokens(self) -> None:
        from socr.core.state import DocumentState

        state = MagicMock(spec=DocumentState)
        state.events = []
        row = "| Energy price guarantee | 24.8 | 26.8 | 3.7 |"
        before = "| a | b |\n| --- | --- |\n" + "\n".join([row] * 40) + "\n| tail | 9.9 |"
        out = PageOutput(page_num=1, text=before, status=PageStatus.SUCCESS, engine="qwen")
        UnifiedPipeline._guard_agentic_page_table_repetition(state, 1, out, False)
        assert out.text != before, "the fixture must actually trigger the guard"
        assert _tokens(out.text) <= _tokens(before), (
            "the repetition guard must be subtractive: every kept line is "
            "byte-identical to one that was already there"
        )


# The equation-sidecar guard cases that stood here stubbed
# ``process_equation_region`` outright, which bypassed the very choke point that
# owns the delimiter rule. Round 3 replaced them with cases that patch only the
# crop reader, so PR #518's real ``contract_delimiter_violation`` and
# ``region_presence_verdict`` run: see
# ``tests/test_p35_cold_review_round3.py::TestEquationGuardsAreMainsGuards``.
# The coverage moved and got stronger; it was not dropped.


# ---------------------------------------------------------------------------
# Finding 3 — a refused re-judge must not change the trust of shipped bytes
# ---------------------------------------------------------------------------


class TestRefusedRejudgeLeavesNoTrace:
    def test_refused_rejudge_events_match_the_no_crop_run(self, tmp_path: Path) -> None:
        """Rung 1 is accepted cleanly; the crop then patches in a digit the
        value guard proves wrong. The re-judge emits a hard-fail and refuses.
        The clean accepted bytes ship -- so the page's audit trail must be the
        no-crop trail plus exactly one `table_reread_rejudged`."""
        refused = _run_pipeline(
            tmp_path / "refused",
            dual_pass_tables=True,
            force_score_signal=True,
            crop_patch_text=_CERTAIN_FAIL,
        )
        baseline = _run_pipeline(
            tmp_path / "baseline",
            dual_pass_tables=False,
            force_score_signal=True,
        )

        assert _final_page_one(refused["out_dir"]) == _final_page_one(baseline["out_dir"])

        refused_kinds = _page_event_kinds(refused["state"])
        baseline_kinds = _page_event_kinds(baseline["state"])
        assert refused_kinds.count("table_reread_rejudged") == 1
        assert [k for k in refused_kinds if k != "table_reread_rejudged"] == baseline_kinds, (
            "a refused candidate's judge-side events describe bytes that never "
            "ship; they must not be appended to the page's audit trail"
        )

    def test_refused_rejudge_does_not_change_the_trust_result(self, tmp_path: Path) -> None:
        refused = _run_pipeline(
            tmp_path / "refused",
            dual_pass_tables=True,
            force_score_signal=True,
            crop_patch_text=_CERTAIN_FAIL,
        )
        baseline = _run_pipeline(
            tmp_path / "baseline",
            dual_pass_tables=False,
            force_score_signal=True,
        )

        def _trust(out_dir: Path):
            found = list(out_dir.rglob("tables_trust.json"))
            return json.loads(found[0].read_text(encoding="utf-8")) if found else None

        assert _trust(refused["out_dir"]) == _trust(baseline["out_dir"]), (
            "the trust result of the shipped bytes must not depend on whether a "
            "refused crop candidate was judged along the way"
        )


# ---------------------------------------------------------------------------
# Finding 4 — the extra judge call is metered
# ---------------------------------------------------------------------------


class TestRejudgeIsMetered:
    def test_rejudge_adds_one_attributed_engine_run(self, tmp_path: Path) -> None:
        rejudged = _run_pipeline(
            tmp_path / "rejudged",
            dual_pass_tables=True,
            force_score_signal=True,
            crop_patch_text=_CERTAIN_FAIL,
        )
        baseline = _run_pipeline(
            tmp_path / "baseline",
            dual_pass_tables=False,
            force_score_signal=True,
        )

        extra = len(rejudged["state"].engine_runs) - len(baseline["state"].engine_runs)
        assert extra == 1, "a second judge call must appear exactly once in the run journal"

        # Round 3 corrected the attribution: the extra call is a JUDGE call, so
        # it is priced by the judge that ran it, not by the OCR rung that won
        # the page. This run judges heuristically, which costs a known 0.00.
        new_run = rejudged["state"].engine_runs[-1]
        assert new_run.engine == (rejudged["state"].agentic_judge_model or "heuristic")
        assert new_run.cost == 0.0
        assert new_run.engine != PROFILE_QWEN_LOCAL.engine.value, (
            "a judge call must not be journalled as the OCR engine that won the page"
        )

    def test_rejudge_event_names_the_judge_not_the_ocr_engine(self, tmp_path: Path) -> None:
        rejudged = _run_pipeline(
            tmp_path / "rejudged",
            dual_pass_tables=True,
            force_score_signal=True,
            crop_patch_text=_CERTAIN_FAIL,
        )
        events = [
            e for e in rejudged["state"].events if getattr(e, "kind", "") == "table_reread_rejudged"
        ]
        assert events
        assert (events[0].data or {}).get("judge_model"), (
            "the event must name the judge that incurred the extra call"
        )
