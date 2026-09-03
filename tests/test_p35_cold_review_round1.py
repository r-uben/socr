"""Cold review round 1 on the P3+P5 branch (`fix/p3-p5-judged-bytes-ship`).

Three findings, one file, each pinned as a DIFFERENCE (CLAUDE.md / #257):

1. **medium** — the crop reread still mutated ``best_output.text`` AFTER
   ``route_page`` had accepted a reading, so once P5's signal fired the P3
   invariant ("judged bytes are shipped bytes") reopened on exactly the path
   P5 added. ``tests/test_p3_judged_bytes_ship.py`` cannot see it: it runs
   with ``dual_pass_tables=False``. Ruling step 4 (2026-09-01 conceptual
   revision): a reread is an escalation tool, run BEFORE the verdict, never
   after an accept. A patched reread is therefore a NEW CANDIDATE and must go
   back through the same judge; if the judge refuses it, the previously
   accepted bytes ship and the refusal is an audit event.
2. **medium** — the new scoring gate (``not is_native and bo.engine not in
   {"native", "chart_asset"}``) dropped #123 TICKET-C2 table scoring for
   pages that reached it before the branch, native-bypass table pages
   included. Scoring is the only surface ``table_not_scorable`` and
   ``table_unexplained_lanes`` ever reach.
3. **low** — ``_score_table_signal or _route_table_signal`` was passed as
   ``needs_escalation`` to ``_escalate_table_page``. Route evidence is true
   when an earlier rung was rejected even if the accepted winner scores 100%
   against the native layer, so GH-96 paid a cloud re-read that
   ``decide_escalation`` cannot keep. The score alone drives GH-96; route
   evidence stays with the crop reread.

Hermetic per CLAUDE.md: the provider ladder, the judge model resolver, the
backend probes, the OCR call and the crop reader are all patched, so nothing
here consults an installed provider or Ollama.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

fitz = pytest.importorskip("fitz")

from ocr_output_contract import split_native_pages  # noqa: E402

from socr.core.config import EngineType, PipelineConfig  # noqa: E402
from socr.core.providers import PROFILE_GEMINI, PROFILE_QWEN_LOCAL  # noqa: E402
from socr.core.result import PageOutput, PageStatus  # noqa: E402
from socr.pipeline.agentic import NativeTableVerifierJudge  # noqa: E402
from socr.pipeline.orchestrator import UnifiedPipeline  # noqa: E402
from socr.tables import locate_tables as _real_locate_tables  # noqa: E402
from socr.tables.extract import TableCropExtractor  # noqa: E402

# ---------------------------------------------------------------------------
# Fixture: the same real ruled 4x3 table used by tests/test_p5_reread_on_signal.py
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

#: A patched reread the judge MUST refuse: one value swapped, which the native
#: value guard proves wrong (CERTAIN_FAIL).
_PATCH_REFUSED = _md(
    [
        ("Total effect of decisions", ["42.8", "30.5", "2.6"]),
        ("September energy package", ["43.2", "26.8", "3.7"]),
        ("Energy price guarantee", ["24.8", "26.8", "9.9"]),
        ("Energy bill relief scheme", ["18.4", "9.1", "5.5"]),
    ]
)

#: A patched reread the judge accepts: identical grid, one extra caption line.
_PATCH_ACCEPTED = _PERFECT + "\n\nSource: reconciled against the crop reading."


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
    candidate_text: str = _PERFECT,
    dual_pass_tables: bool = False,
    available_profiles: list | None = None,
    escalate_ambiguous_tables: bool = False,
    is_native: bool = False,
    force_score_signal: bool | None = None,
    crop_patch_text: str | None = None,
):
    """Drive the real agentic pipeline once and report what it did.

    ``crop_patch_text`` replaces ``_reread_page_tables`` with a stub that
    performs exactly the post-accept mutation the real one performs
    (``ps.best_output.text = <patched>``), which is the defect under test in
    finding 1 -- without needing a live crop VLM.
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
        table_judge_ladder=False,  # docs/log/2026-09-03_p1-prep-latch-and-audit.md
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

    # The text of the LAST accepting judge verdict on page 1. The P3 invariant
    # is that this is exactly what ships, on every path.
    accepted: dict[str, str] = {}
    original_assess = NativeTableVerifierJudge.assess

    def _spy_assess(self, output, provider):
        decision = original_assess(self, output, provider)
        if decision.accept and output.page_num == 1:
            accepted["text"] = output.text
        return decision

    scoring_calls: list[int] = []
    original_scoring = UnifiedPipeline._surface_table_scoring

    def _spy_scoring(self, state, page_num, ps, bo):
        scoring_calls.append(page_num)
        real = original_scoring(self, state, page_num, ps, bo)
        return real if force_score_signal is None else force_score_signal

    escalation_calls: list[dict] = []
    original_escalate = UnifiedPipeline._escalate_table_page

    def _spy_escalate(self, state, page_num, ps, bo, profile, run_provider, pdf, **kwargs):
        escalation_calls.append(dict(kwargs))
        return original_escalate(
            self, state, page_num, ps, bo, profile, run_provider, pdf, **kwargs
        )

    reread_calls: list[int] = []
    original_reread = UnifiedPipeline._reread_page_tables

    def _stub_reread(self, state, page_num, raw_crops, extractor):
        reread_calls.append(page_num)
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
            return_value=available_profiles or [PROFILE_QWEN_LOCAL, PROFILE_GEMINI],
        ),
        patch.object(pipeline, "_resolve_judge_model", return_value=""),
        patch.object(pipeline, "_is_agentic_trusted_native", return_value=is_native),
        patch.object(pipeline, "_probe_backend_idle", return_value=True),
        patch("socr.pipeline.orchestrator.probe_ollama_idle", return_value=True),
        patch("socr.pipeline.orchestrator.probe_openai_server_idle", return_value=True),
        patch.object(
            UnifiedPipeline,
            "_run_engine_on_pages",
            autospec=True,
            side_effect=lambda self, *a, **k: _stub_run_engine_on_pages(*a, **k),
        ),
        patch.object(NativeTableVerifierJudge, "assess", _spy_assess),
        patch.object(UnifiedPipeline, "_surface_table_scoring", _spy_scoring),
        patch.object(UnifiedPipeline, "_escalate_table_page", _spy_escalate),
        patch.object(UnifiedPipeline, "_reread_page_tables", _stub_reread),
        patch.object(UnifiedPipeline, "_phase_assemble", _spy_assemble),
        patch.object(UnifiedPipeline, "_resolve_crop_vlm_model", lambda self: "qwen-test-crop"),
        patch("socr.tables.extract.make_table_reader", MagicMock(return_value=MagicMock())),
        patch.object(TableCropExtractor, "extract", MagicMock(return_value=[MagicMock()])),
        patch("socr.tables.locate_tables", MagicMock(side_effect=_real_locate_tables)),
    ):
        result = pipeline.process(pdf_path, out_dir)

    return {
        "result": result,
        "out_dir": out_dir,
        "accepted_text": accepted.get("text"),
        "scoring_pages": scoring_calls,
        "escalation_kwargs": escalation_calls,
        "reread_pages": reread_calls,
        "events": phase_states[0].events if phase_states else [],
    }


def _final_page_one(out_dir: Path) -> str:
    md_candidates = [
        p for p in out_dir.rglob("*.md") if "pages" not in p.parts and p.name != "README.md"
    ]
    assert md_candidates, f"no final assembled markdown found under {out_dir}"
    pages = split_native_pages(md_candidates[0].read_text(encoding="utf-8"))
    assert pages
    return pages[0]


def _fragment_page_one(out_dir: Path) -> str:
    fragments = list(out_dir.rglob("pages/00001.md"))
    assert fragments, f"no pages/00001.md found under {out_dir}"
    return fragments[0].read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Finding 1 — judged bytes are shipped bytes, on the fired-signal path too
# ---------------------------------------------------------------------------


class TestJudgedBytesOnTheSignalPath:
    def test_clean_path_ships_the_judge_accepted_text(self, tmp_path: Path) -> None:
        """Control: no signal, no reread. Establishes the invariant's baseline
        and that the fixture really does reach a judge acceptance."""
        run = _run_pipeline(tmp_path, dual_pass_tables=False)
        assert run["reread_pages"] == []
        assert run["accepted_text"] is not None, "the judge never accepted page 1"
        assert _fragment_page_one(run["out_dir"]) == run["accepted_text"]
        assert _final_page_one(run["out_dir"]) == run["accepted_text"]

    @pytest.mark.parametrize(
        "patch_text", [_PATCH_REFUSED, _PATCH_ACCEPTED], ids=["refused", "accepted"]
    )
    def test_fired_signal_ships_the_judge_accepted_text(
        self, tmp_path: Path, patch_text: str
    ) -> None:
        """The defect: the crop reread patches ``best_output.text`` after
        ``route_page`` accepted, so the shipped bytes were never judged.

        Whatever the judge LAST accepted for the page is what must reach the
        fragment and the final markdown -- whether that is the patched
        candidate (judge accepted it) or the pre-patch text (judge refused it).
        """
        run = _run_pipeline(
            tmp_path,
            dual_pass_tables=True,
            force_score_signal=True,
            crop_patch_text=patch_text,
        )
        assert run["reread_pages"] == [1], "the crop reread must have fired for this pin"
        assert run["accepted_text"] is not None
        assert _fragment_page_one(run["out_dir"]) == run["accepted_text"]
        assert _final_page_one(run["out_dir"]) == run["accepted_text"]

    def test_refused_patch_keeps_the_previously_accepted_bytes_and_says_so(
        self, tmp_path: Path
    ) -> None:
        """A judge refusal of the patched candidate is not silent: the prior
        bytes ship AND an audit event records the refusal."""
        run = _run_pipeline(
            tmp_path,
            dual_pass_tables=True,
            force_score_signal=True,
            crop_patch_text=_PATCH_REFUSED,
        )
        assert _final_page_one(run["out_dir"]) != _PATCH_REFUSED
        refusals = [
            e
            for e in run["events"]
            if getattr(e, "kind", "") == "table_reread_rejudged"
            and not (getattr(e, "data", None) or {}).get("accepted", True)
        ]
        assert refusals, "a refused crop patch must leave an audit event"

    def test_accepted_patch_is_what_ships(self, tmp_path: Path) -> None:
        """The converse: a patched candidate the judge accepts DOES ship, so
        the fix is a re-judge, not a blanket revert of every reread."""
        run = _run_pipeline(
            tmp_path,
            dual_pass_tables=True,
            force_score_signal=True,
            crop_patch_text=_PATCH_ACCEPTED,
        )
        assert _final_page_one(run["out_dir"]) == _PATCH_ACCEPTED


# ---------------------------------------------------------------------------
# Finding 2 — TICKET-C2 scoring coverage
# ---------------------------------------------------------------------------


class TestTableScoringCoverage:
    def test_native_bypass_table_page_is_still_scored(self, tmp_path: Path) -> None:
        """A native-bypass table page never enters the OCR ladder, which is
        exactly why #123 TICKET-C2 exists: scoring is the only place its
        not-scorable / unexplained-lane content loss reaches a surface."""
        run = _run_pipeline(tmp_path, is_native=True)
        assert run["scoring_pages"] == [1], (
            "a native-bypass table page must still be scored against its own "
            "native layer; the P5 gate `not is_native` dropped it"
        )

    def test_routed_table_page_is_still_scored(self, tmp_path: Path) -> None:
        run = _run_pipeline(tmp_path, is_native=False)
        assert run["scoring_pages"] == [1]


# ---------------------------------------------------------------------------
# Finding 3 — GH-96 escalation uses score evidence, not route evidence
# ---------------------------------------------------------------------------


class TestEscalationUsesScoreEvidence:
    def test_route_evidence_alone_does_not_trigger_the_provider_escalation(
        self, tmp_path: Path
    ) -> None:
        """The score says the incumbent matches the native layer exactly, so
        ``decide_escalation`` could not keep any candidate. Route evidence must
        not spend a cloud read anyway."""
        run = _run_pipeline(
            tmp_path,
            escalate_ambiguous_tables=True,
            force_score_signal=False,
        )
        assert run["escalation_kwargs"], "the GH-96 lane never ran; nothing is pinned"
        for kwargs in run["escalation_kwargs"]:
            assert kwargs.get("needs_escalation") is False, (
                "GH-96 must be driven by the table score, not by route evidence"
            )

    def test_score_evidence_does_trigger_the_provider_escalation(self, tmp_path: Path) -> None:
        run = _run_pipeline(
            tmp_path,
            escalate_ambiguous_tables=True,
            force_score_signal=True,
        )
        assert run["escalation_kwargs"]
        assert all(k.get("needs_escalation") is True for k in run["escalation_kwargs"])
