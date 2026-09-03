"""Cold review round 5 on the P3+P5 branch (`fix/p3-p5-judged-bytes-ship`).

Round 4 closed the marker contract. The page-total spend was still wrong, and
the round-4 review found the root cause: it was DERIVED from ``ps.attempts``,
which is both incomplete live and collapsed on resume.

- Live, a rejected GH-96 escalation journals its spend into ``state.engine_runs``
  but never appends the candidate to ``ps.attempts``, so a derivation misses it
  on the very first sidecar write.
- On resume, ``_restore_terminal_page_state`` rebuilds ``ps.attempts`` as the one
  frozen winner. Assembly then re-flushes every terminal sidecar and RE-DERIVED
  the field from that collapsed list, so the first resumed run overwrote the
  multi-rung total it had just consumed and the second resume regained the
  rejected rung's budget.

Ruling: per-page spend is a RECORDED FACT, never a derived one. It is
incremented wherever an ``EngineResult`` is journaled for the page, persisted,
restored verbatim, and re-written as the restored fact.

Hermetic: no provider, no network, no live model.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

fitz = pytest.importorskip("fitz")

from socr.core.config import PipelineConfig  # noqa: E402
from socr.core.document import DocumentHandle  # noqa: E402
from socr.core.providers import PROFILE_GEMINI, PROFILE_MISTRAL  # noqa: E402
from socr.core.result import (  # noqa: E402
    DocumentStatus,
    EngineResult,
    PageOutput,
    PageStatus,
)
from socr.core.state import DocumentState  # noqa: E402
from socr.pipeline.agentic import AcceptDecision  # noqa: E402
from socr.pipeline.orchestrator import UnifiedPipeline  # noqa: E402

from test_p35_cold_review_round2 import (  # noqa: E402
    _CERTAIN_FAIL,
    _PERFECT,
    _build_fixture_pdf,
    _run_pipeline,
)


class _Accept:
    def assess(self, output, provider):
        return AcceptDecision(accept=True, reason="accepted")


def _sidecar(out_dir: Path) -> dict:
    found = list(out_dir.rglob("pages/00001.json"))
    assert found, f"no terminal sidecar under {out_dir}"
    return json.loads(found[0].read_text(encoding="utf-8"))


def _write_sidecar(out_dir: Path, payload: dict) -> None:
    found = list(out_dir.rglob("pages/00001.json"))
    found[0].write_text(json.dumps(payload), encoding="utf-8")


# ---------------------------------------------------------------------------
# 1 — the multi-rung total survives TWO resumes
# ---------------------------------------------------------------------------


def _multi_rung_state(tmp_path: Path):
    """A page the ladder paid two rungs for, journaled the way production does:
    the EngineResult carries every rung, and only the WINNER is an attempt."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    pdf = tmp_path / "doc.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "source")
    doc.save(pdf)
    doc.close()
    state = DocumentState(DocumentHandle(pdf))
    bo = PageOutput(
        page_num=1,
        text="patched",
        status=PageStatus.SUCCESS,
        engine=PROFILE_MISTRAL.engine.value,
        provider_id=PROFILE_MISTRAL.id,
        cost_usd=PROFILE_MISTRAL.cost_per_page_usd,
    )
    ps = state.pages[1]
    ps.attempts.append(bo)
    ps.best_output = bo
    route_cost = PROFILE_GEMINI.cost_per_page_usd + PROFILE_MISTRAL.cost_per_page_usd
    state.record_engine_run(
        EngineResult(
            document_path=state.handle.path,
            engine=PROFILE_MISTRAL.engine.value,
            status=DocumentStatus.SUCCESS,
            cost=route_cost,
        )
    )
    UnifiedPipeline._add_page_cost(ps, route_cost)
    state.agentic_judge_model = PROFILE_GEMINI.model
    return state, ps, bo


class TestMultiRungSpendSurvivesTwoResumes:
    def test_live_first_resume_and_second_resume_all_agree(self, tmp_path: Path) -> None:
        state, ps, bo = _multi_rung_state(tmp_path / "live")
        config = PipelineConfig(
            quiet=True,
            judge_backend="heuristic",
            reprocess=True,
            table_judge_ladder=False,  # docs/log/2026-09-03_p1-prep-latch-and-audit.md
        )
        pipeline = UnifiedPipeline(config)
        pipeline._rejudge_crop_patched_page(state, 1, ps, bo, "old", _Accept(), PROFILE_MISTRAL)
        live_cost = state.total_cost
        assert live_cost == pytest.approx(
            PROFILE_GEMINI.cost_per_page_usd
            + PROFILE_MISTRAL.cost_per_page_usd
            + PROFILE_GEMINI.cost_per_page_usd
        ), "control: two rungs plus a priced judge"

        out_dir = tmp_path / "out"
        with patch.object(pipeline, "_resolve_judge_model", return_value=None):
            pipeline._flush_page_fragment(state, 1, bo.text, out_dir)
            pipeline._flush_page_sidecar(state, 1, out_dir, terminal=True)

        # --- first resume ---------------------------------------------------
        first = DocumentState(DocumentHandle(state.handle.path))
        first_pipe = UnifiedPipeline(config)
        with patch.object(first_pipe, "_resolve_judge_model", return_value=None):
            loaded = first_pipe._load_terminal_page(first, 1, out_dir)
            assert loaded is not None, "control: the terminal page must be resumable"
            first_pipe._restore_terminal_page_state(first, 1, loaded, out_dir)
            assert first.total_cost == live_cost
            # Assembly re-flushes every terminal sidecar on a resumed run. The
            # rewrite must carry the RESTORED FACT, not a recomputation from the
            # attempt list resume just collapsed to one winner.
            first_pipe._flush_page_sidecar(first, 1, out_dir, terminal=True)

        # --- second resume --------------------------------------------------
        second = DocumentState(DocumentHandle(state.handle.path))
        second_pipe = UnifiedPipeline(config)
        with patch.object(second_pipe, "_resolve_judge_model", return_value=None):
            loaded2 = second_pipe._load_terminal_page(second, 1, out_dir)
            assert loaded2 is not None
            second_pipe._restore_terminal_page_state(second, 1, loaded2, out_dir)
        assert second.total_cost == live_cost, (
            "the second resume regained the rejected rung's budget: the first "
            "resumed run rewrote the sidecar from its collapsed attempt list"
        )

    def test_live_total_does_not_double_count_the_folded_judge_cost(self, tmp_path: Path) -> None:
        """Round-3 control, kept: the judge cost folded onto the page must not
        inflate the live total, which sums ``engine_runs`` alone."""
        state, ps, bo = _multi_rung_state(tmp_path / "nodouble")
        route_cost = PROFILE_GEMINI.cost_per_page_usd + PROFILE_MISTRAL.cost_per_page_usd
        pipeline = UnifiedPipeline(
            PipelineConfig(
                quiet=True,
                reprocess=True,
                table_judge_ladder=False,  # docs/log/2026-09-03_p1-prep-latch-and-audit.md
            )
        )
        pipeline._rejudge_crop_patched_page(state, 1, ps, bo, "old", _Accept(), PROFILE_MISTRAL)
        assert state.total_cost == route_cost + PROFILE_GEMINI.cost_per_page_usd


# ---------------------------------------------------------------------------
# 2 — a REJECTED paid escalation persists its spend on the first write
# ---------------------------------------------------------------------------


class TestRejectedEscalationSpendIsRecorded:
    def test_rejected_gh96_escalation_persists_its_cost(self, tmp_path: Path) -> None:
        """The rejected branch returns without appending the candidate, so a
        derivation from ``ps.attempts`` misses spend that really happened."""
        pdf = _build_fixture_pdf(tmp_path / "esc")
        state = DocumentState(DocumentHandle(pdf))
        incumbent = PageOutput(
            page_num=1,
            text=_PERFECT,
            status=PageStatus.SUCCESS,
            engine="qwen",
            provider_id="qwen-local",
            cost_usd=0.0,
        )
        ps = state.pages[1]
        ps.attempts.append(incumbent)
        ps.best_output = incumbent

        pipeline = UnifiedPipeline(
            PipelineConfig(
                quiet=True,
                table_judge_ladder=False,  # docs/log/2026-09-03_p1-prep-latch-and-audit.md
            )
        )

        def _run_provider(profile, page_num):
            # Measures no better than the incumbent, so decide_escalation refuses
            # it -- after the call has already been paid for.
            return PageOutput(
                page_num=page_num,
                text=_PERFECT,
                status=PageStatus.SUCCESS,
                engine=profile.engine.value,
            )

        degraded, out = pipeline._escalate_table_page(
            state,
            1,
            ps,
            incumbent,
            PROFILE_GEMINI,
            _run_provider,
            pdf,
            needs_escalation=True,
        )
        assert degraded is False
        assert out is incumbent, "control: a refused candidate must not replace the incumbent"
        assert state.total_cost == PROFILE_GEMINI.cost_per_page_usd, (
            "control: the refused call was really paid for"
        )

        out_dir = tmp_path / "out"
        with patch.object(pipeline, "_resolve_judge_model", return_value=None):
            pipeline._flush_page_fragment(state, 1, incumbent.text, out_dir)
            pipeline._flush_page_sidecar(state, 1, out_dir, terminal=True)
        assert _sidecar(out_dir)["page_cost_usd"] == PROFILE_GEMINI.cost_per_page_usd, (
            "the FIRST sidecar write must already carry the refused call's spend"
        )


# ---------------------------------------------------------------------------
# 3 — an old sidecar without the field becomes a fact
# ---------------------------------------------------------------------------


class TestOldSidecarUpgradesToAFact:
    def test_missing_field_falls_back_then_is_written_back(self, tmp_path: Path) -> None:
        state, ps, bo = _multi_rung_state(tmp_path / "old")
        config = PipelineConfig(
            quiet=True,
            judge_backend="heuristic",
            reprocess=True,
            table_judge_ladder=False,  # docs/log/2026-09-03_p1-prep-latch-and-audit.md
        )
        pipeline = UnifiedPipeline(config)
        out_dir = tmp_path / "out"
        with patch.object(pipeline, "_resolve_judge_model", return_value=None):
            pipeline._flush_page_fragment(state, 1, bo.text, out_dir)
            pipeline._flush_page_sidecar(state, 1, out_dir, terminal=True)

        payload = _sidecar(out_dir)
        payload.pop("page_cost_usd", None)  # a sidecar written before the field existed
        _write_sidecar(out_dir, payload)

        resumed = DocumentState(DocumentHandle(state.handle.path))
        resumed_pipe = UnifiedPipeline(config)
        with patch.object(resumed_pipe, "_resolve_judge_model", return_value=None):
            loaded = resumed_pipe._load_terminal_page(resumed, 1, out_dir)
            assert loaded is not None
            resumed_pipe._restore_terminal_page_state(resumed, 1, loaded, out_dir)
            # The fallback is the winner's own cost -- what those runs recorded.
            assert resumed.total_cost == bo.cost_usd
            resumed_pipe._flush_page_sidecar(resumed, 1, out_dir, terminal=True)

        assert _sidecar(out_dir)["page_cost_usd"] == bo.cost_usd, (
            "the fallback must be written back as a fact so the next resume is stable"
        )

        third = DocumentState(DocumentHandle(state.handle.path))
        third_pipe = UnifiedPipeline(config)
        with patch.object(third_pipe, "_resolve_judge_model", return_value=None):
            loaded3 = third_pipe._load_terminal_page(third, 1, out_dir)
            assert loaded3 is not None
            third_pipe._restore_terminal_page_state(third, 1, loaded3, out_dir)
        assert third.total_cost == bo.cost_usd


# ---------------------------------------------------------------------------
# 4 — the real recovery path, end to end, through two resumes
# ---------------------------------------------------------------------------


class TestRealRecoveryPathSpendIsStable:
    def test_two_resumes_of_the_real_exhausted_ladder_page(self, tmp_path: Path) -> None:
        recovered = _run_pipeline(
            tmp_path / "real",
            candidate_text=_CERTAIN_FAIL,
            dual_pass_tables=True,
            crop_patch_text=_PERFECT,
        )
        state = recovered["state"]
        live_cost = state.total_cost
        assert recovered["ps"].best_output.audit_passed is True, "control: the crop promoted"

        out_dir = recovered["out_dir"]
        assert _sidecar(out_dir)["page_cost_usd"] == live_cost

        config = recovered["config"]
        totals = []
        for _ in range(2):
            resumed = DocumentState(DocumentHandle(recovered["pdf_path"]))
            pipe = UnifiedPipeline(config)
            with patch.object(pipe, "_resolve_judge_model", return_value=""):
                loaded = pipe._load_terminal_page(resumed, 1, out_dir)
                if loaded is None:
                    pytest.skip("the live run left no resumable terminal page for this fixture")
                pipe._restore_terminal_page_state(resumed, 1, loaded, out_dir)
                totals.append(resumed.total_cost)
                pipe._flush_page_sidecar(resumed, 1, out_dir, terminal=True)
        assert totals == [live_cost, live_cost]
