"""Cold review round 4 on the P3+P5 branch (`fix/p3-p5-judged-bytes-ship`).

Round 3 closed five of the six open items. Two remain:

1. **The fabricated-ref marker was a test oracle, not a contract.** `SOCR_MARKER_RE`
   recognised socr's marker prose, but `url_provenance` still built its replacement
   ad hoc, so the recognizer only changed what the test compared. The marker
   exception stands -- socr's own receipt for a removal is not document content --
   but it has to be a shipping contract: one definition, and the emitter built from
   it, so a marker the recognizer would not match cannot be emitted at all.
2. **An exhausted multi-rung page regained paid budget on resume.** Live routing
   journals every attempted rung, but the terminal sidecar persisted only the
   WINNING output's per-attempt cost. On the branch's own recovery path -- both
   rungs rejected, the paid rung spent, the crop repairs the table, the re-judge
   promotes the free local winner -- the page's real spend was recorded against a
   rejected attempt and vanished on resume.

Hermetic: no provider, no network, no live model.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import patch

import pytest

fitz = pytest.importorskip("fitz")

from socr.core.config import PipelineConfig  # noqa: E402
from socr.core.document import DocumentHandle  # noqa: E402
from socr.core.manifest import SOCR_MARKER_RE, socr_marker  # noqa: E402
from socr.core.providers import PROFILE_GEMINI, PROFILE_MISTRAL  # noqa: E402
from socr.core.result import (  # noqa: E402
    DocumentStatus,
    EngineResult,
    PageOutput,
    PageStatus,
)
from socr.core.state import DocumentState  # noqa: E402
from socr.core.url_provenance import FABRICATED_IMAGE_MARKER  # noqa: E402
from socr.pipeline.agentic import AcceptDecision  # noqa: E402
from socr.pipeline.orchestrator import UnifiedPipeline  # noqa: E402

from test_p35_cold_review_round2 import (  # noqa: E402
    _CERTAIN_FAIL,
    _PERFECT,
    _run_pipeline,
)

_CONTENT_TOKEN_RE = re.compile(r"[A-Za-z]+|\d+(?:[.,]\d+)?")


def _tokens(text: str) -> set[str]:
    return set(_CONTENT_TOKEN_RE.findall(text or ""))


def _content_tokens(text: str) -> set[str]:
    """Tokens outside every span socr itself authored."""
    return _tokens(SOCR_MARKER_RE.sub(" ", text or ""))


# ---------------------------------------------------------------------------
# 1 — the marker is a contract, not a test oracle
# ---------------------------------------------------------------------------


class TestSocrMarkerIsAContract:
    def test_every_marker_the_builder_emits_is_recognised(self) -> None:
        """The property that makes the exception safe: the emitter cannot
        produce a marker the recognizer would miss, so a sanitizer can never
        smuggle free prose past the content-token check."""
        for note in [
            "fabricated image reference removed",
            "a note with ] a closing bracket",
            "a note with\na newline",
            "",
            "   ",
        ]:
            marker = socr_marker(note)
            match = SOCR_MARKER_RE.fullmatch(marker)
            assert match is not None, f"builder emitted an unrecognised marker: {marker!r}"

    def test_the_fabricated_ref_marker_is_built_from_the_shared_definition(self) -> None:
        """A value check cannot tell "built from the contract" from "happens to
        look like it today", and that difference is the whole finding. So this
        pins the SOURCE, the way tests/test_gh169_judge_reason_persists.py pins
        its production site."""
        import pathlib

        src = pathlib.Path(__file__).resolve().parents[1] / "src" / "socr"
        provenance = (src / "core" / "url_provenance.py").read_text()
        assert "socr_marker(" in provenance, (
            "url_provenance must BUILD its replacement through the shared marker "
            "builder, not assemble its own prose"
        )
        assert "[socr:" not in provenance, (
            "url_provenance still hand-writes the marker shape; a second copy is "
            "how the recognizer and the emitter drift apart"
        )
        assert SOCR_MARKER_RE.fullmatch(FABRICATED_IMAGE_MARKER)

    def test_a_marker_is_one_span_so_stripping_cannot_swallow_content(self) -> None:
        text = f"Revenue 42.8. {FABRICATED_IMAGE_MARKER} Costs 13.1."
        assert _content_tokens(text) == _tokens("Revenue 42.8. Costs 13.1.")


class TestFabricatedRefSanitizerIsSubtractiveOnContent:
    """The cold reviewer's round-2 canary, as the CONTENT-token check."""

    def _run(self, tmp_path: Path):
        pipeline = UnifiedPipeline(
            PipelineConfig(
                quiet=True,
                save_figures=False,
                table_judge_ladder=False,  # docs/log/2026-09-03_p1-prep-latch-and-audit.md
            )
        )
        state = type("_State", (), {"events": [], "pages": {}})()
        before = "Revenue 42.8.\n\n![chart](https://example.invalid/invented999.png)"
        out = PageOutput(page_num=1, text=before, status=PageStatus.SUCCESS, engine="qwen")
        with patch.object(pipeline, "_source_url_index", return_value=frozenset()):
            pipeline._sanitize_agentic_page_image_refs(state, 1, out, tmp_path)
        return before, out

    def test_only_added_tokens_lie_inside_a_recognised_marker(self, tmp_path: Path) -> None:
        before, out = self._run(tmp_path)
        assert out.text != before, "the fixture must actually trigger the fabricated-ref gate"
        assert _content_tokens(out.text) <= _tokens(before)

    def test_the_fabricated_reference_is_removed(self, tmp_path: Path) -> None:
        _before, out = self._run(tmp_path)
        assert "invented999" not in out.text
        assert "example.invalid" not in out.text

    def test_no_content_token_changes(self, tmp_path: Path) -> None:
        """Subtractive means removed or kept -- never rewritten. Everything that
        was NOT the invented reference must survive unaltered."""
        before, out = self._run(tmp_path)
        kept = _tokens("Revenue 42.8.")
        assert kept <= _content_tokens(out.text)
        assert _content_tokens(out.text) <= _tokens(before)


# ---------------------------------------------------------------------------
# 2 — a page's TOTAL spend survives resume
# ---------------------------------------------------------------------------


class _Accept:
    def assess(self, output, provider):
        return AcceptDecision(accept=True, reason="accepted")


def _state(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    pdf = tmp_path / "doc.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "source")
    doc.save(pdf)
    doc.close()
    state = DocumentState(DocumentHandle(pdf))
    rejected = PageOutput(
        page_num=1,
        text="",
        status=PageStatus.SUCCESS,
        engine=PROFILE_GEMINI.engine.value,
        provider_id=PROFILE_GEMINI.id,
        audit_passed=False,
        cost_usd=PROFILE_GEMINI.cost_per_page_usd,
    )
    bo = PageOutput(
        page_num=1,
        text="patched",
        status=PageStatus.SUCCESS,
        engine=PROFILE_MISTRAL.engine.value,
        provider_id=PROFILE_MISTRAL.id,
        cost_usd=PROFILE_MISTRAL.cost_per_page_usd,
    )
    ps = state.pages[1]
    ps.attempts.extend([rejected, bo])
    ps.best_output = bo
    return state, ps, bo


class TestPageSpendSurvivesResume:
    def test_live_total_does_not_double_count_the_folded_judge_cost(self, tmp_path: Path) -> None:
        """The property round 3 established, kept as a control: folding the judge
        cost onto the page must not inflate the LIVE total, which sums
        ``engine_runs`` alone."""
        state, ps, bo = _state(tmp_path)
        route_cost = PROFILE_GEMINI.cost_per_page_usd + PROFILE_MISTRAL.cost_per_page_usd
        state.record_engine_run(
            EngineResult(
                document_path=state.handle.path,
                engine=PROFILE_MISTRAL.engine.value,
                status=DocumentStatus.SUCCESS,
                cost=route_cost,
            )
        )
        state.agentic_judge_model = PROFILE_GEMINI.model
        pipeline = UnifiedPipeline(
            PipelineConfig(
                quiet=True,
                reprocess=True,
                table_judge_ladder=False,  # docs/log/2026-09-03_p1-prep-latch-and-audit.md
            )
        )
        pipeline._rejudge_crop_patched_page(state, 1, ps, bo, "old", _Accept(), PROFILE_MISTRAL)
        judge_cost = PROFILE_GEMINI.cost_per_page_usd
        assert state.total_cost == route_cost + judge_cost

    # The multi-rung resume case that stood here built its ``PageState`` by hand
    # and leaned on round 4's DERIVATION of the page total from ``ps.attempts``.
    # Round 5 showed that derivation is the defect -- it misses a refused
    # escalation live and is destroyed by the first resumed run's sidecar
    # rewrite -- so the scenario moved to
    # ``tests/test_p35_cold_review_round5.py::TestMultiRungSpendSurvivesTwoResumes``,
    # which records the spend the way production does and carries it through TWO
    # resumes. The coverage moved and got stronger; it was not dropped.

    def test_real_exhausted_ladder_promotion_persists_all_spend(self, tmp_path: Path) -> None:
        """The branch's own recovery path, end to end: both rungs rejected, the
        crop repairs the table, the re-judge promotes the free local winner.
        The page's persisted spend must be the whole page's spend, not the
        winning rung's."""
        recovered = _run_pipeline(
            tmp_path / "real",
            candidate_text=_CERTAIN_FAIL,
            dual_pass_tables=True,
            crop_patch_text=_PERFECT,
        )
        state = recovered["state"]
        ps = recovered["ps"]
        assert ps.best_output.audit_passed is True, "control: the crop recovery must have promoted"
        live_cost = state.total_cost
        assert live_cost, "control: this path must have spent something to be worth pinning"
        assert ps.best_output.cost_usd != live_cost, (
            "control: the winning rung's own cost is NOT the page's spend here -- "
            "that gap is the finding"
        )

        # What the live run left on disk is what a resumed run can restore, so
        # the sidecar has to carry the page's whole spend. The reconstruction
        # side is pinned by the resume test above; this pins the persistence.
        sidecars = list(recovered["out_dir"].rglob("pages/00001.json"))
        assert sidecars, "control: the live run must have written a terminal sidecar"
        persisted = json.loads(sidecars[0].read_text(encoding="utf-8"))
        assert persisted.get("page_cost_usd") == live_cost, (
            "the terminal sidecar must persist the page's total spend, not the "
            "winning rung's own cost"
        )
