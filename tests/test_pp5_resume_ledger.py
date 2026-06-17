"""PP-5 tests: per-page resume ledger — skip terminal pages on re-run.

Acceptance criteria (GH-76):
  * Crash/interrupt after page K, re-run → ONLY pages > K are re-OCR'd.
    Proved with a route_page spy: pages <= K (terminal) must NOT be routed
    again; pages > K (non-terminal) MUST be.
  * A run-config / model change (different _run_fingerprint) invalidates the
    terminal fragments → the affected pages ARE re-OCR'd.
  * A NON-terminal page (provisional terminal=false, or missing sidecar) is NOT
    skipped — it is reprocessed.
  * A corrupt / unparseable sidecar → reprocess (no crash, no skip).
  * No second writer of root metadata.json (RootIndex sole author).
  * A partial doc (some terminal pages + some unprocessed) resumes correctly.

Hermetic: every test that drives _phase_agentic / process() patches
_available_engines_for_agentic so the provider ladder is non-empty regardless
of whether ollama/gemini is installed (CI has no provider).  Mirrors PP-2/PP-7.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from socr.core.born_digital import DocumentAssessment, PageAssessment
from socr.core.config import EngineType, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.providers import PROFILE_GEMINI
from socr.core.result import DocumentStatus, PageOutput, PageStatus
from socr.core.state import DocumentState
from socr.pipeline.orchestrator import UnifiedPipeline

# ---------------------------------------------------------------------------
# Shared helpers (mirror the PP-2 harness)
# ---------------------------------------------------------------------------

# Distinct, deterministic per-page OCR bodies (each 50+ words to pass audit if on).
_PAGE_TEXTS = {
    1: (
        "The interest rate channel of monetary transmission operates by altering the "
        "marginal cost of borrowing for households and firms across the economy. When "
        "the central bank tightens policy short term rates rise and the effect propagates "
        "along the yield curve to longer maturities that price investment decisions and "
        "durable consumption over the medium run in most advanced economies studied here."
    ),
    2: (
        "The bank lending channel emphasises the role of financial intermediaries in "
        "amplifying monetary impulses through their balance sheets and funding costs. "
        "Tighter policy reduces the supply of loanable funds at any given rate forcing "
        "credit constrained borrowers to curtail spending more sharply than the frictionless "
        "benchmark would predict in standard new keynesian models of the business cycle."
    ),
    3: (
        "The exchange rate channel links domestic monetary policy to external competitiveness "
        "through capital flows and the uncovered interest parity condition that anchors "
        "expectations. A surprise tightening appreciates the currency dampening net exports "
        "and importing disinflation while simultaneously tightening financial conditions for "
        "firms with foreign currency liabilities on their balance sheets in open economies."
    ),
    4: (
        "Fiscal monetary interactions introduce additional heterogeneity into the empirical "
        "estimates of transmission strength across regimes and institutional settings worldwide. "
        "Periods of consolidation amplify contractionary effects while expansionary stances can "
        "partially offset the pass through of policy impulses to aggregate demand requiring "
        "careful identification grounded in theory to separate the two structural shocks cleanly."
    ),
}


def _make_config(*, enabled_engines=None, **overrides) -> PipelineConfig:
    overrides.setdefault("save_figures", False)
    overrides.setdefault("primary_engine", EngineType.DEEPSEEK)
    return PipelineConfig(
        agentic=True,
        judge_backend="heuristic",
        enabled_engines=enabled_engines or [EngineType.GEMINI],
        dual_pass_tables=False,
        detect_equations=False,
        recover_clean_equations=False,
        quiet=True,
        audit_enabled=False,
        write_manifest=False,
        **overrides,
    )


def _make_pipeline(config: PipelineConfig | None = None) -> UnifiedPipeline:
    return UnifiedPipeline(config or _make_config())


def _bd_assessment(
    page_count: int, born_digital_pages: set[int] | None = None
) -> DocumentAssessment:
    bd = born_digital_pages or set()
    pages = [
        PageAssessment(
            page_num=i,
            is_born_digital=i in bd,
            native_text=f"native text for page {i} " * 5 if i in bd else "",
            confidence=0.9,
        )
        for i in range(1, page_count + 1)
    ]
    return DocumentAssessment(path=Path("/tmp/fake.pdf"), pages=pages)


def _real_pdf(tmp_path: Path, page_count: int = 3) -> Path:
    fitz = pytest.importorskip("fitz")
    path = tmp_path / "doc.pdf"
    doc = fitz.open()
    for i in range(page_count):
        doc.new_page().insert_text((72, 72), f"page {i + 1} text " * 10)
    doc.save(str(path))
    doc.close()
    return path


def _counting_route(route_calls: list[int]):
    """route_page side_effect that records each page routed and returns its text."""

    def _route(page_num, ladder, run_provider, judge, **kwargs):
        from socr.pipeline.agentic import PageDecision, ProviderAttempt

        route_calls.append(page_num)
        out = PageOutput(
            page_num=page_num,
            text=_PAGE_TEXTS[page_num],
            status=PageStatus.SUCCESS,
            engine="gemini",
            audit_passed=True,
        )
        prof = ladder[0]
        att = ProviderAttempt(
            engine=prof.engine,
            output=out,
            cost_usd=prof.cost_per_page_usd,
            accepted=True,
            reason="ok",
            provider_id=prof.id,
            model=prof.model,
            backend=prof.backend,
        )
        return PageDecision(page_num=page_num, final_output=out, attempts=[att], accepted=True)

    return _route


def _run(pipeline: UnifiedPipeline, pdf_path: Path, out_dir: Path, route_calls: list[int]):
    """Drive a full process() run with a hermetic ladder + counting route_page."""
    with (
        patch.object(pipeline, "_available_engines_for_agentic", return_value=[PROFILE_GEMINI]),
        patch("socr.pipeline.orchestrator.route_page", side_effect=_counting_route(route_calls)),
    ):
        return pipeline.process(pdf_path, out_dir)


def _doc_dir(out_dir: Path) -> Path:
    return out_dir / "doc"


def _simulate_crash_after(out_dir: Path, terminal_pages: set[int], all_pages: set[int]) -> None:
    """Reproduce on disk what a hard crash mid-``_phase_agentic`` leaves behind.

    A real crash before ``_phase_assemble`` completes means:
      * The pages that DID finish their per-page lifecycle have terminal sidecars
        + fragments (we keep ``terminal_pages`` intact).
      * The pages that did NOT finish have no terminal artefacts (drop them).
      * ``_phase_assemble`` never ran, so NEITHER the root ``metadata.json`` nor
        the per-doc ``metadata.json`` carries a COMPLETED record — the outer
        RootIndex gate therefore does NOT skip the doc, and ``process()`` re-
        enters ``_phase_agentic`` where the inner ledger takes over.  We delete
        both metadata files to model that (the RootIndex gate stays unchanged).
    """
    pages_dir = _doc_dir(out_dir) / "pages"
    for n in all_pages - terminal_pages:
        for suffix in (".json", ".md"):
            f = pages_dir / f"{n:05d}{suffix}"
            if f.exists():
                f.unlink()
    for meta in (out_dir / "metadata.json", _doc_dir(out_dir) / "metadata.json"):
        if meta.exists():
            meta.unlink()


# ---------------------------------------------------------------------------
# AC: crash after page K → only pages > K reprocessed
# ---------------------------------------------------------------------------


class TestResumeSkipsTerminalPages:
    def test_only_pages_after_k_reprocessed(self, tmp_path: Path) -> None:
        """Full run, then simulate a crash after page 2 (drop p3 terminal sidecar);
        re-run must re-OCR ONLY p3, never p1/p2."""
        pdf_path = _real_pdf(tmp_path, page_count=3)
        config = _make_config()
        pipeline = _make_pipeline(config)
        pipeline.bd_detector = MagicMock()
        pipeline.bd_detector.detect.return_value = _bd_assessment(3)  # all OCR

        # --- Run 1: complete the whole doc. All 3 pages routed. ---
        run1_calls: list[int] = []
        result1 = _run(pipeline, pdf_path, tmp_path, run1_calls)
        assert result1.status == DocumentStatus.SUCCESS
        assert sorted(run1_calls) == [1, 2, 3], f"Run 1 must OCR all pages; got {run1_calls}"

        pages_dir = _doc_dir(tmp_path) / "pages"
        for n in (1, 2, 3):
            meta = json.loads((pages_dir / f"{n:05d}.json").read_text())
            assert meta["terminal"] is True, f"p{n} sidecar must be terminal after run 1"

        # --- Simulate "crashed after page 2": p3 never reached terminal. ---
        # A real crash leaves pages 1-2 terminal, p3 unfinished, and no COMPLETED
        # doc-level record — so the outer RootIndex gate does not skip the doc.
        _simulate_crash_after(tmp_path, terminal_pages={1, 2}, all_pages={1, 2, 3})

        # --- Run 2: re-run. Only p3 must be re-OCR'd. ---
        run2_pipeline = _make_pipeline(_make_config())
        run2_pipeline.bd_detector = MagicMock()
        run2_pipeline.bd_detector.detect.return_value = _bd_assessment(3)
        run2_calls: list[int] = []
        result2 = _run(run2_pipeline, pdf_path, tmp_path, run2_calls)

        assert result2.status == DocumentStatus.SUCCESS
        assert run2_calls == [3], (
            f"Run 2 must re-OCR ONLY p3 (pages 1-2 were terminal); got {run2_calls}"
        )
        # The resumed doc must still contain all three pages' text.
        for n in (1, 2, 3):
            assert _PAGE_TEXTS[n] in result2.markdown, f"p{n} text missing from resumed markdown"

    def test_full_rerun_skips_all_pages(self, tmp_path: Path) -> None:
        """A clean re-run of a fully terminal doc (identical fingerprint) re-OCRs
        NOTHING — every page is a terminal ledger hit."""
        pdf_path = _real_pdf(tmp_path, page_count=3)
        pipeline = _make_pipeline(_make_config())
        pipeline.bd_detector = MagicMock()
        pipeline.bd_detector.detect.return_value = _bd_assessment(3)

        run1_calls: list[int] = []
        _run(pipeline, pdf_path, tmp_path, run1_calls)
        assert sorted(run1_calls) == [1, 2, 3]

        # Re-run with --reprocess so the OUTER doc-level RootIndex gate does not
        # short-circuit process() before _phase_agentic; the INNER ledger must
        # still skip every page on fingerprint match.
        config2 = _make_config(reprocess=True)
        pipeline2 = _make_pipeline(config2)
        pipeline2.bd_detector = MagicMock()
        pipeline2.bd_detector.detect.return_value = _bd_assessment(3)
        run2_calls: list[int] = []
        _run(pipeline2, pdf_path, tmp_path, run2_calls)

        assert run2_calls == [], (
            f"Identical-fingerprint re-run must re-OCR no pages (inner ledger hits); "
            f"got {run2_calls}"
        )

    def test_mixed_native_and_ocr_resume(self, tmp_path: Path) -> None:
        """A doc mixing born-digital native (p2) + OCR (p1,p3): after a crash that
        left p1+p2 terminal and p3 unfinished, resume re-OCRs ONLY p3.  The native
        page resumes from its terminal ledger entry (no native re-bypass needed)."""
        pdf_path = _real_pdf(tmp_path, page_count=3)
        pipeline = _make_pipeline(_make_config())
        pipeline.bd_detector = MagicMock()
        # p2 is born-digital native; p1 and p3 are OCR.
        pipeline.bd_detector.detect.return_value = _bd_assessment(3, born_digital_pages={2})

        run1_calls: list[int] = []
        result1 = _run(pipeline, pdf_path, tmp_path, run1_calls)
        assert result1.status == DocumentStatus.SUCCESS
        # Only the OCR pages (1, 3) were routed in run 1 — p2 took the native bypass.
        assert sorted(run1_calls) == [1, 3], f"Run 1 must OCR only p1,p3; got {run1_calls}"

        _simulate_crash_after(tmp_path, terminal_pages={1, 2}, all_pages={1, 2, 3})

        pipeline2 = _make_pipeline(_make_config())
        pipeline2.bd_detector = MagicMock()
        pipeline2.bd_detector.detect.return_value = _bd_assessment(3, born_digital_pages={2})
        run2_calls: list[int] = []
        result2 = _run(pipeline2, pdf_path, tmp_path, run2_calls)

        assert result2.status == DocumentStatus.SUCCESS
        assert run2_calls == [3], (
            f"Resume must re-OCR ONLY p3; p1 (OCR-terminal) and p2 (native-terminal) "
            f"must both be ledger hits. got {run2_calls}"
        )


# ---------------------------------------------------------------------------
# AC: fingerprint change invalidates the ledger → affected pages re-OCR'd
# ---------------------------------------------------------------------------


class TestFingerprintInvalidation:
    def test_config_change_forces_reocr(self, tmp_path: Path) -> None:
        """A run-config change (different _run_fingerprint) must re-OCR all pages,
        not silently reuse the terminal fragments."""
        pdf_path = _real_pdf(tmp_path, page_count=3)
        pipeline = _make_pipeline(_make_config())
        pipeline.bd_detector = MagicMock()
        pipeline.bd_detector.detect.return_value = _bd_assessment(3)

        run1_calls: list[int] = []
        _run(pipeline, pdf_path, tmp_path, run1_calls)
        assert sorted(run1_calls) == [1, 2, 3]

        fp1 = pipeline._run_fingerprint()

        # Change an output-affecting flag → different fingerprint.  render_dpi is
        # fingerprinted (it changes the bytes fed to the engine).  reprocess=True
        # so the outer RootIndex gate (checksum+fp) does not block re-entry.
        config2 = _make_config(render_dpi=pipeline.config.render_dpi + 100, reprocess=True)
        pipeline2 = _make_pipeline(config2)
        pipeline2.bd_detector = MagicMock()
        pipeline2.bd_detector.detect.return_value = _bd_assessment(3)
        assert pipeline2._run_fingerprint() != fp1, "Test setup: fingerprint must change"

        run2_calls: list[int] = []
        _run(pipeline2, pdf_path, tmp_path, run2_calls)

        assert sorted(run2_calls) == [1, 2, 3], (
            f"A fingerprint change must re-OCR ALL pages; got {run2_calls}"
        )

    def test_enabled_only_engine_model_change_invalidates(self, tmp_path: Path) -> None:
        """A provider present ONLY via ``enabled_engines`` (not primary / local /
        fallback / multi) drives the agentic ladder.  Changing THAT provider's
        model must change ``_run_fingerprint`` AND force re-OCR on resume — else a
        stale terminal SUCCESS sidecar is silently reused after a model swap.

        QWEN is wired into ``enabled_engines`` only; ``primary``/``local`` are
        DEEPSEEK and ``fallback``/``multi`` are empty, so QWEN reaches the
        fingerprint exclusively through the enabled-engine determinants path.
        """

        def _qwen_only_config(**overrides) -> PipelineConfig:
            return _make_config(
                primary_engine=EngineType.DEEPSEEK,
                local_engine=EngineType.DEEPSEEK,
                fallback_chain=[],
                multi_engine=[],
                enabled_engines=[EngineType.QWEN],
                **overrides,
            )

        # --- Fingerprint unit check: qwen_model swap must change the fingerprint. ---
        # Both pinned, differing ONLY in the model id, so the assertion isolates
        # the enabled-engine model determinant (not the pin flag or anything else).
        base_pinned = UnifiedPipeline(
            _qwen_only_config(qwen_model="qwen3-vl:30b-a3b-instruct", qwen_model_pinned=True)
        )
        swapped = UnifiedPipeline(
            _qwen_only_config(qwen_model="qwen3-vl:235b-a22b-instruct", qwen_model_pinned=True)
        )
        assert base_pinned._run_fingerprint() != swapped._run_fingerprint(), (
            "Changing the model of an enabled-only agentic engine must change "
            "_run_fingerprint (per-engine determinants, not just names)"
        )

        # --- End-to-end: a model swap re-OCRs the page on resume (no stale skip). ---
        pdf_path = _real_pdf(tmp_path, page_count=2)
        pipeline = _make_pipeline(
            _qwen_only_config(qwen_model="qwen3-vl:30b-a3b-instruct", qwen_model_pinned=True)
        )
        pipeline.bd_detector = MagicMock()
        pipeline.bd_detector.detect.return_value = _bd_assessment(2)
        run1_calls: list[int] = []
        _run(pipeline, pdf_path, tmp_path, run1_calls)
        assert sorted(run1_calls) == [1, 2]

        # Swap the qwen model and re-run (reprocess=True bypasses the OUTER gate so
        # the INNER ledger is the only thing that could wrongly skip).
        pipeline2 = _make_pipeline(
            _qwen_only_config(
                qwen_model="qwen3-vl:235b-a22b-instruct",
                qwen_model_pinned=True,
                reprocess=True,
            )
        )
        pipeline2.bd_detector = MagicMock()
        pipeline2.bd_detector.detect.return_value = _bd_assessment(2)
        run2_calls: list[int] = []
        _run(pipeline2, pdf_path, tmp_path, run2_calls)

        assert sorted(run2_calls) == [1, 2], (
            f"A model swap on an enabled-only engine must re-OCR ALL pages, not "
            f"reuse stale terminal sidecars; got {run2_calls}"
        )

    def test_enabled_engines_order_does_not_change_fingerprint(self, tmp_path: Path) -> None:
        """Reordering ``enabled_engines`` (SAME set) must NOT change the
        fingerprint: ``provider_ladder`` is COST-ordered and ignores the input
        list order, so the selected output is identical either way.  A different
        order must not spuriously invalidate the resume ledger (wasteful re-OCR).
        Conversely, adding/removing an engine (a real set change) MUST change it.
        """
        fp_ab = UnifiedPipeline(
            _make_config(
                primary_engine=EngineType.DEEPSEEK,
                local_engine=EngineType.DEEPSEEK,
                fallback_chain=[],
                multi_engine=[],
                enabled_engines=[EngineType.QWEN, EngineType.GEMINI],
            )
        )._run_fingerprint()
        fp_ba = UnifiedPipeline(
            _make_config(
                primary_engine=EngineType.DEEPSEEK,
                local_engine=EngineType.DEEPSEEK,
                fallback_chain=[],
                multi_engine=[],
                enabled_engines=[EngineType.GEMINI, EngineType.QWEN],
            )
        )._run_fingerprint()
        assert fp_ab == fp_ba, (
            "Reordering enabled_engines (same set) must NOT change _run_fingerprint "
            "(the cost-ordered ladder ignores input order)"
        )

        # A genuine SET change (drop QWEN) must still change the fingerprint.
        fp_subset = UnifiedPipeline(
            _make_config(
                primary_engine=EngineType.DEEPSEEK,
                local_engine=EngineType.DEEPSEEK,
                fallback_chain=[],
                multi_engine=[],
                enabled_engines=[EngineType.GEMINI],
            )
        )._run_fingerprint()
        assert fp_subset != fp_ab, (
            "Dropping an engine from enabled_engines MUST change _run_fingerprint"
        )


# ---------------------------------------------------------------------------
# AC: non-terminal / missing / corrupt sidecars are NOT skipped
# ---------------------------------------------------------------------------


class TestNonTerminalNotSkipped:
    def test_provisional_terminal_false_reprocessed(self, tmp_path: Path) -> None:
        """A page whose sidecar is terminal=false (provisional) must be reprocessed."""
        pdf_path = _real_pdf(tmp_path, page_count=2)
        pipeline = _make_pipeline(_make_config())
        pipeline.bd_detector = MagicMock()
        pipeline.bd_detector.detect.return_value = _bd_assessment(2)

        run1_calls: list[int] = []
        _run(pipeline, pdf_path, tmp_path, run1_calls)
        assert sorted(run1_calls) == [1, 2]

        pages_dir = _doc_dir(tmp_path) / "pages"
        # Demote p2 to provisional (terminal=false) — same fingerprint, but the
        # page must NOT be skipped: terminal=false means "may be unfinished".
        meta = json.loads((pages_dir / "00002.json").read_text())
        meta["terminal"] = False
        (pages_dir / "00002.json").write_text(json.dumps(meta))

        pipeline2 = _make_pipeline(_make_config(reprocess=True))
        pipeline2.bd_detector = MagicMock()
        pipeline2.bd_detector.detect.return_value = _bd_assessment(2)
        run2_calls: list[int] = []
        _run(pipeline2, pdf_path, tmp_path, run2_calls)

        assert run2_calls == [2], (
            f"A terminal=false page must be reprocessed; p1 (terminal) must be skipped. "
            f"got {run2_calls}"
        )

    def test_missing_sidecar_reprocessed(self, tmp_path: Path) -> None:
        """A page whose sidecar is absent (but fragment present) must be reprocessed."""
        pdf_path = _real_pdf(tmp_path, page_count=2)
        pipeline = _make_pipeline(_make_config())
        pipeline.bd_detector = MagicMock()
        pipeline.bd_detector.detect.return_value = _bd_assessment(2)

        run1_calls: list[int] = []
        _run(pipeline, pdf_path, tmp_path, run1_calls)
        assert sorted(run1_calls) == [1, 2]

        pages_dir = _doc_dir(tmp_path) / "pages"
        (pages_dir / "00002.json").unlink()  # fragment 00002.md stays

        pipeline2 = _make_pipeline(_make_config(reprocess=True))
        pipeline2.bd_detector = MagicMock()
        pipeline2.bd_detector.detect.return_value = _bd_assessment(2)
        run2_calls: list[int] = []
        _run(pipeline2, pdf_path, tmp_path, run2_calls)

        assert run2_calls == [2], f"A page with no sidecar must be reprocessed; got {run2_calls}"

    def test_corrupt_sidecar_reprocessed_no_crash(self, tmp_path: Path) -> None:
        """A corrupt / unparseable sidecar must reprocess the page, not crash or skip."""
        pdf_path = _real_pdf(tmp_path, page_count=2)
        pipeline = _make_pipeline(_make_config())
        pipeline.bd_detector = MagicMock()
        pipeline.bd_detector.detect.return_value = _bd_assessment(2)

        run1_calls: list[int] = []
        _run(pipeline, pdf_path, tmp_path, run1_calls)
        assert sorted(run1_calls) == [1, 2]

        pages_dir = _doc_dir(tmp_path) / "pages"
        # Write garbage that is not valid JSON.
        (pages_dir / "00002.json").write_text("{ this is not valid json ::: ")

        pipeline2 = _make_pipeline(_make_config(reprocess=True))
        pipeline2.bd_detector = MagicMock()
        pipeline2.bd_detector.detect.return_value = _bd_assessment(2)
        run2_calls: list[int] = []
        result2 = _run(pipeline2, pdf_path, tmp_path, run2_calls)

        assert result2.status == DocumentStatus.SUCCESS, "A corrupt sidecar must not crash the run"
        assert run2_calls == [2], (
            f"A corrupt sidecar must reprocess that page (p1 still skipped); got {run2_calls}"
        )

    def test_missing_fragment_reprocessed(self, tmp_path: Path) -> None:
        """A terminal sidecar with NO body fragment must be reprocessed (incomplete)."""
        pdf_path = _real_pdf(tmp_path, page_count=2)
        pipeline = _make_pipeline(_make_config())
        pipeline.bd_detector = MagicMock()
        pipeline.bd_detector.detect.return_value = _bd_assessment(2)

        run1_calls: list[int] = []
        _run(pipeline, pdf_path, tmp_path, run1_calls)
        assert sorted(run1_calls) == [1, 2]

        pages_dir = _doc_dir(tmp_path) / "pages"
        (pages_dir / "00002.md").unlink()  # sidecar stays, fragment gone

        pipeline2 = _make_pipeline(_make_config(reprocess=True))
        pipeline2.bd_detector = MagicMock()
        pipeline2.bd_detector.detect.return_value = _bd_assessment(2)
        run2_calls: list[int] = []
        _run(pipeline2, pdf_path, tmp_path, run2_calls)

        assert run2_calls == [2], (
            f"A terminal sidecar without its fragment must reprocess; got {run2_calls}"
        )

    def test_non_success_status_reprocessed(self, tmp_path: Path) -> None:
        """A terminal sidecar whose winning_output status is NOT success (e.g. an
        ERROR / timed-out page written terminal at assemble) must be RE-OCR'd,
        never skipped — this is the cascade-halt silent-data-loss vector."""
        pdf_path = _real_pdf(tmp_path, page_count=2)
        pipeline = _make_pipeline(_make_config())
        pipeline.bd_detector = MagicMock()
        pipeline.bd_detector.detect.return_value = _bd_assessment(2)

        run1_calls: list[int] = []
        _run(pipeline, pdf_path, tmp_path, run1_calls)
        assert sorted(run1_calls) == [1, 2]

        pages_dir = _doc_dir(tmp_path) / "pages"
        # Demote p2's winning output to ERROR status (terminal stays True, same fp,
        # same checksum) — it must still be reprocessed because it was not clean.
        meta = json.loads((pages_dir / "00002.json").read_text())
        meta["winning_output"]["status"] = PageStatus.ERROR.value
        (pages_dir / "00002.json").write_text(json.dumps(meta))

        pipeline2 = _make_pipeline(_make_config(reprocess=True))
        pipeline2.bd_detector = MagicMock()
        pipeline2.bd_detector.detect.return_value = _bd_assessment(2)
        run2_calls: list[int] = []
        _run(pipeline2, pdf_path, tmp_path, run2_calls)

        assert run2_calls == [2], (
            f"A non-SUCCESS terminal page must be re-OCR'd; p1 (clean) skipped. got {run2_calls}"
        )

    def test_flagged_native_fallback_sidecar_records_winner_not_attempt(
        self, tmp_path: Path
    ) -> None:
        """GH-56: a page that ships a FLAGGED native-text fallback (its OCR table
        attempt was rejected by the table verifier) must record the NATIVE
        fallback as ``winning_output`` — not the rejected OCR attempt.

        Observed on the real Consensus Economics forecaster grid: qwen produced a
        row-major table that the native-table verifier hard-failed; the page then
        shipped the collapsed native-text fallback (engine="native", WARNING). But
        the sidecar recorded ``best_output`` (the qwen attempt, status="success",
        audit_passed=False), so:
          1. the sidecar disagreed with the fragment that actually shipped, and
          2. the resume gate — which reads ``winning_output.status`` — saw
             "success" and would SKIP this flagged page on re-run, silently
             erasing its audit-failed signal (the gate's own contract forbids
             skipping a flagged native fallback).
        The sidecar must instead record the selection ``_winning_page_output``
        freezes into the manifest/fragment.
        """
        pytest.importorskip("fitz")
        pdf_path = _real_pdf(tmp_path, page_count=1)
        out_dir = tmp_path / "out"

        pipeline = _make_pipeline(_make_config())
        pipeline._scan_root = pdf_path.parent

        state = DocumentState(handle=DocumentHandle.from_path(pdf_path))
        ps = state.pages[1]
        ps.is_born_digital = True
        ps.native_text = "JANUARY 2024\n\n| names-collapsed | numbers-newline-stacked |"
        ps.native_table_structure_failed = True
        # A rejected OCR table attempt: extraction SUCCEEDED as an operation, but
        # the native-table verifier hard-failed its grid (audit_passed=False).
        # status stays SUCCESS — the exact shape that fooled the resume gate.
        rejected = PageOutput(
            page_num=1,
            text="# UNITED STATES\n\n| Goldman Sachs | 2.3 | 1.9 |",
            status=PageStatus.SUCCESS,
            engine="qwen",
            audit_passed=False,
        )
        ps.attempts.append(rejected)
        ps.best_output = rejected

        # Flush what actually ships (the native fallback body) + the terminal sidecar.
        pipeline._flush_page_fragment(state, 1, ps.native_text, out_dir)
        pipeline._flush_page_sidecar(state, 1, out_dir, terminal=True)

        sidecar = next(out_dir.rglob("pages/00001.json"))
        meta = json.loads(sidecar.read_text())

        # Provenance honesty: the sidecar names the NATIVE fallback that shipped,
        # not the rejected qwen attempt.
        assert meta["winning_output"]["engine"] == "native", meta["winning_output"]
        assert meta["winning_output"]["status"] == PageStatus.WARNING.value
        assert meta["winning_output"]["audit_passed"] is False
        assert meta["status"] == PageStatus.WARNING.value
        assert meta["engine"] == "native"

        # Resume correctness: same fingerprint + checksum + terminal=True, so the
        # ONLY thing that can force a reprocess is the now-honest non-SUCCESS
        # status. The flagged page must NOT be skipped.
        assert pipeline._load_terminal_page(state, 1, out_dir) is None

    def test_whole_doc_recovery_sidecar_matches_manifest(self, tmp_path: Path) -> None:
        """GH-56 (non-agentic path): a page whose text was recovered from a CLI
        whole-document attempt must record that recovered output in the sidecar —
        the SAME selection ``build_manifest`` and the fragment use — not a failure
        marker.

        ``_flush_page_sidecar`` runs in ``_phase_assemble`` for every pipeline
        path, including non-agentic CLI-engine runs where ``best_output`` is never
        populated and the text lives in ``state.whole_doc_attempts``. The sidecar
        must compute ``whole_doc`` exactly as the manifest does; passing
        ``whole_doc=None`` here would freeze a failure marker that disagrees with
        the shipped fragment.
        """
        pytest.importorskip("fitz")
        pdf_path = _real_pdf(tmp_path, page_count=1)
        out_dir = tmp_path / "out"

        pipeline = _make_pipeline(_make_config())
        pipeline._scan_root = pdf_path.parent

        state = DocumentState(handle=DocumentHandle.from_path(pdf_path))
        # A CLI whole-doc engine populates whole_doc_attempts (page_num=0), never
        # the per-page best_output slots.
        state.whole_doc_attempts.append(
            PageOutput(
                page_num=0,
                text="## Page 1\n\nrecovered whole-doc body for page one",
                status=PageStatus.SUCCESS,
                engine="marker",
                audit_passed=True,
            )
        )
        assert state.pages[1].best_output is None

        pipeline._flush_page_fragment(state, 1, "recovered whole-doc body for page one", out_dir)
        pipeline._flush_page_sidecar(state, 1, out_dir, terminal=True)

        meta = json.loads(next(out_dir.rglob("pages/00001.json")).read_text())
        assert meta["winning_output"]["engine"] == "marker", meta["winning_output"]
        assert "recovered whole-doc body" in meta["winning_output"]["text"]
        assert meta["status"] == PageStatus.SUCCESS.value

    def test_failure_marker_body_reprocessed(self, tmp_path: Path) -> None:
        """A terminal page whose fragment body is a page-failure marker must be
        re-OCR'd (the marker is honesty, not content — never reuse it)."""
        pdf_path = _real_pdf(tmp_path, page_count=2)
        pipeline = _make_pipeline(_make_config())
        pipeline.bd_detector = MagicMock()
        pipeline.bd_detector.detect.return_value = _bd_assessment(2)

        run1_calls: list[int] = []
        _run(pipeline, pdf_path, tmp_path, run1_calls)
        assert sorted(run1_calls) == [1, 2]

        pages_dir = _doc_dir(tmp_path) / "pages"
        (pages_dir / "00002.md").write_text("[page 2 failed: no usable OCR output]")

        pipeline2 = _make_pipeline(_make_config(reprocess=True))
        pipeline2.bd_detector = MagicMock()
        pipeline2.bd_detector.detect.return_value = _bd_assessment(2)
        run2_calls: list[int] = []
        _run(pipeline2, pdf_path, tmp_path, run2_calls)

        assert run2_calls == [2], f"A failure-marker fragment must be reprocessed; got {run2_calls}"

    def test_changed_input_at_same_path_reprocessed(self, tmp_path: Path) -> None:
        """A DIFFERENT PDF at the same relative path (so the same doc dir / page
        sidecars) must re-OCR every page — the per-page input_checksum guard must
        reject the stale fragments even when the outer RootIndex gate is bypassed
        (partial resume / --reprocess)."""
        fitz = pytest.importorskip("fitz")
        pdf_path = tmp_path / "doc.pdf"

        # Run 1 on the original 2-page PDF.
        doc = fitz.open()
        for i in range(2):
            doc.new_page().insert_text((72, 72), f"original page {i + 1} text " * 10)
        doc.save(str(pdf_path))
        doc.close()

        pipeline = _make_pipeline(_make_config())
        pipeline.bd_detector = MagicMock()
        pipeline.bd_detector.detect.return_value = _bd_assessment(2)
        run1_calls: list[int] = []
        _run(pipeline, pdf_path, tmp_path, run1_calls)
        assert sorted(run1_calls) == [1, 2]

        # REPLACE the PDF with DIFFERENT content at the SAME path (same doc dir,
        # same page sidecars on disk).  Its checksum now differs from the recorded
        # input_checksum, so every page must be re-OCR'd.
        pdf_path.unlink()
        doc2 = fitz.open()
        for i in range(2):
            doc2.new_page().insert_text((72, 72), f"COMPLETELY different content {i + 1} " * 12)
        doc2.save(str(pdf_path))
        doc2.close()

        pipeline2 = _make_pipeline(_make_config(reprocess=True))
        pipeline2.bd_detector = MagicMock()
        pipeline2.bd_detector.detect.return_value = _bd_assessment(2)
        run2_calls: list[int] = []
        _run(pipeline2, pdf_path, tmp_path, run2_calls)

        assert sorted(run2_calls) == [1, 2], (
            f"A changed PDF at the same path must re-OCR ALL pages (stale-fragment "
            f"guard); got {run2_calls}"
        )


# ---------------------------------------------------------------------------
# AC: partial doc (some terminal + some unprocessed) resumes correctly
# ---------------------------------------------------------------------------


class TestPartialDocResume:
    def test_partial_doc_resumes(self, tmp_path: Path) -> None:
        """4-page doc: pages 1-2 terminal, pages 3-4 never finished → resume
        re-OCRs only 3-4 and the final markdown carries all four pages."""
        pdf_path = _real_pdf(tmp_path, page_count=4)
        pipeline = _make_pipeline(_make_config())
        pipeline.bd_detector = MagicMock()
        pipeline.bd_detector.detect.return_value = _bd_assessment(4)

        run1_calls: list[int] = []
        _run(pipeline, pdf_path, tmp_path, run1_calls)
        assert sorted(run1_calls) == [1, 2, 3, 4]

        # Simulate a crash that left pages 3 and 4 unfinished.
        _simulate_crash_after(tmp_path, terminal_pages={1, 2}, all_pages={1, 2, 3, 4})

        pipeline2 = _make_pipeline(_make_config())
        pipeline2.bd_detector = MagicMock()
        pipeline2.bd_detector.detect.return_value = _bd_assessment(4)
        run2_calls: list[int] = []
        result2 = _run(pipeline2, pdf_path, tmp_path, run2_calls)

        assert result2.status == DocumentStatus.SUCCESS
        assert sorted(run2_calls) == [3, 4], (
            f"Partial resume must re-OCR ONLY the unfinished pages 3-4; got {run2_calls}"
        )
        for n in (1, 2, 3, 4):
            assert _PAGE_TEXTS[n] in result2.markdown, f"p{n} missing from resumed markdown"


# ---------------------------------------------------------------------------
# AC: RootIndex stays the SOLE author of root metadata.json
# ---------------------------------------------------------------------------


class TestRootIndexSoleAuthor:
    def test_no_second_metadata_writer(self, tmp_path: Path) -> None:
        """A resumed run writes root metadata.json ONLY via RootIndex.record —
        the PP-5 ledger reader/writer must not introduce a second author."""
        pdf_path = _real_pdf(tmp_path, page_count=2)
        pipeline = _make_pipeline(_make_config())
        pipeline.bd_detector = MagicMock()
        pipeline.bd_detector.detect.return_value = _bd_assessment(2)
        run1_calls: list[int] = []
        _run(pipeline, pdf_path, tmp_path, run1_calls)

        # Drop p2 so the resume actually does partial work (faithful crash state).
        _simulate_crash_after(tmp_path, terminal_pages={1}, all_pages={1, 2})

        pipeline2 = _make_pipeline(_make_config())
        pipeline2.bd_detector = MagicMock()
        pipeline2.bd_detector.detect.return_value = _bd_assessment(2)

        # Spy on RootIndex.record — it must be the ONLY path that writes root
        # metadata.json.  We patch the symbol the orchestrator imports lazily.
        record_calls: list = []
        import ocr_output_contract

        orig_record = ocr_output_contract.RootIndex.record

        def _spy_record(self, rel_key, meta):
            record_calls.append(rel_key)
            return orig_record(self, rel_key, meta)

        run2_calls: list[int] = []
        with patch.object(ocr_output_contract.RootIndex, "record", _spy_record):
            _run(pipeline2, pdf_path, tmp_path, run2_calls)

        # Root metadata.json exists and was produced via RootIndex.record only.
        root_meta = tmp_path / "metadata.json"
        assert root_meta.exists(), "root metadata.json must exist after the resumed run"
        assert record_calls, "RootIndex.record must be the writer of root metadata.json"
        # And p2 (the only non-terminal page) was the only re-OCR.
        assert run2_calls == [2], f"Only p2 should be re-OCR'd on resume; got {run2_calls}"


# ---------------------------------------------------------------------------
# AC: figure-doc resume (Option A) — a crash DURING the figure phase leaves
# pre-figure terminal sidecars; resume skips OCR for those pages and re-runs
# the figure phase.  This locks the ':pre-figures' interaction.
# ---------------------------------------------------------------------------


class TestFigureDocResume:
    def test_pre_figure_crash_skips_ocr_reruns_figures(self, tmp_path: Path) -> None:
        """save_figures=True: terminal per-page sidecars are stamped BEFORE the
        figure phase (with the BARE fingerprint).  A crash during figures must,
        on resume, skip OCR for the terminal pages (inner ledger hit) while the
        figure phase runs again — the doc-level ':pre-figures' record keeps the
        whole doc from being skipped at the outer gate."""
        pdf_path = _real_pdf(tmp_path, page_count=2)
        config = _make_config(save_figures=True)
        pipeline = _make_pipeline(config)
        pipeline.bd_detector = MagicMock()
        pipeline.bd_detector.detect.return_value = _bd_assessment(2)

        # Stub the figure phase to a no-op passthrough so the test stays hermetic
        # (no real figure extraction / VLM); the inner-ledger behaviour is what we
        # are validating, not figure embedding itself.
        figure_calls: list[int] = []

        def _noop_figures(state, result, output_dir, text):
            figure_calls.append(1)
            return text

        run1_calls: list[int] = []
        with patch.object(pipeline, "_describe_and_embed_figures", side_effect=_noop_figures):
            result1 = _run(pipeline, pdf_path, tmp_path, run1_calls)
        assert result1.status == DocumentStatus.SUCCESS
        assert sorted(run1_calls) == [1, 2], f"Run 1 must OCR both pages; got {run1_calls}"
        assert figure_calls, "figure phase must have run in run 1 (save_figures=True)"

        # Per-page terminal sidecars must carry the BARE fingerprint (not suffixed)
        # so the inner ledger matches on resume even while the doc record is
        # ':pre-figures'.
        pages_dir = _doc_dir(tmp_path) / "pages"
        for n in (1, 2):
            meta = json.loads((pages_dir / f"{n:05d}.json").read_text())
            assert meta["terminal"] is True
            assert meta["run_fingerprint"] == pipeline._run_fingerprint(), (
                "per-page sidecar must store the bare run fingerprint"
            )
            assert ":pre-figures" not in meta["run_fingerprint"]

        # Simulate a crash DURING the figure phase: both pages are terminal (OCR
        # done + sidecar stamped pre-figures), but assemble never finished so the
        # doc record is absent / not COMPLETED.  Keep BOTH pages terminal.
        for meta_file in (tmp_path / "metadata.json", _doc_dir(tmp_path) / "metadata.json"):
            if meta_file.exists():
                meta_file.unlink()

        pipeline2 = _make_pipeline(_make_config(save_figures=True))
        pipeline2.bd_detector = MagicMock()
        pipeline2.bd_detector.detect.return_value = _bd_assessment(2)
        figure_calls2: list[int] = []

        def _noop_figures2(state, result, output_dir, text):
            figure_calls2.append(1)
            return text

        run2_calls: list[int] = []
        with patch.object(pipeline2, "_describe_and_embed_figures", side_effect=_noop_figures2):
            result2 = _run(pipeline2, pdf_path, tmp_path, run2_calls)

        assert result2.status == DocumentStatus.SUCCESS
        assert run2_calls == [], (
            f"Both pages were terminal pre-figures; resume must re-OCR NONE; got {run2_calls}"
        )
        assert figure_calls2, "the figure phase must RE-RUN on resume (Option A)"


# ---------------------------------------------------------------------------
# AC: resume is PROVIDER-INDEPENDENT — terminal pages load from disk even when
# the live provider ladder is empty (e.g. ollama not running at re-run time).
# ---------------------------------------------------------------------------


class TestResumeWithEmptyLadder:
    def test_resume_works_with_empty_provider_ladder(self, tmp_path: Path) -> None:
        """A partial doc with terminal sidecars for pages <= K, re-run with NO
        provider available (empty ladder), must still load pages <= K from their
        terminal fragments (resuming needs no provider).  Pages > K, which need a
        provider that is now absent, are reported unprocessed — never lost.

        Regression: the empty-ladder early-return used to fire BEFORE the per-page
        resume gate, so terminal pages were not resumed when the ladder was empty.
        """
        pdf_path = _real_pdf(tmp_path, page_count=3)
        pipeline = _make_pipeline(_make_config())
        pipeline.bd_detector = MagicMock()
        pipeline.bd_detector.detect.return_value = _bd_assessment(3)

        # --- Run 1: complete the whole doc with a live provider. ---
        run1_calls: list[int] = []
        result1 = _run(pipeline, pdf_path, tmp_path, run1_calls)
        assert result1.status == DocumentStatus.SUCCESS
        assert sorted(run1_calls) == [1, 2, 3]

        # --- Crash: pages 1-2 terminal, p3 unfinished, no COMPLETED doc record. ---
        _simulate_crash_after(tmp_path, terminal_pages={1, 2}, all_pages={1, 2, 3})

        # --- Run 2: EMPTY provider ladder (no provider available now). ---
        pipeline2 = _make_pipeline(_make_config())
        pipeline2.bd_detector = MagicMock()
        pipeline2.bd_detector.detect.return_value = _bd_assessment(3)

        run2_calls: list[int] = []
        with (
            patch.object(pipeline2, "_available_engines_for_agentic", return_value=[]),
            patch(
                "socr.pipeline.orchestrator.route_page",
                side_effect=_counting_route(run2_calls),
            ),
        ):
            result2 = pipeline2.process(pdf_path, tmp_path)

        # route_page must NEVER be called — there is no provider.  Terminal pages
        # are loaded from disk; the unresumable page is left for the no-provider path.
        assert run2_calls == [], (
            f"route_page must not be called with an empty ladder; got {run2_calls}"
        )

        # Pages 1-2 (terminal) must be resumed from their fragments — their text
        # is present in the output even though no provider was available.
        for n in (1, 2):
            assert _PAGE_TEXTS[n] in result2.markdown, (
                f"p{n} was terminal on disk and must be RESUMED even with an empty "
                f"ladder; it is missing from the output markdown"
            )

        # p3 (non-terminal, needs a provider that is absent) must NOT be silently
        # presented as a clean success: it is reported as a failure / native
        # fallback, never a clean resumed page.  Its real OCR text must be absent.
        assert _PAGE_TEXTS[3] not in result2.markdown, (
            "p3 had no terminal fragment and no provider; its OCR text must NOT appear"
        )
        assert result2.status != DocumentStatus.SUCCESS, (
            "A doc with an unresumable, unprocessable page must not report clean success"
        )

    def test_empty_ladder_no_terminal_pages_still_falls_back(self, tmp_path: Path) -> None:
        """With an empty ladder AND no terminal fragments at all, the no-provider
        path still runs (unchanged behavior): the doc is not a clean success and
        route_page is never called."""
        pdf_path = _real_pdf(tmp_path, page_count=2)
        pipeline = _make_pipeline(_make_config())
        pipeline.bd_detector = MagicMock()
        pipeline.bd_detector.detect.return_value = _bd_assessment(2)

        route_calls: list[int] = []
        with (
            patch.object(pipeline, "_available_engines_for_agentic", return_value=[]),
            patch(
                "socr.pipeline.orchestrator.route_page",
                side_effect=_counting_route(route_calls),
            ),
        ):
            result = pipeline.process(pdf_path, tmp_path)

        assert route_calls == [], "no provider → route_page never called"
        assert result.status != DocumentStatus.SUCCESS, (
            "no provider and no terminal fragments → not a clean success"
        )
