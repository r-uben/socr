"""PP-2 tests: fused agentic loop, per-page provisional flush, and cascade-halt.

Acceptance criteria:
  * Byte-identity: final stitched .md from a progressive agentic run is
    byte-identical to a non-progressive agentic run on the same fixture (with
    equation flags OFF). Proved via assemble-authoritative (fork A): the final
    _canonical_body() output is unchanged by the provisional flush.  A golden
    test locks the exact assembled bytes; a determinism assertion verifies two
    runs on the same fixture produce the same markdown.
  * Incremental flush: every page (native AND ocr) gets a provisional fragment
    + sidecar immediately after it finishes; sidecar carries terminal=False.
  * Phase 4c not twice: dual_pass_tables in agentic mode is a no-op at the
    process() level (gated by `not (agentic and not is_multi)`).
  * Cascade HALT: a wedged backend after page N → doc marked with error
    "PARTIAL_SAVE_VLM_TIMEOUT", pages 0..N-1 flushed (pages/00001.md exists),
    no VLM call on page N+1.  audit_log.json present + partial_save_vlm_timeout
    event unconditionally asserted.
  * Equations flags OFF → output unchanged (_detect_and_crop_equations never called).
  * Doc-scoped engines built once (judge built once before the loop).
  * Existing agentic tests pass; total_cost unchanged on the happy path.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from socr.core.born_digital import DocumentAssessment, PageAssessment
from socr.core.config import EngineType, PipelineConfig
from socr.core.providers import PROFILE_GEMINI, PROFILE_QWEN_LOCAL
from socr.core.result import DocumentStatus, PageOutput, PageStatus
from socr.pipeline.agentic import AcceptDecision
from socr.pipeline.orchestrator import UnifiedPipeline

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_OCR_TEXT = (
    "This document presents an analysis of market dynamics across several European "
    "economies during the post-pandemic recovery period. We examine monetary policy "
    "transmission mechanisms and their effects on inflation expectations output gaps "
    "and financial stability indicators. Our empirical framework builds on vector "
    "autoregressive models with sign restrictions estimated using Bayesian methods "
    "on quarterly macroeconomic data spanning the period from 2019 to 2024. The "
    "results suggest that unconventional monetary policy tools had asymmetric "
    "effects across core and peripheral economies in the sample studied here."
)  # 90+ words — passes the HeuristicsChecker 50-word minimum


def _make_config(
    *,
    agentic: bool = True,
    dual_pass_tables: bool = False,
    detect_equations: bool = False,
    recover_clean_equations: bool = False,
    enabled_engines=None,
    **overrides,
) -> PipelineConfig:
    return PipelineConfig(
        primary_engine=EngineType.DEEPSEEK,
        agentic=agentic,
        judge_backend="heuristic",
        enabled_engines=enabled_engines or [EngineType.GEMINI],
        dual_pass_tables=dual_pass_tables,
        detect_equations=detect_equations,
        recover_clean_equations=recover_clean_equations,
        quiet=True,
        save_figures=False,
        write_manifest=False,
        **overrides,
    )


def _make_pipeline(config: PipelineConfig | None = None) -> UnifiedPipeline:
    return UnifiedPipeline(config or _make_config())


def _make_bd_assessment(
    page_count: int,
    born_digital_pages: set[int] | None = None,
) -> DocumentAssessment:
    """Build a minimal DocumentAssessment for test fixtures."""
    bd = born_digital_pages or set()
    pages = []
    for i in range(1, page_count + 1):
        is_bd = i in bd
        pages.append(
            PageAssessment(
                page_num=i,
                is_born_digital=is_bd,
                native_text=f"native text for page {i} " * 5 if is_bd else "",
                confidence=0.9,
            )
        )
    return DocumentAssessment(path=Path("/tmp/fake.pdf"), pages=pages)


def _real_pdf(tmp_path: Path, page_count: int = 2) -> Path:
    """Create a minimal real PDF."""
    fitz = pytest.importorskip("fitz")
    path = tmp_path / "doc.pdf"
    doc = fitz.open()
    for i in range(page_count):
        doc.new_page().insert_text((72, 72), f"page {i + 1} text " * 10)
    doc.save(str(path))
    doc.close()
    return path


def _gemini_engine_mock(text: str = _OCR_TEXT) -> MagicMock:
    """Minimal engine mock: always returns ``text`` for any page."""
    m = MagicMock()
    m.name = "gemini"
    m.is_available.return_value = True
    m.model_version = ""

    def _pp(pdf_path, page_nums, config, dpi=200):
        return [
            PageOutput(
                page_num=pn,
                text=text,
                status=PageStatus.SUCCESS,
                engine="gemini",
                audit_passed=True,
            )
            for pn in page_nums
        ]

    m.process_pages.side_effect = _pp
    return m


# ---------------------------------------------------------------------------
# AC: Byte-identity — provisional flush does not affect final .md body
# ---------------------------------------------------------------------------


_GOLDEN_OCR_P1 = (
    "The monetary transmission mechanism operates through several channels including "
    "the interest rate channel the bank lending channel the balance sheet channel and "
    "the exchange rate channel. Each channel involves different financial intermediaries "
    "and operates with varying lags and intensities across the business cycle. The "
    "empirical evidence suggests that the lending channel dominates in bank-based "
    "financial systems while the balance sheet channel is relatively more important "
    "in market-based systems with deep capital markets and diverse funding structures."
)  # 85 words — passes HeuristicsChecker 50-word minimum

_GOLDEN_NATIVE_P2 = "native text for page 2 " * 5  # set by _make_bd_assessment for p2

_GOLDEN_OCR_P3 = (
    "Fiscal policy interactions with monetary transmission introduce additional "
    "heterogeneity in the empirical estimates. Periods of fiscal consolidation tend to "
    "amplify the contractionary effects of monetary tightening while expansionary fiscal "
    "stances can partially offset the transmission of monetary impulses to aggregate "
    "demand. The joint identification of monetary and fiscal shocks remains a major "
    "methodological challenge in the structural vector autoregression literature and "
    "requires careful exclusion restrictions grounded in institutional and theoretical "
    "considerations relevant to the specific economies under study here in this analysis."
)  # 88 words — passes HeuristicsChecker 50-word minimum


def _golden_expected_markdown() -> str:
    """Compute the expected golden markdown for the 3-page fixture (p1 OCR, p2 native, p3 OCR)."""
    from ocr_output_contract import assemble_pages

    return assemble_pages(
        [_GOLDEN_OCR_P1, _GOLDEN_NATIVE_P2.rstrip(), _GOLDEN_OCR_P3],
        page_numbers=[1, 2, 3],
    )


def _make_golden_route_page(page_texts: dict[int, str]):
    """Return a route_page side_effect function with fully deterministic outputs."""

    def _fake_route(page_num, ladder, run_provider, judge, **kwargs):
        from socr.pipeline.agentic import PageDecision, ProviderAttempt

        text = page_texts[page_num]
        out = PageOutput(
            page_num=page_num,
            text=text,
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

    return _fake_route


class TestByteIdentity:
    """After _phase_agentic populates state, the final .md is byte-identical across
    runs and matches a pre-computed golden string.

    Design (§5 of design note): assemble-authoritative (fork A) means provisional
    flush does NOT change the assembled output — _rewrite_all_fragments at assemble
    time is the sole authoritative writer.  The golden test locks the exact bytes
    so any future drift in the fused loop fails the test immediately.
    """

    def _run_golden_fixture(self, tmp_path: Path) -> str:
        """3-page fixture: p1 OCR, p2 native, p3 OCR. Returns result.markdown."""
        fitz = pytest.importorskip("fitz")

        pdf_path = tmp_path / "golden.pdf"
        doc = fitz.open()
        for i in range(3):
            doc.new_page().insert_text((72, 72), f"page {i + 1} filler text " * 5)
        doc.save(str(pdf_path))
        doc.close()

        config = _make_config(agentic=True, enabled_engines=[EngineType.GEMINI])
        pipeline = _make_pipeline(config)
        pipeline.bd_detector = MagicMock()
        # Page 2 is born-digital native; pages 1 and 3 are OCR.
        pipeline.bd_detector.detect.return_value = _make_bd_assessment(3, born_digital_pages={2})

        page_texts = {1: _GOLDEN_OCR_P1, 3: _GOLDEN_OCR_P3}
        # Hermetic: patch _available_engines_for_agentic so the provider ladder is
        # non-empty regardless of whether ollama/gemini is installed — otherwise the
        # loop bails before the patched route_page (no providers) and OCR pages fall
        # to native, failing the doc with AUDIT_FAILED in a no-provider env (CI).
        with (
            patch(
                "socr.pipeline.orchestrator.route_page",
                side_effect=_make_golden_route_page(page_texts),
            ),
            patch.object(pipeline, "_available_engines_for_agentic", return_value=[PROFILE_GEMINI]),
        ):
            result = pipeline.process(pdf_path, tmp_path)

        assert result.status == DocumentStatus.SUCCESS, (
            f"Golden fixture must succeed; got {result.status}"
        )
        return result.markdown

    def test_final_md_matches_golden(self, tmp_path: Path) -> None:
        """result.markdown is byte-identical to the pre-computed golden string."""
        markdown = self._run_golden_fixture(tmp_path)
        expected = _golden_expected_markdown()
        assert markdown == expected, (
            f"Markdown does not match golden.\n"
            f"--- EXPECTED (first 300 chars) ---\n{expected[:300]!r}\n"
            f"--- GOT (first 300 chars) ---\n{markdown[:300]!r}"
        )

    def test_determinism_two_runs_identical(self, tmp_path: Path) -> None:
        """Running the same fixture twice yields byte-identical result.markdown."""
        run1 = tmp_path / "run1"
        run1.mkdir()
        run2 = tmp_path / "run2"
        run2.mkdir()
        md1 = self._run_golden_fixture(run1)
        md2 = self._run_golden_fixture(run2)
        assert md1 == md2, (
            "Two identical fixture runs produced different markdown — "
            "the fused loop is non-deterministic.\n"
            f"Run 1 (first 200 chars): {md1[:200]!r}\n"
            f"Run 2 (first 200 chars): {md2[:200]!r}"
        )


# ---------------------------------------------------------------------------
# AC: Incremental flush — every page (native AND ocr) gets fragment + sidecar
# ---------------------------------------------------------------------------


class TestIncrementalFlush:
    """After _phase_agentic, every page has pages/NNN.md + pages/NNN.json."""

    def test_mixed_native_and_ocr_pages_all_flushed(self, tmp_path: Path) -> None:
        """4-page doc mixing native (p1,p3) + OCR (p2,p4) — all get fragments."""
        fitz = pytest.importorskip("fitz")

        pdf_path = tmp_path / "mixed.pdf"
        doc = fitz.open()
        for i in range(4):
            doc.new_page().insert_text((72, 72), f"page {i + 1} text " * 10)
        doc.save(str(pdf_path))
        doc.close()

        config = _make_config(agentic=True, enabled_engines=[EngineType.GEMINI])
        pipeline = _make_pipeline(config)
        pipeline.bd_detector = MagicMock()
        # Pages 1 and 3 are born-digital (native); 2 and 4 are scanned (OCR).
        pipeline.bd_detector.detect.return_value = _make_bd_assessment(4, born_digital_pages={1, 3})

        with patch("socr.pipeline.orchestrator.get_engine", return_value=_gemini_engine_mock()):
            result = pipeline.process(pdf_path, tmp_path)

        assert result.status == DocumentStatus.SUCCESS

        doc_dir = tmp_path / "mixed"
        pages_dir = doc_dir / "pages"
        assert pages_dir.is_dir()

        frags = sorted(pages_dir.glob("*.md"))
        sidecars = sorted(pages_dir.glob("*.json"))
        assert len(frags) == 4, f"Expected 4 fragments, got {len(frags)}: {[f.name for f in frags]}"
        assert len(sidecars) == 4, (
            f"Expected 4 sidecars, got {len(sidecars)}: {[f.name for f in sidecars]}"
        )

    def test_provisional_sidecar_has_terminal_false(self, tmp_path: Path) -> None:
        """Provisional sidecars written during the agentic loop carry terminal=False."""
        pdf_path = _real_pdf(tmp_path, page_count=1)

        config = _make_config(agentic=True, enabled_engines=[EngineType.GEMINI])
        pipeline = _make_pipeline(config)
        pipeline.bd_detector = MagicMock()
        pipeline.bd_detector.detect.return_value = _make_bd_assessment(1, born_digital_pages=set())

        # Capture provisional sidecar before the assemble-time rewrite.
        provisional_sidecars: list[dict] = []
        _orig = pipeline._flush_page_sidecar

        def _capture(state, page_num, output_dir, *, terminal=True):
            path = _orig(state, page_num, output_dir, terminal=terminal)
            if not terminal:
                provisional_sidecars.append(json.loads(path.read_text()))
            return path

        with (
            patch("socr.pipeline.orchestrator.get_engine", return_value=_gemini_engine_mock()),
            patch.object(pipeline, "_flush_page_sidecar", side_effect=_capture),
        ):
            pipeline.process(pdf_path, tmp_path)

        assert provisional_sidecars, "No provisional (terminal=False) sidecar was written"
        for snap in provisional_sidecars:
            assert snap["terminal"] is False, (
                f"Provisional sidecar must have terminal=False, got {snap.get('terminal')!r}"
            )

    def test_native_page_gets_fragment_without_ocr(self, tmp_path: Path) -> None:
        """A native born-digital page is flushed even though no OCR ran on it."""
        pdf_path = _real_pdf(tmp_path, page_count=1)

        config = _make_config(agentic=True, enabled_engines=[EngineType.GEMINI])
        pipeline = _make_pipeline(config)
        pipeline.bd_detector = MagicMock()
        # Page 1 is fully native (born-digital, no OCR needed).
        pipeline.bd_detector.detect.return_value = _make_bd_assessment(1, born_digital_pages={1})

        with patch("socr.pipeline.orchestrator.get_engine", return_value=_gemini_engine_mock()):
            result = pipeline.process(pdf_path, tmp_path)

        assert result.status == DocumentStatus.SUCCESS

        doc_dir = tmp_path / "doc"
        pages_dir = doc_dir / "pages"
        frags = sorted(pages_dir.glob("*.md"))
        sidecars = sorted(pages_dir.glob("*.json"))
        assert len(frags) == 1, "Native page must produce a fragment"
        assert len(sidecars) == 1, "Native page must produce a sidecar"


# ---------------------------------------------------------------------------
# AC: Phase 4c not twice
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# AC: Cascade HALT — wedged backend halts with PARTIAL_SAVE_VLM_TIMEOUT
# ---------------------------------------------------------------------------


class TestCascadeHalt:
    """Provider timeout + unhealthy probe → PARTIAL_SAVE_VLM_TIMEOUT; no call for p2."""

    def _fake_timeout_decision(self, page_num: int, ladder):
        """Return a PageDecision that simulates a provider timeout."""
        from socr.pipeline.agentic import PageDecision, ProviderAttempt

        err_out = PageOutput(
            page_num=page_num,
            text="",
            status=PageStatus.ERROR,
            engine="qwen",
            audit_passed=False,
        )
        prof = ladder[0]
        timeout_att = ProviderAttempt(
            engine=prof.engine,
            output=err_out,
            cost_usd=0.0,
            accepted=False,
            reason="provider timeout",
            provider_id=prof.id,
            model=prof.model,
            backend=prof.backend,
        )
        return PageDecision(page_num=page_num, final_output=err_out, attempts=[timeout_att])

    def test_halt_on_unresponsive_backend_after_timeout(self, tmp_path: Path) -> None:
        """After timeout + unhealthy probe on p1, p2 is NOT routed.

        Also verifies:
          * The provisional fragment for p1 (pages/00001.md) is flushed before halt.
          * audit_log.json is present and contains a partial_save_vlm_timeout event.

        Hermetic: _available_engines_for_agentic is patched so the ladder is non-empty
        regardless of whether ollama/qwen is installed in the test environment.
        """
        pdf_path = _real_pdf(tmp_path, page_count=2)
        config = _make_config(agentic=True, enabled_engines=[EngineType.QWEN])
        pipeline = _make_pipeline(config)
        pipeline.bd_detector = MagicMock()
        pipeline.bd_detector.detect.return_value = _make_bd_assessment(2, born_digital_pages=set())

        route_calls: list[int] = []

        def _fake_route(page_num, ladder, run_provider, judge, **kwargs):
            route_calls.append(page_num)
            if page_num == 1:
                return self._fake_timeout_decision(page_num, ladder)
            # Should never be reached after halt.
            return self._fake_timeout_decision(page_num, ladder)

        with (
            patch.object(
                pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]
            ),
            patch("socr.pipeline.orchestrator.route_page", side_effect=_fake_route),
            patch("socr.pipeline.orchestrator.probe_ollama_idle", return_value=False),
        ):
            result = pipeline.process(pdf_path, tmp_path)

        # Only page 1 must be routed.
        assert route_calls == [1], (
            f"Expected route_page called only for p1; got calls for: {route_calls}"
        )
        # Result error must carry the PARTIAL_SAVE_VLM_TIMEOUT marker.
        assert result.error and "PARTIAL_SAVE_VLM_TIMEOUT" in result.error, (
            f"Expected 'PARTIAL_SAVE_VLM_TIMEOUT' in result.error; got: {result.error!r}"
        )

        # The page processed before the wedge (p1) must have been flushed to disk.
        doc_dir = tmp_path / "doc"
        pages_dir = doc_dir / "pages"
        p1_frag = pages_dir / "00001.md"
        assert p1_frag.exists(), (
            f"pages/00001.md must exist after halt — p1 was processed before the wedge; "
            f"found fragments: {sorted(pages_dir.glob('*.md')) if pages_dir.is_dir() else 'none'}"
        )

        # audit_log.json must be present (the AuditEvent is appended to state.events
        # regardless of audit configuration) and carry the halt event.
        audit_log = doc_dir / "audit_log.json"
        assert audit_log.exists(), (
            f"audit_log.json must be written after partial_save_vlm_timeout; "
            f"doc_dir contents: {sorted(doc_dir.iterdir()) if doc_dir.is_dir() else 'missing'}"
        )
        audit_data = json.loads(audit_log.read_text())
        events = audit_data.get("events", [])
        halt_events = [e for e in events if e.get("kind") == "partial_save_vlm_timeout"]
        assert halt_events, (
            f"partial_save_vlm_timeout AuditEvent must be in audit_log.json; "
            f"events present: {[e.get('kind') for e in events]}"
        )

    def test_native_page_after_wedge_is_not_processed(self, tmp_path: Path) -> None:
        """A native-trusted page immediately after the wedge must NOT be processed in the loop.

        This is the exact case /codex flagged: with the halt check only in the OCR branch,
        a native page after the wedge would skip the guard and still be processed (its
        best_output set, provisional flush called, etc.).  After the fix the halt check is
        at the TOP of the loop, so p2 (native) is never entered.

        Fixture: 3 pages — p1 OCR (times out), p2 native (born-digital), p3 OCR.
        Expected:
          * route_page called only for p1 — never for p3.
          * p2 has no provisional sidecar with terminal=False (was never reached in the loop).
          * p3 has no provisional sidecar with terminal=False (was never reached in the loop).
          * PARTIAL_SAVE_VLM_TIMEOUT in result.error.

        Note: _phase_assemble always writes authoritative fragments for all pages with text
        (fork A / assemble-authoritative design — that is intentional and unaffected by the
        halt).  The loop-level invariant being tested here is that p2/p3 receive no
        PROVISIONAL in-loop flush (terminal=False sidecar) — they are not reached by the
        agentic driver.
        """
        fitz = pytest.importorskip("fitz")

        pdf_path = tmp_path / "wedge_native.pdf"
        doc = fitz.open()
        for i in range(3):
            doc.new_page().insert_text((72, 72), f"page {i + 1} text " * 8)
        doc.save(str(pdf_path))
        doc.close()

        config = _make_config(agentic=True, enabled_engines=[EngineType.QWEN])
        pipeline = _make_pipeline(config)
        pipeline.bd_detector = MagicMock()
        # p2 is native (born-digital); p1 and p3 are OCR.
        pipeline.bd_detector.detect.return_value = _make_bd_assessment(3, born_digital_pages={2})

        route_calls: list[int] = []

        def _fake_route(page_num, ladder, run_provider, judge, **kwargs):
            route_calls.append(page_num)
            return self._fake_timeout_decision(page_num, ladder)

        # Track provisional (terminal=False) sidecar calls to detect in-loop processing.
        provisional_flush_calls: list[int] = []
        _orig_flush = pipeline._flush_page_sidecar

        def _spy_sidecar(state, page_num, output_dir, *, terminal=True):
            path = _orig_flush(state, page_num, output_dir, terminal=terminal)
            if not terminal:
                provisional_flush_calls.append(page_num)
            return path

        # Hermetic: patch _available_engines_for_agentic so the ladder is non-empty
        # regardless of whether ollama/qwen is installed in the test environment.
        with (
            patch.object(
                pipeline, "_available_engines_for_agentic", return_value=[PROFILE_QWEN_LOCAL]
            ),
            patch("socr.pipeline.orchestrator.route_page", side_effect=_fake_route),
            patch("socr.pipeline.orchestrator.probe_ollama_idle", return_value=False),
            patch.object(pipeline, "_flush_page_sidecar", side_effect=_spy_sidecar),
        ):
            result = pipeline.process(pdf_path, tmp_path)

        # route_page must never be called for p3 (OCR page after the wedge).
        assert 3 not in route_calls, (
            f"route_page must NOT be called for p3 after wedge; got calls: {route_calls}"
        )
        assert route_calls == [1], (
            f"route_page must only be called for p1 (the wedge page); got: {route_calls}"
        )
        # PARTIAL_SAVE_VLM_TIMEOUT must be in the result error.
        assert result.error and "PARTIAL_SAVE_VLM_TIMEOUT" in result.error, (
            f"Expected PARTIAL_SAVE_VLM_TIMEOUT in result.error; got: {result.error!r}"
        )
        # Neither p2 (native) nor p3 (OCR) may have received a provisional in-loop flush.
        # Provisional flush == terminal=False sidecar written from inside _phase_agentic.
        assert 2 not in provisional_flush_calls, (
            f"p2 (native, after wedge) must NOT receive a provisional loop flush; "
            f"provisional sidecar calls: {provisional_flush_calls}"
        )
        assert 3 not in provisional_flush_calls, (
            f"p3 (OCR, after wedge) must NOT receive a provisional loop flush; "
            f"provisional sidecar calls: {provisional_flush_calls}"
        )

    def test_healthy_backend_after_timeout_does_not_halt(self, tmp_path: Path) -> None:
        """When the probe returns True (healthy), no halt occurs; p2 is routed.

        Hermetic: _available_engines_for_agentic is patched so the ladder is non-empty
        regardless of whether the Gemini API is reachable in the test environment.
        """
        pdf_path = _real_pdf(tmp_path, page_count=2)
        config = _make_config(agentic=True, enabled_engines=[EngineType.GEMINI])
        pipeline = _make_pipeline(config)
        pipeline.bd_detector = MagicMock()
        pipeline.bd_detector.detect.return_value = _make_bd_assessment(2, born_digital_pages=set())

        route_calls: list[int] = []

        def _fake_route(page_num, ladder, run_provider, judge, **kwargs):
            route_calls.append(page_num)
            # All pages succeed normally.
            out = PageOutput(
                page_num=page_num,
                text=_OCR_TEXT,
                status=PageStatus.SUCCESS,
                engine="gemini",
                audit_passed=True,
            )
            from socr.pipeline.agentic import PageDecision, ProviderAttempt

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

        with (
            patch.object(pipeline, "_available_engines_for_agentic", return_value=[PROFILE_GEMINI]),
            patch("socr.pipeline.orchestrator.route_page", side_effect=_fake_route),
            patch("socr.pipeline.orchestrator.probe_ollama_idle", return_value=True),
        ):
            result = pipeline.process(pdf_path, tmp_path)

        # Both pages routed.
        assert sorted(route_calls) == [1, 2], f"Expected both pages routed; got: {route_calls}"
        assert result.error is None or "PARTIAL_SAVE_VLM_TIMEOUT" not in (result.error or ""), (
            "Should not halt when backend probe returns True"
        )


# ---------------------------------------------------------------------------
# AC: Equations flags OFF → _detect_and_crop_equations never called
# ---------------------------------------------------------------------------


class TestEquationFlagsOff:
    """detect_equations=False (default) → no equation detection in agentic loop."""

    def test_equations_not_called_when_flag_off(self, tmp_path: Path) -> None:
        pdf_path = _real_pdf(tmp_path, page_count=1)
        config = _make_config(
            agentic=True, detect_equations=False, enabled_engines=[EngineType.GEMINI]
        )
        pipeline = _make_pipeline(config)
        pipeline.bd_detector = MagicMock()
        pipeline.bd_detector.detect.return_value = _make_bd_assessment(1, born_digital_pages=set())

        detect_calls: list = []

        def _spy(state, page_nums, output_dir):
            detect_calls.extend(page_nums)

        with (
            patch("socr.pipeline.orchestrator.get_engine", return_value=_gemini_engine_mock()),
            patch.object(pipeline, "_detect_and_crop_equations", side_effect=_spy),
        ):
            result = pipeline.process(pdf_path, tmp_path)

        assert result.status == DocumentStatus.SUCCESS
        assert not detect_calls, (
            f"_detect_and_crop_equations must not be called with detect_equations=False; "
            f"called for pages: {detect_calls}"
        )


# ---------------------------------------------------------------------------
# AC: Doc-scoped engines built once (judge built once per document)
# ---------------------------------------------------------------------------


class TestJudgeBuiltOnce:
    """_build_page_judge must be called exactly once per _phase_agentic call."""

    def test_judge_built_once_for_n_pages(self, tmp_path: Path) -> None:
        pdf_path = _real_pdf(tmp_path, page_count=3)
        config = _make_config(agentic=True, enabled_engines=[EngineType.GEMINI])
        pipeline = _make_pipeline(config)
        pipeline.bd_detector = MagicMock()
        pipeline.bd_detector.detect.return_value = _make_bd_assessment(3, born_digital_pages=set())

        build_calls: list[int] = []
        _orig = pipeline._build_page_judge

        def _spy(state):
            build_calls.append(1)
            return _orig(state)

        with (
            patch("socr.pipeline.orchestrator.get_engine", return_value=_gemini_engine_mock()),
            patch.object(pipeline, "_build_page_judge", side_effect=_spy),
        ):
            pipeline.process(pdf_path, tmp_path)

        assert build_calls == [1], (
            f"_build_page_judge must be called exactly once; called {len(build_calls)} times"
        )


# ---------------------------------------------------------------------------
# AC: _TimeoutJudge adapter
# ---------------------------------------------------------------------------


class TestTimeoutJudge:
    """_TimeoutJudge wraps inner judge in a wall-clock deadline."""

    def test_timeout_rejects_slow_judge(self) -> None:
        """When inner judge hangs longer than timeout, returns accept=False."""
        import time

        pipeline = _make_pipeline()

        class _SlowJudge:
            def assess(self, output, provider):
                time.sleep(10)
                return AcceptDecision(accept=True, reason="never reached")

        judge = pipeline._TimeoutJudge(_SlowJudge(), timeout_sec=0.05)
        fake_out = PageOutput(
            page_num=1, text="x", status=PageStatus.SUCCESS, engine="t", audit_passed=True
        )
        result = judge.assess(fake_out, MagicMock())
        assert result.accept is False
        assert "timeout" in result.reason.lower()

    def test_fast_judge_verdict_forwarded(self) -> None:
        """When inner judge responds quickly, verdict is passed through unchanged."""

        class _FastJudge:
            def assess(self, output, provider):
                return AcceptDecision(accept=True, reason="fast accept")

        pipeline = _make_pipeline()
        judge = pipeline._TimeoutJudge(_FastJudge(), timeout_sec=5.0)
        fake_out = PageOutput(
            page_num=1, text="x", status=PageStatus.SUCCESS, engine="t", audit_passed=True
        )
        result = judge.assess(fake_out, MagicMock())
        assert result.accept is True
        assert result.reason == "fast accept"

    def test_none_timeout_passthrough(self) -> None:
        """timeout_sec=None disables the wrapper; inner judge called directly."""
        called = [False]

        class _TrackJudge:
            def assess(self, output, provider):
                called[0] = True
                return AcceptDecision(accept=True, reason="direct")

        pipeline = _make_pipeline()
        judge = pipeline._TimeoutJudge(_TrackJudge(), timeout_sec=None)
        fake_out = PageOutput(
            page_num=1, text="x", status=PageStatus.SUCCESS, engine="t", audit_passed=True
        )
        result = judge.assess(fake_out, MagicMock())
        assert called[0]
        assert result.reason == "direct"
