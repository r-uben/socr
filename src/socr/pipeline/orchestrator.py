"""Unified OCR pipeline orchestrator.

Drives DocumentState through:
  1. Analyze    -- born-digital detection
  2. Backbone   -- primary engine OCR
  3. Score      -- heuristic quality audit
  4. Repair     -- selective fallback on failed pages
  4b. Consensus -- multi-engine best-output selection (optional)
  5. Assemble   -- stitch final output and save

Replaces StandardPipeline's ad-hoc primary/audit/fallback stages with a
structured loop that operates on the DocumentState blackboard.
"""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path

from rich.console import Console

from socr.audit.heuristics import HeuristicsChecker
from socr.audit.scorer import FailureModeScorer
from socr.core.born_digital import BornDigitalDetector
from socr.core.chunker import PDFChunker
from socr.core.config import EngineType, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.metadata import MetadataManager
from socr.core.result import (
    DocumentStatus,
    EngineResult,
    FailureMode,
    FigureInfo,
    PageOutput,
    PageStatus,
)
from socr.core.state import DocumentState
from socr.engines.base import BaseEngine, sanitize_filename
from socr.engines.registry import get_engine
from socr.figures.extractor import FigureExtractor
from socr.pipeline.consensus import ConsensusEngine
from socr.pipeline.repair import RepairRouter

logger = logging.getLogger(__name__)
console = Console()


class UnifiedPipeline:
    """5-phase OCR pipeline orchestrator.

    Usage::

        pipeline = UnifiedPipeline(config)
        result = pipeline.process(pdf_path, output_dir)
        results = pipeline.process_batch(input_dir, output_dir)
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.heuristics = HeuristicsChecker(min_word_count=config.audit_min_words)
        self.scorer = FailureModeScorer(checker=self.heuristics)
        self.repair_router = RepairRouter(config)
        self.bd_detector = BornDigitalDetector()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, pdf_path: Path, output_dir: Path | None = None) -> EngineResult:
        """Process a single PDF through the 5-phase loop.

        Returns an EngineResult summarising the best extraction.
        """
        pdf_path = Path(pdf_path)
        out_dir = output_dir or self.config.output_dir

        doc = DocumentHandle.from_path(pdf_path)
        state = DocumentState(handle=doc)

        if not self.config.quiet:
            console.print(f"[blue]Processing:[/blue] {doc.filename}")
            console.print(f"[dim]{doc.page_count} pages, {doc.size_mb:.1f} MB[/dim]")

        # Phase 1: Analyze
        self._phase_analyze(state)

        # Phase 2: Backbone OCR
        backbone_result = self._phase_backbone(state, out_dir)

        # Phase 3: Score
        if backbone_result and backbone_result.success and self.config.audit_enabled:
            self._phase_score(state, backbone_result)

        # Phase 4: Selective Repair (loops up to max_retries)
        if self.config.audit_enabled:
            self._phase_repair(state, out_dir)

        # Phase 4b: Consensus (optional, after repair)
        if self.config.consensus_enabled:
            self._phase_consensus(state)

        # Phase 5: Assemble
        final_result = self._phase_assemble(state, out_dir)

        if not self.config.quiet:
            self._print_summary(final_result, state)

        return final_result

    def process_batch(
        self, input_dir: Path, output_dir: Path | None = None
    ) -> list[EngineResult]:
        """Process all PDFs in a directory with incremental tracking."""
        input_dir = Path(input_dir)
        out_dir = output_dir or self.config.output_dir
        meta = MetadataManager(out_dir)

        pdfs = sorted(input_dir.glob("*.pdf"))
        if not pdfs:
            if not self.config.quiet:
                console.print("[yellow]No PDF files found[/yellow]")
            return []

        to_process = []
        for pdf in pdfs:
            if meta.is_processed(pdf) and not self.config.reprocess:
                if self.config.verbose:
                    console.print(f"[dim]Skipping: {pdf.name}[/dim]")
            else:
                to_process.append(pdf)

        if not to_process:
            if not self.config.quiet:
                console.print("[green]All files already processed[/green]")
                console.print("[dim]Use --reprocess to force reprocessing[/dim]")
            return []

        if self.config.dry_run:
            if not self.config.quiet:
                console.print(f"[blue]Would process {len(to_process)} file(s):[/blue]")
                for pdf in to_process:
                    size_mb = pdf.stat().st_size / (1024 * 1024)
                    console.print(f"  {pdf.name} ({size_mb:.1f} MB)")
            return []

        if not self.config.quiet:
            console.print(f"[blue]Processing {len(to_process)} file(s)...[/blue]")
            console.print(f"[blue]Output:[/blue] {out_dir}\n")

        results: list[EngineResult] = []
        start = time.time()

        for pdf in to_process:
            result = self.process(pdf, out_dir)
            results.append(result)
            if result.success:
                meta.record(
                    pdf,
                    engine=result.engine,
                    processing_time=result.processing_time,
                    pages=result.pages_processed,
                )

        if not self.config.quiet:
            ok = sum(1 for r in results if r.success)
            console.print(f"\n[green]Completed:[/green] {ok}/{len(to_process)} files")
            console.print(f"[dim]Total time: {time.time() - start:.1f}s[/dim]")

        return results

    # ------------------------------------------------------------------
    # Phase 1: Analyze
    # ------------------------------------------------------------------

    def _phase_analyze(self, state: DocumentState) -> None:
        """Detect born-digital pages and apply to state."""
        if not self.config.quiet:
            console.print("\n[cyan]Phase 1:[/cyan] Analyze (born-digital detection)")

        assessment = self.bd_detector.detect(state.handle.path)
        state.apply_born_digital(assessment)

        bd_count = assessment.born_digital_count
        if not self.config.quiet:
            if bd_count:
                console.print(
                    f"  {bd_count}/{assessment.page_count} pages born-digital "
                    f"(will skip OCR)"
                )
            else:
                console.print("  No born-digital pages detected")

    # ------------------------------------------------------------------
    # Phase 2: Backbone OCR
    # ------------------------------------------------------------------

    def _phase_backbone(
        self, state: DocumentState, output_dir: Path
    ) -> EngineResult | None:
        """Run the primary engine on the document.

        For per-page HTTP engines (``GEMINI_API``, ``DEEPSEEK_VLLM``), render
        each page as an image and process independently via the HTTP API.

        For CLI engines, if the document exceeds ``config.chunk_threshold``
        pages, split it into chunks and process each chunk independently via
        :meth:`_backbone_chunked`.
        """
        # Per-page HTTP engines — bypass CLI entirely
        if self.config.primary_engine == EngineType.GEMINI_API:
            return self._backbone_per_page(state, output_dir)

        engine = get_engine(self.config.primary_engine)

        if not self.config.quiet:
            console.print(f"\n[cyan]Phase 2:[/cyan] Backbone OCR [{engine.name}]")

        if not engine.is_available():
            logger.warning(f"Primary engine {engine.name} not available")
            if not self.config.quiet:
                console.print(f"[red]Engine {engine.name} not available[/red]")
            err_result = EngineResult(
                document_path=state.handle.path,
                engine=engine.name,
                status=DocumentStatus.ERROR,
                error=(
                    f"Engine {engine.name} not available "
                    f"(CLI not installed or missing API key)"
                ),
            )
            state.apply_result(err_result)
            return err_result

        # Chunk long documents to avoid context-window / timeout issues
        if state.handle.page_count > self.config.chunk_threshold:
            if not self.config.quiet:
                console.print(
                    f"  [dim]{state.handle.page_count} pages > "
                    f"chunk threshold {self.config.chunk_threshold}, "
                    f"splitting into chunks of {self.config.chunk_size}[/dim]"
                )
            return self._backbone_chunked(state, output_dir, engine)

        result = engine.process_document(state.handle.path, output_dir, self.config)
        result.pages_processed = state.handle.page_count
        state.apply_result(result)
        return result

    def _backbone_chunked(
        self,
        state: DocumentState,
        output_dir: Path,
        engine: BaseEngine,
    ) -> EngineResult:
        """Run the engine on each chunk and concatenate the results.

        Splits the PDF via :class:`PDFChunker`, runs the engine on each
        chunk, and combines the per-chunk texts into a single
        ``PageOutput(page_num=0)`` whole-doc result.
        """
        chunker = PDFChunker(max_pages_per_chunk=self.config.chunk_size)
        start_time = time.time()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            chunks = chunker.chunk(state.handle.path, tmp_path / "chunks")

            if not self.config.quiet:
                console.print(f"  Split into {len(chunks)} chunks")

            chunk_texts: list[str] = []
            total_cost = 0.0

            for chunk in chunks:
                if not self.config.quiet:
                    console.print(
                        f"  Chunk {chunk.chunk_num}/{len(chunks)} "
                        f"(pages {chunk.start_page}-{chunk.end_page})"
                    )

                chunk_result = engine.process_document(
                    chunk.path, tmp_path / "out", self.config,
                )

                if chunk_result.success and chunk_result.pages:
                    # CLI engines produce page_num=0 whole-doc output
                    text = chunk_result.markdown
                    if text:
                        chunk_texts.append(text)
                else:
                    logger.warning(
                        f"Chunk {chunk.chunk_num} failed: "
                        f"{chunk_result.error or chunk_result.status.value}"
                    )

                total_cost += chunk_result.cost

        elapsed = time.time() - start_time
        combined_text = "\n\n".join(chunk_texts)

        if combined_text.strip():
            status = DocumentStatus.SUCCESS
            page_status = PageStatus.SUCCESS
        else:
            status = DocumentStatus.ERROR
            page_status = PageStatus.ERROR

        result = EngineResult(
            document_path=state.handle.path,
            engine=engine.name,
            status=status,
            pages=[
                PageOutput(
                    page_num=0,
                    text=combined_text,
                    status=page_status,
                    engine=engine.name,
                    processing_time=elapsed,
                )
            ],
            pages_processed=state.handle.page_count,
            processing_time=elapsed,
            cost=total_cost,
        )
        state.apply_result(result)
        return result

    def _backbone_per_page(
        self,
        state: DocumentState,
        output_dir: Path,
    ) -> EngineResult:
        """Run a per-page HTTP engine on every page individually.

        Renders each page to an image, sends it to the engine's
        ``process_image`` method, and assembles the per-page PageOutputs
        into a single EngineResult.
        """
        from socr.engines.gemini_api import GeminiAPIConfig, GeminiAPIEngine

        engine = GeminiAPIEngine(
            GeminiAPIConfig(model=self.config.gemini_model)
        )

        if not self.config.quiet:
            console.print(
                f"\n[cyan]Phase 2:[/cyan] Backbone OCR [{engine.name}] "
                f"(per-page)"
            )

        if not engine.is_available():
            logger.warning(f"Engine {engine.name} not available")
            if not self.config.quiet:
                console.print(f"[red]Engine {engine.name} not available[/red]")
            err_result = EngineResult(
                document_path=state.handle.path,
                engine=engine.name,
                status=DocumentStatus.ERROR,
                error=f"Engine {engine.name} not available (missing API key)",
            )
            state.apply_result(err_result)
            return err_result

        start_time = time.time()
        page_outputs: list[PageOutput] = []
        total_pages = state.handle.page_count

        for page_num in range(1, total_pages + 1):
            if not self.config.quiet:
                console.print(
                    f"  Page {page_num}/{total_pages}...", end=" "
                )

            image = state.handle.render_page(page_num)
            page_result = engine.process_image(image, page_num=page_num)
            page_outputs.append(page_result)

            if not self.config.quiet:
                if page_result.status == PageStatus.SUCCESS:
                    console.print(
                        f"[green]{page_result.word_count} words[/green]"
                    )
                else:
                    console.print(
                        f"[red]{page_result.failure_mode.value}[/red]"
                    )

        elapsed = time.time() - start_time
        engine.close()

        success_count = sum(
            1 for p in page_outputs if p.status == PageStatus.SUCCESS
        )
        overall_status = (
            DocumentStatus.SUCCESS if success_count > 0
            else DocumentStatus.ERROR
        )

        result = EngineResult(
            document_path=state.handle.path,
            engine=engine.name,
            status=overall_status,
            pages=page_outputs,
            pages_processed=total_pages,
            processing_time=elapsed,
            model_version=engine.model_version,
        )
        state.apply_result(result)
        return result

    # ------------------------------------------------------------------
    # Phase 3: Score
    # ------------------------------------------------------------------

    def _phase_score(
        self, state: DocumentState, backbone_result: EngineResult
    ) -> None:
        """Run quality scoring on engine outputs.

        For CLI engines that produce page_num=0 (whole-doc), score the
        combined text and propagate the result to the whole-doc PageOutput.
        For per-page outputs, score each page individually.
        """
        if not self.config.quiet:
            console.print("\n[cyan]Phase 3:[/cyan] Score (quality audit)")

        has_whole_doc = any(p.page_num == 0 for p in backbone_result.pages)

        if has_whole_doc:
            self._score_whole_doc(state, backbone_result)
        else:
            self._score_per_page(state)

    def _score_whole_doc(
        self, state: DocumentState, result: EngineResult
    ) -> None:
        """Score a whole-document output (CLI engine, page_num=0)."""
        whole_doc_page = next(
            (p for p in result.pages if p.page_num == 0), None
        )
        if not whole_doc_page:
            return

        scoring = self.scorer.score(
            whole_doc_page.text, engine=result.engine,
            expected_pages=state.handle.page_count,
        )

        if scoring.passed:
            whole_doc_page.audit_passed = True
            whole_doc_page.failure_mode = FailureMode.NONE
            result.audit_passed = True
            if not self.config.quiet:
                console.print("  [green]Passed[/green]")
        else:
            whole_doc_page.audit_passed = False
            whole_doc_page.failure_mode = scoring.primary_failure
            result.audit_passed = False
            result.status = DocumentStatus.AUDIT_FAILED
            result.failure_mode = scoring.primary_failure
            if not self.config.quiet:
                console.print(
                    f"  [red]FAIL:[/red] {scoring.primary_failure.value}"
                )
                for mode, detail in scoring.details.items():
                    console.print(f"    {detail}")

    def _score_per_page(self, state: DocumentState) -> None:
        """Score each page's best output individually."""
        failures = 0
        for page_num in sorted(state.pages):
            page_state = state.pages[page_num]
            if page_state.is_born_digital and page_state.native_text:
                continue
            if not page_state.attempts:
                continue

            # Score the most recent attempt
            latest = page_state.attempts[-1]
            scoring = self.scorer.score(latest.text, engine=latest.engine)

            latest.audit_passed = scoring.passed
            if not scoring.passed:
                latest.failure_mode = scoring.primary_failure
                failures += 1
                # If this was the best_output but now fails, clear it
                if page_state.best_output is latest:
                    page_state.best_output = None
            else:
                latest.failure_mode = FailureMode.NONE
                # Promote to best if none set
                if not page_state.best_output:
                    page_state.best_output = latest

        if not self.config.quiet:
            if failures:
                console.print(f"  {failures} page(s) failed audit")
            else:
                console.print("  [green]All pages passed[/green]")

    # ------------------------------------------------------------------
    # Phase 4: Selective Repair
    # ------------------------------------------------------------------

    def _phase_repair(self, state: DocumentState, output_dir: Path) -> None:
        """Repair loop: plan repairs, execute, re-score, repeat.

        Loops up to ``config.max_retries`` times. Each iteration:
          1. Ask RepairRouter for a plan.
          2. For each engine group in the plan, run the engine.
          3. Apply results and re-score.
          4. Stop if no pages need repair or plan is empty.
        """
        # If a CLI engine produced a passing whole-doc output, per-page
        # states won't have best_outputs but the document is covered.
        # Skip repair entirely in that case.
        has_passing_whole_doc = any(
            w.audit_passed for w in state.whole_doc_attempts
        )
        # Also check if there's a failing whole-doc attempt that needs
        # document-level retry (e.g. truncated output).
        has_failing_whole_doc = any(
            not w.audit_passed for w in state.whole_doc_attempts
        )
        needs_whole_doc_retry = (
            has_failing_whole_doc and not has_passing_whole_doc
        )

        if has_passing_whole_doc and not state.pages_needing_repair:
            if not self.config.quiet:
                console.print(
                    "\n[cyan]Phase 4:[/cyan] Repair (not needed)"
                )
            return

        # Retry-on-truncation: if the latest whole-doc attempt failed
        # specifically with TRUNCATED, retry the same engine before
        # falling through to the fallback chain.  Gemini's truncation
        # is non-deterministic, so a simple retry often succeeds.
        if (
            needs_whole_doc_retry
            and self.config.truncation_retries > 0
            and state.whole_doc_attempts
        ):
            latest_whole = state.whole_doc_attempts[-1]
            if (
                not latest_whole.audit_passed
                and latest_whole.failure_mode == FailureMode.TRUNCATED
            ):
                # Identify which engine produced the truncated output
                truncated_engine_name = latest_whole.engine
                truncated_engine_type = None
                for et in EngineType:
                    if et.value == truncated_engine_name:
                        truncated_engine_type = et
                        break

                if truncated_engine_type is not None:
                    for retry_idx in range(self.config.truncation_retries):
                        if not self.config.quiet:
                            console.print(
                                f"\n[cyan]Phase 4:[/cyan] Repair "
                                f"(truncation retry {retry_idx + 1}/"
                                f"{self.config.truncation_retries}) "
                                f"[{truncated_engine_name}]"
                            )
                        engine = get_engine(truncated_engine_type)
                        if not engine.is_available():
                            break
                        retry_result = engine.process_document(
                            state.handle.path, output_dir, self.config
                        )
                        retry_result.pages_processed = (
                            state.handle.page_count
                        )
                        state.apply_result(retry_result)
                        if retry_result.success:
                            self._score_repair_result(
                                state, retry_result, []
                            )
                        # Check if the retry passed
                        if any(
                            w.audit_passed
                            for w in state.whole_doc_attempts
                        ):
                            needs_whole_doc_retry = False
                            has_passing_whole_doc = True
                            break

                    # If truncation retry resolved it, we're done
                    if not needs_whole_doc_retry:
                        if not self.config.quiet:
                            console.print(
                                "  [green]Truncation retry "
                                "succeeded[/green]"
                            )
                        return

        for attempt in range(self.config.max_retries):
            plan = self.repair_router.plan_repairs(state)

            # If per-page plan is empty but whole-doc retry is needed,
            # try the next engine in the fallback chain on the whole doc.
            if plan.is_empty and needs_whole_doc_retry:
                tried = {r.engine for r in state.engine_runs}
                next_engine = None
                for et in self.config.fallback_chain:
                    if et.value not in tried:
                        next_engine = et
                        break
                if next_engine:
                    if not self.config.quiet:
                        console.print(
                            f"\n[cyan]Phase 4:[/cyan] Repair "
                            f"(attempt {attempt + 1}/{self.config.max_retries}) "
                            f"[{next_engine.value}] (whole-doc retry)"
                        )
                    engine = get_engine(next_engine)
                    if engine.is_available():
                        repair_result = engine.process_document(
                            state.handle.path, output_dir, self.config
                        )
                        repair_result.pages_processed = state.handle.page_count
                        state.apply_result(repair_result)
                        if repair_result.success:
                            self._score_repair_result(
                                state, repair_result, []
                            )
                            # Check if the new attempt passed
                            if any(
                                w.audit_passed
                                for w in state.whole_doc_attempts
                            ):
                                needs_whole_doc_retry = False
                                break
                    continue

            if plan.is_empty:
                if not self.config.quiet and attempt == 0:
                    if state.pages_needing_repair:
                        console.print(
                            "\n[cyan]Phase 4:[/cyan] Repair "
                            "(all engines exhausted, skipping)"
                        )
                    else:
                        console.print(
                            "\n[cyan]Phase 4:[/cyan] Repair (not needed)"
                        )
                break

            if not self.config.quiet:
                engines_str = ", ".join(
                    e.value for e in plan.by_engine.keys()
                )
                console.print(
                    f"\n[cyan]Phase 4:[/cyan] Repair "
                    f"(attempt {attempt + 1}/{self.config.max_retries}) "
                    f"[{engines_str}]"
                )
                console.print(
                    f"  {len(plan.repairs)} page(s) to repair"
                )
                if plan.pages_skipped:
                    console.print(
                        f"  {len(plan.pages_skipped)} page(s) skipped "
                        f"(engines exhausted)"
                    )

            # Execute repairs grouped by engine
            for engine_type, repairs in plan.by_engine.items():
                engine = get_engine(engine_type)

                if not engine.is_available():
                    if not self.config.quiet:
                        console.print(
                            f"  [yellow]{engine.name} not available, "
                            f"skipping[/yellow]"
                        )
                    continue

                # CLI engines process the whole document; the orchestrator
                # picks per-page improvements from the result.
                repair_result = engine.process_document(
                    state.handle.path, output_dir, self.config
                )
                repair_result.pages_processed = state.handle.page_count
                state.apply_result(repair_result)

                # Score the repair result
                if repair_result.success:
                    self._score_repair_result(state, repair_result, repairs)

            # If nothing left to repair, stop early
            if not state.pages_needing_repair:
                break

    def _score_repair_result(
        self,
        state: DocumentState,
        result: EngineResult,
        repairs: list,
    ) -> None:
        """Score a repair engine's output.

        For CLI engines (whole-doc, page_num=0): score the whole text and
        update the corresponding whole_doc_attempt.  For per-page outputs,
        score each relevant page.
        """
        has_whole_doc = any(p.page_num == 0 for p in result.pages)

        if has_whole_doc:
            whole_page = next(p for p in result.pages if p.page_num == 0)
            scoring = self.scorer.score(
                whole_page.text, engine=result.engine,
                expected_pages=state.handle.page_count,
            )
            whole_page.audit_passed = scoring.passed
            if not scoring.passed:
                whole_page.failure_mode = scoring.primary_failure
            else:
                whole_page.failure_mode = FailureMode.NONE
        else:
            repair_page_nums = {r.page_num for r in repairs}
            for page_out in result.pages:
                if page_out.page_num not in repair_page_nums:
                    continue
                scoring = self.scorer.score(
                    page_out.text, engine=result.engine
                )
                page_out.audit_passed = scoring.passed
                if not scoring.passed:
                    page_out.failure_mode = scoring.primary_failure
                else:
                    page_out.failure_mode = FailureMode.NONE

    # ------------------------------------------------------------------
    # Phase 4b: Consensus
    # ------------------------------------------------------------------

    def _phase_consensus(self, state: DocumentState) -> None:
        """Run multi-engine consensus on pages with multiple attempts.

        Only runs when ``config.consensus_enabled`` is True.  For each
        page with >1 attempt, selects (or merges) the best output and
        updates ``page_state.best_output``.
        """
        pages_with_multi = [
            pn
            for pn in sorted(state.pages)
            if len(state.pages[pn].attempts) >= 2
            and not (state.pages[pn].is_born_digital and state.pages[pn].native_text)
        ]

        if not pages_with_multi:
            if not self.config.quiet:
                console.print(
                    "\n[cyan]Phase 4b:[/cyan] Consensus (not needed — "
                    "no multi-attempt pages)"
                )
            return

        if not self.config.quiet:
            console.print(
                f"\n[cyan]Phase 4b:[/cyan] Consensus "
                f"({len(pages_with_multi)} page(s) with multiple attempts)"
            )

        engine = ConsensusEngine(
            use_llm=self.config.consensus_use_llm,
            ollama_model=self.config.consensus_ollama_model,
        )
        results = engine.reconcile_document(state)

        if not self.config.quiet:
            for cr in results:
                disc_str = f" [{len(cr.discrepancies)} discrepancies]" if cr.discrepancies else ""
                console.print(
                    f"  Page {cr.page_num}: selected {cr.selected_engine} "
                    f"(agreement={cr.agreement_score:.2f}){disc_str}"
                )

    # ------------------------------------------------------------------
    # Phase 5: Assemble
    # ------------------------------------------------------------------

    def _phase_assemble(
        self, state: DocumentState, output_dir: Path
    ) -> EngineResult:
        """Build the final EngineResult from DocumentState and save to disk."""
        if not self.config.quiet:
            console.print("\n[cyan]Phase 5:[/cyan] Assemble")

        final_text = state.text
        has_text = bool(final_text.strip())

        # Determine overall status.
        # For CLI engines that produce whole-doc output (page_num=0), pages
        # won't have per-page best_outputs.  A passing whole-doc attempt
        # covers the entire document -- treat it as success.
        has_passing_whole_doc = any(
            w.audit_passed for w in state.whole_doc_attempts
        )
        pages_ok = (
            not state.pages_needing_repair or has_passing_whole_doc
        )

        if has_text and pages_ok:
            status = DocumentStatus.SUCCESS
        elif has_text:
            status = DocumentStatus.AUDIT_FAILED
        else:
            status = DocumentStatus.ERROR

        state.status = status

        # Compute total processing time
        total_time = sum(r.processing_time for r in state.engine_runs)

        # Build the final result
        final_result = EngineResult(
            document_path=state.handle.path,
            engine=", ".join(state.engines_used) if state.engines_used else "none",
            status=status,
            pages=[
                PageOutput(
                    page_num=0,
                    text=final_text,
                    status=PageStatus.SUCCESS if has_text else PageStatus.ERROR,
                    engine=", ".join(state.engines_used),
                )
            ],
            pages_processed=state.handle.page_count,
            processing_time=total_time,
            cost=state.total_cost,
            audit_passed=status == DocumentStatus.SUCCESS,
        )

        # Figure extraction
        if self.config.save_figures and has_text:
            self._extract_figures(state, final_result, output_dir)

        # Save markdown
        if has_text:
            saved_path = self._save_markdown(state, final_text, output_dir)
            if not self.config.quiet:
                console.print(f"  [blue]Output:[/blue] {saved_path}")

        return final_result

    def _extract_figures(
        self,
        state: DocumentState,
        result: EngineResult,
        output_dir: Path,
    ) -> None:
        """Extract figures from the PDF."""
        if not self.config.quiet:
            console.print("  Extracting figures...")

        stem = sanitize_filename(state.handle.stem)
        figures_dir = output_dir / stem / "figures"
        extractor = FigureExtractor(
            max_total=self.config.figures_max_total,
            max_per_page=self.config.figures_max_per_page,
            save_dir=figures_dir,
        )
        extracted = extractor.extract(state.handle.path)

        if not self.config.quiet:
            if extracted:
                console.print(
                    f"  Extracted {len(extracted)} figures to {figures_dir}"
                )
            else:
                console.print("  [dim]No figures detected[/dim]")

        result.figures = [
            FigureInfo(
                figure_num=fig.figure_num,
                page_num=fig.page_num,
                figure_type="extracted",
                image_path=fig.saved_path,
            )
            for fig in extracted
        ]

    def _save_markdown(
        self, state: DocumentState, text: str, output_dir: Path
    ) -> Path:
        """Save the assembled markdown to output_dir/{stem}/{stem}.md."""
        stem = sanitize_filename(state.handle.stem)
        doc_dir = output_dir / stem
        doc_dir.mkdir(parents=True, exist_ok=True)
        md_path = doc_dir / f"{stem}.md"
        md_path.write_text(text, encoding="utf-8")
        return md_path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _print_summary(
        self, result: EngineResult, state: DocumentState
    ) -> None:
        """Print a final summary line."""
        if result.success:
            status_str = "[green]Success[/green]"
        else:
            status_str = f"[red]{result.status.value}[/red]"

        console.print(
            f"\n{status_str} | {result.engine} | "
            f"{result.processing_time:.1f}s"
        )
        if state.pages_needing_repair:
            console.print(
                f"[yellow]{len(state.pages_needing_repair)} page(s) "
                f"still failing[/yellow]"
            )
        if result.error:
            console.print(f"[dim]{result.error}[/dim]")
