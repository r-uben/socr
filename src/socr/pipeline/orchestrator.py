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

import fitz
from rich.console import Console

from socr.audit.heuristics import HeuristicsChecker
from socr.audit.scorer import FailureModeScorer
from socr.core.born_digital import BornDigitalDetector, DocumentAssessment
from socr.core.chunker import PDFChunker
from socr.core.config import EngineType, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.metadata import MetadataManager
from socr.core.normalizer import OutputNormalizer
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
from socr.engines.registry import get_engine, resolve_auto_engine
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
        self._last_assessment: DocumentAssessment | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, pdf_path: Path, output_dir: Path | None = None) -> EngineResult:
        """Process a single PDF through the 5-phase loop.

        Returns an EngineResult summarising the best extraction.
        """
        pdf_path = Path(pdf_path)
        out_dir = output_dir or self.config.output_dir

        # Resolve AUTO engine before starting
        if self.config.primary_engine == EngineType.AUTO:
            self.config.primary_engine = resolve_auto_engine()
            if not self.config.quiet:
                console.print(
                    f"[dim]Auto-selected engine: {self.config.primary_engine.value}[/dim]"
                )

        doc = DocumentHandle.from_path(pdf_path)
        state = DocumentState(handle=doc)

        if not self.config.quiet:
            console.print(f"[blue]Processing:[/blue] {doc.filename}")
            console.print(f"[dim]{doc.page_count} pages, {doc.size_mb:.1f} MB[/dim]")

        is_multi = bool(self.config.multi_engine)

        # Phase 1: Analyze
        self._phase_analyze(state)

        if self.config.agentic and not is_multi:
            # Cost-aware agentic routing replaces backbone + score + repair:
            # per page, try the cheapest provider and let the judge escalate.
            self._phase_agentic(state, out_dir)
        elif is_multi:
            # Multi-engine mode: run all engines, score all, consensus
            backbone_results = self._backbone_multi_engine(state, out_dir)

            # Phase 3: Score all engine outputs
            if self.config.audit_enabled:
                self._phase_score_multi(state, backbone_results)

            # Phase 4: Repair — skip (multiple engines already provide coverage)
            if not self.config.quiet:
                console.print(
                    "\n[cyan]Phase 4:[/cyan] Repair "
                    "(skipped — multi-engine mode)"
                )

            # Phase 4b: Consensus — always run in multi-engine mode
            self._phase_consensus(state)
        else:
            # Single-engine mode: original flow
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
        self._last_assessment = assessment
        state.apply_born_digital(assessment)

        bd_count = assessment.born_digital_count
        if not self.config.quiet:
            if bd_count:
                # Count pages needing enhancement vs pure prose
                enhancement_count = sum(
                    1 for pa in assessment.pages
                    if pa.is_born_digital and pa.needs_ocr_enhancement
                )
                prose_count = bd_count - enhancement_count
                scanned_count = assessment.scanned_count
                console.print(
                    f"  {bd_count}/{assessment.page_count} pages born-digital"
                )
                if self.config.native_first and (prose_count or enhancement_count):
                    if prose_count:
                        console.print(
                            f"    {prose_count} prose-only (native text)"
                        )
                    if enhancement_count:
                        console.print(
                            f"    {enhancement_count} complex "
                            f"(tables/figures/equations)"
                        )
                    if scanned_count:
                        console.print(
                            f"    {scanned_count} scanned (no text layer)"
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

        When ``native_first`` is enabled and the document is mostly
        born-digital, uses native text for prose pages and sends only
        complex/scanned pages to a CLI engine via a temp PDF.

        For CLI engines, if the document exceeds ``config.chunk_threshold``
        pages, split it into chunks and process each chunk independently via
        :meth:`_backbone_chunked`.
        """
        # Native-first: use native text for born-digital prose, CLI only
        # for complex/scanned pages.
        if self.config.native_first:
            bd_pages = [
                p for p in state.pages.values() if p.is_born_digital
            ]
            bd_ratio = len(bd_pages) / max(len(state.pages), 1)
            if bd_ratio >= 0.5:
                return self._backbone_native_first(state, output_dir)

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

        # Per-page processing: render all pages to images → CLI
        all_pages = list(range(1, state.handle.page_count + 1))
        if not self.config.quiet:
            console.print(
                f"  Processing {len(all_pages)} pages (per-page)..."
            )

        start_time = time.time()
        page_outputs = engine.process_pages(
            pdf_path=state.handle.path,
            page_nums=all_pages,
            config=self.config,
            dpi=self.config.render_dpi,
        )
        elapsed = time.time() - start_time

        success_count = sum(
            1 for p in page_outputs if p.status == PageStatus.SUCCESS
        )
        overall_status = (
            DocumentStatus.SUCCESS if success_count > 0
            else DocumentStatus.ERROR
        )

        if not self.config.quiet:
            console.print(f"  {success_count}/{len(all_pages)} pages succeeded")

        result = EngineResult(
            document_path=state.handle.path,
            engine=engine.name,
            status=overall_status,
            pages=page_outputs,
            pages_processed=state.handle.page_count,
            processing_time=elapsed,
            model_version=engine.model_version,
        )
        state.apply_result(result)
        return result

    def _backbone_native_first(
        self, state: DocumentState, output_dir: Path
    ) -> EngineResult:
        """3-tier routing: native → local → cloud.

        Tier 1: Born-digital prose → native text (free, instant)
        Tier 2: Easy scanned pages → local engine (free, fast)
        Tier 3: Hard pages (tables, multi-column, degraded) → primary engine (cloud)

        When tiered=False or no local engine is available, tiers 2+3 collapse
        into a single pass using the primary engine (same as before).
        """
        from socr.core.difficulty import PageDifficulty, classify_pages
        from socr.engines.registry import resolve_local_engine

        # Classify pages
        prose_pages: list[int] = []
        enhancement_pages: list[int] = []
        scanned_pages: list[int] = []

        for page_num, ps in sorted(state.pages.items()):
            if ps.is_born_digital and not ps.needs_ocr_enhancement and ps.native_text:
                prose_pages.append(page_num)
            elif ps.is_born_digital and ps.needs_ocr_enhancement:
                enhancement_pages.append(page_num)
            else:
                scanned_pages.append(page_num)

        total = len(state.pages)
        ocr_pages = enhancement_pages + scanned_pages

        # Tier 2/3 split: classify difficulty of OCR pages
        easy_pages: list[int] = []
        hard_pages: list[int] = []

        # Resolve local engine for tiered routing
        local_engine_type = None
        if self.config.tiered and ocr_pages:
            if self.config.local_engine == EngineType.AUTO:
                local_engine_type = resolve_local_engine()
            elif self.config.local_engine != self.config.primary_engine:
                local_engine_type = self.config.local_engine

        if local_engine_type and ocr_pages:
            # Build hints from born-digital assessment
            page_hints: dict[int, dict] = {}
            for page_num in ocr_pages:
                ps = state.pages[page_num]
                bd_assessment = next(
                    (pa for pa in (self._last_assessment or DocumentAssessment(path=state.handle.path, pages=[])).pages
                     if pa.page_num == page_num),
                    None,
                )
                if bd_assessment:
                    page_hints[page_num] = {
                        "has_tables": bd_assessment.has_tables,
                        "has_equations": bd_assessment.has_equations,
                    }
                elif ps.needs_ocr_enhancement:
                    # Fallback: if needs enhancement, assume hard
                    page_hints[page_num] = {
                        "has_tables": True,
                        "has_equations": False,
                    }

            # Classify page difficulty with hints
            difficulty_map = classify_pages(
                str(state.handle.path), ocr_pages,
                page_hints=page_hints,
            )
            for page_num in ocr_pages:
                da = difficulty_map.get(page_num)
                if da and da.difficulty == PageDifficulty.EASY:
                    easy_pages.append(page_num)
                else:
                    hard_pages.append(page_num)
        else:
            # No tiered routing — all OCR pages go to primary
            hard_pages = ocr_pages

        if not self.config.quiet:
            label = "native-first" if not local_engine_type else "tiered"
            console.print(f"\n[cyan]Phase 2:[/cyan] Text extraction ({label})")
            if prose_pages:
                console.print(
                    f"  {len(prose_pages)}/{total} pages: "
                    "native text (born-digital prose)"
                )
            if easy_pages:
                console.print(
                    f"  {len(easy_pages)}/{total} pages: "
                    f"local OCR [{local_engine_type.value}] (easy)"
                )
            if hard_pages:
                console.print(
                    f"  {len(hard_pages)}/{total} pages: "
                    f"cloud OCR [{self.config.primary_engine.value}] (hard)"
                )
            if not ocr_pages:
                console.print("  All pages born-digital")

        start_time = time.time()
        page_outputs: list[PageOutput] = []

        # Tier 1: Native text for prose pages
        for page_num in prose_pages:
            ps = state.pages[page_num]
            page_outputs.append(PageOutput(
                page_num=page_num,
                text=ps.native_text,
                status=PageStatus.SUCCESS,
                engine="native",
                audit_passed=True,
            ))

        # Tier 2: Local engine for easy pages
        escalated_pages: list[int] = []
        escalated_reasons: dict[int, str] = {}
        local_engine_name = ""
        if easy_pages and local_engine_type:
            local_outputs = self._run_engine_on_pages(
                state, easy_pages, enhancement_pages,
                local_engine_type, "local",
            )

            # Per-page quality scoring on local outputs → auto-escalate failures
            passed_outputs: list[PageOutput] = []
            local_engine_name = get_engine(local_engine_type).name
            for po in local_outputs:
                # Native passthrough — born-digital prose, keep as-is
                if po.engine == "native":
                    passed_outputs.append(po)
                    continue
                # Engine error — escalate rather than ship a blank page
                if po.status != PageStatus.SUCCESS:
                    escalated_pages.append(po.page_num)
                    escalated_reasons[po.page_num] = "engine_error"
                    logger.info(
                        "Page %d errored on local engine — escalating to cloud",
                        po.page_num,
                    )
                    continue
                scoring = self.scorer.score(po.text, engine=po.engine)
                if scoring.passed:
                    po.audit_passed = True
                    passed_outputs.append(po)
                else:
                    # Quality failure — escalate to cloud
                    escalated_pages.append(po.page_num)
                    escalated_reasons[po.page_num] = scoring.primary_failure.value
                    logger.info(
                        "Page %d failed local audit (%s) — escalating to cloud",
                        po.page_num, scoring.primary_failure.value,
                    )

            page_outputs.extend(passed_outputs)

            if escalated_pages and not self.config.quiet:
                console.print(
                    f"  [yellow]{len(escalated_pages)} page(s) failed "
                    f"local audit → escalating to cloud[/yellow]"
                )
                for pn in escalated_pages:
                    console.print(f"    p{pn}: {escalated_reasons[pn]}")

        # Tier 3: Primary (cloud) engine for hard pages + escalated pages
        cloud_pages = hard_pages + escalated_pages
        if cloud_pages:
            cloud_outputs = self._run_engine_on_pages(
                state, cloud_pages, enhancement_pages,
                self.config.primary_engine, "cloud",
            )
            # Tag escalated pages so metadata tracks the promotion
            for co in cloud_outputs:
                if co.page_num in escalated_pages:
                    co.escalated_from = local_engine_name
            page_outputs.extend(cloud_outputs)

        elapsed = time.time() - start_time

        success_count = sum(
            1 for p in page_outputs if p.status == PageStatus.SUCCESS
        )
        overall_status = (
            DocumentStatus.SUCCESS if success_count > 0
            else DocumentStatus.ERROR
        )

        engines_used = set()
        for p in page_outputs:
            if p.engine and p.engine != "native":
                engines_used.add(p.engine)
        if engines_used:
            engine_name = "native+" + "+".join(sorted(engines_used))
        else:
            engine_name = "native"

        result = EngineResult(
            document_path=state.handle.path,
            engine=engine_name,
            status=overall_status,
            pages=page_outputs,
            pages_processed=total,
            processing_time=elapsed,
        )
        state.apply_result(result)
        return result

    def _run_engine_on_pages(
        self,
        state: DocumentState,
        page_nums: list[int],
        enhancement_pages: list[int],
        engine_type: EngineType,
        label: str,
    ) -> list[PageOutput]:
        """Render pages to images and run a CLI engine per-page.

        Each page is rendered to a PNG, the CLI processes the image directory,
        and we get back one PageOutput per page with real text. No more
        page_num=0 whole-doc hack.

        Args:
            state: Document state.
            page_nums: 1-indexed page numbers to process.
            enhancement_pages: Pages that have native text fallback.
            engine_type: Which engine to use.
            label: Label for log messages ("local" or "cloud").

        Returns:
            List of PageOutput, one per page_num, with per-page text.
        """
        engine = get_engine(engine_type)

        if not engine.is_available():
            logger.warning(f"{engine.name} not available for {label} OCR")
            if not self.config.quiet:
                console.print(
                    f"  [yellow]{engine.name} not available -- "
                    "using native text as fallback[/yellow]"
                )
            outputs: list[PageOutput] = []
            for page_num in page_nums:
                ps = state.pages[page_num]
                if page_num in enhancement_pages and ps.native_text:
                    outputs.append(PageOutput(
                        page_num=page_num,
                        text=ps.native_text,
                        status=PageStatus.SUCCESS,
                        engine="native",
                        audit_passed=True,
                    ))
                else:
                    outputs.append(PageOutput(
                        page_num=page_num,
                        text="",
                        status=PageStatus.ERROR,
                        engine=engine.name,
                        failure_mode=FailureMode.MODEL_UNAVAILABLE,
                    ))
            return outputs

        if not self.config.quiet:
            console.print(
                f"  Running {engine.name} on "
                f"{len(page_nums)} {label} pages (per-page)..."
            )

        # Render pages to images → CLI processes images → per-page results
        page_outputs = engine.process_pages(
            pdf_path=state.handle.path,
            page_nums=page_nums,
            config=self.config,
            dpi=self.config.render_dpi,
        )

        # For enhancement pages where OCR failed, fall back to native text
        final: list[PageOutput] = []
        for po in page_outputs:
            if (
                po.status != PageStatus.SUCCESS
                and po.page_num in enhancement_pages
            ):
                ps = state.pages[po.page_num]
                if ps.native_text:
                    final.append(PageOutput(
                        page_num=po.page_num,
                        text=ps.native_text,
                        status=PageStatus.SUCCESS,
                        engine="native",
                        audit_passed=True,
                    ))
                    continue
            final.append(po)

        if not self.config.quiet:
            ok = sum(1 for p in final if p.status == PageStatus.SUCCESS)
            console.print(f"  {ok}/{len(page_nums)} pages succeeded")

        return final

    # ------------------------------------------------------------------
    # Agentic: cost-aware per-page routing (replaces backbone+score+repair)
    # ------------------------------------------------------------------

    def _phase_agentic(self, state: DocumentState, output_dir: Path) -> None:
        """Per OCR page: try the cheapest provider, let the judge escalate.

        Born-digital prose still takes free native text (Tier 1). Every other
        page is routed through the cost-ordered ladder; the winning provider and
        its cost are recorded on PageState. Bounded by max_retries and an
        optional cost_budget.
        """
        from socr.core.providers import provider_ladder
        from socr.pipeline.agentic import route_page

        if not self.config.quiet:
            console.print("\n[cyan]Agentic routing[/cyan] (cost-ordered, judge-gated)")

        # Tier 1: born-digital prose -> native text (free, no OCR).
        ocr_pages: list[int] = []
        for page_num, ps in sorted(state.pages.items()):
            if ps.is_born_digital and not ps.needs_ocr_enhancement and ps.native_text:
                native = PageOutput(
                    page_num=page_num, text=ps.native_text, status=PageStatus.SUCCESS,
                    engine="native", audit_passed=True, cost_usd=0.0,
                )
                ps.attempts.append(native)
                ps.best_output = native
            else:
                ocr_pages.append(page_num)

        if not ocr_pages:
            if not self.config.quiet:
                console.print("  All pages born-digital (no OCR needed)")
            return

        available = self._available_engines_for_agentic()
        ladder = provider_ladder(
            available, per_page_only=True, max_cost_per_page=self.config.max_cost_per_page
        )
        if not ladder:
            logger.warning("agentic: no OCR providers available; pages left unprocessed")
            if not self.config.quiet:
                console.print("  [red]No OCR providers available[/red]")
            return

        judge = self._build_page_judge(state)
        max_attempts = max(1, self.config.max_retries + 1)
        enhancement_pages = [p for p in ocr_pages if state.pages[p].needs_ocr_enhancement]

        if not self.config.quiet:
            ladder_str = " -> ".join(f"{p.engine.value}(${p.cost_per_page_usd:g})" for p in ladder)
            console.print(f"  ladder: {ladder_str}")

        def run_provider(engine: EngineType, page_num: int) -> PageOutput:
            outs = self._run_engine_on_pages(
                state, [page_num], enhancement_pages, engine, "agentic"
            )
            return outs[0]

        for page_num in ocr_pages:
            # Budget guard: once the document budget is spent, only try the cheapest.
            cap = max_attempts
            if self.config.cost_budget > 0 and state.total_cost >= self.config.cost_budget:
                cap = 1
            decision = route_page(page_num, ladder, run_provider, judge, max_attempts=cap)

            ps = state.pages[page_num]
            for att in decision.attempts:
                att.output.cost_usd = att.cost_usd
                att.output.audit_passed = att.accepted
                ps.attempts.append(att.output)
            ps.best_output = decision.final_output

            # Record cost so DocumentState.total_cost reflects spend.
            state.engine_runs.append(EngineResult(
                document_path=state.handle.path,
                engine=decision.winning_engine,
                status=DocumentStatus.SUCCESS if decision.accepted else DocumentStatus.AUDIT_FAILED,
                cost=decision.total_cost_usd,
                processing_time=0.0,
            ))

            if not self.config.quiet:
                tag = "accepted" if decision.accepted else "best-effort"
                console.print(
                    f"  p{page_num}: {decision.winning_engine} "
                    f"[{tag}, {len(decision.attempts)} tr, ${decision.total_cost_usd:.4f}]"
                )

        if not self.config.quiet:
            console.print(f"  total cost: ${state.total_cost:.4f}")

    def _available_engines_for_agentic(self) -> set[EngineType]:
        """Probe which known providers are actually usable right now."""
        from socr.core.providers import DEFAULT_PROVIDERS

        available: set[EngineType] = set()
        for engine_type in self.config.enabled_engines:
            if engine_type not in DEFAULT_PROVIDERS:
                continue
            try:
                if get_engine(engine_type).is_available():
                    available.add(engine_type)
            except Exception:  # availability probe must never crash routing
                continue
        return available

    def _build_page_judge(self, state: DocumentState):
        """Select the page judge: VLM if requested+available, else heuristics."""
        from socr.pipeline.agentic import HeuristicPageJudge, VLMPageJudge

        backend = self.config.judge_backend
        if backend in ("vlm", "auto"):
            try:
                from socr.judge.ollama_judge import OllamaVisionJudge

                vj = (
                    OllamaVisionJudge(model=self.config.judge_model)
                    if self.config.judge_model
                    else OllamaVisionJudge()
                )
                if vj.is_available():
                    return VLMPageJudge(vj, self._make_page_renderer(state))
            except Exception as exc:
                logger.warning("VLM judge unavailable (%s); using heuristics", exc)
            if backend == "vlm" and not self.config.quiet:
                console.print("  [yellow]VLM judge unavailable -> heuristic judge[/yellow]")
        return HeuristicPageJudge(self.heuristics)

    def _make_page_renderer(self, state: DocumentState):
        """Return render_image(page_num) -> temp PNG path for the VLM judge."""
        def render(page_num: int) -> Path:
            img = state.handle.render_page(page_num, dpi=self.config.render_dpi)
            tmp = Path(tempfile.gettempdir()) / f"socr_judge_{state.handle.stem}_p{page_num}.png"
            img.save(tmp)
            return tmp

        return render

    def _write_manifest(self, state: DocumentState, output_dir: Path) -> None:
        """Write a reproducibility manifest + blob cache for `socr replay`.

        Non-fatal: a manifest failure must never lose the OCR output.
        """
        try:
            from socr.core.cache import BlobStore
            from socr.core.manifest import build_manifest

            doc_dir = output_dir / sanitize_filename(state.handle.stem)
            store = BlobStore(doc_dir / "cache")
            manifest = build_manifest(state, store, dpi=self.config.render_dpi)
            manifest.save(doc_dir / "manifest.json")
            if not self.config.quiet:
                console.print(f"  [dim]Manifest: {doc_dir / 'manifest.json'} (replayable)[/dim]")
        except Exception as exc:
            logger.warning("manifest write failed (non-fatal): %s", exc)

    # ------------------------------------------------------------------
    # Phase 2 (multi-engine): Backbone OCR with multiple engines
    # ------------------------------------------------------------------

    def _backbone_multi_engine(
        self,
        state: DocumentState,
        output_dir: Path,
    ) -> list[EngineResult]:
        """Run multiple CLI engines on the document and collect all results.

        Each engine's output is applied to DocumentState via
        ``state.apply_result()``, so per-page attempts accumulate across
        engines.  Returns the list of EngineResults for downstream scoring.
        """
        engines = self.config.multi_engine
        engine_names = [e.value for e in engines]

        if not self.config.quiet:
            console.print(
                f"\n[cyan]Phase 2:[/cyan] Multi-engine OCR "
                f"[{', '.join(engine_names)}]"
            )

        results: list[EngineResult] = []

        for idx, engine_type in enumerate(engines, 1):
            if not self.config.quiet:
                console.print(
                    f"  Engine {idx}/{len(engines)}: {engine_type.value}",
                    end="",
                )

            try:
                engine = get_engine(engine_type)
            except ValueError:
                if not self.config.quiet:
                    console.print(" [red]not supported[/red]")
                continue

            if not engine.is_available():
                if not self.config.quiet:
                    console.print(" [yellow]not available[/yellow]")
                continue

            # Per-page processing for all engines
            all_pages = list(range(1, state.handle.page_count + 1))
            page_outputs = engine.process_pages(
                pdf_path=state.handle.path,
                page_nums=all_pages,
                config=self.config,
                dpi=self.config.render_dpi,
            )
            success_count = sum(
                1 for p in page_outputs if p.status == PageStatus.SUCCESS
            )
            result = EngineResult(
                document_path=state.handle.path,
                engine=engine.name,
                status=(
                    DocumentStatus.SUCCESS if success_count > 0
                    else DocumentStatus.ERROR
                ),
                pages=page_outputs,
                pages_processed=state.handle.page_count,
                model_version=engine.model_version,
            )
            state.apply_result(result)

            word_count = sum(p.word_count for p in result.pages)
            if not self.config.quiet:
                if result.success:
                    console.print(f"... [green]{word_count} words[/green]")
                else:
                    console.print(
                        f"... [red]{result.error or result.status.value}[/red]"
                    )

            results.append(result)

        return results

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

        # When the backbone used chunking, each chunk was small enough to
        # avoid truncation.  Skip the doc-level truncation check because
        # dividing chunk output by total pages gives a misleadingly low
        # words-per-page ratio.
        was_chunked = state.handle.page_count > self.config.chunk_threshold
        scoring = self.scorer.score(
            whole_doc_page.text, engine=result.engine,
            expected_pages=0 if was_chunked else state.handle.page_count,
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

    def _phase_score_multi(
        self,
        state: DocumentState,
        backbone_results: list[EngineResult],
    ) -> None:
        """Score all engine outputs from multi-engine mode.

        For each engine result, runs scoring (whole-doc or per-page as
        appropriate) and prints a per-engine summary.
        """
        if not self.config.quiet:
            console.print("\n[cyan]Phase 3:[/cyan] Score (quality audit)")

        for result in backbone_results:
            if not result.success:
                if not self.config.quiet:
                    console.print(
                        f"  {result.engine}: [red]skipped (engine failed)[/red]"
                    )
                continue

            has_whole_doc = any(p.page_num == 0 for p in result.pages)

            if has_whole_doc:
                whole_page = next(p for p in result.pages if p.page_num == 0)
                was_chunked = (
                    state.handle.page_count > self.config.chunk_threshold
                )
                scoring = self.scorer.score(
                    whole_page.text,
                    engine=result.engine,
                    expected_pages=(
                        0 if was_chunked else state.handle.page_count
                    ),
                )
                whole_page.audit_passed = scoring.passed
                if scoring.passed:
                    whole_page.failure_mode = FailureMode.NONE
                    result.audit_passed = True
                else:
                    whole_page.failure_mode = scoring.primary_failure
                    result.audit_passed = False

                if not self.config.quiet:
                    if scoring.passed:
                        console.print(
                            f"  {result.engine}: [green]passed[/green]"
                        )
                    else:
                        console.print(
                            f"  {result.engine}: "
                            f"[red]{scoring.primary_failure.value}[/red]"
                        )
            else:
                # Per-page outputs: score each page
                passed = 0
                failed = 0
                for page_out in result.pages:
                    scoring = self.scorer.score(
                        page_out.text, engine=result.engine
                    )
                    page_out.audit_passed = scoring.passed
                    if scoring.passed:
                        page_out.failure_mode = FailureMode.NONE
                        passed += 1
                        # Promote to best if none set for this page
                        page_state = state.pages.get(page_out.page_num)
                        if page_state and not page_state.best_output:
                            page_state.best_output = page_out
                    else:
                        page_out.failure_mode = scoring.primary_failure
                        failed += 1

                if not self.config.quiet:
                    console.print(
                        f"  {result.engine}: "
                        f"[green]{passed} passed[/green], "
                        f"[red]{failed} failed[/red]"
                    )

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
                        all_pages = list(range(1, state.handle.page_count + 1))
                        page_outputs = engine.process_pages(
                            state.handle.path, all_pages, self.config,
                            dpi=self.config.render_dpi,
                        )
                        retry_result = EngineResult(
                            document_path=state.handle.path,
                            engine=engine.name,
                            status=DocumentStatus.SUCCESS if any(
                                p.status == PageStatus.SUCCESS for p in page_outputs
                            ) else DocumentStatus.ERROR,
                            pages=page_outputs,
                            pages_processed=state.handle.page_count,
                        )
                        state.apply_result(retry_result)
                        if retry_result.success:
                            self._score_repair_result(
                                state, retry_result, []
                            )
                        # Check if per-page results pass
                        ok = sum(
                            1 for p in page_outputs
                            if p.status == PageStatus.SUCCESS and p.audit_passed
                        )
                        if ok == state.handle.page_count:
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
                        all_pages = list(range(1, state.handle.page_count + 1))
                        page_outputs = engine.process_pages(
                            state.handle.path, all_pages, self.config,
                            dpi=self.config.render_dpi,
                        )
                        repair_result = EngineResult(
                            document_path=state.handle.path,
                            engine=engine.name,
                            status=DocumentStatus.SUCCESS if any(
                                p.status == PageStatus.SUCCESS for p in page_outputs
                            ) else DocumentStatus.ERROR,
                            pages=page_outputs,
                            pages_processed=state.handle.page_count,
                        )
                        state.apply_result(repair_result)
                        if repair_result.success:
                            self._score_repair_result(
                                state, repair_result, []
                            )
                            if not state.pages_needing_repair:
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

                # Only process the failed pages, not the whole document
                failed_pages = [r.page_num for r in repairs]
                page_outputs = engine.process_pages(
                    state.handle.path, failed_pages, self.config,
                    dpi=self.config.render_dpi,
                )
                repair_result = EngineResult(
                    document_path=state.handle.path,
                    engine=engine.name,
                    status=DocumentStatus.SUCCESS if any(
                        p.status == PageStatus.SUCCESS for p in page_outputs
                    ) else DocumentStatus.ERROR,
                    pages=page_outputs,
                    pages_processed=len(failed_pages),
                )
                state.apply_result(repair_result)

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
        """Run multi-engine consensus on pages/docs with multiple attempts.

        Handles both per-page attempts (HTTP engines) and whole-doc
        attempts (CLI engines).
        """
        has_multi_pages = any(
            len(state.pages[pn].attempts) >= 2
            and not (state.pages[pn].is_born_digital and state.pages[pn].native_text)
            for pn in state.pages
        )
        has_multi_whole_doc = len(state.whole_doc_attempts) >= 2

        if not has_multi_pages and not has_multi_whole_doc:
            if not self.config.quiet:
                console.print(
                    "\n[cyan]Phase 4b:[/cyan] Consensus (not needed — "
                    "no multi-attempt pages)"
                )
            return

        if not self.config.quiet:
            parts = []
            if has_multi_whole_doc:
                parts.append(f"{len(state.whole_doc_attempts)} whole-doc attempts")
            if has_multi_pages:
                count = sum(
                    1 for pn in state.pages
                    if len(state.pages[pn].attempts) >= 2
                )
                parts.append(f"{count} multi-attempt pages")
            console.print(
                f"\n[cyan]Phase 4b:[/cyan] Consensus ({', '.join(parts)})"
            )

        engine = ConsensusEngine(
            use_llm=self.config.consensus_use_llm,
            ollama_model=self.config.consensus_ollama_model,
            quiet=self.config.quiet,
        )
        results = engine.reconcile_document(state)

        if not self.config.quiet:
            for cr in results:
                disc_str = f" [{len(cr.discrepancies)} discrepancies]" if cr.discrepancies else ""
                label = "Whole doc" if cr.page_num == 0 else f"Page {cr.page_num}"
                console.print(
                    f"  {label}: selected {cr.selected_engine} "
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

        # Strip phantom image references before saving
        normalizer = OutputNormalizer()
        stem = sanitize_filename(state.handle.stem)
        doc_dir = output_dir / stem
        if has_text:
            final_text = normalizer.strip_phantom_images(
                final_text, output_dir=doc_dir
            )

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

        # Figure extraction + description + embedding
        if self.config.save_figures and has_text:
            final_text = self._describe_and_embed_figures(
                state, final_result, output_dir, final_text,
            )
            # Update the page text with embedded figure blocks
            final_result.pages[0].text = final_text

        # Save markdown
        if has_text:
            saved_path = self._save_markdown(state, final_text, output_dir)
            if not self.config.quiet:
                console.print(f"  [blue]Output:[/blue] {saved_path}")

        # Reproducibility manifest (opt-in; default-on in agentic mode).
        if has_text and (self.config.write_manifest or self.config.agentic):
            self._write_manifest(state, output_dir)

        return final_result

    def _describe_and_embed_figures(
        self,
        state: DocumentState,
        result: EngineResult,
        output_dir: Path,
        text: str,
    ) -> str:
        """Extract figures, describe with a vision model, embed in markdown.

        1. Extract figures from the PDF (saves PNGs).
        2. For each figure with an image, send to Gemini API for description.
        3. Append figure blocks at the end of the markdown.
        4. Return modified text.

        Graceful degradation: if GEMINI_API_KEY is unavailable the figures
        are saved without descriptions.
        """
        if not self.config.quiet:
            console.print("  Extracting figures...")

        stem = sanitize_filename(state.handle.stem)
        doc_dir = output_dir / stem
        figures_dir = doc_dir / "figures"
        extractor = FigureExtractor(
            max_total=self.config.figures_max_total,
            max_per_page=self.config.figures_max_per_page,
            save_dir=figures_dir,
        )
        extracted = extractor.extract(state.handle.path)

        if not extracted:
            if not self.config.quiet:
                console.print("  [dim]No figures detected[/dim]")
            result.figures = []
            return text

        if not self.config.quiet:
            console.print(
                f"  Extracted {len(extracted)} figures to {figures_dir}"
            )

        # Try to get a vision engine for descriptions
        vision_engine = self._get_vision_engine()
        figures: list[FigureInfo] = []

        for fig in extracted:
            description = ""
            figure_type = "extracted"

            if vision_engine is not None and fig.image is not None:
                # Get page context for better descriptions
                context = self._get_page_context(state, fig.page_num)
                info = vision_engine.describe_figure(
                    fig.image, context=context,
                )
                description = info.description
                figure_type = info.figure_type or "extracted"

            fig_info = FigureInfo(
                figure_num=fig.figure_num,
                page_num=fig.page_num,
                figure_type=figure_type,
                description=description,
                image_path=fig.saved_path,
                engine=vision_engine.name if vision_engine else "",
            )
            figures.append(fig_info)

        if vision_engine is not None:
            vision_engine.close()

        result.figures = figures

        if not self.config.quiet:
            described = sum(1 for f in figures if f.description)
            console.print(
                f"  {len(figures)} figures processed"
                f" ({described} described)"
            )

        # Build figure blocks and append to text
        figure_blocks = self._build_figure_blocks(figures, doc_dir)
        if figure_blocks:
            text = text.rstrip() + "\n\n" + figure_blocks

        return text

    def _get_vision_engine(self):
        """Try to create a Gemini API engine for figure description.

        Returns None if GEMINI_API_KEY is not available.
        """
        import os

        api_key = os.environ.get("GEMINI_API_KEY", "") or os.environ.get(
            "GOOGLE_API_KEY", ""
        )
        if not api_key:
            if not self.config.quiet:
                console.print(
                    "  [dim]No GEMINI_API_KEY — saving figures "
                    "without descriptions[/dim]"
                )
            return None

        from socr.engines.gemini_api import GeminiAPIConfig, GeminiAPIEngine

        engine = GeminiAPIEngine(
            GeminiAPIConfig(
                api_key=api_key,
                model=self.config.gemini_model,
            )
        )
        if engine.initialize():
            return engine

        if not self.config.quiet:
            console.print(
                "  [dim]Gemini API not reachable — saving figures "
                "without descriptions[/dim]"
            )
        return None

    @staticmethod
    def _get_page_context(state: DocumentState, page_num: int) -> str:
        """Get text context from a page for figure description."""
        page_state = state.pages.get(page_num)
        if page_state and page_state.best_output:
            return (page_state.best_output.text or "")[:500]

        # Fall back to whole-doc attempts — extract a rough slice
        for attempt in state.whole_doc_attempts:
            if attempt.text:
                return attempt.text[:500]
        return ""

    @staticmethod
    def _build_figure_blocks(
        figures: list[FigureInfo], doc_dir: Path,
    ) -> str:
        """Build markdown figure blocks for embedding.

        Each block looks like::

            **Figure N** (page P): [description]

            ![Figure N](figures/figure_N_pageP.png)
        """
        blocks = []
        for fig in figures:
            # Compute a relative image path from the doc directory
            if fig.image_path:
                try:
                    rel_path = Path(fig.image_path).relative_to(doc_dir)
                except ValueError:
                    rel_path = Path(fig.image_path).name
            else:
                continue

            header = f"**Figure {fig.figure_num}** (page {fig.page_num})"
            if fig.description:
                header += f": {fig.description}"

            block = f"{header}\n\n![Figure {fig.figure_num}]({rel_path})"
            blocks.append(block)

        return "\n\n".join(blocks)

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

        engine_str = result.engine
        if self.config.multi_engine:
            engine_str = (
                "+".join(e.value for e in self.config.multi_engine)
                + " (consensus)"
            )

        console.print(
            f"\n{status_str} | {engine_str} | "
            f"{result.processing_time:.1f}s"
        )
        if state.pages_needing_repair:
            console.print(
                f"[yellow]{len(state.pages_needing_repair)} page(s) "
                f"still failing[/yellow]"
            )
        if result.error:
            console.print(f"[dim]{result.error}[/dim]")
