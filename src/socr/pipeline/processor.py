"""Standard OCR pipeline — document-level processing via CLI engines.

4 stages:
  1. Primary OCR (one subprocess per PDF)
  2. Quality audit (heuristics on whole-document markdown)
  3. Fallback OCR (re-run with different engine if audit fails)
  4. Figure extraction (deferred to FigureExtractor — TICKET-4)
"""

import logging
import time
from pathlib import Path

from rich.console import Console

from socr.audit.heuristics import HeuristicsChecker
from socr.core.config import PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.metadata import MetadataManager
from socr.core.result import DocumentStatus, EngineResult
from socr.engines.registry import get_engine
from socr.figures.extractor import FigureExtractor

logger = logging.getLogger(__name__)
console = Console()


class StandardPipeline:
    """Document-level OCR pipeline with cascading fallback."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.heuristics = HeuristicsChecker(min_word_count=config.audit_min_words)

    def process(self, pdf_path: Path, output_dir: Path | None = None) -> EngineResult:
        """Process a single PDF through the full pipeline."""
        doc = DocumentHandle.from_path(pdf_path)
        out_dir = output_dir or self.config.output_dir

        if not self.config.quiet:
            console.print(f"[blue]Processing:[/blue] {doc.filename}")
            console.print(f"[dim]{doc.page_count} pages, {doc.size_mb:.1f} MB[/dim]")

        # Stage 1: Primary OCR
        result = self._run_primary(doc, out_dir)

        # Stage 2: Audit
        if self.config.audit_enabled and result.success:
            audit_ok = self._run_audit(result)

            # Stage 3: Fallback (if audit failed)
            if not audit_ok:
                fallback_result = self._run_fallback(doc, out_dir)
                if fallback_result and fallback_result.success:
                    result = fallback_result

        # Stage 4: Figures
        if self.config.save_figures and result.success:
            self._run_figures(doc, result, out_dir)

        # Save output
        if result.success:
            saved = self._save_result(doc, result, out_dir)
            if not self.config.quiet:
                console.print(f"[blue]Output:[/blue] {saved}")

        if not self.config.quiet:
            self._print_summary(result)

        return result

    def process_batch(self, input_dir: Path, output_dir: Path | None = None) -> list[EngineResult]:
        """Process all PDFs in a directory with incremental tracking."""
        out_dir = output_dir or self.config.output_dir
        meta = MetadataManager(out_dir)

        pdfs = sorted(input_dir.glob("*.pdf"))
        if not pdfs:
            if not self.config.quiet:
                console.print("[yellow]No PDF files found[/yellow]")
            return []

        # Filter already-processed files
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

        results = []
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

    # --- Stages ---

    def _run_primary(self, doc: DocumentHandle, output_dir: Path) -> EngineResult:
        """Stage 1: Primary OCR via CLI engine."""
        engine = get_engine(self.config.primary_engine)

        if not self.config.quiet:
            console.print(f"\n[cyan]Stage 1:[/cyan] Primary OCR [{engine.name}]")

        if not engine.is_available():
            logger.warning(f"Primary engine {engine.name} not available")
            if not self.config.quiet:
                console.print(f"[red]Engine {engine.name} not available[/red]")
            return EngineResult(
                document_path=doc.path,
                engine=engine.name,
                status=DocumentStatus.ERROR,
                error=f"Engine {engine.name} not available (CLI not installed or missing API key)",
            )

        result = engine.process_document(doc.path, output_dir, self.config)
        result.pages_processed = doc.page_count
        return result

    def _run_audit(self, result: EngineResult) -> bool:
        """Stage 2: Quality audit on whole-document markdown. Returns True if passed."""
        if not self.config.quiet:
            console.print(f"\n[cyan]Stage 2:[/cyan] Quality audit")

        check = self.heuristics.check(result.markdown)
        result.audit_passed = check.passed

        if check.errors:
            result.audit_notes = check.errors
            result.status = DocumentStatus.AUDIT_FAILED
            if not self.config.quiet:
                for err in check.errors:
                    console.print(f"  [red]FAIL:[/red] {err}")
        elif not self.config.quiet:
            console.print("  [green]Passed[/green]")

        return check.passed

    def _run_fallback(self, doc: DocumentHandle, output_dir: Path) -> EngineResult | None:
        """Stage 3: Fallback OCR — try each engine in fallback_chain until one succeeds."""
        primary = self.config.primary_engine

        for engine_type in self.config.fallback_chain:
            if engine_type == primary:
                continue

            engine = get_engine(engine_type)

            if not self.config.quiet:
                console.print(f"\n[cyan]Stage 3:[/cyan] Fallback OCR [{engine.name}]")

            if not engine.is_available():
                if not self.config.quiet:
                    console.print(f"[yellow]Fallback engine {engine.name} not available, trying next[/yellow]")
                continue

            result = engine.process_document(doc.path, output_dir, self.config)
            result.pages_processed = doc.page_count
            if result.success:
                return result

            if not self.config.quiet:
                console.print(f"[yellow]Fallback engine {engine.name} failed, trying next[/yellow]")

        logger.info("All fallback engines exhausted")
        return None

    def _run_figures(self, doc: DocumentHandle, result: EngineResult, output_dir: Path) -> None:
        """Stage 4: Extract figure images from the PDF."""
        if not self.config.quiet:
            console.print(f"\n[cyan]Stage 4:[/cyan] Figure extraction")

        from socr.engines.base import sanitize_filename

        figures_dir = output_dir / sanitize_filename(doc.stem) / "figures"
        extractor = FigureExtractor(
            max_total=self.config.figures_max_total,
            max_per_page=self.config.figures_max_per_page,
            save_dir=figures_dir,
        )
        extracted = extractor.extract(doc.path)

        if not self.config.quiet:
            if extracted:
                console.print(f"  Extracted {len(extracted)} figures to {figures_dir}")
            else:
                console.print("  [dim]No figures detected[/dim]")

        from socr.core.result import FigureInfo
        result.figures = [
            FigureInfo(
                figure_num=fig.figure_num,
                page_num=fig.page_num,
                figure_type="extracted",
                image_path=fig.saved_path,
            )
            for fig in extracted
        ]

    def _save_result(self, doc: DocumentHandle, result: EngineResult, output_dir: Path) -> Path:
        """Save the OCR markdown to output_dir/{stem}/{stem}.md."""
        from socr.engines.base import sanitize_filename

        stem = sanitize_filename(doc.stem)
        doc_dir = output_dir / stem
        doc_dir.mkdir(parents=True, exist_ok=True)
        md_path = doc_dir / f"{stem}.md"
        md_path.write_text(result.markdown, encoding="utf-8")
        return md_path

    def _print_summary(self, result: EngineResult) -> None:
        status = "[green]Success[/green]" if result.success else f"[red]{result.status.value}[/red]"
        console.print(f"\n{status} | {result.engine} | {result.processing_time:.1f}s")
        if result.error:
            console.print(f"[dim]{result.error}[/dim]")
