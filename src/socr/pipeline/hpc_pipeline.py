"""HPC pipeline for single-GPU setups with vLLM.

Processes documents per-page via vLLM HTTP API with sequential model swapping:
  1. OCR Phase: DeepSeek-OCR extracts text from all pages
  2. Nougat Phase (optional): LaTeX equation extraction
  3. Reconciliation: Merge DeepSeek + Nougat outputs
  4. Figure Phase: Vision model describes extracted figures

Uses VLLMServerManager to swap models between phases on a single GPU.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
from rich.console import Console

from socr.audit.heuristics import HeuristicsChecker
from socr.core.config import EngineType, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.result import (
    DocumentResult,
    DocumentStatus,
    FigureInfo,
    PageResult,
    PageStatus,
)
from socr.engines.deepseek_vllm import DeepSeekVLLMConfig, DeepSeekVLLMEngine
from socr.engines.vllm import VLLMConfig, VLLMEngine
from socr.engines.vllm_manager import ServerConfig, VLLMServerManager
from socr.figures.extractor import FigureExtractor
from socr.pipeline.reconciler import (
    EngineOutput,
    OutputReconciler,
    create_page_result_from_reconciliation,
)

logger = logging.getLogger(__name__)
console = Console()


class HPCPipeline:
    """HPC pipeline with sequential model loading for single-GPU setups.

    Swaps models between processing phases to fit within single-GPU memory.
    Each phase starts vLLM with the required model, processes all items,
    then stops vLLM and frees GPU memory.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.server_manager = VLLMServerManager(verbose=config.verbose)
        self.reconciler = OutputReconciler(
            use_llm_reconciler=config.hpc.use_llm_reconciler,
            reconciler_model=config.hpc.reconciler_model,
            vllm_url=config.hpc.vllm_url,
        )
        self.heuristics = HeuristicsChecker(
            min_word_count=config.audit_min_words,
            max_garbage_ratio=0.15,
        )
        self._page_images: dict[int, Image.Image] = {}

    def process(self, pdf_path: Path, output_dir: Path | None = None) -> DocumentResult:
        """Process a document through the HPC pipeline."""
        start_time = time.time()
        out_dir = output_dir or self.config.output_dir
        doc = DocumentHandle.from_path(pdf_path)

        if not self.config.quiet:
            console.print(f"[blue]Processing (HPC):[/blue] {doc.filename}")
            console.print(f"[dim]{doc.page_count} pages, {doc.size_mb:.1f} MB[/dim]")
            if self.config.hpc.manage_server:
                console.print("[dim]Sequential mode — single-GPU model swapping[/dim]")

        # Render all pages upfront for reuse across phases
        if not self.config.quiet:
            console.print("[dim]Rendering pages...[/dim]")
        self._page_images = doc.render_all_pages(dpi=self.config.hpc.render_dpi)

        try:
            # Phase 1: OCR with DeepSeek-vLLM
            ocr_outputs = self._run_ocr_phase(doc)

            # Phase 2: Nougat for LaTeX (optional)
            nougat_outputs: dict[int, EngineOutput] = {}
            if self.config.hpc.use_nougat:
                nougat_outputs = self._run_nougat_phase(doc)

            # Phase 3: Reconciliation
            page_results = self._run_reconciliation_phase(
                ocr_outputs, nougat_outputs, doc.page_count
            )

            # Build combined markdown from page results
            markdown = self._assemble_markdown(page_results)

            # Phase 4: Figures
            figures: list[FigureInfo] = []
            if self.config.save_figures:
                figures = self._run_figure_phase(doc, page_results, out_dir)

        finally:
            self.server_manager.stop()
            self._page_images.clear()

        processing_time = time.time() - start_time
        success_pages = sum(1 for r in page_results if r.status == PageStatus.SUCCESS)

        result = DocumentResult(
            document_path=pdf_path,
            engine="hpc-sequential",
            status=DocumentStatus.SUCCESS if success_pages > 0 else DocumentStatus.ERROR,
            markdown=markdown,
            pages_processed=doc.page_count,
            processing_time=processing_time,
            figures=figures,
        )

        # Save output
        if result.success:
            saved = self._save_result(doc, result, out_dir)
            if not self.config.quiet:
                console.print(f"[blue]Output:[/blue] {saved}")

        if not self.config.quiet:
            status = "[green]Success[/green]" if result.success else "[red]Error[/red]"
            console.print(f"\n{status} | hpc-sequential | {processing_time:.1f}s")
            console.print(f"[dim]{success_pages}/{doc.page_count} pages successful[/dim]")

        return result

    # --- Phases ---

    def _run_ocr_phase(self, doc: DocumentHandle) -> dict[int, EngineOutput]:
        """Phase 1: OCR with DeepSeek-vLLM."""
        if not self.config.quiet:
            console.print(f"\n[cyan]Phase 1:[/cyan] OCR [deepseek-vllm]")

        # Start vLLM with OCR model
        base_url, api_key = self._start_server(self.config.hpc.ocr_model)

        engine_config = DeepSeekVLLMConfig(
            base_url=base_url, model=self.config.hpc.ocr_model, api_key=api_key
        )
        engine = DeepSeekVLLMEngine(engine_config)

        if not engine.initialize():
            raise RuntimeError(f"Failed to initialize DeepSeek-vLLM at {base_url}")

        outputs: dict[int, EngineOutput] = {}
        workers = self.config.hpc.parallel_pages

        if not self.config.quiet:
            console.print(f"[dim]Processing {doc.page_count} pages ({workers} workers)[/dim]")

        if workers <= 1:
            for page_num, image in self._page_images.items():
                result = engine.process_image(image, page_num)
                if result.status == PageStatus.SUCCESS:
                    outputs[page_num] = EngineOutput(
                        engine="deepseek-vllm",
                        text=result.text,
                        confidence=result.confidence or 0.85,
                        processing_time=result.processing_time,
                    )
                elif not self.config.quiet:
                    console.print(f"  [red]Page {page_num}:[/red] {result.error_message}")
        else:
            def process_page(pn: int, img: Image.Image):
                return pn, engine.process_image(img, pn)

            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(process_page, pn, img): pn
                    for pn, img in self._page_images.items()
                }
                for future in as_completed(futures):
                    page_num, result = future.result()
                    if result.status == PageStatus.SUCCESS:
                        outputs[page_num] = EngineOutput(
                            engine="deepseek-vllm",
                            text=result.text,
                            confidence=result.confidence or 0.85,
                            processing_time=result.processing_time,
                        )

        engine.close()

        # Stop server to free GPU
        if self.config.hpc.manage_server:
            if not self.config.quiet:
                console.print("[dim]Stopping OCR server...[/dim]")
            self.server_manager.stop()

        # Quality audit + Gemini fallback
        if self.config.hpc.audit_enabled:
            failed_pages = self._audit_ocr_results(outputs)
            if failed_pages and self.config.hpc.cloud_fallback:
                gemini_outputs = self._fallback_to_gemini(failed_pages)
                outputs.update(gemini_outputs)

        if not self.config.quiet:
            console.print(f"  [green]{len(outputs)}/{len(self._page_images)} pages extracted[/green]")

        return outputs

    def _run_nougat_phase(self, doc: DocumentHandle) -> dict[int, EngineOutput]:
        """Phase 2: Nougat for LaTeX extraction (runs locally, no vLLM)."""
        if not self.config.quiet:
            console.print(f"\n[cyan]Phase 2:[/cyan] Nougat [LaTeX extraction]")

        from socr.engines.nougat import NougatEngine

        engine = NougatEngine()
        if not engine.is_available():
            if not self.config.quiet:
                console.print("[yellow]Nougat not available, skipping[/yellow]")
            return {}

        # Nougat is a CLI engine — process the whole document
        result = engine.process_document(doc.path, self.config.output_dir, self.config)
        if not result.success:
            if not self.config.quiet:
                console.print(f"[yellow]Nougat failed: {result.error}[/yellow]")
            return {}

        # Split nougat output into per-page outputs (best effort by page breaks)
        outputs: dict[int, EngineOutput] = {}
        pages = result.markdown.split("\n\n---\n\n")
        for i, page_text in enumerate(pages):
            page_num = i + 1
            if page_text.strip():
                outputs[page_num] = EngineOutput(
                    engine="nougat",
                    text=page_text.strip(),
                    confidence=0.8,
                    processing_time=result.processing_time / max(len(pages), 1),
                )

        if not self.config.quiet:
            console.print(f"  [green]{len(outputs)} pages with LaTeX[/green]")

        return outputs

    def _run_reconciliation_phase(
        self,
        ocr_outputs: dict[int, EngineOutput],
        nougat_outputs: dict[int, EngineOutput],
        total_pages: int,
    ) -> list[PageResult]:
        """Phase 3: Reconcile OCR + Nougat outputs."""
        if not self.config.quiet:
            console.print(f"\n[cyan]Phase 3:[/cyan] Reconciliation")

        results: list[PageResult] = []
        latex_merged = 0

        for page_num in range(1, total_pages + 1):
            outputs = []
            if page_num in ocr_outputs:
                outputs.append(ocr_outputs[page_num])
            if page_num in nougat_outputs:
                outputs.append(nougat_outputs[page_num])

            if not outputs:
                results.append(PageResult(
                    page_num=page_num,
                    status=PageStatus.ERROR,
                    error_message="No engine produced output",
                ))
                continue

            reconciliation = self.reconciler.reconcile(outputs, page_num)
            page_result = create_page_result_from_reconciliation(
                reconciliation, page_num,
                processing_time=sum(o.processing_time for o in outputs),
            )
            results.append(page_result)

            if reconciliation.latex_source:
                latex_merged += reconciliation.conflicts_resolved

        success_count = sum(1 for r in results if r.status == PageStatus.SUCCESS)
        if not self.config.quiet:
            console.print(f"  [green]{success_count}/{total_pages} pages reconciled[/green]")
            if latex_merged > 0:
                console.print(f"  [dim]{latex_merged} LaTeX blocks merged from Nougat[/dim]")

        return results

    def _run_figure_phase(
        self,
        doc: DocumentHandle,
        page_results: list[PageResult],
        output_dir: Path,
    ) -> list[FigureInfo]:
        """Phase 4: Extract and describe figures."""
        if not self.config.quiet:
            console.print(f"\n[cyan]Phase 4:[/cyan] Figure extraction & description")

        from socr.engines.base import sanitize_filename

        figures_dir = output_dir / sanitize_filename(doc.stem) / "figures"
        extractor = FigureExtractor(
            max_total=self.config.figures_max_total,
            max_per_page=self.config.figures_max_per_page,
            save_dir=figures_dir,
        )
        extracted = extractor.extract(doc.path)

        if not extracted:
            if not self.config.quiet:
                console.print("  [dim]No figures detected[/dim]")
            return []

        if not self.config.quiet:
            console.print(f"  [dim]{len(extracted)} figures extracted[/dim]")

        # Start vision model for descriptions
        base_url, api_key = self._start_server(self.config.hpc.vision_model)
        vision_config = VLLMConfig(base_url=base_url, model=self.config.hpc.vision_model, api_key=api_key)
        vision_engine = VLLMEngine(vision_config)

        figures: list[FigureInfo] = []
        if vision_engine.initialize():
            for fig in extracted:
                # Get context from the page result
                context = ""
                for pr in page_results:
                    if pr.page_num == fig.page_num:
                        context = (pr.text or "")[:500]
                        break

                if fig.image is not None:
                    info = vision_engine.describe_figure(fig.image, context=context)
                    info.figure_num = fig.figure_num
                    info.page_num = fig.page_num
                    if fig.saved_path:
                        info.image_path = fig.saved_path
                    figures.append(info)
                else:
                    figures.append(FigureInfo(
                        figure_num=fig.figure_num,
                        page_num=fig.page_num,
                        figure_type="extracted",
                        image_path=fig.saved_path,
                    ))

            vision_engine.close()
        else:
            if not self.config.quiet:
                console.print("[yellow]Vision model not available, saving figures without descriptions[/yellow]")
            figures = [
                FigureInfo(
                    figure_num=fig.figure_num,
                    page_num=fig.page_num,
                    figure_type="extracted",
                    image_path=fig.saved_path,
                )
                for fig in extracted
            ]

        if self.config.hpc.manage_server:
            self.server_manager.stop()

        if not self.config.quiet:
            console.print(f"  [green]{len(figures)} figures processed[/green]")

        return figures

    # --- Helpers ---

    def _start_server(self, model: str) -> tuple[str, str]:
        """Start vLLM with a model, or use existing URL.

        Returns:
            Tuple of (base_url, api_key).
        """
        if self.config.hpc.manage_server:
            if not self.config.quiet:
                console.print(f"[dim]Starting vLLM with {model}...[/dim]")
            server_config = ServerConfig(
                model=model,
                port=self.config.hpc.vllm_port,
                gpu_memory_utilization=self.config.hpc.gpu_memory_utilization,
                max_model_len=self.config.hpc.max_model_len,
            )
            self.server_manager.start(
                server_config,
                timeout=self.config.hpc.server_startup_timeout,
            )
            return self.server_manager.get_base_url(), self.server_manager.get_api_key()
        return self.config.hpc.vllm_url, ""

    def _audit_ocr_results(self, ocr_outputs: dict[int, EngineOutput]) -> list[int]:
        """Run heuristics audit on OCR outputs, return failed page numbers."""
        failed = []
        for page_num, output in ocr_outputs.items():
            result = self.heuristics.check(output.text)
            if not result.passed:
                failed.append(page_num)
                if not self.config.quiet:
                    errors = ", ".join(result.errors[:2])
                    console.print(f"  [yellow]Page {page_num} failed audit: {errors}[/yellow]")

        if not self.config.quiet:
            if failed:
                console.print(f"  [dim]{len(failed)}/{len(ocr_outputs)} pages failed audit[/dim]")
            else:
                console.print(f"  [green]All {len(ocr_outputs)} pages passed audit[/green]")

        return failed

    def _fallback_to_gemini(self, failed_pages: list[int]) -> dict[int, EngineOutput]:
        """Re-OCR failed pages with Gemini cloud fallback."""
        if not failed_pages:
            return {}

        if not self.config.quiet:
            console.print(f"  [dim]Gemini fallback for {len(failed_pages)} pages...[/dim]")

        from socr.engines.gemini import GeminiEngine

        engine = GeminiEngine()
        if not engine.is_available():
            if not self.config.quiet:
                console.print("  [yellow]Gemini not available (check GEMINI_API_KEY)[/yellow]")
            return {}

        # Gemini is a CLI engine — we need to process the whole doc and extract pages
        # For now, use document-level fallback (re-OCR entire doc)
        # A more sophisticated approach would process individual pages
        outputs: dict[int, EngineOutput] = {}

        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            result = engine.process_document(
                self._page_images[1] if 1 in self._page_images else list(self._page_images.values())[0],
                Path(tmpdir),
                self.config,
            ) if False else None  # Gemini needs the PDF path, not images

        # Gemini CLI works on the whole PDF — can't do per-page fallback easily
        # Log this limitation
        if not self.config.quiet:
            console.print("  [dim]Per-page Gemini fallback not yet supported in HPC mode[/dim]")

        return outputs

    @staticmethod
    def _assemble_markdown(page_results: list[PageResult]) -> str:
        """Combine per-page results into a single markdown document."""
        parts = []
        for pr in sorted(page_results, key=lambda r: r.page_num):
            if pr.status == PageStatus.SUCCESS and pr.text:
                parts.append(pr.text)
        return "\n\n---\n\n".join(parts)

    def _save_result(self, doc: DocumentHandle, result: DocumentResult, output_dir: Path) -> Path:
        """Save the OCR markdown to output_dir/{stem}/{stem}.md."""
        from socr.engines.base import sanitize_filename

        stem = sanitize_filename(doc.stem)
        doc_dir = output_dir / stem
        doc_dir.mkdir(parents=True, exist_ok=True)

        # Add frontmatter
        frontmatter = (
            f"---\n"
            f"source: {doc.filename}\n"
            f"engine: hpc-sequential\n"
            f"processed: {datetime.now(timezone.utc).isoformat()}\n"
            f"---\n\n"
        )

        md_path = doc_dir / f"{stem}.md"
        md_path.write_text(frontmatter + result.markdown, encoding="utf-8")
        return md_path
