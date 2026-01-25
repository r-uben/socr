"""HPC-optimized multi-agent OCR pipeline.

Runs multiple OCR engines in parallel locally via vLLM, then reconciles
outputs intelligently. No cloud fallback in HPC mode.

Pipeline flow:
    PDF -> [DeepSeek-OCR-vLLM] -> text/structure  -|
        -> [Nougat]            -> LaTeX equations -+-> [Reconciler] -> merged output
        -> [Vision-vLLM]       -> figure desc.   -|
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from smart_ocr.core.config import AgentConfig, EngineType
from smart_ocr.core.document import Document
from smart_ocr.core.result import OCRResult, PageResult, PageStatus
from smart_ocr.engines.base import BaseEngine
from smart_ocr.engines.deepseek_vllm import DeepSeekVLLMEngine
from smart_ocr.engines.nougat import NougatEngine
from smart_ocr.engines.vllm import VLLMEngine
from smart_ocr.pipeline.processor import OCRPipeline
from smart_ocr.pipeline.reconciler import (
    EngineOutput,
    OutputReconciler,
    create_page_result_from_reconciliation,
)


class HPCPipeline(OCRPipeline):
    """HPC-optimized pipeline with parallel multi-engine processing.

    This pipeline is designed for HPC environments where:
    - vLLM serves DeepSeek-OCR for primary text extraction
    - Nougat provides better LaTeX equation handling
    - Vision models (InternVL2/Qwen2-VL) describe figures
    - All engines run locally (no cloud fallback)

    The outputs are reconciled to produce the best combined result.
    """

    def __init__(self, config: AgentConfig | None = None) -> None:
        """Initialize HPC pipeline with multi-engine support."""
        super().__init__(config)

        # Update vLLM engines with HPC config
        if self.config.hpc.enabled:
            # Configure DeepSeek-vLLM with HPC settings
            self.config.deepseek_vllm.base_url = self.config.hpc.vllm_url
            self.config.deepseek_vllm.model = self.config.hpc.ocr_model

            # Configure vision vLLM with HPC settings
            self.config.vllm.base_url = self.config.hpc.vllm_url
            self.config.vllm.model = self.config.hpc.vision_model

            # Re-initialize engines with updated config
            self.engines[EngineType.DEEPSEEK_VLLM] = DeepSeekVLLMEngine(self.config.deepseek_vllm)
            self.engines[EngineType.VLLM] = VLLMEngine(self.config.vllm)

        # Initialize reconciler
        self.reconciler = OutputReconciler(
            use_llm_reconciler=self.config.hpc.use_llm_reconciler,
            reconciler_model=self.config.hpc.reconciler_model,
            vllm_url=self.config.hpc.vllm_url,
        )

        # Track which engines are available for HPC mode
        self._hpc_engines: list[EngineType] = []

    def process(self, pdf_path: Path | str, output_path: Path | str | None = None) -> OCRResult:
        """Process a document through the HPC pipeline.

        HPC pipeline stages:
        1. Parallel OCR with multiple engines (DeepSeek-vLLM + Nougat)
        2. Reconciliation (merge outputs, prefer Nougat for LaTeX)
        3. Figure processing (reuse from base class)

        Args:
            pdf_path: Path to the PDF file to process
            output_path: Optional custom output path

        Returns:
            OCRResult with processed pages and figures
        """
        self._start_time = time.time()
        pdf_path = Path(pdf_path)
        self._custom_output_path = Path(output_path) if output_path else None

        # Print header
        self.console.print_header()
        self.console.console.print("[info]HPC Mode[/info] - Multi-engine parallel processing\n")

        # Load document
        document = Document.from_pdf(pdf_path, render_dpi=self.config.render_dpi)
        document.classify()

        self.console.print_document_info(
            filename=document.filename,
            pages=document.num_pages,
            size_mb=document.size_mb,
            doc_type=document.doc_type.value,
            detected_features=document.detected_features,
        )

        # Initialize result with HPC metadata
        result = OCRResult(document_path=str(pdf_path))
        default_output_file = self._default_output_file(pdf_path)
        result.metadata.update({
            "doc_type": document.doc_type.value,
            "detected_features": document.detected_features,
            "default_output_file": str(default_output_file),
            "hpc_mode": True,
            "vllm_url": self.config.hpc.vllm_url,
        })

        # Determine available HPC engines
        self._hpc_engines = self._select_hpc_engines()

        if not self._hpc_engines:
            self.console.print_warning("No HPC engines available, falling back to standard pipeline")
            return super().process(pdf_path, output_path)

        # Stage 1: Parallel OCR with multiple engines
        engine_outputs = self._run_parallel_ocr(document)

        # Stage 2: Reconciliation
        page_results = self._run_reconciliation(engine_outputs, document.num_pages)
        for r in page_results:
            result.add_page_result(r)

        # Stage 3: Figure processing (reuse existing _run_stage4)
        if self.config.include_figures:
            self._run_stage4(document, result)

        # Recalculate stats
        result.recalculate_stats()

        # Add engine attribution to metadata
        result.metadata["ocr_engines"] = list({r.engine for r in result.pages if r.engine})
        result.metadata["processed"] = datetime.utcnow().isoformat()

        # Print summary
        elapsed = time.time() - self._start_time
        result.stats.total_time = elapsed

        self.console.print_summary(
            pages_success=result.stats.pages_success,
            pages_total=result.stats.total_pages,
            figures_count=result.stats.figures_detected,
            time_seconds=elapsed,
            cost=result.stats.total_cost,
            engines_used=result.stats.engines_used,
            output_path=str(default_output_file),
        )

        return result

    def _select_hpc_engines(self) -> list[EngineType]:
        """Select available engines for HPC mode."""
        available = []

        # Primary: DeepSeek-vLLM
        deepseek_vllm = self.engines.get(EngineType.DEEPSEEK_VLLM)
        if deepseek_vllm and deepseek_vllm.is_available():
            available.append(EngineType.DEEPSEEK_VLLM)
            self.console.console.print(f"  [success]+[/success] [deepseek-vllm]deepseek-vllm[/deepseek-vllm] available")
        else:
            self.console.print_warning("DeepSeek-vLLM not available")

        # Secondary: Nougat (for LaTeX)
        if self.config.hpc.use_nougat:
            nougat = self.engines.get(EngineType.NOUGAT)
            if nougat and nougat.is_available():
                available.append(EngineType.NOUGAT)
                self.console.console.print(f"  [success]+[/success] [nougat]nougat[/nougat] available")
            else:
                self.console.print_warning("Nougat not available (LaTeX support disabled)")

        return available

    def _run_parallel_ocr(
        self,
        document: Document,
    ) -> dict[int, list[EngineOutput]]:
        """Run OCR with multiple engines in parallel.

        Returns:
            Dict mapping page_num to list of EngineOutput from each engine
        """
        self.console.print_stage_header(1, "PARALLEL OCR", "Multi-engine text extraction")

        engine_outputs: dict[int, list[EngineOutput]] = {
            page.page_num: [] for page in document.pages
        }

        # Process with each engine
        for engine_type in self._hpc_engines:
            engine = self.engines[engine_type]
            workers = self.config.parallel_pages

            parallel_msg = f"processing... ({workers} workers)" if workers > 1 else "processing..."
            self.console.print_engine_active(engine.name, parallel_msg)

            with self.progress.stage_progress(
                stage_name="primary",
                engine=engine.name,
                total=document.num_pages,
                description=f"OCR [{engine.name}]",
            ) as ctx:
                if workers <= 1:
                    # Sequential processing
                    for page in document.pages:
                        result = engine.process_image(page.image, page.page_num)
                        if result.status == PageStatus.SUCCESS:
                            engine_outputs[page.page_num].append(EngineOutput(
                                engine=engine.name,
                                text=result.text,
                                confidence=result.confidence or 0.0,
                                processing_time=result.processing_time,
                            ))

                        status = "success" if result.status == PageStatus.SUCCESS else "error"
                        ctx.add_result(
                            item=page.page_num,
                            status=status,
                            message=result.error_message if result.error_message else "",
                            confidence=result.confidence,
                        )
                        ctx.advance()
                else:
                    # Parallel processing
                    def process_page(page):
                        return page.page_num, engine.process_image(page.image, page.page_num)

                    with ThreadPoolExecutor(max_workers=workers) as executor:
                        futures = {executor.submit(process_page, page): page for page in document.pages}
                        for future in as_completed(futures):
                            page_num, result = future.result()
                            if result.status == PageStatus.SUCCESS:
                                engine_outputs[page_num].append(EngineOutput(
                                    engine=engine.name,
                                    text=result.text,
                                    confidence=result.confidence or 0.0,
                                    processing_time=result.processing_time,
                                ))

                            status = "success" if result.status == PageStatus.SUCCESS else "error"
                            ctx.add_result(
                                item=page_num,
                                status=status,
                                message=result.error_message if result.error_message else "",
                                confidence=result.confidence,
                            )
                            ctx.advance()

                ctx.print_results()

        return engine_outputs

    def _run_reconciliation(
        self,
        engine_outputs: dict[int, list[EngineOutput]],
        total_pages: int,
    ) -> list[PageResult]:
        """Reconcile outputs from multiple engines.

        Args:
            engine_outputs: Dict mapping page_num to list of EngineOutput
            total_pages: Total number of pages

        Returns:
            List of reconciled PageResult
        """
        self.console.print_stage_header(2, "RECONCILIATION", "Merge multi-engine outputs")

        results: list[PageResult] = []
        latex_merged = 0

        for page_num in sorted(engine_outputs.keys()):
            outputs = engine_outputs[page_num]

            if not outputs:
                # No successful outputs for this page
                results.append(PageResult(
                    page_num=page_num,
                    status=PageStatus.ERROR,
                    error_message="No engine produced output",
                ))
                continue

            reconciliation = self.reconciler.reconcile(outputs, page_num)
            page_result = create_page_result_from_reconciliation(
                reconciliation,
                page_num,
                processing_time=sum(o.processing_time for o in outputs),
            )
            results.append(page_result)

            if reconciliation.latex_source:
                latex_merged += reconciliation.conflicts_resolved

        # Print reconciliation summary
        success_count = sum(1 for r in results if r.status == PageStatus.SUCCESS)
        self.console.console.print(
            f"  [success]+[/success] {success_count}/{total_pages} pages reconciled"
        )
        if latex_merged > 0:
            self.console.console.print(
                f"  [info]i[/info] {latex_merged} LaTeX blocks merged from Nougat"
            )

        return results

    def save_output(self, result: OCRResult, output_path: Path | None = None) -> Path:
        """Save OCR result with HPC metadata in YAML frontmatter."""
        # Use parent implementation
        saved_path = super().save_output(result, output_path)

        # If markdown format, prepend YAML frontmatter
        if self.config.output_format == "markdown":
            content = saved_path.read_text()

            # Build frontmatter
            frontmatter_lines = [
                "---",
                f"source: {Path(result.document_path).name}",
            ]

            engines = result.metadata.get("ocr_engines", [])
            if engines:
                frontmatter_lines.append(f"ocr_engines: [{', '.join(engines)}]")

            processed = result.metadata.get("processed", "")
            if processed:
                frontmatter_lines.append(f"processed: {processed}")

            if result.metadata.get("hpc_mode"):
                frontmatter_lines.append("hpc_mode: true")

            frontmatter_lines.append("---")
            frontmatter_lines.append("")

            # Prepend frontmatter to content
            new_content = "\n".join(frontmatter_lines) + content
            saved_path.write_text(new_content)

        return saved_path
