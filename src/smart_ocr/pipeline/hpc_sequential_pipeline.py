"""HPC Sequential Pipeline for single-GPU setups.

Processes documents in phases, swapping models between each phase:
1. OCR Phase: Load DeepSeek-OCR -> extract text from all pages -> stop
2. Figure Phase: Load vision model -> describe figures -> stop
3. (Optional) Nougat Phase: Run Nougat for LaTeX equations

This avoids OOM issues when running multiple large models on a single GPU.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from PIL import Image

from smart_ocr.core.config import AgentConfig, EngineType
from smart_ocr.core.document import Document
from smart_ocr.core.result import FigureResult, OCRResult, PageResult, PageStatus
from smart_ocr.engines.deepseek_vllm import DeepSeekVLLMEngine
from smart_ocr.engines.nougat import NougatEngine
from smart_ocr.engines.vllm import VLLMEngine
from smart_ocr.engines.vllm_manager import ServerConfig, VLLMServerManager
from smart_ocr.pipeline.processor import OCRPipeline
from smart_ocr.pipeline.reconciler import (
    EngineOutput,
    OutputReconciler,
    create_page_result_from_reconciliation,
)


class HPCSequentialPipeline(OCRPipeline):
    """HPC pipeline with sequential model loading for single-GPU setups.

    This pipeline swaps models between processing phases to fit within
    single-GPU memory constraints. Each phase:
    1. Starts vLLM with the required model
    2. Processes all items for that phase
    3. Stops vLLM and frees GPU memory

    Designed for HPC nodes with single high-memory GPUs (H100, A100).
    """

    def __init__(self, config: AgentConfig | None = None) -> None:
        """Initialize sequential HPC pipeline."""
        super().__init__(config)

        # Server manager for controlling vLLM lifecycle
        self.server_manager = VLLMServerManager(verbose=self.config.verbose)

        # Initialize reconciler
        self.reconciler = OutputReconciler(
            use_llm_reconciler=self.config.hpc.use_llm_reconciler,
            reconciler_model=self.config.hpc.reconciler_model,
            vllm_url=self.config.hpc.vllm_url,
        )

        # Cache for page images (avoid re-rendering between phases)
        self._page_images: dict[int, Image.Image] = {}

    def process(self, pdf_path: Path | str, output_path: Path | str | None = None) -> OCRResult:
        """Process a document through sequential HPC pipeline.

        Sequential pipeline phases:
        1. OCR Phase: DeepSeek-vLLM extracts text
        2. Nougat Phase (optional): Nougat extracts LaTeX
        3. Figure Phase: Vision model describes figures
        4. Reconciliation: Merge outputs

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
        self.console.console.print(
            "[info]HPC Sequential Mode[/info] - Single-GPU model swapping\n"
        )

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

        # Initialize result
        result = OCRResult(document_path=str(pdf_path))
        default_output_file = self._default_output_file(pdf_path)
        result.metadata.update({
            "doc_type": document.doc_type.value,
            "detected_features": document.detected_features,
            "default_output_file": str(default_output_file),
            "hpc_mode": True,
            "hpc_sequential": True,
            "vllm_url": self.config.hpc.vllm_url,
        })

        # Cache page images for reuse across phases
        self._cache_page_images(document)

        try:
            # Phase 1: OCR with DeepSeek-vLLM
            ocr_outputs = self._run_ocr_phase(document)

            # Phase 2: Nougat for LaTeX (optional, runs locally without vLLM)
            nougat_outputs = {}
            if self.config.hpc.use_nougat:
                nougat_outputs = self._run_nougat_phase(document)

            # Phase 3: Reconciliation
            page_results = self._run_reconciliation_phase(
                ocr_outputs, nougat_outputs, document.num_pages
            )
            for r in page_results:
                result.add_page_result(r)

            # Phase 4: Figure processing with vision model
            if self.config.include_figures:
                self._run_figure_phase(document, result)

        finally:
            # Ensure server is stopped
            self.server_manager.stop()
            # Clear cached images
            self._page_images.clear()

        # Recalculate stats
        result.recalculate_stats()

        # Add engine attribution
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

    def _cache_page_images(self, document: Document) -> None:
        """Cache page images to avoid re-rendering between phases."""
        self.console.console.print("[dim]Caching page images...[/dim]")
        for page in document.pages:
            self._page_images[page.page_num] = page.image

    def _run_ocr_phase(self, document: Document) -> dict[int, EngineOutput]:
        """Run OCR phase with DeepSeek-vLLM.

        Returns:
            Dict mapping page_num to EngineOutput
        """
        self.console.print_stage_header(1, "OCR PHASE", "DeepSeek-vLLM text extraction")

        # Start vLLM with OCR model
        if self.config.hpc.manage_server:
            self.console.console.print(
                f"[dim]Starting vLLM with {self.config.hpc.ocr_model}...[/dim]"
            )
            server_config = ServerConfig(
                model=self.config.hpc.ocr_model,
                port=self.config.hpc.vllm_port,
                gpu_memory_utilization=self.config.hpc.gpu_memory_utilization,
                max_model_len=self.config.hpc.max_model_len,
            )
            self.server_manager.start(
                server_config,
                timeout=self.config.hpc.server_startup_timeout,
            )
            base_url = self.server_manager.get_base_url()
        else:
            base_url = self.config.hpc.vllm_url

        # Configure and initialize engine
        self.config.deepseek_vllm.base_url = base_url
        self.config.deepseek_vllm.model = self.config.hpc.ocr_model
        engine = DeepSeekVLLMEngine(self.config.deepseek_vllm)

        if not engine.initialize():
            raise RuntimeError(f"Failed to initialize DeepSeek-vLLM at {base_url}")

        outputs: dict[int, EngineOutput] = {}
        workers = self.config.parallel_pages

        self.console.print_engine_active(
            "deepseek-vllm",
            f"processing {document.num_pages} pages ({workers} workers)",
        )

        with self.progress.stage_progress(
            stage_name="primary",
            engine="deepseek-vllm",
            total=document.num_pages,
            description="OCR extraction",
        ) as ctx:
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

                    status = "success" if result.status == PageStatus.SUCCESS else "error"
                    ctx.add_result(
                        item=page_num,
                        status=status,
                        message=result.error_message or "",
                        confidence=result.confidence,
                    )
                    ctx.advance()
            else:
                def process_page(page_num: int, image: Image.Image):
                    return page_num, engine.process_image(image, page_num)

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

                        status = "success" if result.status == PageStatus.SUCCESS else "error"
                        ctx.add_result(
                            item=page_num,
                            status=status,
                            message=result.error_message or "",
                            confidence=result.confidence,
                        )
                        ctx.advance()

            ctx.print_results()

        # Stop server to free GPU memory
        if self.config.hpc.manage_server:
            self.console.console.print("[dim]Stopping OCR server...[/dim]")
            self.server_manager.stop()

        return outputs

    def _run_nougat_phase(self, document: Document) -> dict[int, EngineOutput]:
        """Run Nougat phase for LaTeX extraction.

        Nougat runs locally (no vLLM), so this can run independently.

        Returns:
            Dict mapping page_num to EngineOutput
        """
        self.console.print_stage_header(2, "NOUGAT PHASE", "LaTeX equation extraction")

        engine = self.engines.get(EngineType.NOUGAT)
        if not engine or not engine.is_available():
            self.console.print_warning("Nougat not available, skipping LaTeX extraction")
            return {}

        outputs: dict[int, EngineOutput] = {}

        self.console.print_engine_active("nougat", f"processing {document.num_pages} pages")

        with self.progress.stage_progress(
            stage_name="primary",
            engine="nougat",
            total=document.num_pages,
            description="LaTeX extraction",
        ) as ctx:
            for page_num, image in self._page_images.items():
                result = engine.process_image(image, page_num)
                if result.status == PageStatus.SUCCESS:
                    outputs[page_num] = EngineOutput(
                        engine="nougat",
                        text=result.text,
                        confidence=result.confidence or 0.8,
                        processing_time=result.processing_time,
                    )

                status = "success" if result.status == PageStatus.SUCCESS else "error"
                ctx.add_result(
                    item=page_num,
                    status=status,
                    message=result.error_message or "",
                    confidence=result.confidence,
                )
                ctx.advance()

            ctx.print_results()

        return outputs

    def _run_reconciliation_phase(
        self,
        ocr_outputs: dict[int, EngineOutput],
        nougat_outputs: dict[int, EngineOutput],
        total_pages: int,
    ) -> list[PageResult]:
        """Reconcile outputs from OCR and Nougat phases."""
        self.console.print_stage_header(3, "RECONCILIATION", "Merge multi-engine outputs")

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
                reconciliation,
                page_num,
                processing_time=sum(o.processing_time for o in outputs),
            )
            results.append(page_result)

            if reconciliation.latex_source:
                latex_merged += reconciliation.conflicts_resolved

        success_count = sum(1 for r in results if r.status == PageStatus.SUCCESS)
        self.console.console.print(
            f"  [success]+[/success] {success_count}/{total_pages} pages reconciled"
        )
        if latex_merged > 0:
            self.console.console.print(
                f"  [info]i[/info] {latex_merged} LaTeX blocks merged from Nougat"
            )

        return results

    def _run_figure_phase(self, document: Document, result: OCRResult) -> None:
        """Run figure description phase with vision model."""
        self.console.print_stage_header(4, "FIGURE PHASE", "Vision model figure description")

        # Extract figures first (no GPU needed)
        pending_figures = self._extract_figures(document, result)

        if not pending_figures:
            self.console.console.print("[dim]No figures detected[/dim]")
            return

        self.console.console.print(
            f"[dim]Extracted {len(pending_figures)} figures, starting vision model...[/dim]"
        )

        # Start vLLM with vision model
        if self.config.hpc.manage_server:
            server_config = ServerConfig(
                model=self.config.hpc.vision_model,
                port=self.config.hpc.vllm_port,
                gpu_memory_utilization=self.config.hpc.gpu_memory_utilization,
                max_model_len=self.config.hpc.max_model_len,
            )
            self.server_manager.start(
                server_config,
                timeout=self.config.hpc.server_startup_timeout,
            )
            base_url = self.server_manager.get_base_url()
        else:
            base_url = self.config.hpc.vllm_url

        # Configure vision engine
        self.config.vllm.base_url = base_url
        self.config.vllm.model = self.config.hpc.vision_model
        engine = VLLMEngine(self.config.vllm)

        if not engine.initialize():
            self.console.print_warning(
                f"Failed to initialize vision model at {base_url}"
            )
            if self.config.hpc.manage_server:
                self.server_manager.stop()
            return

        self.console.print_engine_active(
            "vllm",
            f"describing {len(pending_figures)} figures",
        )

        # Describe figures
        for fig_num, page_num, image, context, fig_path in pending_figures:
            fig_result = engine.describe_figure(image, context=context)
            fig_result.figure_num = fig_num
            fig_result.page_num = page_num
            if fig_path:
                fig_result.image_path = fig_path

            page_result = result.get_page(page_num)
            if page_result:
                page_result.figures.append(fig_result)

            self.console.print_figure_result(
                figure_num=fig_num,
                page=page_num,
                fig_type=fig_result.figure_type,
                description=fig_result.description,
            )

        # Stop server
        if self.config.hpc.manage_server:
            self.console.console.print("[dim]Stopping vision server...[/dim]")
            self.server_manager.stop()

    def _extract_figures(
        self,
        document: Document,
        result: OCRResult,
    ) -> list[tuple[int, int, Image.Image, str, str | None]]:
        """Extract figures from document without GPU.

        Returns:
            List of (fig_num, page_num, image, context, fig_path) tuples
        """
        try:
            import fitz
        except ImportError:
            self.console.print_warning("PyMuPDF not available for figure extraction")
            return []

        pending_figures: list[tuple[int, int, Image.Image, str, str | None]] = []
        figure_counter = 1
        max_dim = 1024
        min_area = 80 * 80
        render_dpi = 150

        # Prepare figures directory
        figures_dir: Path | None = None
        if self.config.save_figures:
            if self._custom_output_path:
                figures_dir = self._custom_output_path.parent / "figures"
            else:
                doc_stem = Path(document.path).stem
                figures_dir = self.config.output_dir / doc_stem / "figures"
            figures_dir.mkdir(parents=True, exist_ok=True)

        try:
            with fitz.open(document.path) as pdf:
                for page_index in range(len(pdf)):
                    if figure_counter > self.config.figures_max_total:
                        break

                    page = pdf[page_index]
                    page_num = page_index + 1
                    page_result = result.get_page(page_num)

                    context_text = ""
                    if page_result:
                        context_text = (page_result.text or "")[
                            : self.config.figures_context_max_chars
                        ]

                    per_page = 0

                    # Extract embedded images
                    images = page.get_images(full=True)
                    for img in images:
                        if (
                            figure_counter > self.config.figures_max_total
                            or per_page >= self.config.figures_max_per_page
                        ):
                            break

                        xref = img[0]
                        width, height = img[2], img[3]
                        area = width * height
                        aspect = width / max(height, 1)

                        if area < min_area or aspect > 8 or aspect < 0.125:
                            continue

                        try:
                            pix = fitz.Pixmap(pdf, xref)
                            if pix.colorspace is None:
                                continue

                            if pix.colorspace != fitz.csRGB or pix.alpha or pix.n != 3:
                                rgb = fitz.Pixmap(fitz.csRGB, pix)
                                pix = rgb

                            pil_img = Image.frombytes(
                                "RGB", (pix.width, pix.height), pix.samples
                            )
                        except Exception:
                            continue

                        if max(pil_img.size) > max_dim:
                            pil_img.thumbnail((max_dim, max_dim))

                        fig_path: str | None = None
                        if figures_dir:
                            fig_filename = f"figure_{figure_counter}_page{page_num}.png"
                            fig_path = str(figures_dir / fig_filename)
                            pil_img.save(fig_path)

                        pending_figures.append(
                            (figure_counter, page_num, pil_img, context_text, fig_path)
                        )
                        figure_counter += 1
                        per_page += 1

        except Exception as e:
            self.console.print_warning(f"Figure extraction error: {e}")

        return pending_figures

    def save_output(self, result: OCRResult, output_path: Path | None = None) -> Path:
        """Save OCR result with HPC sequential metadata."""
        saved_path = super().save_output(result, output_path)

        # Add YAML frontmatter for markdown
        if self.config.output_format == "markdown":
            content = saved_path.read_text()

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

            frontmatter_lines.append("hpc_mode: sequential")
            frontmatter_lines.append("---")
            frontmatter_lines.append("")

            new_content = "\n".join(frontmatter_lines) + content
            saved_path.write_text(new_content)

        return saved_path
