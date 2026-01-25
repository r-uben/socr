"""Main OCR pipeline orchestrator."""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.table import Table

from smart_ocr.audit.heuristics import HeuristicsChecker
from smart_ocr.audit.llm_audit import LLMAuditor
from smart_ocr.core.config import AgentConfig, EngineType
from smart_ocr.core.document import Document, DocumentType
from smart_ocr.core.result import OCRResult, PageResult, PageStatus, ProcessingStats
from smart_ocr.pipeline.router import EngineRouter
from smart_ocr.engines.base import BaseEngine
from smart_ocr.engines.deepseek import DeepSeekEngine
from smart_ocr.engines.deepseek_vllm import DeepSeekVLLMEngine
from smart_ocr.engines.gemini import GeminiEngine
from smart_ocr.engines.mistral import MistralEngine
from smart_ocr.engines.nougat import NougatEngine
from smart_ocr.engines.vllm import VLLMEngine
from smart_ocr.ui.console import AgentConsole
from smart_ocr.ui.panels import AuditPanel, StagePanel, SummaryPanel
from smart_ocr.ui.progress import AgentProgress
from smart_ocr.ui.theme import AGENT_THEME


class OCRPipeline:
    """Multi-agent OCR pipeline with cascading fallback."""

    def __init__(self, config: AgentConfig | None = None) -> None:
        self.config = config or AgentConfig()
        self.console = AgentConsole(verbose=self.config.verbose)
        self.progress = AgentProgress(self.console.console)

        # Initialize engines
        self.engines: dict[EngineType, BaseEngine] = {
            EngineType.NOUGAT: NougatEngine(self.config.nougat),
            EngineType.DEEPSEEK: DeepSeekEngine(self.config.deepseek),
            EngineType.DEEPSEEK_VLLM: DeepSeekVLLMEngine(self.config.deepseek_vllm),
            EngineType.MISTRAL: MistralEngine(self.config.mistral),
            EngineType.GEMINI: GeminiEngine(self.config.gemini),
            EngineType.VLLM: VLLMEngine(self.config.vllm),
        }

        # Initialize audit components
        self.heuristics = HeuristicsChecker(
            min_word_count=self.config.audit.min_word_count,
        )
        self.llm_auditor = LLMAuditor(
            model=self.config.audit.model.value,
            ollama_host=self.config.audit.ollama_host,
        ) if self.config.audit.enabled else None

        self._start_time = 0.0
        self._custom_output_path: Path | None = None
        self.router = EngineRouter(self.config, self.engines)

    def process(self, pdf_path: Path | str, output_path: Path | str | None = None) -> OCRResult:
        """Process a document through the full pipeline.

        Args:
            pdf_path: Path to the PDF file to process.
            output_path: Optional custom output path. If provided, figures will be saved
                         alongside this file instead of in the default output directory.
        """
        self._start_time = time.time()
        pdf_path = Path(pdf_path)
        self._custom_output_path = Path(output_path) if output_path else None

        # Print header
        self.console.print_header()

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
        })

        # Stage 1: Primary OCR
        primary_engine = self.router.select_primary(
            document.doc_type,
            warn=self.console.print_warning,
        )
        stage1_results = self._run_stage1(document, primary_engine)
        for r in stage1_results:
            result.add_page_result(r)

        # Stage 2: Quality Audit
        pages_to_reprocess = self._run_stage2(document, result, primary_engine)

        # Stage 3: Fallback OCR (if needed)
        if pages_to_reprocess:
            fallback_results = self._run_stage3(document, pages_to_reprocess, primary_engine)
            for r in fallback_results:
                result.add_page_result(r)

        # Stage 4: Figure Processing (if enabled)
        if self.config.include_figures:
            self._run_stage4(document, result)

        # Recalculate stats (handles page replacements correctly)
        result.recalculate_stats()

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

    def _run_stage1(
        self,
        document: Document,
        engine_type: EngineType,
    ) -> list[PageResult]:
        """Run primary OCR stage with optional parallelization."""
        self.console.print_stage_header(1, "PRIMARY OCR", "Extract text from pages")

        engine = self.engines[engine_type]
        workers = self.config.parallel_pages
        parallel_msg = f"processing... ({workers} workers)" if workers > 1 else "processing..."
        self.console.print_engine_active(engine.name, parallel_msg)

        results: list[PageResult] = []

        with self.progress.stage_progress(
            stage_name="primary",
            engine=engine.name,
            total=document.num_pages,
            description="Extracting text",
        ) as ctx:
            if workers <= 1:
                # Sequential processing
                for page in document.pages:
                    result = engine.process_image(page.image, page.page_num)
                    results.append(result)

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
                page_results: dict[int, PageResult] = {}

                def process_page(page):
                    return page.page_num, engine.process_image(page.image, page.page_num)

                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = {executor.submit(process_page, page): page for page in document.pages}
                    for future in as_completed(futures):
                        page_num, result = future.result()
                        page_results[page_num] = result

                        status = "success" if result.status == PageStatus.SUCCESS else "error"
                        ctx.add_result(
                            item=page_num,
                            status=status,
                            message=result.error_message if result.error_message else "",
                            confidence=result.confidence,
                        )
                        ctx.advance()

                # Sort by page number to maintain order
                results = [page_results[i] for i in sorted(page_results.keys())]

            ctx.print_results()

        if engine.capabilities.cost_per_page > 0:
            total_cost = engine.capabilities.cost_per_page * len(results)
            self.console.print_cost(total_cost, engine.name)

        return results

    def _run_stage2(self, document: Document, result: OCRResult, primary_engine: EngineType) -> list[int]:
        """Run quality audit stage."""
        self.console.print_stage_header(2, "QUALITY AUDIT", "Validate OCR output")

        pages_to_reprocess = []
        audit_panel = AuditPanel()

        # Run heuristics on each page
        self.console.print_engine_active("ollama", "Running quality checks...")

        for page_result in result.pages:
            if page_result.status != PageStatus.SUCCESS:
                pages_to_reprocess.append(page_result.page_num)
                continue

            # Heuristic checks
            heuristics_result = self.heuristics.check(page_result.text)

            for metric in heuristics_result.metrics:
                audit_panel.add_metric(
                    name=f"Page {page_result.page_num} - {metric.name}",
                    value=str(metric.value),
                    threshold=str(metric.threshold) if metric.threshold else None,
                    passed=metric.passed,
                )

            if not heuristics_result.passed:
                page_result.audit_passed = False
                page_result.audit_notes.extend(heuristics_result.errors)
                pages_to_reprocess.append(page_result.page_num)

        # Optional cross-engine spot check for flagged pages (local-only)
        cross_engine = self.router.select_cross_check(primary_engine) if self.config.audit.cross_check_enabled else None
        if cross_engine and pages_to_reprocess:
            engine = self.engines[cross_engine]
            self.console.print_engine_active(engine.name, "cross-checking flagged pages")
            for page_num in list(pages_to_reprocess)[: self.config.audit.cross_check_pages]:
                page = document.get_page(page_num)
                if not page:
                    continue
                alt_result = engine.process_image(page.image, page_num)
                if alt_result.status != PageStatus.SUCCESS:
                    continue
                heuristics_result = self.heuristics.check(alt_result.text)
                if heuristics_result.passed:
                    alt_result.audit_notes.append(f"Replaced via cross-check ({engine.name})")
                    result.add_page_result(alt_result)
                    pages_to_reprocess.remove(page_num)

        # Run LLM audit on suspicious pages (if enabled and available)
        if self.llm_auditor and pages_to_reprocess:
            if not self.llm_auditor.is_available():
                self.console.print_warning("LLM audit skipped (Ollama/model not available)")
                self.llm_auditor.close()
                self.llm_auditor = None
                return pages_to_reprocess
            self.console.print_engine_active("ollama", "LLM review of flagged pages...")

            for page_num in pages_to_reprocess[:3]:  # Limit to first 3
                page_result = result.get_page(page_num)
                if page_result and page_result.text:
                    llm_result = self.llm_auditor.audit(page_result.text)

                    audit_panel.add_llm_review(
                        item=f"Page {page_num}",
                        verdict=llm_result.verdict,
                        reason=llm_result.reasoning[:100] if llm_result.reasoning else "",
                    )

                    if llm_result.verdict == "acceptable":
                        # LLM overrides heuristics
                        pages_to_reprocess.remove(page_num)
                        page_result.audit_passed = True

        # Print audit results
        if audit_panel.metrics or audit_panel.llm_results:
            self.console.console.print(audit_panel.render())

        if pages_to_reprocess:
            self.console.print_warning(
                f"{len(pages_to_reprocess)} pages flagged for reprocessing"
            )
        else:
            self.console.console.print("[success]✓ All pages passed quality checks[/success]")

        return pages_to_reprocess

    def _run_stage3(
        self,
        document: Document,
        pages_to_reprocess: list[int],
        primary_engine: EngineType,
    ) -> list[PageResult]:
        """Run fallback OCR on failed pages with optional parallelization."""
        self.console.print_stage_header(3, "FALLBACK OCR", "Reprocess failed pages")

        # Select fallback engine (different from primary)
        fallback_engine = self.router.select_fallback(
            primary_engine,
            warn=self.console.print_warning,
        )
        if not fallback_engine:
            self.console.print_warning("No fallback engine available; leaving flagged pages as-is")
            return []
        engine = self.engines[fallback_engine]

        workers = self.config.parallel_pages
        parallel_msg = f"reprocessing {len(pages_to_reprocess)} pages ({workers} workers)" if workers > 1 else f"reprocessing {len(pages_to_reprocess)} pages"
        self.console.print_engine_active(engine.name, parallel_msg)

        results: list[PageResult] = []

        with self.progress.stage_progress(
            stage_name="fallback",
            engine=engine.name,
            total=len(pages_to_reprocess),
            description="Reprocessing",
        ) as ctx:
            if workers <= 1:
                # Sequential processing
                for page_num in pages_to_reprocess:
                    page = document.get_page(page_num)
                    if not page:
                        continue

                    result = engine.process_image(page.image, page_num)
                    results.append(result)

                    status = "success" if result.status == PageStatus.SUCCESS else "error"
                    ctx.add_result(
                        item=page_num,
                        status=status,
                        message=result.error_message if result.error_message else "",
                        confidence=result.confidence,
                    )
                    ctx.advance()
            else:
                # Parallel processing
                page_results: dict[int, PageResult] = {}

                def process_page(page_num: int):
                    page = document.get_page(page_num)
                    if not page:
                        return None
                    return page_num, engine.process_image(page.image, page_num)

                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = {executor.submit(process_page, pn): pn for pn in pages_to_reprocess}
                    for future in as_completed(futures):
                        result_tuple = future.result()
                        if result_tuple is None:
                            continue
                        page_num, result = result_tuple
                        page_results[page_num] = result

                        status = "success" if result.status == PageStatus.SUCCESS else "error"
                        ctx.add_result(
                            item=page_num,
                            status=status,
                            message=result.error_message if result.error_message else "",
                            confidence=result.confidence,
                        )
                        ctx.advance()

                # Sort by page number to maintain order
                results = [page_results[i] for i in sorted(page_results.keys())]

            ctx.print_results()

        if engine.capabilities.cost_per_page > 0:
            total_cost = engine.capabilities.cost_per_page * len(results)
            self.console.print_cost(total_cost, engine.name)

        return results

    def _run_stage4(self, document: Document, result: OCRResult) -> None:
        """Run figure detection and description stage with optional parallelization."""
        self.console.print_stage_header(4, "FIGURE PROCESSING", "Describe figures and charts")
        self.console.print_warning("Figure processing is experimental; extracting images from PDF.")

        # Select engine with figure capabilities
        figure_engine = None

        # If user specified an engine, try that first
        if self.config.use_figures_engine_override:
            engine = self.engines[self.config.figures_engine]
            if engine.capabilities.supports_figures and engine.is_available():
                figure_engine = engine
            else:
                self.console.print_warning(
                    f"Requested figure engine '{self.config.figures_engine.value}' not available, "
                    "falling back to auto-select"
                )

        # Auto-select if no override or override failed
        # Prefer local engines (vLLM, DeepSeek) over cloud (Gemini, Mistral)
        if not figure_engine:
            for engine_type in [EngineType.VLLM, EngineType.GEMINI, EngineType.DEEPSEEK, EngineType.MISTRAL]:
                if not self.config.get_engine_config(engine_type).enabled:
                    continue
                engine = self.engines[engine_type]
                if engine.capabilities.supports_figures and engine.is_available():
                    figure_engine = engine
                    break

        if not figure_engine:
            self.console.print_warning("No figure-capable engine available")
            return

        workers = self.config.parallel_figures
        parallel_msg = f"extracting and describing figures ({workers} workers)" if workers > 1 else "describing figures"
        self.console.print_engine_active(figure_engine.name, parallel_msg)

        try:
            import fitz  # PyMuPDF
            from PIL import Image
        except ImportError:
            self.console.print_warning("Figure detection skipped (PyMuPDF/Pillow not available)")
            return

        # Phase 1: Extract all figures first (fast, no API calls)
        pending_figures: list[tuple[int, int, Image.Image, str, str | None]] = []  # (fig_num, page_num, image, context, fig_path)

        figure_counter = 1
        max_dim = 1024
        min_area = 80 * 80
        render_dpi = 150
        min_drawings_for_vector_figure = 5
        min_vector_figure_area_ratio = 0.05
        max_vector_figure_area_ratio = 0.85
        header_footer_margin = 0.1

        # Prepare figures directory if saving is enabled
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
                    if not page_result:
                        continue

                    context_text = (page_result.text or "")[: self.config.figures_context_max_chars]

                    per_page = 0
                    processed_regions: set[tuple[int, int, int, int]] = set()

                    is_landscape = page.rect.width > page.rect.height
                    effective_min_area_ratio = min_vector_figure_area_ratio * 0.5 if is_landscape else min_vector_figure_area_ratio
                    effective_max_area_ratio = 0.98 if is_landscape else max_vector_figure_area_ratio
                    effective_min_drawings = 3 if is_landscape else min_drawings_for_vector_figure

                    # Strategy 0: Detect vector figures
                    try:
                        drawings = page.get_drawings()
                        page_width = page.rect.width
                        page_height = page.rect.height
                        page_area = page_width * page_height

                        if len(drawings) >= effective_min_drawings:
                            figure_regions = self._cluster_drawings_into_figures(
                                drawings, page_width, page_height, cluster_gap=30,
                            )

                            for region_drawings, bbox in figure_regions:
                                if figure_counter > self.config.figures_max_total:
                                    break
                                if per_page >= self.config.figures_max_per_page:
                                    break

                                x0, y0, x1, y1 = bbox
                                width = x1 - x0
                                height = y1 - y0
                                area = width * height
                                area_ratio = area / page_area

                                if area < min_area or width < 50 or height < 50:
                                    continue
                                if area_ratio < effective_min_area_ratio or area_ratio > effective_max_area_ratio:
                                    continue
                                if len(region_drawings) < effective_min_drawings:
                                    continue

                                if not is_landscape:
                                    center_y = (y0 + y1) / 2
                                    in_header = center_y < page_height * header_footer_margin
                                    in_footer = center_y > page_height * (1 - header_footer_margin)
                                    if (in_header or in_footer) and len(region_drawings) < 20:
                                        continue

                                region_key = (int(x0), int(y0), int(x1), int(y1))
                                if region_key in processed_regions:
                                    continue
                                processed_regions.add(region_key)

                                padding = 10
                                clip = fitz.Rect(
                                    max(0, x0 - padding), max(0, y0 - padding),
                                    min(page_width, x1 + padding), min(page_height, y1 + padding)
                                )

                                mat = fitz.Matrix(render_dpi / 72, render_dpi / 72)
                                try:
                                    pix = page.get_pixmap(matrix=mat, clip=clip)
                                    pil_img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                                    if max(pil_img.size) > max_dim:
                                        pil_img.thumbnail((max_dim, max_dim))

                                    fig_path: str | None = None
                                    if figures_dir:
                                        fig_filename = f"figure_{figure_counter}_page{page_num}.png"
                                        fig_path = str(figures_dir / fig_filename)
                                        pil_img.save(fig_path)

                                    pending_figures.append((figure_counter, page_num, pil_img, context_text or "", fig_path))
                                    figure_counter += 1
                                    per_page += 1
                                except Exception:
                                    pass

                            # Fallback for presentations
                            if is_landscape and per_page == 0 and len(drawings) >= 10:
                                margin_x = page_width * 0.05
                                margin_top = page_height * 0.15
                                margin_bottom = page_height * 0.10
                                clip = fitz.Rect(margin_x, margin_top, page_width - margin_x, page_height - margin_bottom)

                                mat = fitz.Matrix(render_dpi / 72, render_dpi / 72)
                                try:
                                    pix = page.get_pixmap(matrix=mat, clip=clip)
                                    pil_img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                                    if max(pil_img.size) > max_dim:
                                        pil_img.thumbnail((max_dim, max_dim))

                                    fig_path: str | None = None
                                    if figures_dir:
                                        fig_filename = f"figure_{figure_counter}_page{page_num}.png"
                                        fig_path = str(figures_dir / fig_filename)
                                        pil_img.save(fig_path)

                                    pending_figures.append((figure_counter, page_num, pil_img, context_text or "", fig_path))
                                    figure_counter += 1
                                    per_page += 1
                                except Exception:
                                    pass
                    except Exception:
                        pass

                    # Strategy 1: Extract IMAGE blocks
                    try:
                        text_dict = page.get_text("dict")
                        for block in text_dict.get("blocks", []):
                            if figure_counter > self.config.figures_max_total or per_page >= self.config.figures_max_per_page:
                                break
                            if block.get("type") != 1:
                                continue

                            bbox = block.get("bbox")
                            if not bbox:
                                continue

                            x0, y0, x1, y1 = bbox
                            width, height = x1 - x0, y1 - y0
                            area = width * height
                            aspect = width / max(height, 1)

                            if area < min_area or aspect > 8 or aspect < 0.125:
                                continue

                            region_key = (int(x0), int(y0), int(x1), int(y1))
                            if region_key in processed_regions:
                                continue
                            processed_regions.add(region_key)

                            clip = fitz.Rect(bbox)
                            mat = fitz.Matrix(render_dpi / 72, render_dpi / 72)
                            try:
                                pix = page.get_pixmap(matrix=mat, clip=clip)
                                pil_img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                            except Exception:
                                continue

                            if max(pil_img.size) > max_dim:
                                pil_img.thumbnail((max_dim, max_dim))

                            fig_path: str | None = None
                            if figures_dir:
                                fig_filename = f"figure_{figure_counter}_page{page_num}.png"
                                fig_path = str(figures_dir / fig_filename)
                                pil_img.save(fig_path)

                            pending_figures.append((figure_counter, page_num, pil_img, context_text or "", fig_path))
                            figure_counter += 1
                            per_page += 1
                    except Exception:
                        pass

                    # Strategy 2: Extract raw embedded images
                    images = page.get_images(full=True)
                    for img in images:
                        if figure_counter > self.config.figures_max_total or per_page >= self.config.figures_max_per_page:
                            break

                        xref = img[0]
                        width, height = img[2], img[3]
                        area = width * height
                        aspect = width / max(height, 1)
                        if area < min_area or aspect > 8 or aspect < 0.125:
                            continue

                        try:
                            raw_image = pdf.extract_image(xref)
                            if len(raw_image.get("image", b"")) < 5000:
                                continue
                        except Exception:
                            pass

                        pix = None
                        rgb = None
                        try:
                            pix = fitz.Pixmap(pdf, xref)
                            if pix.colorspace is None:
                                continue

                            if pix.colorspace != fitz.csRGB or pix.alpha or pix.n != 3:
                                rgb = fitz.Pixmap(fitz.csRGB, pix)
                                pix = rgb

                            pil_img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                        except Exception:
                            continue
                        finally:
                            del rgb
                            del pix

                        if max(pil_img.size) > max_dim:
                            pil_img.thumbnail((max_dim, max_dim))

                        fig_path: str | None = None
                        if figures_dir:
                            fig_filename = f"figure_{figure_counter}_page{page_num}.png"
                            fig_path = str(figures_dir / fig_filename)
                            pil_img.save(fig_path)

                        pending_figures.append((figure_counter, page_num, pil_img, context_text or "", fig_path))
                        figure_counter += 1
                        per_page += 1

        except Exception as e:
            self.console.print_warning(f"Could not open/process PDF for figures: {e}")
            return

        if not pending_figures:
            self.console.console.print("[dim]No figures detected[/dim]")
            return

        self.console.console.print(f"[dim]Extracted {len(pending_figures)} figures, describing...[/dim]")

        # Phase 2: Describe figures (parallelized)
        def describe_figure_task(fig_data: tuple[int, int, Image.Image, str, str | None]):
            fig_num, page_num, pil_img, context, fig_path = fig_data
            fig_result = figure_engine.describe_figure(pil_img, context=context)
            fig_result.figure_num = fig_num
            fig_result.page_num = page_num
            if fig_path:
                fig_result.image_path = fig_path
            return fig_result

        if workers <= 1:
            # Sequential description
            for fig_data in pending_figures:
                fig_result = describe_figure_task(fig_data)
                page_result = result.get_page(fig_result.page_num)
                if page_result:
                    page_result.figures.append(fig_result)
                self.console.print_figure_result(
                    figure_num=fig_result.figure_num,
                    page=fig_result.page_num,
                    fig_type=fig_result.figure_type,
                    description=fig_result.description,
                )
        else:
            # Parallel description
            described_figures: list[tuple[int, any]] = []  # (fig_num, fig_result)

            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(describe_figure_task, fig_data): fig_data[0] for fig_data in pending_figures}
                for future in as_completed(futures):
                    try:
                        fig_result = future.result()
                        described_figures.append((fig_result.figure_num, fig_result))

                        self.console.print_figure_result(
                            figure_num=fig_result.figure_num,
                            page=fig_result.page_num,
                            fig_type=fig_result.figure_type,
                            description=fig_result.description,
                        )
                    except Exception as e:
                        fig_num = futures[future]
                        self.console.print_warning(f"Figure {fig_num} description failed: {e}")

            # Sort by figure number and attach to page results
            for _, fig_result in sorted(described_figures, key=lambda x: x[0]):
                page_result = result.get_page(fig_result.page_num)
                if page_result:
                    page_result.figures.append(fig_result)

    def _cluster_drawings_into_figures(
        self,
        drawings: list[dict],
        page_width: float,
        page_height: float,
        cluster_gap: float = 30,
    ) -> list[tuple[list[dict], tuple[float, float, float, float]]]:
        """Cluster drawings into figure regions using spatial proximity.

        Uses a simple union-find approach: drawings whose bounding boxes are within
        `cluster_gap` pixels of each other belong to the same figure.

        Returns:
            List of (drawings_in_cluster, bounding_box) tuples.
        """
        if not drawings:
            return []

        # Extract bounding boxes
        boxes = []
        for d in drawings:
            rect = d.get("rect")
            if rect:
                boxes.append((rect.x0, rect.y0, rect.x1, rect.y1))
            else:
                boxes.append(None)

        # Filter out drawings without valid boxes
        valid = [(i, boxes[i]) for i in range(len(boxes)) if boxes[i] is not None]
        if not valid:
            return []

        # Union-Find structure
        parent = {i: i for i, _ in valid}

        def find(x: int) -> int:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Check proximity between all pairs (O(n^2) but n is typically small)
        for i, (idx_i, box_i) in enumerate(valid):
            for j, (idx_j, box_j) in enumerate(valid):
                if i >= j:
                    continue
                # Compute gap between boxes
                x0_i, y0_i, x1_i, y1_i = box_i
                x0_j, y0_j, x1_j, y1_j = box_j

                # Horizontal gap
                if x1_i < x0_j:
                    h_gap = x0_j - x1_i
                elif x1_j < x0_i:
                    h_gap = x0_i - x1_j
                else:
                    h_gap = 0  # Overlapping horizontally

                # Vertical gap
                if y1_i < y0_j:
                    v_gap = y0_j - y1_i
                elif y1_j < y0_i:
                    v_gap = y0_i - y1_j
                else:
                    v_gap = 0  # Overlapping vertically

                # If close enough, merge clusters
                if h_gap <= cluster_gap and v_gap <= cluster_gap:
                    union(idx_i, idx_j)

        # Group drawings by cluster
        clusters: dict[int, list[int]] = {}
        for idx, _ in valid:
            root = find(idx)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(idx)

        # Compute bounding box for each cluster
        results = []
        for indices in clusters.values():
            cluster_drawings = [drawings[i] for i in indices]
            cluster_boxes = [boxes[i] for i in indices if boxes[i] is not None]

            if not cluster_boxes:
                continue

            x0 = min(b[0] for b in cluster_boxes)
            y0 = min(b[1] for b in cluster_boxes)
            x1 = max(b[2] for b in cluster_boxes)
            y1 = max(b[3] for b in cluster_boxes)

            results.append((cluster_drawings, (x0, y0, x1, y1)))

        # Sort by position (top-to-bottom, left-to-right)
        results.sort(key=lambda r: (r[1][1], r[1][0]))

        return results

    def save_output(self, result: OCRResult, output_path: Path | None = None) -> Path:
        """Save OCR result to file."""
        document_path = Path(result.document_path)
        extension = self._output_extension()

        # Derive output location
        if output_path:
            if output_path.suffix:
                main_file = output_path
                base_dir = output_path.parent
                # Warn if extension doesn't match format
                expected_ext = f".{extension}"
                if output_path.suffix.lower() != expected_ext:
                    self.console.print_warning(
                        f"Output extension '{output_path.suffix}' doesn't match "
                        f"format '{self.config.output_format}' (expected '{expected_ext}')"
                    )
            else:
                base_dir = output_path
                main_file = base_dir / f"{document_path.stem}.{extension}"
        else:
            main_file = self._default_output_file(document_path)
            base_dir = main_file.parent

        base_dir.mkdir(parents=True, exist_ok=True)

        # Serialize main content
        if self.config.output_format == "markdown":
            content = result.to_markdown()
        elif self.config.output_format == "json":
            import json

            content = json.dumps({
                "document": result.document_path,
                "pages": [
                    {
                        "page_num": p.page_num,
                        "text": p.text,
                        "status": p.status.value,
                        "engine": p.engine,
                        "confidence": p.confidence,
                        "figures": [
                            {
                                "figure_num": f.figure_num,
                                "figure_type": f.figure_type,
                                "description": f.description,
                                "bbox": f.bbox,
                                "image_path": f.image_path,
                            }
                            for f in p.figures
                        ] if p.figures else [],
                    }
                    for p in result.pages
                ],
                "stats": {
                    "total_pages": result.stats.total_pages,
                    "pages_success": result.stats.pages_success,
                    "figures_detected": result.stats.figures_detected,
                    "total_cost": result.stats.total_cost,
                    "total_time": result.stats.total_time,
                },
            }, indent=2)
        else:
            content = result.get_full_text()

        main_file.write_text(content)

        # Save run metadata alongside the main output
        metadata = {
            "document": result.document_path,
            "output_file": str(main_file),
            "format": self.config.output_format,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "stats": {
                "total_pages": result.stats.total_pages,
                "pages_success": result.stats.pages_success,
                "pages_warning": result.stats.pages_warning,
                "pages_error": result.stats.pages_error,
                "total_cost": result.stats.total_cost,
                "total_time": result.stats.total_time,
            },
            "engines_used": result.stats.engines_used,
            "figures": result.stats.figures_detected,
            "pages_needing_reprocessing": result.get_pages_needing_reprocessing(),
            "document_metadata": result.metadata,
        }

        import json
        (base_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

        return main_file

    def _output_extension(self) -> str:
        """Map output format to file extension."""
        return {
            "markdown": "md",
            "json": "json",
            "txt": "txt",
        }.get(self.config.output_format, (self.config.output_format or "txt").lstrip("."))

    def _default_output_file(self, document_path: Path) -> Path:
        """Default location: output/<doc_stem>/<doc_stem>.<ext>."""
        doc_stem = document_path.stem
        ext = self._output_extension()
        return self.config.output_dir / doc_stem / f"{doc_stem}.{ext}"
