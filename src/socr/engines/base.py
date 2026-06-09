"""Base engine adapters for socr.

Two engine families:
  - BaseEngine: CLI-based, one subprocess per document (standard mode)
  - BaseHTTPEngine: HTTP API, per-page processing (HPC mode with vLLM)
"""

import logging
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path

from ocr_output_contract import (
    doc_dir_for,
    markdown_path_for,
    relative_key,
    resolve_output_root,
    split_native_pages,
)

from socr.core.config import PipelineConfig
from socr.core.normalizer import OutputNormalizer
from socr.core.result import (
    DocumentStatus,
    EngineResult,
    FailureMode,
    FigureInfo,
    PageOutput,
    PageStatus,
)

logger = logging.getLogger(__name__)

_normalizer = OutputNormalizer()


def sanitize_filename(name: str) -> str:
    """Sanitize a filename for use as a directory name."""
    return "".join(c if c.isalnum() or c in "._- " else "_" for c in name).strip()


class BaseEngine(ABC):
    """Abstract base class for CLI-based OCR engines.

    Each engine wraps a sibling CLI tool (gemini-ocr, nougat-ocr, etc.). Every
    engine now emits the family-wide *canonical* output structure via the shared
    ``ocr-output-contract`` package, so read-back is the same for all of them:

        ``<output_dir>/<rel/dir>/<stem>/<stem>.md``

    where ``<rel/dir>`` mirrors the input's path relative to the scanned root.
    socr passes a single file (the PDF, or a page-image directory) and reads the
    one aggregated ``<stem>.md`` back via the contract's path helpers — no
    per-engine layout special-casing and no ``rglob`` guessing.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Engine identifier (matches EngineType value)."""
        ...

    @property
    @abstractmethod
    def cli_command(self) -> str:
        """The CLI binary name (e.g., 'gemini-ocr', 'nougat-ocr')."""
        ...

    @property
    def model_version(self) -> str:
        """Model version string. Override in subclasses that know their model."""
        return ""

    def resolved_model_version(self, config: PipelineConfig) -> str:
        """The model id ACTUALLY used for this run, for the manifest fingerprint.

        The static :attr:`model_version` property is a hardcoded literal that
        does not track ``--model`` overrides (e.g. gemini hardcodes
        ``gemini-3-flash-preview`` while the CLI is invoked with
        ``config.gemini_model``). The manifest fingerprint must reflect the
        RESOLVED model so a model swap invalidates the cache.

        Resolution: read ``config.<name>_model`` when the engine exposes a
        configurable model attribute; fall back to the static literal otherwise.
        """
        configured = getattr(config, f"{self.name}_model", None)
        if isinstance(configured, str) and configured.strip():
            return configured
        return self.model_version

    def fingerprint_determinants(self, config: PipelineConfig) -> tuple[str, str | None]:
        """Return ``(backend, task)`` for the run fingerprint.

        Mirrors the resolved-model logic: a swap of backend or task must
        invalidate the cache. Engines without a configurable backend report
        ``"socr"`` (the orchestrator's umbrella backend) and ``None`` task.
        """
        backend = getattr(config, f"{self.name}_backend", None)
        task = getattr(config, f"{self.name}_task", None)
        backend_str = backend if isinstance(backend, str) and backend.strip() else "socr"
        task_str = task if isinstance(task, str) and task.strip() else None
        return backend_str, task_str

    def is_available(self) -> bool:
        """Check if the CLI tool is installed and callable."""
        try:
            result = subprocess.run(
                [self.cli_command, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def process_document(
        self,
        pdf_path: Path,
        output_dir: Path,
        config: PipelineConfig,
    ) -> EngineResult:
        """Process a PDF document via CLI subprocess.

        Calls the CLI once on the whole PDF, reads the output markdown.
        Returns EngineResult with a single PageOutput containing the full text.
        """
        start_time = time.time()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_out = Path(tmpdir)
            cmd = self._build_command(pdf_path, tmp_out, config)

            logger.info(f"[{self.name}] Running: {' '.join(cmd)}")

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=config.timeout,
                )

                if result.returncode != 0:
                    stderr = result.stderr.strip() if result.stderr else "Unknown error"
                    logger.error(f"[{self.name}] CLI failed: {stderr}")
                    return EngineResult(
                        document_path=pdf_path,
                        engine=self.name,
                        status=DocumentStatus.ERROR,
                        failure_mode=FailureMode.CLI_ERROR,
                        error=f"CLI exited {result.returncode}: {stderr[:500]}",
                        processing_time=time.time() - start_time,
                        model_version=self.resolved_model_version(config),
                    )

                # Read output markdown
                markdown = self._read_output(pdf_path, tmp_out)
                if markdown is None:
                    return EngineResult(
                        document_path=pdf_path,
                        engine=self.name,
                        status=DocumentStatus.ERROR,
                        failure_mode=FailureMode.EMPTY_OUTPUT,
                        error="CLI produced no output markdown",
                        processing_time=time.time() - start_time,
                        model_version=self.resolved_model_version(config),
                    )

                elapsed = time.time() - start_time
                return EngineResult(
                    document_path=pdf_path,
                    engine=self.name,
                    status=DocumentStatus.SUCCESS,
                    pages=[
                        PageOutput(
                            page_num=0,
                            text=markdown,
                            status=PageStatus.SUCCESS,
                            engine=self.name,
                            processing_time=elapsed,
                        )
                    ],
                    processing_time=elapsed,
                    model_version=self.resolved_model_version(config),
                )

            except subprocess.TimeoutExpired:
                return EngineResult(
                    document_path=pdf_path,
                    engine=self.name,
                    status=DocumentStatus.ERROR,
                    failure_mode=FailureMode.TIMEOUT,
                    error=f"Timeout after {config.timeout}s",
                    processing_time=time.time() - start_time,
                    model_version=self.resolved_model_version(config),
                )

    def process_pages(
        self,
        pdf_path: Path,
        page_nums: list[int],
        config: PipelineConfig,
        dpi: int = 200,
    ) -> list[PageOutput]:
        """Process specific pages by rendering to images and calling the CLI.

        Renders each page to a PNG image, saves to a temp directory, calls the
        CLI on the directory, and reads back per-page markdown. Returns one
        PageOutput per page with the actual OCR text (no page_num=0 hack).

        Args:
            pdf_path: Path to the source PDF.
            page_nums: 1-indexed page numbers to process.
            config: Pipeline configuration.
            dpi: Render DPI for page images.

        Returns:
            List of PageOutput, one per page_num, in the same order.
        """
        import fitz
        from PIL import Image

        start_time = time.time()

        with tempfile.TemporaryDirectory() as tmpdir:
            images_dir = Path(tmpdir) / "images"
            images_dir.mkdir()
            cli_out = Path(tmpdir) / "out"

            # Render pages to numbered PNG images
            page_num_to_stem: dict[int, str] = {}
            with fitz.open(pdf_path) as doc:
                mat = fitz.Matrix(dpi / 72, dpi / 72)
                for page_num in page_nums:
                    stem = f"page_{page_num:04d}"
                    page_num_to_stem[page_num] = stem
                    pix = doc[page_num - 1].get_pixmap(matrix=mat)
                    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                    img.save(images_dir / f"{stem}.png")

            # Call CLI on the image directory
            cmd = self._build_command(images_dir, cli_out, config)
            logger.info(f"[{self.name}] Processing {len(page_nums)} pages: {' '.join(cmd)}")

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=config.timeout,
                )
            except subprocess.TimeoutExpired:
                return [
                    PageOutput(
                        page_num=pn,
                        status=PageStatus.ERROR,
                        engine=self.name,
                        failure_mode=FailureMode.TIMEOUT,
                        audit_passed=False,
                        error=f"Timeout after {config.timeout}s",
                    )
                    for pn in page_nums
                ]

            if result.returncode != 0:
                stderr = (result.stderr or "").strip()[:500]
                return [
                    PageOutput(
                        page_num=pn,
                        status=PageStatus.ERROR,
                        engine=self.name,
                        failure_mode=FailureMode.CLI_ERROR,
                        audit_passed=False,
                        error=f"CLI exited {result.returncode}: {stderr}",
                    )
                    for pn in page_nums
                ]

            # Read per-page output: CLI writes {cli_out}/{stem}/{stem}.md
            elapsed = time.time() - start_time
            outputs: list[PageOutput] = []
            # A page can come back empty because the model REFUSED, not failed.
            # Gemini's recitation/copyright filter blocks verbatim output of
            # memorized (famous) text and exits 0 with no page file, logging
            # "finish_reason=...RECITATION" to stderr. Detect it so repair can
            # escalate to an open model instead of uselessly retrying Gemini.
            stderr_text = result.stderr or ""
            recitation_in_stderr = "RECITATION" in stderr_text

            # Aggregated read-back (qwen and other canon engines that fold a
            # dir-of-images into ONE '## Page N' doc instead of per-page files).
            # Returns {page_num: text} mapped to socr's original page numbers, or
            # an all-pages count-mismatch error, or None when no aggregate exists.
            aggregate, aggregate_error = self._read_aggregated_pages(
                cli_out, images_dir, page_num_to_stem
            )
            if aggregate_error is not None:
                return [
                    PageOutput(
                        page_num=pn,
                        status=PageStatus.ERROR,
                        engine=self.name,
                        failure_mode=FailureMode.EMPTY_OUTPUT,
                        audit_passed=False,
                        error=aggregate_error,
                    )
                    for pn in page_nums
                ]

            for page_num in page_nums:
                stem = page_num_to_stem[page_num]
                text = self._read_page_output(stem, cli_out, scan_root=images_dir)
                # Fall back to the aggregated split when there is no per-page file.
                if not text and aggregate is not None:
                    text = aggregate.get(page_num)

                if text:
                    text = self._clean_output(text, self.name)
                    outputs.append(
                        PageOutput(
                            page_num=page_num,
                            text=text,
                            status=PageStatus.SUCCESS,
                            engine=self.name,
                            processing_time=elapsed / len(page_nums),
                            audit_passed=True,
                        )
                    )
                elif recitation_in_stderr and stem in stderr_text:
                    outputs.append(
                        PageOutput(
                            page_num=page_num,
                            status=PageStatus.ERROR,
                            engine=self.name,
                            failure_mode=FailureMode.RECITATION,
                            audit_passed=False,
                            error=(
                                "model refused (RECITATION: copyright/recitation "
                                "filter blocked verbatim output)"
                            ),
                        )
                    )
                else:
                    outputs.append(
                        PageOutput(
                            page_num=page_num,
                            status=PageStatus.ERROR,
                            engine=self.name,
                            failure_mode=FailureMode.EMPTY_OUTPUT,
                            audit_passed=False,
                            error=f"No output for page {page_num}",
                        )
                    )

            return outputs

    def _read_page_output(
        self, stem: str, output_dir: Path, scan_root: Path | None = None
    ) -> str | None:
        """Read output markdown for a single page image, canonical-first.

        Each rendered page image ``<stem>.png`` lives under ``scan_root`` and the
        engine was invoked with ``-o <output_dir>``. For a per-page-writing engine
        the canon places its output at ``doc_dir_for(output_dir, "<stem>.png")``,
        i.e. ``<output_dir>/<stem>_png/<stem>.md`` (``doc_dir_for`` disambiguates
        the non-PDF ``.png`` extension as ``<stem>_png``; the page image has no
        subdirectory, so the mirrored rel dir is empty). We resolve that exact
        path via the contract's helpers FIRST (keyed on the ``.png`` rel key, so
        the ``_png`` folder is computed correctly — never hand-built).

        socr ships independently of the sibling engine CLIs, so a not-yet-
        converged (or legacy) engine may write a non-canonical layout. To avoid
        silently yielding an EMPTY page for the DEFAULT per-page pipeline path,
        we keep a GUARDED legacy fallback ladder after the canonical path:

          1. canonical ``<output_dir>/<stem>_png/<stem>.md``  (preferred)
          2. flat ``<output_dir>/<stem>.md``
          3. sanitized-stem variants (``<sanitized>/<sanitized>.md`` + flat)
          4. a LOUD, stem-filtered ``rglob`` net (warns expected-vs-found)

        Each non-canonical hit is logged as an anomaly so divergence surfaces
        rather than passing silently. ``scan_root`` defaults to ``output_dir``
        for backwards compatibility, but callers should pass the image directory
        the page was rendered into so the relative key matches what the engine
        mirrored.
        """
        root = scan_root if scan_root is not None else output_dir
        rel_key = relative_key(root / f"{stem}.png", root)
        doc_dir = doc_dir_for(output_dir, rel_key)
        md_path = markdown_path_for(doc_dir, rel_key)
        if md_path.exists():
            return md_path.read_text(encoding="utf-8")

        legacy = self._legacy_page_md(stem, output_dir, expected=md_path)
        if legacy is not None:
            return legacy.read_text(encoding="utf-8")
        return None

    def _legacy_page_md(self, stem: str, output_dir: Path, *, expected: Path) -> Path | None:
        """Guarded legacy read-back for a single page (non-canonical layouts).

        Tries flat and sanitized-stem variants, then a stem-filtered ``rglob``
        net. Returns the resolved path (logging the anomaly) or ``None``. The
        rglob net requires the filename stem to match the page stem so a stray
        aggregate ``images.md`` (qwen's pre-canon layout) cannot be mistaken for
        a specific page; it also guards against symlinks escaping ``output_dir``.
        """
        sanitized = sanitize_filename(stem)
        candidates = [
            output_dir / f"{stem}.md",
            output_dir / sanitized / f"{sanitized}.md",
            output_dir / f"{sanitized}.md",
        ]
        for cand in candidates:
            if cand != expected and cand.exists():
                logger.warning(
                    f"[{self.name}] Canonical page output {expected} missing; "
                    f"using non-canonical {cand} (legacy layout fallback)"
                )
                return cand

        # Stem-filtered rglob net: only files whose stem matches the page stem
        # (canonical or sanitized), so an aggregate/stray .md is never returned.
        wanted_stems = {stem, sanitized}
        for md_file in sorted(output_dir.rglob("*.md")):
            if md_file.stem not in wanted_stems:
                continue
            if not md_file.resolve().is_relative_to(output_dir.resolve()):
                logger.warning(f"[{self.name}] Skipping symlink outside output dir: {md_file}")
                continue
            logger.warning(
                f"[{self.name}] Canonical page output {expected} missing; "
                f"using non-canonical {md_file} via stem-filtered rglob fallback"
            )
            return md_file
        return None

    def _read_aggregated_pages(
        self,
        cli_out: Path,
        images_dir: Path,
        page_num_to_stem: dict[int, str],
    ) -> tuple[dict[int, str] | None, str | None]:
        """Recover per-page text from a canon engine's AGGREGATED output.

        Some canon engines (qwen) treat the rendered page-image directory as ONE
        document and emit a single ``<images_dir.stem>/<images_dir.stem>.md`` with
        ``## Page N`` headers, rather than one ``.md`` per input image. socr's
        per-page read-back then finds no per-page files and would record every
        page as ``EMPTY_OUTPUT``. This recovers the pages by splitting that
        aggregate on the canonical headers and mapping each section back to the
        original socr page number.

        Returns ``(mapping, None)`` where ``mapping`` is ``{original_page_num:
        text}``; ``(None, None)`` when no aggregate exists (per-page engines);
        or ``(None, error)`` when the aggregate's page count does not match the
        number of rendered images (a corrupt/truncated aggregate — fail ALL pages
        rather than silently mis-aligning the mapping).

        Mapping rule (ordering-safe): qwen orders its ``## Page N`` sections by the
        SORTED image filenames it saw, so we map the k-th split section to the
        page whose rendered image is k-th in filename-sorted order — NOT to
        ``page_nums[k-1]`` (``page_nums`` may be non-contiguous / non-ascending).
        """
        # Locate the aggregate by the contract's canonical helpers (no glob).
        # A canon engine invoked as ``<engine> process <images_dir> -o <cli_out>``
        # treats the pure page-image dir as ONE document and writes its aggregate
        # at ``resolve_output_root(images_dir, cli_out)/<images_dir_stem>/<images_dir_stem>.md``.
        # Round-3 fix: resolve the root through ``resolve_output_root`` keyed on
        # the IMAGES dir (rather than hardcoding ``cli_out`` as the root) so the
        # path matches the engine's own computation exactly even if ``-o``
        # resolution semantics change. With ``-o cli_out`` given,
        # ``resolve_output_root(images_dir, cli_out) == cli_out``; the engine's
        # ``scan_root`` for a single image-dir document is ``images_dir.parent``,
        # so its ``rel_key`` is the dir name and ``doc_dir_for`` mirrors it.
        agg_root = resolve_output_root(images_dir, cli_out)
        rel_key = relative_key(images_dir, images_dir.parent)
        agg_doc_dir = doc_dir_for(agg_root, rel_key)
        agg_md = markdown_path_for(agg_doc_dir, rel_key)
        if not agg_md.exists():
            return None, None

        try:
            blob = agg_md.read_text(encoding="utf-8")
        except OSError as exc:
            return None, f"aggregate read failed: {exc}"

        sections = split_native_pages(self._clean_output(blob, self.name))
        # Engine-saw order: the SAME filename sort the canon engine applies to the
        # image dir. Each rendered image stem is page_{orig:04d}, so a filename
        # sort yields ascending original page order — robust to a non-ascending
        # page_nums request.
        engine_order = sorted(page_num_to_stem.items(), key=lambda kv: f"{kv[1]}.png")
        ordered_page_nums = [orig for orig, _stem in engine_order]

        if len(sections) != len(ordered_page_nums):
            return (
                None,
                (
                    f"aggregated output page count mismatch at {agg_md}: "
                    f"split {len(sections)} section(s) but rendered "
                    f"{len(ordered_page_nums)} image(s)"
                ),
            )
        return {orig: sections[i] for i, orig in enumerate(ordered_page_nums)}, None

    @abstractmethod
    def _build_command(
        self,
        pdf_path: Path,
        output_dir: Path,
        config: PipelineConfig,
    ) -> list[str]:
        """Build the CLI command for this engine."""
        ...

    def _read_output(self, pdf_path: Path, output_dir: Path) -> str | None:
        """Read the aggregated markdown from the CLI's output dir, canonical-first.

        Every engine now emits the canonical structure (via
        ``ocr-output-contract``): ``<output_dir>/<stem>/<stem>.md`` for a
        single-file PDF input. We resolve that path with the contract helpers
        FIRST, then fall back through a GUARDED legacy ladder so a not-yet-
        converged or legacy engine surfaces (logged) rather than silently
        yielding an empty document:

          1. canonical ``<output_dir>/<stem>/<stem>.md``  (preferred)
          2. flat ``<output_dir>/<stem>.md``
          3. sanitized-stem variants (``<sanitized>/<sanitized>.md`` + flat)
          4. a LOUD, stem-filtered ``rglob`` net (no longer returns the first
             arbitrary ``.md`` — only one whose stem matches the doc stem)
        """
        scan_root = pdf_path.parent
        rel_key = relative_key(pdf_path, scan_root)
        doc_dir = doc_dir_for(output_dir, rel_key)
        md_path = markdown_path_for(doc_dir, rel_key)
        if md_path.exists():
            return self._clean_output(md_path.read_text(encoding="utf-8"), self.name)

        legacy = self._legacy_page_md(pdf_path.stem, output_dir, expected=md_path)
        if legacy is not None:
            return self._clean_output(legacy.read_text(encoding="utf-8"), self.name)
        return None

    @staticmethod
    def _clean_output(text: str, engine: str = "") -> str:
        """Remove frontmatter, metadata headers, and normalize CLI output.

        Handles:
          - YAML frontmatter (--- ... ---)
          - Metadata headers (# OCR Results + **Original File:** + **Processed:** + ---)
          - Engine-specific artifact cleanup (via OutputNormalizer)
          - Generic markdown normalization (line endings, whitespace, unicode)
        """
        import re

        # Strip YAML frontmatter
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) >= 3:
                text = parts[2].strip()

        # Strip metadata header block (mistral-ocr format):
        # # OCR Results\n\n**Original File:**...\n**Full Path:**...\n**Processed:**...\n\n---
        # Requires at least one metadata line to avoid stripping real "# OCR Results" headings
        text = re.sub(
            r"^#\s*OCR Results\s*\n+"
            r"(?:\*\*(?:Original File|Full Path|Processed|Processing Time):\*\*[^\n]*\n)+"
            r"\s*(?:---\s*\n)?",
            "",
            text,
        ).strip()

        # Apply engine-specific + generic normalization
        text = _normalizer.normalize(text, engine=engine)

        return text


class BaseHTTPEngine(ABC):
    """Abstract base class for HTTP API engines (vLLM, HPC mode).

    These engines call a local vLLM server per-page via OpenAI-compatible API.
    They are NOT CLI-based — they use httpx to talk to a running server.
    """

    def __init__(self) -> None:
        self._initialized: bool = False

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    def model_version(self) -> str:
        """Model version string. Override in subclasses."""
        return ""

    @abstractmethod
    def initialize(self) -> bool:
        """Connect to the server and verify it's ready."""
        ...

    def is_available(self) -> bool:
        """Check if the engine can be initialized."""
        return self._initialized or self.initialize()

    @abstractmethod
    def process_image(self, image: "Image.Image", page_num: int = 1) -> PageOutput:
        """Process a single page image and return structured output."""
        ...

    def describe_figure(
        self,
        image: "Image.Image",
        figure_type: str = "unknown",
        context: str = "",
    ) -> FigureInfo:
        """Describe a figure image. Override in subclasses that support this."""
        return FigureInfo(
            figure_num=0,
            page_num=0,
            figure_type=figure_type,
            description="Figure description not supported by this engine",
        )

    def close(self) -> None:
        """Clean up resources."""
        pass

    @staticmethod
    def _create_success_result(
        page_num: int,
        text: str,
        engine: str = "",
        confidence: float = 0.0,
        processing_time: float = 0.0,
    ) -> PageOutput:
        return PageOutput(
            page_num=page_num,
            text=text,
            status=PageStatus.SUCCESS,
            engine=engine,
            processing_time=processing_time,
            confidence=confidence,
        )

    @staticmethod
    def _create_error_result(
        page_num: int,
        error: str,
        failure_mode: FailureMode = FailureMode.API_ERROR,
    ) -> PageOutput:
        return PageOutput(
            page_num=page_num,
            status=PageStatus.ERROR,
            failure_mode=failure_mode,
            error=error,
            # PageOutput defaults audit_passed=True; an ERROR output claiming
            # a passing audit gets promoted to best_output by apply_result
            # and ships as clean (review finding, issue #39).
            audit_passed=False,
        )
