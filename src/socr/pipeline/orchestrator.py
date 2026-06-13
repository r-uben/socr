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
from socr.core.born_digital import BornDigitalDetector, DocumentAssessment
from socr.core.config import EngineType, PipelineConfig
from socr.core.document import DocumentHandle
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
from socr.engines.registry import get_engine, resolve_auto_engine
from socr.figures.extractor import FigureExtractor
from socr.pipeline.consensus import ConsensusEngine
from socr.pipeline.repair import RepairRouter

logger = logging.getLogger(__name__)
console = Console()


def _manifest_versions() -> tuple[str, str]:
    """(NORMALIZER_VERSION, ASSEMBLY_VERSION), read at call time.

    Late attribute access (not a from-import) so the run fingerprint always
    reflects the live values and tests can monkeypatch them.
    """
    from socr.core import manifest

    return manifest.NORMALIZER_VERSION, manifest.ASSEMBLY_VERSION


def _resume_skippable(index, rel_key: str, checksum: str, fingerprint: str, out_dir: Path) -> bool:
    """Whether a doc can be skipped by the resume gate.

    Canonically completed docs skip via :meth:`RootIndex.is_completed`. A
    PARTIAL doc additionally skips when its recorded checksum AND run
    fingerprint match and its output still exists: re-running the identical
    config cannot improve a partial result, and without this rule every doc
    demoted to AUDIT_FAILED (flagged native fallback, lost pages) would be
    re-processed at full judge/repair cost on EVERY batch resume, forever.
    ``--reprocess`` still forces a retry (checked by the callers).
    """
    if index.is_completed(rel_key, checksum, fingerprint=fingerprint):
        return True
    entry = index.files.get(rel_key)
    if not entry or entry.get("status") != "partial":
        return False
    if entry.get("checksum") != checksum:
        return False
    if not entry.get("fingerprint") or entry.get("fingerprint") != fingerprint:
        return False
    out = entry.get("output_path") or ""
    if not out:
        return False
    p = Path(out)
    if not p.is_absolute():
        p = out_dir / p
    return p.exists()


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
        # Directory the current input was discovered under (batch input dir or
        # the file's parent). Threaded into every contract key so per-doc output
        # mirrors the input subtree relative to it, not the bare basename.
        self._scan_root: Path | None = None
        # Set by process_batch: the contract RunOutcome that drives the exit code.
        from ocr_output_contract import RunOutcome

        self.last_outcome: RunOutcome = RunOutcome()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _resolve_output_root(self, input_path: Path, output_dir: Path | None) -> Path:
        """Resolve the output root per the canon.

        An explicit ``output_dir`` (``-o``) is used verbatim. A non-default
        ``config.output_dir`` (anything the user set away from the legacy
        ``Path('output')`` sentinel) is also honored verbatim. Otherwise the
        canon default ``<input-parent>/ocr/`` is computed via the contract's
        :func:`resolve_output_root` rather than the legacy ``output`` folder.
        """
        from ocr_output_contract import resolve_output_root

        if output_dir is not None:
            return Path(output_dir)
        configured = self.config.output_dir
        if configured is not None and Path(configured) != Path("output"):
            return Path(configured)
        return resolve_output_root(Path(input_path))

    def _resolve_primary_engine(self) -> EngineType:
        """The concrete primary engine, resolving ``AUTO`` WITHOUT mutating config.

        ``process()`` resolves and writes back ``AUTO`` on entry; the batch resume
        gate runs BEFORE ``process()`` per file and must derive the SAME engine to
        compute an identical run fingerprint. We resolve here read-only so the
        gate and ``process()`` agree.
        """
        if self.config.primary_engine == EngineType.AUTO:
            return resolve_auto_engine()
        return self.config.primary_engine

    def _resume_skip(self, pdf_path: Path, out_dir: Path) -> EngineResult | None:
        """Return a SKIPPED result if this doc is already completed, else None.

        Consults the canonical :meth:`RootIndex.is_completed` (status==completed
        AND input checksum match AND output .md still on disk AND run-fingerprint
        match), so a re-run under a different model / task / output-affecting flag
        reprocesses instead of silently reusing the cached output. ``--reprocess``
        forces a re-run. An unreadable input (``safe_checksum`` None) is never
        treated as completed.
        """
        if self.config.reprocess:
            return None
        try:
            from ocr_output_contract import RootIndex, relative_key, safe_checksum

            checksum = safe_checksum(pdf_path)
            if checksum is None:
                return None
            scan_root = self._scan_root or pdf_path.parent
            rel_key = relative_key(pdf_path, scan_root)
            if not _resume_skippable(
                RootIndex(out_dir), rel_key, checksum, self._run_fingerprint(), out_dir
            ):
                return None
        except Exception as exc:  # never let the resume check break a run
            logger.warning("resume check failed (non-fatal): %s", exc)
            return None

        if not self.config.quiet:
            console.print(f"[dim]Skipping (already processed): {pdf_path.name}[/dim]")
        return EngineResult(
            document_path=pdf_path,
            engine=self.config.primary_engine.value,
            status=DocumentStatus.SKIPPED,
        )

    def _engine_determinants(self, engine_type: EngineType) -> dict[str, str | None]:
        """Resolved ``{model, backend, task}`` for an engine, never raising.

        Used to fold the model/backend/task of EVERY engine that can contribute
        text (primary, local, fallback-chain, multi-engine members) into the run
        fingerprint. A swap of a SECONDARY engine's model/task/backend changes
        the saved output (the orchestrator routes pages to it), so it must
        invalidate the resume cache just like a primary-engine swap. Degrades to
        the engine name on any error so fingerprinting never breaks a run.
        """
        from socr.engines.registry import get_engine

        try:
            engine = get_engine(engine_type)
            backend, task = engine.fingerprint_determinants(self.config)
            return {
                "model": engine.resolved_model_version(self.config),
                "backend": backend or "socr",
                "task": task,
            }
        except Exception:
            return {"model": str(engine_type.value), "backend": "socr", "task": None}

    def _run_fingerprint(self, engine_type: EngineType | None = None) -> str:
        """Run-config fingerprint for idempotency, from the RESOLVED run config.

        Captures what changes *what output an input produces*: the resolved
        primary engine's model id, backend, and task, the resolved determinants
        of every SECONDARY engine that can contribute text (local, fallback
        chain, multi-engine members — codex round-3: a secondary-engine model
        swap changes output without changing the primary), and socr's
        output-affecting orchestration flags. Stored in
        :class:`DocMetadata.fingerprint` and consulted by
        :meth:`RootIndex.is_completed`, so a re-run under a different model / task
        / flag reprocesses instead of silently reusing the cached output.

        Round-3 expansion (HIGH): the prior ``extra`` omitted ``save_figures``
        (and figure limits), the figures/consensus/judge model+backend knobs,
        ``fallback_chain``, ``local_engine``, ``tiered`` routing, and chunking
        thresholds — all of which change the saved ``.md``/figures. Toggling any
        of them now invalidates the cache. ``multi_engine`` is fingerprinted in
        USER ORDER (not sorted) so a reordering that changes consensus tie-break
        / first-best selection reprocesses (codex round-3 under-invalidation).

        Knobs deliberately EXCLUDED (do not change selected output bytes):
        display/scripting (``quiet``/``verbose``/``dry_run``), force-run
        (``reprocess``), parallelism (``workers``), and ``timeout`` — including
        them would force needless reprocessing.

        Resolved (not the AUTO sentinel) so the single-file gate, the batch gate,
        and the recorded metadata all agree on one value.
        """
        from ocr_output_contract import run_fingerprint

        engine_type = engine_type or self._resolve_primary_engine()
        primary = self._engine_determinants(engine_type)

        cfg = self.config
        extra: dict[str, object] = {
            # --- routing / engine selection (all contributing engines) ---
            "primary_engine": engine_type.value,
            "local_engine": cfg.local_engine.value,
            "fallback_chain": [e.value for e in cfg.fallback_chain],
            # User order, NOT sorted: order affects consensus/first-best output.
            "multi_engine": [e.value for e in cfg.multi_engine],
            # Resolved model/backend/task of every secondary engine, so a swap of
            # a local/fallback/multi member's model invalidates the cache too.
            "local_engine_determinants": self._engine_determinants(cfg.local_engine),
            "fallback_determinants": [self._engine_determinants(e) for e in cfg.fallback_chain],
            "multi_engine_determinants": [self._engine_determinants(e) for e in cfg.multi_engine],
            # --- rendering / chunking (change the bytes fed to the engine) ---
            "render_dpi": cfg.render_dpi,
            "native_first": cfg.native_first,
            "tiered": cfg.tiered,
            "chunk_threshold": cfg.chunk_threshold,
            "chunk_size": cfg.chunk_size,
            # --- quality gates / repair / routing ---
            "agentic": cfg.agentic,
            "strict_local": cfg.strict_local,
            "audit": cfg.audit_enabled,
            "audit_min_words": cfg.audit_min_words,
            "judge_hard_pages": cfg.judge_hard_pages,
            "judge_backend": cfg.judge_backend,
            "judge_model": cfg.judge_model,
            "dual_pass_tables": cfg.dual_pass_tables,
            "truncation_retries": cfg.truncation_retries,
            "max_retries": cfg.max_retries,
            # --- consensus ---
            "consensus": cfg.consensus_enabled,
            "consensus_use_llm": cfg.consensus_use_llm,
            "consensus_ollama_model": cfg.consensus_ollama_model,
            # --- figures (save_figures embeds figure blocks into the saved .md) ---
            "save_figures": cfg.save_figures,
            "figures_engine": cfg.figures_engine.value,
            "figures_max_total": cfg.figures_max_total,
            "figures_max_per_page": cfg.figures_max_per_page,
            # --- output-semantics code versions (issue #38) ---
            # The normalizer/assembler can change what bytes an identical
            # engine response produces. Without these, corpora cached under
            # the digit-fusing v1 normalizer pass the resume gate forever and
            # the fix never reaches them. Late import so monkeypatched
            # versions (tests) flow through.
            "normalizer_version": _manifest_versions()[0],
            "assembly_version": _manifest_versions()[1],
        }
        return run_fingerprint(
            primary["model"], primary["backend"] or "socr", primary["task"], None, extra=extra
        )

    def process(
        self,
        pdf_path: Path,
        output_dir: Path | None = None,
        scan_root: Path | None = None,
    ) -> EngineResult:
        """Process a single PDF through the 5-phase loop.

        Returns an EngineResult summarising the best extraction.

        ``output_dir`` overrides the output root verbatim (``-o``); when omitted
        the canon default ``<input-parent>/ocr/`` is resolved via the contract's
        :func:`resolve_output_root`. ``scan_root`` is the directory the input was
        discovered under: the batch input dir for a batch member (so the
        per-doc key mirrors the input subtree relative to it), or the file's
        parent for a standalone single-file run. It is NEVER ``pdf.parent`` for
        a batch member, which would collapse the key to the basename and defeat
        the canon's basename-collision fix.
        """
        pdf_path = Path(pdf_path)
        out_dir = self._resolve_output_root(pdf_path, output_dir)
        self._scan_root = scan_root if scan_root is not None else pdf_path.parent

        # Resolve AUTO engine before starting
        if self.config.primary_engine == EngineType.AUTO:
            self.config.primary_engine = resolve_auto_engine()
            if not self.config.quiet:
                console.print(
                    f"[dim]Auto-selected engine: {self.config.primary_engine.value}[/dim]"
                )

        # Single-file resume gate (canon): skip a doc already completed under the
        # SAME input checksum AND run fingerprint, with its .md still on disk.
        # Consulted here (not just in batch) so a re-run under a different model /
        # task / flag reprocesses. --reprocess forces a re-run. Batch enters via
        # process_batch's own gate, so this only fires on the single-file path.
        skipped = self._resume_skip(pdf_path, out_dir)
        if skipped is not None:
            return skipped

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
                console.print("\n[cyan]Phase 4:[/cyan] Repair (skipped — multi-engine mode)")

            # Phase 4b: Consensus — always run in multi-engine mode
            self._phase_consensus(state)
        else:
            # Single-engine mode: original flow
            # Phase 2: Backbone OCR
            backbone_result = self._phase_backbone(state, out_dir)

            # Phase 3: Score
            if backbone_result and backbone_result.success and self.config.audit_enabled:
                self._phase_score(state, backbone_result)

            # Phase 3b: VLM judge on HARD pages — catch semantic corruption
            # (wrong digits/signs/columns) the heuristics miss; rejects re-route
            # through repair.
            if self.config.audit_enabled and self.config.judge_hard_pages:
                self._phase_judge_hard_pages(state)

            # Phase 4: Selective Repair (loops up to max_retries)
            if self.config.audit_enabled:
                self._phase_repair(state, out_dir)

            # Phase 4b: Consensus (optional, after repair)
            if self.config.consensus_enabled:
                self._phase_consensus(state)

        # Phase 4c: Dual-pass table extraction — crop precisely-located tables,
        # re-read each crop with the judge VLM, and patch the authoritative crop
        # reading back into the page on disagreement. Runs for every mode, on the
        # final per-page text, just before assembly.
        if self.config.dual_pass_tables:
            self._phase_dual_pass_tables(state)

        # Phase 5: Assemble
        final_result = self._phase_assemble(state, out_dir)

        if not self.config.quiet:
            self._print_summary(final_result, state)

        return final_result

    def process_batch(self, input_dir: Path, output_dir: Path | None = None) -> list[EngineResult]:
        """Process all PDFs in a directory with incremental tracking.

        The exit-code-bearing summary is exposed on ``self.last_outcome`` (a
        contract :class:`RunOutcome`); the CLI consults it so the batch exit
        code is nonzero when any file failed (the canon's uniform exit policy).
        """
        from ocr_output_contract import (
            RootIndex,
            RunOutcome,
            Status,
            is_within_output_root,
            relative_key,
            safe_checksum,
        )

        input_dir = Path(input_dir)
        out_dir = self._resolve_output_root(input_dir, output_dir)
        outcome = RunOutcome()
        self.last_outcome = outcome

        # The canonical contract RootIndex is the SINGLE authoritative writer of
        # <root>/metadata.json and the SINGLE resume index. The legacy
        # MetadataManager (basename-keyed, TZ-naive, no fingerprint, no
        # output-existence check) is no longer used here: it used to CLOBBER the
        # RootIndex written by process()->_phase_assemble. RootIndex.record is
        # already called per-doc inside process(); the batch loop only READS it
        # for the resume gate.
        root_index = RootIndex(out_dir)
        # Resolve AUTO -> concrete ONCE so the gate's fingerprint matches the one
        # process() records for every file in this batch.
        run_fp = self._run_fingerprint(self._resolve_primary_engine())

        # Exclude anything under the resolved output root so a re-run never
        # re-ingests socr's own .md/figure outputs as fresh inputs.
        pdfs = sorted(p for p in input_dir.glob("*.pdf") if not is_within_output_root(p, out_dir))
        if not pdfs:
            if not self.config.quiet:
                console.print("[yellow]No PDF files found[/yellow]")
            return []

        to_process = []
        for pdf in pdfs:
            # Resume gate via the canon: an unreadable input (safe_checksum None)
            # is NEVER treated as completed — it falls through to process(), which
            # records a per-file failure rather than aborting the batch (SYS-02).
            checksum = safe_checksum(pdf)
            rel_key = relative_key(pdf, input_dir)
            already_done = checksum is not None and _resume_skippable(
                root_index, rel_key, checksum, run_fp, out_dir
            )
            if already_done and not self.config.reprocess:
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
            # Thread the BATCH input dir as scan_root so each per-doc key is the
            # path relative to input_dir (subtree-mirrored), NOT the basename.
            try:
                result = self.process(pdf, out_dir, scan_root=input_dir)
            except Exception as exc:  # one bad file must not abort the batch
                logger.warning("batch: %s failed: %s", pdf.name, exc)
                outcome.add(Status.FAILED, detail=str(pdf))
                results.append(
                    EngineResult(
                        document_path=pdf,
                        engine="none",
                        status=DocumentStatus.ERROR,
                        error=str(exc),
                    )
                )
                continue
            results.append(result)
            if result.success:
                # process()->_phase_assemble already recorded this doc in the
                # canonical RootIndex with the contract schema (model/backend/
                # fingerprint/UTC timestamp). No legacy second write — that was
                # the clobber that downgraded the root index to legacy shape.
                outcome.add(Status.COMPLETED, output_path=str(pdf))
            elif result.status == DocumentStatus.AUDIT_FAILED:
                outcome.add(Status.PARTIAL, detail=str(pdf))
            else:
                outcome.add(Status.FAILED, detail=str(pdf))

        if not self.config.quiet:
            ok = outcome.completed
            console.print(f"\n[green]Completed:[/green] {ok}/{len(to_process)} files")
            if outcome.has_failures:
                console.print(
                    f"[yellow]Failed/partial:[/yellow] {outcome.failed + outcome.partial}"
                )
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
                # Count pages needing enhancement vs trusted native text.
                enhancement_count = sum(
                    1 for pa in assessment.pages if pa.is_born_digital and pa.needs_ocr_enhancement
                )
                native_count = bd_count - enhancement_count
                scanned_count = assessment.scanned_count
                console.print(f"  {bd_count}/{assessment.page_count} pages born-digital")
                if self.config.native_first and (native_count or enhancement_count):
                    if native_count:
                        console.print(f"    {native_count} trusted native text")
                    if enhancement_count:
                        console.print(f"    {enhancement_count} native-layer recovery needed")
                    if scanned_count:
                        console.print(f"    {scanned_count} scanned (no text layer)")
            else:
                console.print("  No born-digital pages detected")

    # ------------------------------------------------------------------
    # Phase 2: Backbone OCR
    # ------------------------------------------------------------------

    def _phase_backbone(self, state: DocumentState, output_dir: Path) -> EngineResult | None:
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
            bd_pages = [p for p in state.pages.values() if p.is_born_digital]
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
                    f"Engine {engine.name} not available (CLI not installed or missing API key)"
                ),
            )
            state.apply_result(err_result)
            return err_result

        # Per-page processing: render all pages to images → CLI
        all_pages = list(range(1, state.handle.page_count + 1))
        if not self.config.quiet:
            console.print(f"  Processing {len(all_pages)} pages (per-page)...")

        start_time = time.time()
        page_outputs = engine.process_pages(
            pdf_path=state.handle.path,
            page_nums=all_pages,
            config=self.config,
            dpi=self.config.render_dpi,
        )
        elapsed = time.time() - start_time

        success_count = sum(1 for p in page_outputs if p.status == PageStatus.SUCCESS)
        overall_status = DocumentStatus.SUCCESS if success_count > 0 else DocumentStatus.ERROR

        if not self.config.quiet:
            console.print(f"  {success_count}/{len(all_pages)} pages succeeded")

        result = EngineResult(
            document_path=state.handle.path,
            engine=engine.name,
            status=overall_status,
            pages=page_outputs,
            pages_processed=state.handle.page_count,
            processing_time=elapsed,
            model_version=engine.resolved_model_version(self.config),
        )
        state.apply_result(result)
        return result

    def _backbone_native_first(self, state: DocumentState, output_dir: Path) -> EngineResult:
        """3-tier routing: native → local → cloud.

        Tier 1: Born-digital trusted native text (free, instant)
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
        # Born-digital pages whose prose is clean but whose math is font-corrupted:
        # keep native prose, recover only the equation regions to LaTeX (Tier 1).
        math_recovery_pages: set[int] = set()
        corrupt_math_pages = {
            pa.page_num
            for pa in (self._last_assessment.pages if self._last_assessment else [])
            if pa.has_corrupt_math
        }

        for page_num, ps in sorted(state.pages.items()):
            if (
                self.config.recover_corrupt_math
                and page_num in corrupt_math_pages
                and ps.is_born_digital
                and ps.native_text
            ):
                # Trust the native prose layer; math is spliced in Tier 1. This
                # also avoids whole-page VLM OCR that would degrade the prose.
                prose_pages.append(page_num)
                math_recovery_pages.add(page_num)
            elif ps.is_born_digital and not ps.needs_ocr_enhancement and ps.native_text:
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
                    (
                        pa
                        for pa in (
                            self._last_assessment
                            or DocumentAssessment(path=state.handle.path, pages=[])
                        ).pages
                        if pa.page_num == page_num
                    ),
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
                str(state.handle.path),
                ocr_pages,
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
                    f"  {len(prose_pages)}/{total} pages: native text (born-digital prose)"
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

        # Tier 1: Native text for prose pages (with math recovery where flagged)
        math_doc = None
        if math_recovery_pages:
            try:
                import fitz

                math_doc = fitz.open(state.handle.path)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("math recovery disabled (cannot open PDF): %s", exc)
                math_recovery_pages = set()
            if math_recovery_pages and not self.config.quiet:
                console.print(
                    f"  [cyan]{len(math_recovery_pages)} page(s): recovering "
                    f"corrupt math -> LaTeX [{self.config.math_model}][/cyan]"
                )

        math_done = 0
        math_total = len(math_recovery_pages)
        for page_num in prose_pages:
            ps = state.pages[page_num]
            text = ps.native_text
            engine = "native"
            if page_num in math_recovery_pages and math_doc is not None:
                from socr.math.recover import recover_math_regions, splice_math

                try:
                    page = math_doc[page_num - 1]
                    regions = recover_math_regions(page, model=self.config.math_model)
                    text = splice_math(page, ps.native_text, regions)
                    engine = "native+math"
                    math_done += 1
                    recovered = sum(1 for _, tex in regions if tex)
                    if not self.config.quiet:
                        console.print(
                            f"    [dim]math {math_done}/{math_total}: p{page_num} "
                            f"{recovered}/{len(regions)} equations -> LaTeX[/dim]"
                        )
                except Exception as exc:
                    logger.warning("math recovery failed on p%d: %s", page_num, exc)
            page_outputs.append(
                PageOutput(
                    page_num=page_num,
                    text=text,
                    status=PageStatus.SUCCESS,
                    engine=engine,
                    audit_passed=True,
                )
            )
        if math_doc is not None:
            math_doc.close()

        # Tier 2: Local engine for easy pages
        escalated_pages: list[int] = []
        escalated_reasons: dict[int, str] = {}
        local_engine_name = ""
        if easy_pages and local_engine_type:
            local_outputs = self._run_engine_on_pages(
                state,
                easy_pages,
                enhancement_pages,
                local_engine_type,
                "local",
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
                scoring = self.scorer.score(
                    po.text, engine=po.engine, sparse_ok=self._sparse_page_ok(po.page_num)
                )
                if scoring.passed:
                    po.audit_passed = True
                    passed_outputs.append(po)
                else:
                    # Quality failure — escalate to cloud
                    escalated_pages.append(po.page_num)
                    escalated_reasons[po.page_num] = scoring.primary_failure.value
                    logger.info(
                        "Page %d failed local audit (%s) — escalating to cloud",
                        po.page_num,
                        scoring.primary_failure.value,
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
                state,
                cloud_pages,
                enhancement_pages,
                self.config.primary_engine,
                "cloud",
            )
            # Tag escalated pages so metadata tracks the promotion
            for co in cloud_outputs:
                if co.page_num in escalated_pages:
                    co.escalated_from = local_engine_name
            page_outputs.extend(cloud_outputs)

        elapsed = time.time() - start_time

        success_count = sum(1 for p in page_outputs if p.status == PageStatus.SUCCESS)
        overall_status = DocumentStatus.SUCCESS if success_count > 0 else DocumentStatus.ERROR

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
                    outputs.append(
                        PageOutput(
                            page_num=page_num,
                            text=ps.native_text,
                            status=PageStatus.SUCCESS,
                            engine="native",
                            audit_passed=True,
                        )
                    )
                else:
                    outputs.append(
                        PageOutput(
                            page_num=page_num,
                            text="",
                            status=PageStatus.ERROR,
                            engine=engine.name,
                            failure_mode=FailureMode.MODEL_UNAVAILABLE,
                        )
                    )
            return outputs

        if not self.config.quiet:
            console.print(
                f"  Running {engine.name} on {len(page_nums)} {label} pages (per-page)..."
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
            if po.status != PageStatus.SUCCESS and po.page_num in enhancement_pages:
                ps = state.pages[po.page_num]
                if ps.native_text:
                    final.append(
                        PageOutput(
                            page_num=po.page_num,
                            text=ps.native_text,
                            status=PageStatus.SUCCESS,
                            engine="native",
                            audit_passed=True,
                        )
                    )
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
        its cost are recorded on PageState. Bounded by ladder exhaustion,
        ``max_cost_per_page``, and ``cost_budget`` — never by a retry count
        (issue #39: the old ``max_retries + 1`` cap made the paid rungs
        unreachable whenever 3+ free local engines were installed).
        """
        from socr.core.providers import provider_ladder
        from socr.pipeline.agentic import route_page

        if not self.config.quiet:
            console.print("\n[cyan]Agentic routing[/cyan] (cost-ordered, judge-gated)")

        # Tier 1: born-digital trusted native text (free, no OCR).
        # Skipped when --no-native-first: the user wants OCR even here, so
        # every page goes through the cost ladder.
        ocr_pages: list[int] = []
        for page_num, ps in sorted(state.pages.items()):
            if (
                self.config.native_first
                and ps.is_born_digital
                and not ps.needs_ocr_enhancement
                and ps.native_text
            ):
                native = PageOutput(
                    page_num=page_num,
                    text=ps.native_text,
                    status=PageStatus.SUCCESS,
                    engine="native",
                    audit_passed=True,
                    cost_usd=0.0,
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
        # --strict-local: drop cloud rungs so only free/local providers are tried.
        if self.config.strict_local:
            available = [p for p in available if p.is_free]
        ladder = provider_ladder(
            available, per_page_only=True, max_cost_per_page=self.config.max_cost_per_page
        )
        if not ladder:
            logger.warning("agentic: no OCR providers available; pages left unprocessed")
            if not self.config.quiet:
                console.print("  [red]No OCR providers available[/red]")
            return

        # Snapshot the ladder for manifest provenance (B3): ordered list of
        # providers, with their identity/cost/tier fields, frozen at routing time.
        state.agentic_ladder = [
            {
                "provider_id": p.id,
                "model": p.model,
                "backend": p.backend,
                "cost_per_page_usd": p.cost_per_page_usd,
                "tier": p.tier,
            }
            for p in ladder
        ]

        # Record judge model for manifest provenance (B3).
        state.agentic_judge_model = self._resolve_judge_model() or ""
        judge = self._build_page_judge(state)
        enhancement_pages = [p for p in ocr_pages if state.pages[p].needs_ocr_enhancement]

        if not self.config.quiet:
            ladder_str = " -> ".join(f"{p.engine.value}(${p.cost_per_page_usd:g})" for p in ladder)
            console.print(f"  ladder: {ladder_str}")

        def run_provider(engine: EngineType, page_num: int) -> PageOutput:
            outs = self._run_engine_on_pages(
                state, [page_num], enhancement_pages, engine, "agentic"
            )
            return outs[0]

        provider_timeout = getattr(self.config, "agentic_provider_timeout", None)

        for page_num in ocr_pages:
            # Escalation is bounded by the ladder + cost controls, never a
            # retry count (the old max_retries+1 cap made paid rungs
            # unreachable with 3+ free locals installed). The budget is
            # enforced BEFORE each paid rung inside route_page; free rungs
            # always remain available.
            remaining = None
            if self.config.cost_budget > 0:
                remaining = max(self.config.cost_budget - state.total_cost, 0.0)
            decision = route_page(
                page_num,
                ladder,
                run_provider,
                judge,
                remaining_budget=remaining,
                provider_timeout=provider_timeout,
            )

            ps = state.pages[page_num]
            for att in decision.attempts:
                att.output.cost_usd = att.cost_usd
                att.output.audit_passed = att.accepted
                att.output.provider_id = att.provider_id  # B3: agentic provenance
                att.output.provider_model = att.model  # B3
                att.output.provider_backend = att.backend  # B3
                # skip_reason: populated for budget-exceeded stubs (no OCR ran).
                att.output.skip_reason = (
                    att.reason if not att.accepted and not att.output.text else ""
                )  # B3
                ps.attempts.append(att.output)
            ps.best_output = decision.final_output

            # Record cost so DocumentState.total_cost reflects spend.
            state.engine_runs.append(
                EngineResult(
                    document_path=state.handle.path,
                    engine=decision.winning_engine,
                    status=DocumentStatus.SUCCESS
                    if decision.accepted
                    else DocumentStatus.AUDIT_FAILED,
                    cost=decision.total_cost_usd,
                    processing_time=0.0,
                )
            )

            if not self.config.quiet:
                tag = "accepted" if decision.accepted else "best-effort"
                console.print(
                    f"  p{page_num}: {decision.winning_engine} "
                    f"[{tag}, {len(decision.attempts)} tr, ${decision.total_cost_usd:.4f}]"
                )

        if not self.config.quiet:
            console.print(f"  total cost: ${state.total_cost:.4f}")

    def _available_engines_for_agentic(self) -> list:
        """Probe which known providers are actually usable right now.

        Returns a list of ``ProviderProfile`` objects (not EngineType values) so
        that two profiles sharing the same ``EngineType`` — e.g. QWEN local and
        QWEN cloud — can appear as distinct rungs in the ladder. Pass the result
        directly to ``provider_ladder()`` which accepts ``list[ProviderProfile]``.
        """
        from socr.core.providers import DEFAULT_PROVIDERS

        available = []
        for engine_type in self.config.enabled_engines:
            prof = DEFAULT_PROVIDERS.get(engine_type)
            if prof is None:
                continue
            try:
                if get_engine(engine_type).is_available():
                    available.append(prof)
            except Exception:  # availability probe must never crash routing
                continue
        return available

    # Vision models tried (in order) as the hard-page judge when judge_model is
    # unset. Cloud-first so the judge is fast; local small VLM as offline fallback.
    _JUDGE_MODEL_CANDIDATES = ["qwen3.5:cloud", "minicpm-v:8b", "qwen3-vl:8b"]

    def _resolve_judge_model(self) -> str | None:
        """Pick an available vision model for judging, or None if none usable."""
        from socr.judge.ollama_judge import OllamaVisionJudge

        if self.config.judge_model:
            return self.config.judge_model
        for model in self._JUDGE_MODEL_CANDIDATES:
            try:
                if OllamaVisionJudge(model=model).is_available():
                    return model
            except Exception:
                continue
        return None

    def _phase_judge_hard_pages(self, state: DocumentState) -> None:
        """Run a VLM judge on HARD pages (tables/equations) to catch semantic
        corruption the heuristic audit cannot see. Rejected pages lose their
        best_output, so the repair phase re-routes them to another engine.
        """
        from socr.judge.ollama_judge import OllamaVisionJudge
        from socr.pipeline.agentic import VLMPageJudge

        model = self._resolve_judge_model()
        if not model:
            return  # no vision judge available; heuristic audit already ran

        # Which pages are worth the extra call: those with tables or equations,
        # where wrong digits/signs/columns are both likely and costly.
        assessment = self._last_assessment
        hard_pages = {
            pa.page_num
            for pa in (assessment.pages if assessment else [])
            if pa.has_tables or pa.has_equations
        }
        if not hard_pages:
            return

        try:
            judge = VLMPageJudge(OllamaVisionJudge(model=model), self._make_page_renderer(state))
        except Exception as exc:
            logger.warning("hard-page judge unavailable (%s)", exc)
            return

        if not self.config.quiet:
            console.print(f"\n[cyan]Phase 3b:[/cyan] VLM judge on hard pages [{model}]")

        judged = rejected = 0
        for page_num in sorted(state.pages):
            if page_num not in hard_pages:
                continue
            ps = state.pages[page_num]
            bo = ps.best_output
            # Only judge model-produced OCR (native text is character-exact; the
            # corruption risk lives in the VLM transcription). "native+math" is
            # native prose with equation regions already recovered to LaTeX via
            # image-OCR — authoritative; the image-vs-text judge spuriously flags
            # its reading order and would revert the LaTeX to raw mojibake.
            if not bo or not bo.text or (bo.engine or "").startswith("native"):
                continue
            judged += 1
            try:
                # VLMPageJudge.assess(output, provider); provider is unused here.
                decision = judge.assess(bo, None)
            except Exception as exc:  # a judge failure must never drop the page
                logger.warning("judge errored on p%d (%s); keeping output", page_num, exc)
                continue
            if decision.accept:
                continue
            rejected += 1
            issues = decision.reason if decision.reason and decision.reason != "faithful" else ""
            bo.audit_passed = False
            bo.failure_mode = FailureMode.AUDIT_FAILED
            bo.error = "VLM judge rejected (image mismatch)" + (f": {issues}" if issues else "")
            from socr.core.audit_log import AuditEvent

            state.events.append(
                AuditEvent(
                    page_num=page_num,
                    kind="judge_reject",
                    engine=bo.engine,
                    detail=issues or "image mismatch",
                    data={"issues": issues, "judge_model": model},
                )
            )
            ps.best_output = None  # -> needs_repair -> repair escalates
            # Without this flag, born-digital pages are exempt from repair once
            # any attempt exists (the anti-loop rule), so a judge rejection
            # silently reverted to flat native text while reporting SUCCESS.
            ps.judge_rejected = True
            if not self.config.quiet:
                console.print(
                    f"  [yellow]p{page_num}: judge rejected — re-routing to repair"
                    + (f" ({issues})" if issues else "")
                    + "[/yellow]"
                )

        if not self.config.quiet:
            if judged == 0:
                console.print("  No model-OCR'd hard pages to judge")
            else:
                console.print(f"  {judged} judged, {rejected} rejected")

    def _phase_dual_pass_tables(self, state: DocumentState) -> None:
        """Crop precisely-located tables, re-read each with the judge VLM, and
        reconcile against the whole-page OCR.

        On disagreement the crop reading (higher-resolution, table-focused) is
        authoritative and patched into the page; the disagreement is surfaced and
        recorded in the page's audit notes. Tables whose count can't be safely
        mapped, or whose crop reading is malformed, are flagged but never edited.
        Fail-open throughout: any error keeps the page's existing text.
        """
        from socr.tables import locate_tables, reconcile_page_tables
        from socr.tables.extract import OllamaTableReader, TableCropExtractor

        model = self._resolve_judge_model()
        if not model:
            return  # no vision model available; nothing to do

        assessment = self._last_assessment
        table_pages = {
            pa.page_num for pa in (assessment.pages if assessment else []) if pa.has_tables
        }
        if not table_pages:
            return

        try:
            extractor = TableCropExtractor(OllamaTableReader(model=model))
        except Exception as exc:
            logger.warning("dual-pass extractor unavailable (%s)", exc)
            return

        if not self.config.quiet:
            console.print(f"\n[cyan]Phase 4c:[/cyan] dual-pass table extraction [{model}]")

        import fitz

        pdf_path = state.handle.path
        scanned = patched = flagged = 0
        for page_num in sorted(state.pages):
            if page_num not in table_pages:
                continue
            ps = state.pages[page_num]
            bo = ps.best_output
            # Only model-OCR'd pages carry transcription corruption risk; native
            # text is character-exact and its tables come straight from the PDF.
            if not bo or not bo.text or bo.engine == "native":
                continue
            scanned += 1
            try:
                with fitz.open(pdf_path) as doc:
                    boxes = locate_tables(doc[page_num - 1])
                if not boxes:
                    continue
                crops = extractor.extract(pdf_path, page_num, boxes)
                if not crops:
                    continue
                result = reconcile_page_tables(
                    bo.text,
                    [(c.markdown, c.source) for c in crops],
                    auto_patch=self.config.auto_patch_tables,
                )
            except Exception as exc:  # a dual-pass failure must never drop a page
                logger.warning("dual-pass errored on p%d (%s); keeping text", page_num, exc)
                continue

            if result.patched:
                bo.text = result.text
                patched += 1
            from socr.core.audit_log import AuditEvent

            for d in result.disagreements:
                flagged += 1
                bo.audit_notes.append(f"dual-pass {d.action}: {d.summary()}")
                state.events.append(
                    AuditEvent(
                        page_num=page_num,
                        kind=f"dualpass_{d.action}",
                        engine=bo.engine,
                        detail=d.summary(),
                        data={
                            "source": d.source,
                            "note": d.note,
                            "changed_cells": [
                                {
                                    "row": c.row,
                                    "col": c.col,
                                    "page": c.page_value,
                                    "crop": c.crop_value,
                                }
                                for c in d.changed_cells
                            ],
                        },
                    )
                )
            if result.disagreements and not self.config.quiet:
                for d in result.disagreements:
                    color = "green" if d.action == "patched" else "yellow"
                    console.print(f"  [{color}]p{page_num}: {d.action} — {d.summary()}[/{color}]")

        if not self.config.quiet:
            if scanned == 0:
                console.print("  No model-OCR'd table pages to re-read")
            else:
                console.print(f"  {scanned} pages scanned, {patched} patched, {flagged} flagged")

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
        # Sparse-aware at the DECISION point: without this, the heuristic
        # fallback judge rejects correct sparse pages at every rung and the
        # uncapped ladder pays the cloud engines for nothing.
        return HeuristicPageJudge(self.heuristics, sparse_ok=self._sparse_page_ok)

    def _make_page_renderer(self, state: DocumentState):
        """Return render_image(page_num) -> temp PNG path for the VLM judge."""

        def render(page_num: int) -> Path:
            img = state.handle.render_page(page_num, dpi=self.config.render_dpi)
            tmp = Path(tempfile.gettempdir()) / f"socr_judge_{state.handle.stem}_p{page_num}.png"
            img.save(tmp)
            return tmp

        return render

    def _fingerprint_inputs(
        self, state: DocumentState
    ) -> dict[str, tuple[str, str, str | None, str | None]]:
        """Resolve ``{engine: (model, backend, task, prompt)}`` for the manifest.

        For every engine that produced output on this doc, ask the engine
        adapter for its RESOLVED model id and ``(backend, task)`` determinants
        from the live config (not the hardcoded ``model_version`` literal). This
        is what makes the manifest fingerprint actually invalidate on model /
        backend / task drift, uniformly across the configurable-model engines.
        socr does not own the OCR prompt (the sibling CLIs do, selected via
        ``--task``), so ``prompt`` is the task selector's effect, left ``None``.
        """
        from socr.core.config import EngineType
        from socr.engines.registry import get_engine

        inputs: dict[str, tuple[str, str, str | None, str | None]] = {}
        names: set[str] = set()
        for run in state.engine_runs:
            for part in str(run.engine).replace("+", ",").split(","):
                part = part.strip()
                if part and part != "native":
                    # Strip a consensus(...) wrapper to the underlying engine.
                    if part.startswith("consensus(") and part.endswith(")"):
                        part = part[len("consensus(") : -1]
                    names.add(part)
        for name in names:
            try:
                engine = get_engine(EngineType(name))
            except (ValueError, KeyError):
                continue
            backend, task = engine.fingerprint_determinants(self.config)
            inputs[name] = (
                engine.resolved_model_version(self.config),
                backend,
                task,
                None,
            )
        return inputs

    def _write_manifest(
        self, state: DocumentState, output_dir: Path, saved_body: str | None = None
    ) -> None:
        """Write a reproducibility manifest + blob cache for `socr replay`.

        Non-fatal: a manifest failure must never lose the OCR output.
        """
        try:
            from ocr_output_contract import doc_dir_for, relative_key

            from socr.core.cache import BlobStore
            from socr.core.manifest import build_manifest

            pdf_path = state.handle.path
            scan_root = self._scan_root or pdf_path.parent
            doc_dir = doc_dir_for(output_dir, relative_key(pdf_path, scan_root))
            store = BlobStore(doc_dir / "cache")
            manifest = build_manifest(
                state,
                store,
                dpi=self.config.render_dpi,
                fingerprint_inputs=self._fingerprint_inputs(state),
                saved_body=saved_body,
            )
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
            console.print(f"\n[cyan]Phase 2:[/cyan] Multi-engine OCR [{', '.join(engine_names)}]")

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
            success_count = sum(1 for p in page_outputs if p.status == PageStatus.SUCCESS)
            result = EngineResult(
                document_path=state.handle.path,
                engine=engine.name,
                status=(DocumentStatus.SUCCESS if success_count > 0 else DocumentStatus.ERROR),
                pages=page_outputs,
                pages_processed=state.handle.page_count,
                model_version=engine.resolved_model_version(self.config),
            )
            state.apply_result(result)

            word_count = sum(p.word_count for p in result.pages)
            if not self.config.quiet:
                if result.success:
                    console.print(f"... [green]{word_count} words[/green]")
                else:
                    console.print(f"... [red]{result.error or result.status.value}[/red]")

            results.append(result)

        return results

    # ------------------------------------------------------------------
    # Phase 3: Score
    # ------------------------------------------------------------------

    def _phase_score(self, state: DocumentState, backbone_result: EngineResult) -> None:
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

    def _score_whole_doc(self, state: DocumentState, result: EngineResult) -> None:
        """Score a whole-document output (CLI engine, page_num=0)."""
        whole_doc_page = next((p for p in result.pages if p.page_num == 0), None)
        if not whole_doc_page:
            return

        # When the backbone used chunking, each chunk was small enough to
        # avoid truncation.  Skip the doc-level truncation check because
        # dividing chunk output by total pages gives a misleadingly low
        # words-per-page ratio.
        was_chunked = state.handle.page_count > self.config.chunk_threshold
        scoring = self.scorer.score(
            whole_doc_page.text,
            engine=result.engine,
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
                console.print(f"  [red]FAIL:[/red] {scoring.primary_failure.value}")
                for mode, detail in scoring.details.items():
                    console.print(f"    {detail}")

    def _sparse_page_ok(self, page_num: int) -> bool:
        """Whether low word count is expected on this page.

        Derived from the page itself, not an absolute threshold: a
        BORN-DIGITAL page whose own text layer carries fewer words than the
        audit minimum is legitimately sparse — demanding 50 words of OCR
        from a 24-word source page is how good sparse pages used to escalate
        straight to paid engines.

        Deliberately narrow (issue #39 review): ``has_figures`` alone does
        NOT qualify — a dense page that merely contains an image must keep
        the full gate, and scanned pages (whose text-layer word counts are
        junk) never earn leniency from this.
        """
        assessment = self._last_assessment
        if not assessment:
            return False
        pa = next((p for p in assessment.pages if p.page_num == page_num), None)
        if pa is None or not pa.is_born_digital:
            return False
        return (pa.word_count or 0) < self.config.audit_min_words

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
            scoring = self.scorer.score(
                latest.text, engine=latest.engine, sparse_ok=self._sparse_page_ok(page_num)
            )

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
                    console.print(f"  {result.engine}: [red]skipped (engine failed)[/red]")
                continue

            has_whole_doc = any(p.page_num == 0 for p in result.pages)

            if has_whole_doc:
                whole_page = next(p for p in result.pages if p.page_num == 0)
                was_chunked = state.handle.page_count > self.config.chunk_threshold
                scoring = self.scorer.score(
                    whole_page.text,
                    engine=result.engine,
                    expected_pages=(0 if was_chunked else state.handle.page_count),
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
                        console.print(f"  {result.engine}: [green]passed[/green]")
                    else:
                        console.print(
                            f"  {result.engine}: [red]{scoring.primary_failure.value}[/red]"
                        )
            else:
                # Per-page outputs: score each page
                passed = 0
                failed = 0
                for page_out in result.pages:
                    scoring = self.scorer.score(page_out.text, engine=result.engine)
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
        has_passing_whole_doc = any(w.audit_passed for w in state.whole_doc_attempts)
        # Also check if there's a failing whole-doc attempt that needs
        # document-level retry (e.g. truncated output).
        has_failing_whole_doc = any(not w.audit_passed for w in state.whole_doc_attempts)
        needs_whole_doc_retry = has_failing_whole_doc and not has_passing_whole_doc

        if has_passing_whole_doc and not state.pages_needing_repair:
            if not self.config.quiet:
                console.print("\n[cyan]Phase 4:[/cyan] Repair (not needed)")
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
            if not latest_whole.audit_passed and latest_whole.failure_mode == FailureMode.TRUNCATED:
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
                            state.handle.path,
                            all_pages,
                            self.config,
                            dpi=self.config.render_dpi,
                        )
                        retry_result = EngineResult(
                            document_path=state.handle.path,
                            engine=engine.name,
                            status=DocumentStatus.SUCCESS
                            if any(p.status == PageStatus.SUCCESS for p in page_outputs)
                            else DocumentStatus.ERROR,
                            pages=page_outputs,
                            pages_processed=state.handle.page_count,
                        )
                        state.apply_result(retry_result)
                        if retry_result.success:
                            self._score_repair_result(state, retry_result, [])
                        # Check if per-page results pass
                        ok = sum(
                            1
                            for p in page_outputs
                            if p.status == PageStatus.SUCCESS and p.audit_passed
                        )
                        if ok == state.handle.page_count:
                            needs_whole_doc_retry = False
                            has_passing_whole_doc = True
                            break

                    # If truncation retry resolved it, we're done
                    if not needs_whole_doc_retry:
                        if not self.config.quiet:
                            console.print("  [green]Truncation retry succeeded[/green]")
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
                            state.handle.path,
                            all_pages,
                            self.config,
                            dpi=self.config.render_dpi,
                        )
                        repair_result = EngineResult(
                            document_path=state.handle.path,
                            engine=engine.name,
                            status=DocumentStatus.SUCCESS
                            if any(p.status == PageStatus.SUCCESS for p in page_outputs)
                            else DocumentStatus.ERROR,
                            pages=page_outputs,
                            pages_processed=state.handle.page_count,
                        )
                        state.apply_result(repair_result)
                        if repair_result.success:
                            self._score_repair_result(state, repair_result, [])
                            if not state.pages_needing_repair:
                                needs_whole_doc_retry = False
                                break
                    continue

            if plan.is_empty:
                if not self.config.quiet and attempt == 0:
                    if state.pages_needing_repair:
                        console.print(
                            "\n[cyan]Phase 4:[/cyan] Repair (all engines exhausted, skipping)"
                        )
                    else:
                        console.print("\n[cyan]Phase 4:[/cyan] Repair (not needed)")
                break

            if not self.config.quiet:
                engines_str = ", ".join(e.value for e in plan.by_engine.keys())
                console.print(
                    f"\n[cyan]Phase 4:[/cyan] Repair "
                    f"(attempt {attempt + 1}/{self.config.max_retries}) "
                    f"[{engines_str}]"
                )
                console.print(f"  {len(plan.repairs)} page(s) to repair")
                # Surface RECITATION refusals explicitly: a Gemini copyright-filter
                # refusal must never be silent — name the reason and the recovery
                # engine so the user knows the page was refused, not just retried.
                for r in plan.repairs:
                    ps = state.pages.get(r.page_num)
                    if ps and any(a.failure_mode == FailureMode.RECITATION for a in ps.attempts):
                        console.print(
                            f"  [yellow]p{r.page_num}: Gemini refused "
                            f"(RECITATION — copyright/recitation filter) "
                            f"→ recovering via {r.engine.value}[/yellow]"
                        )
                if plan.pages_skipped:
                    console.print(
                        f"  {len(plan.pages_skipped)} page(s) skipped (engines exhausted)"
                    )

            # Execute repairs grouped by engine
            for engine_type, repairs in plan.by_engine.items():
                # A routing bug must skip one engine, never kill the document
                # run (get_engine raises ValueError on non-CLI engine types).
                try:
                    engine = get_engine(engine_type)
                except ValueError as exc:
                    logger.warning("repair: skipping %s (%s)", engine_type.value, exc)
                    continue

                if not engine.is_available():
                    if not self.config.quiet:
                        console.print(f"  [yellow]{engine.name} not available, skipping[/yellow]")
                    continue

                # Only process the failed pages, not the whole document
                failed_pages = [r.page_num for r in repairs]
                page_outputs = engine.process_pages(
                    state.handle.path,
                    failed_pages,
                    self.config,
                    dpi=self.config.render_dpi,
                )
                repair_result = EngineResult(
                    document_path=state.handle.path,
                    engine=engine.name,
                    status=DocumentStatus.SUCCESS
                    if any(p.status == PageStatus.SUCCESS for p in page_outputs)
                    else DocumentStatus.ERROR,
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
                whole_page.text,
                engine=result.engine,
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
                    page_out.text,
                    engine=result.engine,
                    sparse_ok=self._sparse_page_ok(page_out.page_num),
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
                    "\n[cyan]Phase 4b:[/cyan] Consensus (not needed — no multi-attempt pages)"
                )
            return

        if not self.config.quiet:
            parts = []
            if has_multi_whole_doc:
                parts.append(f"{len(state.whole_doc_attempts)} whole-doc attempts")
            if has_multi_pages:
                count = sum(1 for pn in state.pages if len(state.pages[pn].attempts) >= 2)
                parts.append(f"{count} multi-attempt pages")
            console.print(f"\n[cyan]Phase 4b:[/cyan] Consensus ({', '.join(parts)})")

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

    def _canonical_body(self, state: DocumentState) -> tuple[str, bool]:
        """Assemble the document body with canonical ``## Page N`` headers.

        Replaces socr's legacy ``\\n\\n---\\n\\n`` join: the saved ``.md`` now
        carries one ``## Page N`` header per page (via the contract's
        ``assemble_pages``), so the contract's ``split_native_pages`` round-trips
        it and the manifest/replay is bit-consistent with the saved file.
        Validates the resulting marker count against ``handle.page_count`` and
        logs loudly on mismatch (the legacy ``---``-only or dropped-marker case)
        rather than silently corrupting the per-page structure.

        Returns ``(body, has_content)`` where ``has_content`` reflects whether
        any page carries real text — an all-empty document yields header-only
        markup, which must NOT count as content (else an empty doc would be
        recorded as success).
        """
        from ocr_output_contract import PAGE_MARKER_RE, assemble_pages

        from socr.core.manifest import canonical_page_texts, is_page_failed_marker

        texts = canonical_page_texts(state)
        # Failure markers are honesty, not content: an all-failed document
        # must still be recorded as having produced nothing.
        has_content = any(t.strip() and not is_page_failed_marker(t) for t in texts)
        if not has_content:
            return "", False
        body = assemble_pages(texts)
        markers = len(PAGE_MARKER_RE.findall(body))
        if markers != state.handle.page_count:
            logger.warning(
                "assemble: produced %d '## Page N' marker(s) but the document has "
                "%d page(s); per-page structure may be incomplete",
                markers,
                state.handle.page_count,
            )
        return body, True

    def _phase_assemble(self, state: DocumentState, output_dir: Path) -> EngineResult:
        """Build the final EngineResult from DocumentState and save to disk."""
        if not self.config.quiet:
            console.print("\n[cyan]Phase 5:[/cyan] Assemble")

        final_text, has_text = self._canonical_body(state)

        # Pages that produced no usable text anywhere (shipped as explicit
        # failure markers) and enhancement pages that silently reverted to
        # native text after recovery was tried and never passed. Both demote
        # the document from SUCCESS: a run that lost content or shipped known-
        # lossy fallbacks must not report a clean pass.
        from socr.core.manifest import canonical_page_texts, is_page_failed_marker

        page_texts = canonical_page_texts(state)
        failed_pages = [i for i, t in enumerate(page_texts, start=1) if is_page_failed_marker(t)]
        native_fallback_pages = [
            n
            for n, p in sorted(state.pages.items())
            if p.is_born_digital
            and p.native_text
            and p.needs_ocr_enhancement
            and p.attempts
            and not (p.best_output and p.best_output.audit_passed)
        ]

        # Determine overall status.
        # For CLI engines that produce whole-doc output (page_num=0), pages
        # won't have per-page best_outputs.  A passing whole-doc attempt
        # covers the entire document -- treat it as success.
        has_passing_whole_doc = any(w.audit_passed for w in state.whole_doc_attempts)
        pages_ok = not state.pages_needing_repair or has_passing_whole_doc
        pages_ok = pages_ok and not failed_pages and not native_fallback_pages

        if has_text and pages_ok:
            status = DocumentStatus.SUCCESS
        elif has_text:
            status = DocumentStatus.AUDIT_FAILED
        else:
            status = DocumentStatus.ERROR

        state.status = status

        if failed_pages or native_fallback_pages:
            from socr.core.audit_log import AuditEvent

            for n in failed_pages:
                state.events.append(
                    AuditEvent(
                        page_num=n,
                        kind="page_failed",
                        engine="",
                        detail="no usable OCR output; failure marker shipped",
                    )
                )
            for n in native_fallback_pages:
                state.events.append(
                    AuditEvent(
                        page_num=n,
                        kind="native_fallback",
                        engine="native",
                        detail="OCR tried and never passed on an enhancement page; "
                        "native text shipped flagged",
                    )
                )
            if not self.config.quiet:
                if failed_pages:
                    console.print(
                        f"  [red]{len(failed_pages)} page(s) produced no usable "
                        f"output: {failed_pages}[/red]"
                    )
                if native_fallback_pages:
                    console.print(
                        f"  [yellow]{len(native_fallback_pages)} enhancement page(s) "
                        f"fell back to native text: {native_fallback_pages}[/yellow]"
                    )

        # Compute total processing time
        total_time = sum(r.processing_time for r in state.engine_runs)

        # Strip phantom image references before saving
        from ocr_output_contract import doc_dir_for, relative_key

        normalizer = OutputNormalizer()
        scan_root = self._scan_root or state.handle.path.parent
        doc_dir = doc_dir_for(output_dir, relative_key(state.handle.path, scan_root))
        if has_text:
            final_text = normalizer.strip_phantom_images(final_text, output_dir=doc_dir)

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
        if failed_pages:
            from socr.core.result import LOST_CONTENT_NOTE

            final_result.error = (
                f"page(s) {', '.join(str(n) for n in failed_pages)} {LOST_CONTENT_NOTE}"
            )

        # Save markdown + metadata BEFORE the figure phase: the describe loop
        # makes long paid API calls, and any exception there used to lose the
        # fully-extracted OCR text with no record at all (no .md, no
        # metadata.json — observed on real runs). Text is final at this point;
        # figures only append to it. When the figure phase is still pending,
        # the metadata is PROVISIONAL: a ":pre-figures" fingerprint suffix
        # guarantees the resume gate never matches it, so a hard crash during
        # figures leaves a retryable record instead of a skipped-forever doc.
        runs_figures = self.config.save_figures and has_text
        if has_text:
            saved_path = self._save_markdown(state, final_text, output_dir)
            if not self.config.quiet:
                console.print(f"  [blue]Output:[/blue] {saved_path}")
        self._write_metadata(state, final_result, output_dir, has_text, provisional=runs_figures)

        # Figure extraction + description + embedding. A figure-phase failure
        # must never destroy the completed OCR run — fall back to the already-
        # saved un-embedded text.
        if runs_figures:
            try:
                embedded_text = self._describe_and_embed_figures(
                    state,
                    final_result,
                    output_dir,
                    final_text,
                )
            except Exception as exc:
                logger.warning("figure phase failed (%s); keeping the un-embedded markdown", exc)
                if not self.config.quiet:
                    console.print(
                        f"  [yellow]Figure phase failed ({exc}); output saved "
                        "without figure descriptions[/yellow]"
                    )
                embedded_text = final_text
            if embedded_text != final_text:
                final_text = embedded_text
                final_result.pages[0].text = final_text
                self._save_markdown(state, final_text, output_dir)
            # Always finalize the record (real fingerprint, final status),
            # replacing the provisional pre-figures entry.
            self._write_metadata(state, final_result, output_dir, has_text)

        # Reproducibility manifest (opt-in; default-on in agentic mode). Pass the
        # FINAL saved body so the manifest blobs (and thus replay) reproduce the
        # on-disk .md bit-for-bit, not the pre-transform state.
        if has_text and (self.config.write_manifest or self.config.agentic):
            self._write_manifest(state, output_dir, saved_body=final_text)

        # Durable per-run audit log of notable events (RECITATION escalations,
        # judge rejections, dual-pass patches). Always written; never fatal.
        self._write_audit_log(state, doc_dir)

        return final_result

    def _write_metadata(
        self,
        state: DocumentState,
        result: EngineResult,
        output_dir: Path,
        has_text: bool,
        provisional: bool = False,
    ) -> None:
        """Write canonical per-doc + root-index ``metadata.json`` via the contract.

        Uses ``ocr-output-contract`` so socr's sidecars are byte-shape-identical
        to what the engine CLIs emit, keyed by the input-RELATIVE path (never the
        basename — two same-named PDFs in different dirs stay distinct). Both
        levels are kept in sync. Non-fatal: a metadata write must never lose OCR
        output.
        """
        from ocr_output_contract import (
            DocMetadata,
            RootIndex,
            Status,
            doc_dir_for,
            failure_checksum,
            markdown_path_for,
            relative_key,
            utc_timestamp,
        )

        try:
            pdf_path = state.handle.path
            scan_root = self._scan_root or pdf_path.parent
            rel_key = relative_key(pdf_path, scan_root)
            doc_dir = doc_dir_for(output_dir, rel_key)
            doc_dir.mkdir(parents=True, exist_ok=True)
            md_path = markdown_path_for(doc_dir, rel_key)

            if result.status == DocumentStatus.SUCCESS:
                status = Status.COMPLETED
            elif has_text:
                status = Status.PARTIAL
            else:
                status = Status.FAILED
            # Pre-figures record: never COMPLETED (figures pending), and
            # fingerprint-suffixed below so the resume gate cannot match it.
            if provisional and status == Status.COMPLETED:
                status = Status.PARTIAL

            # failure_checksum (round-3): the real sha256 digest if the input is
            # readable, else the contract's UNREADABLE_CHECKSUM sentinel
            # (``sha256:0...0``). This keeps a FAILED/unreadable record schema-
            # conformant (the conformance harness rejects a ``""`` / non-
            # ``sha256:`` checksum) AND never matches on resume (the sentinel or a
            # changed digest forces reprocess), all without raising and losing the
            # metadata record (SYS-02).
            meta = DocMetadata(
                status=status,
                checksum=failure_checksum(pdf_path),
                model=result.engine or "none",
                backend="socr",
                processing_time=result.processing_time,
                timestamp=utc_timestamp(),
                output_path=str(md_path) if has_text else "",
                pages=state.handle.page_count,
                error=result.error or None,
                # Run-config fingerprint: a re-run under a different model / task /
                # output-affecting flag is forced to reprocess instead of being
                # skipped by RootIndex.is_completed on input-checksum alone. By the
                # time _phase_assemble runs, process() has resolved AUTO into a
                # concrete primary_engine, so this matches the batch resume gate.
                # A provisional (pre-figures) record is suffix-marked so the
                # gate can never match it — a crash during figures must leave
                # a retryable record.
                fingerprint=self._run_fingerprint() + (":pre-figures" if provisional else ""),
            )
            from ocr_output_contract import write_doc_metadata

            write_doc_metadata(doc_dir, rel_key, meta)
            RootIndex(output_dir).record(rel_key, meta)
        except Exception as exc:  # never lose output over a metadata write
            logger.warning("metadata write failed (non-fatal): %s", exc)

    def _write_audit_log(self, state: DocumentState, doc_dir: Path) -> None:
        """Write audit_log.json next to the output; surface a one-line summary."""
        try:
            from socr.core.audit_log import build_run_audit

            audit = build_run_audit(state)
            if not audit.events:
                return  # a clean run leaves no audit log to inspect
            audit.save(doc_dir / "audit_log.json")
            if not self.config.quiet:
                console.print(
                    f"  [dim]Audit log: {doc_dir / 'audit_log.json'} ({audit.summary_line()})[/dim]"
                )
        except Exception as exc:  # never lose output over an audit-log write
            logger.warning("audit log write failed (non-fatal): %s", exc)

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

        from ocr_output_contract import doc_dir_for, figures_dir_for, relative_key

        scan_root = self._scan_root or state.handle.path.parent
        doc_dir = doc_dir_for(output_dir, relative_key(state.handle.path, scan_root))
        figures_dir = figures_dir_for(doc_dir)
        extractor = FigureExtractor(
            max_total=self.config.figures_max_total,
            max_per_page=self.config.figures_max_per_page,
            save_dir=figures_dir,
        )
        # Scanned pages have no localizable figures — the whole page is one
        # raster, which the extractor would emit as a full-page "figure" and
        # send to the vision model (#42). Skip them when the born-digital
        # phase classified this document; PageState.is_born_digital defaults
        # to False, so the assessment is the only safe source of skips.
        skip_pages: set[int] | None = None
        if self._last_assessment is not None:
            skip_pages = set(self._last_assessment.scanned_pages()) or None
            if skip_pages and not self.config.quiet:
                console.print(
                    f"  [dim]Skipping {len(skip_pages)} scanned page(s) "
                    f"(no localizable figures)[/dim]"
                )
        extracted = extractor.extract(state.handle.path, skip_pages=skip_pages)

        if not extracted:
            if not self.config.quiet:
                console.print("  [dim]No figures detected[/dim]")
            result.figures = []
            return text

        if not self.config.quiet:
            console.print(f"  Extracted {len(extracted)} figures to {figures_dir}")

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
                    fig.image,
                    context=context,
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
            console.print(f"  {len(figures)} figures processed ({described} described)")

        # Build figure blocks and append to text
        figure_blocks = self._build_figure_blocks(figures, doc_dir)
        if figure_blocks:
            text = text.rstrip() + "\n\n" + figure_blocks

        return text

    def _get_vision_engine(self):
        """Try to create a Gemini API engine for figure description.

        When Ollama is available, returns a LocalFirstFigureEngine that tries
        qwen3-vl:30b-a3b-instruct first on every call and falls back to Gemini
        per-call on empty or error result.  When Ollama is unavailable, returns
        GeminiAPIEngine directly.  Returns None only when both are unreachable.
        """
        import os

        from socr.engines.gemini_api import (
            GeminiAPIConfig,
            GeminiAPIEngine,
            LocalFirstFigureEngine,
            OllamaFigureEngine,
        )

        # Build a Gemini engine if credentials are available (used as fallback)
        def _try_gemini() -> GeminiAPIEngine | None:
            api_key = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
            if not api_key:
                return None
            engine = GeminiAPIEngine(
                GeminiAPIConfig(api_key=api_key, model=self.config.gemini_model)
            )
            return engine if engine.initialize() else None

        ollama = OllamaFigureEngine()
        if ollama.is_available():
            gemini_fallback = _try_gemini()
            if not self.config.quiet:
                fb = " + Gemini fallback" if gemini_fallback else ""
                console.print(
                    f"  [dim]Using local Ollama ({ollama.model}) for figure descriptions{fb}[/dim]"
                )
            return LocalFirstFigureEngine(ollama, gemini_fallback)

        if not self.config.quiet:
            console.print(
                "  [dim]Ollama not available — trying Gemini API for figure descriptions[/dim]"
            )

        gemini = _try_gemini()
        if gemini is not None:
            return gemini

        if not self.config.quiet:
            console.print(
                "  [dim]No figure description engine available"
                " — saving figures without descriptions[/dim]"
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
        figures: list[FigureInfo],
        doc_dir: Path,
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

    def _save_markdown(self, state: DocumentState, text: str, output_dir: Path) -> Path:
        """Save the assembled markdown at the canonical contract path.

        Layout matches the family canon and the read-back path exactly:
        ``output_dir/<rel/dir>/<stem>/<stem>.md`` keyed by the input-relative
        path (via the shared contract helpers), so the written ``.md``, the
        per-doc ``metadata.json`` and ``output_path`` all agree.
        """
        from ocr_output_contract import (
            doc_dir_for,
            markdown_path_for,
            relative_key,
        )

        pdf_path = state.handle.path
        scan_root = self._scan_root or pdf_path.parent
        rel_key = relative_key(pdf_path, scan_root)
        doc_dir = doc_dir_for(output_dir, rel_key)
        doc_dir.mkdir(parents=True, exist_ok=True)
        md_path = markdown_path_for(doc_dir, rel_key)
        md_path.write_text(text, encoding="utf-8")
        return md_path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _print_summary(self, result: EngineResult, state: DocumentState) -> None:
        """Print a final summary line."""
        if result.success:
            status_str = "[green]Success[/green]"
        else:
            status_str = f"[red]{result.status.value}[/red]"

        engine_str = result.engine
        if self.config.multi_engine:
            engine_str = "+".join(e.value for e in self.config.multi_engine) + " (consensus)"

        console.print(f"\n{status_str} | {engine_str} | {result.processing_time:.1f}s")
        if state.pages_needing_repair:
            console.print(
                f"[yellow]{len(state.pages_needing_repair)} page(s) still failing[/yellow]"
            )
        if result.error:
            console.print(f"[dim]{result.error}[/dim]")
