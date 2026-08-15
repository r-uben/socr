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
from socr.core.normalizer import (
    MAX_CONSECUTIVE_IDENTICAL_TABLE_ROWS,
    OutputNormalizer,
    collapse_repeated_table_rows,
)
from socr.core.result import (
    DocumentStatus,
    EngineResult,
    FailureMode,
    FigureInfo,
    PageOutput,
    PageStatus,
)
from socr.core.state import DocumentState, PageState
from socr.engines.registry import get_engine, resolve_auto_engine
from socr.figures.extractor import ExtractionResult, FigureExtractor, has_chart_marks
from socr.pipeline.agentic import route_page
from socr.pipeline.consensus import ConsensusEngine
from socr.pipeline.repair import RepairRouter
from socr.tables.extract import probe_ollama_idle

logger = logging.getLogger(__name__)
console = Console()

# Provenance value recorded when no vision judge could be built and the page gate
# is the heuristic checker. A literal sentinel (not "") so the distinction between
# "heuristics judged this run" and "this field was never populated" survives into
# metadata.json and the run fingerprint (#133).
JUDGE_IDENTITY_HEURISTIC = "heuristic"


def _page_blob_key(page_output_dict: dict) -> str:
    """Content-addressed key for a serialised PageOutput dict.

    Used in per-page sidecars as a lightweight ``page_fingerprint`` that
    changes whenever the winning page text changes.  Mirrors the BlobStore
    key derivation so PP-5 can cross-reference with the manifest without
    opening the manifest file.
    """
    import hashlib
    import json

    payload = json.dumps(page_output_dict, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


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

    # Memoized judge-model resolution (#133). ``_resolve_judge_model`` probes
    # Ollama over HTTP once per entry in ``_JUDGE_MODEL_CANDIDATES``, and it is
    # consulted from ``_run_fingerprint``, which runs ONCE PER PAGE via
    # ``_flush_page_sidecar`` — resolving eagerly each time would cost 3 probes
    # per page. Sentinel ``False`` = not yet resolved (``None`` is a legitimate
    # result meaning "no VLM judge is available").
    #
    # Declared at CLASS level, not in ``__init__``: tests construct pipelines via
    # ``object.__new__(UnifiedPipeline)`` to skip constructor side effects, and a
    # fingerprint call on such an instance must not explode on a missing
    # attribute. Assignment in ``_resolve_judge_model`` shadows it per instance.
    _judge_model_cache: str | None | bool = False

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
            # PP-5: the agentic provider ladder is built from ``enabled_engines``
            # and pruned by ``max_cost_per_page`` / ``cost_budget`` — all three
            # change which provider produces (or is even tried on) a page, so a
            # change to any must invalidate the per-page resume ledger.  SORTED by
            # engine value (NOT user order): ``provider_ladder`` is cost-ordered
            # and ignores the input list order, so two configs with the same
            # enabled SET in a different order select identical output — recording
            # the raw order would spuriously invalidate the ledger (needless
            # re-OCR).  Adding/removing an engine (a real set change) still does.
            "enabled_engines": sorted(e.value for e in cfg.enabled_engines),
            # And the RESOLVED model/backend/task of EVERY enabled engine, NOT
            # just its name: an enabled engine drives the agentic ladder, so a
            # provider present ONLY through ``enabled_engines`` (not also via
            # primary/local/fallback/multi) would otherwise let a model/backend/
            # task swap reuse a stale terminal sidecar on resume.  Keyed by engine
            # value and SORTED for determinism (the ladder is cost-ordered, not
            # list-ordered, so member identity — not position — drives output).
            "enabled_engine_determinants": {
                e.value: self._engine_determinants(e)
                for e in sorted(cfg.enabled_engines, key=lambda x: x.value)
            },
            "max_cost_per_page": cfg.max_cost_per_page,
            "cost_budget": cfg.cost_budget,
            # GH-96: the escalation lane rewrites page text, so a resumed run must
            # not reuse fragments produced with the flag in the other state — that
            # would silently ship a mix of escalated and non-escalated pages.
            "escalate_ambiguous_tables": getattr(cfg, "escalate_ambiguous_tables", False),
            # --- rendering / chunking (change the bytes fed to the engine) ---
            "render_dpi": cfg.render_dpi,
            "native_first": cfg.native_first,
            "native_only": cfg.native_only,
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
            # The RESOLVED judge identity, not the (possibly empty) config field.
            # Under the default auto-resolution ``cfg.judge_model`` is "", so
            # pulling or removing a judge model changed how pages were gated
            # without changing the fingerprint — and terminal pages resumed under
            # the other judge (#133). Availability-dependent BY DESIGN: a
            # different judge is a different run, and the ledger must say so.
            # ``JUDGE_IDENTITY_HEURISTIC`` when no vision judge is available, so
            # heuristic-gated pages never resume as VLM-gated ones. An explicit
            # ``--judge-backend heuristic`` short-circuits the resolver: no VLM
            # can run, so probing Ollama would cost 3 HTTP round-trips to record
            # a model that never judges.
            "judge_model": (
                JUDGE_IDENTITY_HEURISTIC
                if cfg.judge_backend == "heuristic"
                else (self._resolve_judge_model() or JUDGE_IDENTITY_HEURISTIC)
            ),
            "dual_pass_tables": cfg.dual_pass_tables,
            "truncation_retries": cfg.truncation_retries,
            "max_retries": cfg.max_retries,
            # --- consensus ---
            "consensus": cfg.consensus_enabled,
            "consensus_use_llm": cfg.consensus_use_llm,
            "consensus_ollama_model": cfg.consensus_ollama_model,
            # --- figures ---
            # ``save_figures`` controls PNG extraction + image-ref embedding.
            # ``describe_figures`` is the separate opt-in for VLM captions.
            # Both change the saved .md content, so both must invalidate the cache.
            "save_figures": cfg.save_figures,
            "describe_figures": cfg.describe_figures,
            # --- GH-36a: equation region detection ---
            # ``detect_equations`` controls whether display-equation regions are
            # detected, cropped, and recorded in provenance (model-free, GH-36a).
            # Changing it changes the audit log and crop artefacts, so it must
            # invalidate the cache.
            "detect_equations": cfg.detect_equations,
            # --- GH-36b: clean-equation → LaTeX via local VLM + 1A gate ---
            # ``recover_clean_equations`` enables the engine call + pylatexenc
            # 1A gate + 1C sidecar attachment.  Changing it changes the .md
            # content (sidecar blocks appear/disappear) so it invalidates cache.
            "recover_clean_equations": cfg.recover_clean_equations,
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
        # PP-2: skip in agentic mode — tables are handled per-page inside
        # _phase_agentic's fused loop (avoids re-reading tables twice).
        if self.config.dual_pass_tables and not (self.config.agentic and not is_multi):
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

        # GH-147 A2: born_digital.py refuses the native table lane for rotated
        # pages (transposed rowization) and retains prose instead. Surface that
        # refusal as a durable audit event -- apply_born_digital only carries
        # PageState fields, not events, so this is the one place per document
        # run where every refused page is known at once.
        #
        # Keyed on PageAssessment.native_table_lane_refused, set ONLY inside the
        # refusal branch in born_digital.py -- NOT re-derived from
        # ``text_is_rotated and has_tables``. ``has_tables`` is stamped before
        # the early non-born-digital returns, so that conjunction alone would
        # also fire on a rotated scanned/garbled page with a ruled table, where
        # the refusal branch never ran and there is no native text retained.
        from socr.core.audit_log import AuditEvent

        for pa in assessment.pages:
            if pa.native_table_lane_refused:
                state.events.append(
                    AuditEvent(
                        page_num=pa.page_num,
                        kind="landscape_page_refused",
                        engine="native",
                        detail=(
                            "native table reconstruction refused (dominant text direction "
                            "is rotated); prose retained, page routed to OCR"
                        ),
                    )
                )

        # GH-151 TICKET-B1 / GH-200 / GH-211: surface every table-structure
        # defect as a durable audit event -- EXACTLY ONE per affected page.
        #
        # Three independent flags can demote the same page: TR-3's per-region
        # geometry hard-fail, B1's grid-shape defect, and GH-200's header
        # attribution. They are genuinely independent (a header can be
        # destroyed on a perfectly rectangular grid, and TR-3's region check
        # runs separately from B1's grid check), so a page can carry more than
        # one. Emitting one event per flag double-counts that page in the CLI
        # failure totals and in any consumer counting events, so the causes are
        # collected and reported together in ``data["defects"]`` instead.
        #
        # Keyed SOLELY on the flags stamped by born_digital.py -- do NOT re-run
        # structure_check or re-derive any predicate here (the design note is
        # explicit: the flag is authoritative, this loop does not re-inspect
        # the grid).
        _DEFECT_DETAIL = {
            "unverifiable_table_region": (
                "native table region failed deterministic geometry verification; "
                "--native-only kept the native text without OCR and marked the page "
                "untrusted"
            ),
            "grid_shape": (
                "native table grid structurally defective (ragged widths and/or a "
                "detached label row)"
            ),
            "header_unattributed": (
                "native table header band not attributable to any emitted header cell "
                "(destroyed, not merely misplaced)"
            ),
        }
        for pa in assessment.pages:
            defects: list[str] = []
            if self.config.native_only and getattr(pa, "has_unverifiable_table_region", False):
                defects.append("unverifiable_table_region")
            if getattr(pa, "native_table_structure_defective", False):
                defects.append("grid_shape")
            if getattr(pa, "native_table_header_unattributed", False):
                defects.append("header_unattributed")
            if not defects:
                continue
            causes = "; ".join(_DEFECT_DETAIL[d] for d in defects)
            state.events.append(
                AuditEvent(
                    page_num=pa.page_num,
                    kind="table_structure_failed",
                    engine="native",
                    detail=(
                        f"{causes}. The native attempt is demoted to flagged WARNING "
                        "and can no longer pass as a trusted native page (a non-native "
                        "winner, if any, is unaffected)."
                    ),
                    data={"defects": defects},
                )
            )

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
                and not self._page_has_tables(page_num, ps)
            ):
                # Trust the native prose layer; math is spliced in Tier 1. This
                # also avoids whole-page VLM OCR that would degrade the prose.
                prose_pages.append(page_num)
                math_recovery_pages.add(page_num)
            elif self._is_trusted_native_without_ocr(page_num, ps):
                prose_pages.append(page_num)
            elif ps.is_born_digital and ps.native_text:
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
            # Build hints from PageState content-type vector (propagated by
            # apply_born_digital; no need to re-read _last_assessment here).
            page_hints: dict[int, dict] = {}
            for page_num in ocr_pages:
                ps = state.pages[page_num]
                if ps.has_tables or ps.has_equations:
                    page_hints[page_num] = {
                        "has_tables": ps.has_tables,
                        "has_equations": ps.has_equations,
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
            # GH-151 TICKET-B1 / GH-200 / #211: a page can reach prose_pages
            # carrying a table-distrust flag only via --native-only (the
            # non-native-only eligibility predicate already excludes has_tables
            # pages). Demote instead of shipping SUCCESS/audit_passed=True --
            # this is the no-reroute honouring of the flag: no extra OCR
            # attempt is triggered, the native text still ships, just flagged.
            # The TR-3 unverifiable mark stays explicitly scoped to
            # --native-only (#211); B1's grid/header flags are set only on that
            # path anyway.
            native_table_distrusted = bool(
                (self.config.native_only and ps.native_table_unverifiable)
                or getattr(ps, "native_table_structure_defective", False)
                or getattr(ps, "native_table_header_unattributed", False)
            )
            page_outputs.append(
                PageOutput(
                    page_num=page_num,
                    text=text,
                    status=(PageStatus.WARNING if native_table_distrusted else PageStatus.SUCCESS),
                    engine=engine,
                    audit_passed=not native_table_distrusted,
                    failure_mode=(
                        FailureMode.NATIVE_TABLE_STRUCTURE_FAILED
                        if native_table_distrusted
                        else FailureMode.NONE
                    ),
                )
            )
        if math_doc is not None:
            math_doc.close()

        # GH-36a: Deterministic display-equation region detection (model-free).
        # Runs on born-digital prose pages that carry the ``has_equations`` signal
        # and are NOT already handled by the corrupt-math recovery path.  Default-
        # off; gated by ``config.detect_equations``.  Saves crop PNGs beside
        # figures and records provenance in ``state.events`` — no text is modified.
        if self.config.detect_equations and self._last_assessment:
            clean_eq_pages = {
                pa.page_num
                for pa in self._last_assessment.pages
                if pa.has_equations and not pa.has_corrupt_math and pa.is_born_digital
            }
            eq_prose_pages = [p for p in prose_pages if p in clean_eq_pages]
            if eq_prose_pages:
                self._detect_and_crop_equations(state, eq_prose_pages, output_dir)

        # GH-36b: Clean-equation → LaTeX via local VLM + 1A structural gate + 1C sidecar.
        # Runs AFTER GH-36a detection (so crop paths are in state.events) and BEFORE
        # state.apply_result (so we can append sidecar blocks to prose page_outputs).
        # Gated by config.recover_clean_equations (default False).
        # Requires detect_equations to have run (no detected regions → no-op).
        if self.config.recover_clean_equations and self.config.detect_equations:
            self._attach_equation_latex_sidecars(state, page_outputs)

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
                    if po.status == PageStatus.SUCCESS and po.audit_passed:
                        passed_outputs.append(po)
                    else:
                        escalated_pages.append(po.page_num)
                        escalated_reasons[po.page_num] = po.failure_mode.value
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

        def native_fallback(
            page_num: int, failure_mode: FailureMode, error: str = ""
        ) -> PageOutput:
            ps = state.pages[page_num]
            if self._page_has_tables(page_num, ps):
                ps.native_table_structure_failed = True
            return PageOutput(
                page_num=page_num,
                text=ps.native_text or "",
                status=PageStatus.WARNING,
                engine="native",
                failure_mode=failure_mode,
                error=error,
                audit_passed=False,
            )

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
                        native_fallback(
                            page_num,
                            FailureMode.MODEL_UNAVAILABLE,
                            f"{engine.name} unavailable; native text used as fallback",
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
                        native_fallback(
                            po.page_num,
                            po.failure_mode
                            if po.failure_mode != FailureMode.NONE
                            else FailureMode.AUDIT_FAILED,
                            po.error or f"{engine.name} OCR failed; native text used as fallback",
                        )
                    )
                    continue
            final.append(po)

        if not self.config.quiet:
            ok = sum(1 for p in final if p.status == PageStatus.SUCCESS)
            console.print(f"  {ok}/{len(page_nums)} pages succeeded")

        return final

    def _assessment_for_page(self, page_num: int):
        if not self._last_assessment:
            return None
        return next((p for p in self._last_assessment.pages if p.page_num == page_num), None)

    def _page_has_tables(self, page_num: int, ps: PageState | None = None) -> bool:
        """Whether the born-digital detector found table-like structure."""
        if ps is not None and ps.has_tables:
            return True
        pa = self._assessment_for_page(page_num)
        return bool(pa and pa.has_tables)

    def _is_native_eligible_without_ocr(self, page_num: int, ps: PageState) -> bool:
        """Whether a page is native-eligible, WITHOUT the table exclusion.

        Same gates as ``_is_trusted_native_without_ocr`` minus the table check —
        shared by that predicate and by ``_is_chart_asset_page`` (GH-150 TICKET-B1),
        which needs eligibility without unconditionally losing to tables.
        """
        if not self.config.native_first or not ps.is_born_digital or not ps.native_text:
            return False

        if self.config.native_only:
            return True

        if ps.needs_ocr_enhancement:
            return False

        return True

    def _is_trusted_native_without_ocr(self, page_num: int, ps: PageState) -> bool:
        """Whether a page may bypass OCR and ship native text directly.

        Born-digital prose takes free native text. Pages with tables need the
        model/VLM path because PyMuPDF text often flattens grids even when the
        character layer is otherwise clean. ``--native-only`` remains the
        explicit override for born-digital pages — including table pages: it
        short-circuits before the table check, exactly as the pre-split
        predicate did (native_only returns True before ever reaching tables).

        GH-147 A2 narrows that override for the rotated+table conjunction: a
        rotated page's native table lane is already refused in
        ``born_digital.py`` (the rowizer would emit a transposed grid), so
        ``--native-only`` must not bypass OCR for it either. Horizontal table
        pages and rotated table-less pages are unaffected and still take the
        unconditional ``native_only`` short-circuit.
        """
        if not self._is_native_eligible_without_ocr(page_num, ps):
            return False
        if self.config.native_only:
            pa = self._assessment_for_page(page_num)
            rotated = bool(pa and pa.text_is_rotated)
            if not (rotated and self._page_has_tables(page_num, ps)):
                return True
        return not self._page_has_tables(page_num, ps)

    def _is_agentic_trusted_native(self, page_num: int, ps: PageState) -> bool:
        """Backward-compatible alias for the agentic native-bypass predicate."""
        return self._is_trusted_native_without_ocr(page_num, ps)

    # ------------------------------------------------------------------
    # PP-7: Chart-asset routing lane
    # ------------------------------------------------------------------

    def _is_chart_asset_page(self, page_num: int, ps: PageState, pdf_path: Path) -> bool:
        """PP-7: return True when a born-digital page carries chart marks.

        This predicate answers *eligibility*, not the final route. Fires when ALL of:
          - The page is born-digital with native text (otherwise it goes to OCR anyway).
          - It IS native-eligible (i.e. ``_is_native_eligible_without_ocr`` returns True —
            no needs_ocr_enhancement, native_first enabled). Tables no longer exclude a
            page from this check (GH-150 TICKET-B1), because a mixed chart+table page must
            still be *detectable* here in order to be arbitrated by the caller.
          - ``has_chart_marks`` detects at least one vector chart cluster OR embedded
            raster image on the page.

        Eligibility is NOT the page-level chart lane. The caller (``_phase_agentic``)
        splits eligible pages in two:

          - **chart-only** (no table signal) → the page-level chart-asset lane, which
            appends a whole-page PNG ref after ``ps.native_text``. Correct only when the
            chart IS the page.
          - **chart + table** → the *normal* route, where ``rowize_from_words_chart_aware``
            emits an inline placeholder at the chart's own y-position. Appending would
            sink the chart below every table that followed it and break source order.
            The caller emits a ``chart_table_arbitration`` audit event for these.

        This predicate is called BEFORE the native-bypass branch in the agentic loop so
        chart pages are intercepted instead of shipping as pure native word-salad prose.

        Do NOT call this predicate when ``_is_native_eligible_without_ocr`` returns False
        (i.e. the page would go to the OCR ladder anyway); there is nothing to intercept.
        """
        # Must be native-eligible first (chart lane is a sub-case of native).
        if not self._is_native_eligible_without_ocr(page_num, ps):
            return False
        # Open the PDF page and run the vector detector.
        try:
            import fitz

            with fitz.open(pdf_path) as doc:
                page = doc[page_num - 1]
                return has_chart_marks(page)
        except Exception as exc:
            logger.debug(
                "_is_chart_asset_page p%d: fitz open/detect failed (%s); defaulting to False",
                page_num,
                exc,
            )
            return False

    def _render_chart_page_png(
        self,
        pdf_path: Path,
        page_num: int,
        figures_dir: Path,
    ) -> str:
        """PP-7: render the full page as a PNG and save it to ``figures_dir``.

        Renders at ``RENDER_DPI`` (same as FigureExtractor) and saves with a
        deterministic name ``chart_page_{page_num}.png`` so the path is
        predictable and the file is distinguishable from regular extracted figures.

        Returns the saved path as a string.

        Raises ``RuntimeError`` on any render failure — the caller is responsible
        for fail-closed handling (hard audit error, never silent degradation).
        """
        from socr.figures.extractor import RENDER_DPI

        figures_dir.mkdir(parents=True, exist_ok=True)
        try:
            import fitz
            from PIL import Image

            with fitz.open(pdf_path) as doc:
                page = doc[page_num - 1]
                mat = fitz.Matrix(RENDER_DPI / 72, RENDER_DPI / 72)
                pix = page.get_pixmap(matrix=mat)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        except Exception as exc:
            raise RuntimeError(
                f"chart-lane PNG render failed for p{page_num} of {pdf_path.name}: {exc}"
            ) from exc

        out_path = figures_dir / f"chart_page_{page_num}.png"
        try:
            img.save(out_path)
        except Exception as exc:
            raise RuntimeError(f"chart-lane PNG save failed for p{page_num}: {exc}") from exc
        return str(out_path)

    def _render_chart_region_pngs(
        self,
        pdf_path: Path,
        page_num: int,
        native_text: str,
        figures_dir: Path,
    ) -> str:
        """TR-2: render chart region bbox crops and update placeholder paths.

        ``rowize_from_words_chart_aware`` embeds chart region placeholders of
        the form ``![chart region N](chart_region_pP_N.png)`` in
        ``ps.native_text``.  Those bare filenames have no path prefix, so
        ``strip_phantom_images`` (called with ``output_dir=doc_dir``) strips
        them because the files do not exist on disk.

        This method:
        1. Detects any ``chart_region_p{page_num}_{i}.png`` placeholders in
           ``native_text``.
        2. Opens the PDF page, calls ``chart_region_bboxes`` to get the
           cluster bboxes (same as the rowizer used when generating the
           placeholders).
        3. Renders each bbox as a cropped PNG and saves to ``figures_dir``.
        4. Returns updated ``native_text`` with
           ``{figures_dir.name}/{filename}`` paths so the files are resolvable
           against ``doc_dir`` and survive ``strip_phantom_images``.

        On any error, returns ``native_text`` unchanged (fail-open: the
        placeholder stays, strip_phantom_images will remove it, and the page
        is treated as chart-asset-failed by the provenance audit).

        Never raises.
        """
        import re

        pattern = re.compile(rf"\(chart_region_p{page_num}_(\d+)\.png\)")
        if not pattern.search(native_text):
            return native_text

        try:
            import fitz

            from socr.figures.extractor import RENDER_DPI
            from socr.tables.reconstruct import chart_region_bboxes

            figures_dir.mkdir(parents=True, exist_ok=True)

            with fitz.open(str(pdf_path)) as _doc:
                _page = _doc[page_num - 1]
                bboxes = chart_region_bboxes(_page)

            rendered_indices: set[int] = set()
            for m in pattern.finditer(native_text):
                region_idx = int(m.group(1))
                if region_idx < 1 or region_idx > len(bboxes):
                    continue
                bbox = bboxes[region_idx - 1]
                fname = f"chart_region_p{page_num}_{region_idx}.png"
                out_path = figures_dir / fname
                try:
                    with fitz.open(str(pdf_path)) as _doc:
                        _page = _doc[page_num - 1]
                        mat = fitz.Matrix(RENDER_DPI / 72, RENDER_DPI / 72)
                        pix = _page.get_pixmap(matrix=mat, clip=bbox)
                        pix.save(str(out_path))
                    rendered_indices.add(region_idx)
                    logger.debug("TR-2: saved chart region PNG %s", out_path)
                except Exception as render_exc:
                    logger.warning(
                        "TR-2 chart region PNG render failed for p%d region %d: %s",
                        page_num,
                        region_idx,
                        render_exc,
                    )

            if not rendered_indices:
                # No PNGs saved — return original text so strip_phantom_images
                # removes the placeholder and the audit records the failure.
                return native_text

            # Update placeholder paths to ``figures/{filename}`` so the files
            # resolve correctly against ``doc_dir`` in strip_phantom_images.
            figures_dir_name = figures_dir.name

            def _replace_path(m: re.Match[str]) -> str:
                region_idx = int(m.group(1))
                if region_idx in rendered_indices:
                    return f"({figures_dir_name}/chart_region_p{page_num}_{region_idx}.png)"
                return m.group(0)  # keep original if PNG was not saved

            return pattern.sub(_replace_path, native_text)

        except Exception as exc:
            logger.warning(
                "TR-2 _render_chart_region_pngs failed for p%d: %s",
                page_num,
                exc,
            )
            return native_text

    def _render_d3_floor_png(
        self,
        pdf_path: Path,
        page_num: int,
        figures_dir: Path,
    ) -> str:
        """TR-3: render a PNG for a D3 fail-closed floor page and return an image ref.

        For a born-digital table page that failed both OCR and per-region geometry
        verification (D3 floor), the human must still be able to SEE the table.
        This method renders a full-page PNG of the failed page into ``figures_dir``
        and returns a markdown image reference string
        (``![Failed table page N](figures/failed_table_p{N}.png)``).

        A full-page render is used rather than a region crop because the page
        has no verified region bboxes at this point — the per-region verifier
        flagged a hard-fail, which means the detected bboxes cannot be trusted
        to bound the actual table content correctly.  The full page is always
        safe: it preserves everything visible.

        Returns "" on any render failure so the caller can safely append it to
        the failure marker (an empty string produces marker-only output, still
        fail-closed and never a plausible-but-wrong table).
        """
        try:
            fname = f"failed_table_p{page_num}.png"
            saved = self._render_chart_page_png(pdf_path, page_num, figures_dir)
            # _render_chart_page_png saves chart_page_{page_num}.png; rename to
            # our D3-specific name so the file is clearly identifiable in figures/.
            saved_path = Path(saved)
            d3_path = saved_path.parent / fname
            saved_path.rename(d3_path)
            figures_dir_name = figures_dir.name
            ref = f"![Failed table page {page_num}]({figures_dir_name}/{fname})"
            logger.debug("TR-3 D3 floor: saved full-page PNG %s", d3_path)
            return ref
        except Exception as exc:
            logger.warning(
                "TR-3 D3 floor PNG render failed for p%d (%s); marker only",
                page_num,
                exc,
            )
            return ""

    # ------------------------------------------------------------------
    # GH-86: per-page VLM placeholder cleanup (agentic pre-flush seam)
    # ------------------------------------------------------------------

    def _sanitize_agentic_page_image_refs(
        self,
        state: DocumentState,
        page_num: int,
        page_out: PageOutput,
        doc_dir: Path,
    ) -> None:
        """Strip VLM sentinel / phantom image refs before provisional flush.

        PNG extraction and inline embedding remain in the assemble-time
        ``_describe_and_embed_figures`` phase (PP-4 / PP-5 ``:pre-figures``
        contract).  This hook only cleans page text so dead ``image-url`` /
        ``image.png`` placeholders never reach fragments or the stitched body.
        """
        normalizer = OutputNormalizer()
        text = page_out.text or ""
        if not text.strip():
            return

        had_vlm_placeholders = normalizer.text_has_vlm_image_placeholders(text)
        text = normalizer.strip_phantom_images(text, output_dir=doc_dir)

        figures_disabled = not (self.config.save_figures or self.config.describe_figures)
        if had_vlm_placeholders and figures_disabled:
            ps = state.pages.get(page_num)
            if ps and (ps.has_figures or ps.has_tables):
                from socr.core.audit_log import AuditEvent

                state.events.append(
                    AuditEvent(
                        page_num=page_num,
                        kind="figure_placeholder_unresolved",
                        engine="",
                        detail=(
                            "VLM image placeholder stripped; figure extraction disabled "
                            "(save_figures=False)"
                        ),
                    )
                )

        page_out.text = text

    # ------------------------------------------------------------------
    # GH-96: table escalation lane
    # ------------------------------------------------------------------

    def _resolve_table_escalation_provider(self, available: list):
        """Cheapest non-local provider from the ALREADY tier-filtered ladder.

        Derived rather than named. Naming ``EngineType.GEMINI`` literally would
        bypass ``--strict-local`` entirely: that flag filters ``available`` by tier
        before the ladder is built, but ``run_provider`` accepts any engine, so a
        hardcoded escalation engine would still fire on exactly the configuration
        the reference run used. Choosing from ``available`` makes ``--strict-local``
        and ``--max-cost-per-page`` suppress the lane for free.
        """
        if not getattr(self.config, "escalate_ambiguous_tables", False):
            return None
        from socr.core.providers import TIER_LOCAL

        candidates = [p for p in available if p.tier != TIER_LOCAL and p.supports_per_page]
        if not candidates:
            return None
        return min(candidates, key=lambda p: p.cost_per_page_usd)

    def _table_page_needs_escalation(
        self, state: DocumentState, page_num: int, page, ps, bo
    ) -> bool:
        """True when the emitted table disagrees with the page's native text layer.

        Calibrated against whether escalation actually helps: 100% recall and 69%
        precision, versus 56% for ``dualpass_flagged``, which fires on every table
        page in the reference document and so cannot discriminate at all.

        GH-113: the ground truth must look like a GRID before anything is compared.
        The native row parser will read two numeric lines of prose, or a chart's
        axis labels, as a table — on a full document that escalated pages holding a
        cover date ("November=['2022']"), a fragment of "4 per cent"
        ("cent=['4']") and chart axes, each time paying for a cloud call to compare
        an empty result against an empty result. The trigger's measured 69%
        precision came from 16 hand-picked pages that all genuinely contained
        tables, so this class of page was never in the calibration.

        ``PageState.has_tables`` does NOT discriminate here — it is True on those
        pages too — so the test is structural: a grid needs at least two value
        columns, and at least two rows sharing that width. Both are minimums that
        follow from what a table is, not tuned cutoffs.

        Deliberately not "most rows share the modal width": a real +89-point
        recovery on the reference document has only 7 of 17 rows at its modal
        width, and a majority rule would have refused it.

        The grid predicate itself lives in ``native_rows.rows_establish_grid``,
        shared with the GH-96 exactness metric's own not-scorable gate (#123
        TICKET-B1) —
        one predicate, not two copies drifting apart.

        #123 TICKET-C1: this is also the one place every table page's incumbent
        text is scored against its own native layer regardless of whether
        escalation ends up firing, so it doubles as the surfacing point for two
        kinds of content loss that used to reach no surface at all: a not-scorable
        page (B1's grid gate — previously a loud, wrong 0.0% became a silent
        ``pct=None``) and a page carrying unexplained lanes (B2's lane alignment
        found a native column the emitted table has nowhere to put). Both are
        recorded as ``AuditEvent``s so ``tables_trust`` surfaces them at the
        document level; if the page is later escalated and accepted, the existing
        ``table_escalation_accepted`` resolution machinery clears them, same as any
        other distrust kind recorded against text that no longer ships.
        """
        from socr.benchmark.table_exactness import score_page
        from socr.core.audit_log import AuditEvent
        from socr.tables.native_rows import native_rows_from_page, rows_establish_grid

        try:
            gt_rows = native_rows_from_page(page)
        except Exception:
            return False
        if not rows_establish_grid(gt_rows):
            state.events.append(
                AuditEvent(
                    page_num=page_num,
                    kind="table_not_scorable",
                    detail=(
                        f"native text layer parsed {len(gt_rows)} row(s) that do not "
                        "form a grid; not scorable against ground truth"
                    ),
                )
            )
            return False

        try:
            report = score_page(page, bo.text or "")
        except Exception:
            return False

        if report.ceiling_note:
            state.events.append(
                AuditEvent(
                    page_num=page_num,
                    kind="table_not_scorable",
                    detail=report.ceiling_note,
                )
            )
        elif report.unexplained_lanes:
            state.events.append(
                AuditEvent(
                    page_num=page_num,
                    kind="table_unexplained_lanes",
                    detail=(
                        f"{report.unexplained_lanes} native lane(s) carry values in "
                        "matched rows but map to no emitted column"
                    ),
                    data={"unexplained_lanes": report.unexplained_lanes},
                )
            )

        return report.pct is not None and report.pct < 100.0

    def _surface_table_scoring(
        self, state: DocumentState, page_num: int, ps, bo: PageOutput
    ) -> None:
        """Score a table page against its native layer even with no escalation lane.

        #123 TICKET-C2: ``_table_page_needs_escalation`` is the one place both
        kinds of content loss it documents (a not-scorable page, an unexplained
        lane) reach ``tables_trust`` — but until now it only ran from
        ``_escalate_table_page``, which never runs unless a non-local provider is
        configured. socr is local-first by design, so on a local-only run (no
        cloud provider, ``--strict-local``, or a cost cap suppressing the lane)
        every not-scorable page and unexplained-lane page shipped with no trace,
        same as before TICKET-C1. This is the escalation-independent twin call:
        same predicate, same single emitter, no second contract.

        Measured on the OBR reference document (68 pages, 18 clearing the grid
        predicate): ``native_rows_from_page`` + ``score_page`` cost ~137ms mean
        per page, worst single page ~540ms. Note the baseline this is weighed
        against: **65 of those 68 pages are born-digital trusted native text and
        never run a VLM at all**, so on a document like this the cost is not
        hidden behind model inference — it is close to pure addition. It is
        accepted anyway, and kept off an opt-out flag, because the caller gates
        on ``_page_has_tables``: prose pages skip it entirely, and a page with
        table-like structure is exactly the page where silently shipping
        unexplained lanes or an unscorable grid is the failure this exists to
        prevent. Chart pages still reach it — ``has_tables`` is True there, which
        is why TICKET-B1 was needed — so the not-scorable surface keeps them.
        """
        try:
            import fitz

            with fitz.open(state.handle.path) as doc:
                page = doc[page_num - 1]
                self._table_page_needs_escalation(state, page_num, page, ps, bo)
        except Exception as exc:
            logger.warning(
                "table scoring failed on p%d (%s); no distrust events emitted",
                page_num,
                exc,
            )

    def _escalate_table_page(
        self,
        state: DocumentState,
        page_num: int,
        ps,
        bo: PageOutput,
        profile,
        run_provider,
        pdf_path,
    ) -> bool:
        """Re-read one table page with *profile*; keep it only if it measures better.

        Returns True when the lane should be disabled for the rest of the document
        (a wedged provider), False otherwise.

        Every rejection path keeps the incumbent text untouched, so the worst case
        is a wasted call.
        """
        import concurrent.futures

        from socr.core.audit_log import AuditEvent
        from socr.tables.escalation_decision import decide_escalation

        try:
            import fitz

            with fitz.open(pdf_path) as doc:
                page = doc[page_num - 1]
                if not self._table_page_needs_escalation(state, page_num, page, ps, bo):
                    return False
                incumbent_text = bo.text or ""

                # A cloud CLI has no timeout of its own and was observed wedged for
                # 97 minutes. Escalation runs inline in the page-major loop, so an
                # unbounded call stalls the entire document. The worker is released
                # rather than joined: the subprocess may outlive us, but the loop
                # must not.
                deadline = float(getattr(self.config, "escalation_timeout_sec", 120.0))
                ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                future = ex.submit(run_provider, profile.engine, page_num)
                try:
                    out = future.result(timeout=deadline)
                    ex.shutdown(wait=False)
                except concurrent.futures.TimeoutError:
                    ex.shutdown(wait=False)
                    state.events.append(
                        AuditEvent(
                            page_num=page_num,
                            kind="table_escalation_timeout",
                            engine=profile.engine.value,
                            detail=(
                                f"escalation exceeded {deadline:.0f}s; lane disabled "
                                "for the rest of this document"
                            ),
                        )
                    )
                    return True

                # `_run_engine_on_pages` converts a failed engine call into a
                # native-text PageOutput. Assigning that would replace a structured
                # table with flattened native text - silent content loss - so the
                # candidate is only considered when the intended engine actually
                # answered.
                if out is None or out.engine != profile.engine.value:
                    state.events.append(
                        AuditEvent(
                            page_num=page_num,
                            kind="table_escalation_refused",
                            engine=profile.engine.value,
                            detail=(
                                f"provider returned engine={getattr(out, 'engine', None)!r}; "
                                "not the escalation engine, candidate discarded"
                            ),
                        )
                    )
                    return False
                if out.status is not PageStatus.SUCCESS or not (out.text or "").strip():
                    state.events.append(
                        AuditEvent(
                            page_num=page_num,
                            kind="table_escalation_refused",
                            engine=profile.engine.value,
                            detail=f"candidate status={out.status}, no usable text",
                        )
                    )
                    return False

                decision = decide_escalation(page, incumbent_text, out.text)

            # Cost is recorded by hand: `route_page` does this for ladder calls, and
            # a bare `run_provider` does not, so without it the document
            # under-reports what it spent.
            state.engine_runs.append(
                EngineResult(
                    document_path=pdf_path,
                    engine=profile.engine.value,
                    status=DocumentStatus.SUCCESS,
                    pages=[],
                    pages_processed=1,
                    cost=profile.cost_per_page_usd,
                )
            )

            if not decision.accepted:
                if not self.config.quiet:
                    console.print(
                        f"  [dim]Escalation p{page_num} rejected ({profile.engine.value}): "
                        f"{decision.reason}[/dim]"
                    )
                state.events.append(
                    AuditEvent(
                        page_num=page_num,
                        kind="table_escalation_rejected",
                        engine=profile.engine.value,
                        detail=decision.reason,
                        data={"gate": decision.gate, "delta": decision.delta},
                    )
                )
                return False

            bo.text = out.text
            ps.attempts.append(out)
            self._clear_fail_closed_flags(state, page_num, ps, profile)
            if not self.config.quiet:
                console.print(
                    f"  [green]Escalated p{page_num}[/green] via {profile.engine.value}: "
                    f"{decision.reason}"
                )
            state.events.append(
                AuditEvent(
                    page_num=page_num,
                    kind="table_escalation_accepted",
                    engine=profile.engine.value,
                    detail=decision.reason,
                    data={"gate": decision.gate, "delta": decision.delta},
                )
            )
            return False

        except Exception as exc:  # a failed escalation must never lose a page
            logger.warning("table escalation failed on p%d (%s); keeping text", page_num, exc)
            return False

    @staticmethod
    def _clear_fail_closed_flags(state: DocumentState, page_num: int, ps, profile) -> None:
        """Release a fail-closed page once a candidate has measurably improved it.

        ``manifest._winning_page_output`` re-derives the winner from PageState flags,
        not from the text: a page still marked unverifiable ships the
        "[page N failed: unverifiable table]" marker even after its text is fixed,
        so the fragment and the manifest would disagree. Clearing is recorded
        separately because releasing a fail-closed page is a consequential act.
        """
        from socr.core.audit_log import AuditEvent

        cleared = [
            name
            for name in (
                "native_table_structure_failed",
                "native_table_structure_defective",  # GH-151 TICKET-B1
                "native_table_header_unattributed",  # GH-200
                "native_table_unverifiable",
                "scanned_table_evidence_failed",
            )
            if getattr(ps, name, False)
        ]
        if not cleared:
            return
        for name in cleared:
            setattr(ps, name, False)
        state.events.append(
            AuditEvent(
                page_num=page_num,
                kind="table_escalation_recovered_fail_closed",
                engine=profile.engine.value,
                detail=f"cleared {', '.join(cleared)} after a measured improvement",
            )
        )

    # ------------------------------------------------------------------
    # GH-97: degenerate table-row repetition guard (agentic pre-flush seam)
    # ------------------------------------------------------------------

    @staticmethod
    def _guard_agentic_page_table_repetition(
        state: DocumentState,
        page_num: int,
        page_out: PageOutput,
        is_native: bool = False,
    ) -> None:
        """Truncate runaway repeated table rows before the provisional flush.

        Placed here rather than in ``OutputNormalizer`` because that runs at the
        engine boundary, before table assembly: on the GH-97 reference run a
        15-character empty row repeated 865 times survived normalization (too
        short for the generic rule's 20-char floor) and reached 867 of the 3177
        lines of the assembled document - 27% of it - with no audit event and no
        doc-level signal.

        Scope is deliberately the agentic path only. The observed runaway, and every
        instance found since, came from the per-page VLM loop; the phase-major paths
        (single-engine, multi-engine, consensus, repair) have never produced one in
        this corpus. Extending the guard there would mean re-deriving its bound on
        output nobody has measured, so they are left to ``OutputNormalizer``'s
        generic rule until there is evidence they need more.

        Removal is content-safe: every dropped line is byte-identical to one
        that is kept. The event is recorded unconditionally so the failure is
        visible in ``audit_log.json`` and, via ``_repetition_truncated_note``,
        in the document-level error surface.
        """
        text = page_out.text or ""
        if not text:
            return

        # GH-98: never applied to a character-exact native extraction. The guard
        # exists because a VLM degenerates into repeating a row; that cannot happen
        # when the text came straight from the PDF's own layer. And there the
        # collapse is not safe: #97's guarantee is "no DISTINCT line is lost", which
        # is weaker than "no information is lost" - row multiplicity changes. A
        # wide-format appendix table with five consecutive all-zero rows and no
        # per-row label is legitimate content on a native page, and truncating it to
        # two silently drops three real rows.
        if is_native:
            return

        cleaned, removed = collapse_repeated_table_rows(text)
        if not removed:
            return

        page_out.text = cleaned

        from socr.core.audit_log import AuditEvent

        state.events.append(
            AuditEvent(
                page_num=page_num,
                kind="table_row_repetition_truncated",
                engine=page_out.engine or "",
                detail=(
                    f"dropped {removed} consecutive duplicate table row(s); "
                    f"kept at most {MAX_CONSECUTIVE_IDENTICAL_TABLE_ROWS} copies"
                ),
                data={"rows_removed": removed},
            )
        )

    @staticmethod
    def _repetition_truncated_note(events: list) -> str | None:
        """Document-level one-liner naming pages whose tables were truncated.

        Returns ``None`` when no page was affected, so a clean run's error
        surface is left untouched.
        """
        pages = sorted({e.page_num for e in events if e.kind == "table_row_repetition_truncated"})
        if not pages:
            return None
        return (
            f"page(s) {', '.join(str(n) for n in pages)}: degenerate table row repetition truncated"
        )

    # ------------------------------------------------------------------
    # PP-2: Judge deadline adapter (stays in orchestrator; keeps agentic.py
    # contract unchanged). Wraps any PageJudge in a wall-clock deadline so a
    # wedged VLM judge cannot block the orchestrator thread indefinitely.
    # ------------------------------------------------------------------

    class _TimeoutJudge:
        """Wraps a PageJudge with a ThreadPoolExecutor wall-clock deadline.

        On timeout the wrapper returns a rejection (accept=False,
        reason="judge timeout"), which causes ``route_page`` to record the
        attempt and escalate normally.  If the backend is also not idle after
        the timeout, the caller should set ``backend_degraded`` and halt.

        ``timeout_sec`` is a soft wall-clock bound in seconds.  ``None``
        disables the wrapper (forward to the inner judge directly).
        """

        def __init__(self, inner, timeout_sec: float | None) -> None:
            self._inner = inner
            self._timeout_sec = timeout_sec

        def assess(self, output, provider):
            import concurrent.futures

            from socr.pipeline.agentic import AcceptDecision

            if self._timeout_sec is None:
                return self._inner.assess(output, provider)

            ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = ex.submit(self._inner.assess, output, provider)
            try:
                result = future.result(timeout=self._timeout_sec)
                ex.shutdown(wait=False)
                return result
            except concurrent.futures.TimeoutError:
                future.cancel()
                ex.shutdown(wait=False)
                logger.warning(
                    "judge timed out on page %s (%.2fs) — rejecting",
                    output.page_num,
                    self._timeout_sec,
                )
                return AcceptDecision(accept=False, reason="judge timeout")

    # ------------------------------------------------------------------
    # Agentic: cost-aware per-page routing (replaces backbone+score+repair)
    # ------------------------------------------------------------------

    def _phase_agentic(self, state: DocumentState, output_dir: Path) -> None:
        """PP-2 fused page-major loop: one pass over ALL pages (native + OCR).

        Born-digital prose takes free native text; every OCR page is routed
        through the cost ladder.  After each page is final, a PROVISIONAL
        fragment + sidecar (``terminal=False``) is flushed for crash recovery.
        The authoritative fragment bytes come from ``_rewrite_all_fragments``
        at the end of ``_phase_assemble`` (fork A / assemble-authoritative).

        Per-page lifecycle (post-route):
          1. ``_reread_page_tables`` (PP-3) — OCR pages with tables only.
          2. Per-page equation detect+crop (GH-36a) — behind ``detect_equations``.
          3. Per-page equation LaTeX sidecar (GH-36b) — behind ``recover_clean_equations``.
          4. Per-page VLM placeholder cleanup (GH-86) — strip sentinel image
             refs before flush; PNG embed runs later in ``_describe_and_embed_figures``.
          5. Write-through blob cache (``put_page``).
          6. Provisional fragment + sidecar flush (``terminal=False``).

        Cascade HALT (fork B / judge-guard):
          - The judge is wrapped in ``_TimeoutJudge`` so a wedged VLM judge
            cannot block the orchestrator thread.
          - When ``route_page`` records a provider timeout AND ``probe_ollama_idle``
            reports the backend is NOT responding, the document-level
            ``backend_degraded`` latch is set: pages processed so far are already
            flushed; the loop breaks without firing any VLM call for page N+1.
          - An AuditEvent of kind ``partial_save_vlm_timeout`` is appended and the
            final ``EngineResult`` carries ``error="PARTIAL_SAVE_VLM_TIMEOUT"`` so
            callers and tests can detect the HALT condition unambiguously.

        Phase 4c (dual-pass tables) is gated OFF for agentic mode in
        ``process()`` so tables are never re-read twice.

        ``_classify`` remains doc-wide (``_phase_analyze``); the fused loop
        handles only the post-classification per-page lifecycle (fork C2).
        """
        from socr.core.providers import provider_ladder
        from socr.pipeline.agentic import DEFAULT_PROVIDER_TIMEOUTS

        if not self.config.quiet:
            console.print("\n[cyan]Agentic routing[/cyan] (cost-ordered, judge-gated)")

        # -- Doc-scoped native-fallback list (needed by run_provider stub) --------
        # Collect OCR pages first, to build native_fallback_pages before the loop.
        ocr_pages: list[int] = []
        for page_num, ps in sorted(state.pages.items()):
            if not self._is_agentic_trusted_native(page_num, ps):
                ocr_pages.append(page_num)

        native_fallback_pages = [
            p
            for p in ocr_pages
            if self.config.native_first
            and state.pages[p].is_born_digital
            and state.pages[p].native_text
        ]

        # -- PP-5: provider-INDEPENDENT terminal-resume PRE-PASS -----------------
        # Resuming a TERMINAL page loads its fragment from disk and needs NO live
        # provider.  Do this BEFORE the empty-ladder early-return below so that a
        # re-run with no provider available (e.g. ollama down) still restores the
        # pages that were already terminal — instead of treating them as lost.
        # Each resumed page is removed from ``ocr_pages`` (and the native-fallback
        # list) so the empty-ladder check only fires for the genuinely
        # UNresumable OCR pages, and is recorded in ``resumed_pages`` so the main
        # loop below skips it (no double-restore / double-cost-count).  The skip
        # is the SAME conservative gate used in the loop (terminal + fingerprint +
        # checksum + SUCCESS + readable fragment), so semantics are unchanged.
        resumed_pages: set[int] = set()
        for page_num in list(ocr_pages):
            resumed = self._load_terminal_page(state, page_num, output_dir)
            if resumed is not None:
                self._restore_terminal_page_state(state, page_num, resumed, output_dir)
                resumed_pages.add(page_num)
                if not self.config.quiet:
                    console.print(
                        f"  p{page_num}: [dim]resumed (terminal ledger hit, pre-pass)[/dim]"
                    )
        if resumed_pages:
            ocr_pages = [p for p in ocr_pages if p not in resumed_pages]
            native_fallback_pages = [p for p in native_fallback_pages if p not in resumed_pages]

        # -- GH-150 TICKET-B1: precompute chart-vs-table arbitration winners ----
        # Scan ALL of state.pages, not just ocr_pages: a page with chart marks and
        # NO table signal is trusted-native and never appears in ocr_pages, but it
        # still must be caught here or the loop's cached-set gate (below) would
        # drop it back to plain native prose.
        #
        # A page carrying BOTH chart marks and a table signal is deliberately NOT a
        # chart winner. The page-level lane appends its PNG ref after the whole of
        # ``ps.native_text`` (see ``chart_body`` below), which is correct only when
        # the chart IS the page. On a mixed page the chart sits between other
        # regions, so appending sinks it below every table that followed it and
        # breaks source order. Such pages keep their normal route, where
        # ``rowize_from_words_chart_aware`` emits an inline placeholder at the
        # chart's own y-position and ``_render_chart_region_pngs`` resolves it —
        # position-preserving by construction.
        chart_only_pages: set[int] = set()
        chart_mixed_pages: set[int] = set()
        for pn, ps in sorted(state.pages.items()):
            if pn in resumed_pages:
                continue
            if not self._is_chart_asset_page(pn, ps, state.handle.path):
                continue
            if self._page_has_tables(pn, ps):
                chart_mixed_pages.add(pn)
            else:
                chart_only_pages.add(pn)
        chart_winner_pages: set[int] = chart_only_pages

        # Record the arbitration for every mixed page. The decision is made here,
        # so the event is emitted here — not inside the chart lane, which mixed
        # pages no longer enter. No silent routing: a page whose chart competed
        # with a table always leaves a trace naming which lane took it and why.
        if chart_mixed_pages:
            from socr.core.audit_log import AuditEvent as _ArbEvent

            for pn in sorted(chart_mixed_pages):
                state.events.append(
                    _ArbEvent(
                        page_num=pn,
                        kind="chart_table_arbitration",
                        engine="chart_region",
                        detail=(
                            "both chart and table signals fired; the page is held out "
                            "of the page-level chart lane and keeps its normal route, "
                            "so the chart is represented inline at its own position "
                            "rather than appended after the page text"
                        ),
                        data={
                            "winner": "chart_region",
                            "loser": "chart_asset_page",
                            "chart_marks": True,
                            "table_signal": True,
                            "rationale": (
                                "the page-level lane appends its PNG ref after all "
                                "page text, which breaks source order when the chart "
                                "is one region among several"
                            ),
                        },
                    )
                )
        # Prune chart winners from BOTH lists. As of this ticket the prune is a
        # NO-OP by construction, and that is worth stating rather than implying
        # otherwise: ``chart_winner_pages`` is exactly ``chart_only_pages``, whose
        # members have no table signal, so ``_is_trusted_native_without_ocr``
        # returns True for them — they never entered ``ocr_pages`` (built from
        # ``not _is_agentic_trusted_native``) and therefore never entered
        # ``native_fallback_pages`` (a subset of it).
        #
        # It is kept as a self-enforcing invariant: if ``chart_winner_pages`` is
        # ever widened to include pages that DO carry a table signal, those pages
        # would be in both lists, and the empty-ladder early return below would
        # stamp them WARNING/MODEL_UNAVAILABLE (CI's exact no-provider state)
        # before the loop ever ran. The prune makes that widening safe.
        #
        # Note what this does NOT cover: mixed chart+table pages are deliberately
        # not winners, so they stay in both lists and ARE stamped unavailable on a
        # no-provider run — correctly, since they genuinely need the ladder.
        #
        # Pruning alone would not stop routing in any case: the loop's else branch
        # calls route_page unconditionally for non-native, non-chart pages, so the
        # cached-set loop gate (see the chart-asset-lane branch below) is the
        # load-bearing half.
        if chart_winner_pages:
            ocr_pages = [p for p in ocr_pages if p not in chart_winner_pages]
            native_fallback_pages = [
                p for p in native_fallback_pages if p not in chart_winner_pages
            ]

        # -- Doc-scoped provider setup -------------------------------------------
        available = self._available_engines_for_agentic()
        if self.config.strict_local:
            from socr.core.providers import TIER_LOCAL

            available = [p for p in available if p.tier == TIER_LOCAL]
        ladder = provider_ladder(
            available, per_page_only=True, max_cost_per_page=self.config.max_cost_per_page
        )
        # GH-96: escalation provider, chosen from the ALREADY tier-filtered list so
        # --strict-local and --max-cost-per-page suppress the lane for free.
        _escalation_profile = self._resolve_table_escalation_provider(available)
        _escalation_degraded = False

        if not ladder and ocr_pages:
            logger.warning("agentic: no OCR providers available; OCR pages left unprocessed")
            if not self.config.quiet:
                console.print("  [red]No OCR providers available[/red]")
            for page_num in native_fallback_pages:
                ps = state.pages[page_num]
                if self._page_has_tables(page_num, ps):
                    ps.native_table_structure_failed = True
                fallback = PageOutput(
                    page_num=page_num,
                    text=ps.native_text or "",
                    status=PageStatus.WARNING,
                    engine="native",
                    failure_mode=FailureMode.MODEL_UNAVAILABLE,
                    error="no OCR providers available; native text used as fallback",
                    audit_passed=False,
                    cost_usd=0.0,
                )
                ps.attempts.append(fallback)
            return

        # Snapshot the ladder for manifest provenance (B3).
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

        # Judge provenance (B3) is recorded INSIDE _build_page_judge, from the
        # judge it actually built — not from _resolve_judge_model, which reports
        # what was wanted rather than what ran (#133).
        _inner_judge = self._build_page_judge(state)

        # Use calibrated defaults when no explicit override is configured.
        provider_timeout = (
            getattr(self.config, "agentic_provider_timeout", None) or DEFAULT_PROVIDER_TIMEOUTS
        )

        # Wrap the judge in a deadline adapter so a wedged VLM judge cannot
        # block the orchestrator thread (fork B: orchestrator-side adapter).
        # Use the maximum configured provider timeout as the judge bound.
        _judge_timeout: float | None = None
        if provider_timeout:
            _judge_timeout = max(provider_timeout.values()) if provider_timeout else None
        judge = self._TimeoutJudge(_inner_judge, _judge_timeout)

        if not self.config.quiet:
            ladder_str = " -> ".join(f"{p.engine.value}(${p.cost_per_page_usd:g})" for p in ladder)
            console.print(f"  ladder: {ladder_str}")

        def run_provider(engine: EngineType, page_num: int) -> PageOutput:
            outs = self._run_engine_on_pages(
                state, [page_num], native_fallback_pages, engine, "agentic"
            )
            return outs[0]

        # -- Doc-scoped table extractor (PP-3 hoist) ----------------------------
        _table_extractor = None
        _crop_timeout: float | None = None
        if self.config.dual_pass_tables:
            try:
                from socr.tables.extract import TableCropExtractor, make_table_reader

                _table_model = self._resolve_crop_vlm_model()
                if _table_model:
                    _qwen_family = ("qwen",)
                    _crop_timeout = (
                        DEFAULT_PROVIDER_TIMEOUTS.get(EngineType.QWEN, 120.0)
                        if any(_table_model.lower().startswith(p) for p in _qwen_family)
                        else 120.0
                    )
                    _table_extractor = TableCropExtractor(
                        make_table_reader(
                            backend=self.config.qwen_backend,
                            model=_table_model,
                            timeout=_crop_timeout,
                            vllm_url=self.config.qwen_vllm_url,
                        )
                    )
            except Exception as exc:
                logger.warning(
                    "agentic: table extractor unavailable (%s); skipping in-loop tables", exc
                )

        # -- Doc-scoped equation flags -------------------------------------------
        _detect_eq = bool(self.config.detect_equations)
        _recover_eq = bool(self.config.recover_clean_equations) and _detect_eq

        # -- Doc-scoped write-through blob cache ---------------------------------
        _page_blob_store: object = None
        try:
            from ocr_output_contract import doc_dir_for, relative_key

            from socr.core.cache import BlobStore

            pdf_path = state.handle.path
            scan_root = self._scan_root or pdf_path.parent
            doc_dir = doc_dir_for(output_dir, relative_key(pdf_path, scan_root))
            _page_blob_store = BlobStore(doc_dir / "cache")
        except Exception as exc:
            logger.debug("write-through blob store unavailable: %s", exc)

        # -- Document-level cascade-halt latch -----------------------------------
        backend_degraded: bool = False
        # Track whether any page was processed for console summary.
        halt_reason: str = ""

        # -- PP-7: doc-scoped chart-lane figures directory -----------------------
        # Pre-compute figures_dir once so chart-lane PNG saves have a consistent
        # location alongside the rest of the document outputs.  Falls back to a
        # sibling ``figures/`` directory next to the PDF when the contract is not
        # available (unit-test / partial-pipeline scenario).
        _chart_figures_dir: Path | None = None
        _agentic_doc_dir: Path | None = None
        try:
            from ocr_output_contract import doc_dir_for, figures_dir_for, relative_key

            _chart_pdf_path = state.handle.path
            _chart_scan_root = self._scan_root or _chart_pdf_path.parent
            _chart_doc_dir = doc_dir_for(
                output_dir, relative_key(_chart_pdf_path, _chart_scan_root)
            )
            _agentic_doc_dir = _chart_doc_dir
            _chart_figures_dir = figures_dir_for(_chart_doc_dir)
        except Exception as exc:
            logger.debug("PP-7: could not compute chart figures_dir via contract: %s", exc)
            # Fallback: sibling figures/ next to the output dir.
            _chart_figures_dir = output_dir / "figures"
            _agentic_doc_dir = output_dir

        # ====================================================================
        # ONE fused loop over ALL pages in page order (native AND ocr).
        # Native-trusted pages are finalized in-place; OCR pages go through
        # route_page.  Every page gets a provisional fragment + sidecar flush
        # at the end of its lifecycle (salvage-only; authoritative bytes come
        # from _rewrite_all_fragments at assemble time — fork A).
        # ====================================================================
        for page_num in sorted(state.pages):
            ps = state.pages[page_num]
            is_native = self._is_agentic_trusted_native(page_num, ps)

            # Cascade halt guard (top-of-loop): once the backend has degraded
            # (previously timed out AND failed the health probe), halt for ALL
            # subsequent pages — native OR OCR.  No page after the wedge may be
            # processed or flushed; saving pages 0..N-1 is the invariant.
            if backend_degraded:
                if not self.config.quiet:
                    console.print(
                        f"  [red]p{page_num}: backend degraded — halting "
                        "(PARTIAL_SAVE_VLM_TIMEOUT)[/red]"
                    )
                break

            # ----------------------------------------------------------------
            # PP-5: per-page resume gate (INNER ledger; doc-level RootIndex gate
            # stays the outer all-or-nothing fast path).  Before doing ANY work
            # on page N, consult the ledger: if a TERMINAL sidecar
            # (``pages/NNN.json`` terminal=true) AND its fragment
            # (``pages/NNN.md``) exist, parse, and the per-page run fingerprint
            # MATCHES this run's fingerprint, load that fragment as the page's
            # result and SKIP OCR for page N.  This is what makes a re-run after
            # a crash reprocess ONLY the pages that were not yet terminal.
            #
            # CONSERVATIVE (load-bearing): _load_terminal_page returns None on
            # ANY doubt — missing / partial / corrupt sidecar, terminal=false,
            # fingerprint mismatch, unreadable fragment — and the page falls
            # through to normal processing.  Reprocessing a done page is wasteful
            # but safe; skipping an unfinished page is silent data loss and must
            # never happen.  A resumed page is NOT re-flushed here: its terminal
            # sidecar + fragment already on disk are authoritative, and the
            # assemble-time _rewrite_all_fragments rewrites every fragment from
            # the final text regardless.
            #
            # OCR pages already resumed by the provider-independent PRE-PASS above
            # are skipped here so they are not restored (and cost-counted) twice.
            if page_num in resumed_pages:
                continue
            resumed = self._load_terminal_page(state, page_num, output_dir)
            if resumed is not None:
                self._restore_terminal_page_state(state, page_num, resumed, output_dir)
                if not self.config.quiet:
                    console.print(f"  p{page_num}: [dim]resumed (terminal ledger hit)[/dim]")
                continue

            # PP-7: Chart-asset lane — intercept before the native-bypass branch.
            # A born-digital native page that carries vector chart marks (or an
            # embedded raster image) routes here instead of shipping as raw word-
            # salad prose.  B1 representation: native prose retained + chart PNG
            # ref embedded + explicit audit flag.  Force PNG even when --save-
            # figures is off (chart PNGs are mandatory preservation artifacts).
            if page_num in chart_winner_pages:
                if not self.config.quiet:
                    console.print(f"  p{page_num}: [cyan]chart-asset lane[/cyan]")
                from socr.core.audit_log import AuditEvent

                chart_png_ref = ""
                chart_render_failed = False
                if _chart_figures_dir is not None:
                    try:
                        saved_png = self._render_chart_page_png(
                            state.handle.path, page_num, _chart_figures_dir
                        )
                        # Relative image ref (from the doc's root markdown file).
                        png_rel = Path(saved_png).name
                        # figures/ subdir is conventional; use the relative path
                        # within the figures dir so it renders in the markdown.
                        try:
                            _figures_dir_name = _chart_figures_dir.name
                            chart_png_ref = (
                                f"![Chart page {page_num}]({_figures_dir_name}/{png_rel})"
                            )
                        except Exception:
                            chart_png_ref = f"![Chart page {page_num}]({png_rel})"
                    except RuntimeError as render_exc:
                        chart_render_failed = True
                        logger.error(
                            "PP-7 chart-lane: PNG render FAILED for p%d — %s",
                            page_num,
                            render_exc,
                        )
                        state.events.append(
                            AuditEvent(
                                page_num=page_num,
                                kind="chart_asset_render_failed",
                                engine="",
                                detail=str(render_exc),
                            )
                        )
                else:
                    chart_render_failed = True
                    _msg = (
                        f"PP-7 chart-lane: figures_dir unavailable for p{page_num}; PNG not saved"
                    )
                    logger.error(_msg)
                    state.events.append(
                        AuditEvent(
                            page_num=page_num,
                            kind="chart_asset_render_failed",
                            engine="",
                            detail=_msg,
                        )
                    )

                # B1 representation: retain native prose + embed chart PNG ref.
                # If render failed, set status=WARNING so downstream stages know
                # the visual payload is missing (fail-closed, never silent).
                native_prose = ps.native_text or ""
                if chart_png_ref:
                    chart_body = (
                        native_prose.rstrip() + "\n\n" + chart_png_ref
                        if native_prose.strip()
                        else chart_png_ref
                    )
                else:
                    chart_body = native_prose

                # Mirror native_table_structure_failed: set the PageState flag so
                # _winning_page_output (manifest.py) does not re-stamp the page as
                # engine=native / audit_passed=True / status=SUCCESS when it falls
                # through from the non-passing best_output.  Without this the fail-
                # closed intent is scrubbed at manifest-freeze time (bug PP-7-R1).
                if chart_render_failed:
                    ps.chart_asset_render_failed = True

                chart_status = PageStatus.WARNING if chart_render_failed else PageStatus.SUCCESS
                chart_out = PageOutput(
                    page_num=page_num,
                    text=chart_body,
                    status=chart_status,
                    engine="chart_asset",
                    audit_passed=not chart_render_failed,
                    cost_usd=0.0,
                )
                ps.attempts.append(chart_out)
                ps.best_output = chart_out

                # Durable audit event: visual chart semantics saved as image
                # asset; data values are NOT transcribed (explicit, auditable).
                state.events.append(
                    AuditEvent(
                        page_num=page_num,
                        kind="chart_asset_page",
                        engine="chart_asset",
                        detail=(
                            "visual chart semantics represented as image asset; "
                            "data values not transcribed"
                        ),
                        data={
                            "png_saved": not chart_render_failed,
                            "png_path": chart_png_ref,
                        },
                    )
                )

            elif is_native:
                # Tier 1: born-digital trusted native text — free, no OCR.
                # GH-151 TICKET-B1 / GH-200 / #211: a page reaches here
                # carrying a table-distrust flag only via --native-only (the
                # non-native-only ``is_native`` predicate already excludes
                # has_tables pages). Demote instead of shipping SUCCESS: this
                # is the no-reroute honouring of the flag, no extra OCR
                # attempt is triggered here.
                native_table_distrusted = bool(
                    (self.config.native_only and ps.native_table_unverifiable)
                    or getattr(ps, "native_table_structure_defective", False)
                    or getattr(ps, "native_table_header_unattributed", False)
                )
                native_out = PageOutput(
                    page_num=page_num,
                    text=ps.native_text,
                    status=(PageStatus.WARNING if native_table_distrusted else PageStatus.SUCCESS),
                    engine="native",
                    audit_passed=not native_table_distrusted,
                    failure_mode=(
                        FailureMode.NATIVE_TABLE_STRUCTURE_FAILED
                        if native_table_distrusted
                        else FailureMode.NONE
                    ),
                    cost_usd=0.0,
                )
                ps.attempts.append(native_out)
                ps.best_output = native_out

                # #92: born-digital page shipped as native text while carrying
                # unmapped math glyphs (PUA / weak ToUnicode) that equation recovery
                # did not reach. The prose is sound, but the math symbols are
                # font-private and lost — surface it (never silent). Page stays
                # SUCCESS (prose is usable); the event records the math-glyph gap.
                if getattr(ps, "has_unmapped_math_glyphs", False) and not (
                    self.config.detect_equations and self.config.recover_clean_equations
                ):
                    from socr.core.audit_log import AuditEvent

                    state.events.append(
                        AuditEvent(
                            page_num=page_num,
                            kind="native_math_unrecovered",
                            engine="native",
                            detail=(
                                "born-digital native text shipped with unmapped math glyphs "
                                "(private-use codepoints, weak ToUnicode); math symbols not "
                                "recovered — enable --detect-equations --recover-clean-equations "
                                "for region OCR -> LaTeX"
                            ),
                            data={
                                "has_equations": ps.has_equations,
                                "recover_clean_equations": self.config.recover_clean_equations,
                            },
                        )
                    )

                # #136: the text layer showed cosmetic encoding corruption (lost
                # spaces, fused words) in the flag band. The page ships SUCCESS —
                # the content is sound and a reader recovers "JournalofFinance" —
                # but the mark must reach something that ships. Before this, the
                # detector's "marked suspect so it is never silently relied on"
                # went only into PageAssessment.notes, which nothing in the
                # pipeline reads. Digit corruption never reaches here; it is
                # routed to OCR at detection.
                if getattr(ps, "has_encoding_hygiene_suspect", False):
                    from socr.core.audit_log import AuditEvent

                    state.events.append(
                        AuditEvent(
                            page_num=page_num,
                            kind="native_encoding_hygiene_suspect",
                            engine="native",
                            detail=(
                                "born-digital native text shipped from a suspect text layer "
                                "(mid-word capitals / run-on tokens from dropped inter-word "
                                "spaces); content is trusted, spacing is not"
                            ),
                            data={"class": "hygiene"},
                        )
                    )
            else:
                # Route OCR page through the cost ladder.
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

                for att in decision.attempts:
                    att.output.cost_usd = att.cost_usd
                    att.output.audit_passed = att.accepted
                    att.output.provider_id = att.provider_id  # B3: agentic provenance
                    att.output.provider_model = att.model  # B3
                    att.output.provider_backend = att.backend  # B3
                    att.output.skip_reason = (
                        att.reason if not att.accepted and not att.output.text else ""
                    )  # B3
                    ps.attempts.append(att.output)
                ps.best_output = decision.final_output

                # GH-56: deterministic header repair for collapsed multi-band tables.
                if ps.best_output and ps.best_output.text and self._page_has_tables(page_num, ps):
                    try:
                        import fitz

                        from socr.tables.header_repair import repair_table_headers_on_page

                        with fitz.open(state.handle.path) as _hdr_doc:
                            _repaired_text, _hdr_n = repair_table_headers_on_page(
                                _hdr_doc[page_num - 1],
                                ps.best_output.text,
                            )
                        if _hdr_n > 0:
                            ps.best_output.text = _repaired_text
                            from socr.core.audit_log import AuditEvent

                            state.events.append(
                                AuditEvent(
                                    page_num=page_num,
                                    kind="table_header_repair",
                                    engine=ps.best_output.engine or "",
                                    detail=(
                                        f"rebuilt {_hdr_n} collapsed table header(s) "
                                        "from native geometry (post-route)"
                                    ),
                                    data={"repair_count": _hdr_n},
                                )
                            )
                    except Exception as exc:
                        logger.debug(
                            "agentic header repair skipped on p%d (%s)",
                            page_num,
                            exc,
                        )

                # GH-200: the GH-56 repair above MUTATES ps.best_output.text
                # after the judge already accepted it -- so the shipped text
                # is not the text the structural gate saw. Re-run the
                # grid-shape term ONLY (string-only, no geometry, free) on
                # whatever text ships now; if repair produced a ragged/
                # detached-label grid, demote in place. Routing is finished
                # by this point -- do NOT reroute, just stop shipping it as a
                # pass. Exactly one audit event per page: this is the sole
                # recheck site after routing, distinct from the judge's own
                # (pre-repair) escalation event.
                if ps.best_output and ps.best_output.text and ps.best_output.audit_passed:
                    from socr.tables.structure_check import (
                        check_markdown,
                        structural_gate_fires,
                    )

                    if structural_gate_fires(check_markdown(ps.best_output.text)):
                        from socr.core.audit_log import AuditEvent

                        ps.best_output.audit_passed = False
                        ps.best_output.status = PageStatus.WARNING
                        ps.best_output.failure_mode = FailureMode.NATIVE_TABLE_STRUCTURE_FAILED
                        ps.native_table_structure_failed = True
                        state.events.append(
                            AuditEvent(
                                page_num=page_num,
                                kind="table_structure_failed",
                                engine=ps.best_output.engine or "",
                                detail="grid_shape defect found after post-route header repair",
                                data={"defect": "grid_shape", "site": "post_route_recheck"},
                            )
                        )

                # GH-90: scanned-table source-evidence fail-closed floor.
                _source_ev_rejected = any(
                    "source_evidence_table" in (att.reason or "") for att in decision.attempts
                )
                if _source_ev_rejected and not ps.is_born_digital:
                    ps.scanned_table_evidence_failed = True
                    if _chart_figures_dir is not None:
                        ps.d3_floor_png_ref = self._render_d3_floor_png(
                            state.handle.path,
                            page_num,
                            _chart_figures_dir,
                        )
                    d3_marker = f"[page {page_num} failed: unverifiable table — see image]"
                    png_ref = ps.d3_floor_png_ref
                    floor_text = f"{d3_marker}\n\n{png_ref}" if png_ref else d3_marker
                    if ps.best_output is not None:
                        ps.best_output.text = floor_text
                        ps.best_output.status = PageStatus.ERROR
                        ps.best_output.audit_passed = False
                        ps.best_output.failure_mode = FailureMode.HALLUCINATION

                # Provenance guard: when the judge rejected ALL ladder rungs for a
                # born-digital table page, mark the page so _assemble_result treats
                # any native-text fallback as audit-failed.
                if not decision.accepted and self._page_has_tables(page_num, ps):
                    ps.native_table_structure_failed = True
                    # TR-2: render chart region PNG crops and update placeholder
                    # paths in ps.native_text so strip_phantom_images (called in
                    # _phase_assemble with output_dir=doc_dir) finds the files.
                    # If ps.native_text contains no chart region placeholders this
                    # is a no-op; on render failure it returns native_text unchanged
                    # (fail-open: placeholder is stripped as before).
                    if ps.native_text and _chart_figures_dir is not None:
                        ps.native_text = self._render_chart_region_pngs(
                            state.handle.path,
                            page_num,
                            ps.native_text,
                            _chart_figures_dir,
                        )
                    # TR-3 / GH-200: D3 fail-closed floor PNG.  When the per-region
                    # verifier also hard-failed (native_table_unverifiable=True) OR
                    # header attribution found a destroyed header band
                    # (native_table_header_unattributed=True -- TR-3 is blind to
                    # header loss by construction, see header_attribution.py),
                    # render a full-page PNG so the human can still SEE the table.
                    # The image ref is stored on ps.d3_floor_png_ref and picked up
                    # by _winning_page_output (manifest.py) when assembling the
                    # final failed-table marker text. Without this widening the
                    # D3 floor would ship a bare marker with no image on a
                    # header-only defect -- still fail-closed, but a human
                    # cannot see the table (see manifest.py:_winning_page_output).
                    if (
                        getattr(ps, "native_table_unverifiable", False)
                        or getattr(ps, "native_table_header_unattributed", False)
                    ) and _chart_figures_dir is not None:
                        ps.d3_floor_png_ref = self._render_d3_floor_png(
                            state.handle.path,
                            page_num,
                            _chart_figures_dir,
                        )

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

                # Cascade-halt check: did any attempt time out, and is the
                # backend now unresponsive?  Use PP-0's probe_ollama_idle.
                # A judge timeout is encoded as reason="judge timeout" on the
                # last attempt; a provider timeout is encoded similarly.
                _had_timeout = any("timeout" in (att.reason or "") for att in decision.attempts)
                if _had_timeout and not probe_ollama_idle():
                    backend_degraded = True
                    halt_reason = "PARTIAL_SAVE_VLM_TIMEOUT"
                    from socr.core.audit_log import AuditEvent

                    state.events.append(
                        AuditEvent(
                            page_num=page_num,
                            kind="partial_save_vlm_timeout",
                            engine="",
                            detail=(
                                f"backend unresponsive after timeout on p{page_num}; "
                                "halting — pages processed so far are saved"
                            ),
                        )
                    )
                    if not self.config.quiet:
                        console.print(
                            f"  [red]p{page_num}: VLM backend unresponsive after timeout; "
                            "halting document (PARTIAL_SAVE_VLM_TIMEOUT)[/red]"
                        )
                    # Fall through to flush this page's output, then break at
                    # the top of the next iteration.

            # ----------------------------------------------------------------
            # Per-page lifecycle (runs for EVERY page that has best_output).
            # ----------------------------------------------------------------
            bo = ps.best_output
            if bo is None:
                continue

            # PP-3: in-loop table re-read (OCR pages with tables only).
            # Native text is character-exact — its tables need no re-read.
            # The chart_asset clause is currently unreachable (chart winners have no
            # table signal, so _page_has_tables already short-circuits); it is kept
            # to match the escalation-site guard below, which IS load-bearing, and
            # to stay correct if chart winners are ever widened to mixed pages.
            if (
                not is_native
                and _table_extractor is not None
                and self._page_has_tables(page_num, ps)
                and bo.text
                and bo.engine != "chart_asset"
            ):
                try:
                    import fitz

                    from socr.tables import locate_tables

                    with fitz.open(state.handle.path) as _doc:
                        _boxes = locate_tables(_doc[page_num - 1])
                    if _boxes:
                        _raw_crops = _table_extractor.extract(
                            state.handle.path,
                            page_num,
                            _boxes,
                            cascade_probe=True,
                        )
                        self._reread_page_tables(state, page_num, _raw_crops, _table_extractor)
                except Exception as exc:
                    logger.warning(
                        "agentic in-loop table re-read errored on p%d (%s); keeping text",
                        page_num,
                        exc,
                    )

            # GH-96: escalate a table page whose output disagrees with its own
            # native text layer, keeping the candidate only if exactness measurably
            # improves. Placed here so the accepted text still flows through the
            # equation, image-ref and repetition passes below.
            if (
                _escalation_profile is not None
                and not _escalation_degraded
                and bo.text
                and bo.engine != "chart_asset"
            ):
                _escalation_degraded = self._escalate_table_page(
                    state,
                    page_num,
                    ps,
                    bo,
                    _escalation_profile,
                    run_provider,
                    state.handle.path,
                )
            elif bo.text and self._page_has_tables(page_num, ps):
                # #123 TICKET-C2: no escalation lane this page (no provider, the
                # lane is degraded, or the lane is off) — still score against the
                # native layer so not-scorable pages and unexplained lanes surface
                # on a local-only run instead of shipping with no trace.
                #
                # Gated on _page_has_tables so prose pages skip the ~137ms scoring
                # cost entirely: they have no table to lose. Chart pages still
                # reach it (has_tables is True there — that is precisely why
                # TICKET-B1 exists), so they still surface as not-scorable.
                self._surface_table_scoring(state, page_num, ps, bo)

            # GH-36a/36b: per-page equation detect + crop + optional LaTeX
            # sidecar.  Runs ONLY when the flags are on (default-off).  With
            # both flags OFF, the page body is unchanged — output is byte-
            # identical to a non-equation-aware run (AC: "flags OFF → unchanged").
            if _detect_eq and bo.text:
                # Determine whether this page has clean equations.
                _eq_pages_set: set[int] = set()
                if self._last_assessment:
                    _eq_pages_set = {
                        pa.page_num
                        for pa in self._last_assessment.pages
                        if pa.has_equations and not pa.has_corrupt_math and pa.is_born_digital
                    }
                if page_num in _eq_pages_set:
                    # GH-36a: detect + crop (model-free, no text change).
                    self._detect_and_crop_equations(state, [page_num], output_dir)
                    # GH-36b: LaTeX sidecar (behind recover_clean_equations flag).
                    if _recover_eq:
                        self._attach_equation_latex_sidecars(state, [bo])

            # GH-86: strip VLM sentinel image refs before provisional flush.
            if _agentic_doc_dir is not None:
                self._sanitize_agentic_page_image_refs(
                    state,
                    page_num,
                    bo,
                    _agentic_doc_dir,
                )

            # GH-97: truncate degenerate repeated table rows. Runs after table
            # assembly (unlike OutputNormalizer) and before the flush, so the
            # fragment, the sidecar and the stitched body all see one text.
            self._guard_agentic_page_table_repetition(state, page_num, bo, is_native)

            # Write-through blob cache (replay-cache continuity).
            if _page_blob_store is not None:
                try:
                    _page_blob_store.put_page(bo)
                except Exception as exc:
                    logger.debug("write-through blob write failed for p%d: %s", page_num, exc)

            # Provisional fragment + sidecar flush (PP-2 A1: salvage-only).
            # ``terminal=False`` marks these as provisional; the authoritative
            # bytes are written by _rewrite_all_fragments at assemble time.
            try:
                from ocr_output_contract import PAGE_MARKER_RE

                # Strip any leading ## Page N marker from bo.text so the
                # provisional fragment body matches what assemble would produce
                # (modulo post-strip/post-figure transforms, which run later).
                _raw_body = bo.text or ""
                _stripped = _raw_body.lstrip()
                _m = PAGE_MARKER_RE.match(_stripped)
                _body = _stripped[_m.end() :].lstrip("\n") if _m else _raw_body

                self._flush_page_fragment(state, page_num, _body, output_dir)
                self._flush_page_sidecar(state, page_num, output_dir, terminal=False)
            except Exception as exc:
                logger.debug(
                    "PP-2 provisional flush failed for p%d (%s); continuing",
                    page_num,
                    exc,
                )

        # -- Post-loop summary ---------------------------------------------------
        if not self.config.quiet:
            console.print(f"  total cost: ${state.total_cost:.4f}")
            if halt_reason:
                console.print(f"  [red]Halted: {halt_reason}[/red]")

        # If we halted due to backend degradation, record it on the state so
        # _phase_assemble can propagate the reason into EngineResult.error.
        if halt_reason:
            state._pp2_halt_reason = halt_reason  # type: ignore[attr-defined]

    def _available_engines_for_agentic(self) -> list:
        """Probe which known providers are actually usable right now.

        Returns a list of ``ProviderProfile`` objects (not EngineType values) so
        that two profiles sharing the same ``EngineType`` — e.g. QWEN local and
        QWEN cloud — can appear as distinct rungs in the ladder. Pass the result
        directly to ``provider_ladder()`` which accepts ``list[ProviderProfile]``.

        GH-46-E2: ``DEFAULT_PROVIDERS`` is keyed by ``EngineType`` and therefore
        holds at most one profile per engine — ``EngineType.QWEN`` maps to
        ``PROFILE_QWEN_LOCAL``. Iterating it alone could never emit the cloud
        rung, so the declared local -> Ollama-Cloud -> Gemini ladder had no
        middle rung despite this docstring promising one. The cloud profile is
        appended from its own probe instead. ``DEFAULT_PROVIDERS`` is left alone:
        the same-EngineType collision there is deliberate and documented.

        The two QWEN rungs are probed INDEPENDENTLY. A machine with the cloud
        model but no local pull gets the cloud rung alone; a machine with only
        the local build gets the local rung alone. Neither gates the other.

        Tier filtering (``--strict-local``) is NOT applied here — it stays in the
        caller (``_phase_agentic``), which is the only place that knows the run's
        policy. This function reports reachability, not eligibility.
        """
        from socr.core.providers import DEFAULT_PROVIDERS, PROFILE_QWEN_CLOUD
        from socr.engines.qwen import cloud_model_available

        available = []
        for engine_type in self.config.enabled_engines:
            prof = DEFAULT_PROVIDERS.get(engine_type)
            if prof is None:
                continue
            try:
                if get_engine(engine_type).is_available():
                    available.append(prof)
            except Exception:  # availability probe must never crash routing
                pass  # NOT `continue` — the cloud probe below is independent
            if engine_type is EngineType.QWEN:
                try:
                    if cloud_model_available():
                        available.append(PROFILE_QWEN_CLOUD)
                except Exception:  # same rule: a probe must never crash routing
                    pass
        return available

    # Vision models tried (in order) as the hard-page judge when judge_model is
    # unset. Cloud-first so the judge is fast; local small VLM as offline fallback.
    _JUDGE_MODEL_CANDIDATES = ["qwen3.5:cloud", "minicpm-v:8b", "qwen3-vl:8b"]

    def _resolve_judge_model(self) -> str | None:
        """Pick an available vision model for judging, or None if none usable.

        Memoized for the lifetime of the pipeline (#133): every call probes
        Ollama once per candidate, and ``_run_fingerprint`` consults this on
        every page sidecar flush. An explicit ``--judge-model`` short-circuits
        the probing entirely and is returned verbatim, so an operator override
        is never silently discarded because the daemon was briefly unreachable.
        """
        from socr.judge.ollama_judge import OllamaVisionJudge

        if self.config.judge_model:
            return self.config.judge_model
        if self._judge_model_cache is not False:
            return self._judge_model_cache  # type: ignore[return-value]
        resolved: str | None = None
        for model in self._JUDGE_MODEL_CANDIDATES:
            try:
                if OllamaVisionJudge(model=model).is_available():
                    resolved = model
                    break
            except Exception:
                continue
        self._judge_model_cache = resolved
        return resolved

    def _resolve_crop_vlm_model(self) -> str | None:
        """Vision model for bounded table-crop reread (dual-pass / crop fallback).

        Honors ``strict_local``: cloud judge models (e.g. ``qwen3.5:cloud``) are
        never used for crop repair — only the local instruct VLM
        (``qwen3-vl:30b-a3b-instruct``) or another explicitly local override.
        """
        from socr.core.providers import PROFILE_QWEN_LOCAL

        # vLLM/server backend (HPC): use the HF model id served by vLLM, not an
        # ollama tag. The crop-reader factory routes to the OpenAI-compatible reader.
        if self.config.qwen_backend in ("vllm", "sglang", "api"):
            return self.config.qwen_vllm_model

        if self.config.strict_local:
            if self.config.judge_model and "cloud" not in self.config.judge_model:
                return self.config.judge_model
            return PROFILE_QWEN_LOCAL.model

        if self.config.judge_model:
            return self.config.judge_model
        return self._resolve_judge_model()

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
        # Primary source: PageState content-type vector (propagated by apply_born_digital).
        # Fallback: _last_assessment for callers that set it directly without going
        # through apply_born_digital (e.g. unit tests, partial pipeline runs).
        hard_pages = {
            page_num for page_num, ps in state.pages.items() if ps.has_tables or ps.has_equations
        }
        if not hard_pages and self._last_assessment:
            hard_pages = {
                pa.page_num
                for pa in self._last_assessment.pages
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

    def _reread_page_tables(
        self,
        state: DocumentState,
        page_num: int,
        raw_crops: list,
        extractor,
    ) -> tuple[int, int]:
        """Reconcile crop readings against whole-page OCR for one page.

        Called by ``_phase_dual_pass_tables`` AFTER ``locate_tables`` and
        ``extractor.extract`` have already run (those are inside the caller's
        narrow fail-open ``try/except``). This method starts from the raw crop
        list, handles the timeout-sentinel split, reconciles, patches, and emits
        ``dualpass_*`` AuditEvents. The progressive-pages loop (PP-2) will call
        this directly after it obtains crops for the page.

        Returns ``(patched_delta, flagged_delta)``.

        Exception boundary (must match pre-refactor 065e5dd exactly):
        - ``reconcile_page_tables`` is guarded by its own narrow ``try/except``
          (a reconcile failure must not drop the page -- same as original).
        - The patch (``bo.text = result.text``), counter increment, and
          AuditEvent emission run OUTSIDE any ``try/except`` so a bug there
          propagates to the caller rather than being swallowed.  This ensures
          a page is never left with patched ``bo.text`` and a missing event.
        """
        from socr.core.audit_log import AuditEvent
        from socr.tables import reconcile_page_tables

        ps = state.pages[page_num]
        bo = ps.best_output

        # Separate timed-out sentinels from successful crops; emit audit events
        # for each timeout so they appear in audit_log.json.
        had_timeout = False
        crops = []
        for c in raw_crops:
            if getattr(c, "_timed_out", False):
                had_timeout = True
                state.events.append(
                    AuditEvent(
                        page_num=page_num,
                        kind="dualpass_crop_timeout",
                        engine=bo.engine,
                        detail=(
                            "crop VLM reread timed out; "
                            f"degraded={getattr(extractor, '_backend_degraded', False)}"
                        ),
                        data={
                            "source": c.source,
                            "backend_degraded": getattr(extractor, "_backend_degraded", False),
                        },
                    )
                )
                bo.audit_notes.append(
                    f"dual-pass crop timeout p{page_num} ({c.source}); kept existing text"
                )
            else:
                crops.append(c)

        if not crops:
            return 0, 0

        # If ANY crop on this page timed out, do NOT auto-patch even when the
        # config enables it. Partial crop coverage means we cannot safely assert
        # that the remaining crops represent the full table set; patching on
        # incomplete evidence risks data loss. Force flag-only for this page.
        effective_auto_patch = self.config.auto_patch_tables and not had_timeout
        crop_repair_fallback = False
        crop_repair_declined = False
        original_text = bo.text

        def _reconcile_with_optional_fallback(fitz_page=None):
            nonlocal effective_auto_patch, crop_repair_fallback, crop_repair_declined
            from socr.tables.crop_repair import (
                crop_patch_improves_verification,
                page_needs_crop_repair_fallback,
            )

            needs_crop_fallback = not had_timeout and page_needs_crop_repair_fallback(
                original_text,
                native_table_unverifiable=getattr(ps, "native_table_unverifiable", False),
                fitz_page=fitz_page,
            )
            if needs_crop_fallback and not effective_auto_patch:
                candidate = reconcile_page_tables(
                    original_text,
                    [(c.markdown, c.source) for c in crops],
                    auto_patch=True,
                )
                if candidate.patched and crop_patch_improves_verification(
                    original_text,
                    candidate.text,
                    fitz_page=fitz_page,
                ):
                    effective_auto_patch = True
                    crop_repair_fallback = True
                    return candidate
                crop_repair_declined = bool(candidate.patched or candidate.disagreements)
                return reconcile_page_tables(
                    original_text,
                    [(c.markdown, c.source) for c in crops],
                    auto_patch=False,
                )
            return reconcile_page_tables(
                original_text,
                [(c.markdown, c.source) for c in crops],
                auto_patch=effective_auto_patch,
            )

        try:
            import fitz

            with fitz.open(state.handle.path) as _crop_doc:
                result = _reconcile_with_optional_fallback(_crop_doc[page_num - 1])
        except Exception as exc:
            try:
                result = _reconcile_with_optional_fallback(None)
            except Exception as inner_exc:
                logger.warning(
                    "dual-pass reconcile errored on p%d (%s); keeping text",
                    page_num,
                    inner_exc,
                )
                return 0, 0
            logger.debug(
                "crop-repair fallback gate skipped on p%d (%s); reconcile without native geometry",
                page_num,
                exc,
            )

        # Patch and emit AuditEvents OUTSIDE any try/except -- a bug here must
        # propagate so the caller sees it and the page is never left in a
        # half-patched state (bo.text changed but event/counter missing).
        patched_delta = 0
        if result.patched:
            bo.text = result.text
            patched_delta = 1
            if crop_repair_fallback:
                bo.audit_notes.append(
                    f"dual-pass crop-repair fallback p{page_num}: "
                    "verification improved; patched from local VLM crop"
                )
        elif crop_repair_declined:
            bo.audit_notes.append(
                f"dual-pass crop-repair fallback declined p{page_num}: "
                "crop reading did not improve verification"
            )

        flagged_delta = 0
        for d in result.disagreements:
            flagged_delta += 1
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

        return patched_delta, flagged_delta

    def _phase_dual_pass_tables(self, state: DocumentState) -> None:
        """Crop precisely-located tables, re-read each with the judge VLM, and
        reconcile against the whole-page OCR.

        On disagreement the crop reading (higher-resolution, table-focused) is
        authoritative and patched into the page; the disagreement is surfaced and
        recorded in the page's audit notes. Tables whose count can't be safely
        mapped, or whose crop reading is malformed, are flagged but never edited.
        Fail-open throughout: any error keeps the page's existing text.

        Each VLM crop call is wrapped in a ThreadPoolExecutor wall-clock guard
        (``TableCropExtractor._read_with_deadline``). On timeout an audit event
        of kind ``dualpass_crop_timeout`` is appended to ``state.events`` and the
        cascade guard is consulted before issuing the next crop call.

        The per-page reconcile/patch/audit body is delegated to
        ``_reread_page_tables`` so that the progressive-pages loop (PP-2) can
        call it independently per page. The narrow fail-open guard covering
        ``fitz.open + locate_tables + extractor.extract`` stays in this loop,
        matching the pre-refactor exception boundary exactly.
        """
        from socr.tables import locate_tables
        from socr.tables.extract import TableCropExtractor, make_table_reader

        model = self._resolve_crop_vlm_model()
        if not model:
            return  # no vision model available; nothing to do

        # Table pages: primary source is PageState content-type vector (propagated by
        # apply_born_digital).  Fall back to _last_assessment for partial pipeline runs
        # and unit tests that set the assessment directly without apply_born_digital.
        table_pages = {page_num for page_num, ps in state.pages.items() if ps.has_tables}
        if not table_pages and self._last_assessment:
            table_pages = {pa.page_num for pa in self._last_assessment.pages if pa.has_tables}
        if not table_pages:
            return

        # Derive crop-read httpx timeout from the calibrated agentic provider timeouts
        # so slow models (e.g. qwen3-vl:30b-a3b-instruct at 91-125 s on dense tables)
        # don't trip the I/O timeout before they finish. This is the httpx scalar timeout
        # on the network call; a SEPARATE ThreadPoolExecutor wall-clock deadline
        # (crop_wall_clock_deadline) is computed from this value inside TableCropExtractor.
        from socr.core.config import EngineType
        from socr.pipeline.agentic import DEFAULT_PROVIDER_TIMEOUTS

        _qwen_family = ("qwen",)
        crop_timeout = (
            DEFAULT_PROVIDER_TIMEOUTS.get(EngineType.QWEN, 120.0)
            if any(model.lower().startswith(p) for p in _qwen_family)
            else 120.0
        )

        try:
            # Reader and extractor are built ONCE at doc scope; reused per page.
            extractor = TableCropExtractor(
                make_table_reader(
                    backend=self.config.qwen_backend,
                    model=model,
                    timeout=crop_timeout,
                    vllm_url=self.config.qwen_vllm_url,
                )
            )
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
            # Narrow fail-open guard: covers only fitz.open + locate_tables +
            # extractor.extract (the bad-PDF/bad-page cases). Reconcile, patch,
            # and audit-event emission run OUTSIDE this catch via
            # _reread_page_tables, matching the pre-refactor boundary exactly.
            try:
                with fitz.open(pdf_path) as doc:
                    boxes = locate_tables(doc[page_num - 1])
                if not boxes:
                    continue
                # extract() applies the per-crop wall-clock deadline internally
                # and marks timed-out crops with _timed_out=True so the helper
                # can record an audit event and apply the cascade guard.
                raw_crops = extractor.extract(pdf_path, page_num, boxes)
            except Exception as exc:  # a dual-pass failure must never drop a page
                logger.warning("dual-pass errored on p%d (%s); keeping text", page_num, exc)
                continue
            # _reread_page_tables runs outside the try/except above so that any
            # exception in reconcile->patch->audit propagates (not swallowed).
            page_patched, page_flagged = self._reread_page_tables(
                state, page_num, raw_crops, extractor
            )
            patched += page_patched
            flagged += page_flagged

        if not self.config.quiet:
            if scanned == 0:
                console.print("  No model-OCR'd table pages to re-read")
            else:
                console.print(f"  {scanned} pages scanned, {patched} patched, {flagged} flagged")

    def _build_page_judge(self, state: DocumentState):
        """Select the page judge: VLM if requested+available, else heuristics.

        The returned judge is always wrapped in NativeTableVerifierJudge, which
        runs a deterministic two-tier PyMuPDF geometry check on born-digital
        table pages BEFORE the inner judge is called (Consilium decision
        20260615T170212Z-0362, Option C).  Scanned pages bypass cleanly.
        """
        from socr.core.audit_log import AuditEvent
        from socr.pipeline.agentic import (
            HeuristicPageJudge,
            NativeTableVerifierJudge,
            SourceEvidenceTableJudge,
            VLMPageJudge,
        )

        backend = self.config.judge_backend
        inner_judge = None
        judge_identity = JUDGE_IDENTITY_HEURISTIC
        resolved_model: str | None = None
        if backend in ("vlm", "auto"):
            try:
                from socr.judge.ollama_judge import OllamaVisionJudge

                # Build the judge from the RESOLVED model, not the module
                # default. Constructing bare fell back to ``qwen2-vl:7b`` — a
                # model that is not even in ``_JUDGE_MODEL_CANDIDATES`` — so on
                # any machine without that exact pull the judge silently became
                # the heuristic checker while the manifest still named a VLM
                # (#133).
                resolved_model = self._resolve_judge_model()
                if resolved_model:
                    vj = OllamaVisionJudge(model=resolved_model)
                    if vj.is_available():
                        inner_judge = VLMPageJudge(vj, self._make_page_renderer(state))
                        judge_identity = resolved_model
            except Exception as exc:
                logger.warning("VLM judge unavailable (%s); using heuristics", exc)
            if inner_judge is None:
                # Surface REGARDLESS of backend. ``judge_backend`` defaults to
                # "auto", and warning only under an explicit "vlm" meant the
                # default path degraded in total silence — the one place this
                # repo's "failures surface at every level" rule was not applied
                # to the judge itself.
                detail = (
                    f"requested judge model {resolved_model!r} is not available"
                    if resolved_model
                    else "no vision model from the judge candidate ladder is available"
                )
                if not self.config.quiet:
                    console.print(
                        f"  [yellow]VLM judge unavailable ({detail}) -> "
                        "heuristic judge; pages are gated by word-count/structure "
                        "checks, NOT by a vision model[/yellow]"
                    )
                state.events.append(
                    AuditEvent(
                        page_num=0,
                        kind="judge_degraded_to_heuristic",
                        engine="",
                        detail=detail,
                        data={
                            "judge_backend": backend,
                            "requested_model": resolved_model or "",
                            "candidates": list(self._JUDGE_MODEL_CANDIDATES),
                        },
                    )
                )
        # Provenance records the judge that ACTUALLY ran (#133): the previous
        # value came from ``_resolve_judge_model`` regardless of what was built,
        # so metadata.json could name a VLM for pages heuristics had judged.
        state.agentic_judge_model = judge_identity
        if inner_judge is None:
            # Sparse-aware at the DECISION point: without this, the heuristic
            # fallback judge rejects correct sparse pages at every rung and the
            # uncapped ladder pays the cloud engines for nothing.
            inner_judge = HeuristicPageJudge(self.heuristics, sparse_ok=self._sparse_page_ok)

        # Wrap with the deterministic native table verifier.
        # get_fitz_page opens the PDF handle on demand; never caches pages in
        # memory (fitz.Page holds a reference to the open document).
        pdf_path = state.handle.path

        # Single-slot cache: open the PDF once per get_fitz_page call, keep the
        # doc handle alive in the list so the Page stays valid during
        # verify_native_table, then close + evict the previous doc on the next
        # call. The list (not a plain variable) lets the nested closure mutate it.
        # This prevents N open file handles when the judge visits N table pages in
        # sequence, while keeping the Page reference valid for the caller.
        _fitz_doc_cache: list = []

        def get_fitz_page(page_num: int):
            import fitz

            # Close and evict any previously cached doc before opening a new one
            # so we hold at most one open handle at a time.
            if _fitz_doc_cache:
                try:
                    _fitz_doc_cache[0].close()
                except Exception:
                    pass
                _fitz_doc_cache.clear()
            doc = fitz.open(pdf_path)
            _fitz_doc_cache.append(doc)
            # PyMuPDF pages are 0-indexed; socr page numbers are 1-indexed.
            return doc[page_num - 1]

        def is_table_page(page_num: int) -> bool:
            ps = state.pages.get(page_num)
            return self._page_has_tables(page_num, ps)

        def record_event(event) -> None:
            state.events.append(event)

        native_judge = NativeTableVerifierJudge(
            inner=inner_judge,
            get_fitz_page=get_fitz_page,
            is_table_page=is_table_page,
            record_event=record_event,
        )
        return SourceEvidenceTableJudge(
            inner=native_judge,
            get_fitz_page=get_fitz_page,
            record_event=record_event,
        )

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

    def _native_table_structure_gate_applies(
        self, page_num: int, output: PageOutput, ps: PageState | None = None
    ) -> bool:
        """Whether a native output should be audited for table grid loss."""
        if not (output.engine or "").startswith("native"):
            return False
        return self._page_has_tables(page_num, ps)

    def _score_per_page(self, state: DocumentState) -> None:
        """Score each page's best output individually."""
        failures = 0
        for page_num in sorted(state.pages):
            page_state = state.pages[page_num]
            if not page_state.attempts:
                continue

            # Score the most recent attempt
            latest = page_state.attempts[-1]
            if page_state.is_born_digital and page_state.native_text:
                latest_is_native = (latest.engine or "").startswith("native")
                if latest_is_native:
                    tr3_distrust = bool(
                        self.config.native_only and page_state.native_table_unverifiable
                    )
                    shape_distrust = page_state.native_table_structure_defective or getattr(
                        page_state, "native_table_header_unattributed", False
                    )
                    if tr3_distrust or shape_distrust:
                        # GH-151 TICKET-B1 / GH-200 / GH-211: the defect was
                        # found at EXTRACTION time and is authoritative -- do
                        # NOT run the heuristic scorer over it (that scorer is
                        # exactly what missed p26: it tolerates the
                        # ragged/orphan shapes this gate targets, and it would
                        # overwrite TR-3's hard-fail with a pass). Force the
                        # demotion and skip straight past the heuristic path
                        # below so it cannot re-promote this attempt to
                        # audit_passed=True. No OCR is requested either way.
                        latest.audit_passed = False
                        latest.status = PageStatus.WARNING
                        latest.failure_mode = FailureMode.NATIVE_TABLE_STRUCTURE_FAILED
                        if getattr(page_state, "native_table_header_unattributed", False):
                            detail = "native table header not attributable (GH-200 gate)"
                        elif page_state.native_table_structure_defective:
                            detail = "native table grid structurally defective (GH-151 B1 gate)"
                        else:
                            detail = "native table region unverifiable (TR-3, --native-only)"
                        latest.error = detail
                        latest.audit_notes.append(detail)
                        failures += 1
                        if page_state.best_output is latest:
                            page_state.best_output = None
                        continue
                    if self._native_table_structure_gate_applies(page_num, latest, page_state):
                        scoring = self.scorer.score_native_table_structure(latest.text)
                        latest.audit_passed = scoring.passed
                        if not scoring.passed:
                            latest.failure_mode = scoring.primary_failure
                            detail = scoring.details.get(scoring.primary_failure, "")
                            if detail:
                                latest.error = detail
                                latest.audit_notes.append(detail)
                            page_state.native_table_structure_failed = True
                            page_state.needs_ocr_enhancement = True
                            failures += 1
                            if page_state.best_output is latest:
                                page_state.best_output = None
                        else:
                            latest.failure_mode = FailureMode.NONE
                            page_state.native_table_structure_failed = False
                    # Preserve the native-text exemption unless the narrow
                    # table-structure gate above rejected this exact attempt.
                    continue
                if not page_state.needs_ocr_enhancement and not self._page_has_tables(
                    page_num, page_state
                ):
                    continue

            scoring = self.scorer.score(
                latest.text, engine=latest.engine, sparse_ok=self._sparse_page_ok(page_num)
            )

            latest.audit_passed = scoring.passed
            if not scoring.passed:
                latest.failure_mode = scoring.primary_failure
                if self._page_has_tables(page_num, page_state):
                    page_state.native_table_structure_failed = True
                failures += 1
                # If this was the best_output but now fails, clear it
                if page_state.best_output is latest:
                    page_state.best_output = None
            else:
                latest.failure_mode = FailureMode.NONE
                if self._page_has_tables(page_num, page_state):
                    page_state.native_table_structure_failed = False
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
    # Phase 5: Assemble — fragment/sidecar/stitch helpers (PP-1)
    # ------------------------------------------------------------------

    def _page_fragment_path(self, output_dir: Path, state: DocumentState, page_num: int) -> Path:
        """Canonical path for a per-page markdown fragment: ``pages/NNN.md``.

        Zero-padded to five digits so lexicographic order == page order for any
        document up to 99 999 pages.
        """
        from ocr_output_contract import doc_dir_for, relative_key

        scan_root = self._scan_root or state.handle.path.parent
        doc_dir = doc_dir_for(output_dir, relative_key(state.handle.path, scan_root))
        return doc_dir / "pages" / f"{page_num:05d}.md"

    def _flush_page_fragment(
        self,
        state: DocumentState,
        page_num: int,
        text: str,
        output_dir: Path,
    ) -> Path:
        """Write body-only canonical page text to ``pages/NNN.md`` atomically.

        The fragment contains ONLY the body text — no ``## Page N`` header.
        ``_stitch_fragments`` re-adds headers via ``assemble_pages`` so the
        stitched output is byte-identical to the in-memory ``_canonical_body``.

        Atomic: writes to ``.md.tmp`` then renames so a mid-write crash never
        leaves a truncated fragment that a future stitch would silently include.
        """
        frag_path = self._page_fragment_path(output_dir, state, page_num)
        frag_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = frag_path.with_suffix(".md.tmp")
        tmp_path.write_text(text, encoding="utf-8")
        tmp_path.rename(frag_path)
        return frag_path

    def _flush_page_sidecar(
        self,
        state: DocumentState,
        page_num: int,
        output_dir: Path,
        *,
        terminal: bool = True,
    ) -> Path:
        """Write per-page provenance sidecar to ``pages/NNN.json`` atomically.

        Carries the fields that are NOT already in ``PageOutput.to_dict`` but
        are needed by later pipeline stages (PP-5 enrichment) or diagnostics:
        the PageState decision flags (``needs_ocr_enhancement``,
        ``native_table_structure_failed``, ``native_table_structure_defective``,
        ``judge_rejected``), page- and
        run-level fingerprints, the winning output's full serialised dict, and
        a summary of audit events for this page.

        ``terminal=False`` marks a provisional sidecar written mid-run (PP-2
        incremental flush); it may be superseded by the final assemble-time
        flush (``terminal=True``).

        Atomic: writes to ``.json.tmp`` then renames so a reader never sees a
        half-written file.
        """
        import json

        from ocr_output_contract import doc_dir_for, relative_key

        scan_root = self._scan_root or state.handle.path.parent
        doc_dir = doc_dir_for(output_dir, relative_key(state.handle.path, scan_root))
        sidecar_path = doc_dir / "pages" / f"{page_num:05d}.json"
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)

        # PP-5: bind the sidecar to the EXACT input bytes.  The doc-level
        # RootIndex gate already rejects a changed input on checksum mismatch, but
        # a partial-doc resume (no COMPLETED doc record) or ``--reprocess`` re-
        # enters _phase_agentic where the INNER ledger runs; without this binding
        # the inner gate would reuse a stale fragment for a CHANGED PDF at the
        # same relative path (silent stale-output reuse).  ``None`` when the input
        # is unreadable — which never equals a real digest, so it can never match.
        from ocr_output_contract import safe_checksum

        input_checksum = safe_checksum(state.handle.path)

        ps = state.pages.get(page_num)
        # Provenance honesty (GH-56): record the output that ACTUALLY ships for
        # this page — the same selection ``_winning_page_output`` freezes into the
        # manifest (``build_manifest``) and the ``pages/NNN.md`` fragment
        # (``canonical_page_texts``) — NOT the raw per-page OCR attempt
        # (``best_output``).  They diverge precisely in the fallback cases: a
        # rejected OCR attempt (status=SUCCESS yet audit_passed=False) overridden
        # by a flagged born-digital native-text fallback (engine="native",
        # WARNING), or a CLI whole-doc attempt recovered into per-page text.
        # Recording ``best_output`` there made the sidecar disagree with the
        # fragment AND fooled the resume gate, whose skip test reads
        # ``winning_output.status``: a flagged native-fallback page kept
        # status="success" and was SKIPPED on re-run, silently erasing its
        # audit-failed signal.  We compute ``whole_doc`` exactly as the manifest
        # and fragment do (None in the agentic path — no whole-doc CLI attempts —
        # but real for non-agentic CLI runs) so the sidecar matches both paths.
        from socr.core.manifest import (
            _whole_doc_page_texts,
            _winning_page_output,
            is_page_failed_marker,
        )

        whole_doc = _whole_doc_page_texts(state)
        winning_out = _winning_page_output(state, page_num, whole_doc) if ps else None
        winning_dict = winning_out.to_dict() if winning_out is not None else {}

        # Audit events for this page only, serialised as plain dicts.
        page_events = [
            {
                "kind": ev.kind,
                "engine": ev.engine,
                "detail": ev.detail,
                "data": ev.data if hasattr(ev, "data") and ev.data else {},
            }
            for ev in state.events
            if hasattr(ev, "page_num") and ev.page_num == page_num
        ]

        # Figure refs emitted by the description phase and stored on EngineResult.
        # We record only the page-local subset here.
        figure_refs: list[dict] = []
        for run in state.engine_runs:
            for fig in getattr(run, "figures", []):
                if getattr(fig, "page_num", None) == page_num:
                    figure_refs.append(fig.to_dict() if hasattr(fig, "to_dict") else {})

        payload: dict = {
            "page_num": page_num,
            # Status mirrors the winning output's status, or "missing" when
            # no output exists.
            "status": winning_dict.get("status", "missing"),
            # terminal: True when this is the definitive sidecar written at
            # phase-5 assemble time; False for a provisional mid-run incremental
            # flush from PP-2 that may be superseded by the authoritative write.
            "terminal": terminal,
            # Engine / provider provenance.
            "engine": winning_dict.get("engine", ""),
            "provider": winning_dict.get("provider_id", ""),
            "cost_usd": winning_dict.get("cost_usd", 0.0),
            # Full serialised winning PageOutput.  PP-5 reconstructs a skipped
            # page's in-memory PageState.best_output from this dict (paired with
            # the fragment text) so the resumed run carries the SAME status /
            # engine / provider / audit verdict the original produced — not just
            # the stitched body.  Empty dict when no winning output exists.
            "winning_output": winning_dict,
            # Run-level fingerprint (shared across all pages of this run).
            "run_fingerprint": self._run_fingerprint(),
            # Input-PDF checksum: the PP-5 inner resume gate requires an EXACT
            # match so a changed input at the same relative path can never reuse
            # this fragment.  Empty string when the input is unreadable.
            "input_checksum": input_checksum or "",
            # Page-level fingerprint: the blob key of the winning output, if
            # available.  Empty string when no winning output exists (e.g. a
            # failed page).
            # A failure-marker winner (a page with no usable output anywhere) has
            # no cached BlobStore entry to cross-reference, so its fingerprint
            # stays "" — otherwise it would be a sha256 of synthesized marker text
            # pointing at a blob that was never stored.  A real or native-fallback
            # winner carries genuine text worth fingerprinting for change-detection.
            "page_fingerprint": (
                _page_blob_key(winning_dict)
                if winning_dict and not is_page_failed_marker(winning_dict.get("text", ""))
                else ""
            ),
            # PageState decision flags NOT in PageOutput.to_dict (PP-5 consumers
            # need these to understand why a particular output was selected).
            "needs_ocr_enhancement": bool(ps.needs_ocr_enhancement) if ps else False,
            "native_table_structure_failed": (
                bool(ps.native_table_structure_failed) if ps else False
            ),
            # GH-151 TICKET-B1: grid-shape defect found at extraction time.
            "native_table_structure_defective": (
                bool(getattr(ps, "native_table_structure_defective", False)) if ps else False
            ),
            # GH-200: header-attribution HARD verdict found at extraction time.
            "native_table_header_unattributed": (
                bool(getattr(ps, "native_table_header_unattributed", False)) if ps else False
            ),
            # TR-3: per-region geometry hard-fail flag (D3 floor trigger).
            "native_table_unverifiable": (
                bool(getattr(ps, "native_table_unverifiable", False)) if ps else False
            ),
            # TR-3: image ref for the D3 floor PNG (empty string when not rendered).
            "d3_floor_png_ref": (str(getattr(ps, "d3_floor_png_ref", "")) if ps else ""),
            "chart_asset_render_failed": (
                bool(ps.chart_asset_render_failed) if ps else False  # PP-7
            ),
            "judge_rejected": bool(ps.judge_rejected) if ps else False,
            # Audit log subset for this page.
            "audit_events": page_events,
            # Table-pass and figure refs.
            "figure_refs": figure_refs,
        }

        tmp_path = sidecar_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp_path.rename(sidecar_path)
        return sidecar_path

    def _load_terminal_page(
        self,
        state: DocumentState,
        page_num: int,
        output_dir: Path,
    ) -> PageOutput | None:
        """PP-5 ledger reader: a reconstructed ``PageOutput`` for a TERMINAL page.

        Returns the page's winning output, reconstructed from its terminal
        sidecar (``pages/NNN.json``) paired with its body fragment
        (``pages/NNN.md``), ONLY when the page is DEFINITELY complete under the
        CURRENT run fingerprint.  Returns ``None`` on ANY doubt — the caller then
        reprocesses the page.  This is the load-bearing conservative gate: a
        false ``None`` only re-OCRs a page that was already done (wasteful but
        safe); a false non-``None`` would SKIP an unfinished page (silent data
        loss) and must never happen.

        Skip is granted iff ALL hold:
          * ``pages/NNN.json`` exists and parses as JSON.
          * ``terminal`` is exactly ``True`` (a provisional ``terminal=False``
            mid-run flush is NEVER skippable — the page may not be finished).
          * ``run_fingerprint`` equals the current run's fingerprint EXACTLY (a
            model / prompt / render-dpi / flag change yields a different
            fingerprint, forcing re-OCR of affected pages).
          * ``input_checksum`` equals the CURRENT input PDF's checksum EXACTLY (a
            changed input at the same relative path can never reuse a stale
            fragment — the doc-level RootIndex gate catches this on the fast
            path, but a partial-doc resume / ``--reprocess`` re-enters the loop
            where only this inner check stands between us and stale-output reuse).
          * The winning output's status is exactly ``SUCCESS`` and the body is
            NOT a page-failure marker (a timed-out / ERROR / lossy-fallback page
            written terminal at assemble must be RE-OCR'd, never skipped).
          * ``pages/NNN.md`` exists and is readable.
          * The serialised ``winning_output`` is present and rebuilds into a
            ``PageOutput`` without error.

        The fragment is body-only (no ``## Page N`` header), so its text is used
        verbatim as ``PageOutput.text``; the sidecar restores status / engine /
        provider / audit verdict and the PageState decision flags.
        """
        import json

        try:
            from ocr_output_contract import doc_dir_for, relative_key, safe_checksum

            from socr.core.manifest import is_page_failed_marker

            scan_root = self._scan_root or state.handle.path.parent
            doc_dir = doc_dir_for(output_dir, relative_key(state.handle.path, scan_root))
            pages_dir = doc_dir / "pages"
            sidecar_path = pages_dir / f"{page_num:05d}.json"
            frag_path = pages_dir / f"{page_num:05d}.md"

            # Both artefacts MUST be present.  A missing sidecar OR fragment means
            # the page never reached the authoritative assemble-time flush.
            if not sidecar_path.is_file() or not frag_path.is_file():
                return None

            # Parse the sidecar; a corrupt / truncated file is treated as doubt.
            try:
                meta = json.loads(sidecar_path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                return None
            if not isinstance(meta, dict):
                return None

            # terminal MUST be exactly True (not truthy): a provisional sidecar,
            # a missing key, or a non-bool value all force reprocessing.
            if meta.get("terminal") is not True:
                return None

            # Fingerprint MUST match the current run EXACTLY.  A config / model /
            # flag change yields a different fingerprint and re-OCRs the page.
            if meta.get("run_fingerprint") != self._run_fingerprint():
                return None

            # Input-PDF checksum MUST match the CURRENT input EXACTLY.  Guards the
            # partial-resume / --reprocess path where the outer RootIndex gate
            # does not fire: a different PDF at the same relative path must never
            # reuse this fragment.  An unreadable input (safe_checksum None) or a
            # missing recorded checksum can never match a real digest.
            recorded_checksum = meta.get("input_checksum")
            current_checksum = safe_checksum(state.handle.path)
            if not recorded_checksum or recorded_checksum != current_checksum:
                return None

            # The full winning PageOutput dict must be present and rebuildable.
            winning = meta.get("winning_output")
            if not isinstance(winning, dict) or not winning:
                return None

            # Status MUST be SUCCESS.  A page written terminal at assemble time
            # with an ERROR / WARNING / timed-out output (e.g. a cascade-halt page
            # whose best_output is the ERROR attempt, or a flagged native fallback)
            # is NOT a clean result — re-OCR it, never skip it.
            if winning.get("status") != PageStatus.SUCCESS.value:
                return None

            # Read the authoritative body fragment.
            try:
                body = frag_path.read_text(encoding="utf-8")
            except OSError:
                return None

            # A page-failure marker body is honesty, not content: never skip it.
            if not body.strip() or is_page_failed_marker(body):
                return None

            try:
                page_out = PageOutput.from_dict(winning)
            except Exception:
                return None

            # The fragment is the authoritative body bytes; prefer it over the
            # serialised text so a post-OCR fragment rewrite (figures / phantom
            # strip from a prior completed assemble) is honoured on resume.
            page_out.text = body
            page_out.page_num = page_num
            return page_out
        except Exception as exc:  # never let the ledger read break a run
            logger.debug("PP-5 ledger read failed for p%d (%s); reprocessing", page_num, exc)
            return None

    def _restore_terminal_page_state(
        self, state: DocumentState, page_num: int, page_out: PageOutput, output_dir: Path
    ) -> None:
        """Populate ``PageState`` from a resumed terminal page (PP-5).

        Mirrors what the live loop sets for a freshly-processed page: appends the
        reconstructed output to ``attempts``, sets it as ``best_output``, restores
        the PageState decision flags from the sidecar so downstream status
        demotion (native fallback, judge rejection, table-structure loss) behaves
        identically to a non-resumed run, AND folds the page's recorded cost into
        ``state.engine_runs`` so ``state.total_cost`` matches a live run.  The
        latter matters for budget correctness: later pages compute their
        remaining budget from ``state.total_cost``; without the resumed page's
        spend the loop could over-route / overspend on resume.  Best-effort: a
        flag-restore failure is non-fatal (the body text is already correct).
        """
        import json

        ps = state.pages.get(page_num)
        if ps is None:
            return
        ps.attempts.append(page_out)
        ps.best_output = page_out
        # Fold the resumed page's cost into total_cost (budget continuity).  An
        # EngineResult mirrors what the live route_page path appends per page.
        state.engine_runs.append(
            EngineResult(
                document_path=state.handle.path,
                engine=page_out.engine or "resumed",
                status=DocumentStatus.SUCCESS,
                cost=page_out.cost_usd,
                processing_time=0.0,
            )
        )
        try:
            from ocr_output_contract import doc_dir_for, relative_key

            scan_root = self._scan_root or state.handle.path.parent
            doc_dir = doc_dir_for(output_dir, relative_key(state.handle.path, scan_root))
            sidecar_path = doc_dir / "pages" / f"{page_num:05d}.json"
            meta = json.loads(sidecar_path.read_text(encoding="utf-8"))
            ps.needs_ocr_enhancement = bool(meta.get("needs_ocr_enhancement", False))
            ps.native_table_structure_failed = bool(
                meta.get("native_table_structure_failed", False)
            )
            # GH-151 TICKET-B1: restore the grid-shape defect flag so it
            # survives resume instead of evaporating on a resumed run.
            ps.native_table_structure_defective = bool(
                meta.get("native_table_structure_defective", False)
            )
            # GH-200: restore the header-attribution defect flag so it
            # survives resume instead of evaporating on a resumed run.
            ps.native_table_header_unattributed = bool(
                meta.get("native_table_header_unattributed", False)
            )
            # TR-3: restore per-region verifier flag and D3 PNG ref.
            ps.native_table_unverifiable = bool(meta.get("native_table_unverifiable", False))
            ps.d3_floor_png_ref = str(meta.get("d3_floor_png_ref", ""))
            ps.chart_asset_render_failed = bool(meta.get("chart_asset_render_failed", False))
            ps.judge_rejected = bool(meta.get("judge_rejected", False))
        except Exception as exc:
            logger.debug("PP-5 flag restore failed for p%d (%s); body text kept", page_num, exc)

    def _rewrite_all_fragments(
        self,
        state: DocumentState,
        output_dir: Path,
        final_text: str,
    ) -> None:
        """Rewrite every per-page fragment from the FINAL assembled text.

        This is the single authoritative fragment-write pass, called AFTER
        ``strip_phantom_images`` AND AFTER the figure phase, so every
        ``pages/NNN.md`` file contains exactly the body that appears in the
        saved ``.md`` (post-strip, post-inline-figures).

        The earlier PP-1 flush writes raw (pre-strip) fragments for the
        byte-identity verification check.  This pass supersedes those files so
        that ``_stitch_fragments`` == ``final_text`` is guaranteed for every
        page — whether or not the page has figures, and whether or not the page
        carried phantom image references.

        Non-fatal: any error is logged; the on-disk ``.md`` is already correct.
        """
        from ocr_output_contract import split_native_pages

        page_bodies = split_native_pages(final_text)
        if not page_bodies:
            return

        try:
            for idx, body in enumerate(page_bodies):
                page_num = idx + 1
                frag_path = self._page_fragment_path(output_dir, state, page_num)
                frag_path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = frag_path.with_suffix(".md.tmp")
                tmp_path.write_text(body, encoding="utf-8")
                tmp_path.rename(frag_path)
        except Exception as exc:
            logger.warning(
                "PP-4 [%s]: authoritative fragment rewrite failed (%s); "
                "fragments may diverge from final .md",
                state.handle.path.name,
                exc,
            )

    def _stitch_fragments(self, state: DocumentState, output_dir: Path) -> str:
        """Reconstruct the document body from ``pages/NNN.md`` fragments.

        Reads every fragment in page order and joins them via the contract's
        ``assemble_pages`` with EXPLICIT page numbers, producing the canonical
        ``## Page N`` body.  The result is byte-identical to ``_canonical_body``
        when all fragments were written from the same ``canonical_page_texts``
        call.

        Falls back to an empty string when no fragments exist (should never
        happen after a successful flush pass, but defensively handled).
        """
        from ocr_output_contract import assemble_pages, doc_dir_for, relative_key

        scan_root = self._scan_root or state.handle.path.parent
        doc_dir = doc_dir_for(output_dir, relative_key(state.handle.path, scan_root))
        pages_dir = doc_dir / "pages"
        if not pages_dir.is_dir():
            return ""

        # Collect fragment files in lexicographic (== page) order.
        frag_files = sorted(pages_dir.glob("*.md"))
        if not frag_files:
            return ""

        texts: list[str] = []
        page_numbers: list[int] = []
        for frag_file in frag_files:
            try:
                page_num = int(frag_file.stem)
            except ValueError:
                logger.warning("stitch: ignoring non-numeric fragment %s", frag_file.name)
                continue
            texts.append(frag_file.read_text(encoding="utf-8"))
            page_numbers.append(page_num)

        if not texts:
            return ""
        return assemble_pages(texts, page_numbers=page_numbers)

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
        """Build the final EngineResult from DocumentState and save to disk.

        PP-1: fragment/sidecar/stitch scaffold.  After computing the in-memory
        canonical body, this method also:
          1. Flushes one ``pages/NNN.md`` fragment per page (body-only, no header).
          2. Flushes one ``pages/NNN.json`` sidecar per page (provenance / flags).
          3. Stitches the fragments back via ``_stitch_fragments`` and verifies
             the result is byte-identical to the in-memory body.
        The stitched body replaces the in-memory one when they match; on any
        mismatch or error a warning is logged and the in-memory body is kept
        (fail-open, byte-identity invariant preserved via fallback).
        """
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

        # PP-1: flush per-page fragments and sidecars from the in-memory page
        # texts, then verify stitch round-trips byte-identically.
        # Non-fatal: any error keeps the in-memory body.
        if has_text and page_texts:
            try:
                page_nums = list(range(1, state.handle.page_count + 1))
                for pnum, ptext in zip(page_nums, page_texts):
                    self._flush_page_fragment(state, pnum, ptext, output_dir)
                    self._flush_page_sidecar(state, pnum, output_dir)
                stitched = self._stitch_fragments(state, output_dir)
                if stitched == final_text:
                    # Byte-identical: the fragment/stitch path is verified.
                    # Use stitched to confirm we go through the new code path.
                    final_text = stitched
                else:
                    # Mismatch — log and fall back to in-memory body.
                    logger.warning(
                        "PP-1 [%s]: stitched body differs from in-memory body "
                        "(%d vs %d bytes); falling back to in-memory assembly",
                        state.handle.path.name,
                        len(stitched),
                        len(final_text),
                    )
            except Exception as exc:
                logger.warning(
                    "PP-1 [%s]: fragment flush/stitch failed (%s); "
                    "falling back to in-memory assembly",
                    state.handle.path.name,
                    exc,
                )
        failed_pages = [i for i, t in enumerate(page_texts, start=1) if is_page_failed_marker(t)]
        # TR-3: D3 fail-closed floor pages — born-digital table pages where the
        # per-region geometry verifier hard-failed AND the OCR ladder also failed.
        # These ship the explicit failed-table marker (not the collapsed native) so
        # no plausible-but-wrong table is ever emitted.  They are a STRICT SUBSET
        # of failed_pages (every D3 page is also a failed page), counted separately
        # for the distinct audit event and CLI summary.
        # GH-200: widened identically to manifest.py's D3 conjunction -- a
        # header-only defect (native_table_header_unattributed, TR-3 is blind
        # to header loss by construction) is a D3 floor page too, else it is
        # double-counted below (both in d3_floor_pages via the manifest ship
        # and in native_fallback_pages via this list, since the exclusion
        # predicate must match exactly what _winning_page_output ships).
        d3_floor_pages = [
            n
            for n, p in sorted(state.pages.items())
            if p.is_born_digital
            and p.native_table_structure_failed
            and (
                getattr(p, "native_table_unverifiable", False)
                or getattr(p, "native_table_header_unattributed", False)
            )
            and bool(p.attempts)
        ]
        # GH-211 MAJOR-2: under --native-only the OCR ladder never runs, so a
        # page demoted purely because the extraction-time TR-3 geometry check
        # flagged it (``native_table_unverifiable``) never had an OCR attempt
        # -- "OCR tried and never passed" (the native_fallback wording below)
        # would be a lie for these pages. Split them into their own bucket so
        # the audit log and CLI can say "native distrusted, OCR never run"
        # instead. Guarded by ``all(... startswith("native"))`` so the narrow
        # rotated+table exception (which DOES still route through OCR even
        # under --native-only, see ``_is_trusted_native_without_ocr``) is
        # correctly excluded and keeps the "OCR tried and failed" wording.
        native_only_distrust_pages = [
            n
            for n, p in sorted(state.pages.items())
            if p.is_born_digital
            and p.native_text
            and self.config.native_only
            and getattr(p, "native_table_unverifiable", False)
            and not p.native_table_structure_failed
            and p.attempts
            and all((a.engine or "").startswith("native") for a in p.attempts)
            and not (p.best_output and p.best_output.audit_passed)
        ]
        native_fallback_pages = [
            n
            for n, p in sorted(state.pages.items())
            if p.is_born_digital
            and p.native_text
            and (
                p.needs_ocr_enhancement
                or p.native_table_structure_failed
                or getattr(p, "native_table_unverifiable", False)
                or getattr(p, "native_table_structure_defective", False)
                or getattr(p, "native_table_header_unattributed", False)  # GH-200
                or p.chart_asset_render_failed  # PP-7: render failure surfaces at doc level
            )
            # TR-3: D3 floor pages (see d3_floor_pages below --
            # ``native_table_structure_failed AND native_table_unverifiable``)
            # already have their own distinct event; exclude EXACTLY that set
            # from the generic native_fallback list so a page is never
            # double-counted. GH-151 B1's defect flag is OR'd into the
            # *include* side, not into the exclusion: a page can carry
            # ``native_table_structure_defective`` with
            # ``native_table_unverifiable`` also true (the TR-3 per-region
            # geometry check runs independently of B1's grid-shape check)
            # without being a D3 floor page, because D3 requires
            # ``native_table_structure_failed`` too -- which B1's
            # short-circuit in ``_score_per_page`` never sets (it returns
            # before the heuristic scorer that sets it ever runs). Excluding
            # on ``native_table_unverifiable`` alone silently dropped that
            # page from BOTH lists, so it never surfaced as a document
            # failure despite shipping WARNING/audit_passed=False --
            # excluding on the exact d3_floor_pages predicate is the only
            # condition that matches what ``_winning_page_output``
            # (manifest.py) actually ships.
            #
            # GH-151 B1 review round 2: this exclusion previously lived
            # INSIDE the ``native_table_structure_failed`` disjunct above, so
            # a page that ALSO carried ``needs_ocr_enhancement=True`` (e.g. a
            # corrupt-math page whose table region separately hard-failed
            # TR-3's per-region geometry check) satisfied the FIRST disjunct
            # unconditionally and bypassed the exclusion entirely -- counted
            # in both ``d3_floor_pages`` and ``native_fallback_pages``. Moving
            # the exclusion outside the whole OR closes that gap: it now
            # applies uniformly to every reason a page can enter this list.
            and not (
                p.native_table_structure_failed
                and (
                    getattr(p, "native_table_unverifiable", False)
                    or getattr(p, "native_table_header_unattributed", False)  # GH-200
                )
            )
            # GH-211: pages distrusted under --native-only never had an OCR
            # attempt, so they must not be folded into a list whose whole
            # meaning is "OCR was tried and never passed". They surface
            # through native_only_distrust_pages instead.
            and n not in native_only_distrust_pages
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
        pages_ok = pages_ok and not native_only_distrust_pages

        if has_text and pages_ok:
            status = DocumentStatus.SUCCESS
        elif has_text:
            status = DocumentStatus.AUDIT_FAILED
        else:
            status = DocumentStatus.ERROR

        state.status = status

        if failed_pages or native_fallback_pages or d3_floor_pages or native_only_distrust_pages:
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
                        detail="structured/enhancement page did not ship a passing OCR "
                        "result (OCR failed, was skipped, or ran under --native-only); "
                        "native text shipped flagged",
                    )
                )
            # GH-211 MAJOR-2: --native-only distrust pages get their own kind
            # here too. The authoritative "table_structure_failed" event was
            # already recorded once, in ``_phase_analyze``, at extraction time
            # (data={"defect": "unverifiable_table_region"}) -- this repeats
            # the accurate "OCR never attempted" wording at assemble time so a
            # reader scanning the tail of the audit log (or the CLI summary
            # below) is not left with only the misleading native_fallback
            # phrasing for these pages.
            for n in native_only_distrust_pages:
                state.events.append(
                    AuditEvent(
                        page_num=n,
                        kind="native_only_table_distrusted",
                        engine="native",
                        detail="--native-only: table region unverifiable, OCR never "
                        "attempted (ladder disabled); native text shipped flagged",
                    )
                )
            # TR-3: distinct audit event for D3 floor pages — do NOT record
            # native_fallback for these; they were routed to the image-asset lane
            # (failed-table marker + explicit failure), never the collapsed native.
            for n in d3_floor_pages:
                state.events.append(
                    AuditEvent(
                        page_num=n,
                        kind="table_region_unverifiable",
                        engine="native",
                        detail=(
                            "per-region geometry verifier hard-failed"
                            " (geometry_impossible_collapse) and OCR ladder also failed;"
                            " D3 fail-closed: explicit failed-table marker shipped"
                            " — no collapsed/ragged table emitted"
                        ),
                        data={"d3_floor": True},
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
                        f"  [yellow]{len(native_fallback_pages)} structured/enhancement page(s) "
                        f"fell back to native text: {native_fallback_pages}[/yellow]"
                    )
                if native_only_distrust_pages:
                    console.print(
                        f"  [yellow]{len(native_only_distrust_pages)} page(s) shipped native "
                        f"text with an unverifiable table region (--native-only: OCR not "
                        f"attempted): {native_only_distrust_pages}[/yellow]"
                    )
                if d3_floor_pages:
                    console.print(
                        f"  [red]{len(d3_floor_pages)} table page(s) hit the D3 fail-closed"
                        f" floor (unverifiable region → explicit failure marker): "
                        f"{d3_floor_pages}[/red]"
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

        # PP-2 cascade HALT: propagate the halt reason into the result error
        # so callers and tests can detect a partial-save due to a wedged backend.
        _pp2_halt = getattr(state, "_pp2_halt_reason", None)
        if _pp2_halt:
            # Append to any existing per-page error rather than overwriting it.
            if final_result.error:
                final_result.error = f"{final_result.error}; {_pp2_halt}"
            else:
                final_result.error = _pp2_halt

        # GH-97: surface degenerate table-row repetition at document level. The
        # per-page audit events alone are not enough - a consumer gating on
        # ``metadata.json`` must be able to see that a page's table was
        # truncated without parsing the full audit log.
        _rep_note = self._repetition_truncated_note(state.events)
        if _rep_note:
            if final_result.error:
                final_result.error = f"{final_result.error}; {_rep_note}"
            else:
                final_result.error = _rep_note
        # GH-95: document-level table-distrust pointer. ``DocMetadata`` has a
        # fixed field set with no slot for a trust count, so the signal rides in
        # the one free-text field, prefixed with a stable token a downstream CLI
        # can gate on by substring. ``tables_trust.json`` remains authoritative;
        # this only tells a consumer to go look. Appended (never overwriting) so
        # a lost-content or halt reason still reads first.
        _trust_note = self._tables_trust_note(state)
        if _trust_note:
            if final_result.error:
                final_result.error = f"{final_result.error}; {_trust_note}"
            else:
                final_result.error = _trust_note

        # Save markdown + metadata BEFORE the figure phase: the describe loop
        # makes long paid API calls, and any exception there used to lose the
        # fully-extracted OCR text with no record at all (no .md, no
        # metadata.json — observed on real runs). Text is final at this point;
        # figures only append to it. When the figure phase is still pending,
        # the metadata is PROVISIONAL: a ":pre-figures" fingerprint suffix
        # guarantees the resume gate never matches it, so a hard crash during
        # figures leaves a retryable record instead of a skipped-forever doc.
        #
        # ``save_figures`` triggers PNG extraction + image-ref embedding.
        # ``describe_figures`` additionally calls the VLM caption engine.
        # Either flag activates the figure phase; the orchestrator gates the
        # VLM call inside ``_describe_and_embed_figures`` on ``describe_figures``
        # so ``--save-figures`` alone never produces caption prose.
        runs_figures = (self.config.save_figures or self.config.describe_figures) and has_text
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

        # PP-4: single authoritative fragment rewrite from the FINAL text (post-
        # strip_phantom_images, post-inline-figures for figure docs, plain post-
        # strip for figure-free docs).  Runs unconditionally so every pages/NNN.md
        # matches the saved .md byte-for-byte regardless of whether the page has
        # figures or phantom image refs.  Supersedes the PP-1 pre-strip flush for
        # ALL pages.
        if has_text:
            self._rewrite_all_fragments(state, output_dir, final_text)

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
            self._write_tables_trust(state, audit, doc_dir)
        except Exception as exc:  # never lose output over an audit-log write
            logger.warning("audit log write failed (non-fatal): %s", exc)

    def _write_tables_trust(self, state: DocumentState, audit, doc_dir: Path) -> None:
        """GH-95: write tables_trust.json beside audit_log.json.

        Separate file rather than an inline callout in the assembled markdown:
        the ``.md`` is produced by ``_rewrite_all_fragments``, whose byte-identity
        with whole-doc assembly is guarded by golden tests, and #95 accepts a
        sidecar. Prose-only pages therefore stay unmarked in the corpus.

        Non-fatal by the same rule as the audit log: a trust-sidecar write must
        never lose OCR output.
        """
        try:
            from socr.core.tables_trust import build_tables_trust

            trust = build_tables_trust(state.handle.filename, audit.events)
            if not trust.pages:
                return  # every table trusted — no sidecar to write
            trust.save(doc_dir / "tables_trust.json")
            if not self.config.quiet:
                console.print(f"  [yellow]Table trust:[/yellow] {trust.summary_line()}")
        except Exception as exc:
            logger.warning("tables_trust.json write failed (non-fatal): %s", exc)

    @staticmethod
    def _tables_trust_note(state: DocumentState) -> str | None:
        """GH-95: document-level table-distrust pointer, or None when clean.

        Derived from ``state.events`` rather than ``build_run_audit`` because
        every distrust kind is appended at the source; the derived-escalation
        half of the audit contributes none of them. Non-fatal: a note that
        cannot be derived must never fail the run.
        """
        try:
            from socr.core.tables_trust import build_tables_trust, trust_note

            events = list(getattr(state, "events", []))
            filename = getattr(getattr(state, "handle", None), "filename", "")
            return trust_note(build_tables_trust(filename, events))
        except Exception as exc:
            logger.warning("table-trust note derivation failed (non-fatal): %s", exc)
            return None

    def _describe_and_embed_figures(
        self,
        state: DocumentState,
        result: EngineResult,
        output_dir: Path,
        text: str,
    ) -> str:
        """Extract figures, describe them (opt-in), and embed inline per page.

        PP-4 layout: each figure is embedded INSIDE the ``## Page N`` section it
        belongs to (after the page body text), not appended at the document tail.
        This makes each page section self-contained.

        Phase steps:
          1. Extract figures doc-wide (saves PNGs; counter is doc-global +
             monotonic across pages so ``figure_N_pageP.png`` filenames never
             renumber).
          2. Build a vision engine ONCE per document; close it once at the end.
             When ``config.describe_figures`` is False the engine is never
             constructed and no VLM call is made.
          3. Describe each figure (optional) and collect ``FigureInfo`` objects.
          4. Group figures by page.  For each page: build the figure-block
             markdown, splice it into that page's body in the in-memory page
             list, and update the on-disk fragment (``pages/NNN.md``) atomically
             so the fragment layer stays consistent with the final ``.md``.
          5. Reassemble all pages into the updated document body.

        Byte-identity guarantee: a document with no extracted figures returns
        ``text`` unchanged and leaves all fragment files untouched, so the PP-1
        byte-identity invariant is preserved for figure-free documents.

        Ordering: PNGs are saved by the extractor BEFORE the image ref is written
        to the fragment, so ``strip_phantom_images`` (applied to ``text`` before
        this method is called) cannot strip our just-added refs on a subsequent
        pass.
        """
        if not self.config.quiet:
            console.print("  Extracting figures...")

        from ocr_output_contract import (
            assemble_pages,
            doc_dir_for,
            figures_dir_for,
            relative_key,
            split_native_pages,
        )

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
        extraction: ExtractionResult = extractor.extract(state.handle.path, skip_pages=skip_pages)
        extracted = extraction.figures

        # Cap-reached: signal in console and durable audit log so silently
        # dropped later figures are not invisible to the operator.  The event
        # is attached to the page where extraction halted (the first unprocessed
        # page) so per-page audit consumers can localise the cap.
        if extraction.cap_reached:
            cap_msg = (
                f"Figure extraction cap reached ({self.config.figures_max_total} figures); "
                f"later figures may have been dropped."
            )
            if not self.config.quiet:
                console.print(f"  [yellow]{cap_msg}[/yellow]")
            from socr.core.audit_log import AuditEvent

            state.events.append(
                AuditEvent(
                    page_num=extraction.cap_page or 0,
                    kind="figure_cap_reached",
                    engine="",
                    detail=cap_msg,
                    data={"figures_max_total": self.config.figures_max_total},
                )
            )

        if not extracted:
            if not self.config.quiet:
                console.print("  [dim]No figures detected[/dim]")
            result.figures = []
            return text

        if not self.config.quiet:
            console.print(f"  Extracted {len(extracted)} figures to {figures_dir}")

        # VLM caption engine: built ONCE per document, closed ONCE after the
        # per-figure loop.  When ``describe_figures`` is False the engine is
        # never constructed and no VLM call is made (GH-50 parity).
        vision_engine = self._get_vision_engine() if self.config.describe_figures else None
        if not self.config.describe_figures and not self.config.quiet:
            console.print(
                "  [dim]Skipping VLM captions (--describe-figures not set; "
                "writing image refs only)[/dim]"
            )

        figures: list[FigureInfo] = []

        for fig in extracted:
            description = ""
            figure_type = "extracted"
            was_described = False

            if vision_engine is not None and fig.image is not None:
                # Get page context for better descriptions
                context = self._get_page_context(state, fig.page_num)
                info = vision_engine.describe_figure(
                    fig.image,
                    context=context,
                )
                description = info.description
                figure_type = info.figure_type or "extracted"
                was_described = True

            fig_info = FigureInfo(
                figure_num=fig.figure_num,
                page_num=fig.page_num,
                figure_type=figure_type,
                description=description,
                image_path=fig.saved_path,
                engine=vision_engine.name if vision_engine else "",
                bbox=fig.bbox,
            )
            figures.append(fig_info)

            # GH-47C (Option C — log-only): for described figures, collect
            # native word tokens inside the figure bbox and record them as a
            # recoverable-label set in the audit log.  No caption comparison,
            # no warning, no threshold.  A region with zero recoverable tokens
            # is inapplicable — always recorded as an empty set, never a fail.
            if was_described and fig.bbox is not None:
                self._record_figure_recoverable_labels(state, fig_info)

        if vision_engine is not None:
            vision_engine.close()

        result.figures = figures

        if not self.config.quiet:
            if self.config.describe_figures:
                described = sum(1 for f in figures if f.description)
                console.print(f"  {len(figures)} figures processed ({described} described)")
            else:
                console.print(f"  {len(figures)} figures saved (no VLM captions)")

        # PP-4 inline embedding: splice figure blocks into the per-page body
        # rather than appending them at the document tail.
        #
        # ``text`` is the phantom-stripped assembled document body (## Page N
        # sections).  We split it into per-page body texts, augment each page
        # that has figures, re-assemble with explicit page numbers, and also
        # update the on-disk fragment files so the ``pages/NNN.md`` layer stays
        # consistent.
        #
        # Group figures by 1-indexed page number.
        by_page: dict[int, list[FigureInfo]] = {}
        for fig_info in figures:
            by_page.setdefault(fig_info.page_num, []).append(fig_info)

        if not by_page:
            return text

        # Parse the assembled body into per-page body texts (headers stripped).
        page_bodies = split_native_pages(text)
        n_pages = len(page_bodies)

        updated = False
        for page_num, page_figs in sorted(by_page.items()):
            idx = page_num - 1  # 0-indexed
            if idx < 0 or idx >= n_pages:
                logger.warning(
                    "PP-4: figure(s) claim page %d but document has %d page(s); "
                    "skipping inline embed for these figures",
                    page_num,
                    n_pages,
                )
                continue

            page_figure_blocks = self._build_figure_blocks(page_figs, doc_dir)
            if not page_figure_blocks:
                continue

            # Append figure blocks to this page's body text.
            page_bodies[idx] = page_bodies[idx].rstrip() + "\n\n" + page_figure_blocks

            updated = True

        if not updated:
            return text

        # Reassemble using explicit 1-indexed page numbers so assemble_pages
        # emits canonical ## Page N headers matching the original page order.
        page_numbers = list(range(1, n_pages + 1))
        return assemble_pages(page_bodies, page_numbers=page_numbers)

    def _record_figure_recoverable_labels(self, state, fig_info) -> None:
        """GH-47C (Option C — log-only): collect native word tokens inside a figure bbox.

        For each described figure, open the PDF page and filter ``get_text("words")``
        results by the figure's bounding rectangle.  The word token set is recorded in
        the durable audit log as ``figure_recoverable_labels`` — evidence for human
        inspection only.  No comparison is made against the caption, no warning is
        emitted, and no threshold exists.

        A figure region with zero recoverable native words is INAPPLICABLE: rasterized
        or embedded-image figures have no text layer, so an empty set is always correct
        for those figures.  We record the empty set and mark it inapplicable so the
        audit record is self-explanatory.

        This method is a no-op (logs a debug line and returns) on any error, to ensure
        the describe-path cannot be disrupted by a label-recovery failure.
        """
        import fitz

        from socr.core.audit_log import AuditEvent

        if fig_info.bbox is None:
            return

        x0, y0, x1, y1 = fig_info.bbox
        try:
            with fitz.open(state.handle.path) as pdf:
                page_index = fig_info.page_num - 1  # page_num is 1-indexed
                if page_index < 0 or page_index >= len(pdf):
                    return
                page = pdf[page_index]
                # Each word tuple: (x0, y0, x1, y1, word, block_no, line_no, word_no)
                words = page.get_text("words")
        except Exception as exc:
            logger.debug(
                "GH-47C: failed to open PDF for label recovery on figure %d page %d: %s",
                fig_info.figure_num,
                fig_info.page_num,
                exc,
            )
            return

        # Filter: include word tokens whose centre falls inside the figure rect.
        # Strict containment (centre-point) is more reliable than intersection for
        # labels that span a border; it avoids picking up text immediately adjacent
        # to the figure (e.g. a caption line placed just below the bbox).
        recovered: list[str] = []
        for w in words:
            wx0, wy0, wx1, wy1, word_text = w[0], w[1], w[2], w[3], w[4]
            cx = (wx0 + wx1) / 2.0
            cy = (wy0 + wy1) / 2.0
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                recovered.append(word_text)

        inapplicable = len(recovered) == 0
        if inapplicable:
            detail_suffix = "0 native words in region (inapplicable — likely rasterized figure)"
        else:
            detail_suffix = f"{len(recovered)} native word token(s) recovered from figure region"
        state.events.append(
            AuditEvent(
                page_num=fig_info.page_num,
                kind="figure_recoverable_labels",
                engine="",
                detail=(f"figure {fig_info.figure_num} page {fig_info.page_num}: {detail_suffix}"),
                data={
                    "figure_num": fig_info.figure_num,
                    "recoverable_labels": recovered,
                    "inapplicable": inapplicable,
                    "bbox": list(fig_info.bbox),
                },
            )
        )

    def _detect_and_crop_equations(
        self,
        state: DocumentState,
        page_nums: list[int],
        output_dir: Path,
    ) -> None:
        """GH-36a: detect display-equation regions and save crop PNGs.

        Runs the deterministic, model-free detector on each page in
        ``page_nums``.  Saves a crop PNG for every detected region to
        ``equations/`` beside the figures directory, and records provenance in
        ``state.events`` (AuditEvent kind ``equation_region_detected``).

        No text is modified, no model is called.  This is DETECTION + EVIDENCE
        only; the engine/validation/splice layer is GH-36b.
        """
        import fitz
        from ocr_output_contract import doc_dir_for, relative_key

        from socr.core.audit_log import AuditEvent
        from socr.math.detect_equations import (
            EquationDetectionResult,
            detect_display_equations,
            save_equation_crops,
        )

        scan_root = self._scan_root or state.handle.path.parent
        doc_dir = doc_dir_for(output_dir, relative_key(state.handle.path, scan_root))
        equations_dir = doc_dir / "equations"

        total_regions = 0
        try:
            pdf = fitz.open(state.handle.path)
        except Exception as exc:
            logger.warning("equation detection: cannot open PDF: %s", exc)
            return

        try:
            for page_num in page_nums:
                try:
                    page = pdf[page_num - 1]
                except IndexError:
                    logger.warning("equation detection: page %d out of range (skipping)", page_num)
                    continue

                det: EquationDetectionResult = detect_display_equations(page, page_num)
                if not det.regions:
                    continue

                save_equation_crops(det.regions, page, equations_dir, dpi=self.config.render_dpi)

                for region in det.regions:
                    total_regions += 1
                    state.events.append(
                        AuditEvent(
                            page_num=page_num,
                            kind="equation_region_detected",
                            engine="detect_equations",
                            detail=(
                                f"display-equation region detected "
                                f"(bbox={region.source_bbox!r}, "
                                f"eq_num={region.has_eq_number}, "
                                f"crop={region.crop_path!r})"
                            ),
                            data={
                                "source_bbox": list(region.source_bbox),
                                "padded_bbox": list(region.bbox),
                                "has_eq_number": region.has_eq_number,
                                "crop_path": region.crop_path,
                                "detection_time_s": det.detection_time_s,
                            },
                        )
                    )
        finally:
            pdf.close()

        if total_regions and not self.config.quiet:
            console.print(
                f"  [dim]GH-36a: {total_regions} equation region(s) detected "
                f"and cropped to {equations_dir}[/dim]"
            )

    def _attach_equation_latex_sidecars(
        self,
        state: DocumentState,
        page_outputs: list,
    ) -> None:
        """GH-36b: read equation crop PNGs → 1A-validated LaTeX → 1C sidecar.

        For each page that had display-equation regions detected by GH-36a, this
        method:
          1. Reads the crop PNG with the local VLM (``qwen3-vl:30b-a3b-instruct``
             via ``equation_latex.latex_for_crop``; mock-testable via the ``ocr``
             injectable inside ``process_equation_region``).
          2. Validates the VLM output with ``validate_latex.validate_latex_structure``
             (pylatexenc, offline, deterministic — the 1A gate).
          3. Appends a 1C non-destructive sidecar block to the page's
             ``PageOutput.text``:  the crop PNG is ALWAYS inlined; validated LaTeX
             is attached adjacently when 1A passes; native text is kept on failure.
             Neither the crop nor the native text is ever silently replaced.
          4. Records ``equation_latex_accepted`` / ``equation_latex_rejected_kept_crop``
             audit events with raw LaTeX + 1A result + attachment decision + model id.

        Hard invariants (per consilium 20260615T210537Z-6621):
          - Bad/hallucinated/unvalidated LaTeX NEVER silently replaces a faithful
            crop or native text.
          - The crop PNG is always the visual ground truth — always inlined.
          - 1B (full render / image-compare) is NOT performed here.
          - This path stays default-off (config.recover_clean_equations = False).
        """
        from socr.core.audit_log import AuditEvent
        from socr.math.equation_latex import process_equation_region
        from socr.math.recover import DEFAULT_MODEL

        # Collect equation_region_detected events grouped by page.
        regions_by_page: dict[int, list[dict]] = {}
        for event in state.events:
            if event.kind == "equation_region_detected":
                regions_by_page.setdefault(event.page_num, []).append(event.data)

        if not regions_by_page:
            return  # GH-36a found nothing (or didn't run)

        # Build a fast lookup from page_num to the PageOutput entry.
        output_by_page: dict[int, object] = {po.page_num: po for po in page_outputs}

        # GH-36b: use the dedicated clean-equation model field (defaults to
        # qwen3-vl:30b-a3b-instruct — the validated local instruct VLM).
        # Do NOT fall back to math_model here: that field defaults to
        # qwen3.5:cloud for the corrupt-font path and would route every
        # default-config clean-equation run to a cloud endpoint, violating
        # the consilium local-first mandate (20260615T210537Z-6621).
        model = self.config.clean_equation_model or DEFAULT_MODEL
        accepted_total = 0
        rejected_total = 0

        for page_num, region_data_list in sorted(regions_by_page.items()):
            po = output_by_page.get(page_num)
            if po is None:
                # Page not in prose_pages (shouldn't happen, but be defensive).
                logger.warning(
                    "GH-36b: page %d has detected equations but no PageOutput; skipping",
                    page_num,
                )
                continue

            native_text = state.pages[page_num].native_text or ""

            for region_index, rdata in enumerate(region_data_list):
                crop_path = rdata.get("crop_path")

                result = process_equation_region(
                    region_index=region_index,
                    page_num=page_num,
                    crop_path=crop_path,
                    native_text=native_text,
                    model=model,
                    host=self.config.math_model_host
                    if hasattr(self.config, "math_model_host")
                    else "http://localhost:11434",
                )

                # 1C: append sidecar block to page text (never replace).
                if result.sidecar_block:
                    if po.text:
                        po.text = po.text + "\n\n" + result.sidecar_block
                    else:
                        po.text = result.sidecar_block

                # Provenance: emit audit event.
                if result.latex_attached:
                    accepted_total += 1
                    kind = "equation_latex_accepted"
                    detail = (
                        f"1A-validated LaTeX attached adjacently to crop "
                        f"(crop={crop_path!r}, model={model!r})"
                    )
                else:
                    rejected_total += 1
                    kind = "equation_latex_rejected_kept_crop"
                    detail = (
                        f"1A validation failed ({result.validation_reason}); "
                        f"native text kept, crop retained "
                        f"(crop={crop_path!r}, model={model!r})"
                    )

                state.events.append(
                    AuditEvent(
                        page_num=page_num,
                        kind=kind,
                        engine="equation_latex",
                        detail=detail,
                        data={
                            "region_index": result.region_index,
                            "crop_path": result.crop_path,
                            "raw_latex": result.raw_latex,
                            "validation_ok": result.validation_ok,
                            "validation_reason": result.validation_reason,
                            "latex_attached": result.latex_attached,
                            "model_id": result.model_id,
                        },
                    )
                )

        if (accepted_total or rejected_total) and not self.config.quiet:
            console.print(
                f"  [dim]GH-36b: {accepted_total} equation(s) → LaTeX attached; "
                f"{rejected_total} failed 1A gate (native text kept)[/dim]"
            )

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
