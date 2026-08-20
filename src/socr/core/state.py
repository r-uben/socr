"""Central document state blackboard for the OCR pipeline.

DocumentState is the single mutable data structure that accumulates results
as a document flows through pipeline stages (born-digital detection, primary
engine, fallback engines, reconciliation, figure extraction).

Design principles:
  - DUMB DATA: no pipeline logic, no audit rules, no engine calls.
  - The orchestrator calls an engine, gets an EngineResult, then merges it
    via ``state.apply_result(result)``.
  - All attempts are stored per page for reconciliation / voting.
  - ``best_output`` is auto-selected naively (first passing attempt); the
    reconciler will override it later.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from socr.core.born_digital import DocumentAssessment
from socr.core.document import DocumentHandle
from socr.core.result import DocumentStatus, EngineResult, PageOutput


@dataclass
class PageState:
    """Per-page processing state (1-indexed)."""

    page_num: int
    is_born_digital: bool = False
    native_text: str | None = None
    needs_ocr_enhancement: bool = False  # native layer has a known deficiency
    has_tables: bool = False  # page contains table-like structures
    has_figures: bool = False  # page contains embedded raster images
    has_equations: bool = False  # page contains math/equations
    has_unmapped_math_glyphs: bool = False  # PUA glyphs in native layer -> silent math-glyph loss
    #: #136: text layer shows COSMETIC encoding corruption (lost spaces, fused
    #: words) in the flag band. Content is trustworthy; the mark exists so the page
    #: is never *silently* relied on. Digit corruption never sets this — that class
    #: is routed to OCR at detection.
    has_encoding_hygiene_suspect: bool = False
    #: #217: the page's symbol font shipped no ToUnicode map and at least one
    #: glyph it draws has no verified recovery, so those characters are still
    #: whatever the extractor produced. Unlike the cosmetic class above this one
    #: CAN be a digit or an operator -- an unrecovered minus is a sign flip -- so
    #: the page must never be silently trusted.
    has_unrecovered_symbol_glyphs: bool = False
    attempts: list[PageOutput] = field(default_factory=list)  # all engine attempts
    best_output: PageOutput | None = None  # selected/reconciled best
    judge_rejected: bool = False  # VLM judge rejected the best output
    #: GH-225: how many image references on this page had no provenance in
    #: the source document and were removed. Non-zero demotes the DOCUMENT
    #: to AUDIT_FAILED; the page itself keeps its cleaned text and ships.
    fabricated_image_refs: int = 0
    #: GH-195: this page had >=1 text-strategy table grid rejected because a
    #: lane boundary split a native numeric token. The word-geometry rebuild
    #: is lossless, so the page keeps its text — but it is demoted to WARNING
    #: and the document to AUDIT_FAILED, because the issue requires the
    #: rejection to surface at page and document status, not only in a log.
    text_grid_rejected: bool = False
    native_table_structure_failed: bool = False  # native table text lost its grid
    #: GH-151 TICKET-B1: grid-shape defect (ragged / detached-label row pair)
    #: found in the native markdown at extraction time. Deliberately a
    #: SEPARATE field from ``native_table_structure_failed`` above:
    #: ``_score_per_page`` clears that flag on a passing heuristic score
    #: (orchestrator.py), which would silently erase this structural verdict
    #: if they were the same field. Set once in ``apply_born_digital``, never
    #: re-derived downstream.
    native_table_structure_defective: bool = False
    #: GH-200: header-attribution HARD verdict (destroyed header band) found
    #: in the native markdown at extraction time. Same treatment as
    #: ``native_table_structure_defective`` immediately above: a SEPARATE
    #: field, set once in ``apply_born_digital``, never re-derived
    #: downstream, deliberately absent from ``needs_repair`` (see that
    #: property's docstring -- forcing repair under --native-only is barred).
    native_table_header_unattributed: bool = False
    native_table_unverifiable: bool = False  # TR-3: per-region verifier flagged hard-fail
    scanned_table_evidence_failed: bool = False  # GH-90: source-evidence gate rejected table
    d3_floor_png_ref: str = ""  # TR-3: image ref string for the D3 floor PNG (empty if not saved)
    chart_asset_render_failed: bool = False  # PP-7: chart-lane PNG render failed

    @property
    def needs_repair(self) -> bool:
        """Whether this page still needs (re)processing.

        Born-digital pages with trusted native text never need repair.
        Born-digital pages with a known native-layer deficiency, or table pages
        that require structured OCR, need repair until a passing non-native
        attempt exists.
        If OCR has been attempted and failed, native_text serves as fallback
        (needs_repair returns False to avoid infinite repair loops) — UNLESS
        an audit explicitly rejected the output: a judge rejection or failed
        native table-structure gate means the existing reading is semantically
        wrong, so the page must get a real repair pass instead of silently
        reverting to flat native text.
        The repair loop still terminates: the router excludes tried engines,
        so an audit-rejected page runs out of candidates and is skipped.
        """
        if self.is_born_digital and self.native_text:
            # GH-151 TICKET-B1: native_table_structure_defective is
            # DELIBERATELY absent from this condition. Adding it would force
            # a repair pass even under --native-only, which is settled as
            # forbidden (docs/plans/gh151-structural-gate/TICKETS.md
            # TICKET-B1). The flag is honoured downstream instead, at
            # ``_score_per_page`` and the manifest/ship surfaces.
            if (
                self.needs_ocr_enhancement
                or self.native_table_structure_failed
                or self.chart_asset_render_failed  # PP-7: render failure treated as deficient
            ):
                # Prefer enhancement for pages with deficient native text, but
                # if it has been attempted and none passed, fall back to native.
                if self.best_output and self.best_output.audit_passed:
                    return False  # OCR succeeded
                if (
                    self.judge_rejected
                    or self.native_table_structure_failed
                    or self.chart_asset_render_failed
                ):
                    return True  # audit rejection demands a real repair pass
                non_native_attempts = [
                    a for a in self.attempts if not (a.engine or "").startswith("native")
                ]
                if non_native_attempts:
                    return False  # OCR tried but failed; native text is fallback
                return True  # No OCR attempted yet; request it
            return False
        return not self.best_output or not self.best_output.audit_passed

    @property
    def best_attempt(self) -> PageOutput | None:
        """The most usable attempt when no passing ``best_output`` exists.

        Audit-passing attempts win (most recent first); otherwise the attempt
        with the most text. Returns None when no attempt carries any text.
        """
        with_text = [a for a in self.attempts if a.text and a.text.strip()]
        if not with_text:
            return None
        passing = [a for a in with_text if a.audit_passed]
        if passing:
            return passing[-1]
        return max(with_text, key=lambda a: len(a.text))


@dataclass
class DocumentState:
    """Central blackboard for the OCR pipeline.

    Constructed from a ``DocumentHandle``; pre-populates one ``PageState``
    per page.  Engine results are merged via ``apply_result``.
    """

    handle: DocumentHandle
    status: DocumentStatus = DocumentStatus.PENDING
    pages: dict[int, PageState] = field(default_factory=dict)
    whole_doc_attempts: list[PageOutput] = field(
        default_factory=list
    )  # page_num=0 from CLI engines
    engine_runs: list[EngineResult] = field(
        default_factory=list
    )  # all EngineResult objects for telemetry
    events: list = field(default_factory=list)  # AuditEvent stream for the run audit log
    # Agentic routing ladder snapshot (B3) — populated by _phase_agentic, None otherwise.
    # Each entry: {provider_id, model, backend, cost_per_page_usd, tier}.
    agentic_ladder: list[dict] = field(default_factory=list)
    # Judge model used for agentic routing (B3) — "" when heuristic judge was used.
    agentic_judge_model: str = ""
    # PP-2 cascade halt: non-empty when the page loop stopped early because the
    # backend went unresponsive (e.g. "PARTIAL_SAVE_VLM_TIMEOUT"). Set by
    # _phase_agentic, read by _phase_assemble to propagate into EngineResult.error.
    pp2_halt_reason: str = ""

    def __post_init__(self) -> None:
        for i in range(1, self.handle.page_count + 1):
            if i not in self.pages:
                self.pages[i] = PageState(page_num=i)

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def apply_result(self, result: EngineResult) -> None:
        """Merge an engine's output into the blackboard."""
        self.engine_runs.append(result)
        for page_out in result.pages:
            if page_out.page_num == 0:
                self.whole_doc_attempts.append(page_out)
            else:
                page_state = self.pages.get(page_out.page_num)
                if page_state:
                    page_state.attempts.append(page_out)
                    # A passing attempt is promoted when there is no best yet
                    # OR the current best has since FAILED audit (scoring
                    # demotes in place). Without the second clause, a failed
                    # round-1 repair pinned best_output forever and every
                    # later PASSING repair attempt was silently discarded.
                    #
                    # GH-34: also require non-empty text (same predicate used
                    # by PageState.best_attempt). An empty-but-audit_passed
                    # repair must never overwrite or become best_output.
                    if (
                        page_out.audit_passed
                        and page_out.text
                        and page_out.text.strip()
                        and (not page_state.best_output or not page_state.best_output.audit_passed)
                    ):
                        page_state.best_output = page_out

    def apply_born_digital(self, assessment: DocumentAssessment) -> None:
        """Apply born-digital detection results.

        Propagates the full content-type vector (has_tables, has_figures,
        has_equations) onto PageState so downstream routing gates can read
        directly from PageState without re-consulting _last_assessment.
        """
        for pa in assessment.pages:
            if pa.page_num in self.pages:
                ps = self.pages[pa.page_num]
                ps.is_born_digital = pa.is_born_digital
                ps.has_tables = pa.has_tables
                ps.has_figures = pa.has_figures
                ps.has_equations = pa.has_equations
                ps.has_unmapped_math_glyphs = getattr(pa, "has_unmapped_math_glyphs", False)
                ps.has_encoding_hygiene_suspect = getattr(pa, "has_encoding_hygiene_suspect", False)
                ps.has_unrecovered_symbol_glyphs = getattr(
                    pa, "has_unrecovered_symbol_glyphs", False
                )
                if pa.is_born_digital:
                    ps.native_text = pa.native_text
                    ps.needs_ocr_enhancement = pa.needs_ocr_enhancement
                    # GH-151 TICKET-B1: propagate the grid-shape defect flag.
                    # Deliberately NOT added to needs_repair (state.py `needs_repair`
                    # property) so it can never force OCR under --native-only —
                    # that ruling is settled, see docs/plans/gh151-structural-gate/
                    # TICKETS.md TICKET-B1.
                    ps.native_table_structure_defective = getattr(
                        pa, "native_table_structure_defective", False
                    )
                    # GH-200: propagate the header-attribution defect flag.
                    # Same non-repair-forcing treatment as the line above.
                    ps.native_table_header_unattributed = getattr(
                        pa, "native_table_header_unattributed", False
                    )
                    # TR-3: propagate per-region verifier hard-fail flag so the
                    # D3 selection in _winning_page_output can route to the floor.
                    if getattr(pa, "has_unverifiable_table_region", False):
                        ps.native_table_unverifiable = True
                    # GH-195: carry the text-strategy grid rejection onto the page
                    # so it can reach page status and document status.
                    if getattr(pa, "text_grid_rejections", None):
                        ps.text_grid_rejected = True

    # ------------------------------------------------------------------
    # Read-only derived properties
    # ------------------------------------------------------------------

    @property
    def text(self) -> str:
        """Assemble current best document text.

        If only whole-doc attempts exist (CLI engines), return the last one.
        Otherwise, stitch per-page best outputs, preferring native text for
        born-digital pages.
        """
        has_per_page = any(p.best_output for p in self.pages.values())
        has_native = any(p.is_born_digital and p.native_text for p in self.pages.values())

        # If only whole-doc attempts exist (CLI engines) and at least one
        # passed audit, use the best passing attempt.
        if not has_per_page and self.whole_doc_attempts:
            passing = [w for w in self.whole_doc_attempts if w.audit_passed]
            if passing:
                return passing[-1].text
            # All whole-doc attempts failed audit. If we have born-digital
            # native text, prefer that over truncated/failed OCR.
            if has_native:
                return self._assemble_native_text()
            # Last resort: return the latest whole-doc attempt even if failed
            return self.whole_doc_attempts[-1].text

        texts: list[str] = []
        for i in range(1, self.handle.page_count + 1):
            p = self.pages[i]
            if p.best_output and p.best_output.audit_passed:
                # Prefer passing enhanced output where the native layer is known
                # deficient; otherwise native text below is authoritative.
                texts.append(p.best_output.text)
            elif p.is_born_digital and p.native_text:
                texts.append(p.native_text)
            elif p.best_output:
                texts.append(p.best_output.text)
            elif p.best_attempt:
                # A rejected-but-substantial attempt (e.g. judge-cleared
                # best_output) beats silently dropping the page.
                texts.append(p.best_attempt.text)
        return "\n\n---\n\n".join(texts)

    def _assemble_native_text(self) -> str:
        """Assemble document text from born-digital native text per page."""
        texts: list[str] = []
        for i in range(1, self.handle.page_count + 1):
            p = self.pages[i]
            if p.native_text:
                texts.append(p.native_text)
        return "\n\n---\n\n".join(texts)

    @property
    def pages_needing_repair(self) -> list[int]:
        """Page numbers (sorted) that still need (re)processing."""
        return [i for i, p in sorted(self.pages.items()) if p.needs_repair]

    @property
    def total_cost(self) -> float:
        """Sum of cost across all engine runs."""
        return sum(r.cost for r in self.engine_runs)

    @property
    def engines_used(self) -> list[str]:
        """Ordered unique list of engine names used so far."""
        return list(dict.fromkeys(r.engine for r in self.engine_runs))
