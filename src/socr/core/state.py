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

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

from socr.core.born_digital import DocumentAssessment
from socr.core.document import DocumentHandle
from socr.core.result import DocumentStatus, EngineResult, FailureMode, PageOutput

logger = logging.getLogger(__name__)


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
    has_corrupt_math: bool = False  # native equation glyphs are positively font-corrupted
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
    #: GH-271: the corrupt-equation lane (default on) produced a region hybrid
    #: that must ship even though it remains WARNING/audit_passed=False (syntax
    #: is not a mathematical-fidelity oracle). Read only by final winner selection.
    corrupt_math_hybrid: PageOutput | None = None
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
    #: #263: rotated page whose native layer is confetti (one glyph run per
    #: extracted line). Set once in ``apply_born_digital`` from the assessment
    #: flag of the same name, never re-derived. Read by
    #: ``manifest._winning_page_output``, which ships a failure marker instead
    #: of the fragments -- unreadable fragments under a SUCCESS are exactly the
    #: silent loss the cardinal rule forbids.
    native_rotated_text_shredded: bool = False
    native_table_structure_failed: bool = False  # native table text lost its grid
    #: Backward-compatible aggregate of GH-226 raw emission, GH-190 raw
    #: content, and GH-151 parsed grid-shape defects. Deliberately a SEPARATE
    #: field from ``native_table_structure_failed`` above: ``_score_per_page``
    #: clears that flag on a passing heuristic score (orchestrator.py), which
    #: would silently erase this verdict if they were the same field. Set once
    #: in ``apply_born_digital``, never re-derived downstream.
    native_table_structure_defective: bool = False
    #: GH-226 exact raw-emission defect code, kept separate from the aggregate
    #: above so it remains exact emission provenance.
    native_table_emission_defect: str = ""
    # GH-303: the GH-190 empty-body term, carried separately so the audit can name
    # it instead of reporting it as GH-151's `grid_shape`.
    native_table_content_defect: str = ""
    #: GH-200: header-attribution HARD verdict (destroyed header band) found
    #: in the native markdown at extraction time. Same treatment as
    #: ``native_table_structure_defective`` immediately above: a SEPARATE
    #: field, set once in ``apply_born_digital``, never re-derived
    #: downstream, deliberately absent from ``needs_repair`` (see that
    #: property's docstring -- forcing repair under --native-only is barred).
    native_table_header_unattributed: bool = False
    native_table_unverifiable: bool = False  # TR-3: per-region verifier flagged hard-fail
    #: GH-371: zero-based ordinals of native table regions whose per-region
    #: verifier hard-failed.  Empty when no failed region identity is available.
    native_table_unverifiable_ordinals: list[int] = field(default_factory=list)
    #: GH-371: number of native table regions examined by the per-region verifier.
    native_table_region_count: int = 0
    #: GH-375: per-ordinal ``table_grid_identity`` of the y0-sorted native
    #: regions. Empty when identity was not captured. A mismatch against
    #: ``find_table_blocks`` fails the regional splice closed.
    native_table_region_identities: list[str] = field(default_factory=list)
    #: GH-520: the DETECTION-level table count and bboxes, from
    #: ``find_tables()`` before any reconstruction ran. Unlike
    #: ``native_table_region_count`` above -- which the GFM parser produces and
    #: which therefore agrees with whatever that parser managed -- these two can
    #: contradict it, which is the only reason the structure-class floor is
    #: allowed to splice regionally again. Persisted and restored, so a resumed
    #: page reaches the same verdict as the run that measured it.
    detected_table_count: int = 0
    detected_table_bboxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    scanned_table_evidence_failed: bool = False  # GH-90: source-evidence gate rejected table
    #: What this page has actually SPENT, as a recorded fact rather than a
    #: derivation (cold review rounds 4-5). Every site that journals an
    #: ``EngineResult`` for this page adds to it: each ladder rung ``route_page``
    #: tried, the GH-96 escalation call INCLUDING its rejected branch (which pays
    #: for the call and then returns without appending a candidate), and the
    #: post-route crop re-judge.
    #:
    #: It cannot be recomputed from ``attempts``: that list never held the
    #: refused escalation candidate, and resume rebuilds it as the single frozen
    #: winner -- so a derivation lost the spend of every rung the ladder paid for
    #: and refused, and lost it again on every later sidecar rewrite.
    #:
    #: ``None`` means "unknown", exactly as ``DocumentState.total_cost`` uses it:
    #: one unmetered call makes the page total unknowable, and an unknown
    #: subtotal must never be treated as zero spend.
    page_cost_usd: float | None = 0.0
    d3_floor_png_ref: str = ""  # TR-3: image ref string for the D3 floor PNG (empty if not saved)
    #: #263: image ref for the shredded-rotated-page floor. Deliberately NOT
    #: ``d3_floor_png_ref``: that field means "the table region was routed to
    #: the image lane", and a consumer must be able to tell the two floors
    #: apart.
    rotated_shred_png_ref: str = ""
    chart_asset_render_failed: bool = False  # PP-7: chart-lane PNG render failed
    #: GH-318: chart ELIGIBILITY detection raised and the page fell through to
    #: the non-chart route. Distinct from ``chart_asset_render_failed`` above,
    #: which means the page WAS a chart and its PNG failed to render; here the
    #: pipeline never learned whether the page is a chart at all, so the routing
    #: decision is unresolved rather than wrong. Deliberately absent from
    #: ``needs_repair`` (below): a detector crash must not force the chart lane
    #: or a repair pass — it is honoured downstream at the document buckets.
    chart_asset_detection_failed: bool = False
    #: S1/MAJOR-7(b): persisted answer to ``structure_class_grid_winner(p) is
    #: not None`` from the run that produced the terminal sidecar, restored by
    #: ``_restore_terminal_page_state`` on resume. Needed because resume
    #: collapses ``p.attempts`` down to the single frozen winner, so a resumed
    #: run's own attempt list can no longer prove a non-native rung authored a
    #: grid on the ORIGINAL pass -- without this flag,
    #: ``_reaches_structure_class_branch``'s early-return mirror sees a clean
    #: non-native ``audit_passed`` winner and (correctly, by its own logic)
    #: treats it as an ordinary passing page, silently dropping it from
    #: ``structure_class_model_pages`` on the second run even though the same
    #: input produced it on the first.
    structure_class_model_kept_on_resume: bool = False
    #: P6 cold review round 2: the ``PageDisposition`` the run that produced the
    #: terminal sidecar published for this page, as its serialized dict, restored
    #: by ``_restore_terminal_page_state``. Same problem shape as the flag above --
    #: resume collapses ``p.attempts`` to the single frozen winner, so a resumed
    #: re-flush recomputes a DIFFERENT disposition for the same page (an accepted
    #: winner demoted by the ladder guard comes back as ``UNACCEPTED_OUTPUT_KEPT``
    #: rather than the ``ACCEPTED_OUTPUT`` run 1 recorded) and the sidecar bytes
    #: move on a run that reprocessed nothing.
    #:
    #: Read only when building the page's published disposition. Nothing in the
    #: resume GATE consults it, and no bucket is derived from it, so it cannot
    #: change which pages are skipped or how a document is scored.
    resumed_disposition: dict | None = None
    #: P4-R (cold review round 1, finding 5): the equation-region lane wanted to
    #: read this page and could not, because no available provider served the
    #: clean-equation model. That is TRANSIENT external state, not configuration,
    #: so the run fingerprint does not describe it -- and a page whose only
    #: defect is "the model was not up" would otherwise be written terminal
    #: SUCCESS and restored forever, leaving a default-on lane permanently inert
    #: on that page. Persisted in the page sidecar and read by
    #: ``_load_terminal_page``, which refuses to skip such a page so the next run
    #: with a provider actually reads it. Deliberately NOT a status demotion: the
    #: page's shipped bytes, status and audit verdict are exactly what they are
    #: with the lane switched off, per the no-provider parity requirement.
    equation_lane_retry_pending: bool = False
    #: GH-353 TICKET-B1: the table judge ladder's page-level disposition
    #: (``FailureMode.TABLE_REJECTED`` / ``TABLE_UNVERIFIED`` / ``None``).
    #: This is the durable, pre-guard signal C3's manifest guard
    #: (``_apply_ladder_disposition_guard``) reads to demote a page AFTER
    #: winner selection, and C2's ``_table_ladder_terminal`` reads for
    #: document aggregation. Set once, in the agentic loop, by the gate;
    #: never derived from ``best_output`` (mutating a shipped attempt's
    #: ``audit_passed``/``failure_mode`` in place would make assemble
    #: discard the page's text -- the #252 round-1 rule).
    table_ladder_disposition: FailureMode | None = None
    #: GH-359 (cubic P1): True when at least one emitted table on this page
    #: reached assemble with NO ladder terminal of its own. The disposition
    #: above is a page-level reduction, so a page holding one REJECTED table
    #: keeps ``TABLE_REJECTED`` even when a SECOND table was never witnessed.
    #: D1b's resume exception skips a REJECTED page on the grounds that "both
    #: rungs looked and said no" -- true of the rejected table, false of the
    #: unwitnessed one. This flag withholds that exception so the page is
    #: reprocessed and the unwitnessed table finally gets a look.
    table_ladder_incomplete: bool = False
    #: GH-367: per-table_id record of the last binding-contradiction
    #: adjudication on this page (``{"status": "lifted"|"held", ...}``).
    #: Restored from the sidecar so a resumed run does not silently
    #: re-clamp a table whose contradictions were already disproved.
    binding_adjudication: dict = field(default_factory=dict)
    #: P1 retry latch: transient external state absent from the run fingerprint.
    #: Set when at least one table on this page reached an UNVERIFIED terminal
    #: caused by an unavailable table judge rung (transport failure, missing
    #: binary, or unexpected exception), so resume re-judges the page when the
    #: rung returns rather than restoring a terminal caused by a temporary outage.
    table_judge_retry_pending: bool = False
    #: Which rung KINDS ("ollama", "gemini") were unavailable when that latch
    #: was set. Cold review round 3: a bare bit let a healthy rung 1 stand in
    #: for a rung 2 that was still down, so resume reopened the document and
    #: re-ran the whole ladder every time. Empty means "not recorded" -- a
    #: sidecar written before this field existed -- and the gate then falls
    #: back to asking about any rung.
    table_judge_retry_rungs: list[str] = field(default_factory=list)
    #: VI-B1: exclusive wall-clock seconds for this page's agentic loop.
    #: Keys: route, extract, tables, ladder, adjudication, figures, equations,
    #: flush, total. Nested children (extract under route for OCR; ladder and
    #: adjudication under tables) are subtracted from the parent so the
    #: exclusive keys sum to ``total``. Measurement only: never consulted by
    #: resume, fingerprint, or winner selection. Empty when the page was never
    #: timed (old sidecar, or a flush that did not run the page loop).
    timings_s: dict[str, float] = field(default_factory=dict)

    def is_structure_class(self) -> bool:
        """C2: pages whose native branch may never author a GRID.

        Backed by ``has_tables`` -- the field ``apply_born_digital`` sets from
        detected table structures -- so the orchestrator's OCR-bypass routing
        (``_page_has_tables``) and the manifest's winner selection
        (``_winning_page_output``) read the same source and cannot diverge on
        what counts as structure-class.

        Equations are deliberately OUT of scope (BLOCKING 1 on #269): the S1
        case (i)/(iii) branch only accepts a GFM table as a "grid was
        authored" signal (``_grid_authored_attempt`` requires
        ``find_table_blocks``), so routing an equation page through the same
        branch cannot select anything a ``$...$``-reading model attempt
        produces -- it either ships an accepted hallucination or falls
        through to case (iii) and demotes a page that shipped a free native
        SUCCESS before S1 to WARNING, flipping the document to
        AUDIT_FAILED for no gain. Equations get the model-rung guarantee and
        grid-authorship rule once a non-GFM acceptance path exists to
        actually select between a native and model equation reading; until
        then this stays tables-only.
        """
        return bool(self.has_tables)

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
        The explicit corrupt-math hybrid is handled but remains unverified: its
        crop-backed region candidate must not trigger a generic whole-page repair
        that can degrade surrounding native prose.
        The repair loop still terminates: the router excludes tried engines,
        so an audit-rejected page runs out of candidates and is skipped.
        """
        if self.corrupt_math_hybrid is not None:
            return False
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


def add_page_cost(ps, cost: float | None) -> None:
    """Charge *cost* to a page's recorded spend (``None`` absorbing both ways).

    Rounds 4-6. Page spend is a RECORDED FACT, never derived: ``attempts`` never
    held a refused escalation candidate, and resume collapses it to the single
    frozen winner. ``None`` means unknown, exactly as ``total_cost`` uses it --
    an unmetered call makes the page total unknowable, and unknown never decays
    back to a number.

    Fails OPEN, loudly. The only caller that can raise here holds a page object
    without the field, and letting that propagate would abort a lane inside its
    own fail-open guard and silently keep the incumbent text. Under-recorded
    spend is a warning; lost content is not recoverable.
    """
    if ps is None:
        return
    try:
        if cost is None or getattr(ps, "page_cost_usd", 0.0) is None:
            ps.page_cost_usd = None
            return
        ps.page_cost_usd = (getattr(ps, "page_cost_usd", 0.0) or 0.0) + cost
    except Exception as exc:
        logger.warning("page spend not recorded (%s); the page total is understated", exc)


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
    #: PRIVATE. Cold review round 7: rounds 5-6 tried to hold the
    #: "journal and record page spend in one call" contract with an AST guard,
    #: and the review defeated it four ways -- an alias then ``append``, a
    #: ``list(...) + [...]`` reassignment, a ``getattr`` hop, and a subclass
    #: method that simply took the exempt name. A pattern-matcher cannot win
    #: that game, so the contract is enforced by ENCAPSULATION instead: the
    #: journal is private, ``engine_runs`` below is a read-only view, and every
    #: one of those shapes now raises where it is written.
    #: ``init=False``: the journal is not a constructor or ``dataclasses.replace``
    #: input either (cold review round 7) -- a fresh list installed that way
    #: would carry runs no page was charged for. Reflective writes
    #: (``object.__setattr__``, ``__dict__``) are Python, not a contract this
    #: class can close, and are out of scope.
    _engine_runs: list[EngineResult] = field(
        init=False, default_factory=list, repr=False
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

    @property
    def engine_runs(self) -> tuple[EngineResult, ...]:
        """The run journal, read-only.

        A tuple rather than the list itself, and a property with NO setter, so
        ``append`` / ``extend`` / ``insert`` raise, ``+=`` and plain assignment
        raise, and a ``getattr`` hop or a local alias inherits the same tuple --
        every bypass the round-6 review found now fails at the line that writes
        it. Readers only ever iterate, index or count, all of which a tuple does.
        """
        return tuple(self._engine_runs)

    def record_engine_run(
        self, result: EngineResult, page_nums: Sequence[int] | None = None
    ) -> None:
        """Journal an engine run AND charge its cost to the pages it ran on.

        Cold review round 6 closed the CLASS, not the instance. Rounds 4-5 made
        per-page spend a recorded fact and wired the sites that existed; the
        corrupt-math recovery lane still journaled a page run with an UNKNOWN
        cost and recorded nothing, so the page kept the default known zero and a
        resumed run read unmetered spend as no spend. That is what a two-call
        contract buys you: the obvious way to add a lane is the wrong way.

        So this is the ONE place the private journal is appended to. Round 7 made
        that structural rather than advisory: ``engine_runs`` is a read-only view,
        so there is no other way in, and a small scoped guard
        (``tests/test_p35_cold_review_round7.py``) keeps the private name from
        being reached for outside this class.

        ``page_nums`` names the pages the run is charged to. Omitted, it is taken
        from the result's own page outputs. An EMPTY sequence means "already
        recorded" -- the resume path, which restores the page's persisted fact
        verbatim and must not add to it.
        """
        self._engine_runs.append(result)
        charged = page_nums
        if charged is None:
            charged = [p.page_num for p in (result.pages or []) if getattr(p, "page_num", 0)]
        for num in dict.fromkeys(charged):
            add_page_cost(self.pages.get(num), result.cost)

    def apply_result(self, result: EngineResult) -> None:
        """Merge an engine's output into the blackboard."""
        self.record_engine_run(result)
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
                ps.has_corrupt_math = pa.has_corrupt_math
                ps.has_unmapped_math_glyphs = getattr(pa, "has_unmapped_math_glyphs", False)
                ps.has_encoding_hygiene_suspect = getattr(pa, "has_encoding_hygiene_suspect", False)
                ps.has_unrecovered_symbol_glyphs = getattr(
                    pa, "has_unrecovered_symbol_glyphs", False
                )
                if pa.is_born_digital:
                    ps.native_text = pa.native_text
                    ps.needs_ocr_enhancement = pa.needs_ocr_enhancement
                    # Propagate the backward-compatible native-table aggregate
                    # (raw emission, raw content, and parsed shape defects).
                    # Deliberately NOT added to needs_repair (state.py `needs_repair`
                    # property) so it can never force OCR under --native-only —
                    # that ruling is settled, see docs/plans/gh151-structural-gate/
                    # TICKETS.md TICKET-B1.
                    ps.native_table_structure_defective = getattr(
                        pa, "native_table_structure_defective", False
                    )
                    # Preserve the exact GH-226 raw-emission provenance
                    # separately; content and shape defects do not populate it.
                    ps.native_table_emission_defect = str(
                        getattr(pa, "native_table_emission_defect", "") or ""
                    )
                    # GH-303: same treatment for the GH-190 content term.
                    ps.native_table_content_defect = str(
                        getattr(pa, "native_table_content_defect", "") or ""
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
                    # GH-371: preserve failed-region identity for the regional
                    # D3 splice and its sidecar resume path.
                    ps.native_table_unverifiable_ordinals = list(
                        getattr(pa, "native_table_unverifiable_ordinals", []) or []
                    )
                    ps.native_table_region_count = getattr(pa, "native_table_region_count", 0)
                    ps.native_table_region_identities = list(
                        getattr(pa, "native_table_region_identities", []) or []
                    )
                    # GH-520: the independent signal, carried alongside the
                    # parser-derived one it exists to contradict.
                    ps.detected_table_count = getattr(pa, "detected_table_count", 0)
                    ps.detected_table_bboxes = [
                        tuple(b) for b in (getattr(pa, "detected_table_bboxes", []) or [])
                    ]
                    # GH-195: carry the text-strategy grid rejection onto the page
                    # so it can reach page status and document status.
                    if getattr(pa, "text_grid_rejections", None):
                        ps.text_grid_rejected = True
                    # #263: carry the rotated-shredded verdict so the ship
                    # surface can refuse the confetti.
                    ps.native_rotated_text_shredded = getattr(
                        pa, "native_rotated_text_shredded", False
                    )

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
            if p.corrupt_math_hybrid is not None:
                texts.append(p.corrupt_math_hybrid.text)
            elif p.best_output and p.best_output.audit_passed:
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
    def total_cost(self) -> float | None:
        """Sum known spend, or ``None`` when any executed run is unmetered."""
        costs = [run.cost for run in self.engine_runs]
        if any(cost is None for cost in costs):
            return None
        return sum(cost for cost in costs if cost is not None)

    @property
    def engines_used(self) -> list[str]:
        """Ordered unique list of engine names used so far."""
        return list(dict.fromkeys(r.engine for r in self.engine_runs))
