"""Per-document execution ledger + reproducibility manifest.

A manifest records, for every page of a document, (a) an *input fingerprint*
that captures everything determining what the output should be, and (b) a
*blob_ref* pointing at the frozen winning ``PageOutput`` in the content-addressed
``BlobStore``.

Two distinct jobs, deliberately not conflated (this was the load-bearing
correction from the design review):

  - **Replay** = reconstruct the document by fetching cached blobs. NO engine is
    invoked, so output is bit-identical regardless of VLM non-determinism or
    provider model drift. This is what ``socr replay`` does.
  - **Invalidation** = decide whether a cached page is still trustworthy. The
    fingerprint covers the rendered-image hash (not just PDF bytes — a renderer
    upgrade changes pixels), render params, engine id + model version, prompt
    template, and the normalizer/assembly versions. If any component changes, the
    fingerprint changes and the entry is known-stale; a fresh ``socr agent`` run
    can re-OCR only those pages.

The manifest is a plain JSON file; the blobs live in the BlobStore. Together they
are the corpus's reproducible record.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ocr_output_contract import (
    PAGE_MARKER_RE,
    assemble_pages,
    run_fingerprint,
    split_native_pages,
)

from socr.core.cache import BlobStore
from socr.core.document import DocumentHandle
from socr.core.result import FailureMode, PageOutput, PageStatus
from socr.core.state import DocumentState

logger = logging.getLogger(__name__)

# Bump these when the corresponding logic changes in a way that should
# invalidate cached pages. They are part of every page fingerprint.
# v2 (issue #38): normalizer — HTML tables converted to markdown instead of
# tag-stripped into fused digit-streams, math-preserving NFKC; assembly —
# attempts fallback, explicit failure markers, flagged native fallback.
# Pages cached under v1 carry the fabricated-number corruption and MUST be
# invalidated.
# v3 (issue #92): native-layer cleaning at the born-digital extraction boundary —
# zero-width spaces / soft hyphens stripped, exotic spaces normalized. This changes
# the saved native bytes for any born-digital page that carried publisher invisibles
# (native text bypasses OutputNormalizer), so pages cached under v2 must reprocess.
MANIFEST_SCHEMA_VERSION = "1"
NORMALIZER_VERSION = "3"
ASSEMBLY_VERSION = "2"

# Legacy page separator. socr now assembles bodies and replays with the
# contract's ``assemble_pages`` (``## Page N`` headers); this constant is kept
# only for backward-compatible imports and is no longer used to join pages.
PAGE_SEPARATOR = "\n\n---\n\n"

# Matches the canonical failure marker `[page N failed: no usable OCR output]`.
_PAGE_FAILED_RE = re.compile(r"^\[page \d+ failed: no usable OCR output\]$")
# TR-3: also matches the D3 fail-closed marker `[page N failed: unverifiable table …]`
# which may be followed by a PNG image ref on a second block (the full text starts
# with the marker, possibly with trailing whitespace and an image ref).  We match
# on the FIRST LINE only so both formats are recognised by is_page_failed_marker.
_PAGE_FAILED_ANY_RE = re.compile(r"^\[page \d+ failed:.*", re.MULTILINE)


def page_failed_marker(page_num: int) -> str:
    """Explicit in-document marker for a page that produced nothing.

    Shipped instead of silent emptiness: a reader diffing the output against
    the source must not need to count page headers to notice a missing page
    (the Kuttner-Table-2 failure mode).

    The D3 fail-closed floor (TR-3) ships a VARIANT marker:
    ``[page N failed: unverifiable table — see image]\\n\\n![...](...)``.
    That marker is also recognised by ``is_page_failed_marker``.
    """
    return f"[page {page_num} failed: no usable OCR output]"


def is_page_failed_marker(text: str) -> bool:
    """True if a page's canonical text is a failure marker (not real content).

    Matches both the original ``[page N failed: no usable OCR output]`` marker
    AND the TR-3 D3 fail-closed variant ``[page N failed: unverifiable table …]``
    (which may be followed by a PNG image ref block).
    """
    stripped = text.strip()
    # Fast path: exact original marker.
    if _PAGE_FAILED_RE.match(stripped):
        return True
    # TR-3 D3 variant: text starts with `[page N failed: ...]` on its first line.
    return bool(_PAGE_FAILED_ANY_RE.match(stripped))


def compute_image_hash(handle: DocumentHandle, page_num: int, dpi: int) -> str:
    """SHA-256 of the rendered page bytes at a given DPI.

    Hashing the *rendered* image (not the PDF bytes) means a PyMuPDF upgrade that
    changes rasterization invalidates the cache entry, as it should.
    """
    img = handle.render_page(page_num, dpi=dpi)
    return hashlib.sha256(img.tobytes()).hexdigest()


@dataclass
class PageFingerprint:
    """Everything that determines what a page's OCR output should be."""

    pdf_file_hash: str
    page_num: int
    render_dpi: int
    engine: str
    model_version: str = ""
    image_hash: str = ""  # empty for native-text pages (no rasterization involved)
    prompt_hash: str = ""
    normalizer_version: str = NORMALIZER_VERSION
    assembly_version: str = ASSEMBLY_VERSION

    def key(self) -> str:
        """Stable hash of the fingerprint — the invalidation identity of a page."""
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class ManifestEntry:
    """One page's record: fingerprint + pointer to its frozen output blob."""

    page_num: int
    blob_ref: str  # content hash of the winning PageOutput in the BlobStore
    fingerprint: PageFingerprint
    journal: list[dict] = field(default_factory=list)  # provenance: attempts tried

    def to_dict(self) -> dict:
        return {
            "page_num": self.page_num,
            "blob_ref": self.blob_ref,
            "fingerprint": asdict(self.fingerprint),
            "journal": self.journal,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ManifestEntry:
        return cls(
            page_num=d["page_num"],
            blob_ref=d["blob_ref"],
            fingerprint=PageFingerprint(**d["fingerprint"]),
            journal=d.get("journal", []),
        )


@dataclass
class Manifest:
    """Document-level reproducibility record."""

    pdf_filename: str
    pdf_file_hash: str
    page_count: int
    render_dpi: int
    entries: dict[int, ManifestEntry] = field(default_factory=dict)
    schema_version: str = MANIFEST_SCHEMA_VERSION
    # Agentic routing ladder snapshot (B3): ordered list of providers tried,
    # with their cost/tier info. None when run was NOT in agentic mode.
    agentic_ladder: list[dict] | None = None
    # Judge model used for agentic routing (B3) — "" when heuristic judge was used.
    agentic_judge_model: str = ""

    def to_dict(self) -> dict:
        d = {
            "schema_version": self.schema_version,
            "pdf_filename": self.pdf_filename,
            "pdf_file_hash": self.pdf_file_hash,
            "page_count": self.page_count,
            "render_dpi": self.render_dpi,
            "entries": {str(k): v.to_dict() for k, v in sorted(self.entries.items())},
        }
        if self.agentic_ladder is not None:
            d["agentic_ladder"] = self.agentic_ladder
        if self.agentic_judge_model:
            d["agentic_judge_model"] = self.agentic_judge_model
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Manifest:
        return cls(
            pdf_filename=d["pdf_filename"],
            pdf_file_hash=d["pdf_file_hash"],
            page_count=d["page_count"],
            render_dpi=d["render_dpi"],
            schema_version=d.get("schema_version", MANIFEST_SCHEMA_VERSION),
            entries={int(k): ManifestEntry.from_dict(v) for k, v in d.get("entries", {}).items()},
            agentic_ladder=d.get("agentic_ladder"),
            agentic_judge_model=d.get("agentic_judge_model", ""),
        )

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> Manifest:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class _WholeDoc:
    """The whole-document attempt chosen to recover per-page text from."""

    texts: dict[int, str]
    engine: str
    audit_passed: bool


def _whole_doc_page_texts(state: DocumentState) -> _WholeDoc | None:
    """Per-page texts recovered from a whole-document CLI attempt.

    CLI engines that process a whole PDF in one shot return a single
    ``PageOutput(page_num=0)`` stored in ``state.whole_doc_attempts``; the
    per-page ``best_output`` slots are never populated. Without this, the
    manifest would freeze empty pages and ``replay`` would reconstruct an empty
    document even though the saved ``.md`` has the full text (the historical
    replay/manifest bug).

    We recover per-page text by splitting the winning whole-doc markdown on the
    canonical ``## Page N`` headers via the shared contract splitter. socr now
    emits ``## Page N`` bodies (via the contract's ``assemble_pages``), so a
    well-formed whole-doc blob round-trips. A blob with no markers (a legacy or
    non-converged engine) splits to a single page; the caller reconciles the
    split count against ``handle.page_count`` rather than trusting it blindly.

    Returns ``None`` when there is no usable whole-doc attempt (so the per-page
    path is used unchanged). The chosen attempt's REAL ``engine`` and
    ``audit_passed`` are carried through so the synthetic per-page output is NOT
    fabricated as a passing page when the only attempt FAILED audit.
    """
    if not state.whole_doc_attempts:
        return None
    passing = [w for w in state.whole_doc_attempts if w.audit_passed]
    chosen = passing[-1] if passing else state.whole_doc_attempts[-1]
    text = chosen.text or ""
    if not text.strip():
        return None
    pages = split_native_pages(text)
    return _WholeDoc(
        texts={i: t for i, t in enumerate(pages, start=1)},
        engine=chosen.engine or "cli",
        audit_passed=bool(chosen.audit_passed),
    )


def _native_text_with_appends(p) -> str:
    """The native text INCLUDING content appended after extraction.

    ``p.native_text`` is frozen when the text layer is read. Later phases splice
    extra content onto the ``PageOutput`` object in place -- notably GH-36b's
    equation LaTeX sidecar -- without ever updating ``native_text``. Demoting a
    page's table trust must not silently revert it to that pre-append snapshot.

    Reading from ``attempts`` rather than ``best_output`` is deliberate and closes
    two distinct holes:

    * ``DocumentState.apply_result`` only promotes an ``audit_passed`` output to
      ``best_output``, so a page demoted for table distrust never has one at all
      on the deterministic path.
    * ``_score_per_page`` explicitly clears ``best_output`` when it demotes the
      latest attempt under ``--native-only``.

    ``attempts`` survives both. Only a *native* attempt whose text EXTENDS the
    frozen snapshot is accepted, so this can never substitute OCR text or
    different content for the native reading -- if nothing extends it, the
    snapshot is returned unchanged.
    """
    base = p.native_text or ""
    if not base:
        return base
    for attempt in reversed(p.attempts or ()):
        text = attempt.text or ""
        if (attempt.engine or "").startswith("native") and len(text) > len(base):
            if text.startswith(base):
                return text
    return base


def _winning_page_output(
    state: DocumentState,
    page_num: int,
    whole_doc: _WholeDoc | None = None,
) -> PageOutput:
    """The PageOutput that should be frozen for this page.

    Mirrors ``DocumentState.text`` selection: a passing OCR best_output wins;
    otherwise born-digital native text; otherwise text recovered from a
    whole-document CLI attempt (split on ``## Page N``); otherwise the best
    attempt we have. Native and whole-doc fallbacks are wrapped in a synthetic
    PageOutput so the manifest always records real content, never an empty page.

    The whole-doc fallback carries the CHOSEN attempt's real ``engine`` and
    ``audit_passed``/status — a blob that FAILED audit is frozen as
    ``AUDIT_FAILED`` / ``audit_passed=False``, never fabricated as SUCCESS.
    """
    p = state.pages[page_num]
    if p.best_output and p.best_output.audit_passed:
        # An ``audit_passed`` native output on a page whose table was flagged
        # unverifiable is a CONTRADICTION, and the contradiction must lose to
        # the distrust flag rather than short-circuit past it. This state is
        # reachable without any live scoring bug: the resume ledger's
        # fingerprint has no source-version component (#214), so a page marked
        # terminal SUCCESS by an older build is restored verbatim -- carrying
        # ``audit_passed=True`` alongside ``native_table_unverifiable=True``.
        # Falling through here re-demotes it through the normal path below.
        native_unverifiable = (p.best_output.engine or "").startswith("native") and getattr(
            p, "native_table_unverifiable", False
        )
        if not native_unverifiable:
            return p.best_output
    # GH-90: scanned-table fail-closed floor.  When the source-evidence gate
    # rejected a VLM-emitted markdown table on a scan, shipping the fluent
    # hallucination is worse than an explicit failure marker — same D3 pattern.
    if (
        not p.is_born_digital
        and getattr(p, "scanned_table_evidence_failed", False)
        and bool(p.attempts)
    ):
        d3_marker = f"[page {page_num} failed: unverifiable table — see image]"
        png_ref = getattr(p, "d3_floor_png_ref", "")
        d3_text = f"{d3_marker}\n\n{png_ref}" if png_ref else d3_marker
        return PageOutput(
            page_num=page_num,
            text=d3_text,
            status=PageStatus.ERROR,
            engine=p.best_output.engine if p.best_output else "qwen",
            audit_passed=False,
            failure_mode=FailureMode.HALLUCINATION,
        )
    if p.is_born_digital and p.native_text:
        # TR-3: D3 fail-closed floor.  When the OCR ladder failed for a table
        # page AND the per-region geometry verifier flagged a hard-fail
        # (geometry_impossible_collapse), shipping the collapsed native text
        # risks a plausible-but-wrong artifact (silent column-shift or merged
        # rows).  The panel verdict Q1=D3: "ship neither flawed table — emit an
        # explicit failed-table marker; route the region to the image-asset lane."
        # A wrong/shifted number is worse than an obviously-missing one.
        if (
            p.native_table_structure_failed
            and getattr(p, "native_table_unverifiable", False)
            and bool(p.attempts)
        ):
            # TR-3: D3 fail-closed floor text = failed-table marker + image ref.
            # The marker is always present (makes the failure loud and greppable);
            # the PNG image ref lets a human SEE the table without transcription.
            # ``ps.d3_floor_png_ref`` is set by ``_render_d3_floor_png`` in
            # ``_phase_agentic`` right after ``native_table_structure_failed``
            # is set.  If the render failed (or no figures_dir was available),
            # d3_floor_png_ref is "" and only the marker is shipped — still
            # fail-closed, still no plausible-but-wrong table text.
            d3_marker = f"[page {page_num} failed: unverifiable table — see image]"
            png_ref = getattr(p, "d3_floor_png_ref", "")
            d3_text = f"{d3_marker}\n\n{png_ref}" if png_ref else d3_marker
            return PageOutput(
                page_num=page_num,
                text=d3_text,
                status=PageStatus.ERROR,
                engine="native",
                audit_passed=False,
                failure_mode=FailureMode.NATIVE_TABLE_STRUCTURE_FAILED,
            )

        # An enhancement page (native layer known deficient) whose recovery was
        # tried and never passed ships native text
        # as a FALLBACK, not a success: flagged WARNING / audit_passed=False
        # so the manifest and run summary stop stamping silent reversions as
        # passing pages.
        native_table_defect = p.native_table_structure_failed or getattr(
            p, "native_table_unverifiable", False
        )
        native_is_fallback = (
            p.needs_ocr_enhancement
            or native_table_defect
            or p.chart_asset_render_failed  # PP-7: render failure must stay WARNING
        ) and bool(p.attempts)
        # GH-211 MAJOR-1: never ship the frozen ``p.native_text`` snapshot when a
        # native attempt carries content appended after extraction (GH-36b's
        # equation sidecar). See ``_native_text_with_appends``: it reads from
        # ``attempts``, which survives both ``apply_result``'s audit_passed gate
        # and ``_score_per_page``'s explicit ``best_output = None`` on demotion.
        # Reading ``best_output`` here instead would drop the sidecar on the
        # deterministic --native-only path, which is exactly the path this
        # ticket is about.
        fallback_text = _native_text_with_appends(p)
        return PageOutput(
            page_num=page_num,
            text=fallback_text,
            status=PageStatus.WARNING if native_is_fallback else PageStatus.SUCCESS,
            engine="native",
            audit_passed=not native_is_fallback,
            failure_mode=(
                FailureMode.NATIVE_TABLE_STRUCTURE_FAILED
                if native_table_defect and native_is_fallback
                else FailureMode.NONE
            ),
        )
    # Whole-document CLI path: recover this page's text from the split markdown.
    # Consulted BEFORE a FAILED per-page best_output so a whole-doc attempt that
    # carries real content for this page is not shadowed (the prior ordering left
    # whole-doc recovery dead-coded behind any non-None best_output).
    if whole_doc and page_num in whole_doc.texts and whole_doc.texts[page_num].strip():
        # A blob that FAILED audit is frozen with audit_passed=False and a
        # non-SUCCESS status (WARNING: content present, audit not passed) so the
        # manifest never fabricates known-bad output as a passing page. An
        # EMPTY section in the split (``## Page N`` headers with nothing
        # between) falls through to the attempts/marker logic below instead of
        # shipping a silent empty page stamped with the blob's passing audit.
        return PageOutput(
            page_num=page_num,
            text=whole_doc.texts[page_num],
            status=PageStatus.SUCCESS if whole_doc.audit_passed else PageStatus.WARNING,
            engine=whole_doc.engine,
            audit_passed=whole_doc.audit_passed,
        )
    # A failed per-page attempt (content present, audit not passed) beats an
    # empty page so the manifest preserves what little we have.
    if p.best_output:
        return p.best_output
    # The documented-but-previously-missing fallback: when scoring/judging
    # cleared ``best_output`` and repair produced nothing, the rejected text
    # still lives in ``attempts``. Ship it flagged rather than erasing the
    # page (the silent-empty-page failure mode).
    attempt = p.best_attempt
    if attempt is not None:
        return PageOutput(
            page_num=page_num,
            text=attempt.text,
            status=PageStatus.WARNING,
            engine=attempt.engine,
            audit_passed=False,
            failure_mode=attempt.failure_mode,
        )
    # Nothing anywhere produced text: ship an EXPLICIT failure marker, never
    # a silent gap between page headers.
    return PageOutput(
        page_num=page_num,
        text=page_failed_marker(page_num),
        status=PageStatus.ERROR,
        audit_passed=False,
    )


def _strip_leading_page_marker(text: str) -> str:
    """Drop a leading ``## Page N`` header from a per-page text, if present.

    A page's recovered text may already carry its own ``## Page N`` header (e.g.
    a whole-doc CLI blob split back into pages, or a page that came from a CLI
    that emitted the canonical header). ``assemble_pages`` re-adds the canonical
    header, so stripping a pre-existing leading marker prevents a DOUBLE header
    (``## Page 1\\n\\n## Page 1\\n\\n...``) and keeps the marker count == pages.
    """
    stripped = text.lstrip()
    m = PAGE_MARKER_RE.match(stripped)
    if m:
        return stripped[m.end() :].lstrip("\n")
    return text


def canonical_page_texts(state: DocumentState) -> list[str]:
    """Per-page winning texts for the document, length == ``handle.page_count``.

    The SINGLE source of truth for both the saved ``.md`` body and the manifest
    blobs, so the saved document and ``replay`` are bit-consistent. Each entry is
    the winning page's text selected exactly as :func:`_winning_page_output`
    selects it (passing OCR > native > best attempt > whole-doc split), with any
    pre-existing leading ``## Page N`` header stripped so ``assemble_pages`` adds
    exactly one canonical header per page. ``split_native_pages`` round-trips it.
    """
    whole_doc = _whole_doc_page_texts(state)
    return [
        _strip_leading_page_marker(_winning_page_output(state, page_num, whole_doc).text)
        for page_num in range(1, state.handle.page_count + 1)
    ]


def _base_engine_name(engine: str) -> str:
    """Strip the ``consensus(<engine>)`` wrapper to the underlying engine name.

    The LLM-consensus producer labels its output ``consensus(qwen)`` etc., which
    does not match any ``EngineResult.engine`` key in the model/fingerprint maps.
    Stripping the wrapper lets the fingerprint resolve the real model/prompt for
    the consensus-frozen page instead of recording an empty determinant.
    """
    if engine.startswith("consensus(") and engine.endswith(")"):
        return engine[len("consensus(") : -1]
    return engine


def build_manifest(
    state: DocumentState,
    blobs: BlobStore,
    *,
    dpi: int | None = None,
    fingerprint_inputs: dict[str, tuple[str, str, str | None, str | None]] | None = None,
    saved_body: str | None = None,
) -> Manifest:
    """Freeze a completed ``DocumentState`` into (manifest, cached blobs).

    For each page: select the winning PageOutput, store it in the BlobStore, and
    record a fingerprinted ManifestEntry pointing at it. The rendered-image hash
    is computed only for pages that were actually OCR'd (an engine touched the
    raster); native-text pages don't depend on rasterization.

    ``fingerprint_inputs`` maps an engine name to its RESOLVED run determinants
    ``(model, backend, task, prompt)`` (computed by the orchestrator from the
    live config). When present, the page's ``prompt_hash`` is the contract's
    :func:`run_fingerprint` of those determinants AND ``model_version`` is taken
    from the resolved model — so a model/backend/task/prompt swap invalidates
    the cache, across configurable-model engines AND the consensus producer.
    Without it, the per-engine ``EngineResult.model_version`` is used as before.

    ``saved_body`` is the FINAL ``## Page N`` markdown actually written to disk
    (post strip-phantom-images / figure-embed). When given, each page blob's
    TEXT is taken from splitting that saved body, so ``replay`` reproduces the
    on-disk document bit-for-bit instead of diverging via pre-transform state.
    The fingerprint/engine metadata still comes from the winning PageOutput.
    """
    handle = state.handle
    dpi = dpi if dpi is not None else 200
    fingerprint_inputs = fingerprint_inputs or {}
    saved_pages: list[str] | None = None
    if saved_body is not None:
        saved_pages = split_native_pages(saved_body)
        if len(saved_pages) != handle.page_count:
            logger.warning(
                "manifest: saved body split into %d page(s) but the document has "
                "%d page(s); replay may diverge from the saved .md",
                len(saved_pages),
                handle.page_count,
            )
    manifest = Manifest(
        pdf_filename=handle.filename,
        pdf_file_hash=handle.file_hash,
        page_count=handle.page_count,
        render_dpi=dpi,
        agentic_ladder=state.agentic_ladder if state.agentic_ladder else None,
        agentic_judge_model=getattr(state, "agentic_judge_model", ""),
    )
    # Recover per-page text from a whole-document CLI attempt (page_num=0) so the
    # manifest never freezes empty pages when per-page best_outputs are absent.
    whole_doc = _whole_doc_page_texts(state)
    # Validate the recovered split count against the real page count: a mismatch
    # (the '---'-only legacy case, or dropped/merged markers) would silently
    # freeze trailing pages empty or drop extras. Log loudly rather than corrupt.
    if whole_doc is not None and len(whole_doc.texts) != handle.page_count:
        logger.warning(
            "manifest: whole-doc split yielded %d page(s) but the document has "
            "%d page(s) (engine=%s); trailing pages may be empty or extras dropped",
            len(whole_doc.texts),
            handle.page_count,
            whole_doc.engine,
        )
    # Model version per engine, so a model swap/drift invalidates the fingerprint.
    model_versions = {r.engine: r.model_version for r in state.engine_runs if r.model_version}
    for page_num in range(1, handle.page_count + 1):
        page = _winning_page_output(state, page_num, whole_doc)
        # Freeze the EXACT on-disk page text when the saved body is supplied, so
        # replay reproduces the saved .md (post-transform); fall back to the
        # winning page text otherwise. Engine/fingerprint metadata is unchanged.
        if saved_pages is not None and page_num - 1 < len(saved_pages):
            from dataclasses import replace as _dc_replace

            page = _dc_replace(page, text=saved_pages[page_num - 1])
        blob_ref = blobs.put_page(page)
        image_hash = ""
        if page.engine and page.engine != "native":
            image_hash = compute_image_hash(handle, page_num, dpi)

        # Resolve the run determinants for this page's engine (consensus-aware).
        base_engine = _base_engine_name(page.engine)
        determinants = fingerprint_inputs.get(base_engine) or fingerprint_inputs.get(page.engine)
        prompt_hash = ""
        if determinants is not None:
            model, backend, task, prompt = determinants
            model_version = (
                model or model_versions.get(base_engine) or model_versions.get(page.engine, "")
            )
            prompt_hash = run_fingerprint(model_version, backend, task, prompt)
        else:
            model_version = model_versions.get(base_engine) or model_versions.get(page.engine, "")

        fp = PageFingerprint(
            pdf_file_hash=handle.file_hash,
            page_num=page_num,
            render_dpi=dpi,
            engine=page.engine,
            model_version=model_version,
            image_hash=image_hash,
            prompt_hash=prompt_hash,
        )
        _judge_model = getattr(state, "agentic_judge_model", "")
        journal = [
            {
                "engine": a.engine,
                "provider_id": getattr(a, "provider_id", ""),
                "model": getattr(a, "provider_model", ""),
                "backend": getattr(a, "provider_backend", ""),
                "cost_usd": a.cost_usd,
                "accepted": a.audit_passed,
                "confidence": a.confidence,
                "failure_mode": a.failure_mode.value,
                "reason": getattr(a, "skip_reason", "") or a.failure_mode.value,
                "judge_model": _judge_model,
            }
            for a in state.pages[page_num].attempts
        ]
        manifest.entries[page_num] = ManifestEntry(
            page_num=page_num, blob_ref=blob_ref, fingerprint=fp, journal=journal
        )
    return manifest


def replay(manifest: Manifest, blobs: BlobStore) -> str:
    """Reconstruct the document markdown purely from cached blobs.

    Invokes NO engine. Raises KeyError if a referenced blob is missing (a broken
    or partially-deleted cache), which is preferable to silently emitting a
    document with holes.

    Joined with the contract's ``assemble_pages`` (``## Page N`` headers), the
    SAME assembler socr uses for the saved ``.md`` body, so replay output is
    canonical and consistent with the document written to disk.
    """
    texts: list[str] = []
    for page_num in range(1, manifest.page_count + 1):
        entry = manifest.entries.get(page_num)
        if entry is None:
            raise KeyError(f"manifest has no entry for page {page_num}")
        page = blobs.get_page(entry.blob_ref)
        texts.append(page.text)
    return assemble_pages(texts)


def stale_pages(manifest: Manifest, blobs: BlobStore) -> list[int]:
    """Pages whose referenced blob is missing from the cache (need re-OCR)."""
    return [
        pn
        for pn in range(1, manifest.page_count + 1)
        if pn not in manifest.entries or not blobs.has(manifest.entries[pn].blob_ref)
    ]
