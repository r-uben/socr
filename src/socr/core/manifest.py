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
import math
import re
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path

from ocr_output_contract import (
    PAGE_MARKER_RE,
    assemble_pages,
    run_fingerprint,
    split_native_pages,
)

from socr.core.cache import BlobStore
from socr.core.document import DocumentHandle
from socr.core.result import (
    REJECTION_AMBIGUOUS_DEFERRED,
    REJECTION_JUDGE_ONLY,
    FailureMode,
    PageOutput,
    PageStatus,
)
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
ASSEMBLY_VERSION = "3"

# VI-B1: exclusive per-page stage keys. Nested children (extract under route
# for OCR; ladder / adjudication under tables) are subtracted from the parent.
# ``timings_s.total`` is the independent page wall, not this sum.
PAGE_TIMING_EXCLUSIVE_KEYS: tuple[str, ...] = (
    "route",
    "extract",
    "tables",
    "ladder",
    "adjudication",
    "figures",
    "equations",
    "flush",
)


def coerce_page_timings(raw) -> dict[str, float]:
    """Keep only finite non-negative numeric exclusive keys (and ``total``)."""
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for key in (*PAGE_TIMING_EXCLUSIVE_KEYS, "total"):
        value = raw.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0:
            out[key] = float(value)
    return out


def exclusive_timings_sum(timings: dict[str, float] | None) -> float:
    """Sum of exclusive stage keys. Does not read ``total`` (independent wall)."""
    if not timings:
        return 0.0
    return sum(float(timings.get(key, 0.0) or 0.0) for key in PAGE_TIMING_EXCLUSIVE_KEYS)


def rollup_page_timings(state: DocumentState) -> dict[str, float]:
    """Sum per-page ``timings_s`` into one document dict."""
    acc = {key: 0.0 for key in PAGE_TIMING_EXCLUSIVE_KEYS}
    acc["total"] = 0.0
    for ps in state.pages.values():
        timings = coerce_page_timings(getattr(ps, "timings_s", None) or {})
        if not timings:
            continue
        for key in PAGE_TIMING_EXCLUSIVE_KEYS:
            acc[key] += float(timings.get(key, 0.0) or 0.0)
        acc["total"] += float(timings.get("total", 0.0) or 0.0)
    return acc


# Legacy page separator. socr now assembles bodies and replays with the
# contract's ``assemble_pages`` (``## Page N`` headers); this constant is kept
# only for backward-compatible imports and is no longer used to join pages.
PAGE_SEPARATOR = "\n\n---\n\n"

# Matches the canonical failure marker `[page N failed: no usable OCR output]`.
_PAGE_FAILED_RE = re.compile(r"^\[page \d+ failed: no usable OCR output\]$")
# TR-3 and the other page floors use a closed, single-line marker.  Keep the
# reason generic because rotated-text and table-emission floors have their own
# marker text, but do not let a malformed/truncated first line pass as a page
# failure.
_PAGE_FAILED_ANY_RE = re.compile(r"^\[page \d+ failed:[^\]\r\n]+\]$")
_PAGE_FAILED_IMAGE_RE = re.compile(r"^!\[[^\]\r\n]*\]\([^\)\r\n]+\)$")
_TABLE_EMISSION_FAILED_RE = re.compile(
    r"^\[page (?P<page>\d+) failed: invalid table emission — (?P<defect>[^\]]+)\]$"
)


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


#: Every span socr itself authors INSIDE a page body: the fail-closed page
#: markers and the ``[socr: …]`` notes a sanitizer leaves where it removed
#: something. These are socr's own prose about what it did, not document
#: content, which is what makes them the one exception to "no post-verdict step
#: adds content": a subtractive step that replaces an invented image reference
#: with a note has removed content and added a receipt for the removal.
#:
#: Defined here, beside ``is_page_failed_marker``, so there is ONE recognizer.
#: A second copy would drift, and a drifted copy is how a real addition gets
#: waved through as "just a marker".
SOCR_MARKER_RE = re.compile(r"\[socr:[^\]\r\n]*\]|\[page \d+ failed:[^\]\r\n]+\]")


def socr_marker(note: str) -> str:
    """Build one socr marker, guaranteed to match :data:`SOCR_MARKER_RE`.

    Cold review round 4. The recognizer alone was a test oracle: it changed what
    a test compared while every emitter still assembled its own prose, so
    "socr's markers are not content" rested on nothing enforceable. Emitters
    build their marker HERE, which makes the exception a contract -- the
    recognizer cannot miss a marker, because the builder cannot emit one it
    would miss.

    The note is flattened to a single line and stripped of the closing bracket
    that would end the span early. That is what keeps the guarantee true for
    free-form text (a source path, a reason), so a marker may carry detail
    without the detail escaping the marker.
    """
    flattened = " ".join(str(note or "").split()).replace("]", ")")
    return f"[socr: {flattened}]" if flattened else "[socr: ]"


def is_page_failed_marker(text: str) -> bool:
    """True if a page's canonical text is a failure marker (not real content).

    Matches both the original ``[page N failed: no usable OCR output]`` marker
    AND the TR-3 D3 fail-closed variant ``[page N failed: unverifiable table …]``
    (which may be followed by a PNG image ref block).

    GH-371: Returns False if the text contains substantial prose beyond the marker
    and optional image reference, indicating a regional splice with preserved content.
    """
    stripped = text.strip()
    lines = stripped.splitlines()
    if not lines:
        return False

    # A whole-page body is exactly one closed marker, optionally followed by
    # one Markdown image block.  Blank lines are formatting around the optional
    # block; any other text means the marker is a regional splice or malformed
    # output and therefore represents usable page content (or an unknown body).
    marker = lines[0].strip()
    if not _PAGE_FAILED_ANY_RE.fullmatch(marker):
        return False
    content_lines = [line.strip() for line in lines[1:] if line.strip()]
    return len(content_lines) <= 1 and (
        not content_lines or _PAGE_FAILED_IMAGE_RE.fullmatch(content_lines[0]) is not None
    )


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
    disposition: PageDisposition | None = None

    def to_dict(self) -> dict:
        d = {
            "page_num": self.page_num,
            "blob_ref": self.blob_ref,
            "fingerprint": asdict(self.fingerprint),
            "journal": self.journal,
        }
        if self.disposition is not None:
            d["disposition"] = self.disposition.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ManifestEntry:
        disp_dict = d.get("disposition")
        disposition = PageDisposition.from_dict(disp_dict) if disp_dict is not None else None
        return cls(
            page_num=d["page_num"],
            blob_ref=d["blob_ref"],
            fingerprint=PageFingerprint(**d["fingerprint"]),
            journal=d.get("journal", []),
            disposition=disposition,
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
    # VI-B1: document rollup of per-page exclusive stage timings. None when no
    # page recorded ``timings_s`` (pre-B1 sidecar / non-agentic run).
    timings_s: dict[str, float] | None = None

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
        if self.timings_s:
            d["timings_s"] = self.timings_s
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Manifest:
        raw_timings = d.get("timings_s")
        timings_s = coerce_page_timings(raw_timings) or None if raw_timings is not None else None
        return cls(
            pdf_filename=d["pdf_filename"],
            pdf_file_hash=d["pdf_file_hash"],
            page_count=d["page_count"],
            render_dpi=d["render_dpi"],
            schema_version=d.get("schema_version", MANIFEST_SCHEMA_VERSION),
            entries={int(k): ManifestEntry.from_dict(v) for k, v in d.get("entries", {}).items()},
            agentic_ladder=d.get("agentic_ladder"),
            agentic_judge_model=d.get("agentic_judge_model", ""),
            timings_s=timings_s,
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


def kept_table_grid_defect(text: str) -> str:
    """The structural gate's own predicate, run for SURFACING, never for gating.

    #259 round 3. Round 2's blind spot was that a candidate which is both
    structurally defective and an ambiguous deferral is never gate-inspected --
    ``_apply_structural_gate`` returns a rejecting inner decision unchanged. The
    owner's ruling settles what to DO about that: a ragged grid is a thing to
    flag, not a reason to hand the page back to an ungridded native. So the
    predicate runs here, and its answer decorates the page instead of deciding
    it. String-only (no geometry, no page object), like the post-route recheck
    in ``orchestrator._phase_agentic``.
    """
    try:
        from socr.tables.structure_check import table_output_defect

        return table_output_defect(text, None, None) or ""
    except Exception:  # surfacing must never be able to break assembly
        return ""


def kept_table_flag_note(state, page_num: int, text: str) -> str:
    """The visible, greppable flag carried on a kept-but-flagged table.

    The owner's ruling is "keep the model version, carry the flag visibly".
    Status fields and audit JSON are not visible to someone reading the ``.md``,
    and this corpus is read by humans who cite from it -- so the doubt goes in
    the body, naming the disputed values, and says what to do about it.
    """
    reasons: list[str] = []
    for event in getattr(state, "events", []) or []:
        if getattr(event, "page_num", None) != page_num:
            continue
        if getattr(event, "kind", "") != "table_value_drift_unadjudicated":
            continue
        from socr.tables.native_verifier import describe_drift

        drifted = (getattr(event, "data", None) or {}).get("drifted_rows") or []
        described = describe_drift(drifted) if drifted else ""
        reasons.append(f"value drift ({described})" if described else "value drift")
    defect = kept_table_grid_defect(text)
    if defect:
        reasons.append(f"grid shape: {defect}")
    if not reasons:
        return ""
    return (
        f"[page {page_num}: flagged table kept — "
        + "; ".join(reasons)
        + " — verify against the source before citing]"
    )


def flagged_model_page_output(p) -> PageOutput | None:
    """#259: the model output a flagged table page must ship instead of native.

    ``_winning_page_output`` returns ``best_output`` only while
    ``audit_passed`` is True, and the agentic router sets
    ``att.output.audit_passed = att.accepted`` for every attempt. So on a
    born-digital table page where the ladder accepted nothing, a model output
    that is fully PRESENT — real text, no floor, no hallucination verdict —
    fell through to the native branch and was replaced wholesale by the native
    reading.

    That substitution collapses two different outcomes: *the model produced
    nothing* and *the model produced something a check flagged*. Only the first
    is a reason to fall back. The second hands the page to native precisely
    when native is least able to arbitrate: the flag on these pages is a
    native-table-distrust flag, i.e. the pipeline's own record that the native
    text does not represent this table's grid. On the reference page
    ``table_not_scorable`` had already fired — the native layer parsed four rows
    that do not form a grid — at the moment native was chosen as the
    replacement.

    Returns the output to keep, or ``None`` to leave selection unchanged. The
    caller demotes it; this predicate never mutates ``audit_passed``, which is
    the winner-SELECTION flag and not a page-quality flag (flipping it discards
    the page's content, the #252 round-1 defect).

    Deliberately NOT widened beyond a table defect. A page demoted only by
    ``needs_ocr_enhancement`` (deficient native prose) keeps today's behaviour:
    there the model output was rejected on its own merits and native is a
    legitimate reading, so preferring a rejected model page could ship garbage
    over clean text.
    """
    if not (p.is_born_digital and p.native_text):
        return None
    # The two fail-closed floors are hard verdicts, not flags, and both ship an
    # explicit failure marker rather than native — so neither is the
    # substitution this fixes, and neither may be bypassed here.
    if getattr(p, "scanned_table_evidence_failed", False):
        return None
    if p.native_table_structure_failed and (
        getattr(p, "native_table_unverifiable", False)
        or getattr(p, "native_table_header_unattributed", False)
    ):
        return None
    if not (
        p.native_table_structure_failed
        or getattr(p, "native_table_unverifiable", False)
        or getattr(p, "native_table_structure_defective", False)
        or getattr(p, "native_table_header_unattributed", False)
    ):
        return None

    bo = p.best_output
    if bo is None or bo.audit_passed:
        return None
    if (bo.engine or "").startswith("native"):
        return None
    # ALLOWLIST, deliberately not a denylist of bad dispositions. A hard
    # rejection mutates nothing on the PageOutput -- the verifier's CERTAIN_FAIL
    # (``agentic.py``, ``vr.hard_fail``) and the winner-side structural gate both
    # return ``accept=False`` and leave status SUCCESS / failure_mode NONE, and
    # the orchestrator stores only ``accepted`` into ``audit_passed``. So a
    # "not ERROR and not HALLUCINATION" test cannot tell a table the value guard
    # positively proved wrong from the reference page's ambiguous
    # "paired/spanning headers — deferring to VLM". Keeping the former would
    # replace native with a table socr knows is corrupt. Unless the refusal was
    # positively identified as the soft kind, behave exactly as before.
    # GH-322 / GH-326: the allowlist widens from one soft rejection to both, but
    # only behind the presence gate.
    #
    # It was narrowed to AMBIGUOUS_DEFERRED alone because a hard rejection mutates
    # nothing observable on the PageOutput, so "not ERROR and not HALLUCINATION"
    # could not tell a table the value guard positively disproved from one it
    # merely deferred on. `rejection_class` was the only available proxy for "is
    # this model reading trustworthy".
    #
    # It is no longer the only one. The presence gate answers that question
    # directly from the page's own numbers, and measurement says it should be
    # trusted over the alternative: native was the LEAST accurate of three
    # readings (8/13 rows exact against a free local model's 12/13), so keeping a
    # correct model table out on the strength of a rejection label -- while
    # substituting a reading that loses row labels -- is the failure GH-259 named.
    #
    # JUDGE_ONLY is admitted because a judge refusal is a model's opinion, not
    # evidence; the page's own numbers outrank it. Everything else still falls
    # back, and an EMPTY rejection_class stays out: it is indistinguishable from
    # "never judged", and absence of evidence is not evidence.
    rejection = getattr(bo, "rejection_class", "")
    if rejection not in D3_SUPERSEDING_REJECTIONS:
        return None
    if rejection != REJECTION_AMBIGUOUS_DEFERRED:
        from socr.tables.escalation_canary import presence_verdict_from_text

        verdict = presence_verdict_from_text(
            p.native_text or "",
            bo.text or "",
            encoding_suspect=bool(
                getattr(p, "has_encoding_hygiene_suspect", False)
                or getattr(p, "has_corrupt_math", False)
            ),
        )
        # Only invention blocks. UNVERIFIABLE does not: a page whose text layer is
        # too damaged to adjudicate is exactly the page where native is least able
        # to arbitrate, which is this predicate's whole premise.
        if verdict.blocks_success:
            return None
    # "The model produced nothing" — the case that must still fall back.
    if not (bo.text or "").strip():
        return None
    # A positively-rejected reading, not a flagged one: an ERROR output, or one
    # a floor already overwrote with a failure marker, is not evidence of
    # anything and must not displace the native reading.
    if bo.status is PageStatus.ERROR or bo.failure_mode is FailureMode.HALLUCINATION:
        return None
    # The comparison being fixed is between two readings of a TABLE. If the
    # model emitted no grid at all on a table page it did not produce the thing
    # under comparison -- that is the "produced nothing" case for this
    # predicate's purposes, and native remains the only table reading there is.
    # Note this is a property of the model's own output alone; native structure
    # is never consulted as ground truth, which is the root error #259 names.
    from socr.tables.reconcile import has_authored_table_grid

    if not has_authored_table_grid(bo.text):
        return None
    return bo


#: #262: the dispositions on which an attempt may supersede the D3 fail-closed
#: floor. An ALLOWLIST of positively-identified SOFT refusals -- the reading was
#: refused by a judge, not refuted by a deterministic gate. Anything else,
#: including an empty ``rejection_class``, keeps the failed-table marker.
D3_SUPERSEDING_REJECTIONS: frozenset[str] = frozenset(
    {REJECTION_AMBIGUOUS_DEFERRED, REJECTION_JUDGE_ONLY}
)


def d3_superseded_note(page_num: int) -> str:
    """The in-body flag a superseded D3 page carries, in the #259 note's shape.

    The page ships a table every rung of the ladder refused, in place of a
    marker that said so in the document itself. Status and audit surfaces carry
    that, but a reader holding only the ``.md`` sees neither, so the flag has to
    be in the body as well. Phrased so it can never match
    ``is_page_failed_marker`` -- this page did not fail, it shipped flagged.
    """
    return (
        f"[page {page_num}: unverifiable table — kept a model reading over the "
        "failed-table marker; the native table region failed verification and no "
        "OCR rung was accepted — verify against the page image before citing]"
    )


def d3_floor_kept_model_output(p) -> PageOutput | None:
    """#262: the model attempt that must ship instead of the D3 marker.

    The D3 conjunction is a verdict on the native lane. When a model attempt
    authored a grid, shipping the failed-table marker would discard the page's
    model reading, prose, and equations along with the native table.

    Returns the output to keep, or ``None`` to leave the floor firing. The
    caller demotes a COPY; this predicate never mutates anything, and in
    particular never touches ``audit_passed``, which is the winner-SELECTION
    flag and not a page-quality flag (flipping it discards the page's content —
    the #252 round-1 defect).

    ALLOWLIST, never a denylist. Round 1 of this fix reasoned that because the
    alternative here is a marker with zero content rather than #259's complete
    native reading, "we do not know why this rung was refused" was good enough
    to keep. It is not, and the counterexample is exact: the verifier's
    CERTAIN_FAIL path (``agentic.py``, ``vr.hard_fail``) returns
    ``accept=False`` and mutates NOTHING on the ``PageOutput`` -- status stays
    SUCCESS, ``failure_mode`` stays NONE -- so a grid the value guard
    POSITIVELY REFUTED for a numeric-multiset or label-binding mismatch passed
    every one of those rules and would have shipped over a fail-closed floor.
    A denylist of bad dispositions cannot see a disposition that was never
    written; only an allowlist of good ones is fail-safe. An empty
    ``rejection_class`` means "socr cannot say why this was refused" and keeps
    the marker.

    The allowlist is the two dispositions on which socr can positively say the
    refusal was SOFT -- no deterministic gate refuted the reading, a judge did:
    ``REJECTION_AMBIGUOUS_DEFERRED`` (#259: the verifier reached AMBIGUOUS and
    deferred) and ``REJECTION_JUDGE_ONLY`` (#262: the verifier found nothing to
    refute and the inner judge alone refused). Both are written before the
    structural gate runs, so a gate rejection is never mistaken for either.
    ``flagged_model_page_output`` deliberately keeps the narrower one-value
    allowlist: its fallback is a real native reading, so it can afford to.

    Selecting among qualifying candidates is then a choice between readings
    socr has NO evidence to rank -- ladder position is not quality, and the
    most escalated rung is not the best one. So ``best_output`` wins when it
    qualifies: ``_best_effort`` already picked it as the most trustworthy
    attempt, which is a judgement rather than an ordering. Only when the winner
    was cleared or does not qualify does ladder order break the tie, and the
    audit event records that the choice was unranked.

    Also excluded, on either list: an ERROR output, a HALLUCINATION verdict,
    empty text, a native-lane reading (native is precisely what the floor
    distrusts), and text that is itself a failure marker (the GH-90 floor
    overwrites ``best_output.text`` in place).
    """
    if not (
        p.is_born_digital
        and p.native_text
        and p.native_table_structure_failed
        and (
            getattr(p, "native_table_unverifiable", False)
            or getattr(p, "native_table_header_unattributed", False)
        )
        and bool(p.attempts)
    ):
        return None
    # GH-90's scanned floor is a different lane and is not reachable from here
    # (it requires ``not is_born_digital``), but a page carrying its flag must
    # never be rescued by this path either.
    if getattr(p, "scanned_table_evidence_failed", False):
        return None
    # #263 owns the disposition of a page whose native text is shredded. D3 is
    # earlier in winner selection, so declining here preserves the existing D3
    # floor instead of letting #262 silently choose a model on #263's behalf.
    if getattr(p, "native_rotated_text_shredded", False):
        return None

    from socr.tables.reconcile import has_strict_table_grid

    def _qualifies(out: PageOutput) -> bool:
        if out is None:
            return False
        if getattr(out, "rejection_class", "") not in D3_SUPERSEDING_REJECTIONS:
            return False
        # ``chart_asset`` ships native_text plus a PNG reference. Relabelling
        # the text does not make it independent evidence capable of overriding
        # the D3 floor that distrusts that same native text (#265).
        if (out.engine or "").startswith(_NATIVE_TEXT_LANES):
            return False
        text = (out.text or "").strip()
        if not text or is_page_failed_marker(text):
            return False
        if out.status is PageStatus.ERROR or out.failure_mode is FailureMode.HALLUCINATION:
            return False
        return has_strict_table_grid(text)

    if _qualifies(p.best_output):
        return p.best_output
    for out in reversed(list(p.attempts)):
        if _qualifies(out):
            return out
    return None


#: #263 round 2: the engines whose winning text IS (or embeds) the page's
#: ``PageState.native_text``. ``native`` ships it directly; ``chart_asset``
#: (PP-7) ships ``native_text`` with a whole-page PNG ref appended. A flag that
#: says "this page's native text is not a reading of the page" therefore
#: contradicts a passing winner from EITHER lane, and keying the contradiction
#: guard below on ``engine.startswith("native")`` alone let the chart lane
#: return before the fail-closed floor could run (the #265 review finding).
#: Every other engine label is a real model output and is unaffected.
_NATIVE_TEXT_LANES = ("native", "chart_asset")


def _grid_shaped_attempt(out: PageOutput | None) -> bool:
    """Whether ``out`` is a non-native reading that authored a table grid,
    with NO judge/rejection verdict applied either way.

    TICKET-A1b (#634): split out of ``_grid_authored_attempt`` so the row-
    corroboration fallback below can score every STRUCTURALLY plausible
    candidate, including one the value guard hard-rejected on a stale
    multiset check -- second-guessing exactly that guard is why the fallback
    exists (a flattened/relabelled row is multiset-identical to a correct one
    and can be hard-rejected today even though the row-level ORDERED check
    below would clear it). ``_grid_authored_attempt`` keeps the judge gate for
    S1 case (i)'s own pool; both pools call this same structural half so they
    cannot drift apart on what "authored a grid" means in the first place.

    A native-engine reading is never a candidate -- C1 forbids native from
    authoring the GRID on a structure-class page, full stop; native's PROSE
    ships separately, untouched, via the last-resort WARNING branch below.

    GH-268 centralizes "authored a grid" in ``has_strict_table_grid``. S1
    must use the same structural contract as the earlier D3 and #259 branches;
    otherwise an output one branch rejects can be selected by this later one.
    """
    if out is None:
        return False
    if (out.engine or "").startswith(_NATIVE_TEXT_LANES):
        return False
    text = (out.text or "").strip()
    if not text or is_page_failed_marker(text):
        return False
    if out.status is PageStatus.ERROR or out.failure_mode is FailureMode.HALLUCINATION:
        return False

    from socr.tables.reconcile import has_strict_table_grid

    return has_strict_table_grid(text)


def _grid_reading_attempt(out: PageOutput | None) -> bool:
    """Whether ``out`` is a non-native reading that authored ANY table
    reading, ragged body allowed -- the row-corroboration fallback's own
    candidate pool.

    TICKET-A1b (#634), measured against the real ECB fixture cache: a
    statistical table's own printed layout routinely stacks more than one
    spanning-header-shaped row above the numeric body (a units line, a
    grouped-column label row, a bare column-index legend row -- see
    ``row_corroboration.is_column_index_row``'s own docstring for the same
    convention), so ``has_strict_table_grid``'s uniform-body-width
    requirement rejects the ENTIRE block before row corroboration ever gets
    to look at it -- measured 0/6 real qwen/gemini fixture candidates
    passing ``has_strict_table_grid`` where the numeric body rows
    underneath corroborate cleanly. ``has_authored_table_grid`` (#259's own
    ragged-body-tolerant predicate) is used instead, matching
    ``row_corroboration``'s own module contract: ``table_blocks`` is
    deliberately not ``binding.parse_grid`` for the identical reason ("gives
    up on the WHOLE block the first time a row does not [match column
    counts]"). Deliberately a SEPARATE predicate from ``_grid_shaped_attempt``
    rather than a parameter on it, so S1 case (i)'s own strict pool
    (``_grid_authored_attempt``) and its existing floor-fixture tests are
    untouched by this widening.
    """
    if out is None:
        return False
    if (out.engine or "").startswith(_NATIVE_TEXT_LANES):
        return False
    text = (out.text or "").strip()
    if not text or is_page_failed_marker(text):
        return False
    if out.status is PageStatus.ERROR or out.failure_mode is FailureMode.HALLUCINATION:
        return False

    from socr.tables.reconcile import has_authored_table_grid

    return has_authored_table_grid(text)


def _grid_authored_attempt(out: PageOutput | None) -> bool:
    """Whether ``out`` is a model-authored reading that authored a table grid
    and was never POSITIVELY hard-rejected by the still-live value guard /
    structural gate.

    S1 ships no binder (R2: "selection ships first WITHOUT the binder"). The
    multiset-based value guard that produces today's hard rejects is not
    deleted until S2's verifier rewiring (C4: deleting it early, without the
    two-directional replacement, "is a regression") -- so until S2 lands, a
    CERTAIN_FAIL is still the strongest signal this codebase has that a grid
    is wrong, and S1 must not override it. Gated on the SAME allowlist
    ``flagged_model_page_output`` (#259) already uses --
    ``REJECTION_AMBIGUOUS_DEFERRED`` -- for the same fail-safe reason theirs is
    an allowlist and not a denylist: a hard reject (``vr.hard_fail``, the
    winner-side structural gate) mutates NOTHING on the ``PageOutput`` --
    status stays SUCCESS, ``failure_mode`` stays NONE, ``rejection_class``
    stays "" -- so an empty ``rejection_class`` is indistinguishable from "no
    judge ever ran" and must default to distrust, not to "authored a grid,
    ship it."
    """
    if not _grid_shaped_attempt(out):
        return False
    return bool(
        out.audit_passed or getattr(out, "rejection_class", "") == REJECTION_AMBIGUOUS_DEFERRED
    )


def _reaches_structure_class_branch(p) -> bool:
    """Whether ``_winning_page_output`` would actually reach the S1
    structure-class branch for this page, mirroring EVERY precondition that
    branch sits behind, in the SAME order, rather than a subset of them.

    BLOCKING 2 on #269: the prior version reproduced only the early-return
    short-circuit and then unconditionally returned True -- so a page whose
    ``best_output`` was never a clean non-native pass (e.g. any AUDIT_FAILED
    or born-digital-with-no-native-text page) satisfied this gate regardless
    of ``is_structure_class()``, born-digital status, or whether the D3/#263
    fail-closed floors or #259's flagged-model substitution actually fired
    first. Concrete fallout: a PROSE page with a refused model GFM landed in
    ``structure_class_model_pages`` and flipped the document to
    AUDIT_FAILED while ``_winning_page_output`` shipped native SUCCESS and
    never entered the S1 branch at all; a D3-floor page's bucket claimed a
    model grid shipped when the real winner was the fail-closed ERROR
    marker; a #259 page's bucket described the undemoted attempt rather
    than the ``replace()`` copy that actually ships.

    Fixed by walking the SAME branches ``_winning_page_output`` walks, in
    the same order, so a caller outside that function (the document-level
    buckets in ``_phase_assemble``) cannot disagree with what the manifest
    actually ships. ``_winning_page_output`` itself now calls this function
    for its own S1 entry-point check instead of duplicating it inline, so
    the two cannot drift apart again.
    """
    if p.best_output and p.best_output.audit_passed:
        winning_engine = p.best_output.engine or ""
        native_distrusted = winning_engine.startswith("native") and (
            p.is_structure_class()
            or getattr(p, "native_table_unverifiable", False)
            or getattr(p, "native_table_structure_defective", False)
            or getattr(p, "native_table_header_unattributed", False)
        )
        native_text_shredded = winning_engine.startswith(_NATIVE_TEXT_LANES) and getattr(
            p, "native_rotated_text_shredded", False
        )
        # MAJOR 7(b): resume collapses ``p.attempts`` to the single frozen
        # winner (``_restore_terminal_page_state``), so a resumed run's own
        # attempt list can no longer prove a non-native rung authored a grid
        # on the run that actually produced this winner. The persisted flag
        # (set at flush time, restored on resume -- see ``PageState``) is
        # this predicate's only source of truth for that case; a live run
        # never sets it, so it is inert everywhere else.
        if not (
            native_distrusted
            or native_text_shredded
            or getattr(p, "structure_class_model_kept_on_resume", False)
        ):
            return False
    # Everything below mirrors, in order, the preconditions
    # ``_winning_page_output`` itself checks before reaching the S1 branch:
    # born-digital native text (the branch's own containing ``if``), the
    # TR-3 D3 fail-closed floor, the #263 rotated-shredded floor, and #259's
    # flagged-model substitution. Any one of these means the S1 branch is
    # never reached for this page -- a different, earlier return ships.
    if not (p.is_born_digital and p.native_text):
        return False
    if (
        p.native_table_structure_failed
        and (
            getattr(p, "native_table_unverifiable", False)
            or getattr(p, "native_table_header_unattributed", False)
        )
        and bool(p.attempts)
    ):
        return False
    if getattr(p, "native_rotated_text_shredded", False):
        return False
    if flagged_model_page_output(p) is not None:
        return False
    if not p.is_structure_class():
        return False
    return any(not (a.engine or "").startswith("native") for a in p.attempts)


def _truncated_grid_reading_ids(p) -> frozenset[int]:
    """TICKET-A2 (#645): ``id()`` of every ``_grid_reading_attempt`` candidate
    on this page that ``structure_check.table_truncated`` flags, scored
    across ALL of them -- best_output and every attempt, deduplicated -- not
    scoped to whichever narrower pool (strict or corroboration) happens to be
    asking.

    That wider universe is load-bearing, not cosmetic: the ticket's own live
    fixture (ECB economic bulletin p2, 2026-09-07) has the S1 STRICT pool
    holding ONLY the truncated qwen candidate (14/417 words, ends mid-flag)
    while the complete qwen candidate (414/417) only ever clears the WIDER
    ``_grid_reading_attempt`` pool the corroboration fallback scores against.
    Comparing truncation only within the strict pool would see a single-member
    pool and have nothing to compare against, so the truncated one would ship
    unchallenged -- scoring against the union of both pools' membership is
    what lets ``_strict_grid_authored_pool`` see that a complete reading
    exists at all and empty itself in favour of it.

    Returns an empty set when every candidate truncates (a truncated reading
    is still the only evidence there is -- TICKET-A2's own "if it is the only
    candidate, it still ships, flagged" clause) or when fewer than two
    candidates exist to compare.
    """
    words = getattr(p, "native_words", None) or []
    seen: set[int] = set()
    universe: list[PageOutput] = []
    for out in [p.best_output, *p.attempts]:
        if out is None or id(out) in seen or not _grid_reading_attempt(out):
            continue
        seen.add(id(out))
        universe.append(out)
    if len(universe) < 2:
        return frozenset()

    from socr.tables.structure_check import table_truncated

    flags = {id(out): table_truncated(out.text or "", words) for out in universe}
    if not any(flags.values()) or all(flags.values()):
        return frozenset()
    return frozenset(oid for oid, truncated in flags.items() if truncated)


def structure_class_truncated_engines(p) -> tuple[str, ...]:
    """TICKET-A2 (#645): engines dropped as truncated for this page's S1
    branch, in candidate order, for the caller's ``candidate_truncated``
    audit event.

    Mirrors ``structure_class_grid_corroboration``'s own gate
    (``_reaches_structure_class_branch``) so the two functions describe the
    same page consistently. Empty when nothing was dropped -- including the
    "every candidate truncates" case, where TICKET-A2 says nothing is dropped
    by design.
    """
    if not _reaches_structure_class_branch(p):
        return ()
    truncated_ids = _truncated_grid_reading_ids(p)
    if not truncated_ids:
        return ()
    seen: set[int] = set()
    engines: list[str] = []
    for out in [p.best_output, *p.attempts]:
        if out is None or id(out) in seen or not _grid_reading_attempt(out):
            continue
        seen.add(id(out))
        if id(out) in truncated_ids:
            engines.append(out.engine or "")
    return tuple(engines)


def _truncated_candidate_events(page_num: int, truncated_engines: tuple[str, ...]) -> list:
    """TICKET-A2 (#645): one ``candidate_truncated`` ``AuditEvent`` per
    dropped engine.

    Pulled out of ``_select_page_output_tagged`` as a comprehension-built
    list rather than an in-place ``for`` loop: R7 (test_r7_winner_kind_tags.py)
    asserts that cascade function's body is loop-free by AST inspection, since
    a loop there could break the "exactly one ending per page" guarantee the
    ``SelectionProvenance`` tag depends on. This helper's own loop is outside
    that function entirely, so the guarantee is untouched.
    """
    from socr.core.audit_log import AuditEvent

    return [
        AuditEvent(
            page_num=page_num,
            kind="candidate_truncated",
            engine=truncated_engine,
            detail=(
                "TICKET-A2 (#645): dropped as a grid-winner "
                "candidate -- ends mid-emission (a final row "
                "breaking the block's own style, or a numeric "
                "row-count shortfall past A1b's own row-count "
                "allowance) and another candidate on this page "
                "is not truncated"
            ),
        )
        for truncated_engine in truncated_engines
    ]


def _strict_grid_authored_pool(p) -> list[PageOutput]:
    """S1 case (i)'s own candidate pool: every judge-cleared grid-authored
    attempt, ``best_output`` first.

    Factored out of ``structure_class_grid_winner`` (TICKET-A1b, #634) so it
    and the corroboration fallback's own gate agree, byte-for-byte, on when
    this pool is non-empty -- the fallback must never fire when case (i)
    already has something to ship.

    TICKET-A2 (#645): a candidate ``_truncated_grid_reading_ids`` flags is
    dropped from this pool whenever ANY candidate for the page (not only this
    pool) is not truncated -- a truncated reading must never beat a complete
    one, and must not block the corroboration fallback from reaching a
    complete one either. If dropping truncated candidates would empty an
    otherwise non-empty pool, it empties: that is the whole point (the
    ticket's own live fixture -- see ``_truncated_grid_reading_ids``'s
    docstring), and lets ``_row_corroborated_grid_winner`` run instead of a
    truncated candidate shipping via case (i).
    """
    if _grid_authored_attempt(p.best_output):
        pool = [p.best_output]
    else:
        pool = [out for out in p.attempts if _grid_authored_attempt(out)]
    truncated_ids = _truncated_grid_reading_ids(p)
    if truncated_ids:
        pool = [out for out in pool if id(out) not in truncated_ids]
    return pool


def _union_bbox(
    bboxes: list[tuple[float, float, float, float]],
) -> tuple[float, float, float, float] | None:
    """Smallest box covering every bbox in *bboxes*, or ``None`` if empty.

    TICKET-A1b (#634) disclosed simplification: a page can carry more than
    one ``detected_table_bboxes`` entry, and ``corroborate_rows`` takes a
    single region. Scoring against the union rather than splitting per-table
    is deliberate: ``corroborate_rows`` already scores every markdown table
    BLOCK it finds against one region, so a multi-table page is handled at
    the block level already -- the union only widens the native-word filter,
    it does not merge table blocks.
    """
    if not bboxes:
        return None
    return (
        min(b[0] for b in bboxes),
        min(b[1] for b in bboxes),
        max(b[2] for b in bboxes),
        max(b[3] for b in bboxes),
    )


#: TICKET-A1b (#634) round 2 (owner correction, 2026-09-06): the minimum
#: share of a candidate's own numeric body rows (``RowCorroboration.total``)
#: that the union of a page's ``detected_table_bboxes`` must expose as
#: native numeric bands (``RowCorroboration.native_numeric_rows``) before
#: that bbox-scoped region is trusted at all. Below this share the detector
#: has under-covered the table (a partial box, or one block of a
#: multi-block annex table), and scoring against it turns a genuinely
#: corroborating candidate's own numbers into false "extras" -- the exact
#: defect the owner's second ruling identified: all six census fixtures'
#: bbox-union coverage measured 0.10-0.69 (16/39, 18/39, 5/39, 4/14, 25/36,
#: 23/36) while the SAME six fixtures' whole-PAGE coverage measured 1.28-1.93
#: (50/39, 51/39, 50/39, 27/14, 50/36, 46/36) -- see
#: ``docs/log/2026-09-06_A1b-corroboration-selection.md``. 0.75 sits with
#: margin below every measured full-page ratio and above every measured
#: under-covering one; no finer anchor exists in the data than that gap.
REGION_COVERAGE_MIN_SHARE: float = 0.75


def _row_shape_reconciliation_ok(words: list, markdown: str) -> bool:
    """TICKET-A1b (#634) round 3 (owner redesign, 2026-09-06): closes the
    A1->A2 truncation window with a SHAPE-based check, not a geometric one.

    Rounds 1-2 anchored the row-count-allowance and edge-row checks to
    ``detected_table_bboxes``, then (when that under-covered) to a
    band-run extended by a distance tolerance. Both failed: measured
    against all six census fixtures, the detector's own bboxes sit
    entirely OUTSIDE the numeric body (bulletin p1's bbox ends at y=258;
    its data rows run 275.8-635.2 -- see the filed detector issue), and a
    pitch-based distance tolerance cannot separate "next block of the same
    multi-block table" (report p3's own internal section-heading gap,
    48.4pt, 6.31x its row pitch) from "table ended, footnote starts"
    (bulletin p1's real boundary, 20.15pt, 2.62x its row pitch) -- the
    former is LARGER than the latter, so no threshold, raw or
    pitch-normalized, admits one while excluding the other.

    This check instead asks whether native page has as many TABLE-SHAPED
    rows as the candidate claims to have emitted, using the candidate's OWN
    numeric body rows to define what "table-shaped" means -- no distance or
    geometry involved:

    - ``ROW_SHAPE_MIN`` is data-derived per candidate: the minimum numeric-
      token count over the candidate's own ``numeric_body_rows``. A footnote
      band (one bare number), a date/label-only band, and a value-less
      section-heading band (zero numeric tokens) are all narrower than any
      real data row and fall out on their own -- no named constant needed.
    - ``native_table_rows`` counts every native baseline band on the WHOLE
      PAGE (``region=None``, matching what ``corroborate_rows`` already
      scores the ``clears`` gate against once under-coverage widens it --
      see ``REGION_COVERAGE_MIN_SHARE`` above) whose own numeric-token count
      is ``>= ROW_SHAPE_MIN`` and that is not the printed column-index
      legend row (``is_column_index_row``, a table convention, not data).
    - Reconciles with A1a's own ``ROW_CORROBORATION_MIN`` allowance: the
      candidate's total must be at least that share of ``native_table_rows``,
      or the candidate dropped whole rows and is not eligible, regardless of
      how well the rows it DID emit corroborate.

    Measured on all six fixtures (see the log): every real winning
    candidate reconciles EXACTLY (ratio 1.000 -- 39/39, 39/39, 39/39, 14/14,
    36/36, 36/36; no strays). The reviewer's last-10-rows-deleted repro
    measures 29 vs a 39-row native floor (threshold 36, fails); a
    first-5-rows-deleted variant measures 34 vs the same floor (also fails,
    leading edge caught the same way -- no separate edge-row walk needed,
    since every unmatched table-shaped native band counts wherever it sits,
    not just at the bound range's boundary).

    A page with two independent tables where the candidate covers only one
    is REJECTED here by design: the page-level ``native_table_rows`` counts
    both tables' rows, so a candidate missing an entire second table cannot
    reconcile -- the second table's rows are lost content, same as any
    other dropped rows.
    """
    from socr.tables.row_corroboration import (
        ROW_CORROBORATION_MIN,
        numeric_body_rows,
        table_blocks,
        table_shaped_native_row_count,
    )

    candidate_rows = [
        row for rows in table_blocks(markdown) for row in numeric_body_rows(rows) if row
    ]
    if not candidate_rows:
        return True

    row_shape_min = min(len(row) for row in candidate_rows)
    native_table_rows = table_shaped_native_row_count(words, row_shape_min)
    if native_table_rows <= 0:
        return True

    return len(candidate_rows) >= math.ceil(native_table_rows * ROW_CORROBORATION_MIN)


def _row_corroborated_grid_winner(p):
    """TICKET-A1b (#634): S1 case (i)-b -- a candidate the strict grid-authored
    pool discarded, kept anyway because its rows measurably reproduce the
    native page.

    Returns ``(PageOutput, RowCorroboration)`` for the winning candidate, or
    ``None`` if nothing corroborates (including plain abstention: no native
    words cached, no detected bbox, or ``corroborate_rows`` itself abstains).

    GATE (owner ruling, 2026-09-06 -- see the ticket's own motivating
    evidence): this is called ONLY by ``structure_class_grid_winner`` /
    ``structure_class_grid_corroboration`` once ``_strict_grid_authored_pool``
    is already empty, i.e. the S1 floor would otherwise apply -- NOT gated on
    ``native_table_header_unattributed``. The ticket's literal spec named
    that flag, but all six ECB defect-census fixtures that motivate this
    ticket measure it ``False`` (the header-loss signal in the census came
    from a MODEL-side ``table_structure_failed: header_unattributed`` audit
    event on the losing candidate, not this native page flag); gating on the
    native flag would make the fallback a no-op on every one of its own
    motivating pages. The owner ruling is about OUTCOME, not flag: whenever
    the floor would ship the fail-closed marker, consult native row
    corroboration first. C1's invariant is untouched by this -- native still
    never authors the grid; what can now ship is a MODEL candidate the
    native layer corroborates row-for-row, exactly the defect class A1a's
    ``row_corroboration`` module was built to measure (a numerically
    reproducing candidate the multiset-based value guard hard-rejected
    anyway).

    Scored against ``_grid_reading_attempt``'s pool, not
    ``_grid_authored_attempt``'s -- deliberately wider than S1 case (i): a
    candidate the value guard hard-rejected on a stale multiset comparison is
    exactly what row corroboration exists to re-examine (a flattened row is
    multiset-identical to a correct one; the ordered per-row check is not).
    Wider still than ``_grid_shaped_attempt`` -- measured against the real
    ECB fixture cache, 0/6 real qwen/gemini candidates pass
    ``has_strict_table_grid``'s uniform-body-width requirement (a spanning
    units/legend row above the numeric body is routine in real statistical
    tables), so ``_grid_reading_attempt`` uses ``has_authored_table_grid``
    (ragged body allowed) instead -- see its own docstring.

    Tie-break (disclosed judgment call): among clearing candidates, prefer
    one whose page-level ``table_ladder_disposition`` is NOT a recorded
    negative verdict, then the higher corroboration share, then the existing
    ``(audit_passed, confidence, word_count)`` ordering S1 case (i) uses.

    TICKET-A1b (#634) round 2 (owner correction, 2026-09-06): the region
    scored against is NOT unconditionally the bbox union. Measured against
    all six census fixtures, ``detected_table_bboxes`` drastically
    under-covers a real annex table (a partial box, or one block of a
    multi-block stacked table) -- 16-18 native numeric words captured
    against 410-606 on the whole page -- which turns a genuinely
    corroborating candidate's own numbers into false "extras" and a false
    ``clears=False``. Per candidate: score against the bbox union first; if
    its own native-row coverage against THIS candidate's row count
    (``native_numeric_rows / total``) is below ``REGION_COVERAGE_MIN_SHARE``,
    the detector has under-covered and the ``clears`` gate is re-scored
    against the whole page (``region=None`` -- ``words_in_region``'s own
    no-op sentinel). Recorded, not silently substituted: each scored
    candidate carries its ``corroboration_region`` ("bbox_union" or "page")
    and the pre-widen coverage share, both surfaced through to the sidecar
    by ``_apply_row_corroboration_disclosure``.

    TICKET-A1b (#634) round 3 (owner redesign, 2026-09-06): the row-count
    reconciliation below (``_row_shape_reconciliation_ok``) runs
    UNCONDITIONALLY on every candidate, regardless of ``region_kind`` --
    see its own docstring for why the region-anchored (round 1) and
    band-run-distance-anchored (round 2 follow-up) designs both broke on
    real fixtures, and why the shape-based replacement does not need a
    region argument at all (it always reconciles against the whole page).

    TICKET-A2 (#645): a candidate ``_truncated_grid_reading_ids`` flags is
    dropped from the pool below whenever any other page candidate is not
    truncated -- the ticket's own live fixture reaches this fallback
    precisely because A2 emptied the strict pool of the truncated candidate,
    so this pool must not turn around and score that same truncated
    candidate back in.
    """
    words = getattr(p, "native_words", None) or []
    if not words:
        return None
    bbox_region = _union_bbox(list(getattr(p, "detected_table_bboxes", None) or []))
    if bbox_region is None:
        return None

    from socr.tables.row_corroboration import corroborate_rows

    seen_ids: set[int] = set()
    candidates: list[PageOutput] = []
    for out in [p.best_output, *p.attempts]:
        if out is None or id(out) in seen_ids or not _grid_reading_attempt(out):
            continue
        seen_ids.add(id(out))
        candidates.append(out)

    truncated_ids = _truncated_grid_reading_ids(p)
    if truncated_ids:
        candidates = [out for out in candidates if id(out) not in truncated_ids]
    if not candidates:
        return None

    scored = []
    for out in candidates:
        text = out.text or ""
        rc_bbox = corroborate_rows(words, text, bbox_region)
        region_kind = "bbox_union"
        coverage_share = None
        rc = rc_bbox
        if rc_bbox.total > 0:
            coverage_share = rc_bbox.native_numeric_rows / rc_bbox.total
            if coverage_share < REGION_COVERAGE_MIN_SHARE:
                region_kind = "page"
                rc = corroborate_rows(words, text, None)
        if rc.clears is not True:
            continue
        # TICKET-A1b round 3 (owner redesign, 2026-09-06): shape-based row
        # reconciliation, always scored against the whole page regardless
        # of ``region_kind`` -- see ``_row_shape_reconciliation_ok``'s own
        # docstring for why neither the bbox nor a distance-bounded region
        # around it can anchor this check on the real fixtures.
        if not _row_shape_reconciliation_ok(words, text):
            continue
        scored.append((out, rc, region_kind, coverage_share))
    if not scored:
        return None

    ladder_not_rejected = getattr(p, "table_ladder_disposition", None) is None

    def _tie_break(pair):
        out, rc, _region_kind, _coverage_share = pair
        return (
            ladder_not_rejected,
            rc.share or 0.0,
            out.audit_passed,
            out.confidence,
            out.word_count,
        )

    return max(scored, key=_tie_break)


def structure_class_grid_corroboration(p):
    """TICKET-A1b (#634): the ``(RowCorroboration, region_kind, coverage_share)``
    detail behind ``structure_class_grid_winner``'s corroboration-fallback
    winner, for the caller's per-row surfacing (markdown splice + audit
    event). ``region_kind`` is ``"bbox_union"`` or ``"page"`` (round 2:
    which region the winner was actually scored against, after the
    coverage-based widening check); ``coverage_share`` is the bbox union's
    own pre-widen ``native_numeric_rows / total`` ratio, or ``None`` if the
    winning candidate had no numeric body rows to divide by.

    ``None`` unless ``structure_class_grid_winner`` actually shipped a
    fallback candidate -- mirrors its own gating (``_reaches_structure_class_branch``,
    empty strict pool) so the two functions can never disagree about which
    page this applies to.
    """
    if not _reaches_structure_class_branch(p):
        return None
    if _strict_grid_authored_pool(p):
        return None
    corroborated = _row_corroborated_grid_winner(p)
    if corroborated is None:
        return None
    _out, rc, region_kind, coverage_share = corroborated
    return rc, region_kind, coverage_share


def _table_block_layout(markdown: str) -> list[dict]:
    """TICKET-A1b (#634): per markdown table block, the header line(s)' raw
    text and the ORIGINAL SOURCE LINE NUMBER of each numeric body row, in
    the same order ``row_corroboration.numeric_body_rows`` returns them
    for that block.

    ``corroborate_rows``/``RowCorroboration.unbound_rows`` work entirely in
    COUNTS -- a block-relative index into a candidate's numeric body rows --
    because that is all selection needs to score a candidate. Turning an
    unbound row's index back into an actual markdown LINE to annotate needs
    line numbers that return type does not carry, so this mirrors
    ``row_corroboration``'s own block/row detection (``table_blocks``,
    ``numeric_body_rows``) rather than re-deriving the filtering rules --
    every predicate call below is that module's own function, imported,
    never duplicated logic. The one necessarily-duplicated piece is the
    line-grouping loop itself (pipe-line runs, separator/header exclusion),
    since ``table_blocks`` discards line numbers before returning; kept
    line-for-line identical to that function on purpose.
    """
    from socr.tables.row_corroboration import is_separator_row, numeric_body_rows, split_cells

    lines = markdown.splitlines()
    pipe_idxs = [i for i, ln in enumerate(lines) if "|" in ln and ln.strip()]
    layouts: list[dict] = []
    i = 0
    while i < len(pipe_idxs):
        j = i
        while j + 1 < len(pipe_idxs) and pipe_idxs[j + 1] == pipe_idxs[j] + 1:
            j += 1
        run = pipe_idxs[i : j + 1]
        if len(run) >= 2:
            run_cells = [split_cells(lines[k]) for k in run]
            separator_positions = {
                pos for pos, cells in enumerate(run_cells) if is_separator_row(cells)
            }
            header_positions = {pos - 1 for pos in separator_positions if pos - 1 >= 0}
            kept = [
                (run[pos], cells)
                for pos, cells in enumerate(run_cells)
                if pos not in separator_positions and pos not in header_positions
            ]
            if kept:
                body_line_nums = [
                    line_num for line_num, cells in kept if numeric_body_rows([cells])
                ]
                layouts.append(
                    {
                        "header_lines": [lines[run[pos]] for pos in sorted(header_positions)],
                        "row_line_nums": body_line_nums,
                    }
                )
        i = j + 1
    return layouts


def _splice_unverified_row_markers(markdown: str, unbound_rows: tuple[tuple[int, ...], ...]) -> str:
    """TICKET-A1b (#634): append an inline ``<!-- row unverified -->`` marker
    to every table row ``row_corroboration`` could not bind to a native
    baseline band, so the doubt travels with the shipped markdown itself,
    not only the sidecar/audit log (CLAUDE.md: no silent content loss -- a
    table clearing the share floor with one wrong row must not read as the
    whole table having cleared).

    Fails safe: a block-count or row-index mismatch against
    ``_table_block_layout``'s own reading of the SAME markdown (which
    should not be possible, since both derive from the same text, but nothing
    here is worth ever crashing assembly over) is silently skipped for that
    row rather than raising or corrupting an unrelated line.
    """
    if not any(unbound_rows):
        return markdown
    layouts = _table_block_layout(markdown)
    lines = markdown.splitlines()
    for block_idx, row_idxs in enumerate(unbound_rows):
        if not row_idxs or block_idx >= len(layouts):
            continue
        row_line_nums = layouts[block_idx]["row_line_nums"]
        for row_idx in row_idxs:
            if row_idx >= len(row_line_nums):
                continue
            line_num = row_line_nums[row_idx]
            if line_num >= len(lines) or "row unverified" in lines[line_num]:
                continue
            lines[line_num] = f"{lines[line_num].rstrip()} <!-- row unverified -->"
    return "\n".join(lines)


def _apply_row_corroboration_disclosure(
    state, page_num: int, grid_winner, corroboration, region_kind: str, coverage_share: float | None
):
    """TICKET-A1b (#634): the visible half of the row-corroboration fallback.

    Splices per-row ``<!-- row unverified -->`` markers into the shipped
    markdown (see ``_splice_unverified_row_markers``) and appends an
    ``AuditEvent`` naming the header text AS EMITTED (the ticket's own
    "owner may rule for neutral ``col N`` stubs later" -- recording the
    header either way keeps that choice reversible) plus the corroboration
    counts, so batch analysis and the sidecar can see this page shipped via
    the fallback rather than S1 case (i)'s ordinary grid-authored pool.

    ``region_kind`` / ``coverage_share`` (round 2, owner correction
    2026-09-06): which region the winner was actually scored against
    (``"bbox_union"`` or ``"page"``) and the bbox union's own pre-widen
    coverage ratio, recorded so a partial detected bbox that forced a
    page-wide fallback is visible in the sidecar, never silent.
    """
    from socr.core.audit_log import AuditEvent

    text = grid_winner.text or ""
    spliced = _splice_unverified_row_markers(text, corroboration.unbound_rows)
    header_lines = [line for layout in _table_block_layout(text) for line in layout["header_lines"]]
    share = corroboration.share
    detail = (
        f"row corroboration kept this candidate over the structure-class floor: "
        f"{corroboration.bound}/{corroboration.total} rows bound"
        + (f" (share {share:.3f})" if share is not None else "")
        + f" [corroboration_region={region_kind}]"
    )
    state.events.append(
        AuditEvent(
            page_num=page_num,
            kind="structure_class_row_corroborated",
            engine=grid_winner.engine or "",
            detail=detail,
            data={
                "bound": corroboration.bound,
                "total": corroboration.total,
                "unbound_rows": [list(idxs) for idxs in corroboration.unbound_rows],
                "header_lines": header_lines,
                "corroboration_region": region_kind,
                "region_coverage_share": coverage_share,
            },
        )
    )
    if spliced == text:
        return grid_winner
    return replace(grid_winner, text=spliced)


def structure_class_grid_winner(p) -> PageOutput | None:
    """S1 case (i): the grid-authoring model attempt a structure-class page ships.

    Ranks every qualifying attempt by ``(audit_passed, confidence,
    word_count)`` and takes the best -- MAJOR 4 on #269: the prior version
    walked ``reversed(p.attempts)`` and took the LAST qualifying one, so a
    refused earlier attempt with a better reading could lose to a later,
    worse one purely by ladder position (and if native would have beaten
    that worse grid, shipping it is a text regression the 7/8 measurement
    does not license). ``best_output`` is checked first regardless -- the
    agentic scorer already judged it the most trustworthy attempt, a
    judgement rather than an ordering, and #259's own tie-break defers to it
    the same way.

    Returned UNCHANGED, body untouched, flagged only per its own status (S1
    spec, verbatim) -- unlike the #259 branch, S1 case (i) does not demote on
    a copy, because there is no binder yet (R2) to justify a stronger warning
    than whatever the ladder's own routing already left on the attempt. This
    function mutates nothing either way: ``audit_passed`` is the
    winner-SELECTION flag, not a page-quality flag, and flipping it on the
    stored attempt discards the page (the #252 round-1 defect).

    TICKET-A1b (#634) case (i)-b: when the strict pool above is empty, falls
    back to ``_row_corroborated_grid_winner`` -- a candidate the strict pool
    discarded but whose rows measurably reproduce the native page, per
    A1a's row-corroboration check. Consult
    ``structure_class_grid_corroboration(p)`` for the detail behind that
    winner (share, unbound rows) when this returns one.
    """
    if not _reaches_structure_class_branch(p):
        return None
    strict_pool = _strict_grid_authored_pool(p)
    if strict_pool:
        if p.best_output in strict_pool:
            return p.best_output
        return max(strict_pool, key=lambda out: (out.audit_passed, out.confidence, out.word_count))
    corroborated = _row_corroborated_grid_winner(p)
    return corroborated[0] if corroborated is not None else None


def structure_class_floor_applies(p) -> bool:
    """S1/P2 case (iii): whether the structure-class branch demotes THIS
    page's winner to the fail-closed floor (ERROR/audit_passed=False).

    Exported so ``_phase_assemble``'s document-level buckets (``pages_ok``,
    the audit log, the CLI summary) can see this demotion too. Nothing in
    ``_score_per_page``'s upstream scoring touches ``p.best_output`` for this
    case -- a clean-looking, unflagged native table is exactly what that
    scorer misses (the 2026-08-20 measurement: the winner-side chain up to
    this point -- ``native_verifier``, ``source_evidence``, header anchors --
    compares numeric multisets, blind to binding loss) -- so
    ``p.best_output.audit_passed`` stays True and a bucket keyed off it alone
    would silently miss the page. CLAUDE.md: a failure must surface at every
    level, not just one.

    ``_reaches_structure_class_branch`` is the single unified gate (BLOCKING 2
    on #269) and already checks ``is_structure_class()`` and R3 (at least one
    non-native model rung ran -- inlined as its own final ``return``, not a
    separate named predicate) itself, in the same order ``_winning_page_output``
    does -- this function's own job is only to ask whether that gate passes and
    no attempt authored a usable grid (``structure_class_grid_winner`` returns
    ``None``).
    """
    if not _reaches_structure_class_branch(p):
        return False
    return structure_class_grid_winner(p) is None


def structure_class_floor_text(p, page_num: int) -> str:
    """P2 / GH-317 / GH-520: fail-closed text for an exhausted structure-class page.

    Every table on the page becomes the standard unverifiable-table marker plus
    the ``d3_floor_png_ref``. The page's prose survives ONLY when the page's
    tables can be enumerated independently of the parser that produced them;
    otherwise the whole page becomes the marker and no native byte ships.

    Why the guard is what it is
    ---------------------------
    Cold review rounds 1 and 2, finding 1. Round 1 shipped a regional splice via
    ``splice_all_table_regions``, which proves only that it replaced every block
    ITS OWN PARSER could find -- so a page with two detected tables, one emitted
    as valid GFM and the other collapsed to ragged lines, replaced the parseable
    one and shipped the collapsed one inside text labelled "preserved prose".
    Round 1's fix required coverage against ``native_table_region_count`` /
    ``native_table_region_identities``, which does not close it: those are
    recorded by ``_verify_regions`` over ``table_regions``, and ``table_regions``
    is built only from SUCCESSFUL reconstructions. A sibling that failed
    reconstruction is absent from the count, so the check agrees with the very
    parser it is meant to audit.

    P2 therefore removed the splice entirely and accepted the cost -- the page's
    prose -- as a known limitation, pending an independent signal.

    GH-520 records that signal: ``detected_table_count`` / ``detected_table_bboxes``
    come from ``find_tables()`` at detection time, before reconstruction runs, so
    a table that failed to parse is still counted. The splice returns behind it.

    What the guard requires
    -----------------------
    All four of these, or the whole page floors:

    * at least one table was detected -- a zero count is no evidence, not a
      licence (a borderless table seen only by the lane-cooccupancy pass
      contributes no bbox and is not counted at all, so zero is common);
    * every detected table has a usable bbox
      (``count == len(bboxes)``) -- a table nobody can point at cannot be shown
      to be covered;
    * **every detected table was successfully reconstructed**
      (``native_table_region_count == count``);
    * the parser found exactly that many blocks
      (``len(find_table_blocks(text)) == count``).

    The third condition is the one that carries the argument, and it is the
    exact inversion of round 1's mistake. Round 1 compared the parser's region
    count against ITSELF, which is circular. Comparing it against the DETECTION
    count is not: it asks "did reconstruction produce a region for every table
    the detector found", and a sibling that collapsed makes the two disagree.

    Without it, counting blocks is not enough (cubic P1 on #571): a page with
    two detected tables, one of them collapsed and never parsed, plus an
    unrelated pipe-shaped prose block, has two parsed blocks and two detected
    tables. The counts match by coincidence, the splice replaces the two blocks
    it can see, and the collapsed table ships as preserved prose -- round 1's
    bug exactly. That is why an earlier draft of this docstring was wrong to say
    equal counts establish "no detected table is missing from the parser's block
    list". They do not. The reconstruction-count agreement does.

    What it still does not claim
    ----------------------------
    Block *i* is not proven to be the text of bbox *i*. A markdown block carries
    no geometry, so the ordering is document order and nothing verifies the
    pairing. The guard does not need that pairing: it establishes that every
    detected table has a reconstructed region AND that the parser sees exactly
    as many blocks as there are tables, and ``splice_all_table_regions``
    then replaces every block it finds. Which marker landed on which table is
    not a property anything downstream reads.
    """
    return table_floor_text_for_source(p, page_num, getattr(p, "native_text", "") or "")


def _table_bbox_sane(p) -> bool:
    """B1 / #591 bbox sanity checks the panel asked for.

    A detected-table bbox that is structurally not the table it claims to
    bound must fail the GH-520 guard closed, rather than licence a splice
    around the wrong box (#639: on ECB annex pages ``detected_table_bboxes``
    is the caption/units box, not the body).

    Too small: the bbox union's own native bands (clustered the same way
    row corroboration clusters them) contain no genuine numeric token at
    all -- the box left the table's numeric rows outside it.

    Too large: within the bbox union, bands carrying no numeric token
    (prose) outnumber the bands that do -- the box swallowed paragraph
    text beyond the table's own rows. A single spanning units/legend band
    is normal and does not trip this; a majority of prose bands does.

    Neither check is a licence on its own -- both run alongside the four
    coverage/reconstruction conditions below, and any one failing floors
    the whole page. With no bbox claimed at all there is nothing to check
    and this returns True, leaving the existing four conditions to decide
    (they already floor a page whose ``detected_table_count`` is 0).

    #652 P2a -- MISSING EVIDENCE IS NOT A PASS, AND RESUME MUST NOT RE-DECIDE.
    ``native_words`` is a live-run cache: ``_restore_terminal_page_state``
    restores ``detected_table_count`` and ``detected_table_bboxes`` from the
    sidecar but never the words, and ``_select_and_finalize_page`` re-runs
    selection over that restored state. So a resumed page reached this check
    with bboxes but no words, the old ``return True`` skipped both sanity
    checks, and the resumed run could stamp a different floor outcome than
    the run that wrote the bytes. Two changes close it, in this order:

    * the live run's verdict is PERSISTED (``PageState.table_bbox_sane``,
      written to and restored from the sidecar) and is authoritative when
      the words are gone -- resume replays the decision instead of retaking
      it, so live and resumed reach the same outcome;
    * with neither words nor a persisted verdict, this fails CLOSED. A bbox
      was claimed and nothing can check it; absence of evidence is not
      sanity.
    """
    words = getattr(p, "native_words", None) or []
    bboxes = getattr(p, "detected_table_bboxes", None) or []
    if not bboxes:
        return True
    if not words:
        persisted = getattr(p, "table_bbox_sane", None)
        if persisted is None:
            return False
        return bool(persisted)

    from socr.tables.row_corroboration import baseline_bands, words_in_region

    region = _union_bbox(list(bboxes))
    if region is None:
        return True

    bands = baseline_bands(words_in_region(words, region))
    if not bands:
        return False  # too small: no native band at all inside the claimed box
    numeric_bands = [b for b in bands if b.tokens]
    if not numeric_bands:
        return False  # too small: the box captured no numeric row
    prose_bands = [b for b in bands if not b.tokens]
    return len(prose_bands) <= len(numeric_bands)  # too large otherwise


def table_bbox_sanity_verdict(p) -> bool:
    """Public entry for :func:`_table_bbox_sane`, for the orchestrator.

    #652 P2a: the verdict has to be taken while ``native_words`` still exists
    -- once the page is flushed and later resumed those words are gone -- so
    the orchestrator evaluates it at extraction time and records it on the
    page. Crossing a package boundary for a private name is a layering
    violation (``test_package_layering``), so the boundary gets a name.
    """
    return _table_bbox_sane(p)


def table_floor_text_for_source(
    p, page_num: int, source_text: str, *, fallback_marker: str | None = None
) -> str:
    """The GH-520 coverage check plus splice, over ANY source.

    Factored out of ``structure_class_floor_text`` for P1's withhold path,
    which must splice the SELECTED output's text -- a model winner, not
    necessarily the native layer. Substituting ``p.native_text`` there would
    ship a different page's bytes than the one selection chose.

    B1 / #591 extended this to a fifth call site (the ``page_failed`` ending,
    ``_select_page_output_tagged``'s no-text-anywhere fallback) via
    ``fallback_marker``: that caller wants its own whole-page marker text
    (``page_failed_marker``, "no usable OCR output") on a guard failure, not
    this function's D3-style "unverifiable table" default -- the page may
    have failed for a reason that has nothing to do with a table at all.
    Every early return below uses it in place of the D3 marker when given;
    the SPLICE itself is unaffected (the D3 marker still stamps each
    covered table region, unchanged, because the guard passed there).

    The conditions themselves are unchanged and unweakened: they are the
    argument, and ``structure_class_floor_text``'s docstring is where that
    argument lives. ``_table_bbox_sane`` (B1) adds two more: a bbox that is
    structurally not the table it claims to bound (too small or too large)
    fails closed the same as an uncovered or unreconstructed one.
    """
    d3_marker = f"[page {page_num} failed: unverifiable table — see image]"
    png_ref = getattr(p, "d3_floor_png_ref", "")
    whole_page = (
        fallback_marker
        if fallback_marker is not None
        else (f"{d3_marker}\n\n{png_ref}" if png_ref else d3_marker)
    )

    source_text = source_text or ""
    if not source_text.strip():
        return whole_page

    detected_count = getattr(p, "detected_table_count", 0)
    detected_bboxes = getattr(p, "detected_table_bboxes", []) or []
    if type(detected_count) is not int or detected_count <= 0:
        return whole_page
    if len(detected_bboxes) != detected_count:
        return whole_page
    # Every detected table reconstructed. Not circular: the count on the right
    # comes from the detector, not from the parser being audited (cubic P1 on
    # #571 -- block counting alone is satisfied by coincidence).
    if getattr(p, "native_table_region_count", 0) != detected_count:
        return whole_page

    from socr.tables.reconcile import find_table_blocks

    blocks = find_table_blocks(source_text)
    if len(blocks) != detected_count:
        return whole_page

    if not _table_bbox_sane(p):
        return whole_page

    spliced = splice_all_table_regions(source_text, d3_marker, png_ref)
    return spliced if spliced else whole_page


def native_region_text(words: list) -> str:
    """The printed text of *words*, one line per native baseline band.

    #652/#649: both the trusted-layer check and the recovered-prose body need
    the region's text as it was PRINTED, not as a flat bag of words -- one
    line per band, words left to right. Reuses ``cluster_band_words`` so this
    reconstruction and the prose/table partition can never disagree about
    where a line begins.
    """
    from socr.tables.row_corroboration import cluster_band_words

    lines = [
        " ".join(str(w[4]) for w in sorted(band, key=lambda w: w[0]))
        for band in cluster_band_words(words)
    ]
    return "\n".join(line for line in lines if line.strip())


def _page_prose_partition(p) -> list:
    """The prose/table partition of a page's native words, whole page.

    #652 round 9 introduced this because TWO decisions read a page's layout --
    whether a model's prose could be corroborated, and what #649 ships when it
    cannot -- on DIFFERENT populations: the corroboration side partitioned only
    the words outside the detected table bboxes, so an incomplete bbox hid a
    page's numerals from the very gate that asked whether it printed any.

    Round 10 removed the corroboration reader outright (no model prose ships
    from the scanned-table-failure branch at all), so one caller is left and
    the two cannot diverge by construction. The function stays as the single
    named place a page's partition is taken, on every native word it has --
    never a filtered subset, which is the mistake worth keeping named.
    """
    from socr.tables.row_corroboration import partition_prose_bands

    words = getattr(p, "native_words", None) or []
    if not words:
        return []
    return partition_prose_bands(words)


#: Banner stamped above prose recovered by ``native_prose_floor_text``. The
#: page is still an unverified scan whose table was withheld, and the body no
#: longer starts with a failure marker, so the flag is what tells a reader --
#: and ``is_page_failed_marker``, which correctly stops calling this page
#: marker-only -- that these paragraphs are unverified. Deliberately NOT
#: matched by ``_PAGE_FAILED_ANY_RE``: this page ships content.
SCANNED_PROSE_RECOVERED_FLAG = (
    "[page {page_num}: unverified scan — the paragraphs below are this page's own "
    "text layer; every numeric row is withheld]"
)

#: Audit note recorded on the rebuilt output, so the recovery is visible in the
#: page sidecar and not only in the bytes.
SCANNED_PROSE_RECOVERED_NOTE = (
    "scanned_prose_recovered: no OCR attempt could be spliced around the "
    "withheld table; the page's own trusted prose bands ship flagged instead"
)


def _is_restored_prose_recovery(p, page_num: int) -> bool:
    """Whether this page's winner IS an already-finalized prose recovery.

    #649 rounds 2-3. Both halves of the evidence must hold, and neither alone
    is enough:

    * the winner carries ``PageOutput.scanned_prose_recovered``, the TYPED
      field this module sets when it builds such a body. It survives resume
      because the sidecar serialises the winning output and
      ``_restore_terminal_page_state`` rebuilds the ``PageOutput`` from that
      record.
    * the winner's text starts with this module's own banner. A flag without
      the banner would mean something rewrote the body after the recovery, and
      that body is not this function's to vouch for.

    Round 2 asked whether a note SUBSTRING would do, and Astra showed it would
    not: ``orchestrator`` appends ``dual-pass {action}: {summary}``, and that
    summary quotes model-controlled cell text verbatim, so an attempt carrying
    the banner at the top and the note text inside a table cell was shipped as
    an already-finalized recovery -- invented sentence, invented numbers and
    all. Every note author happens to prefix its text today, so an EXACT
    standalone match would close that particular route, but a credential whose
    soundness depends on auditing every present and future note formatter is
    not a credential. A typed field cannot be reached by free text at all, and
    the note stays for human and corpus visibility rather than as evidence.

    The banner alone is likewise not enough, and for the same reason: a model
    that echoes the banner line must not have its whole output shipped past
    the floor.
    """
    out = getattr(p, "best_output", None)
    if out is None:
        return False
    if getattr(out, "scanned_prose_recovered", False) is not True:
        return False
    return (out.text or "").startswith(SCANNED_PROSE_RECOVERED_FLAG.format(page_num=page_num))


def native_prose_floor_text(p, page_num: int, *, marker_line: str, png_ref: str) -> str | None:
    """The page's own prose, flagged, around a withheld table -- or ``None``.

    #649 (owner ruling, 2026-09-10). Fed 1989-11-14 p3 reaches
    ``UNVERIFIABLE_TABLE_SCANNED`` with ``detected_table_count == 0`` and a
    corrupt-but-usable text layer. Its only cached attempt read the page's real
    vocabulary but emitted the swap-arrangement table as column runs with no
    markdown table syntax at all, so ``splice_all_table_regions`` returns
    ``None`` and the marker shipped alone -- taking three paragraphs of the
    FOMC policy directive with it. Nothing was wrong with those paragraphs;
    they were collateral of a table that could not be verified.

    With no table geometry to splice against, the prose region is delimited by
    the page's own native baseline bands (``prose_region_words``): a band below
    ``ROW_SHAPE_MIN`` numeric tokens is prose and ships; every band at or above
    it is the table and is withheld, replaced in place by *marker_line*. The
    withheld half is exactly the numeric content the D3 floor exists to
    protect, so this recovers prose without ever relaxing the floor.

    Three ways to abstain, all of which leave the caller's bare marker:

    * no native words -- no page text to recover;
    * nothing withheld -- there is no table-shaped band here, so this function
      cannot say what it would be shipping prose "around", and a page that
      reached the scanned-table floor with no numeric band at all is a shape
      this has no evidence about;
    * the prose region's own text fails ``text_layer_trusted`` (#652). The
      page is a scan because its layer is corrupt; shipping that corruption as
      recovered text would be the silent loss this ticket is trying to stop,
      wearing the opposite mask. Measured on the ticket's own fixture the
      corruption is INSIDE the table -- prose region 0.5%, numeric bands 33.3%
      -- which is why the check is applied to the region that ships rather
      than to the page.

    What ships is the native layer's own bytes, never a model's: the attempt
    that failed here failed on structure, and re-deriving prose from it would
    put the reordered text back on the page. The page keeps ERROR status and
    its failure mode; only the body changes.
    """
    from socr.core.born_digital import text_layer_trusted

    words = getattr(p, "native_words", None) or []
    if not words:
        return None

    bands = _page_prose_partition(p)
    prose_bands = [band for is_prose, band in bands if is_prose]
    if not prose_bands or all(is_prose for is_prose, _band in bands):
        return None

    prose_words = [word for band in prose_bands for word in band]
    if not text_layer_trusted(native_region_text(prose_words)):
        return None

    blocks: list[str] = [SCANNED_PROSE_RECOVERED_FLAG.format(page_num=page_num)]
    marker_block = f"{marker_line}\n\n{png_ref}" if png_ref else marker_line
    paragraph: list[str] = []
    in_withheld_run = False

    def _flush() -> None:
        if paragraph:
            blocks.append("\n".join(paragraph))
            paragraph.clear()

    from socr.tables.reconcile import table_syntax_line_indices

    # A native line that parses as markdown TABLE SYNTAX can never ship as
    # prose here, whatever its digits say: the rows beneath it are withheld by
    # definition on this page, so emitting it would assemble a header and a
    # separator over content the floor just refused to verify -- exactly what
    # ``_apply_table_emission_guard`` catches downstream. It joins the withheld
    # run instead.
    #
    # #649 rounds 3-4 (Astra): "is this line table syntax" is a question about
    # CONTEXT, not about the line. Round 2 asked ``_is_table_line``, whose
    # regex accepts any line containing a pipe, so a numeral-free sentence
    # carrying one was withheld. Round 3 asked ``find_table_blocks``, which
    # knows a run of pipe lines but not where that run's table BEGINS, so the
    # same sentence still vanished when it sat directly before or after a real
    # table. The boundaries come from the table's own structure -- the
    # separator row, its header, and the delimited rows beneath it.
    band_lines = [" ".join(str(w[4]) for w in band).strip() for _is_prose, band in bands]
    table_syntax = table_syntax_line_indices(band_lines)

    for idx, (is_prose, band) in enumerate(bands):
        line = band_lines[idx]
        if is_prose and idx in table_syntax:
            is_prose = False
        if is_prose:
            in_withheld_run = False
            # Consecutive printed lines join into one paragraph rather than
            # becoming one block each: these ARE the page's lines, and a
            # directive split into twenty one-line paragraphs is not the page.
            if line:
                paragraph.append(line)
            continue
        if in_withheld_run:
            continue
        # #649 round 2 (Astra): ONE marker per contiguous withheld run, not one
        # per page. The withholding predicate covers every printed digit, so a
        # withheld band is no longer always inside the table -- a prose line
        # carrying a printed value ("...remained around 5-1/4 percent...", the
        # one such line on the ticket's fixture) is withheld mid-paragraph. A
        # single marker at the top of the page would elide that line in
        # silence, which is the exact loss this lane exists to stop. The marker
        # names withheld content, not a table count: a run count is not a table
        # count either, and this branch runs with ``detected_table_count == 0``,
        # so no claim about how many tables the page holds is made anywhere.
        _flush()
        blocks.append(marker_block)
        in_withheld_run = True

    _flush()
    return "\n\n".join(blocks)


class PageEnding(str, Enum):
    """Normalized ending vocabulary for what actually ships on a page.

    DEMOTED_NATIVE is the panel-approved temporary deviation from the
    three-ending ruling. Exit criterion: enumerate corpus pages by
    needs_ocr_enhancement, chart_asset_render_failed, text_grid_rejected,
    and residual native-table-defect trigger; hand-check each trigger's
    fidelity; assign each trigger independently to N or F in a later ticket.
    """

    NATIVE_PROSE = "native_prose"
    MODEL_OUTPUT = "model_output"
    FAIL_CLOSED_MARKER = "fail_closed_marker"
    #: DEMOTED_NATIVE is the panel-approved temporary deviation from the
    #: three-ending ruling. Exit criterion: enumerate corpus pages by
    #: needs_ocr_enhancement, chart_asset_render_failed, text_grid_rejected,
    #: and residual native-table-defect trigger; hand-check each trigger's
    #: fidelity; assign each trigger independently to N or F in a later ticket.
    DEMOTED_NATIVE = "demoted_native"


class PagePrimaryReason(str, Enum):
    """Normalized primary cause vocabulary explaining why a page received its ending."""

    CORRUPT_MATH_HYBRID = "corrupt_math_hybrid"
    ACCEPTED_OUTPUT = "accepted_output"
    SCANNED_TABLE_UNVERIFIABLE = "scanned_table_unverifiable"
    NATIVE_TABLE_UNVERIFIABLE = "native_table_unverifiable"
    ROTATED_NATIVE_TEXT_SHREDDED = "rotated_native_text_shredded"
    NATIVE_TABLE_DISTRUST = "native_table_distrust"
    STRUCTURE_CLASS = "structure_class"
    DEMOTED_NATIVE_RECOVERY_EXHAUSTION = "demoted_native_recovery_exhaustion"
    CLEAN_NATIVE_PROSE = "clean_native_prose"
    WHOLE_DOCUMENT_SECTION = "whole_document_section"
    UNACCEPTED_OUTPUT_KEPT = "unaccepted_output_kept"
    NO_USABLE_OUTPUT = "no_usable_output"
    INVALID_TABLE_EMISSION = "invalid_table_emission"
    #: P1 (owner ruling Q2): the page shipped a fail-closed marker in place of a
    #: table because the readers rejected it and neither ruled guard cleared it.
    #: Distinct from SHIPPED_FAILURE_MARKER, which means socr cannot attribute
    #: the marker it is looking at; here it can, exactly.
    TABLE_JUDGE_WITHHELD = "table_judge_withheld"
    #: The shipped bytes are a recognised failure marker that selection did not
    #: account for. The page shipped no content, and socr will not name a cause it
    #: cannot read off the bytes.
    SHIPPED_FAILURE_MARKER = "shipped_failure_marker"


@dataclass(frozen=True)
class PageDisposition:
    """Public, finalization-aware page outcome (ending + normalized primary cause)."""

    ending: PageEnding
    primary_reason: PagePrimaryReason

    def to_dict(self) -> dict[str, str]:
        return {
            "ending": self.ending.value,
            "primary_reason": self.primary_reason.value,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PageDisposition:
        ending_val = d["ending"]
        ending = PageEnding(ending_val) if isinstance(ending_val, str) else ending_val
        reason_val = d["primary_reason"]
        reason = PagePrimaryReason(reason_val) if isinstance(reason_val, str) else reason_val
        return cls(ending=ending, primary_reason=reason)


class SelectionProvenance(str, Enum):
    """R7: which of ``_select_page_output_tagged``'s endings shipped this page.

    The cascade is 16 returns and **zero loops** (AST-verified), so exactly one
    ending runs per page and this tag is a total, exclusive classification of the
    SHIP axis. It is deliberately not a classification of the page: orthogonal
    alerts (``value_drift``, ``fabricated_ref``, ``text_grid_rejected``) co-occur
    with a page that ships perfectly well and are NOT members here.

    **It names the ending SELECTION took, not the final shipped bytes.**
    ``_winning_page_output`` applies ``_apply_table_emission_guard`` after the tag
    is dropped, and that guard can replace any selected output with a failure
    marker (``FailureMode.TABLE_EMISSION_INVALID``). A page tagged
    ``PASSING_BEST_OUTPUT`` can therefore still ship a marker. Consumers that need
    "what shipped" must still inspect the emitted text; the tag answers "which
    branch chose it".

    The tag exists so callers stop re-deriving "which branch shipped?" with mirror
    predicates that must be kept in lockstep with this function -- the drift that
    ``_reaches_structure_class_branch`` was written to repair. It is INTERNAL:
    ``_select_page_output`` drops it, so every existing caller sees byte-identical
    output.

    Order of definition follows the cascade's own order, which is the only
    authority on precedence. In particular ``CORRUPT_MATH_HYBRID`` outranks the
    model-kept endings; nothing in the codebase had to state that before, and
    re-deriving it elsewhere would mean inventing it.
    """

    #: native+math hybrid attempt kept over the ladder winner
    CORRUPT_MATH_HYBRID = "corrupt_math_hybrid"
    #: the ladder's passing best_output ships clean -- the ordinary success
    PASSING_BEST_OUTPUT = "passing_best_output"
    #: scanned page, source-evidence table check failed: fail-closed marker.
    #: ("D3" in the surrounding identifiers is Option D3 of the 2026-06-17 table-repair
    #: design menu -- a numbered choice, carrying no meaning. Named for what it does.)
    UNVERIFIABLE_TABLE_SCANNED = "unverifiable_table_scanned"
    #: #262: same conjunction, but an attempt authored a grid -- the model reading is
    #: kept over the fail-closed marker, shipped flagged
    UNVERIFIABLE_TABLE_MODEL_KEPT = "unverifiable_table_model_kept"
    #: born-digital page whose native table failed verification: fail-closed marker
    UNVERIFIABLE_TABLE_NATIVE = "unverifiable_table_native"
    #: rotated-text extraction shredded the native layer: fail-closed marker
    ROTATED_TEXT_SHREDDED = "rotated_text_shredded"
    #: #259: ladder accepted nothing but the model produced a table -- kept flagged
    FLAGGED_MODEL_KEPT = "flagged_model_kept"
    #: TICKET-A1c (#641): the winner came from A1b's row-corroboration fallback
    #: (the strict grid-authored pool was empty) -- ships WARNING /
    #: ``FailureMode.HEADER_BINDING_UNVERIFIED`` UNCONDITIONALLY, regardless of
    #: the candidate's own ``audit_passed``, because only its ROW shape was ever
    #: checked, never its HEADER/column binding. Deliberately a SEPARATE member
    #: from ``STRUCTURE_CLASS_GRID_FLAGGED`` (R7: two endings must never share
    #: one tag) even though both map to the same base disposition below -- they
    #: are different SHIP reasons (a rescued fallback candidate vs. an
    #: ordinarily-authored one the judge rejected).
    STRUCTURE_CLASS_GRID_CORROBORATED = "structure_class_grid_corroborated"
    #: structure-class: an attempt authored a grid and it passed audit
    STRUCTURE_CLASS_GRID_PASSING = "structure_class_grid_passing"
    #: structure-class: grid winner kept but demoted to WARNING
    STRUCTURE_CLASS_GRID_FLAGGED = "structure_class_grid_flagged"
    #: structure-class (iii): no attempt authored a grid -- fail-closed floor
    #: (whole-page marker + image ref; no native byte ships)
    STRUCTURE_CLASS_FLOOR = "structure_class_floor"
    #: native layer deficient, recovery tried and never passed: native as FALLBACK,
    #: shipped WARNING / audit_passed=False
    NATIVE_FALLBACK = "native_fallback"
    #: the SAME ending, undemoted: a born-digital page with native text and no
    #: distrust flag ships ordinary native SUCCESS. Split from NATIVE_FALLBACK
    #: because that ending's ``native_demoted`` switch produces two dispositions
    #: from one return -- tagging both as "fallback" would have made part two
    #: count every clean --native-only page as a fallback page, flipping the
    #: document to AUDIT_FAILED and emitting fallback warnings for healthy pages.
    NATIVE_CLEAN = "native_clean"
    #: text recovered from a whole-document attempt, split on ``## Page N``
    WHOLE_DOC_SECTION = "whole_doc_section"
    #: a per-page attempt that failed audit still beats an empty page
    BEST_OUTPUT_UNVERIFIED = "best_output_unverified"
    #: best_output was cleared; the rejected text in ``attempts`` ships flagged
    BEST_ATTEMPT_FLAGGED = "best_attempt_flagged"
    #: nothing anywhere produced text: explicit failure marker, never a silent gap
    NO_TEXT_MARKER = "no_text_marker"


_PROVENANCE_TO_DISPOSITION: dict[SelectionProvenance, PageDisposition] = {
    SelectionProvenance.CORRUPT_MATH_HYBRID: PageDisposition(
        PageEnding.MODEL_OUTPUT, PagePrimaryReason.CORRUPT_MATH_HYBRID
    ),
    SelectionProvenance.PASSING_BEST_OUTPUT: PageDisposition(
        PageEnding.MODEL_OUTPUT, PagePrimaryReason.ACCEPTED_OUTPUT
    ),
    SelectionProvenance.UNVERIFIABLE_TABLE_SCANNED: PageDisposition(
        PageEnding.FAIL_CLOSED_MARKER, PagePrimaryReason.SCANNED_TABLE_UNVERIFIABLE
    ),
    SelectionProvenance.UNVERIFIABLE_TABLE_MODEL_KEPT: PageDisposition(
        PageEnding.MODEL_OUTPUT, PagePrimaryReason.NATIVE_TABLE_UNVERIFIABLE
    ),
    SelectionProvenance.UNVERIFIABLE_TABLE_NATIVE: PageDisposition(
        PageEnding.FAIL_CLOSED_MARKER, PagePrimaryReason.NATIVE_TABLE_UNVERIFIABLE
    ),
    SelectionProvenance.ROTATED_TEXT_SHREDDED: PageDisposition(
        PageEnding.FAIL_CLOSED_MARKER, PagePrimaryReason.ROTATED_NATIVE_TEXT_SHREDDED
    ),
    SelectionProvenance.FLAGGED_MODEL_KEPT: PageDisposition(
        PageEnding.MODEL_OUTPUT, PagePrimaryReason.NATIVE_TABLE_DISTRUST
    ),
    SelectionProvenance.STRUCTURE_CLASS_GRID_CORROBORATED: PageDisposition(
        PageEnding.MODEL_OUTPUT, PagePrimaryReason.STRUCTURE_CLASS
    ),
    SelectionProvenance.STRUCTURE_CLASS_GRID_PASSING: PageDisposition(
        PageEnding.MODEL_OUTPUT, PagePrimaryReason.STRUCTURE_CLASS
    ),
    SelectionProvenance.STRUCTURE_CLASS_GRID_FLAGGED: PageDisposition(
        PageEnding.MODEL_OUTPUT, PagePrimaryReason.STRUCTURE_CLASS
    ),
    SelectionProvenance.STRUCTURE_CLASS_FLOOR: PageDisposition(
        PageEnding.FAIL_CLOSED_MARKER, PagePrimaryReason.STRUCTURE_CLASS
    ),
    SelectionProvenance.NATIVE_FALLBACK: PageDisposition(
        PageEnding.DEMOTED_NATIVE, PagePrimaryReason.DEMOTED_NATIVE_RECOVERY_EXHAUSTION
    ),
    SelectionProvenance.NATIVE_CLEAN: PageDisposition(
        PageEnding.NATIVE_PROSE, PagePrimaryReason.CLEAN_NATIVE_PROSE
    ),
    SelectionProvenance.WHOLE_DOC_SECTION: PageDisposition(
        PageEnding.MODEL_OUTPUT, PagePrimaryReason.WHOLE_DOCUMENT_SECTION
    ),
    SelectionProvenance.BEST_OUTPUT_UNVERIFIED: PageDisposition(
        PageEnding.MODEL_OUTPUT, PagePrimaryReason.UNACCEPTED_OUTPUT_KEPT
    ),
    SelectionProvenance.BEST_ATTEMPT_FLAGGED: PageDisposition(
        PageEnding.MODEL_OUTPUT, PagePrimaryReason.UNACCEPTED_OUTPUT_KEPT
    ),
    SelectionProvenance.NO_TEXT_MARKER: PageDisposition(
        PageEnding.FAIL_CLOSED_MARKER, PagePrimaryReason.NO_USABLE_OUTPUT
    ),
}


def provenance_to_disposition(provenance: SelectionProvenance) -> PageDisposition:
    """Total mapping from every private selection provenance member to its base PageDisposition."""
    return _PROVENANCE_TO_DISPOSITION[provenance]


@dataclass(frozen=True)
class _FinalizedPageRecord:
    """Authoritative per-page outcome combining output, disposition, and selection provenance."""

    output: PageOutput
    disposition: PageDisposition
    selection_provenance: SelectionProvenance


FinalizedPageRecord = _FinalizedPageRecord


def _select_page_output(
    state: DocumentState,
    page_num: int,
    whole_doc: _WholeDoc | None = None,
) -> PageOutput:
    """The PageOutput that should be frozen for this page.

    Thin wrapper over :func:`_select_page_output_tagged` that drops the
    selection provenance tag. Byte-identical to the pre-R7 function for every
    caller; callers that need to know WHICH ending shipped call the tagged
    form rather than re-deriving it.
    """
    return _select_page_output_tagged(state, page_num, whole_doc)[0]


def _select_page_output_tagged(
    state: DocumentState,
    page_num: int,
    whole_doc: _WholeDoc | None = None,
) -> tuple[PageOutput, SelectionProvenance]:
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
    # GH-271: the corrupt-equation lane intentionally remains non-passing because
    # syntax validation cannot establish mathematical fidelity.  It is nonetheless
    # the selected region hybrid: substituting ``p.native_text`` here would erase
    # the retained crop and restore the known-corrupt glyphs.  This narrow field is
    # set only by that lane; it is not a general licence to keep rejected outputs.
    math_hybrid = getattr(p, "corrupt_math_hybrid", None)
    math_hybrid_blocked_by_table = bool(
        p.is_structure_class()
        or p.native_table_structure_failed
        or getattr(p, "native_table_unverifiable", False)
        or getattr(p, "native_table_structure_defective", False)
        or getattr(p, "native_table_header_unattributed", False)
        or getattr(p, "scanned_table_evidence_failed", False)
    )
    if (
        math_hybrid is not None
        and not getattr(p, "native_rotated_text_shredded", False)
        and not math_hybrid_blocked_by_table
        and math_hybrid in p.attempts
        and (math_hybrid.engine or "") == "native+math"
    ):
        return replace(
            math_hybrid,
            status=PageStatus.WARNING,
            audit_passed=False,
            failure_mode=(
                FailureMode.AUDIT_FAILED
                if math_hybrid.failure_mode is FailureMode.NONE
                else math_hybrid.failure_mode
            ),
        ), SelectionProvenance.CORRUPT_MATH_HYBRID
    if p.best_output and p.best_output.audit_passed:
        # A passing NATIVE best_output that also carries a table-distrust flag
        # is a CONTRADICTION, and the contradiction must lose to the flag
        # rather than short-circuit past it. Two independent ways to reach it:
        #
        # * GH-151 B1: the flag is set AFTER best_output was assigned, so a
        #   page slips through with audit_passed=True -- the PP-7-R1 shape,
        #   where a flag the manifest does not read re-stamps SUCCESS and makes
        #   the gate inert.
        # * #214: the resume ledger's fingerprint has no source-version
        #   component, so a page marked terminal SUCCESS by an older build is
        #   restored verbatim, carrying audit_passed=True next to the flag.
        #
        # Falling through re-demotes it through the normal path below. A
        # passing NON-native best_output is unaffected and returns immediately.
        winning_engine = p.best_output.engine or ""
        native_distrusted = winning_engine.startswith("native") and (
            # S1/C1: a structure-class page's native reading may NEVER author
            # the grid, unconditionally -- not only when a distrust flag
            # happened to catch it. Subsumes the three flags below for any
            # page they could ever be set on, kept as an explicit OR for
            # pages this predicate reaches by some path C2 does not cover.
            p.is_structure_class()
            or getattr(p, "native_table_unverifiable", False)
            or getattr(p, "native_table_structure_defective", False)
            or getattr(p, "native_table_header_unattributed", False)
        )
        # #263: same contradiction, for a rotated page whose native layer is
        # confetti -- but scoped to ``_NATIVE_TEXT_LANES`` rather than the
        # ``native`` prefix. The table flags above are about a native TABLE
        # reconstruction, which only the native lane performs; this flag is
        # about ``native_text`` itself, and the chart lane ships that too.
        # Without the wider scope, ``--native-only`` routed the page to the
        # chart lane and its passing winner returned here before the
        # fail-closed floor below could run.
        native_text_shredded = winning_engine.startswith(_NATIVE_TEXT_LANES) and getattr(
            p, "native_rotated_text_shredded", False
        )
        if not (native_distrusted or native_text_shredded):
            return p.best_output, SelectionProvenance.PASSING_BEST_OUTPUT
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

        # B1 (#591): GH-520's four-condition coverage guard
        # (table_floor_text_for_source) cannot apply to this branch -- its
        # first condition requires detected_table_count > 0, but a page
        # reaches here (``not p.is_born_digital``) precisely because native
        # table DETECTION found nothing on it (measured: Fed 1989-11-14 p3,
        # detected_table_count=0, 0 detected bboxes) -- there is no detected
        # geometry to reconcile splice_all_table_regions's blocks against.
        #
        # #652 round 10 (Astra's ruling, 2026-09-10): NO attempt is spliced
        # here at all. The mechanical check that used to stand in for the
        # missing geometry -- does the attempt's vocabulary overlap the page's
        # native words? -- was a corroboration guard, and #652 is the record of
        # it failing that job in six different shapes. The last one closes the
        # question rather than narrowing it again: a page in this branch is
        # here BECAUSE something flagged a table on it, and a native layer with
        # no printed numeral and no detected bbox does not establish that the
        # table is absent -- a text-only Bank/Status table has neither, and its
        # own institution names were vouching for an invented sentence beside
        # the marker. Nothing available on this page distinguishes a table's
        # vocabulary from its prose's, so the attempt is refused without
        # consulting it: an OCR attempt whose audit already flagged
        # HALLUCINATION ships no prose from this branch.
        #
        # This costs the page no TEXT. What the branch ships is #649's native
        # recovery just below -- the page's own trusted text layer, flagged,
        # with every withheld band replaced in place by the marker -- or the
        # bare marker where even that cannot be proven. What is refused is the
        # MODEL's wording, which is the only thing the corroboration check ever
        # authorised. Model-prose salvage on such a page needs independent
        # source evidence for the region AND its transcription (#707), not
        # another vocabulary or geometry threshold.
        d3_text = None
        best_output_text = (p.best_output.text or "") if p.best_output else ""

        # #649: no attempt could be spliced -- on this page's own fixture
        # because the attempt emitted the table as column runs and authored no
        # markdown table at all, so there was no block to work around. The
        # marker then shipped ALONE and took the page's prose with it. Recover
        # that prose from the page's own trusted text layer instead, with the
        # withheld numeric bands replaced in place by the same marker. Returns
        # None whenever it cannot prove what it would be shipping, which
        # leaves the bare marker exactly as before.
        prose_recovered = False
        if d3_text is None and _is_restored_prose_recovery(p, page_num):
            # #649 round 2 (Astra): a page RESTORED from its terminal sidecar
            # has already shipped this recovery, and the words it was built
            # from are gone -- ``native_words`` is a live-run cache the sidecar
            # deliberately does not carry. Recomputing therefore returned None
            # and the finalized body was replaced by the bare marker: a
            # transient missing cache erased text that had already shipped.
            # The frozen result stands. It is not recomputed and not
            # second-guessed; the evidence that it IS this lane's own output
            # is the audit note socr wrote on it, which a model attempt cannot
            # forge, plus the banner in the bytes.
            d3_text = best_output_text
            prose_recovered = True
        if d3_text is None:
            d3_text = native_prose_floor_text(p, page_num, marker_line=d3_marker, png_ref=png_ref)
            prose_recovered = d3_text is not None

        if d3_text is None:
            d3_text = f"{d3_marker}\n\n{png_ref}" if png_ref else d3_marker

        return PageOutput(
            page_num=page_num,
            text=d3_text,
            status=PageStatus.ERROR,
            engine=p.best_output.engine if p.best_output else "qwen",
            audit_passed=False,
            # #649: the recovery is a fact about what shipped, so it is
            # recorded where the corpus reads it, not only in the bytes. The
            # page stays ERROR with its own failure mode either way -- prose
            # coming back does not mean the table was read.
            audit_notes=([SCANNED_PROSE_RECOVERED_NOTE] if prose_recovered else []),
            # The credential the restore path actually reads (see
            # ``_is_restored_prose_recovery``). The note above is for readers;
            # this is for the machine, and only this module sets it.
            scanned_prose_recovered=prose_recovered,
            # #658: this branch REBUILDS the shipped output from scratch, so a
            # fixed HALLUCINATION here overwrote the honest attempt-level reason
            # and the sidecar the corpus actually reads still said the model
            # invented the table. The floor is unchanged -- same marker text,
            # same ERROR, same provenance -- only the recorded cause follows the
            # page's own flag.
            failure_mode=(
                FailureMode.NO_WITNESS_BACKEND
                if getattr(p, "scanned_table_no_witness", False)
                else FailureMode.HALLUCINATION
            ),
        ), SelectionProvenance.UNVERIFIABLE_TABLE_SCANNED
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
            and (
                getattr(p, "native_table_unverifiable", False)
                # GH-200: TR-3 is blind by construction to header loss (the
                # 2026-08-15 hand judgement: 4/4 damaged pages). A header-only
                # defect satisfies native_table_structure_failed but never
                # native_table_unverifiable, so without this OR it fell
                # through to the native_is_fallback WARNING branch below and
                # SHIPPED the header-destroyed native table text.
                or getattr(p, "native_table_header_unattributed", False)
            )
            and bool(p.attempts)
        ):
            # #262: unless some attempt did author a grid, in which case the
            # failed-table marker would be the lossier outcome. This D3-specific
            # decision stays before the broader S1 structure-class branch; the
            # latter's reachability predicate mirrors this precondition order.
            kept_model = d3_floor_kept_model_output(p)
            if kept_model is not None:
                # The marker carried a PNG of the page so a human could SEE the
                # table it refused to transcribe. That backstop matters MORE
                # here, not less: what ships in the marker's place is a grid
                # every rung refused, and the image is the only way a reader can
                # check it. Kept text = model reading + in-body flag + the same
                # image ref the floor would have shipped.
                kept_text = kept_model.text.rstrip()
                kept_text = f"{kept_text}\n\n{d3_superseded_note(page_num)}\n"
                png_ref = getattr(p, "d3_floor_png_ref", "")
                if png_ref:
                    kept_text = f"{kept_text}\n{png_ref}\n"
                return replace(
                    kept_model,
                    text=kept_text,
                    status=PageStatus.WARNING,
                    audit_passed=False,
                    failure_mode=FailureMode.MODEL_TABLE_OVER_FAILED_FLOOR,
                ), SelectionProvenance.UNVERIFIABLE_TABLE_MODEL_KEPT

            # TR-3: D3 fail-closed floor. Try regional splice if ordinals/counts
            # are available; fall back to whole-page marker if isolation is
            # unprovable.
            #
            # GH-375: ``native_table_header_unattributed`` is page-level — there
            # is no per-table header identity. Regional splice of only TR-3
            # ordinals would ship a header-destroyed sibling as unmarked GFM.
            # Refuse isolation and replace every table (GH-90): every table is
            # untrusted, surrounding prose is not.
            native_text = p.native_text or ""
            d3_marker = f"[page {page_num} failed: unverifiable table — see image]"
            png_ref = getattr(p, "d3_floor_png_ref", "")

            d3_text = None
            header_unattributed = bool(getattr(p, "native_table_header_unattributed", False))
            if header_unattributed:
                d3_text = splice_all_table_regions(
                    native_text, marker_line=d3_marker, png_ref=png_ref
                )
            else:
                failed_ordinals = getattr(p, "native_table_unverifiable_ordinals", None)
                region_count = getattr(p, "native_table_region_count", None)
                identities = list(getattr(p, "native_table_region_identities", []) or [])
                # Identities are recorded by ``_verify_regions`` for every
                # examined region, so a missing or short list means the state
                # predates GH-375 (stale sidecar) or was never captured. An
                # ordinal splice without a 1:1 identity match is exactly the
                # equal-count swap hole — refuse it and take the whole-page
                # marker instead of an unverified splice.
                if (
                    failed_ordinals is not None
                    and region_count is not None
                    and len(identities) == region_count
                ):
                    d3_text = splice_failed_table_regions(
                        native_text,
                        failed_ordinals=failed_ordinals,
                        expected_count=region_count,
                        marker_line=d3_marker,
                        png_ref=png_ref,
                        region_identities=identities,
                    )

            if d3_text is None:
                d3_text = f"{d3_marker}\n\n{png_ref}" if png_ref else d3_marker

            return PageOutput(
                page_num=page_num,
                text=d3_text,
                status=PageStatus.ERROR,
                engine="native",
                audit_passed=False,
                failure_mode=FailureMode.NATIVE_TABLE_STRUCTURE_FAILED,
            ), SelectionProvenance.UNVERIFIABLE_TABLE_NATIVE

        # #263: rotated-shredded fail-closed floor. The native layer of a
        # rotated page can come back as one glyph run per line -- 177 chars
        # over 47 lines on the reference page, 32 of them two characters or
        # fewer. Those fragments are not a reading of the page: two independent
        # judges rated the shipped output unusable, and the caption they
        # encode is only recoverable by reversing and re-joining them, which
        # is a repair this floor deliberately does not attempt (a wrong
        # reading is worse than a missing one). So the page ships the marker
        # plus the page image, exactly like the D3 table floor above.
        #
        # Deliberately NOT gated on ``bool(p.attempts)``, following the GH-195
        # precedent immediately below: the damage is found during native
        # extraction, on a page that may never reach the OCR ladder at all
        # (``--native-only``), and the attempt gate would leave exactly those
        # pages stamped SUCCESS over confetti.
        if getattr(p, "native_rotated_text_shredded", False):
            shred_marker = f"[page {page_num} failed: rotated text extraction shredded — see image]"
            shred_png = getattr(p, "rotated_shred_png_ref", "")
            return PageOutput(
                page_num=page_num,
                text=f"{shred_marker}\n\n{shred_png}" if shred_png else shred_marker,
                status=PageStatus.ERROR,
                engine="native",
                audit_passed=False,
                failure_mode=FailureMode.NATIVE_TEXT_SHREDDED,
            ), SelectionProvenance.ROTATED_TEXT_SHREDDED

        # #259: a flagged-but-PRESENT model output stays the winner. Placed
        # AFTER the D3 floor above so a hard-fail still fails closed, and before
        # the native fallback below, which is the substitution being fixed.
        # Demotion is by ``status``/``failure_mode`` on a COPY -- selection is
        # settled at this point, and ``audit_passed`` on the frozen record keeps
        # the resume ledger re-OCRing the page exactly as the native fallback did.
        flagged_model = flagged_model_page_output(p)
        if flagged_model is not None:
            kept_text = flagged_model.text
            note = kept_table_flag_note(state, page_num, kept_text)
            if note:
                kept_text = f"{kept_text.rstrip()}\n\n{note}\n"
            return replace(
                flagged_model,
                text=kept_text,
                status=PageStatus.WARNING,
                audit_passed=False,
                failure_mode=(
                    FailureMode.MODEL_OUTPUT_FLAGGED
                    if flagged_model.failure_mode is FailureMode.NONE
                    else flagged_model.failure_mode
                ),
            ), SelectionProvenance.FLAGGED_MODEL_KEPT

        # S1: the general structure-class case (C2, tables only). Originally
        # scoped to "tables or equations"; BLOCKING 1 on #269's review found
        # this forced every equation-only page through a check
        # (``_grid_authored_attempt``) that asks for a markdown TABLE grid --
        # meaningless for an equation reading, which authors no grid at all.
        # That produced both directions of harm: a correct native equation
        # transcription got wrongly demoted to WARNING purely because no
        # attempt authored a *table* grid on a page that was never going to
        # have one, and a model attempt whose text coincidentally matched the
        # grid-shape check (stray pipe characters near a matrix or an
        # absolute-value bar) could ship over a fine native reading with
        # nothing to actually verify it. Narrowed to ``bool(p.has_tables)`` in
        # ``PageState.is_structure_class`` -- equation pages keep their
        # pre-existing native-fallback/R3 behaviour, untouched by S1.
        # Every branch above this point already handles the pages where a
        # native-distrust flag positively fired (the scanned floor, TR-3's D3
        # floor, #263's rotated-shredded floor, #259's flagged-model
        # substitution) -- what reaches here is the 2026-08-20 measurement's
        # actual bug: 7 of 8 losing pages set NO distrust flag at all,
        # because the winner-side chain up to this point (native_verifier,
        # source_evidence, header anchors) compares numeric multisets, and a
        # flattened table is multiset-identical to a correct one (that chain
        # was blind to the only thing broken). C1's rule is unconditional for
        # a structure-class page: native may not author the GRID, flag or no
        # flag -- WHEN a candidate to select between exists.
        #
        # R3 in its own words: "a structure-class page must run at least one
        # model rung before selection, or C1's rule has nothing to select
        # between." That is a guarantee about the 2-candidate case this
        # initiative measures (default agentic mode, where routing is now
        # fixed to always try a rung on a structure-class page -- see
        # ``_is_trusted_native_without_ocr``). It is explicitly NOT a mandate
        # to punish a page that never had a candidate to begin with:
        # ``--native-only``'s table-only OCR bypass is deliberately
        # unchanged by S1 (the spec's own "Open, needs the owner" question),
        # and GH-211 / GH-195-198 both have pre-existing, deliberately-tested
        # coverage of a clean structure-class native page shipping SUCCESS
        # when no OCR ladder ever ran for it at all -- there is nothing here
        # for C1 to have picked over. Gated on an attempt that is NOT itself
        # native-labelled (``native``, ``native+math``, ...): those exist
        # without any external rung ever having run, so their presence alone
        # must not trip this branch either.
        if _reaches_structure_class_branch(p):
            truncated_engines = structure_class_truncated_engines(p)
            if truncated_engines:
                state.events.extend(_truncated_candidate_events(page_num, truncated_engines))
            grid_winner = structure_class_grid_winner(p)
            if grid_winner is not None:
                # TICKET-A1b (#634) case (i)-b: this winner came from the
                # row-corroboration fallback (the strict grid-authored pool
                # was empty), so the doubt this candidate still carries must
                # be made visible before it ships -- per-row markers in the
                # text itself and an audit event naming the header and the
                # counts, not just a silent pass through the same two
                # endings case (i) uses.
                corroboration_detail = structure_class_grid_corroboration(p)
                if corroboration_detail is not None:
                    corroboration, region_kind, coverage_share = corroboration_detail
                    grid_winner = _apply_row_corroboration_disclosure(
                        state, page_num, grid_winner, corroboration, region_kind, coverage_share
                    )
                    # TICKET-A1c (#641): a corroboration-fallback winner never
                    # cleared the strict grid-authored pool, so nothing ever
                    # verified its HEADER binding -- only A1a's row check ran,
                    # and that check is blind to header/column identity by
                    # construction (it compares ordered NUMBER runs, not
                    # labels). The "clean pass ships untouched" rule two
                    # paragraphs below is stated for an ORDINARY grid-authored
                    # winner, whose header attribution the strict pool gate
                    # itself already vetted -- it does not hold for this
                    # candidate regardless of its own ``audit_passed``. Ship
                    # WARNING / ``HEADER_BINDING_UNVERIFIED`` unconditionally
                    # (never the undemoted PASSING ending) so a clean-passing
                    # corroboration winner cannot ship as a silent SUCCESS --
                    # exactly the gap A1b's own review left open.
                    header_text = "\n".join(
                        line
                        for layout in _table_block_layout(grid_winner.text or "")
                        for line in layout["header_lines"]
                    )
                    kept_text = grid_winner.text
                    note = kept_table_flag_note(state, page_num, kept_text)
                    if note:
                        kept_text = f"{kept_text.rstrip()}\n\n{note}\n"
                    return replace(
                        grid_winner,
                        text=kept_text,
                        status=PageStatus.WARNING,
                        audit_passed=False,
                        failure_mode=FailureMode.HEADER_BINDING_UNVERIFIED,
                        table_corroboration={
                            "engine": grid_winner.engine or "",
                            "bound": corroboration.bound,
                            "total": corroboration.total,
                            "share": corroboration.share,
                            "extra_numbers": list(corroboration.extra_numbers),
                            "skipped_native_rows": corroboration.skipped_native_rows,
                            "unbound_rows": [list(idxs) for idxs in corroboration.unbound_rows],
                            "corroboration_region": region_kind,
                            "coverage_share": coverage_share,
                            "header_text": header_text,
                        },
                    ), SelectionProvenance.STRUCTURE_CLASS_GRID_CORROBORATED
                # (i) a grid-authoring model attempt from ``p.attempts``, body
                # untouched, flagged only per its own status (S1 spec,
                # verbatim) WHEN it is a clean pass. MAJOR 7(a) on #269: a
                # ``grid_winner`` accepted only via the
                # ``REJECTION_AMBIGUOUS_DEFERRED`` allowlist
                # (``_grid_authored_attempt``) is a SOFT reject, not a clean
                # one -- shipping it unchanged left the page SUCCESS /
                # failure_mode NONE while the document-level bucket flipped
                # to AUDIT_FAILED, a direct contradiction at two different
                # surfaces of the SAME page. Demoted via a ``replace()`` copy
                # exactly like #259 does immediately above: status /
                # failure_mode on the COPY, never ``audit_passed`` on the
                # stored attempt (the #252 round-1 defect). A clean pass
                # (``audit_passed`` already True) ships exactly as before --
                # this adds nothing on top of an ordinary passing attempt.
                if grid_winner.audit_passed:
                    return grid_winner, SelectionProvenance.STRUCTURE_CLASS_GRID_PASSING
                kept_text = grid_winner.text
                note = kept_table_flag_note(state, page_num, kept_text)
                if note:
                    kept_text = f"{kept_text.rstrip()}\n\n{note}\n"
                return replace(
                    grid_winner,
                    text=kept_text,
                    status=PageStatus.WARNING,
                    audit_passed=False,
                    failure_mode=(
                        FailureMode.MODEL_OUTPUT_FLAGGED
                        if grid_winner.failure_mode is FailureMode.NONE
                        else grid_winner.failure_mode
                    ),
                ), SelectionProvenance.STRUCTURE_CLASS_GRID_FLAGGED
            # (iii) no attempt authored a grid (R3's model-rung guarantee
            # found nothing usable, or -- under --native-only -- no rung ran
            # at all). P2 / GH-317: ship the fail-closed floor -- the whole-page
            # failed-table marker plus the rendered PNG ref. No native byte
            # ships: the region count that would license a regional splice
            # is produced by the same parser it would validate (cold review
            # round 2), so isolation is unprovable and the page fails closed.
            floor_text = structure_class_floor_text(p, page_num)
            return PageOutput(
                page_num=page_num,
                text=floor_text,
                status=PageStatus.ERROR,
                engine="native",
                audit_passed=False,
                failure_mode=FailureMode.STRUCTURE_CLASS_LADDER_EXHAUSTED,
            ), SelectionProvenance.STRUCTURE_CLASS_FLOOR

        # An enhancement page (native layer known deficient) whose recovery was
        # tried and never passed ships native text
        # as a FALLBACK, not a success: flagged WARNING / audit_passed=False
        # so the manifest and run summary stop stamping silent reversions as
        # passing pages.
        # Union of every table-distrust flag: #211's TR-3 unverifiable mark and
        # GH-151 B1's grid-shape / header flags all demote the same way.
        native_table_defect = (
            p.native_table_structure_failed
            or getattr(p, "native_table_unverifiable", False)
            or getattr(p, "native_table_structure_defective", False)
            or getattr(p, "native_table_header_unattributed", False)
        )
        native_is_fallback = (
            p.needs_ocr_enhancement
            or native_table_defect
            or p.chart_asset_render_failed  # PP-7: render failure must stay WARNING
        ) and bool(p.attempts)
        # GH-195: a text-strategy grid that had to be REJECTED for destroying a
        # native numeric token demotes the page too. Deliberately NOT gated on
        # ``p.attempts``: the rejection happens during native extraction, on a
        # page that may never reach the OCR ladder at all, so the existing
        # conjunction would leave exactly those pages stamped SUCCESS.
        #
        # This is a status-only demotion of an output that is ALREADY the
        # selected winner — the text is the lossless word-geometry rebuild and is
        # unchanged. It is not the #252 mistake of flipping ``audit_passed`` on
        # ``best_output``, which is the winner-SELECTION flag and would discard a
        # page; by this point selection is settled and this synthetic output is
        # what ships either way.
        grid_rejected = bool(getattr(p, "text_grid_rejected", False))
        native_demoted = native_is_fallback or grid_rejected
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
            status=PageStatus.WARNING if native_demoted else PageStatus.SUCCESS,
            engine="native",
            audit_passed=not native_demoted,
            # GH-151 B1: the attempt-level PageOutput this synthetic page
            # replaces already carries FailureMode.NATIVE_TABLE_STRUCTURE_FAILED
            # (set at ``_score_per_page`` / the native ship sites) -- but that
            # attempt is not reachable here (best_output was cleared when it
            # was demoted). Re-derive the failure mode from the same flags
            # rather than silently defaulting to NONE, so the shipped page
            # matches the ticket's doneWhen at the surface that actually ships.
            failure_mode=(
                FailureMode.NATIVE_TABLE_STRUCTURE_FAILED
                if native_table_defect and native_is_fallback
                else FailureMode.NONE
            ),
        ), (
            SelectionProvenance.NATIVE_FALLBACK
            if native_demoted
            else SelectionProvenance.NATIVE_CLEAN
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
        ), SelectionProvenance.WHOLE_DOC_SECTION
    # A failed per-page attempt (content present, audit not passed) beats an
    # empty page so the manifest preserves what little we have.
    if p.best_output:
        return p.best_output, SelectionProvenance.BEST_OUTPUT_UNVERIFIED
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
            # GH-158: this page's text was produced by a model, and rebuilding
            # the output here dropped every field saying WHICH one -- so the
            # rejected-but-shipped page fingerprinted with no model identity and
            # a model swap could not invalidate it. The engine name alone does
            # not distinguish two tags of the same engine. (The whole-document
            # branch above has the same shape, but ``_WholeDoc`` carries no
            # provider fields to forward; that is a wider change and is not
            # silently claimed fixed here.)
            provider_id=attempt.provider_id,
            provider_model=attempt.provider_model,
            provider_backend=attempt.provider_backend,
        ), SelectionProvenance.BEST_ATTEMPT_FLAGGED
    # Nothing anywhere produced text: ship an EXPLICIT failure marker, never
    # a silent gap between page headers -- B1 / #591: unless native prose
    # outside a detected table survives the same GH-520 coverage guard the
    # structure-class floor and the withhold ending already apply, in which
    # case that prose ships and only the table region(s) marker.
    whole_page_marker = page_failed_marker(page_num)
    native_text = getattr(p, "native_text", "") or ""
    floor_text = table_floor_text_for_source(
        p, page_num, native_text, fallback_marker=whole_page_marker
    )
    # R7: the cascade must be single-return-per-ending (test_r7_winner_kind_tags.py),
    # so both outcomes share ONE return, differing only in which values they carry
    # -- not two returns tagged with the same SelectionProvenance member.
    prose_kept = floor_text != whole_page_marker and bool(native_text.strip())
    return PageOutput(
        page_num=page_num,
        text=floor_text if prose_kept else whole_page_marker,
        status=PageStatus.ERROR,
        engine="native" if prose_kept else "",
        audit_passed=False,
    ), SelectionProvenance.NO_TEXT_MARKER


_select_page_output_with_provenance = _select_page_output_tagged


def _apply_table_emission_guard(output: PageOutput, page_num: int) -> PageOutput:
    """Return *output* normalized and hard-failed on a GH-226 or GH-190 defect.

    GH-302. This is the LAST shipping backstop -- whole-document CLI attempts
    that never reach the agentic judge or the post-route recheck have nothing
    else between them and the reader. It ran ``table_emission_defect`` alone,
    and an empty but well-formed table is not an EMISSION defect, so GH-190's
    own fixture still shipped SUCCESS here.

    The content term is added; the shape term is deliberately NOT. Running the
    whole of ``table_output_defect`` was tried first and is too wide for this
    seam: ``structural_gate_fires`` turns pages that ``--native-only`` ships
    FLAGGED into hard failures, which is a routing change this ticket rules
    out. Shape keeps its existing owner. Neither term is a density rule.
    """
    from socr.tables.reconcile import table_content_defect, table_emission_defect

    text = output.text or ""
    if text is not output.text:
        output = replace(output, text=text)
    marker = _TABLE_EMISSION_FAILED_RE.fullmatch(text.strip())
    emission_defect = None if marker else table_emission_defect(text)
    # GH-302 review: the content term must NOT take the text-replacing branch.
    # An emission defect means the markdown itself is malformed, so replacing
    # the page with a marker loses nothing that could be trusted. An empty
    # table is different: `table_content_defect` fires on ONE table run, while
    # the replacement is whole-page, so a page carrying real prose beside an
    # empty table had ALL of it swapped for the marker. That is a content loss
    # introduced by a no-content-loss fix -- the page must be DEMOTED, not
    # discarded (cf. #252: never destroy a page to flag it).
    content_defect = None if (marker or emission_defect) else (table_content_defect(text) or None)
    defect = marker.group("defect") if marker else (emission_defect or content_defect)
    if not defect:
        return output

    detail = f"invalid final table emission: {defect}"
    return replace(
        output,
        text=(
            text
            if content_defect
            else (
                text.strip()
                if marker
                else f"[page {page_num} failed: invalid table emission — {defect}]"
            )
        ),
        status=PageStatus.ERROR,
        audit_passed=False,
        failure_mode=FailureMode.TABLE_EMISSION_INVALID,
        error=detail,
        audit_notes=[*output.audit_notes, detail],
    )


#: GH-353 C3: the two ladder terminals a page's disposition may carry. Kept as
#: its own frozenset (rather than reusing D3_SUPERSEDING_REJECTIONS's shape)
#: because these are OUTCOMES the ladder reducer writes on ``PageState``, not
#: an allowlist of soft-refusal dispositions on a single attempt.
_LADDER_TERMINAL_FAILURE_MODES: frozenset[FailureMode] = frozenset(
    {FailureMode.TABLE_REJECTED, FailureMode.TABLE_UNVERIFIED, FailureMode.TABLE_WITHHELD}
)


def _apply_ladder_disposition_guard(output: PageOutput, page_num: int, p) -> PageOutput:
    """Enforce the table-judge ladder's page-level disposition (GH-353 C3).

    ``_select_page_output_tagged`` has many endings, and several of them --
    the native-only reconstruction chief among them (the ``NATIVE_CLEAN``
    ending) -- ship plain SUCCESS / ``audit_passed=True`` whenever no OTHER
    distrust flag happened to fire on THIS page. The ladder's REJECTED /
    UNVERIFIED verdict is judged AFTER routing (B1, not yet wired), so no
    cascade branch above can see it and a rejected table could otherwise be
    reconstructed as clean SUCCESS downstream of selection. Read via
    ``getattr`` with no default flag on ``PageState`` yet -- B1 owns adding
    and setting the attribute; until then this is inert for every page.

    Semantics: a REJECTED/UNVERIFIED candidate can still lose SELECTION to a
    better attempt -- this guard never touches which text ships, only the
    PAGE's final status/audit flag. What it forbids is the page regaining
    SUCCESS while its disposition says otherwise. An output already demoted
    for some other, more specific reason keeps that reason; the disposition's
    own failure mode is written only when nothing more specific already
    explains the demotion (e.g. GH-226's table-emission guard, applied first,
    wins on its own more precise diagnosis).
    """
    disposition = getattr(p, "table_ladder_disposition", None)
    if disposition not in _LADDER_TERMINAL_FAILURE_MODES:
        return output

    if disposition is FailureMode.TABLE_WITHHELD:
        # P1 (owner ruling Q2). REJECTED and UNVERIFIED are LABELS: the text
        # ships, demoted. WITHHELD is not -- the readers refused this table
        # and neither guard cleared it, so its bytes do not ship at all.
        #
        # Rewritten HERE, in the guard that already runs before
        # ``finalized_page_records`` / ``canonical_page_texts``, so the saved
        # .md, the page fragment, the sidecar's winning_output, the manifest
        # blob and replay all see the same bytes. The splice reads the
        # SELECTED output's text -- never ``p.native_text``, which on a model
        # winner is a different page.
        #
        # Prose survives only under GH-520's four coverage conditions;
        # otherwise the whole page floors, exactly as the structure-class
        # floor does, because an unenumerable table cannot be shown to have
        # been covered by the splice.
        text = output.text or ""
        from socr.tables.reconcile import find_table_blocks

        already_withheld = f"failed: unverifiable table" in text and not find_table_blocks(text)
        # A page that ALREADY shipped the fail-closed floor (the structure-class
        # floor reached the same page first) carries no table bytes and may have
        # legitimately kept its prose through GH-520's own coverage proof.
        # Re-running the coverage check over that already-spliced text sees zero
        # table blocks, fails the count condition, and floors the whole page --
        # destroying prose the floor was entitled to keep. Nothing is left to
        # withhold, so only the label changes.
        return replace(
            output,
            text=text if already_withheld else table_floor_text_for_source(p, page_num, text),
            status=PageStatus.ERROR,
            audit_passed=False,
            failure_mode=disposition,
        )

    if output.audit_passed:
        return replace(
            output,
            status=PageStatus.WARNING,
            audit_passed=False,
            failure_mode=disposition,
        )
    # HEAD's condition, restored (cold review round 2, finding 2). Widening this
    # to ``or output.failure_mode in _LADDER_TERMINAL_FAILURE_MODES`` overwrites an
    # already-recorded terminal with a different one -- an output carrying
    # TABLE_REJECTED on a page whose ladder disposition is TABLE_UNVERIFIED came
    # out TABLE_UNVERIFIED -- which changes the shipped output, the sidecar, the
    # manifest blob and the retry semantics. Stage A/B preserves behaviour.
    if output.failure_mode is FailureMode.NONE:
        return replace(output, failure_mode=disposition)
    return output


def _apply_label_unverified_guard(output: PageOutput) -> PageOutput:
    """Apply the #659 label-unverified REPORTING status, post-selection only.

    Astra review round 2 (P1): ``SourceEvidenceTableJudge`` used to set
    ``output.status = WARNING`` itself, at judge time, before handing the SAME
    output to the inner judge (``HeuristicPageJudge`` / ``VLMPageJudge``).
    Both treat any non-SUCCESS status as empty/error input, so the mutation
    rejected the very candidate the ticket exists to ship -- the ladder
    escalated or fell back instead of shipping the flagged table. The judge now
    only sets the DATA field (``table_label_unverified``) and leaves status
    alone; this guard applies the reporting WARNING here, after selection has
    already chosen this output and no judge will see it again -- the same
    post-selection-guard shape ``_apply_ladder_disposition_guard`` above and
    the #658 mode-carry use for a status decided after the fact.

    Never overrides a MORE severe status: a page already ERROR (a harder
    failure) or already WARNING for some other reason keeps it. Only a plain
    SUCCESS candidate whose only doubt is this label gets demoted.
    """
    if not output.table_label_unverified:
        return output
    if output.status is not PageStatus.SUCCESS:
        return output
    return replace(output, status=PageStatus.WARNING)


def _apply_ditto_guard(output: PageOutput, page_num: int) -> PageOutput:
    """#625: detect ditto-mark cells in the FINALIZED table text, post-selection.

    Same post-selection-guard shape as ``_apply_label_unverified_guard`` and
    ``_apply_unresolved_math_guard`` above, but with no judge behind it: the
    ditto mark is a property of the shipped BYTES, so detection runs here,
    directly on ``output.text``, exactly once selection has settled which
    candidate ships. Sets the data field
    (``PageOutput.table_ditto_columns``) unconditionally when found, and --
    like the two guards above -- only ever turns SUCCESS into WARNING; a page
    already ERROR or WARNING for a more specific reason keeps that status. The
    cell text itself is never touched (owner ruling, #625: no fill-down).
    """
    from socr.tables.ditto import detect_ditto_columns

    columns = detect_ditto_columns(output.text or "", page_num)
    if not columns:
        return output
    data = [c.to_dict() for c in columns]
    if output.status is PageStatus.SUCCESS:
        return replace(output, status=PageStatus.WARNING, table_ditto_columns=data)
    return replace(output, table_ditto_columns=data)


#: The marker families socr itself authors, keyed by the prose each builder emits
#: after ``failed: ``. Cold review round 2, finding 3: the ending must be read from
#: the SHIPPED BYTES through the one shared recogniser (``is_page_failed_marker``),
#: never from selection provenance alone -- a page whose body is
#: ``[page 1 failed: timeout during extraction]`` reached ``BEST_OUTPUT_UNVERIFIED``
#: and was published as ``(MODEL_OUTPUT, UNACCEPTED_OUTPUT_KEPT)``, which is exactly
#: the misclassification the public contract exists to close.
#:
#: The table names the family so the reason can say WHICH marker shipped. An
#: unrecognised family is not a hole: it falls back to ``SHIPPED_FAILURE_MARKER``,
#: which still says fail-closed. ``tests/test_p6_disposition_contract.py`` asserts
#: every family the tree can build is listed here.
_MARKER_FAMILY_REASONS: tuple[tuple[str, "PagePrimaryReason"], ...] = (
    ("invalid table emission", PagePrimaryReason.INVALID_TABLE_EMISSION),
    ("rotated text extraction shredded", PagePrimaryReason.ROTATED_NATIVE_TEXT_SHREDDED),
    ("no usable OCR output", PagePrimaryReason.NO_USABLE_OUTPUT),
    # The unverifiable-table family is DELIBERATELY absent. Its marker prose does
    # not say which lane distrusted the table, and every path that authors it --
    # the two D3 floors and the structure-class floor -- already carries a
    # FAIL_CLOSED_MARKER provenance whose lane-specific reason is kept above. A
    # page reaching the fallback with those bytes is therefore one socr genuinely
    # cannot attribute, and SHIPPED_FAILURE_MARKER says exactly that. Inventing a
    # fourth member to name a lane the bytes do not carry would be worse.
)


def _shipped_marker_reason(text: str) -> PagePrimaryReason:
    """Which marker family the shipped bytes are, as a primary reason."""
    marker = text.strip().splitlines()[0].strip() if text.strip() else ""
    body = marker.partition("failed:")[2].strip().rstrip("]").strip()
    for prose, reason in _MARKER_FAMILY_REASONS:
        if body.startswith(prose):
            return reason
    return PagePrimaryReason.SHIPPED_FAILURE_MARKER


def _apply_chart_region_guard(output: PageOutput, p) -> PageOutput:
    """GH-189: a page whose chart region was lost or unplaceable is not clean.

    Status-only, on the FINALIZED copy. The attempt is untouched and selection
    is settled by the time this runs, so this can neither reroute the page nor
    discard its text -- the #252 mistake was flipping ``audit_passed`` on
    ``best_output``, the winner-SELECTION flag, and that is deliberately not
    done here either. What it forbids is a page serializing SUCCESS beside a
    chart that is gone, sits at an unestablished position, or was never checked.

    A page already demoted for a more specific reason keeps that status: this
    only ever prevents a clean SUCCESS, it never upgrades or re-diagnoses.
    """
    if output.status is not PageStatus.SUCCESS:
        return output
    if not (
        getattr(p, "chart_region_render_failed", False)
        or getattr(p, "chart_region_placement_unresolved", False)
        or getattr(p, "chart_region_inventory_failed", False)
    ):
        return output
    return replace(output, status=PageStatus.WARNING)


def _apply_unresolved_math_guard(output: PageOutput, p) -> PageOutput:
    """#165: demote a page whose detected math-glyph damage survived into its body.

    A REPORTING guard, not a routing one. It runs after selection because the
    question it answers is about the bytes that ship: a recovery the selector
    discarded covered nothing, and a page can only be judged on the copy that
    wins. Nothing here re-selects -- the text, engine, provenance, table
    disposition, ``audit_passed`` and any existing ``failure_mode`` are all
    carried through untouched, so no candidate changes rank because of it. The
    document-level demotion is carried by the explicit unresolved-math page set
    in ``_phase_assemble``, not by a new ``FailureMode`` the ladder would read.

    SUCCESS becomes WARNING; an existing WARNING or ERROR is already at least as
    loud and is left alone. Idempotent, because every finalization seam
    (provisional records, assemble pre-records, final body records, terminal
    sidecars, manifest replay) runs it again on its own output.
    """
    from socr.math.accounting import unresolved_math_detail

    detail = unresolved_math_detail(
        has_unmapped_math_glyphs=bool(getattr(p, "has_unmapped_math_glyphs", False)),
        evidence=getattr(p, "math_recovery_evidence", None),
        text=output.text or "",
    )
    if detail is None:
        return output
    notes = list(output.audit_notes or [])
    if detail.detail not in notes:
        notes.append(detail.detail)
    if output.status is PageStatus.SUCCESS:
        return replace(output, status=PageStatus.WARNING, audit_notes=notes)
    return replace(output, audit_notes=notes)


def _select_and_finalize_page(
    state: DocumentState,
    page_num: int,
    whole_doc: _WholeDoc | None = None,
    saved_text: str | None = None,
) -> _FinalizedPageRecord:
    """Select and finalize a single page through guards, producing one authoritative record.

    Performs, in order:
      1. _select_page_output_with_provenance (the unchanged 16-way selector)
      2. Optional saved-body text replacement
      3. _apply_table_emission_guard
      4. _apply_ladder_disposition_guard
      5. _apply_unresolved_math_guard
      6. _apply_label_unverified_guard
      7. _apply_ditto_guard
      8. _apply_chart_region_guard
      9. Disposition construction from the guarded output and provenance.
    """
    output, provenance = _select_page_output_with_provenance(state, page_num, whole_doc)
    if saved_text is not None:
        output = replace(output, text=saved_text)
    output = _apply_table_emission_guard(output, page_num)
    p = state.pages.get(page_num)
    if p is not None:
        output = _apply_ladder_disposition_guard(output, page_num, p)
        output = _apply_unresolved_math_guard(output, p)
    output = _apply_label_unverified_guard(output)
    output = _apply_ditto_guard(output, page_num)
    if p is not None:
        # Last of the three status-only guards. Order among them is immaterial --
        # each only ever turns SUCCESS into WARNING and none of them upgrades --
        # but it is fixed here so the chain reads in one direction.
        output = _apply_chart_region_guard(output, p)

    text = (output.text or "").strip()

    # The BASE disposition -- what this page would be called if nothing about the
    # final bytes said otherwise. A page restored from a terminal sidecar takes the
    # base its ORIGINAL run published, because resume rebuilds ``p.attempts`` as the
    # single frozen winner and recomputing here would answer a question about the
    # reconstruction rather than about the run that shipped the bytes.
    #
    # This is a base, and only a base (cold review round 3). It is applied HERE,
    # before the byte-derived classification below, so a guard that rewrote the
    # CURRENT shipped bytes still wins: a restored value may stabilise a resume that
    # changed nothing, and may never outrank what the page actually ships now.
    base = provenance_to_disposition(provenance)
    restored = getattr(p, "resumed_disposition", None) if p is not None else None
    if restored:
        try:
            base = PageDisposition.from_dict(restored)
        except (KeyError, ValueError, TypeError):
            logger.debug("P6: unreadable persisted disposition on p%d; recomputing", page_num)

    withheld = (
        p is not None and getattr(p, "table_ladder_disposition", None) is FailureMode.TABLE_WITHHELD
    )
    if withheld and base.primary_reason is not PagePrimaryReason.STRUCTURE_CLASS:
        # P1 (owner ruling Q2). Named BEFORE the byte-derived branches below,
        # because those cannot see this cause. A whole-page withhold falls into
        # the generic marker family (SHIPPED_FAILURE_MARKER, "socr cannot say
        # which lane"), and a REGIONAL withhold is not a marker page at all --
        # it would keep selection's own base, publishing a page whose table was
        # deliberately removed as an ordinary accepted model output. socr knows
        # exactly why these bytes look like this, so it says so.
        #
        # A page that ALSO shipped the structure-class floor is the exception
        # above: it keeps ``STRUCTURE_CLASS``, which is the more specific
        # diagnosis (no attempt authored a grid at all) and -- load-bearing --
        # is what the resume gate reads to refuse the content-terminal skip for
        # a floored page. The failure mode still says TABLE_WITHHELD, and the
        # page still lands in the withheld bucket and note, both of which read
        # ``PageState.table_ladder_disposition`` first.
        disposition = PageDisposition(
            ending=PageEnding.FAIL_CLOSED_MARKER,
            primary_reason=PagePrimaryReason.TABLE_JUDGE_WITHHELD,
        )
    elif _TABLE_EMISSION_FAILED_RE.fullmatch(text):
        # The emission guard's own rewrite outranks whatever branch selected the
        # page: the body it replaced is gone, and the defect it names is the most
        # specific true statement about what shipped.
        disposition = PageDisposition(
            ending=PageEnding.FAIL_CLOSED_MARKER,
            primary_reason=PagePrimaryReason.INVALID_TABLE_EMISSION,
        )
    elif is_page_failed_marker(text):
        # Any OTHER recognised whole-page marker. When selection already knew the
        # page was fail-closed its reason is kept -- it is strictly more specific
        # than the family the bytes can reveal. When it did not, the bytes win and
        # the reason names the marker.
        disposition = (
            base
            if base.ending is PageEnding.FAIL_CLOSED_MARKER
            else PageDisposition(
                ending=PageEnding.FAIL_CLOSED_MARKER,
                primary_reason=_shipped_marker_reason(text),
            )
        )
    else:
        disposition = base

    return _FinalizedPageRecord(
        output=output,
        disposition=disposition,
        selection_provenance=provenance,
    )


def finalized_page_record(
    state: DocumentState,
    page_num: int,
    whole_doc: _WholeDoc | None = None,
    saved_text: str | None = None,
) -> FinalizedPageRecord:
    """Select and finalize a single page through guards, producing one authoritative record."""
    return _select_and_finalize_page(state, page_num, whole_doc=whole_doc, saved_text=saved_text)


def page_disposition(
    state: DocumentState,
    page_num: int,
    whole_doc: _WholeDoc | None = None,
) -> PageDisposition:
    """Public, finalization-aware page outcome (ending + normalized primary cause)."""
    return _select_and_finalize_page(state, page_num, whole_doc=whole_doc).disposition


def finalized_page_records(
    state: DocumentState,
    saved_body: str | None = None,
) -> list[_FinalizedPageRecord]:
    """Compute exactly one finalized page record per page in one pass."""
    saved_pages = split_native_pages(saved_body) if saved_body is not None else None
    whole_doc = _whole_doc_page_texts(state)
    records: list[_FinalizedPageRecord] = []
    for page_num in range(1, state.handle.page_count + 1):
        saved_text = (
            saved_pages[page_num - 1]
            if saved_pages is not None and page_num - 1 < len(saved_pages)
            else None
        )
        record = _select_and_finalize_page(
            state,
            page_num,
            whole_doc=whole_doc,
            saved_text=saved_text,
        )
        records.append(record)
    return records


def _winning_page_output(
    state: DocumentState,
    page_num: int,
    whole_doc: _WholeDoc | None = None,
) -> PageOutput:
    """Select and final-validate the exact page text that will ship."""
    return _select_and_finalize_page(state, page_num, whole_doc=whole_doc).output


def finalized_page_outputs(
    state: DocumentState,
    saved_body: str | None = None,
) -> list[PageOutput]:
    """Page outputs matching the exact body that ships, with final guards."""
    return [rec.output for rec in finalized_page_records(state, saved_body=saved_body)]


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


def canonical_page_texts(
    state: DocumentState,
    records: list[FinalizedPageRecord] | None = None,
) -> list[str]:
    """Per-page winning texts for the document, length == ``handle.page_count``.

    The SINGLE source of truth for both the saved ``.md`` body and the manifest
    blobs, so the saved document and ``replay`` are bit-consistent. Each entry is
    the winning page's text selected exactly as :func:`_winning_page_output`
    selects it (passing OCR > native > best attempt > whole-doc split), with any
    pre-existing leading ``## Page N`` header stripped so ``assemble_pages`` adds
    exactly one canonical header per page. ``split_native_pages`` round-trips it.
    """
    outputs = (
        [rec.output for rec in records] if records is not None else finalized_page_outputs(state)
    )
    return [_strip_leading_page_marker(page.text) for page in outputs]


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
    records: list[FinalizedPageRecord] | None = None,
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

    ``records`` is the optional internal finalization snapshot. The assemble
    phase passes the records it already computed for the exact saved body so
    manifest construction cannot select or guard those pages a second time.
    Direct callers omit it and get one record per page from
    :func:`finalized_page_records`, using the same seam.
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
    has_page_timings = any(getattr(ps, "timings_s", None) for ps in state.pages.values())
    manifest = Manifest(
        pdf_filename=handle.filename,
        pdf_file_hash=handle.file_hash,
        page_count=handle.page_count,
        render_dpi=dpi,
        agentic_ladder=state.agentic_ladder if state.agentic_ladder else None,
        agentic_judge_model=getattr(state, "agentic_judge_model", ""),
        timings_s=rollup_page_timings(state) if has_page_timings else None,
    )
    # Recover/finalize every exact page body once so the manifest, cache and
    # replay cannot diverge from the saved Markdown or its failure status. The
    # assemble phase supplies its already-finalized snapshot; direct callers
    # use this function as the single computation seam.
    frozen_records = records if records is not None else finalized_page_records(state, saved_body)
    if len(frozen_records) != handle.page_count:
        raise ValueError(
            "manifest finalization records must contain exactly one record per page "
            f"(got {len(frozen_records)}, expected {handle.page_count})"
        )
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
        record = frozen_records[page_num - 1]
        page = record.output
        blob_ref = blobs.put_page(page)
        image_hash = ""
        if page.engine and page.engine != "native":
            image_hash = compute_image_hash(handle, page_num, dpi)

        # Resolve the run determinants for this page's engine (consensus-aware).
        base_engine = _base_engine_name(page.engine)
        determinants = fingerprint_inputs.get(base_engine) or fingerprint_inputs.get(page.engine)
        # GH-158: the page's OWN resolved model is the last fallback, and until
        # now it was not consulted at all. An agentic page carries
        # ``provider_model`` (it is already written into the journal two blocks
        # below), yet the fingerprint took the model only from the caller's
        # ``fingerprint_inputs`` or from a doc-level ``EngineResult`` -- neither
        # of which exists on a per-page provider run. So a page whose model was
        # recorded correctly still fingerprinted with ``model_version=""``, and
        # swapping the model tag left ``replay`` believing the cached page was
        # still valid. Reading it from the page cannot invent identity: it is
        # empty exactly when the page had no model (a native page), which is the
        # honest value there -- no sentinel string, because "no model ran" and
        # "the model is called n/a" must not be the same record.
        # The page's OWN resolved model outranks every engine-level source
        # (cubic P2 on #507). ``determinants`` and ``EngineResult.model_version``
        # describe what was CONFIGURED for an engine; ``provider_model`` records
        # what actually ran on this page. An agentic run can escalate a single
        # page to a different rung, and taking the configured value there would
        # fingerprint the page under a model that never read it -- the precise
        # failure this ticket is named for.
        page_model = getattr(page, "provider_model", "") or ""
        prompt_hash = ""
        if page.engine == "native":
            # A native page had NO model, and every source below describes some
            # other engine's model (cubic P2 on #507). On a mixed document the
            # OCR engine's `EngineResult.model_version` is populated, so without
            # this the native pages were stamped with a model that never read
            # them -- erasing the very distinction this ticket argues the empty
            # value exists to preserve. Short-circuit before any of them.
            model_version = ""
        elif determinants is not None:
            model, backend, task, prompt = determinants
            model_version = (
                page_model
                or model
                or model_versions.get(base_engine)
                or model_versions.get(page.engine, "")
            )
            prompt_hash = run_fingerprint(model_version, backend, task, prompt)
        else:
            model_version = (
                page_model or model_versions.get(base_engine) or model_versions.get(page.engine, "")
            )

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
                # GH-169: skip_reason first (the rung was never tried), then the
                # judge's verdict, then the failure mode. Previously a rejected
                # non-empty attempt fell straight through to the mode and read
                # "none".
                "reason": (
                    getattr(a, "skip_reason", "")
                    or getattr(a, "judge_reason", "")
                    or a.failure_mode.value
                ),
                "judge_model": _judge_model,
            }
            for a in state.pages[page_num].attempts
        ]
        manifest.entries[page_num] = ManifestEntry(
            page_num=page_num,
            blob_ref=blob_ref,
            fingerprint=fp,
            journal=journal,
            disposition=record.disposition,
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


def splice_all_table_regions(
    page_text: str,
    marker_line: str,
    png_ref: str = "",
) -> str | None:
    """Replace every parsed markdown table, preserving surrounding prose.

    GH-90 scanned floor distrusts every model table. GH-375 native D3 floor
    does the same when ``native_table_header_unattributed`` is set: that flag
    is page-level, so no table on the page can be named as clean. Isolation
    here is the parser's own block list — ordinals are not drawn from a
    second enumeration, so an equal-count swap cannot arise.

    Returns None when no table block can be identified (caller falls back to
    the whole-page marker).
    """
    from socr.tables.reconcile import find_table_blocks

    if not page_text:
        return None
    blocks = find_table_blocks(page_text)
    if not blocks:
        return None
    return splice_failed_table_regions(
        page_text,
        failed_ordinals=list(range(len(blocks))),
        expected_count=len(blocks),
        marker_line=marker_line,
        png_ref=png_ref,
    )


def splice_failed_table_regions(
    page_text: str,
    failed_ordinals: list[int],
    expected_count: int,
    marker_line: str,
    png_ref: str = "",
    region_identities: list[str] | None = None,
) -> str | None:
    """Replace only failed markdown-table blocks in page text, preserving surrounding prose.

    GH-371: when the D3 fail-closed floor (or GH-90 scanned floor) fires on a
    native table region, preserve the page's surrounding prose by splicing out only
    the failed table blocks instead of replacing the entire page with a marker.

    Args:
        page_text: Full page markdown text containing tables and prose.
        failed_ordinals: Zero-based ordinal indices of table blocks to remove.
        expected_count: Expected number of markdown table blocks in the text.
        marker_line: The visible failed-table marker to insert (e.g.
            "[page N failed: unverifiable table — see image]").
        png_ref: Optional full-page PNG reference (e.g. "![ref](figures/p1.png)").
            Included only once, at the first failed position (in document order).
        region_identities: Optional per-ordinal fingerprints of the y0-sorted
            native regions (``table_grid_identity`` of each region's grid).
            When supplied, each parsed block must match the corresponding
            identity; an equal-count swap fails closed to None.

    Returns:
        Spliced text with failed blocks removed and markers inserted, or None if
        validation fails (empty ordinals, count mismatch, out-of-range/duplicate
        ordinals, identity mismatch, or no parsed tables). Fail-closed: any
        ambiguity returns None.
    """
    from socr.tables.reconcile import find_table_blocks, table_grid_identity

    if type(expected_count) is not int or expected_count <= 0:
        return None

    if not failed_ordinals or not isinstance(failed_ordinals, list):
        return None

    if any(type(ordinal) is not int for ordinal in failed_ordinals):
        return None

    if len(failed_ordinals) != len(set(failed_ordinals)):
        return None

    if any(ordinal < 0 or ordinal >= expected_count for ordinal in failed_ordinals):
        return None

    blocks = find_table_blocks(page_text)
    if len(blocks) != expected_count:
        # Identity assumption: ordinal N in the caller's region enumeration
        # (born_digital's y0-sorted table_regions) names the Nth markdown table
        # block that find_table_blocks parses out of the assembled text. Count
        # equality is the first check tying the two enumerations together — a
        # divergence (a region emitted without pipes, prose that parses as a
        # table) normally breaks the count and lands here (fail closed).
        return None

    if region_identities is not None:
        # GH-375: count equality cannot see a permutation. When the caller
        # captured per-region fingerprints, require a 1:1 match against the
        # parsed blocks; a swap of two different tables fails closed.
        if (
            not isinstance(region_identities, list)
            or len(region_identities) != expected_count
            or any(type(item) is not str for item in region_identities)
            or any(
                table_grid_identity(block.grid) != ident
                for block, ident in zip(blocks, region_identities, strict=True)
            )
        ):
            return None

    lines = page_text.splitlines(keepends=True)

    first_failed_ordinal = min(failed_ordinals)
    for ordinal in sorted(failed_ordinals, reverse=True):
        block = blocks[ordinal]
        start_line = block.start
        end_line = block.end

        replacement = marker_line
        if ordinal == first_failed_ordinal and png_ref:
            replacement = f"{marker_line}\n\n{png_ref}"

        lines[start_line : end_line + 1] = [f"{replacement}\n"]

    result = "".join(lines).rstrip() + "\n"
    return result
