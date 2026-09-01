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

    A native-engine reading is never a candidate -- C1 forbids native from
    authoring the GRID on a structure-class page, full stop; native's PROSE
    ships separately, untouched, via the last-resort WARNING branch below.

    GH-268 centralizes "authored a grid" in ``has_strict_table_grid``. S1
    must use the same structural contract as the earlier D3 and #259 branches;
    otherwise an output one branch rejects can be selected by this later one.
    """
    if out is None:
        return False
    if (out.engine or "").startswith("native"):
        return False
    text = (out.text or "").strip()
    if not text or is_page_failed_marker(text):
        return False
    if out.status is PageStatus.ERROR or out.failure_mode is FailureMode.HALLUCINATION:
        return False
    if not (
        out.audit_passed or getattr(out, "rejection_class", "") == REJECTION_AMBIGUOUS_DEFERRED
    ):
        return False

    from socr.tables.reconcile import has_strict_table_grid

    return has_strict_table_grid(text)


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
    """
    if not _reaches_structure_class_branch(p):
        return None
    if _grid_authored_attempt(p.best_output):
        return p.best_output
    candidates = [out for out in p.attempts if _grid_authored_attempt(out)]
    if not candidates:
        return None
    return max(candidates, key=lambda out: (out.audit_passed, out.confidence, out.word_count))


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
    """P2 / GH-317: fail-closed text representation for an exhausted structure-class page.

    The WHOLE page becomes the standard unverifiable-table marker plus the
    existing ``d3_floor_png_ref``. No regional splice, and therefore no native
    prose: on this ending nothing on the page can be proven to be outside a
    table region.

    Cold review rounds 1 and 2, finding 1. Round 1 shipped a regional splice
    via ``splice_all_table_regions``, which proves only that it replaced every
    block ITS OWN PARSER could find -- so a page with two detected tables where
    reconstruction emitted one as valid GFM and collapsed the other to ragged
    lines replaced the parseable one and shipped the collapsed one. Round 1's
    fix required coverage against ``native_table_region_count`` /
    ``native_table_region_identities``, which does NOT close it: those are
    recorded by ``_verify_regions`` (born_digital.py:2186-2217) counting only
    separator-bearing regions of ``table_regions``, and ``table_regions`` is
    itself built only from SUCCESSFUL reconstructions (born_digital.py:1939-2003).
    A sibling that failed reconstruction is absent from the count, so the check
    agrees with the very parser it is meant to audit and passes.

    Splicing safely needs an INDEPENDENT, detection-level region count recorded
    BEFORE reconstruction. No such signal exists today: ``_detect_tables``
    (born_digital.py:1766-1804) reduces ``page.find_tables()`` to a bool and
    discards the count, and no ``PageAssessment``/``PageState`` field carries a
    detection-level count or bbox list. Until one exists, a regional splice on
    this ending cannot be justified, so it is gone. The cost -- native prose on
    a floored page -- is recorded as a known limitation in
    docs/log/2026-09-01_p2-structure-class-floor.md.
    """
    d3_marker = f"[page {page_num} failed: unverifiable table — see image]"
    png_ref = getattr(p, "d3_floor_png_ref", "")
    return f"{d3_marker}\n\n{png_ref}" if png_ref else d3_marker


class WinnerKind(str, Enum):
    """R7: which of ``_select_page_output_tagged``'s endings shipped this page.

    The cascade is 15 returns and **zero loops** (AST-verified), so exactly one
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


def _select_page_output(
    state: DocumentState,
    page_num: int,
    whole_doc: _WholeDoc | None = None,
) -> PageOutput:
    """The PageOutput that should be frozen for this page.

    Thin wrapper over :func:`_select_page_output_tagged` that drops the R7
    disposition tag. Byte-identical to the pre-R7 function for every caller;
    callers that need to know WHICH ending shipped call the tagged form rather
    than re-deriving it.
    """
    return _select_page_output_tagged(state, page_num, whole_doc)[0]


def _select_page_output_tagged(
    state: DocumentState,
    page_num: int,
    whole_doc: _WholeDoc | None = None,
) -> tuple[PageOutput, WinnerKind]:
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
        ), WinnerKind.CORRUPT_MATH_HYBRID
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
            return p.best_output, WinnerKind.PASSING_BEST_OUTPUT
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

        best_output_text = (p.best_output.text or "") if p.best_output else ""
        d3_text = splice_all_table_regions(best_output_text, marker_line=d3_marker, png_ref=png_ref)

        if d3_text is None:
            d3_text = f"{d3_marker}\n\n{png_ref}" if png_ref else d3_marker

        return PageOutput(
            page_num=page_num,
            text=d3_text,
            status=PageStatus.ERROR,
            engine=p.best_output.engine if p.best_output else "qwen",
            audit_passed=False,
            failure_mode=FailureMode.HALLUCINATION,
        ), WinnerKind.UNVERIFIABLE_TABLE_SCANNED
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
                ), WinnerKind.UNVERIFIABLE_TABLE_MODEL_KEPT

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
            ), WinnerKind.UNVERIFIABLE_TABLE_NATIVE

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
            ), WinnerKind.ROTATED_TEXT_SHREDDED

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
            ), WinnerKind.FLAGGED_MODEL_KEPT

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
            grid_winner = structure_class_grid_winner(p)
            if grid_winner is not None:
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
                    return grid_winner, WinnerKind.STRUCTURE_CLASS_GRID_PASSING
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
                ), WinnerKind.STRUCTURE_CLASS_GRID_FLAGGED
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
            ), WinnerKind.STRUCTURE_CLASS_FLOOR

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
        ), (WinnerKind.NATIVE_FALLBACK if native_demoted else WinnerKind.NATIVE_CLEAN)
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
        ), WinnerKind.WHOLE_DOC_SECTION
    # A failed per-page attempt (content present, audit not passed) beats an
    # empty page so the manifest preserves what little we have.
    if p.best_output:
        return p.best_output, WinnerKind.BEST_OUTPUT_UNVERIFIED
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
        ), WinnerKind.BEST_ATTEMPT_FLAGGED
    # Nothing anywhere produced text: ship an EXPLICIT failure marker, never
    # a silent gap between page headers.
    return PageOutput(
        page_num=page_num,
        text=page_failed_marker(page_num),
        status=PageStatus.ERROR,
        audit_passed=False,
    ), WinnerKind.NO_TEXT_MARKER


def _apply_table_emission_guard(output: PageOutput, page_num: int) -> PageOutput:
    """Return *output* normalized and hard-failed on a GH-226 defect."""
    from socr.tables.reconcile import table_emission_defect

    text = output.text or ""
    if text is not output.text:
        output = replace(output, text=text)
    marker = _TABLE_EMISSION_FAILED_RE.fullmatch(text.strip())
    defect = marker.group("defect") if marker else table_emission_defect(text)
    if not defect:
        return output

    detail = f"invalid final table emission: {defect}"
    return replace(
        output,
        text=(
            text.strip()
            if marker
            else f"[page {page_num} failed: invalid table emission — {defect}]"
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
    {FailureMode.TABLE_REJECTED, FailureMode.TABLE_UNVERIFIED}
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
    if output.audit_passed:
        return replace(
            output,
            status=PageStatus.WARNING,
            audit_passed=False,
            failure_mode=disposition,
        )
    if output.failure_mode is FailureMode.NONE:
        return replace(output, failure_mode=disposition)
    return output


def _finalize_page_output(state: DocumentState, output: PageOutput, page_num: int) -> PageOutput:
    """Apply every final-validation guard a selected page output must pass.

    Single seam for both callers below so the ladder disposition guard (C3)
    and the GH-226 table-emission guard can never drift apart between the
    manifest/replay path and the assembled-Markdown path -- the exact drift
    ``_reaches_structure_class_branch``'s own docstring warns against.
    """
    output = _apply_table_emission_guard(output, page_num)
    p = state.pages.get(page_num)
    return _apply_ladder_disposition_guard(output, page_num, p) if p is not None else output


def _winning_page_output(
    state: DocumentState,
    page_num: int,
    whole_doc: _WholeDoc | None = None,
) -> PageOutput:
    """Select and final-validate the exact page text that will ship.

    GH-226 puts the last table-emission guard here because this seam is shared
    by canonical fragments, assembled Markdown, manifest blobs, and replay,
    including whole-document CLI attempts that never pass through the agentic
    acceptance gate. Earlier checks still drive repair and routing; this one is
    the fail-closed backstop after those choices are exhausted. GH-353 C3 adds
    the ladder disposition guard at the same seam for the same reason.
    """
    return _finalize_page_output(
        state,
        _select_page_output(state, page_num, whole_doc),
        page_num,
    )


def finalized_page_outputs(
    state: DocumentState,
    saved_body: str | None = None,
) -> list[PageOutput]:
    """Page outputs matching the exact body that ships, with final guards.

    ``saved_body`` is post-transform Markdown (phantom-ref cleanup, figures,
    captions). When supplied, its per-page text overrides the selected text
    *before* the same GH-226 / GH-353 guards are applied, closing the
    manifest/replay and post-figure bypass without duplicating validation
    policy.
    """
    saved_pages = split_native_pages(saved_body) if saved_body is not None else None
    whole_doc = _whole_doc_page_texts(state)
    outputs: list[PageOutput] = []
    for page_num in range(1, state.handle.page_count + 1):
        output = _select_page_output(state, page_num, whole_doc)
        if saved_pages is not None and page_num - 1 < len(saved_pages):
            output = replace(output, text=saved_pages[page_num - 1])
        outputs.append(_finalize_page_output(state, output, page_num))
    return outputs


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
    return [_strip_leading_page_marker(page.text) for page in finalized_page_outputs(state)]


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
    # Recover/finalize every exact page body once so the manifest, cache and
    # replay cannot diverge from the saved Markdown or its failure status.
    frozen_pages = finalized_page_outputs(state, saved_body)
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
        page = frozen_pages[page_num - 1]
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
