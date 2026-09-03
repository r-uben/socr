"""Unified OCR pipeline orchestrator.

Drives DocumentState through analysis, cost-aware agentic per-page extraction,
and assembly. Replaces StandardPipeline's ad-hoc extraction stages with a
structured loop that operates on the DocumentState blackboard.
"""

from __future__ import annotations

import logging
import re
import tempfile
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from socr.math.detect_equations import EquationRegion

from rich.console import Console

from socr.audit.heuristics import HeuristicsChecker
from socr.audit.scorer import FailureModeScorer
from socr.core.born_digital import BornDigitalDetector, DocumentAssessment
from socr.core.audit_log import VISUAL_VALUES_NOT_TRANSCRIBED_KIND
from socr.core.config import EngineType, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.manifest import (
    FinalizedPageRecord,
    PageDisposition,
    PageEnding,
    PagePrimaryReason,
)
from socr.core.normalizer import (
    MAX_CONSECUTIVE_IDENTICAL_TABLE_ROWS,
    OutputNormalizer,
    collapse_repeated_table_rows,
)
from socr.core.providers import (
    ProviderProfile,
    execution_overrides,
    is_cloud_qwen,
    profile_by_id,
    profile_by_model,
    resolved_provenance,
)
from socr.core.result import (
    DocumentStatus,
    EngineResult,
    FailureMode,
    FigureInfo,
    PageOutput,
    PageStatus,
)
from socr.core.state import DocumentState, PageState, add_page_cost
from socr.engines.registry import get_engine, resolve_auto_engine
from socr.figures.extractor import ExtractionResult, FigureExtractor, has_chart_marks
from socr.judge.table_rung_gemini import gemini_rung_reachable as table_judge_gemini_rung_reachable
from socr.judge.table_rung_ollama import ollama_rung_reachable as table_judge_ollama_rung_reachable
from socr.judge.table_verdict import RUNG_KIND_GEMINI, RUNG_KIND_OLLAMA, rung_kind
from socr.pipeline.agentic import route_page
from socr.tables.extract import probe_ollama_idle, probe_openai_server_idle
from socr.tables.extract import resolve_ollama_host as _resolve_ollama_host

#: Exact PageDisposition contract pairs for the three migrated buckets (P6 stage C).
_MIGRATED_DISPOSITION_BUCKETS: dict[str, PageDisposition] = {
    "structure_class_model_pages": PageDisposition(
        PageEnding.MODEL_OUTPUT, PagePrimaryReason.STRUCTURE_CLASS
    ),
    "structure_class_floor_pages": PageDisposition(
        PageEnding.FAIL_CLOSED_MARKER, PagePrimaryReason.STRUCTURE_CLASS
    ),
    "corrupt_math_hybrid_pages": PageDisposition(
        PageEnding.MODEL_OUTPUT, PagePrimaryReason.CORRUPT_MATH_HYBRID
    ),
}


def _derive_disposition_buckets(
    state: DocumentState, records: list[FinalizedPageRecord]
) -> dict[str, set[int]]:
    """The six selection-shaped assemble buckets.

    **Three buckets are disposition-derived** (``structure_class_model_pages``,
    ``structure_class_floor_pages``, ``corrupt_math_hybrid_pages``). Each is keyed
    on exact equality with the public ``PageDisposition`` that shipped (see
    :data:`_MIGRATED_DISPOSITION_BUCKETS`). ``SelectionProvenance`` is never read for
    membership: a page whose candidate was rewritten by the post-selection emission
    guard ships ``FAIL_CLOSED_MARKER / INVALID_TABLE_EMISSION`` and is absent from all
    three buckets. A genuine structure-class floor ships
    ``FAIL_CLOSED_MARKER / STRUCTURE_CLASS`` and remains in
    ``structure_class_floor_pages``.

    **Three buckets stay flag-derived** (``d3_model_table_pages``, ``d3_floor_pages``,
    ``flagged_model_pages``). These are not questions about which branch won at all.
    Each is keyed on a native-lane verdict -- the D3 flag conjunction, or
    ``flagged_model_page_output`` -- that a page can carry while selection ends
    somewhere else entirely. The measured case is a page with the full D3 conjunction
    (or a flagged-model candidate) AND a passing non-native ``best_output``: selection
    returns at ``PASSING_BEST_OUTPUT`` long before either branch is reached, so the
    page carries the tag and the disposition an ordinary clean model page carries, and
    neither can recover it. Deriving these three from either vocabulary flipped such a
    document from AUDIT_FAILED to SUCCESS and dropped its D3 / flagged-model audit
    events, CLI lines and tables-trust note.

    Whether a native-lane verdict SHOULD outrank a passing winner is a real question,
    and a separate one from the disposition derivation above.
    """
    from socr.core.manifest import (
        d3_floor_kept_model_output,
        flagged_model_page_output,
    )

    struct_model: set[int] = set()
    struct_floor: set[int] = set()
    corrupt_math: set[int] = set()

    for r in records:
        pn = r.output.page_num
        disp = r.disposition
        if disp == _MIGRATED_DISPOSITION_BUCKETS["structure_class_model_pages"]:
            struct_model.add(pn)
        elif disp == _MIGRATED_DISPOSITION_BUCKETS["structure_class_floor_pages"]:
            struct_floor.add(pn)
        elif disp == _MIGRATED_DISPOSITION_BUCKETS["corrupt_math_hybrid_pages"]:
            corrupt_math.add(pn)

    # The three native-lane verdicts, unchanged from the pre-P6 assemble predicates.
    d3_model: set[int] = {
        n for n, p in state.pages.items() if d3_floor_kept_model_output(p) is not None
    }
    d3_floor: set[int] = {
        n
        for n, p in state.pages.items()
        if p.is_born_digital
        and p.native_table_structure_failed
        and (
            getattr(p, "native_table_unverifiable", False)
            or getattr(p, "native_table_header_unattributed", False)
        )
        and bool(p.attempts)
        and n not in d3_model
    }
    flagged_model: set[int] = {
        n for n, p in state.pages.items() if flagged_model_page_output(p) is not None
    }

    return {
        "d3_model_table_pages": d3_model,
        "d3_floor_pages": d3_floor,
        "flagged_model_pages": flagged_model,
        "structure_class_model_pages": struct_model,
        "structure_class_floor_pages": struct_floor,
        "corrupt_math_hybrid_pages": corrupt_math,
    }


logger = logging.getLogger(__name__)
console = Console()

# Provenance value recorded when no vision judge could be built and the page gate
# is the heuristic checker. A literal sentinel (not "") so the distinction between
# "heuristics judged this run" and "this field was never populated" survives into
# metadata.json and the run fingerprint (#133).
JUDGE_IDENTITY_HEURISTIC = "heuristic"


#: Baseline for the "you set something I ignore" warning (GH-525) -- and ONLY
#: that. These fields gate nothing (GH-142 rejected their CLI flags for it) and
#: are absent from the run fingerprint entirely, so there is nothing here to
#: freeze; `_warn_inert_config` diffs against these values to decide whether a
#: config is worth warning about.
#:
#: An earlier draft of GH-525 DID freeze them into the fingerprint, to avoid
#: changing it for existing runs. That reason turned out not to exist --
#: `_socr_source_digest` already invalidates every fingerprint on any source
#: edit, by design -- so the keys were dropped instead. Do not reintroduce the
#: freeze on the strength of a cost that is paid by every release anyway.
#:
#: Taken from PipelineConfig's own defaults rather than written out, so the two
#: cannot drift: a changed default moves both together.
def _inert_field_defaults() -> dict:
    from socr.core.config import PipelineConfig

    defaults = PipelineConfig()
    return {
        "judge_hard_pages": defaults.judge_hard_pages,
        "fallback_chain": list(defaults.fallback_chain),
    }


_INERT_FIELD_DEFAULTS = _inert_field_defaults()


def _warn_inert_config(cfg) -> list[str]:
    """Names of inert fields this config sets away from their defaults.

    Ignoring a setting silently is the failure this ticket family is about, so
    the run says which ones it ignored rather than leaving the operator to infer
    it from output that did not change.
    """
    differing = []
    if bool(cfg.judge_hard_pages) != bool(_INERT_FIELD_DEFAULTS["judge_hard_pages"]):
        differing.append("judge_hard_pages")
    if list(cfg.fallback_chain) != list(_INERT_FIELD_DEFAULTS["fallback_chain"]):
        differing.append("fallback_chain")
    return differing


def _page_blob_key(page_output_dict: dict) -> str:
    """Content-addressed key for a serialised PageOutput dict.

    Used in per-page sidecars as a lightweight ``page_fingerprint`` that
    changes whenever the winning page text changes.  Delegates to the
    BlobStore's own ``blob_hash`` so the canonicalisation is identical BY
    CONSTRUCTION and PP-5 can cross-reference with the manifest without
    opening the manifest file.  A re-implementation here diverged on
    ``ensure_ascii`` and disagreed with the store for every non-ASCII page
    (#235) — hence the delegation rather than a copied helper.

    The ``sha256:`` prefix is part of the sidecar's value shape and is NOT
    part of the store's key; only the digest below it is shared.
    """
    from socr.core.cache import blob_hash

    return "sha256:" + blob_hash(page_output_dict)


# GH-214: process-lifetime cache for the source digest. Computed once, not per page.
_SOURCE_DIGEST_CACHE: str | None = None


def _manifest_versions() -> tuple[str, str]:
    """(NORMALIZER_VERSION, ASSEMBLY_VERSION), read at call time.

    Late attribute access (not a from-import) so the run fingerprint always
    reflects the live values and tests can monkeypatch them.
    """
    from socr.core import manifest

    return manifest.NORMALIZER_VERSION, manifest.ASSEMBLY_VERSION


def _socr_version() -> str:
    """socr's declared package version, read at call time (monkeypatchable in tests)."""
    import socr

    return str(getattr(socr, "__version__", ""))


def _socr_source_digest() -> str:
    """SHA-256 over every shipped ``socr`` ``.py`` file, computed once per process.

    GH-214. ``_run_fingerprint`` records the run *configuration*, never socr's own
    code, so fixing an extraction bug without touching config leaves the fingerprint
    byte-identical: already-terminal pages pass the resume gate and the corrected
    code never reaches them. The document then reports SUCCESS carrying output the
    fix was supposed to replace.

    Hashing the whole package rather than a curated module list is deliberate. A
    curated list has to be maintained, and a module someone forgets to register
    fails DANGEROUS — it silently reuses stale output, which is the very bug this
    closes. Hashing everything can only fail SAFE: an output-neutral edit (a
    comment, a docstring) costs one needless reprocess. Re-OCR is cheap; a stale
    number in a citation corpus is not.

    ``normalizer_version`` / ``assembly_version`` (issue #38) are the narrow,
    hand-maintained form of this idea and stay where they are — they cover two
    modules, this covers the rest, and a stale hand-bumped constant is exactly the
    failure mode above.

    Determinism matters more than speed here: paths are sorted and recorded
    package-relative (so two checkouts of the same code agree), and generated
    artefacts are excluded.
    """
    global _SOURCE_DIGEST_CACHE
    if _SOURCE_DIGEST_CACHE is not None:
        return _SOURCE_DIGEST_CACHE

    import hashlib
    from pathlib import Path as _Path

    import socr

    root = _Path(socr.__file__).resolve().parent
    digest = hashlib.sha256()
    try:
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            digest.update(str(path.relative_to(root)).encode("utf-8"))
            digest.update(b"\x00")
            digest.update(path.read_bytes())
            digest.update(b"\x00")
        _SOURCE_DIGEST_CACHE = digest.hexdigest()
    except OSError:
        # An unreadable tree must not silently degrade into "no code identity",
        # which would re-open GH-214. A FIXED sentinel is not enough: two runs from
        # different unreadable checkouts would share it and resume each other's
        # pages. Tagging it with a per-process value keeps one run internally
        # consistent while guaranteeing the next process reprocesses.
        import uuid

        _SOURCE_DIGEST_CACHE = f"unreadable-source-tree:{uuid.uuid4().hex}"
    return _SOURCE_DIGEST_CACHE


def _table_judge_prompt_digest() -> str:
    """SHA-256 of the table-judge prompt policy files, read at call time.

    GH-353 TICKET-B1: a wording-only edit to ``prompts/table_judge.md``
    (A0) changes what every rung is asked without moving
    ``table_judge_rung1_model`` / ``table_judge_rung2_binary`` /
    ``table_judge_timeout_sec`` — those are identity/timeout knobs, not the
    prompt itself. Without this, a prompt-wording fix would leave
    already-terminal ladder-judged pages resumable under the OLD wording
    (the same GH-214 gap ``_socr_source_digest`` closes for the rest of the
    package, narrowed to the one file that is data, not code, and therefore
    outside the source-tree hash). Not cached process-wide (unlike
    ``_socr_source_digest``): the file is a few KB, read once per document,
    and tests routinely monkeypatch ``load_table_judge_prompt``/the file
    itself, so a permanent module-global cache would go stale across runs
    in the same process.

    GH-373: the page-scope fragment is hashed too. A wording-only edit to
    ``prompts/table_judge_scope_page.md`` changes what a degraded-scope
    look is asked without touching the main template.
    """
    import hashlib

    from socr.judge.table_prompt import load_table_judge_prompt, load_table_judge_scope_note

    try:
        blob = load_table_judge_prompt() + "\0" + load_table_judge_scope_note("page")
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()
    except OSError:
        # An unreadable prompt file must not silently degrade to "no prompt
        # identity" (the ``_socr_source_digest`` precedent): tag with a
        # per-call sentinel so a run under a missing/corrupt prompt file
        # never resumes a page a readable prompt produced.
        import uuid

        return f"unreadable-table-judge-prompt:{uuid.uuid4().hex}"


def _cell_transcribe_prompt_digest() -> str:
    """SHA-256 of the cell-transcribe prompt, read at call time.

    GH-367: a wording-only edit to ``prompts/cell_transcribe.md`` changes
    the constrained transcriber without moving rung identity or timeout.
    Same non-caching rule as ``_table_judge_prompt_digest``.
    """
    import hashlib

    from socr.judge.cell_transcribe import load_cell_transcribe_prompt

    try:
        return hashlib.sha256(load_cell_transcribe_prompt().encode("utf-8")).hexdigest()
    except OSError:
        import uuid

        return f"unreadable-cell-transcribe-prompt:{uuid.uuid4().hex}"


class _LatchedDocMetadata:
    """``DocMetadata`` plus lane pending-retry latches on the root entry.

    ``DocMetadata.to_entry`` is a fixed contract shape and lane retry latches
    (equation lane, table judge ladder) are socr-side state, so they are
    layered on here rather than pushed into the contract. Passing this to
    ``RootIndex.record`` keeps that method the sole author of root
    metadata.json -- the PP-5 invariant -- while still committing the status
    and any latches in a single save (cold review round 3, finding 5).

    Everything other than ``to_entry`` delegates to the wrapped metadata.
    ``to_entry()`` merges ``pending`` into the entry. Values, not just flags:
    the table lane records WHICH rung kinds were unavailable (cold review
    round 3, finding 1), so the resume gate can ask about those rungs instead
    of about "some rung somewhere".
    """

    def __init__(self, meta, pending):
        self._meta = meta
        # Accepts either a mapping of key -> value, or a plain iterable of key
        # names for the flag-only lanes (the equation lane's original shape).
        if isinstance(pending, Mapping):
            self._pending = dict(pending)
        else:
            self._pending = {str(key): True for key in pending}

    def __getattr__(self, name):
        return getattr(self._meta, name)

    def to_entry(self) -> dict:
        entry = self._meta.to_entry()
        entry.update(self._pending)
        return entry


def _resume_skippable(
    index,
    rel_key: str,
    checksum: str,
    fingerprint: str,
    out_dir: Path,
    *,
    equation_lane_retry_blocks: bool | Callable[[], bool] = False,
    table_judge_retry_blocks: bool | Callable[[list[str]], bool] = False,
) -> bool:
    """Whether a doc can be skipped by the resume gate.

    Canonically completed docs skip via :meth:`RootIndex.is_completed`. A
    PARTIAL doc additionally skips when its recorded checksum AND run
    fingerprint match and its output still exists: re-running the identical
    config cannot improve a partial result, and without this rule every doc
    demoted to AUDIT_FAILED (flagged native fallback, lost pages) would be
    re-processed at full provider cost on EVERY batch resume, forever.
    ``--reprocess`` still forces a retry (checked by the callers).

    ``equation_lane_retry_blocks`` closes the hole cold review round 2 found in
    P4-R's retry latch (finding 5). The per-page ledger check is downstream of
    THIS gate: ``process()`` consults the document gate before a
    ``DocumentState`` exists, so a document whose equation pages never reached
    a provider was skipped whole and the per-page latch was never read. Provider
    availability is transient and deliberately absent from the fingerprint, so
    everything this function checks can match while the document still holds an
    unread equation. When the caller passes True -- meaning the equation lane is
    enabled on this run -- a recorded pending retry refuses the skip.

    ``table_judge_retry_blocks`` (P1) applies the identical rule to the table
    judge ladder: when a recorded entry carries ``table_judge_retry_pending``
    and a table-judge rung is reachable now, the document skip is refused so
    the pending table is re-judged. When the ladder is disabled, strict_local is
    on, or all rungs remain unavailable, the entry remains skippable. The
    predicate is evaluated lazily only when the entry actually carries the
    latch, avoiding probes on ordinary or flag-off resumes.

    Re-running is then cheap rather than free: pages that DID finish are still
    restored by the per-page ledger, which is provider-aware and keeps skipping
    them, so an offline rerun re-opens the document without re-reading anything.
    """
    entry = index.files.get(rel_key)
    if entry:
        if entry.get("equation_lane_retry_pending") is True:
            blocks = (
                equation_lane_retry_blocks()
                if callable(equation_lane_retry_blocks)
                else equation_lane_retry_blocks
            )
            if blocks:
                return False
        if entry.get("table_judge_retry_pending") is True:
            # Cold review round 3, finding 1: ask about the rung(s) that were
            # ACTUALLY unavailable. A healthy rung 1 must not stand in for a
            # rung 2 that is still down -- that reopened the document and
            # re-ran the whole ladder on every resume. An entry written before
            # the rung list existed carries no kinds; the empty list means
            # "unknown", and the predicate falls back to asking about any rung,
            # which is the conservative reading for an old record.
            recorded = entry.get("table_judge_retry_rungs")
            kinds = [str(k) for k in recorded] if isinstance(recorded, list) else []
            blocks = (
                table_judge_retry_blocks(kinds)
                if callable(table_judge_retry_blocks)
                else table_judge_retry_blocks
            )
            if blocks:
                return False
    if index.is_completed(rel_key, checksum, fingerprint=fingerprint):
        return True
    if not entry or entry.get("status") != "partial":
        return False
    # NOTE (cold review round 1, finding 1): no unconditional lane-latch check
    # belongs here. A latch records a TRANSIENT external outage, and the only
    # question the gate asks is whether that outage is over: both lanes already
    # refused the skip above when their rung/provider is reachable NOW. Refusing
    # it a second time regardless of reachability would re-run the whole lane on
    # every resume of a persistent outage -- paying timeout x tables x rungs to
    # rediscover that the rung is still down -- which is exactly what the
    # docstring above rules out.
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


#: GH-353: the two table judge ladder terminals. Local to this module,
#: deliberately not imported from ``core.manifest``'s
#: ``_LADDER_TERMINAL_FAILURE_MODES`` -- that name is private to C3's guard
#: and this predicate reads a different (pre-guard) surface, see
#: ``_table_ladder_terminal`` below.
_LADDER_TERMINAL_MODES = (FailureMode.TABLE_REJECTED, FailureMode.TABLE_UNVERIFIED)


def _table_ladder_terminal(p) -> FailureMode | None:
    """GH-353 C2 review fix: the ladder terminal a page carries, disposition-first.

    C3's decision log (``docs/log/2026-08-30_ticket-c3.md``) contracts B1 to
    set ``PageState.table_ladder_disposition`` and rely on
    ``_apply_ladder_disposition_guard`` (``core/manifest.py``) to enforce it.
    That guard demotes only the finalized COPY it returns from
    ``_winning_page_output``/``finalized_page_outputs`` -- it deliberately
    never mutates ``best_output`` in place (the #252 round-1 rule: flipping a
    shipped attempt's ``audit_passed``/``failure_mode`` in place would make
    assemble discard the page's text). So reading only
    ``best_output.failure_mode`` here is blind to every ladder terminal
    whenever no OTHER, more specific guard already demoted the kept attempt
    -- exactly the silent-miss this ticket exists to prevent. ``PageState``
    is the durable, pre-guard signal (mirrors every other bucket in
    ``_phase_assemble``, which all read ``PageState`` flags rather than
    re-deriving from a finalized copy), so it is read FIRST here; the
    finalized-output failure_mode is kept as a belt-and-braces fallback for
    the case where the disposition attribute is unset but a more specific
    caller already stamped the terminal directly onto ``best_output``. A
    page can therefore land in exactly one bucket: when both are set, the
    disposition wins.
    """
    disposition = getattr(p, "table_ladder_disposition", None)
    if disposition in _LADDER_TERMINAL_MODES:
        return disposition
    fm = p.best_output.failure_mode if p.best_output else None
    return fm if fm in _LADDER_TERMINAL_MODES else None


_ORTHOGONAL_ASSEMBLE_BUCKET_NAMES = (
    "native_only_distrust_pages",
    "value_drift_pages",
    "fabricated_ref_pages",
    "text_grid_rejected_pages",
    "chart_detection_failed_pages",
    "table_rejected_pages",
    "table_unverified_pages",
)


def _derive_orthogonal_assemble_buckets(state: DocumentState) -> dict[str, list[int]]:
    """Derive the assemble buckets that are orthogonal to page selection.

    These predicates intentionally remain based on configuration, page flags,
    events, and ladder terminals.  The helper is an observation seam only: it
    does not mutate ``state`` or invoke any extraction, chart, verifier, ladder,
    selector, reconstruction, or emission code.

    ``_phase_assemble`` supplies the current pipeline config on the private
    assemble-state context because ``native_only`` is pipeline configuration,
    not a property of ``DocumentState`` itself.  States created without that
    context use the normal ``False`` default, matching ``PipelineConfig``.
    """
    config = getattr(state, "_assemble_config", None)
    native_only = bool(getattr(config, "native_only", False))

    native_only_distrust_pages = [
        n
        for n, p in sorted(state.pages.items())
        if p.is_born_digital
        and p.native_text
        and native_only
        and getattr(p, "native_table_unverifiable", False)
        and not p.native_table_structure_failed
        and p.attempts
        and all((a.engine or "").startswith("native") for a in p.attempts)
        and not (p.best_output and p.best_output.audit_passed)
    ]
    value_drift_pages = sorted(
        {
            getattr(e, "page_num", 0)
            for e in state.events
            if getattr(e, "kind", "") == "table_value_drift_unadjudicated"
            and getattr(e, "page_num", 0)
        }
    )
    fabricated_ref_pages = sorted(
        n for n, p in state.pages.items() if getattr(p, "fabricated_image_refs", 0)
    )
    text_grid_rejected_pages = sorted(
        n for n, p in state.pages.items() if getattr(p, "text_grid_rejected", False)
    )
    chart_detection_failed_pages = sorted(
        n for n, p in state.pages.items() if getattr(p, "chart_asset_detection_failed", False)
    )
    table_rejected_pages = sorted(
        n for n, p in state.pages.items() if _table_ladder_terminal(p) == FailureMode.TABLE_REJECTED
    )
    table_unverified_pages = sorted(
        n
        for n, p in state.pages.items()
        if _table_ladder_terminal(p) == FailureMode.TABLE_UNVERIFIED
    )

    return {
        "native_only_distrust_pages": native_only_distrust_pages,
        "value_drift_pages": value_drift_pages,
        "fabricated_ref_pages": fabricated_ref_pages,
        "text_grid_rejected_pages": text_grid_rejected_pages,
        "chart_detection_failed_pages": chart_detection_failed_pages,
        "table_rejected_pages": table_rejected_pages,
        "table_unverified_pages": table_unverified_pages,
    }


class UnifiedPipeline:
    """OCR pipeline orchestrator.

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
    _final_records: dict[int, FinalizedPageRecord] | None = None

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.heuristics = HeuristicsChecker(min_word_count=config.audit_min_words)
        self.scorer = FailureModeScorer(checker=self.heuristics)
        self.bd_detector = BornDigitalDetector()
        # GH-222: swappable liveness probe for the local VLM backend. ``None``
        # means "use the resolved-host Ollama probe" (see ``_probe_backend_idle``);
        # assign a zero-argument callable to substitute a different backend's
        # health check without touching the cascade-halt logic.
        self.backend_probe: Callable[[], bool] | None = None
        self._last_assessment: DocumentAssessment | None = None
        # Directory the current input was discovered under (batch input dir or
        # the file's parent). Threaded into every contract key so per-doc output
        # mirrors the input subtree relative to it, not the bare basename.
        self._scan_root: Path | None = None
        #: GH-525: the inert-config warning is emitted once per pipeline, not
        #: once per document -- `process_batch` calls `process` per PDF and the
        #: config cannot change between them.
        self._warned_inert_config = False
        #: Table-judge rung reachability, per rung kind, probed at most once per
        #: RUN (cleared by ``_reset_table_judge_rung_probes`` at every public run
        #: boundary) alongside the per-run refusal breaker.
        self._table_rung_available_cache: dict[str, bool] = {}
        self._table_rung_refused_this_run: set[str] = set()
        #: The rung CALLABLES that refused us this run, held by identity. Cold
        #: review round 4: the kind set alone cannot spare pages 2..N, because
        #: the gate receives opaque callables and only the RESULT names a kind.
        #: Positional correspondence with ``run_table_ladder``'s results gives
        #: the mapping, so the object that produced a refusal is not called
        #: again this run whether or not it advertises its kind.
        self._table_rung_refused_callables: list = []
        #: The rung callables this run has already CALLED without a refusal.
        #: Cold review round 5: identity is authoritative for a callable we
        #: have observed; the kind-level breaker applies only to callables we
        #: have not. Without that split, either a batch's rebuilt rung escapes
        #: the breaker, or a healthy same-kind sibling is wrongly dropped.
        self._table_rung_seen_callables: list = []
        #: The rung list handed to the CURRENT ``_run_table_judge_gate`` call.
        #: Only used to resolve the historical positional executing identity
        #: for a rung that advertises nothing; see ``_executing_identity``.
        self._table_judge_gate_rungs: list = []
        #: True while ``process_batch`` is driving ``process`` per file, so the
        #: batch counts as ONE reachability epoch rather than one per file.
        self._in_batch_run = False
        self._final_records = None
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
                RootIndex(out_dir),
                rel_key,
                checksum,
                self._run_fingerprint(),
                out_dir,
                equation_lane_retry_blocks=self._equation_lane_retry_blocks_resume(),
                table_judge_retry_blocks=self._table_judge_retry_blocks_resume,
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

    def _equation_lane_retry_blocks_resume(self) -> bool:
        """Whether a recorded P4-R pending retry should refuse a document skip.

        Only when the lane could actually act on the retry: it is enabled and
        this is an agentic run. With the lane switched off, a marker left by an
        earlier run is inert history and must not force endless reprocessing.
        """
        return bool(self.config.equation_region_lane and self.config.agentic)

    def _table_judge_retry_blocks_resume(self, rung_kinds: list[str] | None = None) -> bool:
        """Whether a recorded table-judge pending retry should refuse a document skip.

        Only when the ladder is enabled, strict_local is false, and one of the
        rungs THAT WAS UNAVAILABLE is attemptable now (cold review round 3,
        finding 1). ``rung_kinds`` empty or None means the record does not say
        which rung failed -- an entry written before the list existed -- and the
        question falls back to "is any rung reachable", the conservative reading.
        """
        return bool(self._table_judge_rung_available_now(rung_kinds))

    def _invalidate_root_entry_for_rerun(
        self, pdf_path: Path, out_dir: Path
    ) -> EngineResult | None:
        """Mark an existing root entry non-resumable before this run rewrites it.

        Returns None to proceed, or an ERROR result the caller must return.

        Only acts when a record already exists: a first run has nothing to
        invalidate. The replacement is written with ``Status.FAILED`` -- neither
        ``completed`` nor ``partial``, so ``_resume_skippable`` refuses it in
        both of its branches -- and carries an explicit error string saying a run
        is in progress, so nothing reads it as a result.

        If that write fails the run REFUSES to start. Proceeding would re-create
        the output the stale entry points at and hand the next run a skippable
        record describing work this run may never finish, which is the
        lost-retry outcome P4-R's latch exists to prevent. Refusing costs a
        re-run; proceeding silently loses an unread equation page.
        """
        from ocr_output_contract import (
            DocMetadata,
            RootIndex,
            Status,
            relative_key,
            safe_checksum,
            utc_timestamp,
        )

        try:
            checksum = safe_checksum(pdf_path)
            if checksum is None:
                return None
            scan_root = self._scan_root or pdf_path.parent
            rel_key = relative_key(pdf_path, scan_root)
            index = RootIndex(out_dir)
            if index.files.get(rel_key) is None:
                return None
        except Exception as exc:
            # Reading the index is not the write this guard protects; a broken
            # read leaves the run to the ordinary metadata path.
            logger.warning("root-entry invalidation check failed (non-fatal): %s", exc)
            return None

        marker = DocMetadata(
            status=Status.FAILED,
            checksum=checksum,
            model=self.config.primary_engine.value,
            backend="socr",
            processing_time=0.0,
            timestamp=utc_timestamp(),
            output_path="",
            pages=0,
            error="run in progress; the previous record was invalidated before reprocessing",
            fingerprint=self._run_fingerprint(),
        )
        try:
            index.record(rel_key, marker)
        except Exception as exc:
            message = (
                f"refusing to reprocess {pdf_path.name}: the stale root index entry could "
                f"not be invalidated ({exc}). Proceeding would re-create the output that "
                "entry points at and let the next run skip this document."
            )
            logger.error(message)
            if not self.config.quiet:
                console.print(f"[red]{message}[/red]")
            return EngineResult(
                document_path=pdf_path,
                engine=self.config.primary_engine.value,
                status=DocumentStatus.ERROR,
                error=message,
            )
        return None

    def _engine_determinants(self, engine_type: EngineType) -> dict[str, str | None]:
        """Resolved ``{model, backend, task}`` for an engine, never raising.

        Used to fold the model/backend/task of every configured engine that can
        contribute text -- the primary, the local engine, and each member of
        ``enabled_engines`` (the agentic ladder is built from those) -- into the
        run fingerprint. A swap of a secondary engine's model/task/backend
        changes the saved output (the orchestrator routes pages to it), so it
        must invalidate the resume cache just like a primary-engine swap.
        Degrades to the engine name on any error so fingerprinting never breaks
        a run.

        The FALLBACK chain is no longer among the sources (GH-525): the
        ``fallback_chain`` field is not read for routing, so changing it alone
        cannot select a provider. That is narrower than "its members cannot
        contribute" (cubic P3 on #532) -- a member enabled independently still
        can, and is fingerprinted through ``enabled_engines``.
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

        Three keys are deliberately ABSENT -- ``judge_hard_pages``,
        ``fallback_chain`` and its determinants -- because none gates anything
        (GH-142 / GH-525). See the note beside ``local_engine_determinants``
        below for why dropping them, rather than freezing them, is free.

        Captures what changes *what output an input produces*: the resolved
        primary engine's model id, backend, and task, the resolved determinants
        of every secondary engine that can contribute text (the local engine and
        each member of ``enabled_engines``, which is what the agentic ladder is
        built from), and socr's output-affecting orchestration flags. Stored in
        :class:`DocMetadata.fingerprint` and consulted by
        :meth:`RootIndex.is_completed`, so a re-run under a different model / task
        / flag reprocesses instead of silently reusing the cached output.

        Round-3 expansion (HIGH): the prior ``extra`` omitted ``save_figures``
        (and figure limits), the figures/judge model+backend knobs,
        ``local_engine``, ``tiered`` routing, and chunking thresholds — all of
        which change the saved ``.md``/figures. Toggling any of them now
        invalidates the cache.

        ``fallback_chain`` was in that list and has since been removed from it
        (GH-525): it does NOT change the saved output, because no execution path
        reads it -- the multi-engine branches that did were deleted in #298.

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
            # Resolved model/backend/task of the local engine, so a swap of its
            # model invalidates the cache too. (Not the fallback chain -- see
            # the GH-525 note below.)
            "local_engine_determinants": self._engine_determinants(cfg.local_engine),
            # GH-525: `judge_hard_pages`, `fallback_chain` and its determinants
            # are deliberately ABSENT. Neither field is read for routing
            # (GH-142 rejected their flags for that), so including them made a
            # config-only toggle invalidate every terminal page and force a
            # reprocess producing byte-identical output.
            #
            # Narrower than "a fallback member cannot contribute" (cubic P3 on
            # #532): a member ENABLED independently still can, and is
            # fingerprinted through `enabled_engines` below. What changing
            # `fallback_chain` alone cannot do is select a provider.
            #
            # Dropping them changes this fingerprint once, for everybody. That
            # cost was the reason not to -- until `_socr_source_digest` was
            # checked: it hashes every shipped .py file, so ANY source edit
            # already invalidates every fingerprint, deliberately ("an
            # output-neutral edit costs one needless reprocess. Re-OCR is
            # cheap"). This edit is one of those, so the cost is not additional.
            #
            # `_warn_inert_config` names them when a config sets them, because
            # ignoring a setting silently is the failure this ticket family is
            # about.
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
            # provider present only through ``enabled_engines`` (not also via
            # primary/local/fallback) would otherwise let a model/backend/
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
            # --- quality gates / routing ---
            "agentic": cfg.agentic,
            "strict_local": cfg.strict_local,
            "audit_min_words": cfg.audit_min_words,
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
            # #229: ``auto_patch_tables`` rewrites table cells in the saved page.
            # Toggling it changed the .md without changing the fingerprint, so a
            # resumed run mixed patched and unpatched pages under one document
            # status. It is read only by the in-loop ``_reread_page_tables`` call,
            # which is itself behind ``dual_pass_tables``; recording it only when
            # that lane is enabled avoids invalidation when the lane is inactive.
            "auto_patch_tables": cfg.auto_patch_tables if cfg.dual_pass_tables else None,
            # --- GH-353: table judge ladder ---
            # Flag + both rung identities + the per-call timeout, all recorded
            # only while the gate is on: it rewrites PageState.table_ladder_
            # disposition (and, via the manifest guard, the shipped
            # audit_passed/failure_mode), so a resumed run under a different
            # rung, model, or timeout must not reuse a terminal sidecar the
            # OTHER configuration produced. The prompt file digest catches a
            # wording-only edit to prompts/table_judge.md, which changes what
            # the judges are asked without moving any of the scalars above.
            "table_judge_ladder": cfg.table_judge_ladder,
            "table_judge_rung1_model": (
                cfg.table_judge_rung1_model if cfg.table_judge_ladder else None
            ),
            # GH-366: the HOST too, not just the model tag. A tag is a label,
            # not the thing that answered -- two hosts can serve different
            # builds under the same name, so pointing rung 1 elsewhere while
            # keeping the tag would resume a verdict a different judge produced.
            # That is the same shape as #133 (judge identity) and #229, and this
            # repo has a history of fingerprint-omission bugs (#214).
            #
            # The RESOLVED host, not the config field -- the same distinction
            # ``judge_model`` above makes, and for the same reason. The field
            # defaults to None and ``resolve_ollama_host`` then falls back to
            # OLLAMA_HOST and finally localhost, so two runs against genuinely
            # different daemons both record None and share a fingerprint. That
            # is the omission this ticket exists to close, one level down.
            #
            # SCOPE: rung 1 only, which is what this ticket covers. The broader
            # question -- whether ``ollama_host`` / ``math_model_host`` should be
            # fingerprinted on the same reasoning -- is deliberately NOT settled
            # here: those invalidate every cached page on a laptop/HPC move, and
            # that trade is a separate decision, not a side effect of this one.
            "table_judge_rung1_host": (
                _resolve_ollama_host(cfg.table_judge_rung1_host) if cfg.table_judge_ladder else None
            ),
            "table_judge_rung2_binary": (
                cfg.table_judge_rung2_binary if cfg.table_judge_ladder else None
            ),
            "table_judge_timeout_sec": (
                cfg.table_judge_timeout_sec if cfg.table_judge_ladder else None
            ),
            "table_judge_prompt_digest": (
                _table_judge_prompt_digest() if cfg.table_judge_ladder else None
            ),
            # GH-367: the cell-transcribe prompt is the constrained model
            # role that can lift a binding clamp. A wording-only edit
            # changes what counts as a raster disproof and must not reuse
            # a previously lifted sidecar.
            "cell_transcribe_prompt_digest": (
                _cell_transcribe_prompt_digest() if cfg.table_judge_ladder else None
            ),
            # --- figures ---
            # ``save_figures`` controls PNG extraction + image-ref embedding.
            # ``describe_figures`` is the separate opt-in for VLM captions.
            # Both change the saved .md content, so both must invalidate the cache.
            "save_figures": cfg.save_figures,
            "describe_figures": cfg.describe_figures,
            # --- corrupt-font math recovery (separate from GH-36) ---
            # ``recover_corrupt_math`` re-renders equations through a VLM and
            # replaces page text, so the flag AND the model identity change the
            # saved bytes. The model is recorded only while the flag is on: an
            # unused model default must not force a needless reprocess (#233).
            "recover_corrupt_math": cfg.recover_corrupt_math,
            "math_model": cfg.math_model if cfg.recover_corrupt_math else None,
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
            # --- P4-R: the agentic equation-region lane ---
            # The lane routes table-free equation pages out of the free native
            # bypass and can attach crop-backed LaTeX beside the native slice,
            # so toggling it changes the saved .md and must invalidate the
            # per-page resume ledger.
            "equation_region_lane": cfg.equation_region_lane,
            # #230: the model that produces those LaTeX sidecars. Swapping it
            # changes the sidecar text, so it must invalidate. It now has TWO
            # consumers -- the legacy GH-36b lane, which runs iff BOTH of its
            # flags are set (``recover_clean_equations and detect_equations``),
            # and P4-R, which reads it whenever the region lane is on. Record
            # the model when EITHER consumer can run; recording it when neither
            # can would force a needless reprocess over an unused default (#233).
            "clean_equation_model": (
                cfg.clean_equation_model
                if (
                    (cfg.recover_clean_equations and cfg.detect_equations)
                    or cfg.equation_region_lane
                )
                else None
            ),
            "figures_engine": cfg.figures_engine.value,
            # #232: the caption model was invisible to the fingerprint. Note the
            # trap: ``figures_engine`` above is NOT the caption engine -- nothing
            # in src/ reads that field. Captions come from ``_get_vision_engine``
            # (:5934), which builds a local Ollama figure engine with a per-call
            # Gemini fallback constructed from ``gemini_model``. The Ollama model
            # is a source-level default, already covered by ``socr_source_digest``;
            # ``gemini_model`` is config, and under a custom ``enabled_engines``
            # excluding GEMINI it reached the fingerprint through no other route,
            # so a swap left stale captions resumable. Recorded only while
            # captions are produced, so the default never forces a reprocess.
            "figure_caption_fallback_model": (cfg.gemini_model if cfg.describe_figures else None),
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
            # --- GH-214: socr's own source identity ---
            # The two versions above cover the normalizer and assembler only. Every
            # OTHER output-affecting module (born_digital, tables, judge, engines)
            # could change without moving any value in this dict, so a correctness
            # fix would leave already-terminal pages skippable and never reach them.
            # Hashing the shipped package closes that generally, and fails safe:
            # an output-neutral edit costs a reprocess, never a stale result.
            "socr_source_digest": _socr_source_digest(),
        }
        return run_fingerprint(
            primary["model"], primary["backend"] or "socr", primary["task"], None, extra=extra
        )

    def _report_inert_config(self) -> None:
        """Name any config field the run is ignoring, once per pipeline.

        GH-525: a config that sets an inert field must not be ignored in
        silence -- that is the same failure the rejected flags had, moved to
        the YAML layer.

        Once per PIPELINE, not per document: `process_batch` calls `process` for
        every PDF, so an unguarded warning repeated once per file and became
        noise an operator learns to skip. The config cannot change between those
        calls, so one line says everything repeating it would.

        Called from BOTH entry points (cubic P2 on #529). Calling it only from
        `process` meant an empty batch -- nothing to do, everything skipped --
        reported nothing, even though its fingerprint ignored the fields just
        the same. The guard makes the second call a no-op.
        """
        if self._warned_inert_config:
            return
        inert = _warn_inert_config(self.config)
        if not inert:
            return
        self._warned_inert_config = True
        message = (
            f"ignoring config field(s) {', '.join(inert)}: they gate nothing on "
            "any path (GH-142/GH-525) and are excluded from the run fingerprint, "
            "so setting them changes neither the output nor which pages are reused"
        )
        logger.warning("%s", message)
        if not self.config.quiet:
            console.print(f"  [yellow]{message}[/yellow]")

    def process(
        self,
        pdf_path: Path,
        output_dir: Path | None = None,
        scan_root: Path | None = None,
    ) -> EngineResult:
        """Process a single PDF through analysis, agentic extraction, and assembly.

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
        # Cold review round 3, new finding 1: a public run starts a fresh
        # reachability epoch. A pipeline object outlives a run, and a cache
        # that never resets means a rung that came back between runs is never
        # observed. Suppressed under ``process_batch``, which is ONE run whose
        # pre-gate and per-file answers must agree.
        if not self._in_batch_run:
            self._reset_table_judge_rung_probes()
        self._report_inert_config()
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

        # Fail closed BEFORE re-emitting output, not only after (cold review
        # round 4). The terminal write at the end is atomic, but atomicity only
        # protects what it writes: a matching OLDER entry -- completed,
        # latch-free, its output since deleted -- survives a failed terminal
        # write, and this run will have re-created the output it points at, so
        # it becomes resumable again with no latch and the next run skips the
        # document whole. Invalidating that record first means a failure
        # anywhere in the run leaves an entry the gate refuses.
        refused = self._invalidate_root_entry_for_rerun(pdf_path, out_dir)
        if refused is not None:
            return refused

        doc = DocumentHandle.from_path(pdf_path)
        state = DocumentState(handle=doc)

        if not self.config.quiet:
            console.print(f"[blue]Processing:[/blue] {doc.filename}")
            console.print(f"[dim]{doc.page_count} pages, {doc.size_mb:.1f} MB[/dim]")

        # Analyze the document before entering the per-page agentic lane.
        self._phase_analyze(state)
        # The fused loop owns all per-page extraction, including its optional
        # in-loop table reread, before final assembly.
        self._phase_agentic(state, out_dir)

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
        self._reset_table_judge_rung_probes()
        self._report_inert_config()

        from ocr_output_contract import (
            RootIndex,
            RunOutcome,
            Status,
            is_within_output_root,
            relative_key,
            safe_checksum,
        )

        from socr.core.result import contract_status_for

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

        # Cold review round 3, finding 1: the pre-gate predicate is now
        # per-file, because each entry names the rung kinds IT is waiting on.
        # The run-local memoization that used to live here would have collapsed
        # those different questions into one answer; the per-kind cache on the
        # pipeline does the memoizing instead, so a batch still probes each rung
        # kind at most once.
        for pdf in pdfs:
            # Resume gate via the canon: an unreadable input (safe_checksum None)
            # is NEVER treated as completed — it falls through to process(), which
            # records a per-file failure rather than aborting the batch (SYS-02).
            checksum = safe_checksum(pdf)
            rel_key = relative_key(pdf, input_dir)
            already_done = checksum is not None and _resume_skippable(
                root_index,
                rel_key,
                checksum,
                run_fp,
                out_dir,
                equation_lane_retry_blocks=self._equation_lane_retry_blocks_resume(),
                table_judge_retry_blocks=self._table_judge_retry_blocks_resume,
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

        # Cold review round 3, new finding 1: the batch is ONE reachability
        # epoch. It was reset on entry above, and the per-file ``process`` calls
        # below must not reset it again, or the pre-gate's admission decision and
        # the per-file resume decision could disagree inside one run. ``finally``
        # so an exception cannot leave the pipeline believing it is still in a batch.
        self._in_batch_run = True
        try:
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
                # GH-177: one mapping, shared with the single-file path, so the two
                # cannot drift into opposite exit codes for the same document.
                # process()->_phase_assemble already recorded this doc in the
                # canonical RootIndex with the contract schema (model/backend/
                # fingerprint/UTC timestamp). No legacy second write — that was
                # the clobber that downgraded the root index to legacy shape.
                doc_status = contract_status_for(result)
                if doc_status is Status.COMPLETED:
                    outcome.add(Status.COMPLETED, output_path=str(pdf))
                else:
                    outcome.add(doc_status, detail=str(pdf))

        finally:
            self._in_batch_run = False

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

        # #263: same shape for the rotated page with NO detected table, whose
        # native layer came back as one glyph run per line. Keyed on the flag
        # set inside the refusal branch in born_digital.py, never re-derived
        # from ``text_is_rotated`` here -- for the same reason the GH-147 loop
        # above is not re-derived.
        for pa in assessment.pages:
            if getattr(pa, "native_rotated_text_shredded", False):
                state.events.append(
                    AuditEvent(
                        page_num=pa.page_num,
                        kind="rotated_text_shredded",
                        engine="native",
                        detail=(
                            "native text layer refused (dominant text direction is rotated "
                            "and the extracted lines are pieces of one text run); "
                            "page routed to OCR"
                        ),
                    )
                )

        self._emit_native_table_structure_events(state, assessment)

        # GH-195: surface a rejected text-strategy grid. GH-144 A2 rejects a
        # find_tables(strategy="text") grid when a lane boundary split a native
        # numeric token (0.67 -> "0" + ".67") and rebuilds the table with the
        # word-geometry rowizer. That is a real content-loss event on the
        # original rendering path — before A2 existed the page shipped the wrong
        # numbers as a silent SUCCESS — and until now it was visible only as a
        # logger.warning inside reconstruct.py. A log line is not a surface.
        #
        # This IS a demotion, at page and document level (round 2 of the #254
        # review). An earlier version of this comment argued the opposite — that
        # the fallback rowizer being lossless in isolation (GH-144 A1 §2 control)
        # meant the page must not be flagged — and the review did not accept it:
        # #195 requires the rejection to reach page status and document status,
        # not only metadata and the CLI.
        #
        # The demotion is STATUS-ONLY and the page keeps its rebuilt text. What
        # it says is "this page's layout is adversarial to the text strategy and
        # a grid had to be thrown away and rebuilt" — worth spot-checking, and
        # operationally different from a clean first-pass render. See
        # ``_winning_page_output``'s ``grid_rejected`` term for the page surface
        # and ``text_grid_rejected_pages`` in ``_phase_assemble`` for the
        # document one.
        for pa in assessment.pages:
            rejections = getattr(pa, "text_grid_rejections", None) or []
            if not rejections:
                continue
            destroyed_total = sum(int(r.get("destroyed_count", 0)) for r in rejections)
            values: list[str] = []
            for rec in rejections:
                values.extend(str(v) for v in rec.get("values", []))
            state.events.append(
                AuditEvent(
                    page_num=pa.page_num,
                    kind="text_grid_rejected",
                    engine="native",
                    detail=(
                        f"{len(rejections)} text-strategy table grid(s) rejected: a lane "
                        f"boundary split {destroyed_total} native numeric token(s) "
                        f"({', '.join(values)}). Rebuilt with the lossless word-geometry "
                        "rowizer — the shipped values are correct, but this page's layout "
                        "is adversarial to find_tables(strategy='text')."
                    ),
                    data={
                        "rejected_grids": len(rejections),
                        "destroyed_tokens": destroyed_total,
                        "values": values,
                    },
                )
            )
        _rejected_pages = sorted(
            pa.page_num for pa in assessment.pages if getattr(pa, "text_grid_rejections", None)
        )
        if _rejected_pages and not self.config.quiet:
            console.print(
                f"  [yellow]{len(_rejected_pages)} page(s) had a text-strategy table grid "
                f"rejected for numeric-token destruction and rebuilt losslessly: "
                f"{_rejected_pages}[/yellow]"
            )

        self._emit_tr3_detection_events(state, assessment)

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
    # Analysis helpers
    # ------------------------------------------------------------------

    def _emit_tr3_detection_events(self, state: DocumentState, assessment) -> None:
        """GH-205: surface the TR-3 per-region geometry hard-fail UNCONDITIONALLY.

        ``has_unverifiable_table_region`` is computed on every native table page,
        but until now it only ever reached a surface IN CONJUNCTION with something
        else: the ``--native-only`` demotion loop in :meth:`_phase_analyze`, the
        D3-floor event at assemble (which additionally requires the OCR ladder to
        have failed), and ``_winning_page_output`` (which reads it only next to
        ``native_table_structure_failed``).  On a page where no conjunction holds,
        the verdict reached no page status, no document status, no metadata field
        and no CLI line.  A detected failure that nothing surfaces is the
        no-silent-content-loss red line.

        SURFACING ONLY, deliberately.  Nothing here keys page status, document
        status or routing on the flag, and that restraint is the point rather than
        an omission: the flag's FIRING rate is not its defect rate, that precision
        has never been measured, and TR-3 shares the ``is_numeric_token`` machinery
        whose notation gaps (``.034``, U+2217 significance stars) plausibly inflate
        it.  The issue blocks its step 3 on hand-judging the firing set first;
        GH-151 B1 read a firing rate as a defect rate and cost three review rounds
        and a redesign.  Routing on this unattended could delete good tables.

        The kind is TR-3's own DETECTION kind, distinct from the D3 fail-closed
        ``table_region_unverifiable`` emitted at assemble.  Reusing that kind would
        tell a consumer of ``tables_trust.json`` that the region had been routed to
        the image-asset lane when in fact nothing acted on it, and would make a D3
        page carry one kind twice instead of a detection plus its disposition.
        Both are in ``TABLE_DISTRUST_KINDS`` and both are ranked in
        ``audit_log.rank``, so this reaches ``audit_log.json``,
        ``tables_trust.json``, the document metadata note and the CLI summary.

        Kept as its own method so the scope guard can disable EXACTLY this
        emission and re-run an otherwise byte-identical pipeline: any difference
        in status or routing between the two runs is then attributable to this
        surfacing alone, whatever the ambient environment does.
        """
        from socr.core.audit_log import AuditEvent

        for pa in assessment.pages:
            if not getattr(pa, "has_unverifiable_table_region", False):
                continue
            state.events.append(
                AuditEvent(
                    page_num=pa.page_num,
                    kind="table_region_geometry_hard_fail",
                    engine="native",
                    detail=(
                        "per-region geometry verifier hard-failed on this page's native "
                        "table (numeric-token multiset mismatch against the native words). "
                        "Recorded as a detection only: it is deliberately NOT keyed to page "
                        "status, document status or routing pending the hand-judgement of "
                        "the firing set (GH-205 step 2), because the firing rate is not a "
                        "measured defect rate."
                    ),
                    data={"detection_only": True, "native_only": bool(self.config.native_only)},
                )
            )

    def _run_engine_on_pages(
        self,
        state: DocumentState,
        page_nums: list[int],
        enhancement_pages: list[int],
        engine_type: EngineType,
        label: str,
        profile: ProviderProfile | None = None,
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
            profile: The ladder rung being executed, when there is one. Supplies
                the backend/model overrides that make an ambiguous engine run as
                the rung declares (GH-159) — without it, the cloud Qwen rung
                silently executed the local build. ``None`` keeps the pure
                config-derived behaviour used by the non-agentic phases.

        Returns:
            List of PageOutput, one per page_num, with per-page text.
        """
        engine = get_engine(engine_type)

        # GH-159: a rung's declared (backend, model) must actually run, not merely
        # be recorded as provenance. Overrides are empty for every unambiguous
        # profile, so this is a no-op except on the cloud Qwen rung.
        run_config = self.config
        if profile is not None:
            overrides = execution_overrides(profile)
            if overrides:
                run_config = replace(self.config, **overrides)
                logger.info(
                    "[GH-159] provider %s pins backend=%r model=%r for this call",
                    profile.id or engine_type.value,
                    overrides.get("qwen_backend"),
                    overrides.get("qwen_model"),
                )

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

        # GH-159: `is_available()` probes the LOCAL tier -- `VLLM_BASE_URL` or the
        # local instruct build -- so it returns False on a machine that can only
        # reach Ollama Cloud. Asking it about the cloud rung would refuse that rung
        # before its pinned config ever ran, which is the "cloud-only environments
        # successfully use the qwen-cloud rung" acceptance criterion. The two probes
        # are deliberately separate (see `qwen.cloud_model_available`); pick the one
        # that answers for the rung actually being executed.
        if is_cloud_qwen(profile):
            from socr.engines.qwen import cloud_model_available

            try:
                available = cloud_model_available()
            except Exception:  # a probe must never crash routing
                available = False
        else:
            available = engine.is_available()

        if not available:
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
            config=run_config,
            dpi=run_config.render_dpi,
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

    def _emit_native_table_structure_events(self, state, assessment) -> None:
        """Record one durable audit event per page with a native table defect.

        Extracted from ``_phase_analyze`` (GH-303) so the defect-name mapping can
        be exercised directly. Pure move: same flags read, same event emitted.
        """
        from socr.core.audit_log import AuditEvent

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
        defect_detail = {
            "unverifiable_table_region": (
                "native table region failed deterministic geometry verification; "
                "--native-only kept the native text without OCR and marked the page "
                "untrusted"
            ),
            "grid_shape": (
                "native table grid structurally defective (ragged widths and/or a "
                "detached label row)"
            ),
            "table_content_empty": (
                "native table has a valid header and delimiter but no body content "
                "(every body cell is a placeholder)"
            ),
            "table_latex_leak": "native table Markdown contains residual LaTeX table syntax",
            "table_width_mismatch": (
                "native table delimiter width disagrees with its rectangular content"
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
            emission_defect = str(getattr(pa, "native_table_emission_defect", "") or "")
            content_defect = str(getattr(pa, "native_table_content_defect", "") or "")
            if emission_defect:
                defects.append(emission_defect)
            elif content_defect:
                # GH-303. The GH-190 content term feeds the
                # `native_table_structure_defective` aggregate, so an empty native
                # table reached the `elif` below and was reported as `grid_shape` --
                # GH-151's ragged-widths / detached-label defect, whose
                # `defect_detail` text describes something that did not happen.
                # Anyone counting GH-151 against GH-190 mis-attributed the page.
                #
                # Named on its own term, exactly as emission defects already are.
                # Disposition is unchanged: still demoted, never restamped SUCCESS,
                # and `--native-only` is not overridden.
                defects.append(content_defect)
            elif getattr(pa, "native_table_structure_defective", False):
                defects.append("grid_shape")
            if getattr(pa, "native_table_header_unattributed", False):
                defects.append("header_unattributed")
            if not defects:
                continue
            causes = "; ".join(defect_detail[d] for d in defects)
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

        R3 (the model-rung guarantee) originally widened this bypass-exclusion
        to equation pages too, on the theory that "native may never author a
        GRID on a structure-class page" (C1) is meaningless unless a model
        attempt actually runs to author one. BLOCKING 1 on #269 reverted that:
        S1's case (i)/(iii) branch only accepts a GFM table as a "grid was
        authored" signal, so forcing a model rung on an equation page cannot
        select anything a ``$...$``-reading attempt produces -- it only turns
        a free native SUCCESS into an accepted hallucination or a false
        AUDIT_FAILED demotion. Equations stay outside C2 until a non-GFM
        acceptance path exists; this bypass is table-only again, matching
        ``PageState.is_structure_class``.

        ``--native-only``'s own bypass is left on its narrower table-only
        check deliberately (see the "Open" question in the S1 build spec):
        forcing a model rung under an explicit native-only run is a separate,
        unresolved policy call.
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
        """The agentic native-bypass predicate: trusted native MINUS the P4-R lane.

        The shared ``_is_trusted_native_without_ocr`` is deliberately left alone
        (ruling 2 / BLOCKING 1 on #269): it is read by the non-agentic
        single-engine, consensus and repair paths, which have no region
        machinery to fall into and would push an equation page into whole-page
        OCR instead. P4-R's widening therefore lives HERE, in the agentic lane
        set only, and it is a re-route rather than a demotion: the page leaves
        the free bypass for ``_agentic_equation_region_page``, which ships the
        same native prose as its floor.
        """
        if self._is_equation_region_lane_page(page_num, ps):
            return False
        return self._is_trusted_native_without_ocr(page_num, ps)

    def _is_equation_region_lane_page(self, page_num: int, ps: PageState) -> bool:
        """P4-R: whether the region-scoped equation lane owns this page.

        The trigger is ``has_equations`` AS DETECTED -- no threshold, no count,
        no new signal. That is the ratified choice, made from the measured
        table in ``docs/log/2026-09-02_p4m-trigger-rates.md`` (36.1% of today's
        free lane, with no natural break in the math-character distribution that
        would justify inventing a cut). Over-routing is a cost, not a
        correctness risk, because the lane is advisory: it never replaces
        whole-page native prose.

        Excluded, and each for its own reason:
          * tables -- a mixed page is already routed by the table lane, and
            ``PageState.is_structure_class()`` stays table-only so the P2
            structure-class floor can never reach a page this predicate accepts;
          * corrupt math -- the GH-271 region lane owns those pages already;
          * shredded rotated native text -- the native slice is not a usable
            anchor for in-place attachment;
          * ``--native-only`` -- an explicit request for the free lane;
          * anything ``_is_native_eligible_without_ocr`` refuses (not
            born-digital, no native text, needs OCR enhancement), which goes to
            the ladder as a whole page regardless.
        """
        return bool(
            self.config.equation_region_lane
            and self.config.agentic
            and not self.config.native_only
            and self._is_native_eligible_without_ocr(page_num, ps)
            and ps.is_born_digital
            and ps.native_text
            and ps.has_equations
            and not ps.has_corrupt_math
            and not ps.native_rotated_text_shredded
            and not self._page_has_tables(page_num, ps)
        )

    def _corrupt_math_model_disabled_reason(self) -> str:
        """Why the direct equation-model call is forbidden by run policy."""
        model = self.config.math_model or ""
        if self.config.strict_local and "cloud" in model.casefold():
            return f"model call skipped: strict-local forbids remote model {model}"
        if "cloud" in model.casefold() and (
            self.config.max_cost_per_page > 0 or self.config.cost_budget > 0
        ):
            return "model call skipped: remote equation model has no configured price"
        return ""

    def _is_corrupt_math_recovery_page(self, page_num: int, ps: PageState) -> bool:
        """Whether the opt-in region lane owns this page instead of whole-page OCR."""
        return bool(
            self.config.native_first
            and not self.config.native_only
            and self.config.recover_corrupt_math
            and ps.is_born_digital
            and ps.native_text
            and ps.has_corrupt_math
            and not ps.native_rotated_text_shredded
            and not self._page_has_tables(page_num, ps)
        )

    def _recover_corrupt_math_page(
        self,
        state: DocumentState,
        page_num: int,
        output_dir: Path,
    ) -> PageOutput:
        """Build a crop-backed native-prose/LaTeX hybrid for one damaged page.

        The result deliberately remains ``WARNING``/``audit_passed=False``:
        structural LaTeX validation rejects malformed output but cannot prove that
        a parseable candidate matches the source mathematics.
        """
        from ocr_output_contract import doc_dir_for, relative_key

        from socr.core.audit_log import AuditEvent
        from socr.core.pdf import open_pdf
        from socr.math.recover import recover_math_regions, splice_math

        ps = state.pages[page_num]
        native_text = ps.native_text or ""
        scan_root = self._scan_root or state.handle.path.parent
        doc_dir = doc_dir_for(output_dir, relative_key(state.handle.path, scan_root))
        crop_dir = doc_dir / "equations"
        regions = []
        error = ""
        model_disabled_reason = self._corrupt_math_model_disabled_reason()
        try:
            with open_pdf(state.handle.path) as pdf:
                regions = recover_math_regions(
                    pdf[page_num - 1],
                    model=self.config.math_model,
                    host=self._local_backend_host(),
                    dpi=self.config.render_dpi,
                    crop_dir=crop_dir,
                    crop_ref_dir="equations",
                    page_num=page_num,
                    model_disabled_reason=model_disabled_reason,
                )
                text = splice_math(pdf[page_num - 1], native_text, regions)
        except Exception as exc:
            error = f"corrupt-equation region recovery failed: {exc}"
            logger.warning("%s on p%d", error, page_num)
            text = native_text

        if not regions:
            unresolved = (
                "[corrupt equation unresolved: page detector flagged font corruption, "
                "but no recoverable region was retained]"
            )
            text = f"{text}\n\n{unresolved}" if text else unresolved

        recovered = sum(1 for region in regions if region.resolved)
        unresolved = len(regions) - recovered if regions else 1
        crop_paths = [region.crop_path for region in regions if region.crop_path]
        validation = [
            {
                "crop_path": region.crop_path,
                "source_text_aligned": region.source_aligned,
                "validation_ok": region.validation_ok,
                "validation_reason": region.validation_reason,
                "model_id": region.model_id,
                "attempts": region.attempts,
            }
            for region in regions
        ]
        crop_detail = (
            f"{len(crop_paths)} crop(s) retained as visual evidence"
            if crop_paths
            else "no crop was retained"
        )
        detail = (
            f"region-only corrupt-equation recovery adopted {recovered} syntax-valid "
            f"candidate(s) and left {unresolved} unresolved region(s); {crop_detail}; "
            "candidates are non-authoritative"
        )
        call_cost_usd = 0.0 if model_disabled_reason else None
        state.events.append(
            AuditEvent(
                page_num=page_num,
                kind="corrupt_math_region_recovery",
                engine="native+math",
                detail=detail,
                data={
                    "recovered_regions": recovered,
                    "unresolved_regions": unresolved,
                    "crop_paths": crop_paths,
                    "regions": validation,
                    "validation_scope": "latex_syntax_only",
                    "semantic_fidelity_verified": False,
                    "model_id": self.config.math_model,
                    "backend": "ollama-compatible",
                    "cost_usd": call_cost_usd,
                    "model_call_skipped": bool(model_disabled_reason),
                    "model_disabled_reason": model_disabled_reason,
                    "error": error,
                },
            )
        )
        page_output = PageOutput(
            page_num=page_num,
            text=text,
            status=PageStatus.WARNING,
            engine="native+math",
            failure_mode=FailureMode.AUDIT_FAILED,
            error=error,
            audit_passed=False,
            audit_notes=[detail],
            provider_id="corrupt-math-region",
            provider_model=self.config.math_model,
            provider_backend="ollama-compatible",
            skip_reason=model_disabled_reason,
            cost_usd=call_cost_usd,
        )
        if self.config.agentic:
            # Round 6: ``call_cost_usd`` is None whenever the model actually ran,
            # so this lane is the one that most needs the recorder -- the page's
            # spend is UNKNOWN, and the default 0.0 would persist a known zero
            # and hand a resumed run budget it never had.
            state.record_engine_run(
                EngineResult(
                    document_path=state.handle.path,
                    engine="native+math",
                    status=DocumentStatus.AUDIT_FAILED,
                    pages=[page_output],
                    model_version=self.config.math_model,
                    cost=call_cost_usd,
                    pages_processed=1,
                    audit_passed=False,
                    audit_notes=[detail],
                ),
                page_nums=[page_num],
            )
        return page_output

    # ------------------------------------------------------------------
    # P4-R: region-scoped equation lane (advisory, never replaces prose)
    # ------------------------------------------------------------------

    #: Every terminal disposition the lane records for a region, plus its two
    #: page-level outcomes. Named here because ``_restore_terminal_page_state``
    #: replays audit events from an explicit allowlist: a kind missing from this
    #: set is persisted in the page sidecar but disappears from
    #: ``audit_log.json`` the moment the page resumes, which is exactly how a
    #: rejected reading would become invisible (the #252 / GH-353 D1a shape).
    EQUATION_LANE_EVENT_KINDS = frozenset(
        {
            "equation_region_reading_attached",
            "equation_region_reading_rejected",
            # GH-522 (cubic P1 on #537): the refusal record and, with it, the
            # `crop_path` pointing at the evidence. Refusing a reading is only
            # acceptable BECAUSE the crop survives for a human to check; without
            # this the record was dropped on resume and the guarantee held only
            # until the next run.
            "equation_region_reading_unverifiable",
            # GH-540: the legacy seam's refusal, for the same reason and one
            # stronger. The shipped markdown does say a reading was withheld --
            # an HTML comment beside the crop carries the validation reason --
            # but the WITHHELD READING ITSELF (`raw_latex`) lives only in this
            # event. Drop it on resume and nobody can tell what was refused, only
            # that something was. Replaying it also removes an asymmetry with no
            # principle behind it: the region lane's refusal survived a resume
            # and the legacy one did not.
            "equation_sidecar_refused",
            "equation_region_reading_unvalidated",
            "equation_region_reading_unaligned",
            "equation_region_reading_unsafe_markup",
            "equation_lane_no_region",
            "equation_lane_detection_failed",
        }
    )

    @classmethod
    def resume_restore_kinds(cls) -> frozenset[str]:
        """Audit-event kinds `_restore_terminal_page_state` replays.

        A method rather than an inline expression (cubic P2 on #551) so a test
        can drive the REAL assembly. The guard for it previously rebuilt this
        union from the same three sources: dropping a term here left that test
        green while the filter silently stopped replaying a whole family --
        exactly the failure the guard exists to catch, reproduced in the guard.
        """
        from socr.judge.table_verdict import (
            TABLE_BINDING_ADJUDICATED_KIND,
            TABLE_LADDER_EVENT_KINDS,
        )

        return frozenset(
            TABLE_LADDER_EVENT_KINDS
            | {TABLE_BINDING_ADJUDICATED_KIND}
            | cls.EQUATION_LANE_EVENT_KINDS
            # GH-519: the chart lane's debt is a standing property of the page,
            # not of the run that noticed it. GH-563 is the cautionary case: a
            # record that lives only in memory tells the truth once and then
            # tells the resumed operator the opposite. The note and the CLI
            # summary below both read this event, so dropping it on resume
            # would silently retire the debt.
            | {VISUAL_VALUES_NOT_TRANSCRIBED_KIND}
        )

    #: The backends the lane's transport can actually address. ``latex_for_crop``
    #: POSTs to ``{host}/api/generate`` -- the Ollama generate API -- so a
    #: profile serving the same model over vLLM, Gemini or anything else is NOT
    #: a provider for this call, however identical its model tag looks.
    #: Cold review round 2, finding 4.
    EQUATION_LANE_BACKENDS = frozenset({"ollama", "ollama-cloud"})

    def _equation_lane_backend_addressable(self, profile) -> bool:
        """Whether this lane's transport can reach ``profile`` under THIS config.

        Cold review round 3, finding 4. Reading ``profile.backend`` is not
        enough: the registry entry is a DECLARATION, and the same
        ``PROFILE_QWEN_LOCAL`` that declares ``ollama`` resolves through
        ``qwen_backend`` and lands on vLLM on an HPC deployment. The lane would
        then post a crop to the local Ollama socket on the strength of a rung
        that is actually serving somewhere else, and lend the call that rung's
        price and provenance. ``resolved_provenance`` is the function the
        manifest and the CLI invocation already agree on, so asking it is asking
        the same question the rest of the pipeline asks.

        ``auto`` is resolvable only in context: it means Ollama unless
        ``VLLM_BASE_URL`` is exported, which ``qwen_auto_resolves_to_openai``
        decides -- "not a corner case, it is the HPC deployment".
        """
        from socr.core.providers import qwen_auto_resolves_to_openai, resolved_provenance

        try:
            backend = (resolved_provenance(profile, self.config) or ("", ""))[0] or ""
        except Exception as exc:  # a resolver failure is doubt, and doubt refuses
            logger.debug("equation lane: cannot resolve backend for %s: %s", profile, exc)
            return False
        if backend == "auto":
            return not qwen_auto_resolves_to_openai(self.config)
        return backend in self.EQUATION_LANE_BACKENDS

    def _equation_lane_provider(self, available_profiles=None):
        """The ladder rung that can serve the clean-equation model, or why not.

        Returns ``(profile, reason)``; exactly one is populated.

        Cold review round 2, structural ruling: the lane does not re-implement
        the ladder's guarantees, it uses them. Candidates are filtered by
        ``--strict-local`` and then run through ``provider_ladder`` with the
        SAME ``per_page_only`` / ``max_cost_per_page`` arguments
        ``_phase_agentic`` uses to build the routing ladder, so a rung priced
        out of the ladder is priced out of this lane by construction rather
        than by a second copy of the rule.

        Selection is then ``(model, backend)``: the model tag the lane is about
        to send AND a backend this lane's transport can address. Round 1 matched
        the model alone, which let a same-model vLLM rung authorise an Ollama
        call and then lend it its price and provenance (finding 4).

        Fail-closed: no rung, no call, native prose ships.
        """
        from socr.core.providers import TIER_LOCAL, provider_ladder
        from socr.math.recover import DEFAULT_MODEL

        model = self.config.clean_equation_model or DEFAULT_MODEL
        is_cloud = "cloud" in model.casefold()
        if self.config.strict_local and is_cloud:
            return None, f"model call skipped: strict-local forbids remote model {model}"
        if is_cloud and (self.config.max_cost_per_page > 0 or self.config.cost_budget > 0):
            return None, "model call skipped: remote equation model has no configured price"

        profiles = (
            list(available_profiles)
            if available_profiles is not None
            else list(self._available_engines_for_agentic())
        )
        if self.config.strict_local:
            profiles = [p for p in profiles if getattr(p, "tier", None) == TIER_LOCAL]

        ladder = provider_ladder(
            profiles,
            per_page_only=True,
            max_cost_per_page=self.config.max_cost_per_page,
        )
        for profile in ladder:
            if (getattr(profile, "model", "") or "") != model:
                continue
            if self._equation_lane_backend_addressable(profile):
                return profile, ""

        # Name the ACTUAL reason, so an operator can tell "nothing is running"
        # from "you priced it out" from "that rung cannot serve this transport".
        same_model = [p for p in profiles if (getattr(p, "model", "") or "") == model]
        if not same_model:
            served = ", ".join(sorted(getattr(p, "id", "") or "?" for p in profiles)) or "none"
            return None, (
                f"model call skipped: no available provider serves the equation model "
                f"{model} (available: {served}); the page ships its native prose unchanged"
            )
        wrong_backend = [p for p in same_model if not self._equation_lane_backend_addressable(p)]
        if len(wrong_backend) == len(same_model):
            from socr.core.providers import resolved_provenance

            resolved = []
            for p in same_model:
                try:
                    resolved.append((resolved_provenance(p, self.config) or ("?", ""))[0] or "?")
                except Exception:
                    resolved.append("?")
            backends = ", ".join(sorted(set(resolved)))
            return None, (
                f"model call skipped: {model} resolves to backend(s) {backends} under this "
                f"configuration, which this lane's transport cannot address"
            )
        return None, (
            f"model call skipped: every rung serving {model} is priced above "
            f"--max-cost-per-page ({self.config.max_cost_per_page})"
        )

    @staticmethod
    def _equation_lane_availability_refusal(reason: str) -> bool:
        """Whether a refusal reflects TRANSIENT availability rather than config.

        Only an availability refusal latches the page for retry. A refusal the
        run fingerprint already describes -- strict-local versus a cloud model,
        a per-page cap, an exhausted budget -- reproduces identically on every
        rerun, so latching it would make the document permanently unskippable
        and change nothing about the outcome. Cold review round 2, finding 5.
        """
        if not reason:
            return False
        settled_by_config = (
            "strict-local forbids",
            "no configured price",
            "priced above",
            "cost cap",
            "cost budget",
            # Cold review round 4, N2: a model that RESOLVES to a backend this
            # transport cannot address is a supported deployment (the HPC vLLM
            # setup), not an outage. Latching it would refuse the document skip
            # on every rerun, restore the same page, and rewrite the latch --
            # idempotent resume defeated forever for that configuration.
            "resolves to backend(s)",
        )
        return not any(marker in reason for marker in settled_by_config)

    def _equation_lane_remaining_budget(self, state: DocumentState) -> float | None:
        """Paid budget still available to this page, or None when uncapped.

        The same computation, and the same fail-closed rule, the generic OCR
        branch applies before ``route_page``: an unmetered earlier call makes
        the remaining budget unknowable, and an unknown subtotal must never be
        treated as zero spend. Cold review round 2, finding 3.
        """
        if self.config.cost_budget <= 0:
            return None
        total_cost = state.total_cost
        if total_cost is None:
            return 0.0
        return max(self.config.cost_budget - total_cost, 0.0)

    def _equation_lane_call_cost(self, profile, calls: int) -> float:
        """What ``calls`` region reads on ``profile`` cost, as a KNOWN number.

        Cold review round 1, finding 3. The lane used to stamp ``cost_usd=None``
        (unknown) and append no ``EngineResult``, which made an executed model
        call invisible to ``--cost-budget`` live and, on resume, turned
        ``state.total_cost`` into None -- so the SAME document routed a later
        page differently live than resumed. A local profile is priced 0.00,
        which is a known number and not an unknown one.

        One region read is charged as one page-equivalent at the serving
        profile's rate. That is the only unit the profile publishes; it is an
        estimate for a whole-page call and this is a crop, so it is an upper
        bound, and for every local rung it is exactly zero either way.
        """
        rate = float(getattr(profile, "cost_per_page_usd", 0.0) or 0.0)
        return rate * max(calls, 0)

    def _agentic_equation_region_page(
        self,
        state: DocumentState,
        page_num: int,
        ps: PageState,
        output_dir: Path,
        available_profiles=None,
        *,
        ocr=None,
    ) -> None:
        """P4-R: read this page's display-equation regions and attach what survives.

        The page's floor is the ORDINARY native page, built first by
        ``_agentic_native_page`` so a lane that attaches nothing ships bytes,
        status, ``audit_passed`` and audit events identical to a run with the
        lane off. Everything below is additive on top of that floor.

        Per region: crop (already saved by the model-free detector) -> local VLM
        -> 1A structural gate -> numeric-presence REJECTION guard -> in-place
        attachment beside the region's own native slice. Any step that fails
        drops that ONE region's reading and leaves the page text untouched; it
        never demotes the page, never flips ``audit_passed`` (the winner-selection
        flag), and never lets a whole-page model read compete with native prose.
        """
        from socr.core.audit_log import AuditEvent
        from socr.math.equation_latex import (
            attach_equation_sidecars_in_place,
            contract_delimiter_violation,
            process_equation_region,
        )
        from socr.math.recover import DEFAULT_MODEL
        from socr.tables.escalation_canary import (
            PRESENCE_INVENTED,
            PRESENCE_UNVERIFIABLE,
            region_presence_verdict,
            text_value_tokens,
        )

        # 1. The floor. Identical to the lane-off page in every respect.
        self._agentic_native_page(state, page_num, ps)
        baseline = ps.best_output
        native_text = ps.native_text or ""

        # 2. Deterministic, model-free region location. The page trigger is
        #    ``has_equations``; this detector is the LOCATOR, not a second
        #    trigger -- a signalled page with no display region simply has
        #    nothing to read, costs no model call, and ships the floor.
        try:
            regions = self._detect_and_crop_equation_page(state, page_num, output_dir)
        except Exception as exc:
            logger.warning("equation lane: region detection failed on p%d: %s", page_num, exc)
            regions = []
            state.events.append(
                AuditEvent(
                    page_num=page_num,
                    kind="equation_lane_detection_failed",
                    engine="native+equations",
                    detail=(
                        "display-equation region detection failed; the page ships its "
                        "native prose unchanged"
                    ),
                    data={"error_type": type(exc).__name__, "error": str(exc)},
                )
            )
        if not regions:
            state.events.append(
                AuditEvent(
                    page_num=page_num,
                    kind="equation_lane_no_region",
                    engine="native+equations",
                    detail=(
                        "page carries the equation signal but no display-equation region "
                        "was located; no model call was made and the native prose ships "
                        "unchanged"
                    ),
                    data={"has_equations": bool(ps.has_equations), "regions": 0},
                )
            )
            return

        model = self.config.clean_equation_model or DEFAULT_MODEL
        profile, skip_reason = self._equation_lane_provider(available_profiles)
        host = self._local_backend_host()
        calls_made = 0
        rate = float(getattr(profile, "cost_per_page_usd", 0.0) or 0.0)
        remaining_budget = self._equation_lane_remaining_budget(state)
        spent_here = 0.0
        # Cold review round 1 finding 5, tightened in round 2. The latch means
        # exactly one thing: THE EQUATION MODEL DID NOT RUN ON THIS PAGE, for a
        # reason the run fingerprint does not describe. Provider availability is
        # transient external state, so a page skipped because nothing was up
        # would otherwise be written terminal SUCCESS and restored forever,
        # leaving a default-on lane permanently inert on that page.
        #
        # Set for unavailability (round 2: regardless of ``strict_local`` -- a
        # strict-local run whose local rung is briefly down did not run the
        # model either) and, below, for a transport failure. NOT set for refusals
        # the fingerprint DOES describe -- strict-local versus a cloud model, or
        # a budget/cap decision -- because those are identical on every rerun
        # under the same config, so latching them would make the document
        # permanently unskippable while changing nothing.
        if profile is None and self._equation_lane_availability_refusal(skip_reason):
            ps.equation_lane_retry_pending = True

        if not self.config.quiet:
            console.print(
                f"  p{page_num}: [cyan]equation region lane [{model}] "
                f"({len(regions)} region(s))[/cyan]"
            )

        attachable: list = []
        for region_index, region in enumerate(regions):
            crop_path = region.crop_path
            crop_ref = f"equations/{Path(crop_path).name}" if crop_path else ""
            # Cold review round 2, finding 3: the per-page cap and the remaining
            # document budget are checked BEFORE each call, and region reads
            # count CUMULATIVELY -- two reads on one page cost twice the rate,
            # and the cap is per page. Accounting after an unauthorised call
            # does not make a cap respected.
            budget_reason = ""
            if rate > 0.0:
                cap = self.config.max_cost_per_page
                if cap > 0 and spent_here + rate > cap:
                    budget_reason = (
                        f"per-page cost cap reached: {spent_here:.4f} already spent on this "
                        f"page and one more region read at {rate:.4f} would exceed "
                        f"--max-cost-per-page {cap}"
                    )
                elif remaining_budget is not None and spent_here + rate > remaining_budget:
                    budget_reason = (
                        f"document cost budget exhausted: {remaining_budget:.4f} remained and "
                        f"one more region read at {rate:.4f} would exceed it"
                    )
            if skip_reason or budget_reason or not crop_path:
                reason = skip_reason or budget_reason or "no crop was retained for this region"
                state.events.append(
                    AuditEvent(
                        page_num=page_num,
                        kind="equation_region_reading_unvalidated",
                        engine="native+equations",
                        detail=(
                            f"no LaTeX candidate for region {region_index} ({reason}); "
                            "native text for the region is unchanged"
                        ),
                        data={
                            "region_index": region_index,
                            "crop_path": crop_path,
                            "model_id": model,
                            "model_call_skipped": True,
                            "skip_reason": reason,
                            "validation_ok": False,
                        },
                    )
                )
                continue

            result = process_equation_region(
                region_index=region_index,
                page_num=page_num,
                crop_path=crop_path,
                native_text=region.source_text or "",
                source_text=region.source_text or "",
                crop_ref=crop_ref,
                ocr=ocr,
                model=model,
                host=host,
            )
            calls_made += 1
            spent_here += rate

            # Cold review round 1, finding 1: a reading carrying a delimiter the
            # output contract owns is refused outright. `process_equation_region`
            # already fails such a candidate at the contract gate; this branch
            # exists so the refusal lands under its OWN audit kind instead of
            # being filed as an ordinary validation miss -- a model writing page
            # boundaries into a page body is worth being able to grep for.
            markup_violation = contract_delimiter_violation(result.raw_latex)
            if markup_violation:
                state.events.append(
                    AuditEvent(
                        page_num=page_num,
                        kind="equation_region_reading_unsafe_markup",
                        engine="native+equations",
                        detail=(
                            f"region {region_index} reading refused: {markup_violation}; "
                            "embedding it would let a model reading split the document, so "
                            "the page keeps its native prose and is NOT demoted"
                        ),
                        data={
                            "region_index": region_index,
                            "crop_path": crop_path,
                            "model_id": model,
                            "violation": markup_violation,
                            "raw_latex": result.raw_latex,
                        },
                    )
                )
                continue

            if not result.raw_latex.strip():
                # Cold review round 2, finding 5: ``latex_for_crop`` returns ""
                # when the transport fails (unreadable crop, URLError, timeout,
                # a response with no body). A provider was present at selection
                # time and the model still did not run, so the page is NOT a
                # finished result. Conservative by design: a model that genuinely
                # answers with nothing is indistinguishable here and costs a
                # retry, which is the safe direction.
                ps.equation_lane_retry_pending = True

            if not (result.validation_ok and result.raw_latex.strip()):
                state.events.append(
                    AuditEvent(
                        page_num=page_num,
                        kind="equation_region_reading_unvalidated",
                        engine="native+equations",
                        detail=(
                            f"region {region_index} produced no 1A-valid LaTeX "
                            f"({result.validation_reason}); native text for the region "
                            "is unchanged"
                        ),
                        data={
                            "region_index": region_index,
                            "crop_path": crop_path,
                            "model_id": model,
                            "model_call_skipped": False,
                            "validation_ok": result.validation_ok,
                            "validation_reason": result.validation_reason,
                        },
                    )
                )
                continue

            # Ruling 4: numeric presence is a REJECTION guard, not an acceptance
            # contract. Containment is one-way and proves only "not invented" --
            # never that a value is correctly placed or bound.
            encoding_suspect = bool(getattr(ps, "has_encoding_hygiene_suspect", False))
            corrupt_math = bool(ps.has_corrupt_math)
            verdict = region_presence_verdict(
                native_text,
                result.raw_latex,
                encoding_suspect=encoding_suspect,
                corrupt_math=corrupt_math,
            )
            if verdict.status == PRESENCE_INVENTED:
                state.events.append(
                    AuditEvent(
                        page_num=page_num,
                        kind="equation_region_reading_rejected",
                        engine="native+equations",
                        detail=(
                            f"region {region_index} reading rejected by the numeric-presence "
                            f"guard ({verdict.reason}); the page keeps its native prose and "
                            "is NOT demoted"
                        ),
                        data={
                            "region_index": region_index,
                            "crop_path": crop_path,
                            "model_id": model,
                            "presence_status": verdict.status,
                            "presence_reason": verdict.reason,
                            "invented": list(verdict.invented),
                            "oracle_size": verdict.oracle_size,
                            "raw_latex": result.raw_latex,
                        },
                    )
                )
                continue

            # GH-522: an UNVERIFIABLE verdict is an ABSTENTION, not a pass. The
            # guard returns it when the page has no numeric oracle, or when the
            # text layer shows decode damage (ruling 4: never FAIL on damaged
            # text). Attaching regardless put a crop-backed LaTeX sidecar --
            # possibly carrying invented values -- beside the authoritative
            # native slice, tagged only by `presence_status` on an audit event
            # that a reader of the .md never sees.
            #
            # Refused under two narrowings, both load-bearing (GH-543 corrected
            # the first, which this lane shipped too broad):
            #
            #  - only when the TEXT LAYER IS DAMAGED. UNVERIFIABLE also covers
            #    "the page has no numeric oracle", and there a numeral in the
            #    reading is usually NOTATION -- the 2 in `E = mc^2`, an equation
            #    tag -- not a data value. Refusing on that convicts notation-only
            #    LaTeX on prose pages. The damaged-text case is the one this
            #    ticket is about: an oracle exists but cannot be trusted.
            #  - only when the reading HAS numeric tokens. A pure-symbol equation
            #    carries nothing that can be invented, and dropping those would
            #    discard safe LaTeX -- the "dropped is worse than missing" half
            #    of the corpus rule.
            #
            # So the test is the CANDIDATE's own numbers, read with the same
            # extractor the guard itself uses, not the verdict alone.
            #
            # The crop stays on disk exactly as for a rejection, so the reading
            # remains available to anyone who wants to check it by hand. The page
            # is NOT demoted and native prose stays: this removes an unchecked
            # addition, it does not take anything away.
            if verdict.status == PRESENCE_UNVERIFIABLE and (encoding_suspect or corrupt_math):
                unchecked = text_value_tokens(result.raw_latex)
                if unchecked:
                    state.events.append(
                        AuditEvent(
                            page_num=page_num,
                            kind="equation_region_reading_unverifiable",
                            engine="native+equations",
                            detail=(
                                f"region {region_index} reading NOT attached: it carries "
                                f"{sum(unchecked.values())} numeric value(s) the presence "
                                f"guard could not check ({verdict.reason}); the page keeps "
                                "its native prose and is NOT demoted"
                            ),
                            data={
                                "region_index": region_index,
                                "crop_path": crop_path,
                                "model_id": model,
                                "presence_status": verdict.status,
                                "presence_reason": verdict.reason,
                                "unchecked_values": sorted(unchecked.elements()),
                                "oracle_size": verdict.oracle_size,
                                "raw_latex": result.raw_latex,
                            },
                        )
                    )
                    continue

            result.presence_status = verdict.status
            result.presence_reason = verdict.reason
            attachable.append(result)

        # Cold review round 1, finding 3: every executed call is metered before
        # any early return below, so a rejected or unaligned reading cannot
        # erase the spend it cost.
        self._meter_equation_lane_calls(state, ps, page_num, profile, model, calls_made)

        if not attachable:
            return

        text, unaligned = attach_equation_sidecars_in_place(native_text, attachable)
        unaligned_set = set(unaligned)
        for result in attachable:
            if result.region_index in unaligned_set:
                state.events.append(
                    AuditEvent(
                        page_num=page_num,
                        kind="equation_region_reading_unaligned",
                        engine="native+equations",
                        detail=(
                            f"region {result.region_index}'s native slice was not found in "
                            "the page text, so its reading was dropped rather than appended "
                            "somewhere it does not belong"
                        ),
                        data={
                            "region_index": result.region_index,
                            "crop_path": result.crop_path,
                            "model_id": model,
                            "source_text": result.source_text,
                        },
                    )
                )
            else:
                state.events.append(
                    AuditEvent(
                        page_num=page_num,
                        kind="equation_region_reading_attached",
                        engine="native+equations",
                        detail=(
                            f"region {result.region_index}: crop-backed, 1A-validated LaTeX "
                            "attached beside its own native slice; the native text is "
                            "retained and remains authoritative"
                        ),
                        data={
                            "region_index": result.region_index,
                            "crop_path": result.crop_path,
                            "crop_ref": result.crop_ref,
                            "model_id": model,
                            "presence_status": getattr(result, "presence_status", ""),
                            "presence_reason": getattr(result, "presence_reason", ""),
                            "validation_scope": "latex_syntax_only",
                            "semantic_fidelity_verified": False,
                        },
                    )
                )

        if text == native_text:
            return

        # A passing winner, never a demotion: status / audit_passed / failure_mode
        # are carried from the native floor rather than restated, so this lane
        # cannot upgrade or demote a page relative to the lane-off run.
        #
        # ``cost_usd`` carries the metered figure, never None: the resume path
        # rebuilds an ``EngineResult`` from this field, and an unknown there
        # makes the whole run's total cost unknown and fails every later paid
        # rung closed (finding 3).
        attached_out = replace(
            baseline,
            text=text,
            engine="native+equations",
            provider_id=(getattr(profile, "id", "") or "equation-region-lane"),
            provider_model=model,
            provider_backend=(getattr(profile, "backend", "") or "ollama-compatible"),
            cost_usd=self._equation_lane_call_cost(profile, calls_made),
        )
        ps.attempts.append(attached_out)
        ps.best_output = attached_out

    def _meter_equation_lane_calls(
        self,
        state: DocumentState,
        ps: PageState,
        page_num: int,
        profile,
        model: str,
        calls_made: int,
    ) -> None:
        """Record the lane's executed model calls in ``state.engine_runs``.

        Cold review round 1, finding 3. Without this the calls were invisible to
        ``--cost-budget``, ``--max-cost-per-page``, the reported total, the
        metadata and the resume ledger. A local rung costs 0.00 -- a KNOWN
        number -- so recording it never makes the total unknown, and recording
        it is what keeps a resumed run's arithmetic identical to a live one.

        Also stamps the cost on the page's current output so a page whose
        reading was REJECTED still carries the spend the refused call cost; the
        resume path rebuilds its ``EngineResult`` from that field.
        """
        if calls_made <= 0:
            return
        cost = self._equation_lane_call_cost(profile, calls_made)
        if ps.best_output is not None:
            ps.best_output.cost_usd = cost
        state.record_engine_run(
            EngineResult(
                document_path=state.handle.path,
                engine="native+equations",
                status=DocumentStatus.SUCCESS,
                pages=[ps.best_output] if ps.best_output is not None else [],
                model_version=model,
                cost=cost,
                pages_processed=1,
                audit_passed=True,
            ),
            page_nums=[page_num],
        )

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

        Exceptions from open_pdf or has_chart_marks propagate to the sole caller
        (_phase_agentic's chart-eligibility scan) for audited fail-soft handling.
        """
        # Must be native-eligible first (chart lane is a sub-case of native).
        if not self._is_native_eligible_without_ocr(page_num, ps):
            return False
        # Open the PDF page and run the vector detector.
        from socr.core.pdf import open_pdf

        with open_pdf(pdf_path) as doc:
            page = doc[page_num - 1]
            return has_chart_marks(page)

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

            from socr.core.pdf import open_pdf

            with open_pdf(pdf_path) as doc:
                page = doc[page_num - 1]
                # GH-307: the chart PNG is the ONLY artifact carrying this page's
                # visual semantics -- the lane records "data values not
                # transcribed" and hands the reader the image instead. A sideways
                # image is the whole payload, rendered unreadable.
                from socr.core.born_digital import upright_rotation_for

                mat = fitz.Matrix(RENDER_DPI / 72, RENDER_DPI / 72)
                rotation = upright_rotation_for(page)
                if rotation:
                    mat = mat.prerotate(rotation)
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

            from socr.core.pdf import open_pdf
            from socr.figures.extractor import RENDER_DPI
            from socr.tables.reconstruct import chart_region_bboxes

            figures_dir.mkdir(parents=True, exist_ok=True)

            with open_pdf(str(pdf_path)) as _doc:
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
                    with open_pdf(str(pdf_path)) as _doc:
                        _page = _doc[page_num - 1]
                        # GH-307: clip-scoped, matching the crop path -- a region
                        # can run against the page's dominant direction.
                        from socr.core.born_digital import upright_rotation_for

                        mat = fitz.Matrix(RENDER_DPI / 72, RENDER_DPI / 72)
                        _rotation = upright_rotation_for(_page, clip=bbox)
                        if _rotation:
                            mat = mat.prerotate(_rotation)
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
        *,
        stem: str = "failed_table",
        label: str = "Failed table page",
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
            fname = f"{stem}_p{page_num}.png"
            saved = self._render_chart_page_png(pdf_path, page_num, figures_dir)
            # _render_chart_page_png saves chart_page_{page_num}.png; rename to
            # our floor-specific name so the file is clearly identifiable in
            # figures/. ``stem``/``label`` distinguish the TR-3 table floor from
            # #263's rotated-shred floor: the two mean different things and must
            # not overwrite each other's PNG on a page that reaches both.
            saved_path = Path(saved)
            d3_path = saved_path.parent / fname
            saved_path.rename(d3_path)
            figures_dir_name = figures_dir.name
            ref = f"![{label} {page_num}]({figures_dir_name}/{fname})"
            logger.debug("TR-3 D3 floor: saved full-page PNG %s", d3_path)
            return ref
        except Exception as exc:
            logger.warning(
                "TR-3 D3 floor PNG render failed for p%d (%s); marker only",
                page_num,
                exc,
            )
            return ""

    def _apply_scanned_table_floor(
        self,
        ps: PageState,
        pdf_path: Path,
        page_num: int,
        figures_dir: Path | None,
    ) -> None:
        """GH-90: mark a scanned page whose table evidence failed for the D3 floor.

        Sets the flags and demotes ``best_output`` — nothing else. The demotion
        is what lets ``_select_page_output_tagged`` fall past its passing-winner
        return into the GH-90 branch, which is the SOLE writer of the floor text
        (regional splice, whole-page marker fallback). GH-371: mutating
        ``best_output.text`` here too would strip the tables before that branch
        re-reads them, so its splice would find no blocks and fall back to the
        whole-page marker, discarding the prose.
        """
        ps.scanned_table_evidence_failed = True
        if figures_dir is not None:
            ps.d3_floor_png_ref = self._render_d3_floor_png(
                pdf_path,
                page_num,
                figures_dir,
            )
        if ps.best_output is not None:
            ps.best_output.status = PageStatus.ERROR
            ps.best_output.audit_passed = False
            ps.best_output.failure_mode = FailureMode.HALLUCINATION

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

        # GH-225: provenance gate.  strip_phantom_images cannot adjudicate an
        # absolute URL — it has no view of the source — so a fabricated
        # ``![](https://i.imgur.com/…)`` survives it and every other gate,
        # because a hyperlink is text.  This is the one FABRICATION check in a
        # pipeline whose other gates all check for loss, so it runs on the same
        # seam, before the provisional flush, and demotes the page.
        self._guard_fabricated_image_refs(state, page_num, page_out, doc_dir)

    def _guard_fabricated_image_refs(
        self,
        state: DocumentState,
        page_num: int,
        page_out: PageOutput,
        doc_dir: Path | None,
    ) -> None:
        """Remove image refs with no provenance in the source, loudly (GH-225).

        Deterministic and model-free: legitimacy is decided by the document
        itself (its link annotations and text layer) plus the asset directory
        socr wrote, never by a host allowlist and never by a network fetch —
        reachability is not provenance.  See ``socr.core.url_provenance``.

        The page is DEMOTED, not merely annotated.  #225's whole point is that
        two pages shipped invented content under ``status: success`` with
        ``judge_rejected: false`` and, on one of them, zero audit events; a
        marker alone would leave a consumer gating on status none the wiser.
        Demotion also makes the resume ledger refuse the page on a re-run
        (GH-161), so the fabrication is re-OCR'd rather than restored.

        Removal is content-safe: an image ref is a pure pointer, and a pointer
        to an asset that does not exist carries nothing.  The invented target
        and its alt text are preserved in the audit event, so nothing is lost
        for forensics — only for the shipped corpus, which is the point.
        """
        text = page_out.text or ""
        if "![" not in text:
            return
        try:
            from socr.core.url_provenance import redact_fabricated_image_refs

            cleaned, removed = redact_fabricated_image_refs(
                text,
                source_urls=self._source_url_index(state),
                doc_dir=doc_dir,
            )
        except Exception as exc:  # a gate failure must never drop the page
            logger.warning("GH-225: provenance gate errored on p%d (%s); text kept", page_num, exc)
            return
        if not removed:
            return

        page_out.text = cleaned

        from socr.core.audit_log import AuditEvent
        from socr.core.result import FailureMode

        targets = [entry["target"] for entry in removed]
        state.events.append(
            AuditEvent(
                page_num=page_num,
                kind="fabricated_image_ref",
                engine=page_out.engine or "",
                detail=(
                    f"removed {len(removed)} image reference(s) absent from the source "
                    "document's link annotations, text layer and asset directory"
                ),
                data={"targets": targets, "alts": [entry["alt"] for entry in removed]},
            )
        )
        # #225 asks to FAIL the page, not warn on it, and the round-2 review is
        # right that my reason for softening that to WARNING was a false
        # dichotomy.  ERROR does NOT delete the cleaned text: winner selection is
        # controlled by ``audit_passed`` (which stays True — see below), and
        # ``failed_pages`` is derived from a page-failure MARKER in the body
        # text, not from status, so an ERROR page with real text is never
        # substituted for a marker.  Verified on the born-digital shape: the
        # table still ships.  So the criterion is met as written.
        #
        # ``audit_passed`` is deliberately NOT flipped, and this is the whole
        # correctness of the demotion.  In this codebase ``audit_passed=False``
        # on ``best_output`` does not mean "flag this page" — it means "this
        # attempt is not the winner": ``_winning_page_output`` (manifest.py:314)
        # returns ``best_output`` only while it is True, and otherwise falls
        # through to the born-digital native-text branch.  Flipping it here
        # therefore threw away the CLEANED OCR page and shipped flattened native
        # text under a fresh SUCCESS — on #225's own document class, the OBR
        # born-digital pages.  The invented URL went, and the extracted table
        # went with it, and the page was restamped clean.  That is a worse
        # silent content loss than the fabrication this gate exists to remove.
        #
        # So the winner stays the winner and carries the demotion itself, and
        # the document-level signal rides on ``fabricated_image_refs`` below
        # rather than on the winner-selection flag.
        if page_out.status == PageStatus.SUCCESS:
            page_out.status = PageStatus.ERROR
        if page_out.failure_mode == FailureMode.NONE:
            page_out.failure_mode = FailureMode.HALLUCINATION
        ps = state.pages.get(page_num)
        if ps is not None:
            ps.fabricated_image_refs = getattr(ps, "fabricated_image_refs", 0) + len(removed)
        if not self.config.quiet:
            console.print(
                f"  [red]p{page_num}: {len(removed)} fabricated image reference(s) removed "
                f"(no provenance in the source)[/red]"
            )

    def _source_url_index(self, state: DocumentState) -> frozenset[str]:
        """Per-document cache of the source PDF's own URLs (GH-225).

        Indexing walks every page's link annotations and text layer, so it is
        computed once per document rather than once per page.  Keyed by the
        resolved input path so a batch run does not reuse one document's index
        for the next.
        """
        from socr.core.url_provenance import source_url_index

        key = str(state.handle.path)
        cached = getattr(self, "_source_url_cache", None)
        if cached is None:
            cached = {}
            self._source_url_cache = cached
        if key not in cached:
            cached[key] = source_url_index(state.handle.path)
        return cached[key]

    def _guard_fabricated_image_refs_document(
        self, state: DocumentState, final_text: str, doc_dir: Path | None
    ) -> str:
        """Document-level provenance sweep during assembly (GH-225).

        The page-level agentic loop demotes the individual page; the assembly
        pass also records the document-level signal
        against page 0 (the document) and surfaces through ``metadata.json`` via
        ``_fabricated_url_note``.  Stated plainly because it is a real gap: on
        these lanes the fabrication is caught and removed, and the document is
        flagged, but no individual PAGE status is demoted.
        """
        if "![" not in final_text:
            return final_text
        try:
            from socr.core.url_provenance import redact_fabricated_image_refs

            cleaned, removed = redact_fabricated_image_refs(
                final_text,
                source_urls=self._source_url_index(state),
                doc_dir=doc_dir,
            )
        except Exception as exc:  # never lose the document over a gate
            logger.warning("GH-225: document provenance gate errored (%s); text kept", exc)
            return final_text
        if not removed:
            return final_text

        from socr.core.audit_log import AuditEvent

        state.events.append(
            AuditEvent(
                page_num=0,
                kind="fabricated_image_ref",
                engine=", ".join(state.engines_used) if state.engines_used else "",
                detail=(
                    f"removed {len(removed)} image reference(s) absent from the source "
                    "document's link annotations, text layer and asset directory"
                ),
                data={
                    "targets": [entry["target"] for entry in removed],
                    "alts": [entry["alt"] for entry in removed],
                    "scope": "document",
                },
            )
        )
        if not self.config.quiet:
            console.print(
                f"  [red]{len(removed)} fabricated image reference(s) removed from the "
                f"document (no provenance in the source)[/red]"
            )
        return cleaned

    @staticmethod
    def _fabricated_url_note(events: list) -> str | None:
        """Document-level one-liner naming pages that carried invented refs.

        Mirrors ``_repetition_truncated_note``: a consumer gating on
        ``metadata.json`` must see that a page shipped fabricated content
        without parsing the full audit log.  ``None`` on a clean run.
        """
        pages = sorted({e.page_num for e in events if e.kind == "fabricated_image_ref"})
        if not pages:
            return None
        # Page 0 is the document-level sweep, which has no page to name; render
        # it as such rather than as a page called "0".
        labels = ["document" if n == 0 else str(n) for n in pages]
        return (
            f"page(s) {', '.join(labels)}: "
            "fabricated image reference(s) removed (no provenance in the source document)"
        )

    @staticmethod
    def _chart_detection_failed_note(state) -> str | None:
        """Document-level one-liner naming pages whose chart routing never resolved.

        Mirrors ``_fabricated_url_note``: a consumer gating on ``metadata.json``
        must see that a page's chart-vs-table routing decision was never made,
        without parsing the full audit log.  Read from the PageState flag rather
        than ``state.events`` because the flag survives resume
        (``_restore_terminal_page_state``) while the events list does not.
        ``None`` on a clean run.
        """
        pages = sorted(
            n for n, p in state.pages.items() if getattr(p, "chart_asset_detection_failed", False)
        )
        if not pages:
            return None
        return (
            f"page(s) {', '.join(str(n) for n in pages)}: "
            "chart eligibility detection failed; page took the non-chart route with "
            "its chart-vs-table routing unresolved (content preserved)"
        )

    @staticmethod
    def _visual_values_split(state) -> tuple[list[int], list[int]]:
        """GH-519/566: (pages whose figure IS in a saved PNG, pages where it is not).

        One derivation for the document note and the CLI summary alike. #566 is
        why: the cubic P2 fix on #565 split the note on ``png_saved`` and left
        the CLI line printing "preserved in the page image only" for every debt
        page, render failures included -- the same false comfort, one surface
        along. Two copies of a sentence drift; one cannot.

        A page that saved its PNG on one event and failed on another counts as
        lost: the harsher sentence wins.
        """
        preserved: set[int] = set()
        lost: set[int] = set()
        for ev in state.events:
            if getattr(ev, "kind", "") != VISUAL_VALUES_NOT_TRANSCRIBED_KIND:
                continue
            data = getattr(ev, "data", None) or {}
            (preserved if data.get("png_saved") else lost).add(ev.page_num)
        preserved -= lost
        return sorted(preserved), sorted(lost)

    @staticmethod
    def _visual_values_not_transcribed_note(state) -> str | None:
        """GH-519: name the pages whose figure text ships only as an image.

        Mirrors ``_chart_detection_failed_note``: a consumer gating on
        ``metadata.json`` must see the debt without opening
        ``audit_log.json``. Read from the audit events rather than a PageState
        flag because this kind is replayed by ``_restore_terminal_page_state``
        (see ``resume_restore_kinds``), so the events are the durable record --
        the flag route is what GH-563 had to undo. ``None`` on a run with no
        chart-asset page.
        """
        preserved, lost = UnifiedPipeline._visual_values_split(state)
        if not preserved and not lost:
            return None

        parts = []
        if preserved:
            parts.append(
                f"page(s) {', '.join(str(n) for n in preserved)}: visual values not "
                "transcribed; in-image text on these figures is preserved in the page image "
                "only (no model read it)"
            )
        if lost:
            # cubic P2 on #565. The debt event fires on the render-failure path
            # too, where NO page PNG was saved -- so "preserved in the page
            # image" is false comfort, and worse than the debt it describes.
            # These pages are already WARNING with a chart_asset_render_failed
            # event; the note must not quietly upgrade them.
            parts.append(
                f"page(s) {', '.join(str(n) for n in lost)}: visual values not "
                "transcribed AND the page image was not saved; in-image text on these "
                "figures is preserved nowhere"
            )
        return "; ".join(parts)

    @staticmethod
    def _unverified_wording_split(state, pages: list[int]) -> tuple[list[int], list[int]]:
        """Split TABLE_UNVERIFIED pages into (retryable, unwitnessed).

        GH-563: derived from the page's own ``table_ladder_unverified`` events,
        never from in-run PageState flags. #562 used flags, and they did not
        survive a resume -- the sidecar persists ``table_ladder_incomplete`` and
        not those -- so a SKIPPED no-witness page came back with empty flags and
        the note fell through to "retryable on resume". That is the exact empty
        promise #560 was filed against, re-told to the one operator least able
        to check it.

        The events are the durable record: ``_restore_terminal_page_state``
        replays every ``TABLE_LADDER_EVENT_KINDS`` event with its ``data``
        intact, so ``retryable: False`` is present on resume as it was on the
        first run. One source, and the resumed run reads what the original run
        wrote.

        A page with no unverified event at all -- a disposition reached through
        ``best_output.failure_mode`` with no terminal of its own -- keeps the
        retryable wording, which is what it said before any of this.
        """
        from socr.judge.table_verdict import TABLE_LADDER_UNVERIFIED_KIND

        by_page: dict[int, list] = {}
        for ev in state.events:
            if getattr(ev, "kind", "") == TABLE_LADDER_UNVERIFIED_KIND:
                by_page.setdefault(getattr(ev, "page_num", 0), []).append(ev)

        retryable: list[int] = []
        unwitnessed: list[int] = []
        for page_num in pages:
            events = by_page.get(page_num) or []
            marked = [(getattr(ev, "data", None) or {}).get("retryable") is False for ev in events]
            if any(marked):
                unwitnessed.append(page_num)
            # A page can hold BOTH kinds, and then both sentences are true of
            # it (cubic P2 on #562). No event at all falls here too.
            if not marked or not all(marked):
                retryable.append(page_num)
        return retryable, unwitnessed

    @staticmethod
    def _table_judge_ladder_note(state) -> str | None:
        """GH-353: document-level one-liner naming the table judge ladder terminals.

        Mirrors ``_chart_detection_failed_note``: a consumer gating on
        ``metadata.json`` must see a REJECTED/UNVERIFIED table without parsing
        the full audit log. Uses ``_table_ladder_terminal`` -- the same
        disposition-first, ``best_output.failure_mode``-fallback predicate the
        ``table_rejected_pages``/``table_unverified_pages`` buckets in
        ``_phase_assemble`` use, so the note and the document-status
        aggregation can never disagree about which pages are affected. Names
        both terminal modes by their ``FailureMode`` value so the CLI summary
        (which prints ``result.error`` verbatim) surfaces the exact terminal,
        not a paraphrase. ``None`` on a clean run.
        """
        rejected = sorted(
            n
            for n, p in state.pages.items()
            if _table_ladder_terminal(p) == FailureMode.TABLE_REJECTED
        )
        unverified = sorted(
            n
            for n, p in state.pages.items()
            if _table_ladder_terminal(p) == FailureMode.TABLE_UNVERIFIED
        )
        if not rejected and not unverified:
            return None
        parts = []
        if rejected:
            parts.append(
                f"page(s) {', '.join(str(n) for n in rejected)}: "
                f"{FailureMode.TABLE_REJECTED.value} (table judge ladder rejected; "
                "not retryable)"
            )
        # GH-560: a page whose UNVERIFIED terminal came from a table with no
        # witness at all is not retryable -- no rung was ever asked, and a
        # re-run reaches the same empty witness. Split so neither group carries
        # the other's claim. The BUCKET is untouched: this is wording only, and
        # the page is still TABLE_UNVERIFIED for document status.
        retryable, unwitnessed = UnifiedPipeline._unverified_wording_split(state, unverified)
        if retryable:
            parts.append(
                f"page(s) {', '.join(str(n) for n in retryable)}: "
                f"{FailureMode.TABLE_UNVERIFIED.value} (table judge ladder exhausted "
                "without an answer; retryable on resume)"
            )
        if unwitnessed:
            parts.append(
                f"page(s) {', '.join(str(n) for n in unwitnessed)}: "
                f"{FailureMode.TABLE_UNVERIFIED.value} (no table witness could be "
                "prepared, so no rung ran; not retryable)"
            )
        return "; ".join(parts)

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

    @staticmethod
    def _route_page_table_escalation_signal(decision, ladder: list) -> bool:
        """Return route evidence that should join the table score signal.

        ``route_page`` deliberately owns provider/judge escalation, while the
        table lane runs immediately after it.  A native verifier rejection that
        caused another rung to run is therefore useful evidence even when the
        final rung happens to look perfect to the native-text score.  Likewise,
        an exhausted ladder on a table page should get the bounded crop/tool
        opportunity before the later table-judge terminal is produced.

        This reads the route decision and attempt reasons only.  In particular,
        ``PageOutput.audit_passed`` is a winner-selection field, not a routing
        signal, and must not be used here.
        """
        attempts = getattr(decision, "attempts", ())
        if any(
            not attempt.accepted and (attempt.reason or "").startswith("native_table_verifier:")
            for attempt in attempts[:-1]
        ):
            return True
        return bool(ladder) and not decision.accepted and len(attempts) >= len(ladder)

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
    ) -> bool:
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
        prevent. A chart page reaches it only if it still has a table signal:
        a chart WINNER is by construction a ``chart_only`` page (``has_tables``
        False), while a chart page that also carries a table is arbitrated to
        the table lane and keeps the not-scorable surface. The caller adds the
        pages the GH-96 escalation lane would otherwise have scored itself, so
        the coverage does not depend on whether that lane is live.
        """
        try:
            from socr.core.pdf import open_pdf

            with open_pdf(state.handle.path) as doc:
                page = doc[page_num - 1]
                return self._table_page_needs_escalation(state, page_num, page, ps, bo)
        except Exception as exc:
            logger.warning(
                "table scoring failed on p%d (%s); no distrust events emitted",
                page_num,
                exc,
            )
            return False

    def _escalate_table_page(
        self,
        state: DocumentState,
        page_num: int,
        ps,
        bo: PageOutput,
        profile,
        run_provider,
        pdf_path,
        *,
        needs_escalation: bool | None = None,
    ) -> tuple[bool, PageOutput]:
        """Re-read one table page with *profile*; keep it only if it measures better.

        Returns whether the lane should be disabled for the rest of the document
        (a wedged provider), plus the output the caller must use downstream.

        Every rejection path returns the incumbent untouched. Acceptance promotes
        the candidate object itself so text, provenance, cost, and audit metadata
        remain one atomic engine result and the incumbent attempt remains historical
        evidence rather than being mutated in place.
        """
        import concurrent.futures

        from socr.core.audit_log import AuditEvent
        from socr.tables.escalation_decision import decide_escalation

        try:
            from socr.core.pdf import open_pdf

            with open_pdf(pdf_path) as doc:
                page = doc[page_num - 1]
                if needs_escalation is None:
                    needs_escalation = self._table_page_needs_escalation(
                        state, page_num, page, ps, bo
                    )
                if not needs_escalation:
                    return False, bo
                incumbent_text = bo.text or ""

                # A cloud CLI has no timeout of its own and was observed wedged for
                # 97 minutes. Escalation runs inline in the page-major loop, so an
                # unbounded call stalls the entire document. The worker is released
                # rather than joined: the subprocess may outlive us, but the loop
                # must not.
                deadline = float(getattr(self.config, "escalation_timeout_sec", 120.0))
                ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                future = ex.submit(run_provider, profile, page_num)
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
                    return True, bo

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
                    return False, bo
                if out.status is not PageStatus.SUCCESS or not (out.text or "").strip():
                    state.events.append(
                        AuditEvent(
                            page_num=page_num,
                            kind="table_escalation_refused",
                            engine=profile.engine.value,
                            detail=f"candidate status={out.status}, no usable text",
                        )
                    )
                    return False, bo

                decision = decide_escalation(page, incumbent_text, out.text)

            # Cost is recorded by hand: `route_page` does this for ladder calls, and
            # a bare `run_provider` does not, so without it the document
            # under-reports what it spent. Recorded against the PAGE here, ABOVE
            # the accept/reject branch: a refused candidate is never appended to
            # ``ps.attempts``, so this is the only place its real spend is seen
            # (round 5).
            state.record_engine_run(
                EngineResult(
                    document_path=pdf_path,
                    engine=profile.engine.value,
                    status=DocumentStatus.SUCCESS,
                    pages=[],
                    pages_processed=1,
                    cost=profile.cost_per_page_usd,
                ),
                page_nums=[page_num],
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
                return False, bo

            out.escalated_from = bo.engine
            out.cost_usd = profile.cost_per_page_usd
            out.provider_id = profile.id
            # GH-370: record the backend/model that ACTUALLY ran, not the
            # registry's descriptive label. For QWEN the two diverge whenever
            # --qwen-backend is not ollama.
            out.provider_backend, out.provider_model = resolved_provenance(profile, self.config)
            ps.attempts.append(out)
            ps.best_output = out
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
            return False, out

        except Exception as exc:  # a failed escalation must never lose a page
            logger.warning("table escalation failed on p%d (%s); keeping text", page_num, exc)
            return False, bo

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
                "native_table_emission_defect",  # GH-226 exact provenance
                "native_table_content_defect",  # GH-303, added by GH-346
                "native_table_header_unattributed",  # GH-200
                "native_table_unverifiable",
                "scanned_table_evidence_failed",
            )
            if getattr(ps, name, False)
        ]
        if not cleared:
            return
        for name in cleared:
            # GH-346: the two defect terms are strings and clear to ""; the rest
            # are booleans. native_table_content_defect was added as a sibling
            # of the emission term in GH-303 but never added here, so a
            # recovered page kept empty-table provenance the clear path claimed
            # to have released.
            setattr(
                ps,
                name,
                ""
                if name in ("native_table_emission_defect", "native_table_content_defect")
                else False,
            )
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
        instance found since, came from the per-page VLM loop. Native page text is
        left to ``OutputNormalizer``'s generic rule because repeated native rows
        can be legitimate content.

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
    # GH-353 TICKET-B1: the table judge ladder gate.
    #
    # Content terminals are produced post-repetition-guard in the agentic
    # loop (B0 witnesses -> A4 ladder -> A1 events -> PageState disposition).
    # Assemble is a completeness backfill for missing events, not a second
    # judge. Native lane included; ``engine == "chart_asset"`` still skips
    # inside the helper (B1); assemble catches residual markdown tables.
    # ------------------------------------------------------------------

    def _build_table_judge_rungs(self) -> list:
        """Construct the ladder's rung sequence once per document.

        Returns ``[]`` when the ladder flag is off, or when ``strict_local``
        is set: both rungs are cloud (ollama-cloud, gemini CLI), so
        ``strict_local and table_judge_ladder`` makes every rung unavailable
        BEFORE the first call (G1's documented interaction) -- an empty rung
        list is the fail-open signal ``_run_table_judge_gate`` reads to
        UNVERIFY every witnessed table on this run without ever attempting
        cloud egress, rather than constructing a rung that would immediately
        refuse. A separate method (not inlined in ``_phase_agentic``) so
        tests can override it to inject fake ``RungCallable``s without a
        live ollama daemon or a ``gemini`` binary on disk.
        """
        if not self.config.table_judge_ladder or self.config.strict_local:
            return []

        from socr.judge.table_rung_gemini import make_gemini_rung
        from socr.judge.table_rung_ollama import build_ollama_rung

        return [
            build_ollama_rung(
                self.config.table_judge_rung1_model,
                self.config.table_judge_rung1_host,
                self.config.table_judge_timeout_sec,
            ),
            make_gemini_rung(self.config),
        ]

    #: The rung kinds the gate builds, in ladder order. Named here so the
    #: "which rung recovered" question has one list to widen to.
    TABLE_JUDGE_RUNG_KINDS: tuple[str, ...] = (RUNG_KIND_OLLAMA, RUNG_KIND_GEMINI)

    def _table_judge_rung_available_now(self, rung_kinds: list[str] | None = None) -> bool:
        """Whether a table-judge rung worth retrying is attemptable now.

        Cold review round 2, finding 1: the gate's notion of reachability and
        the rungs' notion of "unavailable" must be the SAME notion, or this
        function authorizes a state transition its own evidence cannot
        support. It used to answer True on ``shutil.which`` alone and on a
        bare ``/api/tags`` liveness ping -- both of which say yes to a rung
        that is guaranteed to fail again.

        Cold review round 3, finding 1: and it must be asked about the RIGHT
        rung. ``rung_kinds`` names the kinds the latch recorded as unavailable;
        only those are asked. Empty or None means the record does not say, and
        the question widens to "any rung", which is what an entry written
        before the rung list existed deserves.

        Each rung kind owns one cheap, no-model reachability function:

        * ``ollama`` -- daemon up AND the configured judge model actually pulled.
        * ``gemini`` -- on PATH AND a trivial no-model health invocation succeeds.

        Both rungs are cloud rungs, so the ladder and ``strict_local`` gates
        are checked before touching either external seam.
        """
        if not self.config.table_judge_ladder or self.config.strict_local:
            return False

        kinds = [k for k in (rung_kinds or []) if k]
        # GH-554: a kind with no probe of its own -- the ``"unknown"`` the
        # whole-ladder guard synthesizes, or an injected test rung -- can never
        # be shown reachable. Keeping it as the ONLY question is not the
        # conservative reading it looks like: it latches the skip permanently
        # shut, even when every real rung is back. Such a record says no more
        # than an empty one does, so it widens the same way. A record that also
        # names a real kind keeps that kind alone -- the unknown adds no
        # information, the real kind does.
        probeable = [k for k in kinds if k in self.TABLE_JUDGE_RUNG_KINDS]
        kinds = probeable or list(self.TABLE_JUDGE_RUNG_KINDS)
        return any(self._table_judge_rung_kind_available_now(kind) for kind in kinds)

    def _table_judge_rung_kind_available_now(self, kind: str) -> bool:
        """Reachability of ONE rung kind, probed at most once per run.

        The resume gate asks per file and the batch pre-gate per candidate: a
        subprocess health check plus an HTTP model listing must not be paid per
        document. The cache is cleared at every public run boundary
        (``process`` / ``process_batch``), never at construction only, so a
        long-lived pipeline object cannot carry a stale "unreachable" from one
        run into the next (cold review round 3, new finding 1).
        """
        if kind in self._table_rung_refused_this_run:
            # Cold review round 3, new finding 2: this rung already refused us
            # for a recognised external reason on a REAL call in this run. It
            # will refuse the next table identically, so the rest of the run
            # treats it as unreachable. The page latch still persists, so a
            # LATER run retries it.
            return False
        cached = self._table_rung_available_cache.get(kind)
        if cached is None:
            cached = self._probe_table_judge_rung_kind(kind)
            self._table_rung_available_cache[kind] = cached
        return cached

    def _probe_table_judge_rung_kind(self, kind: str) -> bool:
        """The uncached, per-kind probe. Best effort: an unexpected exception
        means that rung is not attemptable, never a broken run."""
        try:
            if kind == RUNG_KIND_GEMINI:
                return bool(table_judge_gemini_rung_reachable(self.config.table_judge_rung2_binary))
            if kind == RUNG_KIND_OLLAMA:
                return bool(
                    table_judge_ollama_rung_reachable(
                        self.config.table_judge_rung1_model,
                        self.config.table_judge_rung1_host,
                    )
                )
        except Exception as exc:
            logger.debug("table-judge rung %r availability probe failed: %s", kind, exc)
            return False
        # An unrecognised kind (an injected test rung, or a synthesized
        # "unknown" from the whole-ladder guard) has no probe of its own, so it
        # cannot be shown reachable and never reopens a document on its own.
        # Nor does it veto the kinds that can be probed: the caller drops
        # unprobeable kinds before asking, and widens to every configured kind
        # when nothing probeable is left (GH-554).
        logger.debug("table-judge rung kind %r has no reachability probe", kind)
        return False

    def _reset_table_judge_rung_probes(self) -> None:
        """Start a fresh reachability epoch for one public run.

        Cold review round 3, new finding 1. ``process_batch`` is ONE run: it
        resets once and its nested ``process`` calls do not reset again, so the
        pre-gate's answer and the per-file answers agree for the whole batch.
        A bare ``process`` call is its own run and resets on entry.
        """
        self._table_rung_available_cache = {}
        self._table_rung_refused_this_run = set()
        self._table_rung_refused_callables = []
        self._table_rung_seen_callables = []

    def _executing_identity(self, rr, executed_rungs: list, index: int) -> str:
        """Who actually ran this rung result. Identity first, position last.

        Resolution order (cold review round 5):

        1. The tag the rung callable carries (``executing``), matched to the
           callable that produced THIS result -- ``run_table_ladder`` appends
           one result per call, in order, so within the list actually executed
           the index is a true identity. That list is not the configured
           ladder once the breaker has filtered it.
        2. The rung KIND, mapped to the configured identity for that kind.
           This covers a synthesized refused result, which had no executor.
        3. The callable's position in the list handed to the gate, which is the
           historical rule and stays correct for an unfiltered ladder of rungs
           that do not advertise themselves.

        An unrecognised rung with no tag and no position resolves to "" rather
        than borrowing a configured identity it never had.
        """
        if index < len(executed_rungs):
            tagged = getattr(executed_rungs[index], "executing", "") or ""
            if tagged:
                return tagged

        kind = rung_kind(rr.rung or "")
        if kind == RUNG_KIND_OLLAMA:
            return self.config.table_judge_rung1_model
        if kind == RUNG_KIND_GEMINI:
            return self.config.table_judge_rung2_binary

        if index < len(executed_rungs):
            configured = self._table_judge_rung_position(executed_rungs[index])
            if configured == 0:
                return self.config.table_judge_rung1_model
            if configured == 1:
                return self.config.table_judge_rung2_binary
        return ""

    def _table_judge_rung_position(self, rung) -> int:
        """The callable's index in the rung list this gate call was given.

        Set for the duration of one ``_run_table_judge_gate`` call; -1 when the
        callable is not one of them.
        """
        for position, candidate in enumerate(self._table_judge_gate_rungs):
            if candidate is rung:
                return position
        return -1

    def _table_rung_callable_refused(self, rung) -> bool:
        """Whether this rung callable is off the table for the rest of this run.

        Cold review round 5. Two questions, answered in order:

        1. **Identity.** Did THIS object refuse us? Authoritative, and the only
           thing that can distinguish two same-kind callables.
        2. **Identity again, the other way.** Have we already called this object
           without a refusal? Then it is fine, whatever its kind. This is what
           keeps a healthy sibling alive when a same-kind neighbour refused.
        3. **Kind.** An object we have never called, of a kind that refused us
           earlier in this run, is the SAME rung rebuilt -- which is exactly
           what ``_build_table_judge_rungs`` does for every document in a
           batch. Calling it again re-pays the refusal we already know about.
        """
        if any(rung is refused for refused in self._table_rung_refused_callables):
            return True
        if any(rung is seen for seen in self._table_rung_seen_callables):
            return False
        kind = getattr(rung, "rung_kind", "") or ""
        return bool(kind) and kind in self._table_rung_refused_this_run

    def _live_table_judge_rungs(self, rungs) -> list:
        """The rungs still worth calling on THIS page.

        Cold review round 4, item 6. The per-run breaker used to live only in
        the reachability seam, which is a resume decision -- so within one run
        every remaining page still called the rung that had already refused
        page 1. Filtering here is what actually stops the amplification.
        """
        if not self._table_rung_refused_this_run and not self._table_rung_refused_callables:
            return list(rungs)
        return [rung for rung in rungs if not self._table_rung_callable_refused(rung)]

    def _record_table_rung_refusals(self, rungs_used, result) -> None:
        """Record refusals from ONE ladder run, by kind and by callable identity.

        ``run_table_ladder`` calls ``rungs_used`` in order and appends one
        result per call, so ``rung_results[i]`` came from ``rungs_used[i]``.
        That positional correspondence is the only reliable way back from a
        result to the object that produced it.
        """
        for index, rr in enumerate(getattr(result, "rung_results", []) or []):
            if index >= len(rungs_used):
                continue
            rung = rungs_used[index]
            if getattr(rr, "refusal", False):
                if not any(rung is refused for refused in self._table_rung_refused_callables):
                    self._table_rung_refused_callables.append(rung)
            elif not any(rung is seen for seen in self._table_rung_seen_callables):
                self._table_rung_seen_callables.append(rung)
        self._note_table_rung_refusals([result])

    def _refused_ladder_result(self, table_id: str):
        """The terminal for a table whose whole ladder is refused this run.

        Not the same thing as the empty-rung (strict_local / configured-off)
        terminal, which is settled by configuration and must not latch. This
        one IS transient: the synthesized results name the refused kinds so the
        page latches and a LATER run retries them.
        """
        from socr.judge.table_ladder import TableLadderOutcome, TableLadderResult
        from socr.judge.table_verdict import RungResult

        return TableLadderResult(
            table_id=table_id,
            outcome=TableLadderOutcome.UNVERIFIED,
            rung_results=[
                RungResult(
                    rung=kind,
                    ok=False,
                    error="rung refused earlier in this run; not called again",
                    unavailable=True,
                    refusal=True,
                )
                for kind in sorted(self._table_rung_refused_this_run)
            ],
        )

    def _note_table_rung_refusals(self, table_results) -> None:
        """Trip the per-run breaker for any rung that refused us on a real call.

        Cold review round 3, new finding 2. A quota or credential refusal is
        not a per-table fact: the next table in this run gets the same answer.
        Without this, a batch pre-gate can admit every latched document and
        every table in every one of them pays the same refused call -- the exact
        cost amplification the latch was added to prevent.
        """
        for result in table_results or []:
            for rr in result.rung_results:
                if getattr(rr, "refusal", False):
                    kind = rung_kind(rr.rung)
                    if kind not in self._table_rung_refused_this_run:
                        logger.info(
                            "table judge rung %r refused (%s); not retried again this run",
                            kind,
                            rr.error,
                        )
                    self._table_rung_refused_this_run.add(kind)

    # ------------------------------------------------------------------
    # GH-359 ruling 5: mechanical binding evidence at the gate.
    #
    # ``tables/binding.py bind()`` is a pure, local, cloud-free geometric
    # check that catches the GH-273 shape judges miss: a value multiset
    # that is completely correct but bound to the wrong row/column.
    #
    # CONTRADICTION-ONLY. ``fully_checked``, ``no_known_contradiction``,
    # ``structural_agreement``, ``native_unbound`` and ``model_unbound``
    # are NOT gates (GH-330: the binder fully-checks 0/15 real pages;
    # wiring any of those as SUCCESS re-blocks the wave). Only
    # ``contradicted_cells`` / ``row_label_contradictions`` fire. Absence
    # of coverage stays NEUTRAL.
    #
    # On fire: withhold acceptance (TABLE_UNVERIFIED). Never REJECTED on
    # mechanical evidence alone -- the native text layer can be the
    # culprit (GH-334). A judge REJECTED is left untouched. The check is
    # not a fake judge rung and does not inject findings into the prompt
    # (ruling 4: crop + markdown, nothing else).
    # ------------------------------------------------------------------

    def _binding_contradiction_for_witness(self, state: DocumentState, page_num: int, witness):
        """Run the mechanical binding check for one LOCATED witness.

        Returns the ``BindingResult`` when a GH-273-class contradiction was
        found, else None (no box, no native words, an unparseable
        candidate, or nothing checkable disagreed). Never raises: a
        page-open/geometry failure is logged and treated as an absence of
        evidence.
        """
        if witness.box is None:
            return None

        from socr.core.pdf import open_pdf
        from socr.tables.binding import bind

        try:
            with open_pdf(state.handle.path) as doc:
                words = doc[page_num - 1].get_text("words")
        except Exception as exc:
            logger.warning(
                "mechanical binding check: could not read native words on p%d (%s: %s)",
                page_num,
                type(exc).__name__,
                exc,
            )
            return None
        if not words:
            return None

        try:
            binding_result = bind(words, witness.markdown, region=witness.box.bbox)
        except Exception as exc:
            logger.warning(
                "mechanical binding check errored on p%d table %s (%s: %s); ignored",
                page_num,
                witness.table_id,
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            return None

        # GH-359 ruling 5: fully_checked is not a gate. Only a genuine
        # cell/label contradiction withholds acceptance.
        if binding_result.contradicted_cells or binding_result.row_label_contradictions:
            return binding_result
        return None

    def _run_table_judge_gate(
        self,
        state: DocumentState,
        page_num: int,
        ps: PageState,
        bo: PageOutput,
        rungs: list,
    ) -> None:
        """Judge every table this page emits against its own page crop.

        Helper, not the content choke by itself. Content terminals are
        produced by the ``if self.config.table_judge_ladder`` call in
        ``_phase_agentic``. Assemble then backfills any emitted table that
        still has no ladder event as TABLE_UNVERIFIED (completeness). A
        helper-unit test going green while that ``if`` is commented out is
        not a content gate; a missing assemble backfill lets an unjudged
        table ship SUCCESS.

        Never raises and never silently passes: a table with no image
        (MISSING, corroboration-contradicted AMBIGUOUS, or a failed page
        render), or any infra error while preparing witnesses or running
        the ladder, demotes that table to UNVERIFIED rather than being
        skipped. Count-mismatch AMBIGUOUS is judged against a full-page
        crop (GH-373) rather than abstained. Sets
        ``ps.table_ladder_disposition`` -- NEVER ``bo.audit_passed`` (the
        winner-selection trap) and never mutates ``bo.failure_mode`` in
        place (the #252 round-1 rule: a finalized copy is demoted later, at
        the manifest guard, so the shipped attempt itself is never touched
        here).

        GH-359 ruling 5: a LOCATED witness also gets the mechanical
        binding-shift detector. A genuine contradiction CAPS the table at
        UNVERIFIED -- a later judge PASS can no longer resolve it to
        ACCEPTED. REJECTED is left untouched. Mechanical evidence is not a
        fake judge rung and is never injected into the prompt (ruling 4).
        Empty rungs plus a contradiction is UNVERIFIED, not REJECTED
        (GH-334: the native layer can be the culprit).

        GH-367: the cap may be lifted only by ``tables.adjudication``
        disproving EACH contradiction (encoding-garbage native token, or
        an independent cell-raster transcription that matches markdown
        and not native). An ordinary PASS never lifts it. A failed or
        partial adjudication leaves the clamp in place.
        """
        if bo.engine == "chart_asset" or not bo.text:
            return

        from socr.core.audit_log import AuditEvent
        from socr.judge.table_ladder import (
            TableLadderOutcome,
            TableLadderResult,
            reduce_page_ladder,
            run_table_ladder,
        )
        from socr.judge.table_prompt import table_judge_prompt_scope
        from socr.judge.table_verdict import (
            TABLE_BINDING_ADJUDICATED_KIND,
            TABLE_LADDER_ACCEPTED_KIND,
            TABLE_LADDER_REJECTED_KIND,
            TABLE_LADDER_UNVERIFIED_KIND,
            RungResult,
            is_availability_exception,
        )
        from socr.tables.binding import BindingResult
        from socr.tables.witness import WitnessScope, prepare_table_witnesses

        # table_id -> BindingResult iff bind() found a GH-273-class contradiction.
        forced_by_binding: dict = {}
        markdown_by_table: dict[str, str] = {}
        scope_by_table: dict[str, str] = {}
        # table_id -> the callables that ACTUALLY ran for it, in call order.
        # The audit trail's executing identity is resolved from these, never
        # from a result's position in the configured ladder (round 5).
        executed_rungs_by_table: dict[str, list] = {}
        self._table_judge_gate_rungs = list(rungs)

        try:
            with prepare_table_witnesses(state.handle.path, page_num, bo.text) as witnesses:
                if not witnesses:
                    # No table blocks in bo.text. Assemble completeness
                    # backfills if the shipped text later contains tables.
                    return
                table_results: list[TableLadderResult] = []
                for witness in witnesses:
                    scope_by_table[witness.table_id] = witness.scope.value
                    if witness.crop_path is None:
                        # MISSING / corroboration-contradicted AMBIGUOUS /
                        # page-render failed: not S1-shaped -- nobody could
                        # look at this table at all. Count-mismatch
                        # AMBIGUOUS with a page crop falls through and is
                        # judged (GH-373).
                        table_results.append(
                            TableLadderResult(
                                table_id=witness.table_id,
                                outcome=TableLadderOutcome.UNVERIFIED,
                            )
                        )
                        continue

                    binding = self._binding_contradiction_for_witness(state, page_num, witness)
                    if isinstance(binding, BindingResult) and (
                        binding.contradicted_cells or binding.row_label_contradictions
                    ):
                        forced_by_binding[witness.table_id] = binding
                        markdown_by_table[witness.table_id] = witness.markdown

                    if not rungs:
                        # No CLI available (strict_local, or flag-on with
                        # injected empty rungs): fail open to UNVERIFIED.
                        # A mechanical contradiction does not upgrade this
                        # to REJECTED (GH-359 ruling 5).
                        table_results.append(
                            TableLadderResult(
                                table_id=witness.table_id,
                                outcome=TableLadderOutcome.UNVERIFIED,
                            )
                        )
                        continue

                    # Cold review round 4, item 6: re-asked per table, because a
                    # refusal on the PREVIOUS table must spare this one.
                    live_rungs = self._live_table_judge_rungs(rungs)
                    if not live_rungs:
                        # Every configured rung refused us earlier in this run.
                        # Unlike the empty-rung case above this is transient, so
                        # the terminal names the refused kinds and latches.
                        table_results.append(self._refused_ladder_result(witness.table_id))
                        continue
                    try:
                        prompt_scope = "page" if witness.scope is WitnessScope.PAGE else "located"
                        with table_judge_prompt_scope(prompt_scope):
                            ladder_result = run_table_ladder(
                                live_rungs, witness.crop_path, witness.markdown, witness.table_id
                            )
                        executed_rungs_by_table[witness.table_id] = list(live_rungs)
                        self._record_table_rung_refusals(live_rungs, ladder_result)
                        table_results.append(ladder_result)
                    except Exception as exc:
                        logger.warning(
                            "table judge ladder errored on p%d table %s (%s: %s); UNVERIFIED",
                            page_num,
                            witness.table_id,
                            type(exc).__name__,
                            exc,
                            exc_info=True,
                        )
                        table_results.append(
                            TableLadderResult(
                                table_id=witness.table_id,
                                outcome=TableLadderOutcome.UNVERIFIED,
                                rung_results=[
                                    RungResult(
                                        rung="unknown",
                                        ok=False,
                                        error=f"{type(exc).__name__}: {exc}",
                                        # Cold review round 2, finding 2: same
                                        # typed rule as the ladder's own guard.
                                        # A transport failure is an outage; a
                                        # defect in this machinery is not, and
                                        # must not be retried on every resume.
                                        unavailable=is_availability_exception(exc),
                                    )
                                ],
                            )
                        )
        except Exception as exc:
            # Witness preparation itself is documented never to raise (B0),
            # but a page whose crop/witness machinery fails entirely must
            # still demote -- belt-and-braces against the no-silent-loss
            # rule, not a case this repo's own contract expects to hit.
            logger.warning(
                "table witness preparation errored on p%d (%s: %s); page UNVERIFIED",
                page_num,
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            state.events.append(
                AuditEvent(
                    page_num=page_num,
                    kind=TABLE_LADDER_UNVERIFIED_KIND,
                    engine=bo.engine or "",
                    detail=f"table witness preparation failed: {type(exc).__name__}: {exc}",
                )
            )
            ps.table_ladder_disposition = FailureMode.TABLE_UNVERIFIED
            return

        # Cold review round 3, new finding 2: a rung that refused us on a real
        # call is done for this run, whatever this page's terminal turns out to
        # be. Recorded before the latch derivation so a refusal on an ACCEPTED
        # page still spares the rest of the run.
        self._note_table_rung_refusals(table_results)

        # P1 retry latch: derive the page latch from the terminal's cause
        # BEFORE any mechanical-binding ACCEPTED-to-UNVERIFIED clamp changes
        # the outcome. Set when at least one table's original ladder result is
        # UNVERIFIED and contains an unavailable rung result -- and record WHICH
        # rung kinds those were (cold review round 3, finding 1), so resume asks
        # about the rung that actually failed rather than about any rung.
        unavailable_kinds = {
            rung_kind(rr.rung)
            for result in table_results
            if result.outcome is TableLadderOutcome.UNVERIFIED
            for rr in result.rung_results
            if rr.unavailable
        }
        if unavailable_kinds:
            ps.table_judge_retry_pending = True
            ps.table_judge_retry_rungs = sorted(unavailable_kinds)

        # GH-359 ruling 5: a genuine mechanical contradiction withholds
        # acceptance. REJECTED is untouched (ceiling on accept, not a
        # floor on reject). Empty-rung UNVERIFIED is already UNVERIFIED.
        # GH-367: the cap lifts only when adjudication disproves EACH
        # contradiction. Run that only for tables the clamp would
        # actually withhold (ladder ACCEPTED). Partial/failed
        # adjudication leaves the clamp in place.
        lifted_ids: set[str] = set()
        accepted_ids = {
            result.table_id
            for result in table_results
            if result.outcome is TableLadderOutcome.ACCEPTED
        }
        for table_id, binding in forced_by_binding.items():
            if table_id not in accepted_ids:
                continue
            record = self._adjudicate_clamped_table(
                state,
                page_num,
                table_id,
                markdown_by_table.get(table_id, ""),
                binding,
                ps,
            )
            ps.binding_adjudication[table_id] = record.to_dict()
            state.events.append(
                AuditEvent(
                    page_num=page_num,
                    kind=TABLE_BINDING_ADJUDICATED_KIND,
                    engine=bo.engine or "",
                    detail=(
                        f"table {table_id} binding adjudication {record.status}: "
                        f"{sum(1 for o in record.items if o.disproof)}/"
                        f"{len(record.items)} contradictions disproved"
                    ),
                    data={"table_id": table_id, **record.to_dict()},
                )
            )
            if record.status == "lifted":
                lifted_ids.add(table_id)

        clamped_results: list[TableLadderResult] = []
        for result in table_results:
            if (
                forced_by_binding.get(result.table_id) is not None
                and result.outcome is TableLadderOutcome.ACCEPTED
                and result.table_id not in lifted_ids
            ):
                result = replace(result, outcome=TableLadderOutcome.UNVERIFIED)
            clamped_results.append(result)
        table_results = clamped_results

        for result in table_results:
            unwitnessed = False
            if result.outcome is TableLadderOutcome.ACCEPTED:
                kind = TABLE_LADDER_ACCEPTED_KIND
                detail = f"table {result.table_id} accepted by the judge ladder"
            elif result.outcome is TableLadderOutcome.REJECTED:
                kind = TABLE_LADDER_REJECTED_KIND
                detail = f"table {result.table_id} rejected by the judge ladder (content problem, not retryable)"
            elif forced_by_binding.get(result.table_id) is not None:
                kind = TABLE_LADDER_UNVERIFIED_KIND
                held = (ps.binding_adjudication.get(result.table_id) or {}).get("status") == "held"
                if held:
                    detail = (
                        f"table {result.table_id} unverified: mechanical binding check found a "
                        "contradiction that adjudication did not disprove "
                        "(acceptance withheld; retryable on resume)"
                    )
                else:
                    detail = (
                        f"table {result.table_id} unverified: mechanical binding check found a "
                        "contradiction (acceptance withheld; retryable on resume)"
                    )
            elif (
                not result.rung_results
                and scope_by_table.get(result.table_id, "none") == WitnessScope.NONE.value
            ):
                # GH-560: no rung ever ran AND no table witness was ever
                # located, so there was nothing to send. Calling that "infra
                # problem, retryable on resume" promises a retry that cannot
                # happen: the P1 latch correctly does not fire for a no-witness
                # terminal (there is no unavailable rung to wait for), so the
                # document is skipped on the next run and the label is a lie.
                #
                # The empty rung trail is NOT sufficient on its own. A witness
                # that was located and then found no reachable rung has an empty
                # trail too, and that one genuinely IS retryable -- the rung can
                # come back. #560 named the empty trail as the pin; the witness
                # is what actually separates the two.
                #
                # Latch semantics are deliberately unchanged; only the wording
                # is, because the wording is what was wrong.
                kind = TABLE_LADDER_UNVERIFIED_KIND
                unwitnessed = True
                detail = (
                    f"table {result.table_id} not judged: no table witness could be "
                    "prepared, so no rung ran (not retryable -- a re-run reaches the "
                    "same empty witness)"
                )
            else:
                kind = TABLE_LADDER_UNVERIFIED_KIND
                detail = f"table {result.table_id} unverified by the judge ladder (infra problem, retryable on resume)"
            # GH-353 review fix (post-A3 "agy" amendment): ``RungResult.rung``
            # names the judge model FAMILY ("gemini"), not the literal binary
            # that ran it (``agy``, per config) -- and rung 1's model is
            # exactly as config-dependent as rung 2's binary. Record the
            # executing identity so a sidecar reader never has to guess what
            # actually produced a verdict. Deliberately minimal: no latencies,
            # no verdict duplication -- ``detail`` above already says what
            # happened; this only says who executed it.
            #
            # Cold review round 5: derived from IDENTITY, never from the
            # position of a result in the configured ladder. That mapping was
            # true only while every run called rungs 1 and 2 in order; once the
            # refusal breaker could hand the ladder a filtered sublist, a lone
            # surviving rung 2 was recorded as having been executed by rung 1's
            # model. Synthesized refused results have no executor at all and
            # are sorted by kind, so their indices were never ladder positions
            # either. False provenance in a citation corpus's audit trail is
            # exactly the failure this trail exists to prevent.
            rung_trail = [
                {
                    "rung": rr.rung,
                    "ok": rr.ok,
                    "executing": self._executing_identity(
                        rr, executed_rungs_by_table.get(result.table_id) or [], idx
                    ),
                }
                for idx, rr in enumerate(result.rung_results)
            ]
            state.events.append(
                AuditEvent(
                    page_num=page_num,
                    kind=kind,
                    engine=bo.engine or "",
                    detail=detail,
                    data={
                        "table_id": result.table_id,
                        "rung_trail": rung_trail,
                        "witness_scope": scope_by_table.get(result.table_id, "none"),
                        # GH-560: stated, not left to be inferred from an empty
                        # rung_trail by every consumer separately.
                        **({"retryable": False} if unwitnessed else {}),
                    },
                )
            )

        page_result = reduce_page_ladder(table_results)
        if page_result.outcome is TableLadderOutcome.REJECTED:
            ps.table_ladder_disposition = FailureMode.TABLE_REJECTED
        elif page_result.outcome is TableLadderOutcome.UNVERIFIED:
            ps.table_ladder_disposition = FailureMode.TABLE_UNVERIFIED

    def _adjudicate_clamped_table(
        self,
        state: DocumentState,
        page_num: int,
        table_id: str,
        markdown: str,
        binding,
        ps: PageState,
    ):
        """GH-367: try to disprove each bind() contradiction on one table.

        Patch seam for tests: ``_transcribe_cell_token``. Encoding-garbage
        items never call it. A missing transcriber is not a disproof.
        """
        from socr.tables.adjudication import adjudicate, items_from_binding

        items = items_from_binding(binding)
        prior = (ps.binding_adjudication or {}).get(table_id)

        def _transcribe(bbox: tuple[float, float, float, float]) -> str | None:
            crop = self._render_adjudication_crop(state.handle.path, page_num, bbox)
            if crop is None:
                return None
            try:
                return self._transcribe_cell_token(crop)
            finally:
                crop.unlink(missing_ok=True)

        return adjudicate(items, markdown=markdown, prior=prior, transcribe=_transcribe)

    def _transcribe_cell_token(self, crop_path: Path) -> str | None:
        """Constrained transcriber seam. Returns a token or None. Never raises.

        ``strict_local`` skips the cloud POST; encoding-garbage disproof
        still runs. Tests patch this method rather than httpx.
        """
        if self.config.strict_local:
            return None
        from socr.judge.cell_transcribe import transcribe_cell

        try:
            return transcribe_cell(
                crop_path,
                model=self.config.table_judge_rung1_model,
                host=self.config.table_judge_rung1_host,
                timeout=self.config.table_judge_timeout_sec,
            )
        except Exception as exc:
            logger.warning(
                "cell transcribe failed (%s: %s); not a disproof",
                type(exc).__name__,
                exc,
            )
            return None

    def _render_adjudication_crop(
        self,
        pdf_path: Path,
        page_num: int,
        bbox: tuple[float, float, float, float],
    ) -> Path | None:
        """Render *bbox* to a temp PNG using the same padding/DPI as table witnesses."""
        import os
        import tempfile

        import fitz
        from PIL import Image

        from socr.core.born_digital import upright_rotation_for
        from socr.core.pdf import open_pdf
        from socr.tables.extract import CROP_PADDING_PT, DEFAULT_CROP_DPI

        try:
            doc = open_pdf(pdf_path)
        except Exception as exc:
            logger.warning("adjudication crop: cannot open %s (%s)", pdf_path, exc)
            return None
        try:
            page = doc[page_num - 1]
            page_rect = page.rect
            x0, y0, x1, y1 = bbox
            clip = fitz.Rect(
                max(page_rect.x0, x0 - CROP_PADDING_PT),
                max(page_rect.y0, y0 - CROP_PADDING_PT),
                min(page_rect.x1, x1 + CROP_PADDING_PT),
                min(page_rect.y1, y1 + CROP_PADDING_PT),
            )
            if clip.is_empty or clip.is_infinite:
                return None
            rotation = upright_rotation_for(page, clip=clip)
            mat = fitz.Matrix(DEFAULT_CROP_DPI / 72, DEFAULT_CROP_DPI / 72)
            if rotation != 0:
                mat.prerotate(rotation)
            pix = page.get_pixmap(matrix=mat, clip=clip)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        except Exception as exc:
            logger.warning(
                "adjudication crop failed p%d (%s: %s)",
                page_num,
                type(exc).__name__,
                exc,
            )
            return None
        finally:
            doc.close()

        fd, name = tempfile.mkstemp(prefix="socr_adjudicate_", suffix=".png")
        path = Path(name)
        try:
            os.close(fd)
            img.save(path)
        except Exception as exc:
            logger.warning("adjudication crop save failed p%d (%s)", page_num, exc)
            path.unlink(missing_ok=True)
            return None
        return path

    def _apply_binding_adjudication_meta(
        self, state: DocumentState, page_num: int, meta: dict
    ) -> None:
        """Restore GH-367 lift records from a sidecar payload. Malformed → ignore."""
        raw = meta.get("binding_adjudication")
        if not isinstance(raw, dict):
            return
        cleaned: dict = {}
        for table_id, record in raw.items():
            if isinstance(table_id, str) and isinstance(record, dict):
                cleaned[table_id] = record
        if cleaned:
            state.pages[page_num].binding_adjudication = cleaned

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

        def __init__(self, inner, timeout_sec: float | None, owner=None) -> None:
            self._inner = inner
            self._timeout_sec = timeout_sec
            # Round 3, finding 3: the worker runs on its own thread, so the
            # pipeline's per-invocation event binding has to be carried across
            # explicitly. ``None`` keeps the wrapper usable standalone (tests
            # construct it directly); the pipeline always passes itself.
            self._owner = owner

        def assess(self, output, provider):
            import concurrent.futures

            from socr.pipeline.agentic import AcceptDecision

            if self._timeout_sec is None:
                return self._inner.assess(output, provider)

            owner = self._owner
            binding = owner._current_judge_binding() if owner is not None else None

            def _run():
                if owner is not None:
                    owner._bind_judge_events(binding)
                    try:
                        return self._inner.assess(output, provider)
                    finally:
                        owner._bind_judge_events(None)
                return self._inner.assess(output, provider)

            if owner is not None:
                owner._enter_judge_call()
            ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = ex.submit(_run)
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
            finally:
                # Decremented in the CALLER exactly once, including on timeout:
                # from here on nothing is waiting on that worker, so anything it
                # still emits is late by definition.
                if owner is not None:
                    owner._leave_judge_call()

    # ------------------------------------------------------------------
    # Agentic: cost-aware per-page routing
    # ------------------------------------------------------------------

    def _phase_agentic(self, state: DocumentState, output_dir: Path) -> None:
        """PP-2 fused page-major loop: one pass over ALL pages (native + OCR).

        Born-digital prose takes free native text; every OCR page is routed
        through the cost ladder, except an opt-in corrupt-equation page: that page
        takes the crop-backed ``native+math`` region lane before whole-page routing.
        After each page is final, a PROVISIONAL fragment + sidecar
        (``terminal=False``) is flushed for crash recovery.
        The authoritative fragment bytes come from ``_rewrite_all_fragments``
        at the end of ``_phase_assemble`` (fork A / assemble-authoritative).

        Per-page lifecycle:
          0. Eligible corrupt-equation pages run the region-only crop guardrail;
             all other non-native pages enter ``route_page``.
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

        Optional dual-pass table rereads run inside this fused loop, so each table
        is handled before the page is flushed and assembled. The crop reread runs
        AFTER ``route_page``'s verdict, so its patched text is a new candidate and
        goes back through the same judge (``_rejudge_crop_patched_page``) before it
        can ship.

        **Post-verdict content contract** (P3, scoped by the round-2 ruling).
        "Judged bytes are shipped bytes" means no post-verdict step may ADD or
        ALTER content. There are exactly three enumerated exceptions, and any new
        one needs the same justification:

          * ``_sanitize_agentic_page_image_refs`` — SUBTRACTIVE. Removes VLM
            sentinel placeholders and fabricated image refs. An image ref is a
            pure pointer; a pointer to something that does not exist carries
            nothing, so removal cannot lose content.
          * ``_guard_agentic_page_table_repetition`` — SUBTRACTIVE. Truncates a
            runaway row repetition; every dropped line is byte-identical to one
            that is kept.
          * ``_attach_equation_latex_sidecars`` — ADDITIVE, and therefore GUARDED
            by REUSING the equation lane's own guards rather than copies:
            ``contract_delimiter_violation`` at the ``process_equation_region``
            choke point refuses a page-assembly delimiter (and a fence that would
            break out of the block), and ``_guard_equation_sidecar_block`` runs
            ``region_presence_verdict`` so model LaTeX carrying a number absent
            from the page's own source is refused.

        "Adds content" is about CONTENT tokens. A subtractive helper that leaves
        socr's own marker where it removed something (``[socr: …]``,
        ``[page N failed: …]``, recognised by ``SOCR_MARKER_RE``) has added a
        receipt, not content. The two subtractive helpers are pinned by
        invariant tests (``tests/test_p35_cold_review_round2.py``,
        ``tests/test_p35_cold_review_round3.py``): outside those marker spans,
        every token in the output exists in the input, never the reverse.

        ``_classify`` remains doc-wide (``_phase_analyze``); the fused loop
        handles only the post-classification per-page lifecycle (fork C2).
        """
        from socr.core.providers import provider_ladder
        from socr.pipeline.agentic import DEFAULT_PROVIDER_TIMEOUTS

        if not self.config.quiet:
            console.print("\n[cyan]Agentic routing[/cyan] (cost-ordered, judge-gated)")

        # -- Doc-scoped native-fallback list (needed by run_provider stub) --------
        # The corrupt-math guardrail is a separate region lane, not a whole-page
        # OCR rung. Remove its pages before provider/resume setup so an empty
        # provider ladder cannot suppress a crop recovery that does not use it.
        math_recovery_pages = {
            page_num
            for page_num, ps in sorted(state.pages.items())
            if self._is_corrupt_math_recovery_page(page_num, ps)
        }
        # P4-R: the equation-region lane is likewise a region lane, not a
        # whole-page OCR rung. Built here, beside the corrupt-math set and
        # BEFORE provider setup, so the empty-ladder branch below cannot stamp
        # one of its pages WARNING/MODEL_UNAVAILABLE -- with no provider the
        # page must ship its native prose exactly as it does with the lane off.
        equation_lane_pages = {
            page_num
            for page_num, ps in sorted(state.pages.items())
            if page_num not in math_recovery_pages
            and self._is_equation_region_lane_page(page_num, ps)
        }
        ocr_pages: list[int] = []
        for page_num, ps in sorted(state.pages.items()):
            if (
                page_num not in math_recovery_pages
                and page_num not in equation_lane_pages
                and not self._is_agentic_trusted_native(page_num, ps)
            ):
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
            try:
                if not self._is_chart_asset_page(pn, ps, state.handle.path):
                    continue
            except Exception as exc:
                logger.warning(
                    "chart eligibility detection failed for p%d: %s: %s",
                    pn,
                    type(exc).__name__,
                    exc,
                    exc_info=True,
                )
                from socr.core.audit_log import AuditEvent

                state.events.append(
                    AuditEvent(
                        page_num=pn,
                        kind="chart_asset_detection_failed",
                        engine="chart_asset",
                        detail="chart eligibility detection failed; page proceeds through non-chart route",
                        data={
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        },
                    )
                )
                # GH-318: the audit event alone is not enough. #297 made this
                # route fail SOFT and durable, but page status, document status,
                # metadata and the CLI all still reported a clean SUCCESS, so the
                # skip was invisible unless someone opened audit_log.json — the
                # #252 / #211 class. Flag the page so the document buckets in
                # _phase_assemble can demote the run to AUDIT_FAILED
                # ("completed with warnings, output written") without discarding
                # the text or forcing the chart lane.
                ps.chart_asset_detection_failed = True
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
        # P4-R precedence, stated once and enforced here rather than only in the
        # loop's elif order: corrupt math > chart asset > equation region >
        # generic no-provider > plain native > whole-page route_page. A chart
        # page keeps the chart lane unchanged; its equation regions are simply
        # not read (deliberate, and recorded in the P4-R decision log).
        equation_lane_pages -= chart_winner_pages
        equation_lane_pages -= resumed_pages
        # Pages the lane actually ran on this pass. The legacy GH-36a/36b in-loop
        # block is skipped for exactly these, so `--detect-equations
        # --recover-clean-equations` plus the lane cannot detect, crop or attach
        # the same region twice.
        equation_lane_handled: set[int] = set()

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

        no_ocr_provider_pages: set[int] = set()
        if not ladder and ocr_pages:
            logger.warning("agentic: no OCR providers available; OCR pages left unprocessed")
            if not self.config.quiet:
                console.print("  [red]No OCR providers available[/red]")
            no_ocr_provider_pages = set(ocr_pages)
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
                ps.best_output = fallback

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
        judge = self._TimeoutJudge(_inner_judge, _judge_timeout, owner=self)

        if not self.config.quiet:
            ladder_str = " -> ".join(f"{p.engine.value}(${p.cost_per_page_usd:g})" for p in ladder)
            console.print(f"  ladder: {ladder_str}")

        def run_provider(profile: ProviderProfile, page_num: int) -> PageOutput:
            outs = self._run_engine_on_pages(
                state,
                [page_num],
                native_fallback_pages,
                profile.engine,
                "agentic",
                profile=profile,
            )
            return outs[0]

        # -- Lazy doc-scoped table extractor (P5) -------------------------------
        # Crop rereads are an escalation tool, so resolving a model or probing a
        # reader at document setup would make an otherwise clean document pay for
        # unused work.  The initialized latch also memoizes fail-open outcomes:
        # a missing model or construction error must not be retried on every page.
        _table_extractor = None
        _table_extractor_initialized = False
        _dual_pass_tables_enabled = bool(self.config.dual_pass_tables)

        def _get_table_extractor():
            nonlocal _table_extractor, _table_extractor_initialized
            if _table_extractor_initialized:
                return _table_extractor
            _table_extractor_initialized = True
            if not _dual_pass_tables_enabled:
                return None
            try:
                from socr.tables.extract import TableCropExtractor, make_table_reader

                _table_model = self._resolve_crop_vlm_model()
                if not _table_model:
                    return None
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
                    "agentic: table extractor unavailable (%s); skipping signal rereads", exc
                )
            return _table_extractor

        # -- Doc-scoped table judge ladder rungs (GH-353 TICKET-B1) -------------
        # Constructed once per document, injected into every page's gate call.
        # [] when the flag is off or strict_local forbids the (cloud-only)
        # rungs; the gate reads an empty list as "fail open to UNVERIFIED".
        _table_judge_rungs = self._build_table_judge_rungs()

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
            _route_table_signal = False
            _winning_profile: ProviderProfile | None = ladder[0] if ladder else None

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

            # GH-271: region-only corrupt-equation lane. It owns the page before
            # whole-page OCR and keeps surrounding native prose untouched.
            if page_num in math_recovery_pages:
                self._agentic_math_recovery_page(
                    state, page_num, ps, output_dir, chart_winner_pages, _chart_figures_dir
                )
            # PP-7: Chart-asset lane — intercept before the native-bypass branch.
            # A born-digital native page that carries vector chart marks (or an
            # embedded raster image) routes here instead of shipping as raw word-
            # salad prose.  B1 representation: native prose retained + chart PNG
            # ref embedded + explicit audit flag.  Force PNG even when --save-
            # figures is off (chart PNGs are mandatory preservation artifacts).
            elif page_num in chart_winner_pages:
                self._agentic_chart_asset_page(state, page_num, ps, _chart_figures_dir)
            # P4-R: region-scoped equation lane. It sits AFTER the corrupt-math
            # and chart lanes and BEFORE plain native, so a page those two own
            # keeps its existing route. The lane's floor is the plain-native
            # output it builds first, which is why it is safe here: with no
            # provider, no region, or a refused reading it ships exactly what
            # the `elif is_native` branch below would have shipped.
            elif page_num in equation_lane_pages:
                self._agentic_equation_region_page(state, page_num, ps, output_dir, available)
                equation_lane_handled.add(page_num)
            elif page_num in no_ocr_provider_pages:
                self._agentic_no_provider_page(page_num, ps)
            elif is_native:
                self._agentic_native_page(state, page_num, ps)
            else:
                # Route OCR page through the cost ladder.
                remaining = None
                if self.config.cost_budget > 0:
                    total_cost = state.total_cost
                    # An unmetered earlier call makes remaining paid budget
                    # unknowable. Fail closed by admitting only free rungs; an
                    # unknown subtotal must never be treated as zero spend.
                    remaining = (
                        0.0
                        if total_cost is None
                        else max(self.config.cost_budget - total_cost, 0.0)
                    )
                decision = route_page(
                    page_num,
                    ladder,
                    run_provider,
                    judge,
                    remaining_budget=remaining,
                    provider_timeout=provider_timeout,
                )
                _route_table_signal = self._route_page_table_escalation_signal(decision, ladder)
                # The profile whose reading won. Needed to re-judge a candidate
                # produced after routing (crop reread) through the same judge
                # with the same provider context.
                for _att in decision.attempts:
                    if _att.output is decision.final_output:
                        _winning_profile = profile_by_id(_att.provider_id) or _winning_profile
                        break

                for att in decision.attempts:
                    att.output.cost_usd = att.cost_usd
                    att.output.audit_passed = att.accepted
                    att.output.provider_id = att.provider_id  # B3: agentic provenance
                    # GH-370: ``att.model``/``att.backend`` carry the rung's
                    # REGISTRY identity. The manifest's provenance fields must
                    # carry what executed -- for a vLLM run those differ, and
                    # recording the registry label named a backend that was not
                    # installed on the host.
                    att_profile = profile_by_id(att.provider_id)
                    if att_profile is not None:
                        att.output.provider_backend, att.output.provider_model = (
                            resolved_provenance(att_profile, self.config)
                        )
                    else:
                        att.output.provider_model = att.model
                        att.output.provider_backend = att.backend
                    att.output.skip_reason = (
                        att.reason if not att.accepted and not att.output.text else ""
                    )  # B3
                    # GH-169: keep the judge's verdict for EVERY attempt, not
                    # only the ones whose output was empty. A provider whose
                    # reading the judge refused journaled reason "none", so the
                    # one question the manifest exists to answer -- why did the
                    # ladder escalate past this rung -- had no answer.
                    att.output.judge_reason = att.reason or ""

                    ps.attempts.append(att.output)
                ps.best_output = decision.final_output

                # GH-90: scanned-table source-evidence fail-closed floor.
                _source_ev_rejected = any(
                    "source_evidence_table" in (att.reason or "") for att in decision.attempts
                )
                if _source_ev_rejected and not ps.is_born_digital:
                    self._apply_scanned_table_floor(
                        ps, state.handle.path, page_num, _chart_figures_dir
                    )

                # #263: rotated-shredded floor PNG. The page's native layer is
                # confetti and the OCR ladder accepted nothing, so the marker
                # that ships in its place carries the page image -- otherwise a
                # human has no way to read a caption socr just refused. Render
                # only when nothing was accepted: an accepted model output wins
                # in _winning_page_output and the floor never applies.
                if getattr(ps, "native_rotated_text_shredded", False) and not decision.accepted:
                    if _chart_figures_dir is not None:
                        ps.rotated_shred_png_ref = self._render_d3_floor_png(
                            state.handle.path,
                            page_num,
                            _chart_figures_dir,
                            stem="shredded_rotated_page",
                            label="Shredded rotated page",
                        )

                # Provenance guard: when the judge rejected ALL ladder rungs for a
                # born-digital table page or a page where table structure failed,
                # mark the page so _assemble_result treats any native-text fallback as
                # audit-failed.
                if not decision.accepted and (
                    self._page_has_tables(page_num, ps)
                    or any(
                        getattr(e, "kind", None) == "table_structure_failed"
                        for e in state.events
                        if getattr(e, "page_num", None) == page_num
                    )
                ):
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

                    # P2 / GH-317: structure-class fail-closed floor PNG. When
                    # no attempt authored a usable grid on a born-digital table
                    # page, render a full-page PNG so the fail-closed floor can
                    # reference the image instead of shipping the unverified native grid.
                    from socr.core.manifest import structure_class_floor_applies

                    if (
                        structure_class_floor_applies(ps)
                        and _chart_figures_dir is not None
                        and not getattr(ps, "d3_floor_png_ref", "")
                    ):
                        ps.d3_floor_png_ref = self._render_d3_floor_png(
                            state.handle.path,
                            page_num,
                            _chart_figures_dir,
                            stem="failed_table",
                            label="Failed table page",
                        )

                # Record cost so DocumentState.total_cost reflects spend, and
                # against the PAGE so it survives the sidecar and resume (round 5).
                state.record_engine_run(
                    EngineResult(
                        document_path=state.handle.path,
                        engine=decision.winning_engine,
                        status=DocumentStatus.SUCCESS
                        if decision.accepted
                        else DocumentStatus.AUDIT_FAILED,
                        cost=decision.total_cost_usd,
                        processing_time=0.0,
                    ),
                    page_nums=[page_num],
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
                if _had_timeout and not self._probe_backend_idle():
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

            # #123 TICKET-C2 scoring is NOT gated on the P5 signal: it must reach
            # every page it reached before this branch, because it is the only
            # surface `table_not_scorable` and `table_unexplained_lanes` ever get.
            # That set is "a page with tables" (the old TICKET-C2 arm) UNION "a
            # page the GH-96 lane would have scored itself" (the old escalation
            # arm, which never gated on has_tables) -- native-bypass and
            # chart-asset table pages included.
            _lane_live = _escalation_profile is not None and not _escalation_degraded
            _score_table_signal = False
            if bo.text and (
                self._page_has_tables(page_num, ps) or (_lane_live and bo.engine != "chart_asset")
            ):
                _score_table_signal = bool(self._surface_table_scoring(state, page_num, ps, bo))

            # P5: the crop reread is an escalation tool fired by a signal, never a
            # trunk pass over every accepted table page.  Route evidence (a native
            # verifier rejection that forced another rung, or an exhausted ladder)
            # joins the score HERE and only here: it is evidence that the page is
            # worth a bounded second look, but it is not evidence that a paid
            # provider re-read could be kept, which is what GH-96 below needs.
            _crop_changed_text = False
            if (
                _dual_pass_tables_enabled
                and (_score_table_signal or _route_table_signal)
                and not is_native
                and bo.text
                and bo.engine != "chart_asset"
                and self._page_has_tables(page_num, ps)
            ):
                # A failed/missing extractor is fail-open and memoized by the
                # document-scoped getter above.
                _table_extractor = _get_table_extractor()
                if _table_extractor is not None:
                    _accepted_text = bo.text
                    try:
                        from socr.core.pdf import open_pdf
                        from socr.tables import locate_tables

                        with open_pdf(state.handle.path) as _doc:
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
                            "agentic signal table re-read errored on p%d (%s); keeping text",
                            page_num,
                            exc,
                        )
                    bo = ps.best_output or bo
                    # P3 / ruling step 4: the reread's patched text is a NEW
                    # CANDIDATE, not shipped bytes. Run it back through the same
                    # judge before it can replace what the judge already accepted.
                    # Outside the try/except above on purpose: _reread_page_tables
                    # patches bo.text outside its own guard, so a raise there must
                    # still not leave unjudged bytes on the page.
                    bo = self._rejudge_crop_patched_page(
                        state, page_num, ps, bo, _accepted_text, judge, _winning_profile
                    )
                    _crop_changed_text = (bo.text or "") != _accepted_text

            # GH-96: escalate a table page whose output disagrees with its own
            # native text layer, keeping the candidate only if exactness measurably
            # improves. Driven by the SCORE, not by route evidence: an earlier rung
            # being rejected says nothing about whether `decide_escalation` could
            # keep a new candidate, and when the incumbent already scores 100% it
            # provably cannot. Passing the score prevents a second scoring pass and
            # its duplicate table_not_scorable / table_unexplained_lanes events --
            # except when the crop reread changed the shipped bytes, where the
            # stale score describes text that no longer ships and the page is
            # re-scored on what does.
            if _lane_live and bo.text and bo.engine != "chart_asset":
                _escalation_degraded, bo = self._escalate_table_page(
                    state,
                    page_num,
                    ps,
                    bo,
                    _escalation_profile,
                    run_provider,
                    state.handle.path,
                    needs_escalation=None if _crop_changed_text else _score_table_signal,
                )

            # GH-36a/36b: per-page equation detect + crop + optional LaTeX
            # sidecar.  Runs ONLY when the flags are on (default-off).  With
            # both flags OFF, the page body is unchanged — output is byte-
            # identical to a non-equation-aware run (AC: "flags OFF → unchanged").
            # P4-R: a page the equation lane already handled has been detected,
            # cropped, read and attached once. Running the legacy block over it
            # would detect and crop the same regions again and append a SECOND
            # sidecar at page end, so it is skipped here.
            if _detect_eq and bo.text and page_num not in equation_lane_handled:
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

            # GH-359 ruling 6: THIS `if` produces content terminals. Assemble
            # does not re-judge; it only backfills a missing terminal as
            # UNVERIFIED (completeness). Placed AFTER the repetition guard so
            # the judged table is the exact text that ships. Runs on native
            # AND OCR pages alike; the helper itself skips chart_asset pages
            # (assemble catches residual markdown tables those pages still emit).
            if self.config.table_judge_ladder:
                self._run_table_judge_gate(state, page_num, ps, bo, _table_judge_rungs)

            # GH-318: this page's chart-vs-table routing was never decided --
            # the eligibility detector raised and the page fell through to the
            # non-chart route. Demote the page's STATUS only; ``audit_passed``
            # is the winner-SELECTION flag and flipping it would discard this
            # page's content (manifest.py: the #252 round-1 defect), which is
            # exactly what the fail-soft route of #297 exists to avoid.
            #
            # Deliberate consequence: the resume ledger accepts terminal pages
            # only at SUCCESS, so a WARNING page is re-processed on every resume
            # rather than skipped. That is the correct trade here -- the routing
            # decision is unresolved, so re-deciding it next run is the point --
            # but it does mean a deterministically-failing detector re-runs this
            # page every time.
            if bo is not None and getattr(ps, "chart_asset_detection_failed", False):
                if bo.status == PageStatus.SUCCESS:
                    bo.status = PageStatus.WARNING

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

                from socr.core.manifest import (
                    _whole_doc_page_texts,
                    finalized_page_record,
                )

                # GH-550: select and finalise ONCE, then hand the same record to
                # both writers.
                #
                # GH-539 made the two agree by calling `finalized_page_record`
                # for the fragment while `_flush_page_sidecar` called it again.
                # Same function and same inputs, so they agreed -- but that is a
                # coincidence maintained by hand, and #549 spent three rounds on
                # exactly this class: same function, one argument apart; same
                # guard, different output. A second call is a second chance to
                # diverge.
                #
                # The `structure_class_floor_text` special case goes too: the
                # floor disposition already lives inside the record, so deriving
                # it separately was a third path to the same bytes.
                _record = finalized_page_record(state, page_num, _whole_doc_page_texts(state))

                # Strip any leading ## Page N marker so the provisional fragment
                # body matches what assemble would produce (modulo
                # post-strip/post-figure transforms, which run later).
                _raw_body = _record.output.text or ""
                _stripped = _raw_body.lstrip()
                _m = PAGE_MARKER_RE.match(_stripped)
                _body = _stripped[_m.end() :].lstrip("\n") if _m else _raw_body

                self._flush_page_fragment(state, page_num, _body, output_dir)
                self._flush_page_sidecar(
                    state, page_num, output_dir, terminal=False, record=_record
                )
            except Exception as exc:
                logger.debug(
                    "PP-2 provisional flush failed for p%d (%s); continuing",
                    page_num,
                    exc,
                )

        # -- Post-loop summary ---------------------------------------------------
        if not self.config.quiet:
            total_cost = state.total_cost
            if total_cost is None:
                console.print("  total cost: unknown (unmetered direct call)")
            else:
                console.print(f"  total cost: ${total_cost:.4f}")
            if halt_reason:
                console.print(f"  [red]Halted: {halt_reason}[/red]")

        # If we halted due to backend degradation, record it on the state so
        # _phase_assemble can propagate the reason into EngineResult.error.
        if halt_reason:
            state.pp2_halt_reason = halt_reason

    def _agentic_math_recovery_page(
        self,
        state: DocumentState,
        page_num: int,
        ps: PageState,
        output_dir: Path,
        chart_winner_pages: set[int],
        chart_figures_dir: Path | None,
    ) -> None:
        """Run the region-only corrupt-equation lane for one page.

        R4 (behaviour-identical extraction): verbatim the
        ``if page_num in math_recovery_pages:`` branch of ``_phase_agentic``'s
        page loop. GH-271: this lane owns the page before whole-page OCR and
        leaves the surrounding native prose untouched.

        When the page is ALSO a chart winner, the mandatory chart PNG is retained
        alongside the region hybrid and the arbitration is recorded — a page whose
        chart competed with an equation always leaves a trace naming which lane
        took it and why.
        """
        if not self.config.quiet:
            console.print(
                f"  p{page_num}: [cyan]corrupt-equation region lane "
                f"[{self.config.math_model}][/cyan]"
            )
        math_out = self._recover_corrupt_math_page(state, page_num, output_dir)
        if page_num in chart_winner_pages:
            from socr.core.audit_log import AuditEvent as _MathChartEvent

            chart_png_ref = ""
            chart_render_error = ""
            if chart_figures_dir is not None:
                try:
                    saved_png = self._render_chart_page_png(
                        state.handle.path,
                        page_num,
                        chart_figures_dir,
                    )
                    chart_png_ref = (
                        f"![Chart page {page_num}]({chart_figures_dir.name}/{Path(saved_png).name})"
                    )
                except (RuntimeError, OSError) as exc:
                    chart_render_error = str(exc)
            else:
                chart_render_error = "figures directory unavailable"

            if chart_png_ref:
                math_out.text = f"{math_out.text.rstrip()}\n\n{chart_png_ref}"
                math_out.audit_notes.append(
                    "chart asset retained alongside corrupt-equation recovery"
                )
            else:
                # GH-568: the FLAG too, not only the event. The chart-only path
                # sets it (PP-7-R1) so `_winning_page_output`, the sidecar and
                # every disposition trigger that ORs it stay honest; this path
                # appended the event alone, leaving the audit log saying the
                # render failed while PageState and the sidecar said it had
                # not. The math hybrid already ships WARNING, so this is not a
                # SUCCESS restamp today -- it is a metadata lie for the
                # consumers that read the flag, resume included.
                ps.chart_asset_render_failed = True
                state.events.append(
                    _MathChartEvent(
                        page_num=page_num,
                        kind="chart_asset_render_failed",
                        engine="native+math",
                        detail=chart_render_error,
                    )
                )

            state.events.append(
                _MathChartEvent(
                    page_num=page_num,
                    kind="chart_math_arbitration",
                    engine="native+math",
                    detail=(
                        "both chart marks and corrupt-equation signals fired; "
                        "the region hybrid retained the mandatory chart PNG"
                        if chart_png_ref
                        else "both chart marks and corrupt-equation signals fired; "
                        "chart rendering failed and the region hybrid stayed WARNING"
                    ),
                    data={
                        "winner": ("native+math+chart_asset" if chart_png_ref else "native+math"),
                        "chart_png_rendered": bool(chart_png_ref),
                        "chart_png_path": chart_png_ref,
                    },
                )
            )
            state.events.append(
                _MathChartEvent(
                    page_num=page_num,
                    kind="chart_asset_page",
                    engine="chart_asset",
                    detail=(
                        "visual chart semantics represented as image asset alongside "
                        "the corrupt-equation region hybrid; data values not transcribed"
                    ),
                    data={
                        "png_saved": bool(chart_png_ref),
                        "png_path": chart_png_ref,
                    },
                )
            )
            state.events.append(
                _MathChartEvent(
                    page_num=page_num,
                    kind=VISUAL_VALUES_NOT_TRANSCRIBED_KIND,
                    engine="chart_asset",
                    detail=(
                        "in-image text on this figure -- axis labels, legend, any embedded "
                        "table -- is preserved only in the page image; it is not in the "
                        "markdown and no model read it"
                        if chart_png_ref
                        else "in-image text on this figure is preserved NOWHERE: it is not "
                        "in the markdown, no model read it, and the page image failed to save"
                    ),
                    data={"png_saved": bool(chart_png_ref), "png_path": chart_png_ref},
                )
            )
        ps.attempts.append(math_out)
        ps.best_output = math_out
        ps.corrupt_math_hybrid = math_out

    def _agentic_chart_asset_page(
        self,
        state: DocumentState,
        page_num: int,
        ps: PageState,
        chart_figures_dir: Path | None,
    ) -> None:
        """Ship a chart page as native prose plus a mandatory chart PNG.

        R3 (behaviour-identical extraction): verbatim the
        ``elif page_num in chart_winner_pages:`` branch of ``_phase_agentic``'s
        page loop. PP-7 / GH-150 TICKET-B1 representation: native prose is
        retained, the chart PNG ref is embedded, and the audit flag is explicit.
        The PNG is forced even when ``--save-figures`` is off, because chart PNGs
        are mandatory preservation artifacts.

        A render failure is fail-closed, never silent: status drops to WARNING and
        ``ps.chart_asset_render_failed`` is set so ``_winning_page_output`` cannot
        re-stamp the page as a clean native SUCCESS at manifest-freeze time.

        ``chart_figures_dir`` is the loop's ``_chart_figures_dir``; ``None`` means
        the output contract was unavailable and is itself a render failure.
        """
        if not self.config.quiet:
            console.print(f"  p{page_num}: [cyan]chart-asset lane[/cyan]")
        from socr.core.audit_log import AuditEvent

        chart_png_ref = ""
        chart_render_failed = False
        if chart_figures_dir is not None:
            try:
                saved_png = self._render_chart_page_png(
                    state.handle.path, page_num, chart_figures_dir
                )
                # Relative image ref (from the doc's root markdown file).
                png_rel = Path(saved_png).name
                # figures/ subdir is conventional; use the relative path
                # within the figures dir so it renders in the markdown.
                try:
                    _figures_dir_name = chart_figures_dir.name
                    chart_png_ref = f"![Chart page {page_num}]({_figures_dir_name}/{png_rel})"
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
            _msg = f"PP-7 chart-lane: figures_dir unavailable for p{page_num}; PNG not saved"
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
        #
        # GH-369: this lane declares "data values not transcribed" and then
        # shipped the native layer whole, axis tick scales included -- a column
        # of bare numbers indistinguishable from real values, under a clean
        # SUCCESS. Fence those lines instead: they stay in the file verbatim
        # (nothing is dropped) but stop reading as body prose beside the image
        # they belong to. A page with no bare-numeric lines is unchanged.
        from socr.figures.extractor import fence_chart_axis_residue

        native_prose = fence_chart_axis_residue(ps.native_text or "")
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

        # #263 round 2: this page's native half is confetti, so
        # _winning_page_output will refuse ``chart_body`` and ship the
        # failure marker instead. The PNG is the half that IS good --
        # on a chart page the image is the content -- so hand the ref
        # to the floor rather than re-rendering the same page. The
        # chart lane's own behaviour is untouched for every page that
        # is not flagged.
        if getattr(ps, "native_rotated_text_shredded", False) and chart_png_ref:
            ps.rotated_shred_png_ref = chart_png_ref

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
                    "visual chart semantics represented as image asset; data values not transcribed"
                ),
                data={
                    "png_saved": not chart_render_failed,
                    "png_path": chart_png_ref,
                },
            )
        )

        # GH-519: and the debt as a kind of its own, so counting it never means
        # parsing the sentence above. Both events, not one replacing the other:
        # `chart_asset_page` says what the lane DID, this says what it did not.
        state.events.append(
            AuditEvent(
                page_num=page_num,
                kind=VISUAL_VALUES_NOT_TRANSCRIBED_KIND,
                engine="chart_asset",
                detail=(
                    "in-image text on this figure -- axis labels, legend, any embedded "
                    "table -- is preserved only in the page image; it is not in the "
                    "markdown and no model read it"
                    if not chart_render_failed
                    else "in-image text on this figure is preserved NOWHERE: it is not in "
                    "the markdown, no model read it, and the page image failed to save"
                ),
                data={"png_saved": not chart_render_failed, "png_path": chart_png_ref},
            )
        )

    def _agentic_no_provider_page(self, page_num: int, ps: PageState) -> None:
        """Stamp a page that had no OCR provider available.

        R2 (behaviour-identical extraction): verbatim the
        ``elif page_num in no_ocr_provider_pages:`` branch of ``_phase_agentic``'s
        page loop. Born-digital fallbacks were already materialized during
        provider setup; a scan has no native text, so the missing provider is
        made explicit here rather than left silent.

        Reads no doc-scoped loop state and does not touch ``state``.
        """
        # Born-digital fallbacks were materialized during provider setup.
        # A scan has no native text, so make the missing provider explicit.
        if ps.best_output is None:
            from socr.core.manifest import page_failed_marker

            unavailable = PageOutput(
                page_num=page_num,
                text=page_failed_marker(page_num),
                status=PageStatus.ERROR,
                engine="",
                failure_mode=FailureMode.MODEL_UNAVAILABLE,
                error="no OCR providers available",
                audit_passed=False,
            )
            ps.attempts.append(unavailable)
            ps.best_output = unavailable

    def _agentic_native_page(
        self,
        state: DocumentState,
        page_num: int,
        ps: PageState,
    ) -> None:
        """Finalize a trusted-native page inside the fused agentic loop.

        R1 (behaviour-identical extraction): this is verbatim the
        ``elif is_native:`` branch of ``_phase_agentic``'s page loop. It reads
        no doc-scoped loop state -- only ``self.config``, ``state``, ``page_num``
        and ``ps`` -- which is why it is the first slice of the decomposition
        (see ``docs/log/2026-08-23_orchestrator-seams.md`` section 5).

        Effects are mutations on objects passed in: appends the native
        ``PageOutput`` to ``ps.attempts``, sets ``ps.best_output``, and appends
        up to three audit events (#92 unmapped math glyphs, #136 encoding
        hygiene, #217 unrecovered symbol glyphs) to ``state.events``.
        """
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
            or getattr(ps, "native_table_emission_defect", "")
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

        # #217: the page's symbol font had no ToUnicode map and at least
        # one glyph it draws has no verified recovery, so those
        # characters are still the wrong ones. Unlike the hygiene class
        # above, this class CAN corrupt a digit or an operator -- an
        # unrecovered minus is a sign flip, a negative coefficient
        # shipping as a large positive one -- so it must be visible on
        # the page's own audit trail and not only in a log line.
        if getattr(ps, "has_unrecovered_symbol_glyphs", False):
            from socr.core.audit_log import AuditEvent

            state.events.append(
                AuditEvent(
                    page_num=page_num,
                    kind="native_unrecovered_symbol_glyphs",
                    engine="native",
                    detail=(
                        "born-digital native text shipped from a symbol font with no "
                        "ToUnicode map; some drawn glyphs have no verified recovery "
                        "and remain whatever the extractor produced (a minus can read "
                        "as a digit, flipping a coefficient's sign)"
                    ),
                    data={"class": "symbol_glyph"},
                )
            )

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

    # -- Judge-side audit events: per-invocation ownership --------------------
    #
    # Round 2, finding 3. The composed page judge closes over ``state.events``
    # through ``record_event``, so anything it emits lands on the document the
    # instant it is emitted -- including while judging a candidate that is about
    # to be REFUSED and never ship. ``native_table_verifier_hard_fail`` is in
    # ``TABLE_DISTRUST_KINDS``, so one refused crop candidate was enough to mark
    # the shipped bytes untrusted in ``tables_trust.json``.
    #
    # Round 3, finding 3. A temporarily swapped instance-global list does not
    # close that: ``_TimeoutJudge`` returns a rejection while its worker keeps
    # running, so a LATE event arrived after the swap was restored and reached
    # the document anyway -- or, worse, landed in the scratch list of a LATER
    # re-judge. Ownership therefore has to be per invocation, not per instance:
    #
    #   * every ``_judge_events_to`` block mints a token and binds (token, sink)
    #     to the calling thread; ``_TimeoutJudge`` carries that binding into its
    #     worker thread, so the emitter always knows whose call it is on;
    #   * the token is RETIRED when the block exits, and an event arriving on a
    #     retired token is dropped -- its candidate has already been disposed of;
    #   * a worker whose wrapper could not carry a binding (a ``_TimeoutJudge``
    #     built without an owner) is still recognised as late: it is not the
    #     thread that drives the page loop, and no judge call is in flight.
    _judge_event_binding: tuple | None = None
    _judge_calls_in_flight: int = 0
    _judge_scratch_blocks: int = 0

    @property
    def _judge_sink_state(self):
        state = getattr(self, "_judge_sink_state_local", None)
        if state is None:
            state = threading.local()
            self._judge_sink_state_local = state
        return state

    def _current_judge_binding(self):
        return getattr(self._judge_sink_state, "binding", None)

    def _bind_judge_events(self, binding) -> None:
        self._judge_sink_state.binding = binding

    def _enter_judge_call(self) -> None:
        with self._judge_call_lock:
            self._judge_calls_in_flight += 1

    def _leave_judge_call(self) -> None:
        with self._judge_call_lock:
            self._judge_calls_in_flight = max(0, self._judge_calls_in_flight - 1)

    @property
    def _judge_call_lock(self):
        lock = getattr(self, "_judge_call_lock_obj", None)
        if lock is None:
            lock = threading.Lock()
            self._judge_call_lock_obj = lock
        return lock

    def _record_judge_event(self, state: DocumentState, event) -> None:
        binding = self._current_judge_binding()
        if binding is not None:
            token, sink = binding
            if token in self._retired_judge_tokens:
                return  # the invocation that made this call is over
            sink.append(event)
            return

        active = self._judge_event_binding
        if active is not None:
            # A worker with no carried binding, emitting while a block is open:
            # capture it rather than let it reach the document unattributed.
            active[1].append(event)
            return

        if (
            self._judge_scratch_blocks
            and threading.get_ident() != getattr(self, "_judge_loop_thread", None)
            and self._judge_calls_in_flight <= 0
        ):
            # Not the loop thread, nothing waiting on any judge: an abandoned
            # worker from a call whose verdict was already taken as a rejection.
            return

        state.events.append(event)

    @property
    def _retired_judge_tokens(self) -> set:
        retired = getattr(self, "_retired_judge_tokens_set", None)
        if retired is None:
            retired = set()
            self._retired_judge_tokens_set = retired
        return retired

    @contextmanager
    def _judge_events_to(self, sink: list):
        """Own every judge-side event of one re-judge, on any thread."""
        token = object()
        previous_binding = self._judge_event_binding
        previous_thread_binding = self._current_judge_binding()
        self._judge_event_binding = (token, sink)
        self._bind_judge_events((token, sink))
        self._judge_loop_thread = threading.get_ident()
        self._judge_scratch_blocks += 1
        try:
            yield
        finally:
            self._retired_judge_tokens.add(token)
            self._judge_event_binding = previous_binding
            self._bind_judge_events(previous_thread_binding)

    @staticmethod
    def _add_page_cost(ps, cost: float | None) -> None:
        """Charge a page's recorded spend. Delegates to the one recorder.

        Kept as a name because tests and older call sites reach for it; the rule
        itself lives with the journal helper in ``socr.core.state`` so journaling
        and recording cannot drift apart (round 6).
        """
        add_page_cost(ps, cost)

    @staticmethod
    def _page_total_cost(ps) -> float | None:
        """Read the page's recorded spend. A pure reader -- it derives nothing.

        Round 4 computed this by summing ``ps.attempts``, and round 5 showed why
        that cannot work: the list omits a refused GH-96 escalation candidate
        live, and resume rebuilds it as the single frozen winner, so the first
        resumed run's sidecar rewrite destroyed the very total it had restored
        and the second resume regained the rejected rung's budget.
        ``_add_page_cost`` records the fact instead; this reads it.
        """
        if ps is None:
            return None
        return getattr(ps, "page_cost_usd", None)

    def _rejudge_crop_patched_page(
        self,
        state: DocumentState,
        page_num: int,
        ps,
        bo: PageOutput,
        accepted_text: str,
        judge,
        provider,
    ) -> PageOutput:
        """Send a crop-patched page back through the judge before it can ship.

        P3's invariant is that judged bytes are shipped bytes on EVERY path, and
        ruling step 4 (``docs/log/2026-09-01_conceptual-revision.md``) says a
        crop reread is an escalation tool that runs before the verdict, never
        after an accept.  ``_reread_page_tables`` patches ``best_output.text``
        after ``route_page`` has already accepted or exhausted the ladder, so
        its output is a NEW CANDIDATE: it goes back through the same judge, with
        the same provider context as the reading that won.

        Three things follow, and all three are load-bearing (cold review round 2):

        1. **The candidate is judged clean.**  It is a fresh reading, so it is
           presented the way a first rung's output is presented -- SUCCESS, no
           failure mode, no rejection class, ``audit_passed`` unset by any
           earlier verdict -- and as a COPY, because the judge legitimately
           mutates what it is handed (pre-verdict header repair,
           ``rejection_class``) and none of that may reach shipped output unless
           the verdict accepts.
        2. **An acceptance PROMOTES.**  Copying only the text left the page
           carrying the fail-closed state ladder exhaustion had stamped
           (``audit_passed=False``, ``native_table_structure_failed``, the floor
           PNG), so ``_grid_authored_attempt`` still refused it and the page
           shipped the structure-class floor rather than the bytes the judge had
           just accepted -- the recovery was invisible.  An accepting re-judge
           therefore leaves exactly the state a first-time acceptance leaves.
        3. **A refusal leaves no trace on the shipped bytes.**  Judge-side events
           are captured to a scratch list; they join ``state.events`` only if the
           candidate is accepted.  Otherwise they describe bytes that never ship,
           and one of them (``native_table_verifier_hard_fail``) would mark the
           page untrusted in ``tables_trust.json``.  The refusal itself is
           recorded as exactly one ``table_reread_rejudged`` event.

        Returns the output the caller must keep using.
        """
        from socr.core.audit_log import AuditEvent

        if bo is None:
            return bo
        patched_text = bo.text or ""
        if patched_text == (accepted_text or ""):
            return bo

        # Round 3, finding 1: the crop lane is reachable on ladder exhaustion
        # with a non-empty OPERATIONAL failure -- a provider whose read was
        # truncated, or errored. Presenting the patched text as a fresh SUCCESS
        # candidate would launder that failure into a terminal SUCCESS page,
        # because the promotion below writes exactly the state a first-time
        # acceptance writes. A crop reread repairs a TABLE; it does not repair a
        # truncated read, and it has no evidence about the part of the page the
        # provider never returned. Refuse before spending a judge call.
        if (
            bo.status is PageStatus.ERROR
            or bo.failure_mode is FailureMode.TRUNCATED
            or (bo.error or "").strip()
        ):
            self._refuse_crop_patch(
                state,
                page_num,
                bo,
                accepted_text,
                reason=(
                    "the winning attempt is an operational failure "
                    f"(status={bo.status.value}, failure_mode={bo.failure_mode.value}); "
                    "a crop reread cannot repair it"
                ),
                event_data={
                    "accepted": False,
                    "judged": False,
                    "provider_id": getattr(provider, "id", ""),
                },
            )
            return bo

        candidate = replace(
            bo,
            text=patched_text,
            status=PageStatus.SUCCESS,
            failure_mode=FailureMode.NONE,
            audit_passed=True,
            rejection_class="",
            judge_reason="",
            audit_notes=list(bo.audit_notes),
            figures=list(bo.figures),
        )

        judge_events: list = []
        started = time.time()
        try:
            with self._judge_events_to(judge_events):
                decision = judge.assess(candidate, provider)
        except Exception as exc:
            logger.warning(
                "re-judge of the crop-patched p%d errored (%s); keeping the accepted text",
                page_num,
                exc,
            )
            decision = None
        elapsed = time.time() - started
        accepted = decision is not None and decision.accept

        # Metering (round 2 finding 4, corrected in round 3). The re-judge is a
        # second JUDGE call on a page the ladder has already paid for, and it was
        # invisible to ``DocumentState.total_cost``, the run journal and the
        # budget view. Round 2 attributed it to the winning OCR profile, which is
        # wrong in both directions: a heuristic decision was journalled as the
        # cloud engine at that engine's page price, and a paid remote judge over
        # a local winner recorded zero.
        #
        # The call is priced by the model that RAN it. ``agentic_judge_model`` is
        # the judge ``_build_page_judge`` actually built -- never re-resolved from
        # config here, so a run degraded to heuristics cannot name the VLM that
        # did not run. A judge model that is not one of the metered rungs runs on
        # a host socr provides, so it costs the known 0.00 rather than "unknown".
        judge_model = state.agentic_judge_model or JUDGE_IDENTITY_HEURISTIC
        judge_profile = profile_by_model(judge_model)
        judge_cost = 0.0 if judge_profile is None else judge_profile.cost_per_page_usd
        state.record_engine_run(
            EngineResult(
                document_path=state.handle.path,
                engine=judge_model,
                status=DocumentStatus.SUCCESS if accepted else DocumentStatus.AUDIT_FAILED,
                cost=judge_cost,
                processing_time=elapsed,
            ),
            page_nums=[page_num],
        )
        # Durability: the sidecar persists only the winning output's
        # ``cost_usd``, and resume rebuilds the page's ``EngineResult`` from that
        # one field. Folding the judge call in there is what keeps a resumed
        # run's arithmetic identical to the live one, so a partial resume cannot
        # regain budget already spent. Same seam the equation lane uses.
        if bo.cost_usd is None or judge_cost is None:
            bo.cost_usd = None
        else:
            bo.cost_usd = bo.cost_usd + judge_cost

        event_data = {
            "accepted": accepted,
            "judged": True,
            "reason": (decision.reason if decision is not None else "judge errored") or "",
            "judge_model": judge_model,
            "judge_cost_usd": judge_cost,
            "provider_id": getattr(provider, "id", ""),
        }

        if accepted:
            # Promote exactly as a first-time acceptance would leave the page.
            # The candidate's own fields are taken, so any repair the judge made
            # before its verdict is part of the judged bytes.
            bo.text = candidate.text
            bo.status = candidate.status
            bo.failure_mode = candidate.failure_mode
            bo.rejection_class = candidate.rejection_class
            bo.audit_passed = True
            bo.judge_reason = decision.reason or ""
            if decision.confidence:
                bo.confidence = decision.confidence
            # Exhaustion stamps set BEFORE the crop lane ran. A first-time
            # acceptance leaves none of them, and while they stand the fail-closed
            # floor outranks the reading that was just accepted. Facts about the
            # NATIVE layer (``native_table_unverifiable``,
            # ``native_table_header_unattributed``, ``native_rotated_text_shredded``)
            # are NOT cleared: they are true of the page either way.
            ps.native_table_structure_failed = False
            ps.scanned_table_evidence_failed = False
            ps.d3_floor_png_ref = ""
            ps.rotated_shred_png_ref = ""
            state.events.extend(judge_events)
            state.events.append(
                AuditEvent(
                    page_num=page_num,
                    kind="table_reread_rejudged",
                    engine=bo.engine or "",
                    detail=(
                        "crop reread patched the page and the judge accepted the "
                        f"patched text ({decision.reason or 'accepted'})"
                    ),
                    data=event_data,
                )
            )
            return bo

        self._refuse_crop_patch(
            state,
            page_num,
            bo,
            accepted_text,
            reason=event_data["reason"],
            event_data=event_data,
        )
        return bo

    def _refuse_crop_patch(
        self,
        state: DocumentState,
        page_num: int,
        bo: PageOutput,
        accepted_text: str,
        *,
        reason: str,
        event_data: dict,
    ) -> None:
        """Put back the bytes that were already judged, and say why, once.

        ``bo.judge_reason`` is deliberately NOT overwritten: it records the route
        verdict on the bytes that still ship, not a verdict on bytes that do not.
        """
        from socr.core.audit_log import AuditEvent

        bo.text = accepted_text or ""
        bo.audit_notes.append(
            f"crop reread patch refused p{page_num} ({reason}); "
            "shipped the previously accepted text"
        )
        data = dict(event_data)
        data["reason"] = reason
        data["accepted"] = False
        state.events.append(
            AuditEvent(
                page_num=page_num,
                kind="table_reread_rejudged",
                engine=bo.engine or "",
                detail=(
                    f"crop reread patched the page but it was refused ({reason}); "
                    "the previously accepted bytes ship"
                ),
                data=data,
            )
        )

    def _reread_page_tables(
        self,
        state: DocumentState,
        page_num: int,
        raw_crops: list,
        extractor,
    ) -> tuple[int, int]:
        """Reconcile crop readings against whole-page OCR for one page.

        Called by the fused agentic loop AFTER ``locate_tables`` and
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
        failed_crops: list = []
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
            elif getattr(c, "_failed", ""):
                # GH-166: a located crop that produced nothing. Previously these
                # were dropped inside the extractor with no sentinel at all, so
                # a page whose crops ALL failed returned (0, 0) and left the
                # incumbent table looking verified -- the check that would have
                # contradicted it simply left no trace.
                failed_crops.append(c)
                state.events.append(
                    AuditEvent(
                        page_num=page_num,
                        kind="dualpass_crop_failed",
                        engine=bo.engine,
                        detail=(
                            f"crop reread produced no reading ({c._failed}); "
                            "the incumbent table is unverified"
                        ),
                        data={"source": c.source, "reason": c._failed},
                    )
                )
                bo.audit_notes.append(
                    f"dual-pass crop failed p{page_num} ({c.source}, {c._failed}); "
                    "kept existing text"
                )
            else:
                crops.append(c)

        if not crops:
            # Every located crop failed or timed out. The page keeps its
            # incumbent table, so a consumer must be told the verification never
            # completed -- the per-crop events above carry that, and both kinds
            # are in TABLE_DISTRUST_KINDS so `tables_trust.json` shows the page.
            return 0, 0

        # If ANY crop on this page timed out, do NOT auto-patch even when the
        # config enables it. Partial crop coverage means we cannot safely assert
        # that the remaining crops represent the full table set; patching on
        # incomplete evidence risks data loss. Force flag-only for this page.
        # GH-166 review (P2): a FAILED crop means the same thing a timed-out one
        # does -- partial coverage. The comment above says patching on
        # incomplete evidence risks data loss; that reasoning does not depend on
        # which way the crop failed, so both force flag-only.
        effective_auto_patch = (
            self.config.auto_patch_tables and not had_timeout and not failed_crops
        )
        crop_repair_fallback = False
        crop_repair_declined = False
        original_text = bo.text

        def _reconcile_with_optional_fallback(fitz_page=None):
            nonlocal effective_auto_patch, crop_repair_fallback, crop_repair_declined
            from socr.tables.crop_repair import (
                crop_patch_improves_verification,
                page_needs_crop_repair_fallback,
            )

            # GH-166 review (P2), second half: the fallback can turn auto-patch
            # back ON, so gating only the initial assignment would let a page
            # with a FAILED crop patch anyway by this route. `failed_crops`
            # joins `had_timeout` here for the same reason it does above --
            # partial coverage is partial however the crop failed.
            needs_crop_fallback = (
                not had_timeout
                and not failed_crops
                and page_needs_crop_repair_fallback(
                    original_text,
                    native_table_unverifiable=getattr(ps, "native_table_unverifiable", False),
                    fitz_page=fitz_page,
                )
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
            from socr.core.pdf import open_pdf

            with open_pdf(state.handle.path) as _crop_doc:
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
            from socr.core.pdf import open_pdf

            # Close and evict any previously cached doc before opening a new one
            # so we hold at most one open handle at a time.
            if _fitz_doc_cache:
                try:
                    _fitz_doc_cache[0].close()
                except Exception:
                    pass
                _fitz_doc_cache.clear()
            doc = open_pdf(pdf_path)
            _fitz_doc_cache.append(doc)
            # PyMuPDF pages are 0-indexed; socr page numbers are 1-indexed.
            return doc[page_num - 1]

        def is_table_page(page_num: int) -> bool:
            ps = state.pages.get(page_num)
            return self._page_has_tables(page_num, ps)

        def record_event(event) -> None:
            self._record_judge_event(state, event)

        native_judge = NativeTableVerifierJudge(
            inner=inner_judge,
            get_fitz_page=get_fitz_page,
            is_table_page=is_table_page,
            record_event=record_event,
        )

        def native_trusted(page_num: int) -> bool | None:
            # GH-163: the source-evidence verifier must select the native
            # verifier on TRUST, not on the presence of words. A scanned page
            # with a baked-in OCR layer has words and no trustworthy reading.
            ps = state.pages.get(page_num)
            return None if ps is None else bool(ps.is_born_digital)

        return SourceEvidenceTableJudge(
            inner=native_judge,
            get_fitz_page=get_fitz_page,
            record_event=record_event,
            native_trusted=native_trusted,
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
        has_native_math = any(
            {"native", "math"}
            <= {
                part.strip()
                for part in str(run.engine).replace(",", "+").split("+")
                if part.strip()
            }
            for run in state.engine_runs
        )
        if has_native_math:
            from socr.math.recover import CORRUPT_MATH_PROMPT

            inputs["native+math"] = (
                self.config.math_model,
                "ollama-compatible",
                None,
                CORRUPT_MATH_PROMPT,
            )
        names: set[str] = set()
        for run in state.engine_runs:
            for part in str(run.engine).replace("+", ",").split(","):
                part = part.strip()
                if part and part != "native":
                    # Historical manifests may label a run consensus(<engine>).
                    # Strip that compatibility wrapper to the underlying engine
                    # so cached manifests remain replayable after consensus was
                    # removed from the active pipeline.
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
        self,
        state: DocumentState,
        output_dir: Path,
        saved_body: str | None = None,
        records: list[FinalizedPageRecord] | None = None,
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
                records=records,
            )
            manifest.save(doc_dir / "manifest.json")
            if not self.config.quiet:
                console.print(f"  [dim]Manifest: {doc_dir / 'manifest.json'} (replayable)[/dim]")
        except Exception as exc:
            logger.warning("manifest write failed (non-fatal): %s", exc)

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

    # ------------------------------------------------------------------
    # Assembly persistence helpers (PP-1)
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
        extra_figures: list | None = None,
        record: FinalizedPageRecord | None = None,
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
        # and fragment do so the sidecar remains consistent with the selected
        # output, including historical or reconstructed state.
        from socr.core.manifest import (
            _whole_doc_page_texts,
            finalized_page_record,
            is_page_failed_marker,
            structure_class_grid_winner,
        )

        if record is None and self._final_records:
            record = self._final_records.get(page_num)
        if record is None and ps:
            whole_doc = _whole_doc_page_texts(state)
            record = finalized_page_record(state, page_num, whole_doc)

        winning_out = record.output if record is not None else None
        winning_dict = winning_out.to_dict() if winning_out is not None else {}
        disp = record.disposition if record is not None else None
        disposition_dict = disp.to_dict() if disp is not None else None
        # MAJOR 7(b) on #269: persisted so a RESUMED run can re-derive
        # ``structure_class_model_pages`` membership without the full attempt
        # history. Resume collapses ``p.attempts`` to the single frozen
        # winner (``_restore_terminal_page_state``), so on a resumed run a
        # clean-passing S1 case (i) grid winner is indistinguishable, from
        # ``p.attempts`` alone, from a page that was never an S1 case at all
        # -- it silently drops out of the bucket on run 2 even though it was
        # correctly counted on run 1. See ``PageState.structure_class_model_kept_on_resume``.
        structure_class_model_kept = bool(ps and structure_class_grid_winner(ps) is not None)

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
        # GH-171: figures reach the sidecar from TWO places, and only the first
        # existed. `_describe_and_embed_figures` attaches its results to the
        # returned `EngineResult`, not to `state.engine_runs` -- and it runs
        # AFTER the assemble-time flush. So the sidecar the pipeline calls
        # authoritative shipped with an empty `figure_refs` on every page that
        # had figures, and no later pass corrected it (`_rewrite_all_fragments`
        # rewrites `.md` only). `extra_figures` carries that second source; the
        # re-flush after the figure phase supplies it.
        figure_sources: list = list(state.engine_runs)
        seen_figs: list = []
        for run in figure_sources:
            seen_figs.extend(getattr(run, "figures", []) or [])
        seen_figs.extend(extra_figures or [])

        figure_refs: list[dict] = []
        for fig in seen_figs:
            if getattr(fig, "page_num", None) == page_num:
                ref = fig.to_dict() if hasattr(fig, "to_dict") else {}
                if ref not in figure_refs:  # the same figure can reach both sources
                    figure_refs.append(ref)

        payload: dict = {
            "page_num": page_num,
            "disposition": disposition_dict,
            # Status mirrors the winning output's status, or "missing" when
            # no output exists.
            "status": winning_dict.get("status", "missing"),
            # Keep the terminal disposition directly addressable for consumers
            # that inspect the sidecar without unpacking ``winning_output``.
            "failure_mode": winning_dict.get("failure_mode", "none"),
            "audit_passed": winning_dict.get("audit_passed", False),
            # terminal: True when this is the definitive sidecar written at
            # assembly time; False for a provisional mid-run incremental
            # flush from PP-2 that may be superseded by the authoritative write.
            "terminal": terminal,
            # Engine / provider provenance.
            "engine": winning_dict.get("engine", ""),
            "provider": winning_dict.get("provider_id", ""),
            "cost_usd": winning_dict.get("cost_usd", 0.0),
            # Cold review round 4: the WHOLE page's spend, not the winner's own
            # per-attempt cost. The two differ exactly when the ladder paid for a
            # rung it then rejected -- including this branch's crop-recovery path,
            # where a paid rung is refused and a FREE local winner is promoted.
            # Resume rebuilds the page's EngineResult from this field, so
            # persisting only the winner handed that budget back and let a
            # resumed run spend it a second time. ``None`` when any attempt was
            # unmetered: an unknown subtotal must never restore as zero.
            "page_cost_usd": self._page_total_cost(state.pages.get(page_num)),
            # Full serialised winning PageOutput.  PP-5 reconstructs a skipped
            # page's in-memory PageState.best_output from this dict (paired with
            # the fragment text) so the resumed run carries the SAME status /
            # engine / provider / audit verdict the original produced — not just
            # the stitched body.  Empty dict when no winning output exists.
            "winning_output": winning_dict,
            # Run-level fingerprint (shared across all pages of this run).
            "run_fingerprint": self._run_fingerprint(),
            # GH-214 provenance: which socr produced this page. The fingerprint
            # above PREVENTS stale reuse from here on, but it is opaque -- it
            # cannot answer "was this page made by older code?" for a page
            # already on disk. These two fields make staleness detectable after
            # the fact, which is what an existing corpus needs.
            "socr_version": _socr_version(),
            "socr_source_digest": _socr_source_digest(),
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
            # #263: rotated page whose native layer was refused as shredded,
            # and the image ref its floor ships in place of the fragments.
            "native_rotated_text_shredded": (
                bool(getattr(ps, "native_rotated_text_shredded", False)) if ps else False
            ),
            "rotated_shred_png_ref": (str(getattr(ps, "rotated_shred_png_ref", "")) if ps else ""),
            # GH-151 TICKET-B1: grid-shape defect found at extraction time.
            "native_table_structure_defective": (
                bool(getattr(ps, "native_table_structure_defective", False)) if ps else False
            ),
            "native_table_emission_defect": (
                str(getattr(ps, "native_table_emission_defect", "")) if ps else ""
            ),
            # GH-346: persisted next to the emission term. Without it a resume
            # dropped the content provenance entirely, so a page reloaded from
            # its sidecar disagreed with the page that wrote it.
            "native_table_content_defect": (
                str(getattr(ps, "native_table_content_defect", "")) if ps else ""
            ),
            # GH-200: header-attribution HARD verdict found at extraction time.
            "native_table_header_unattributed": (
                bool(getattr(ps, "native_table_header_unattributed", False)) if ps else False
            ),
            # TR-3: per-region geometry hard-fail flag (D3 floor trigger).
            "native_table_unverifiable": (
                bool(getattr(ps, "native_table_unverifiable", False)) if ps else False
            ),
            # GH-371: native failed-region identity and the expected number of
            # regions, used to preserve surrounding prose on D3 resume.
            "native_table_unverifiable_ordinals": (
                list(getattr(ps, "native_table_unverifiable_ordinals", [])) if ps else []
            ),
            "native_table_region_count": (getattr(ps, "native_table_region_count", 0) if ps else 0),
            "native_table_region_identities": (
                list(getattr(ps, "native_table_region_identities", [])) if ps else []
            ),
            # TR-3: image ref for the D3 floor PNG (empty string when not rendered).
            "d3_floor_png_ref": (str(getattr(ps, "d3_floor_png_ref", "")) if ps else ""),
            # GH-90: scanned-table evidence check failed; D3 floor applies (prose splice).
            "scanned_table_evidence_failed": (
                bool(getattr(ps, "scanned_table_evidence_failed", False)) if ps else False
            ),
            "chart_asset_render_failed": (
                bool(ps.chart_asset_render_failed) if ps else False  # PP-7
            ),
            # GH-318: chart eligibility never resolved for this page.
            # P4-R: the lane could not reach its provider on the run that wrote
            # this sidecar, so the page is not a finished result even though it
            # ships a clean native SUCCESS (finding 5).
            "equation_lane_retry_pending": (
                bool(getattr(ps, "equation_lane_retry_pending", False)) if ps else False
            ),
            "chart_asset_detection_failed": (
                bool(getattr(ps, "chart_asset_detection_failed", False)) if ps else False
            ),
            "judge_rejected": bool(ps.judge_rejected) if ps else False,
            # MAJOR 7(b): S1 case (i) resume-idempotency flag (see above).
            "structure_class_model_kept": structure_class_model_kept,
            # GH-353 TICKET-D1a: the table judge ladder's durable page-level
            # disposition (set by B1's gate, read by C3's manifest guard and
            # C2's document aggregation). ``None`` when the page never
            # reached a ladder terminal (flag off, ladder ACCEPTED, or no
            # tables on the page). Persisted as the bare ``FailureMode``
            # value string so a resumed run's ``_restore_terminal_page_state``
            # can rebuild the SAME disposition without re-judging -- without
            # this, a skipped page would silently lose its REJECTED/UNVERIFIED
            # verdict on resume (``page_events`` below already carries the
            # per-table audit trail; this field is the page-level reduction
            # ``reduce_page_ladder`` produced from it).
            "table_ladder_disposition": (
                ps.table_ladder_disposition.value
                if ps and ps.table_ladder_disposition is not None
                else None
            ),
            # GH-359 (cubic P1): an emitted table on this page reached
            # assemble with no ladder terminal. Persisted so a resumed run
            # refuses D1b's REJECTED skip-and-keep for THIS page -- the
            # disposition alone cannot express "rejected, but not everything
            # here was witnessed".
            "table_ladder_incomplete": bool(ps and ps.table_ladder_incomplete),
            # GH-367: per-table lift/hold record so a resumed run does not
            # silently re-clamp a table whose contradictions were disproved.
            "binding_adjudication": dict(ps.binding_adjudication) if ps else {},
            # Audit log subset for this page.
            "audit_events": page_events,
            # Table-pass and figure refs.
            "figure_refs": figure_refs,
        }

        # P1: sparse table retry latch -- persisted ONLY when True so default-off
        # sidecars remain byte-identical and satisfy P6 disposition persistence contracts.
        if bool(getattr(ps, "table_judge_retry_pending", False)):
            payload["table_judge_retry_pending"] = True
            rungs = list(getattr(ps, "table_judge_retry_rungs", []) or [])
            if rungs:
                payload["table_judge_retry_rungs"] = rungs

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
            written terminal at assemble must be RE-OCR'd, never skipped) --
            UNLESS the page carries a GH-353 table-judge-ladder ``REJECTED``
            disposition (see below), the ONE deliberate exception.
          * The winning output's ``audit_passed`` is exactly ``True`` (GH-161).
            Status says the extraction succeeded as an OPERATION; only the audit
            verdict says the CONTENT was accepted.  A judge-rejected scanned page
            keeps status SUCCESS with ``audit_passed=False`` and must be re-OCR'd
            -- again subject to the same ``REJECTED``-disposition exception.
          * ``pages/NNN.md`` exists and is readable.
          * The serialised ``winning_output`` is present and rebuilds into a
            ``PageOutput`` without error.

        GH-353 TICKET-D1b, the one deliberate exception: a page whose sidecar
        ``table_ladder_disposition`` is ``FailureMode.TABLE_REJECTED`` **and
        whose ``table_ladder_incomplete`` is false** skips
        the two checks above and IS skip-and-kept even though C3's
        ``_apply_ladder_disposition_guard`` demotes such a page's finalized
        output to ``status=WARNING, audit_passed=False``. REJECTED is a
        corroborated CONTENT verdict (both ladder rungs looked and said no),
        not an infra doubt, so under the SAME input+config it is final --
        unlike every other condition here, which reprocesses on ANY doubt.
        GH-359 (cubic P1) narrows the exception: it is forfeited when assemble
        had to backfill a terminal for any emitted table on the page. The
        disposition is a page-level reduction, so a page holding one REJECTED
        table keeps ``TABLE_REJECTED`` even when a second table was never
        witnessed; without the flag that page would be skip-and-kept forever
        and the unwitnessed table could never be looked at.
        ``FailureMode.TABLE_UNVERIFIED`` gets NO such exception: the ladder
        ran out of witnesses/rungs without an answer, which IS infra-shaped
        doubt, so an UNVERIFIED page falls through the SUCCESS/audit checks
        like any other unresolved page and is reprocessed. Both dispositions
        are covered by ``run_fingerprint`` above, which already binds B1's
        ladder extras (flag, rung identities, timeout, prompt digest) -- a
        changed rung config forces reprocessing before either disposition is
        ever consulted.

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

            # P4-R (cold review round 1, finding 5): the sidecar was written by a
            # run whose equation lane never reached a provider. Provider
            # availability is transient and invisible to the fingerprint, so
            # everything above can match while the page still has an unread
            # equation on it. Refuse the skip when a provider is available NOW;
            # keep it when there is still none, so a genuinely offline rerun is
            # not forced to redo work it cannot do any better.
            if meta.get("equation_lane_retry_pending") is True:
                ps_now = state.pages.get(page_num)
                if ps_now is not None and self._is_equation_region_lane_page(page_num, ps_now):
                    profile, _reason = self._equation_lane_provider()
                    if profile is not None:
                        logger.debug(
                            "PP-5: p%d not resumed; its equation lane had no provider and "
                            "one is available now",
                            page_num,
                        )
                        return None

            # GH-367: fingerprint+checksum matched, so this sidecar is for
            # the current run configuration. Restore the per-table lift
            # record even when the page itself is NOT skippable
            # (UNVERIFIED reprocess): without it, bind() re-fires and
            # silently re-clamps a table whose contradictions were
            # already disproved.
            self._apply_binding_adjudication_meta(state, page_num, meta)

            # P1: the sidecar was written by a run whose table judge ladder
            # encountered transient rung unavailability. When a rung is attemptable
            # NOW, refuse the skip so the page is re-judged. When rungs remain
            # unavailable, do not force reprocessing here: permit an otherwise-valid
            # D1b REJECTED restore (which carries the latch forward in
            # _restore_terminal_page_state), while UNVERIFIED pages will fall through
            # to reprocess via the audit/status gate below.
            if meta.get("table_judge_retry_pending") is True:
                sidecar_rungs = meta.get("table_judge_retry_rungs")
                if self._table_judge_rung_available_now(
                    [str(k) for k in sidecar_rungs] if isinstance(sidecar_rungs, list) else []
                ):
                    logger.debug(
                        "PP-5: p%d not resumed; table judge rung was unavailable, reachable now",
                        page_num,
                    )
                    return None

            # GH-353 TICKET-D1b: the table judge ladder's REJECTED terminal is a
            # DELIBERATE exception to "doubt reprocesses" -- a REJECTED verdict is
            # a corroborated CONTENT judgment (rung 1 + rung 2 both looked and
            # said no; see ``judge/table_ladder.py``), not an infra doubt, so it
            # is final for the SAME input+config and must skip-and-keep rather
            # than fall through the SUCCESS/audit_passed checks below (which a
            # REJECTED page's finalized output never satisfies -- C3's
            # ``_apply_ladder_disposition_guard`` demotes it to
            # ``status=WARNING, audit_passed=False`` precisely so a naive resume
            # gate would refuse to skip it). UNVERIFIED is the opposite case: the
            # ladder ran out of witnesses/rungs without an answer, which IS an
            # infra-shaped doubt, so it deliberately gets NO exception here and
            # falls through to reprocess like any other non-SUCCESS page.
            # ``run_fingerprint`` above already binds B1's ladder extras (flag,
            # rung identities, timeout, prompt digest), so a changed rung
            # config already forced reprocessing before this line is reached --
            # this check never needs to re-verify rung identity itself.
            # The full winning PageOutput dict must be present and rebuildable.
            # Read BEFORE the D1b exception is decided: the exception's scope
            # depends on what actually shipped, not on the disposition alone.
            winning = meta.get("winning_output")
            if not isinstance(winning, dict) or not winning:
                return None

            disposition_raw = meta.get("table_ladder_disposition")
            # GH-359 (cubic P1): the exception is for a page whose tables were
            # ALL adjudicated. If assemble had to backfill a terminal for any
            # emitted table here, "both rungs looked and said no" is false for
            # that table, so the page falls through and is reprocessed.
            #
            # P2 / GH-317 (cold review round 1, finding 2): and the exception is
            # about a JUDGED TABLE, never about a page whose every ladder rung
            # was refused. Those two can co-occur -- a rejected rung output can
            # record TABLE_REJECTED while still authoring no usable grid, so the
            # page ships the fail-closed floor. Without this clause D1b bypasses
            # the SUCCESS and audit_passed gates, and the only guard left is
            # ``is_page_failed_marker``, which deliberately returns False for a
            # REGIONAL floor (marker surrounded by preserved prose). The floored
            # page would then be restored verbatim and never re-OCR'd. The floor
            # is a page-level fail-closed disposition; it is never skippable.
            floor_shipped = (
                winning.get("failure_mode") == FailureMode.STRUCTURE_CLASS_LADDER_EXHAUSTED.value
            )
            is_ladder_rejected = (
                disposition_raw == FailureMode.TABLE_REJECTED.value
                and not bool(meta.get("table_ladder_incomplete"))
                and not floor_shipped
            )

            if not is_ladder_rejected:
                # Status MUST be SUCCESS.  A page written terminal at assemble time
                # with an ERROR / WARNING / timed-out output (e.g. a cascade-halt page
                # whose best_output is the ERROR attempt, or a flagged native fallback)
                # is NOT a clean result — re-OCR it, never skip it.
                if winning.get("status") != PageStatus.SUCCESS.value:
                    return None

                # GH-161: the audit verdict MUST be an explicit pass.  Status alone is
                # not a quality signal — it says the extraction SUCCEEDED as an
                # operation, not that the content was accepted.  Agentic best-effort
                # (``_best_effort`` in pipeline/agentic.py, reached when the judge
                # accepted nothing) keeps the most trustworthy attempt with its
                # provider status still SUCCESS while ``att.output.audit_passed =
                # att.accepted`` makes it False; on a SCANNED page no native fallback
                # exists to demote that winner, so ``_winning_page_output`` returns it
                # verbatim and the sidecar records status="success" beside
                # audit_passed=false.  Skipping there restores text that EVERY judge
                # rejected — silent corpus poisoning on resume, and the one gate
                # condition that the born-digital sibling of this shape (demoted to
                # WARNING) never needed.  Exactly ``True``, mirroring ``terminal``: a
                # missing key or a non-bool is doubt, and doubt re-OCRs.
                if winning.get("audit_passed") is not True:
                    self._record_ledger_audit_reject(state, page_num, winning)
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

    def _record_ledger_audit_reject(
        self, state: DocumentState, page_num: int, winning: dict
    ) -> None:
        """GH-161: make a refused resume of a judge-rejected page VISIBLE.

        Every other condition in ``_load_terminal_page`` refuses silently, and for
        those that is defensible: a missing sidecar or a fingerprint change is
        bookkeeping, not a quality verdict.  This one is a quality verdict — the
        ledger holds a page whose content every judge rejected — so refusing it
        silently would trade one silent failure (restoring rejected text) for
        another (reprocessing it with no record that the ledger was poisoned).

        The event reaches three surfaces without any new plumbing: the page's
        ``pages/NNN.json`` sidecar (``page_events``, filtered by ``page_num``),
        the run's ``audit_log.json`` (``_write_audit_log`` reads
        ``state.events``), and — via that writer's summary line plus the console
        line below — the CLI.  Document status is deliberately untouched: the page
        is about to be re-OCR'd, so the document's outcome must reflect the FRESH
        result, not a stale verdict that no longer describes what shipped.

        Idempotent: the gate runs twice for an OCR page (the pre-pass over
        ``ocr_pages`` and again in the main loop), so a duplicate event for the
        same page is suppressed rather than double-counted in the audit summary.
        """
        try:
            from socr.core.audit_log import AuditEvent

            kind = "resume_ledger_audit_reject"
            if any(
                getattr(ev, "page_num", None) == page_num and getattr(ev, "kind", "") == kind
                for ev in state.events
            ):
                return
            engine = str(winning.get("engine", "") or "")
            state.events.append(
                AuditEvent(
                    page_num=page_num,
                    kind=kind,
                    engine=engine,
                    detail=(
                        "terminal ledger entry was SUCCESS but audit_passed=False "
                        "(judge-rejected content); refusing the resume skip and "
                        "re-OCRing the page"
                    ),
                    data={
                        "status": str(winning.get("status", "")),
                        "audit_passed": winning.get("audit_passed"),
                        "skip_reason": str(winning.get("skip_reason", "") or ""),
                    },
                )
            )
            if not self.config.quiet:
                console.print(
                    f"  [yellow]p{page_num}: ledger entry judge-rejected "
                    f"(audit_passed=false) — re-OCRing[/yellow]"
                )
        except Exception as exc:  # surfacing must never break a run
            logger.debug("GH-161: could not record ledger audit reject for p%d (%s)", page_num, exc)

    def _probe_backend_idle(self) -> bool:
        """Liveness probe for THIS RUN's VLM backend, behind a swappable seam.

        GH-222: the cascade-halt guard used to call ``probe_ollama_idle()`` with
        no arguments, and that function defaulted to a hardcoded
        ``http://localhost:11434``.  On any deployment that does not run an
        Ollama daemon there — vLLM, HPC, a remote Ollama host — the probe
        reported the backend dead for a perfectly healthy machine, so a single
        timeout anywhere in the ladder truncated the document and blamed a
        hardware failure that never happened.  ``--strict-local`` on a vLLM
        backend is the worst case: the ladder is local-only, so one slow page
        ends the run.

        Which server is asked follows ``qwen_backend`` through
        ``OPENAI_COMPATIBLE_BACKENDS`` — the SAME predicate ``make_table_reader``
        uses to decide which server to send crops to, so the probe and the reader
        can never end up asking about different machines.  A vLLM/SGLang backend
        is probed at ``qwen_vllm_url``; anything else is an Ollama daemon, whose
        host is RESOLVED (explicit config, then ``OLLAMA_HOST``, then the
        localhost default) rather than assumed.  ``auto`` resolves to a local
        Ollama daemon everywhere else in this codebase, so it probes Ollama here
        too; deriving "vLLM" from a non-default ``qwen_vllm_url`` alone would
        make the probe disagree with the crop reader, which is the class of bug
        this is fixing.

        ``backend_probe`` remains as an override seam for a backend socr does not
        model, but the default path no longer depends on a caller knowing to
        assign it.

        Deliberately unchanged here: WHICH attempts arm the guard.  ``_had_timeout``
        still scans every attempt and still matches the bare substring
        ``"timeout"``, so a cloud-provider timeout or a judge's free-prose remark
        can still arm it.  Narrowing that needs the timed-out attempt to carry
        its backend identity (#159), which is #221/#227's dependency, not this
        one's — and #227 warns that fixing half of that pair makes behaviour
        worse.  This change only stops the probe asking the wrong machine; it
        does not change what the probe can detect.
        """
        probe = getattr(self, "backend_probe", None)
        if probe is not None:
            return bool(probe())
        # Module-level names on purpose: existing tests patch
        # ``socr.pipeline.orchestrator.probe_ollama_idle``, and routing around
        # that name would silently neuter every one of those patches.
        if self._local_backend_is_openai_compatible():
            return probe_openai_server_idle(self.config.qwen_vllm_url)
        return probe_ollama_idle(self._local_backend_host())

    def _local_backend_is_openai_compatible(self) -> bool:
        """True when this run's local VLM is served over an OpenAI-compatible API.

        Two ways to be one, matching the gate ``QwenEngine.is_available``
        (``engines/qwen.py``) already uses to decide the qwen rung is usable:

        1. ``qwen_backend`` names an OpenAI-compatible backend explicitly.
        2. ``qwen_backend`` is ``"auto"`` AND ``VLLM_BASE_URL`` is set.

        Case 2 is not a corner case, it is the HPC deployment: ``is_available``
        short-circuits True on ``VLLM_BASE_URL`` alone ("vLLM path (e.g. HPC,
        where Ollama is forbidden on server GPUs)"), and ``PipelineConfig``
        adopts that variable into ``qwen_vllm_url`` in ``__post_init__`` while
        leaving ``qwen_backend`` at its ``"auto"`` default. So a user reaches
        "auto backend, remote vLLM server" by exporting ONE environment
        variable and never touching a flag — and treating that as Ollama is
        exactly #222's named failure: a healthy vLLM node reported dead, one
        timeout truncating a ``--strict-local`` run.

        An EXPLICIT ``qwen_backend`` always wins, in both directions:
        ``"ollama"`` stays Ollama even with ``VLLM_BASE_URL`` exported, because
        a value the user typed outranks one the environment happens to carry.
        """
        from socr.core.providers import qwen_auto_resolves_to_openai
        from socr.tables.extract import OPENAI_COMPATIBLE_BACKENDS

        backend = getattr(self.config, "qwen_backend", "")
        if backend in OPENAI_COMPATIBLE_BACKENDS:
            return True
        # GH-370 (cubic P2): case 2 lives in ONE place. Provenance recording
        # asks the same question, and a second copy of this rule is exactly the
        # execution/recording drift this ticket exists to remove.
        return qwen_auto_resolves_to_openai(self.config)

    def _local_backend_host(self) -> str:
        """Where this run's local Ollama daemon actually listens (GH-222)."""
        from socr.tables.extract import resolve_ollama_host

        return resolve_ollama_host(getattr(self.config, "ollama_host", None))

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

        # Read the sidecar BEFORE metering: the page's total spend lives there,
        # and it is not the winning output's own cost whenever the ladder paid
        # for a rung it rejected (round 4).
        meta: dict = {}
        try:
            from ocr_output_contract import doc_dir_for, relative_key

            scan_root = self._scan_root or state.handle.path.parent
            doc_dir = doc_dir_for(output_dir, relative_key(state.handle.path, scan_root))
            sidecar_path = doc_dir / "pages" / f"{page_num:05d}.json"
            meta = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.debug("PP-5: sidecar unreadable for p%d (%s)", page_num, exc)

        # Fold the resumed page's cost into total_cost (budget continuity).  An
        # EngineResult mirrors what the live route_page path appends per page.
        # A sidecar written before ``page_cost_usd`` existed falls back to the
        # winner's cost, which is what those runs recorded.
        # Restored VERBATIM, never recomputed: the attempt list this resume just
        # rebuilt holds only the frozen winner, so a recomputation here would
        # discard every rung the original run paid for and refused -- and the
        # assembly re-flush would then write that loss back to disk (round 5).
        # A sidecar written before the field existed falls back to the winner's
        # cost, which is what those runs recorded, and the fallback becomes the
        # page's fact so the next resume is stable.
        page_cost = meta["page_cost_usd"] if "page_cost_usd" in meta else page_out.cost_usd
        ps.page_cost_usd = page_cost
        state.record_engine_run(
            EngineResult(
                document_path=state.handle.path,
                engine=page_out.engine or "resumed",
                status=DocumentStatus.SUCCESS,
                cost=page_cost,
                processing_time=0.0,
            ),
            # Already recorded: the line above restored the persisted fact, and
            # charging it again would double the page's spend on every resume.
            page_nums=(),
        )
        try:
            ps.needs_ocr_enhancement = bool(meta.get("needs_ocr_enhancement", False))
            ps.native_table_structure_failed = bool(
                meta.get("native_table_structure_failed", False)
            )
            # GH-151 TICKET-B1: restore the grid-shape defect flag so it
            # survives resume instead of evaporating on a resumed run.
            ps.native_table_structure_defective = bool(
                meta.get("native_table_structure_defective", False)
            )
            ps.native_table_emission_defect = str(
                meta.get("native_table_emission_defect", "") or ""
            )
            # GH-346: restored alongside emission, so the resumed PageState
            # carries the same content provenance the sidecar recorded.
            ps.native_table_content_defect = str(meta.get("native_table_content_defect", "") or "")
            # GH-200: restore the header-attribution defect flag so it
            # survives resume instead of evaporating on a resumed run.
            ps.native_table_header_unattributed = bool(
                meta.get("native_table_header_unattributed", False)
            )
            # TR-3: restore per-region verifier flag and D3 PNG ref.
            ps.native_table_unverifiable = bool(meta.get("native_table_unverifiable", False))
            # GH-371: restore failed-region identity for the regional splice.
            # Validate the complete list: one malformed entry invalidates the
            # identity rather than allowing a partial list to select a table.
            raw_ords = meta.get("native_table_unverifiable_ordinals")
            if isinstance(raw_ords, list) and all(
                type(ordinal) is int and ordinal >= 0 for ordinal in raw_ords
            ):
                ps.native_table_unverifiable_ordinals = list(raw_ords)
            else:
                ps.native_table_unverifiable_ordinals = []
            # Validate the expected region count independently; old sidecars
            # have no key and therefore restore the safe zero default.
            raw_count = meta.get("native_table_region_count")
            ps.native_table_region_count = (
                raw_count if type(raw_count) is int and raw_count >= 0 else 0
            )
            raw_idents = meta.get("native_table_region_identities")
            if isinstance(raw_idents, list) and all(type(item) is str for item in raw_idents):
                ps.native_table_region_identities = list(raw_idents)
            else:
                ps.native_table_region_identities = []
            ps.d3_floor_png_ref = str(meta.get("d3_floor_png_ref", ""))
            # P6 cold review round 2: carry run 1's published disposition forward
            # so a resumed re-flush reproduces the sidecar byte for byte. A
            # sidecar written before the field existed restores None and the
            # disposition is recomputed, exactly as it is on a fresh run.
            _restored_disposition = meta.get("disposition")
            ps.resumed_disposition = (
                dict(_restored_disposition) if isinstance(_restored_disposition, dict) else None
            )
            # GH-90: restore scanned-table evidence failure so the scanned floor applies on resume.
            ps.scanned_table_evidence_failed = bool(
                meta.get("scanned_table_evidence_failed", False)
            )
            # #263: restore the shredded-page image ref too, so a resumed run's
            # floor ships marker + image exactly as the first run did instead of
            # silently degrading to a bare marker.
            ps.rotated_shred_png_ref = str(meta.get("rotated_shred_png_ref", ""))
            ps.chart_asset_render_failed = bool(meta.get("chart_asset_render_failed", False))
            # GH-318: OR, never assign. OCR pages can resume BEFORE chart
            # eligibility runs, while native pages detect first and only then
            # restore their sidecar — a plain assignment would let an older
            # clean sidecar erase a failure this run has already recorded.
            ps.chart_asset_detection_failed = bool(
                getattr(ps, "chart_asset_detection_failed", False)
            ) or bool(meta.get("chart_asset_detection_failed", False))
            # P4-R: carry the unread-equation latch forward, so a page resumed
            # while STILL offline keeps saying so and is re-read on the first
            # run that has a provider.
            ps.equation_lane_retry_pending = bool(meta.get("equation_lane_retry_pending", False))
            # P1: restore table retry latch (sparse key defaults to False when omitted).
            ps.table_judge_retry_pending = bool(meta.get("table_judge_retry_pending", False))
            restored_rungs = meta.get("table_judge_retry_rungs")
            ps.table_judge_retry_rungs = (
                [str(k) for k in restored_rungs] if isinstance(restored_rungs, list) else []
            )
            ps.judge_rejected = bool(meta.get("judge_rejected", False))
            # MAJOR 7(b): restore the S1 resume-idempotency flag so
            # ``_reaches_structure_class_branch`` can re-derive the bucket
            # membership a resumed run's collapsed ``attempts`` list can no
            # longer prove on its own.
            ps.structure_class_model_kept_on_resume = bool(
                meta.get("structure_class_model_kept", False)
            )
            # GH-353 TICKET-D1a: restore the table judge ladder's durable
            # disposition AND replay its per-table audit events into
            # ``state.events``. Without this, a resumed page's REJECTED /
            # UNVERIFIED verdict would evaporate from ``ps`` (silently
            # reverting the page to un-demoted at C3's manifest guard) and
            # its ``table_ladder_*`` events would be missing from
            # ``state.events`` (silently dropping the page from
            # ``tables_trust.json`` and the assemble-time metadata note,
            # both of which derive from ``state.events`` -- see
            # ``_tables_trust_note`` / ``build_tables_trust``). Restoring the
            # events, never re-judging: no rung is invoked here.
            disposition_raw = meta.get("table_ladder_disposition")
            ps.table_ladder_disposition = FailureMode(disposition_raw) if disposition_raw else None
            ps.table_ladder_incomplete = bool(meta.get("table_ladder_incomplete"))
            self._apply_binding_adjudication_meta(state, page_num, meta)
            from socr.core.audit_log import AuditEvent

            # P4-R joins the allowlist for the same reason D1a wrote it: the
            # lane's dispositions -- above all a presence-guard REJECTION --
            # are the only record that a reading was looked at and refused.
            # Dropping them on resume would leave a page that silently ships
            # native prose with no trace of the refusal.
            restore_kinds = self.resume_restore_kinds()
            for ev in meta.get("audit_events", []) or []:
                if not isinstance(ev, dict) or ev.get("kind") not in restore_kinds:
                    continue
                state.events.append(
                    AuditEvent(
                        page_num=page_num,
                        kind=str(ev.get("kind", "")),
                        engine=str(ev.get("engine", "") or ""),
                        detail=str(ev.get("detail", "") or ""),
                        data=dict(ev.get("data") or {}),
                    )
                )
        except Exception as exc:
            logger.debug("PP-5 flag restore failed for p%d (%s); body text kept", page_num, exc)

    def _rewrite_all_fragments(
        self,
        state: DocumentState,
        output_dir: Path,
        final_text: str,
        records: list[FinalizedPageRecord] | None = None,
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

        if records is not None:
            page_bodies = [rec.output.text for rec in records]
        else:
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
    # Assemble
    # ------------------------------------------------------------------

    def _canonical_body(
        self, state: DocumentState, page_texts: list[str] | None = None
    ) -> tuple[str, bool]:
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

        texts = page_texts if page_texts is not None else canonical_page_texts(state)
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

    def _backfill_missing_table_ladder_terminals(
        self, state: DocumentState, page_texts: list[str]
    ) -> None:
        """Fail closed when an emitted table reaches assemble unwitnessed.

        Completeness, not content. The agentic helper produces ACCEPTED /
        REJECTED / UNVERIFIED for tables it saw. This sweep does not re-judge:
        any markdown table in the shipped page text that has no ladder terminal
        event is TABLE_UNVERIFIED. A missing event is an infra miss (the
        helper returned early, a resume skipped the gate, chart_asset skipped,
        or the agentic ``if`` never ran), not a content FAIL.

        Runs before document status, metadata, and the CLI summary are built
        so an unwitnessed table cannot restamp SUCCESS.
        """
        if not self.config.table_judge_ladder:
            return

        from socr.core.audit_log import AuditEvent
        from socr.core.manifest import is_page_failed_marker
        from socr.judge.table_verdict import (
            TABLE_LADDER_ACCEPTED_KIND,
            TABLE_LADDER_REJECTED_KIND,
            TABLE_LADDER_UNVERIFIED_KIND,
        )
        from socr.tables.reconcile import find_table_blocks

        terminal_kinds = {
            TABLE_LADDER_ACCEPTED_KIND,
            TABLE_LADDER_REJECTED_KIND,
            TABLE_LADDER_UNVERIFIED_KIND,
        }
        observed: set[tuple[int, str]] = set()
        for event in state.events:
            if event.kind not in terminal_kinds:
                continue
            table_id = str((event.data or {}).get("table_id", "") or "")
            if table_id:
                observed.add((event.page_num, table_id))

        for page_num, page_text in enumerate(page_texts, start=1):
            if not page_text or is_page_failed_marker(page_text):
                continue
            page_state = state.pages.get(page_num)
            if page_state is None:
                continue
            for table_index, _block in enumerate(find_table_blocks(page_text)):
                table_id = f"p{page_num}-t{table_index}"
                if (page_num, table_id) in observed:
                    continue
                # GH-359 (cubic P1): record the completeness miss separately
                # from the page-level disposition. A page holding one REJECTED
                # table keeps that content verdict, but this flag withholds
                # D1b's resume skip so the unwitnessed table is looked at.
                page_state.table_ladder_incomplete = True
                if page_state.table_ladder_disposition is not FailureMode.TABLE_REJECTED:
                    page_state.table_ladder_disposition = FailureMode.TABLE_UNVERIFIED
                state.events.append(
                    AuditEvent(
                        page_num=page_num,
                        kind=TABLE_LADDER_UNVERIFIED_KIND,
                        engine=(page_state.best_output.engine if page_state.best_output else ""),
                        detail=(
                            f"table {table_id} reached assemble with no table-judge "
                            "terminal (infra problem, retryable on resume)"
                        ),
                        data={"table_id": table_id, "rung_trail": []},
                    )
                )

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
            console.print("\n[cyan]Assemble:[/cyan]")

        from socr.core.manifest import (
            canonical_page_texts,
            d3_floor_kept_model_output,
            finalized_page_records,
            is_page_failed_marker,
            kept_table_grid_defect,
            structure_class_grid_winner,
        )

        pre_records = finalized_page_records(state)
        page_texts = canonical_page_texts(state, records=pre_records)
        final_text, has_text = self._canonical_body(state, page_texts=page_texts)

        self._backfill_missing_table_ladder_terminals(state, page_texts)

        # PP-1: flush per-page fragments from the in-memory page texts, then
        # verify stitch round-trips byte-identically. Non-fatal: any error
        # keeps the in-memory body.
        #
        # Fragment write + stitch check MUST run here, before the
        # ``strip_phantom_images`` / ``_guard_fabricated_image_refs_document``
        # mutations below (and before the S1 bucket/event derivation further
        # down) -- ``final_text`` at this point is still the pre-strip value
        # ``_stitch_fragments`` (built from these same pre-strip ``page_texts``)
        # is meant to match. A prior version of this fix (MAJOR 6(a) on #269)
        # moved this ENTIRE block, sidecar flush included, down past the strip
        # mutations to fix sidecar event delivery -- which also silently moved
        # this comparison past the point where ``final_text`` is mutated,
        # tripping the byte-identity guard on every document where the strip
        # actually removes a phantom image ref (output stayed correct; the
        # self-check did not). Splitting the two flushes apart fixes both:
        # this loop stays early, only the sidecar flush (below, near the S1
        # event derivation) needs to run late.
        if has_text and page_texts:
            try:
                page_nums = list(range(1, state.handle.page_count + 1))
                for pnum, ptext in zip(page_nums, page_texts):
                    self._flush_page_fragment(state, pnum, ptext, output_dir)
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

        # P6: derive the six selection-shaped buckets from the pre-computed records in one pass.
        disposition_buckets = _derive_disposition_buckets(state, pre_records)
        d3_model_table_pages = sorted(disposition_buckets["d3_model_table_pages"])
        d3_floor_pages = sorted(disposition_buckets["d3_floor_pages"])
        flagged_model_pages = sorted(disposition_buckets["flagged_model_pages"])
        structure_class_model_pages = sorted(disposition_buckets["structure_class_model_pages"])
        structure_class_floor_pages = sorted(disposition_buckets["structure_class_floor_pages"])
        corrupt_math_hybrid_pages = sorted(disposition_buckets["corrupt_math_hybrid_pages"])

        # The six orthogonal bucket groups (native-only distrust, value drift,
        # fabrication, text-grid rejection, chart-detection failure, and
        # table-ladder rejected/unverified) are alerts or post-selection events,
        # not PageDisposition, so they remain flag/config/event/terminal-derived.
        # Keep the pipeline config on this assemble-only state context so the
        # one-argument helper can remain a pure state observation seam.
        state._assemble_config = self.config
        orthogonal_buckets = _derive_orthogonal_assemble_buckets(state)
        native_only_distrust_pages = orthogonal_buckets["native_only_distrust_pages"]
        value_drift_pages = orthogonal_buckets["value_drift_pages"]
        fabricated_ref_pages = orthogonal_buckets["fabricated_ref_pages"]
        text_grid_rejected_pages = orthogonal_buckets["text_grid_rejected_pages"]
        chart_detection_failed_pages = orthogonal_buckets["chart_detection_failed_pages"]
        table_rejected_pages = orthogonal_buckets["table_rejected_pages"]
        table_unverified_pages = orthogonal_buckets["table_unverified_pages"]

        def _kept_defect(page_num: int) -> str:
            # ``best_output``, not the finalized record (cold review round 2,
            # finding 5). The flagged-model event describes the CANDIDATE that was
            # kept; reading the record instead reads it after the emission guard has
            # replaced the body with a marker, so a defect the event exists to name
            # -- a live LaTeX leak in the kept grid -- silently became "".
            ps_ = state.pages.get(page_num)
            bo_ = ps_.best_output if ps_ else None
            return kept_table_grid_defect(bo_.text) if bo_ and bo_.text else ""

        # Contract: native_fallback_pages means OCR was tried for a
        # non-S1/non-D3/non-flagged-model reason and native bytes ultimately shipped
        # demoted; it is intentionally not equivalent to the DEMOTED_NATIVE ending
        # or the old NATIVE_FALLBACK provenance.
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
            # short-circuit in page analysis never sets (it returns
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
            # #259: the model's table ships on these; native did not.
            and n not in flagged_model_pages
            # S1: the model's grid ships on these; native did not.
            and n not in structure_class_model_pages
            # S1 case (iii): this page has its OWN bucket
            # (``structure_class_floor_pages``) and its own event
            # kind/CLI line. BLOCKING 2 on #269: this list previously OR'd
            # ``p.is_structure_class()`` straight into its attempts-gate below
            # AND its include-clause above, so a structure-class page could
            # satisfy BOTH this generic bucket and its dedicated S1 bucket at
            # once (two events, two CLI lines for one page) -- or, with zero
            # attempts (a cascade-halt page), land in this list even though
            # ``_winning_page_output`` never demoted it and shipped SUCCESS.
            # Excluding the dedicated bucket here and reverting the
            # attempts-gate below to plain ``p.attempts`` restores this list
            # to "OCR was tried [for a REASON OTHER THAN S1] and never
            # passed" -- S1's own bucket owns S1's own pages exclusively.
            and n not in structure_class_floor_pages
            # GH-271: the region hybrid ships, so this is not a native fallback.
            and n not in corrupt_math_hybrid_pages
            # GH-293: a page that ships a FAILURE MARKER did not fall back to
            # native text -- no native text ships at all. The whole meaning of
            # this list is "OCR was tried and never passed, so the page shipped
            # its native body demoted"; a marker page shipped no body.
            #
            # The concrete case is GH-263's shredded lane, which takes the
            # ROTATED_TEXT_SHREDDED ending and never got an exclusion when it
            # was added. Verified: such a page is in this list AND in
            # `failed_pages`, so it emits two audit events and two CLI lines for
            # one page, and the native_fallback line asserts something false.
            #
            # Excluding `failed_pages` rather than naming the shredded ending
            # closes the CLASS, not the instance. `failed_pages` is derived from
            # the SHIPPED TEXT (`is_page_failed_marker`), so this holds for any
            # future ending that ships a marker without remembering to add an
            # exclusion here -- which is exactly how #293 happened. It also
            # makes "no page is in both lists" true by construction rather than
            # by six predicates staying in lockstep.
            and n not in failed_pages
            and p.attempts
            and not (p.best_output and p.best_output.audit_passed)
        ]

        # A reconstructed or historical state may contain a whole-document
        # attempt (page_num=0) without per-page winners. Treat a passing one as
        # covering the document for status calculation.
        has_passing_whole_doc = any(w.audit_passed for w in state.whole_doc_attempts)
        # GH-225: a page the model invented content on must not leave the run
        # reporting a clean SUCCESS.  Because the fabrication demotion keeps the
        # cleaned page as the winner (see ``_guard_fabricated_image_refs``), it
        # deliberately does NOT flip ``best_output.audit_passed``, so it cannot
        # reach the document through ``pages_needing_repair`` the way the other
        # defects do.  It gets its own term, exactly like the three lists above.
        # AUDIT_FAILED rather than ERROR: no real content was lost — the page
        # ships correct text — so the CLI's "completed with warnings, output
        # written" path is the honest one, not a hard failure.
        # Strip phantom image references, and sweep for fabricated ones, BEFORE
        # the status calculation below (#252 review, blocking).  This block used
        # to sit after the status was already frozen, so a document
        # whose ONLY defect was a fabricated link had the ref removed and an
        # error note appended while still finishing SUCCESS / audit_passed=True —
        # the document-status surface the issue requires, silently absent on
        # exactly the lanes that have no per-page seam.  Running it here also
        # means the saved body and the reported status are computed from the same
        # text, which they previously were not.
        from ocr_output_contract import doc_dir_for, relative_key

        normalizer = OutputNormalizer()
        scan_root = self._scan_root or state.handle.path.parent
        doc_dir = doc_dir_for(output_dir, relative_key(state.handle.path, scan_root))
        if has_text:
            final_text = normalizer.strip_phantom_images(final_text, output_dir=doc_dir)
            # The per-page seam already cleans refs emitted during routing; this
            # assembly-time pass is idempotent and also covers any refs introduced
            # by later document-level transformations.
            final_text = self._guard_fabricated_image_refs_document(state, final_text, doc_dir)

        # The document-level sweep has no PageState to increment, so its removals
        # ride on the page-0 document event it records.  Without this term the
        # sweep could redact a fabricated ref and still leave the run SUCCESS.
        doc_fabrication = any(
            getattr(e, "kind", "") == "fabricated_image_ref" and getattr(e, "page_num", None) == 0
            for e in state.events
        )
        pages_ok = not state.pages_needing_repair or has_passing_whole_doc
        pages_ok = pages_ok and not failed_pages and not native_fallback_pages
        pages_ok = pages_ok and not native_only_distrust_pages
        # #259: the kept model page carries a table flag, so the document
        # cannot report a clean SUCCESS. AUDIT_FAILED, not ERROR: the page
        # ships the better of the two readings, nothing was lost.
        pages_ok = pages_ok and not flagged_model_pages
        # S1: the general structure-class case (C2). Same reasoning as #259 --
        # the page ships the kept model reading over native, so it is not a
        # page failure, but the document cannot report a clean SUCCESS either.
        pages_ok = pages_ok and not structure_class_model_pages
        # S1/P2 case (iii): the fail-closed floor ships a whole-page marker plus
        # image and withholds every native byte; it is never SUCCESS because no
        # usable grid candidate survived selection.
        pages_ok = pages_ok and not structure_class_floor_pages
        # #262: the model's reading superseded a HARD fail-closed floor. That is
        # strictly more alarming than #259's flag, and it must not leave the run
        # reporting a clean SUCCESS. AUDIT_FAILED rather than ERROR: the page
        # ships real, scored content -- the marker it replaced had none.
        pages_ok = pages_ok and not d3_model_table_pages
        # GH-271: syntax-valid crop transcription is still mathematically
        # unverified, so the document must not report a clean success.
        pages_ok = pages_ok and not corrupt_math_hybrid_pages
        # NOT a page failure -- the owner was explicit that the page is not failed
        # and the table is kept. AUDIT_FAILED at the document level is the
        # "completed with warnings, output written" path, which is the honest
        # one: content shipped, and a named value on it is disputed.
        pages_ok = pages_ok and not value_drift_pages
        pages_ok = pages_ok and not fabricated_ref_pages and not doc_fabrication
        pages_ok = pages_ok and not text_grid_rejected_pages
        # GH-318: chart eligibility raised and the page took the non-chart route
        # without the pipeline ever learning whether it was a chart. The text it
        # shipped may be right, but the routing decision behind it was never made,
        # so the run must not report a clean SUCCESS. AUDIT_FAILED, not ERROR: the
        # fail-soft route is deliberate (#297) and the content is kept, so this is
        # the "completed with warnings, output written" path.
        pages_ok = pages_ok and not chart_detection_failed_pages

        # GH-353: table judge ladder terminals (C2). Keyed off
        # ``PageState.table_ladder_disposition`` FIRST -- the durable field
        # C3 built for exactly this consumer -- with ``best_output.failure_mode``
        # as a belt-and-braces fallback (see ``_table_ladder_terminal`` above
        # for why ``best_output`` alone is blind to the disposition in
        # production: the manifest guard demotes only the finalized copy it
        # returns, never ``best_output`` itself). NOT keyed off
        # ``best_output.status`` either way -- a page can legitimately arrive
        # as SUCCESS/audit_passed=False (GH-161, ``:4347``), and neither the
        # disposition guard nor a direct failure_mode stamp ever rewrites
        # ``.status`` to match. TABLE_REJECTED (content problem, not
        # retryable) and TABLE_UNVERIFIED (infra problem, retryable on
        # resume) are kept as separate, mutually exclusive buckets -- same
        # reasoning as every other pairing above: one bucket per disposition,
        # because they need distinct audit kinds, CLI wording and (D1b,
        # later) distinct resume policy.
        pages_ok = pages_ok and not table_rejected_pages and not table_unverified_pages

        if has_text and pages_ok:
            status = DocumentStatus.SUCCESS
        elif has_text:
            status = DocumentStatus.AUDIT_FAILED
        else:
            status = DocumentStatus.ERROR

        state.status = status

        if (
            failed_pages
            or native_fallback_pages
            or d3_floor_pages
            or native_only_distrust_pages
            or flagged_model_pages
            or structure_class_model_pages
            or structure_class_floor_pages
            or d3_model_table_pages
            or corrupt_math_hybrid_pages
            or value_drift_pages
            or table_rejected_pages
            or table_unverified_pages
        ):
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
            for n in corrupt_math_hybrid_pages:
                hybrid = state.pages[n].corrupt_math_hybrid
                recovery_event = next(
                    (
                        event
                        for event in reversed(state.events)
                        if event.page_num == n and event.kind == "corrupt_math_region_recovery"
                    ),
                    None,
                )
                crop_paths = (
                    list(recovery_event.data.get("crop_paths", [])) if recovery_event else []
                )
                state.events.append(
                    AuditEvent(
                        page_num=n,
                        kind="corrupt_math_hybrid_shipped",
                        engine="native+math",
                        detail=(
                            "native prose plus crop-backed equation candidate(s) shipped "
                            "WARNING; LaTeX passed at most a syntax gate and mathematical "
                            "fidelity remains unverified"
                        ),
                        data={
                            "provider_id": hybrid.provider_id if hybrid else "",
                            "provider_model": hybrid.provider_model if hybrid else "",
                            "provider_backend": hybrid.provider_backend if hybrid else "",
                            "cost_usd": hybrid.cost_usd if hybrid else None,
                            "crop_paths": crop_paths,
                            "audit_passed": False,
                        },
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
            # #259: the model's flagged table was KEPT. Distinct from
            # native_fallback (nothing fell back) and from the D3 floor (nothing
            # failed closed): the page ships model content under a flag.
            for n in flagged_model_pages:
                state.events.append(
                    AuditEvent(
                        page_num=n,
                        kind="flagged_model_table_kept",
                        engine=(
                            state.pages[n].best_output.engine if state.pages[n].best_output else ""
                        ),
                        detail=(
                            "no OCR rung was accepted, but the model produced a table and"
                            " the native layer carries a table-distrust flag; the model's"
                            " reading ships FLAGGED rather than being replaced by native"
                            + (
                                f"; kept grid is structurally defective ({_kept_defect(n)})"
                                if _kept_defect(n)
                                else ""
                            )
                        ),
                        # #259 round 3: a ragged kept grid is a thing to FLAG, not a
                        # reason to fall back to an ungridded native — so the
                        # structural predicate's answer is recorded here rather than
                        # gating the keep.
                        data={
                            "flagged_model_kept": True,
                            "grid_defect": _kept_defect(n),
                        },
                    )
                )
            # #262: the D3 floor was SUPERSEDED on these pages -- distinct from
            # ``table_region_unverifiable`` (the floor fired and a marker
            # shipped) and from ``flagged_model_table_kept`` (#259: no floor was
            # involved, only a distrust flag). A consumer of the audit log must
            # be able to tell "we shipped nothing" from "we shipped the model
            # over a hard fail", so it gets its own kind rather than reusing one.
            for n in d3_model_table_pages:
                # The engine named here is the CANDIDATE d3_floor_kept_model_output
                # chose, not the record that shipped. On a page whose passing
                # best_output wins selection outright the two differ, and the
                # audit log must name the reading the D3 supersession is about.
                _kept_d3 = d3_floor_kept_model_output(state.pages[n])
                state.events.append(
                    AuditEvent(
                        page_num=n,
                        kind="d3_floor_model_table_kept",
                        engine=(_kept_d3.engine if _kept_d3 else ""),
                        detail=(
                            "the native table region failed geometry/header verification"
                            " and no OCR rung was accepted, but a model attempt authored a"
                            " grid; the model's reading ships FLAGGED instead of the"
                            " failed-table marker — verify it against the source image"
                            " before citing"
                        ),
                        data={"d3_floor_superseded": True},
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
            # S1: the general structure-class case (C2). Distinct from #259's
            # ``flagged_model_table_kept`` (which only fires once an OLD
            # native-distrust flag is set) and from #262's D3 supersession
            # (which requires the TR-3 hard-fail conjunction): this fires on
            # ANY structure-class page whose native branch may not author the
            # grid, flag or no flag -- the 2026-08-20 measurement's actual bug.
            for n in structure_class_model_pages:
                # Same reason as the D3 event above: name the grid candidate the
                # structure-class branch selected, not whatever finally shipped.
                _kept_sc = structure_class_grid_winner(state.pages[n])
                state.events.append(
                    AuditEvent(
                        page_num=n,
                        kind="structure_class_model_table_kept",
                        engine=(_kept_sc.engine if _kept_sc else ""),
                        detail=(
                            "structure-class page (table); native may not"
                            " author the grid (C1), and a model attempt did -- the"
                            " model's reading ships instead of native, flagged per its"
                            " own status"
                        ),
                        data={"structure_class_model_kept": True},
                    )
                )
            # S1/P2 case (iii): every usable grid candidate was refused or
            # absent, so the fail-closed floor ships a marker plus page image
            # and withholds every native byte. Floor pages also remain in
            # ``failed_pages`` (a whole-page floor produced no usable output);
            # this event is the floor-specific surface on top of that.
            for n in structure_class_floor_pages:
                state.events.append(
                    AuditEvent(
                        page_num=n,
                        kind="structure_class_ladder_exhausted_floor",
                        engine="native",
                        detail=(
                            "every usable grid candidate was refused/absent; marker plus "
                            "page image was selected, and the native geometry grid was "
                            "withheld (fail-closed floor)"
                        ),
                        data={"structure_class_floor": True},
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
                if corrupt_math_hybrid_pages:
                    console.print(
                        f"  [yellow]{len(corrupt_math_hybrid_pages)} page(s) shipped "
                        "crop-backed equation candidate(s); mathematical fidelity "
                        f"remains unverified: {corrupt_math_hybrid_pages}[/yellow]"
                    )
                if native_only_distrust_pages:
                    console.print(
                        f"  [yellow]{len(native_only_distrust_pages)} page(s) shipped native "
                        f"text with an unverifiable table region (--native-only: OCR not "
                        f"attempted): {native_only_distrust_pages}[/yellow]"
                    )
                if flagged_model_pages:
                    console.print(
                        f"  [yellow]{len(flagged_model_pages)} table page(s) shipped the "
                        f"model's FLAGGED reading (no rung accepted; native table "
                        f"distrusted): {flagged_model_pages}[/yellow]"
                    )
                if structure_class_model_pages:
                    console.print(
                        f"  [yellow]{len(structure_class_model_pages)} structure-class page(s) "
                        f"shipped the model's grid reading over native (native may not author "
                        f"a grid): {structure_class_model_pages}[/yellow]"
                    )
                if structure_class_floor_pages:
                    console.print(
                        f"  [red]{len(structure_class_floor_pages)} structure-class page(s) hit "
                        f"the fail-closed floor (usable grid candidates refused/absent; marker "
                        f"plus page image selected; native geometry grid withheld): "
                        f"{structure_class_floor_pages}[/red]"
                    )
                if value_drift_pages:
                    from socr.tables.native_verifier import describe_drift

                    console.print(
                        f"  [red]{len(value_drift_pages)} kept table(s) carry a DISPUTED "
                        f"value (detected, not adjudicated): {value_drift_pages}[/red]"
                    )
                    for _e in state.events:
                        if getattr(_e, "kind", "") != "table_value_drift_unadjudicated":
                            continue
                        _rows = (getattr(_e, "data", None) or {}).get("drifted_rows") or []
                        if _rows:
                            console.print(f"    [red]p{_e.page_num}: {describe_drift(_rows)}[/red]")
                if d3_model_table_pages:
                    console.print(
                        f"  [red]{len(d3_model_table_pages)} table page(s) shipped a MODEL"
                        f" reading over a failed-closed native table region (verify against"
                        f" the source image before citing): {d3_model_table_pages}[/red]"
                    )
                if d3_floor_pages:
                    console.print(
                        f"  [red]{len(d3_floor_pages)} table page(s) hit the D3 fail-closed"
                        f" floor (unverifiable region → explicit failure marker): "
                        f"{d3_floor_pages}[/red]"
                    )
                if table_rejected_pages:
                    console.print(
                        f"  [red]{len(table_rejected_pages)} table page(s) failed the judge "
                        f"ladder — TABLE_REJECTED (models looked and said no; not retryable): "
                        f"{table_rejected_pages}[/red]"
                    )
                _visual_kept, _visual_lost = self._visual_values_split(state)
                if _visual_kept:
                    # GH-519: the debt gets a line of its own. It is not a
                    # failure -- the lane did the right thing -- so it is
                    # stated, not coloured as an error.
                    console.print(
                        f"  [cyan]{len(_visual_kept)} chart-asset page(s): visual values not "
                        f"transcribed; in-image text is preserved in the page image only: "
                        f"{_visual_kept}[/cyan]"
                    )
                if _visual_lost:
                    # GH-566: and the render-failure pages are NOT that. Nothing
                    # holds their figure, so this line is a loss, not a note.
                    console.print(
                        f"  [red]{len(_visual_lost)} chart-asset page(s): visual values not "
                        f"transcribed AND the page image was not saved; in-image text is "
                        f"preserved nowhere: {_visual_lost}[/red]"
                    )
                if table_unverified_pages:
                    # GH-560: same split as the document note -- an operator
                    # told "retryable on resume" will re-run, and for a
                    # no-witness terminal that run changes nothing.
                    _retryable, _unwitnessed = self._unverified_wording_split(
                        state, table_unverified_pages
                    )
                    if _retryable:
                        console.print(
                            f"  [yellow]{len(_retryable)} table page(s) could not be "
                            f"judged — TABLE_UNVERIFIED (ladder exhausted without an answer; "
                            f"retryable on resume): {_retryable}[/yellow]"
                        )
                    if _unwitnessed:
                        console.print(
                            f"  [yellow]{len(_unwitnessed)} table page(s) were not judged — "
                            f"TABLE_UNVERIFIED (no table witness could be prepared, so no rung "
                            f"ran; not retryable): {_unwitnessed}[/yellow]"
                        )

        # MAJOR 6(a) on #269: only the SIDECAR flush belongs here, after the
        # bucket derivation and audit-event-append block above.
        # `_flush_page_sidecar` filters `state.events` by page_num at the
        # MOMENT it writes each sidecar -- so flushing before the S1 events
        # (`structure_class_model_table_kept` /
        # `structure_class_ladder_exhausted_floor`)
        # were appended meant those events never reached `pages/NNN.json`'s
        # `audit_events` field at all, and no later pass corrects a sidecar
        # (`_rewrite_all_fragments` only rewrites `.md` fragments).
        #
        # The fragment write + byte-identity stitch check stays at its
        # ORIGINAL position, right after `page_texts` is computed (above,
        # before `failed_pages`) -- an earlier version of this fix moved that
        # check down here too, past where `strip_phantom_images` /
        # `_guard_fabricated_image_refs_document` mutate `final_text`, which
        # tripped the check on every document where the strip actually
        # changed something (output stayed correct; only the self-check
        # false-alarmed). Sidecar delivery and the stitch guard have
        # different correctness requirements -- one wants the LATEST events,
        # the other wants the PRE-strip text -- so they no longer share a loop.
        # Non-fatal: any error here leaves whatever sidecars were already
        # written, in-memory body is unaffected either way.
        if page_texts:
            try:
                page_nums = list(range(1, state.handle.page_count + 1))
                for pnum in page_nums:
                    # GH-485: explicitly EMPTY, not omitted. This flush runs
                    # before the figure phase, so there is nothing to carry --
                    # but every terminal writer now names its figure source, so
                    # "no figures yet" and "forgot to pass them" cannot look
                    # alike. The post-figure re-flush below supplies the real
                    # ones.
                    self._flush_page_sidecar(state, pnum, output_dir, extra_figures=[])
            except Exception as exc:
                logger.warning(
                    "PP-1 [%s]: sidecar flush failed (%s)",
                    state.handle.path.name,
                    exc,
                )

        # Compute total processing time
        total_time = sum(r.processing_time for r in state.engine_runs)

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
        if corrupt_math_hybrid_pages:
            _math_note = "corrupt equation candidate unverified on page(s) " + ", ".join(
                str(n) for n in corrupt_math_hybrid_pages
            )
            if final_result.error:
                final_result.error = f"{final_result.error}; {_math_note}"
            else:
                final_result.error = _math_note

        # PP-2 cascade HALT: propagate the halt reason into the result error
        # so callers and tests can detect a partial-save due to a wedged backend.
        _pp2_halt = state.pp2_halt_reason
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
        # GH-225: surface fabricated image references at document level, for
        # the same reason GH-97 does above — a consumer reading metadata.json
        # must be able to see that a page shipped invented content.
        _fab_note = self._fabricated_url_note(state.events)
        if _fab_note:
            if final_result.error:
                final_result.error = f"{final_result.error}; {_fab_note}"
            else:
                final_result.error = _fab_note
        # GH-318: surface an unresolved chart-routing decision at document level,
        # for the same reason GH-225 does above — the audit event is durable but
        # a consumer reading metadata.json must not have to open audit_log.json
        # to learn that a page's routing was never decided.
        _chart_note = self._chart_detection_failed_note(state)
        if _chart_note:
            if final_result.error:
                final_result.error = f"{final_result.error}; {_chart_note}"
            else:
                final_result.error = _chart_note
        _trust_note = self._tables_trust_note(state)
        if _trust_note:
            if final_result.error:
                final_result.error = f"{final_result.error}; {_trust_note}"
            else:
                final_result.error = _trust_note
        # GH-353: surface the table judge ladder terminals at document level,
        # for the same no-silent-loss reason GH-318/GH-225 do above.
        _ladder_note = self._table_judge_ladder_note(state)
        if _ladder_note:
            if final_result.error:
                final_result.error = f"{final_result.error}; {_ladder_note}"
            else:
                final_result.error = _ladder_note
        _floor_note = self._structure_class_floor_note(state)
        if _floor_note:
            if final_result.error:
                final_result.error = f"{final_result.error}; {_floor_note}"
            else:
                final_result.error = _floor_note
        # GH-519: the chart lane's debt, at document level for the same
        # no-silent-loss reason as every note above it.
        _visual_note = self._visual_values_not_transcribed_note(state)
        if _visual_note:
            if final_result.error:
                final_result.error = f"{final_result.error}; {_visual_note}"
            else:
                final_result.error = _visual_note

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
        # GH-503: set by the figure phase's crash handler, and read by every
        # later metadata write in this method. A metadata write that forgets it
        # re-finalises the record and restores the skipped-forever bug.
        figure_phase_failed = False
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
                # GH-503: the record STAYS provisional. The pre-figures write a
                # few lines up says a crash here "leaves a retryable record
                # instead of a skipped-forever doc"; finalising it in this
                # handler overwrote exactly the suffix that promise depends on,
                # so an ordinary re-run of a document whose figure phase died
                # was SKIPPED and its provenance stayed empty forever under a
                # SUCCESS. Only --reprocess repaired it, and nothing told the
                # operator to reach for it.
                #
                # The cost is that a PERSISTENTLY failing figure phase
                # reprocesses on every run. That is what retryable means, and it
                # is the lesser evil against shipping empty provenance silently.
                figure_phase_failed = True
                logger.warning("figure phase failed (%s); keeping the un-embedded markdown", exc)
                if not self.config.quiet:
                    console.print(
                        f"  [yellow]Figure phase failed ({exc}); output saved "
                        "without figure descriptions. The record stays retryable: "
                        "a plain re-run will re-enter the figure phase[/yellow]"
                    )
                # Durable, page-independent record of the loss. The console line
                # scrolls away; the audit trail is what a later reader has.
                from socr.core.audit_log import AuditEvent as _FigureCrashEvent

                state.events.append(
                    _FigureCrashEvent(
                        page_num=0,
                        kind="figure_phase_failed",
                        engine="",
                        detail=f"{type(exc).__name__}: {exc}",
                        data={"record_left_provisional": True},
                    )
                )
                embedded_text = final_text
            if embedded_text != final_text:
                final_text = embedded_text
                final_result.pages[0].text = final_text
                self._save_markdown(state, final_text, output_dir)
            # Finalize the record (real fingerprint, final status), replacing
            # the provisional pre-figures entry -- unless the phase raised, in
            # which case there is nothing to finalize and the provisional entry
            # is the honest one (GH-503).
            self._write_metadata(
                state, final_result, output_dir, has_text, provisional=figure_phase_failed
            )

            # GH-171: and re-flush the sidecars, for the same reason the flush
            # was already moved below the audit-event append (see the MAJOR 6(a)
            # note above): a sidecar is written from a snapshot, and nothing
            # corrects it afterwards. Flushing before this phase left the
            # authoritative `pages/NNN.json` without the figure paths, bboxes
            # and caption engine that the final markdown and the manifest both
            # carry -- three records of one page disagreeing.
            #
            # Non-fatal, like the first flush: a failure here leaves the
            # pre-figure sidecars, which is what shipped before this fix.
            if has_text and page_texts:
                try:
                    for pnum in range(1, state.handle.page_count + 1):
                        self._flush_page_sidecar(
                            state,
                            pnum,
                            output_dir,
                            extra_figures=list(getattr(final_result, "figures", []) or []),
                        )
                except Exception as exc:
                    logger.warning(
                        "GH-171 [%s]: post-figure sidecar re-flush failed (%s)",
                        state.handle.path.name,
                        exc,
                    )

        # GH-226 review round: validate the EXACT post-transform body, not only
        # the pre-figure winner. Captions and other final transforms are model
        # output too; the saved Markdown, authoritative fragments, sidecars and
        # replay blobs must all freeze the same guarded page text and status.
        final_page_outputs: list[PageOutput] | None = None
        final_records: list[FinalizedPageRecord] | None = None
        if has_text:
            from ocr_output_contract import assemble_pages, split_native_pages

            from socr.core.manifest import finalized_page_records

            final_bodies = split_native_pages(final_text)
            if len(final_bodies) == state.handle.page_count:
                final_records = finalized_page_records(state, final_text)
                final_page_outputs = [record.output for record in final_records]
                emission_failures = [
                    page
                    for page in final_page_outputs
                    if page.failure_mode is FailureMode.TABLE_EMISSION_INVALID
                ]
                if emission_failures:
                    from socr.core.audit_log import AuditEvent

                    final_text = assemble_pages(
                        [page.text for page in final_page_outputs],
                        page_numbers=list(range(1, state.handle.page_count + 1)),
                    )
                    final_result.pages[0].text = final_text
                    final_result.status = DocumentStatus.AUDIT_FAILED
                    final_result.audit_passed = False
                    state.status = DocumentStatus.AUDIT_FAILED
                    failed_nums = [page.page_num for page in emission_failures]
                    note = "invalid final table emission on page(s) " + ", ".join(
                        str(page_num) for page_num in failed_nums
                    )
                    final_result.error = (
                        f"{final_result.error}; {note}" if final_result.error else note
                    )
                    existing = {
                        (event.page_num, (getattr(event, "data", None) or {}).get("site"))
                        for event in state.events
                        if getattr(event, "kind", "") == "table_structure_failed"
                    }
                    for page in emission_failures:
                        if (page.page_num, "final_body") in existing:
                            continue
                        defect = page.error.rsplit(": ", 1)[-1]
                        state.events.append(
                            AuditEvent(
                                page_num=page.page_num,
                                kind="table_structure_failed",
                                engine=page.engine or "",
                                detail=f"{defect} defect found in exact final page body",
                                data={"defect": defect, "site": "final_body"},
                            )
                        )
                    self._save_markdown(state, final_text, output_dir)
                    self._write_metadata(
                        state,
                        final_result,
                        output_dir,
                        has_text,
                        # GH-503: the LAST writer wins, so this one carries the
                        # flag too or the fix above is undone here.
                        provisional=figure_phase_failed,
                    )
            else:
                logger.warning(
                    "GH-226 final-body guard: split yielded %d page(s), expected %d; "
                    "leaving body unchanged for existing mismatch handling",
                    len(final_bodies),
                    state.handle.page_count,
                )

        # PP-4: single authoritative fragment rewrite from the FINAL text (post-
        # strip_phantom_images, post-inline-figures for figure docs, plain post-
        # strip for figure-free docs).  Runs unconditionally so every pages/NNN.md
        # matches the saved .md byte-for-byte regardless of whether the page has
        # figures or phantom image refs.  Supersedes the PP-1 pre-strip flush for
        # ALL pages.
        if has_text:
            self._rewrite_all_fragments(state, output_dir, final_text, records=final_records)

        # The earlier terminal sidecar flush intentionally precedes the figure
        # phase. Rewrite it once more from the exact final outputs so captions,
        # failure markers, statuses and the final-body audit event cannot diverge
        # from fragments/manifest/replay.
        #
        # GH-485: this LAST write must carry the figures too. It is the
        # authoritative one on the happy path, and without `extra_figures` it
        # scanned `state.engine_runs` only -- which the figure phase never
        # writes to -- so it silently undid GH-171's re-flush above and shipped
        # empty `figure_refs` again. Fixing one call site was not enough; every
        # writer of this file needs the same sources or the last one wins.
        if has_text and final_records is not None:
            self._final_records = {rec.output.page_num: rec for rec in final_records}
            _final_figures = list(getattr(final_result, "figures", []) or [])
            try:
                for rec in final_records:
                    self._flush_page_sidecar(
                        state,
                        rec.output.page_num,
                        output_dir,
                        extra_figures=_final_figures,
                        record=rec,
                    )
            finally:
                self._final_records = None

        # Reproducibility manifest (opt-in; default-on in agentic mode). Pass the
        # FINAL saved body so the manifest blobs (and thus replay) reproduce the
        # on-disk .md bit-for-bit, not the pre-transform state.
        if has_text and (self.config.write_manifest or self.config.agentic):
            self._write_manifest(
                state,
                output_dir,
                saved_body=final_text,
                records=final_records,
            )

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

            # P4-R (cold review round 3, finding 5): the root entry and its
            # pending-retry latch are ONE write, made by ``RootIndex.record``.
            #
            # Round 2 called ``record()`` -- which saves immediately -- and then
            # mutated the saved entry and saved again. Between those two saves
            # the index holds a RESUMABLE entry with NO latch (``partial``
            # counts: the gate accepts a partial entry whose checksum,
            # fingerprint and output all match), and an interruption or a
            # failure of the second save leaves it there permanently. The next
            # run's document gate then skips the whole file and the equation
            # page is never read -- the original finding-5 failure with a
            # smaller window.
            #
            # The latch rides in on ``to_entry`` instead, so ``record`` stays the
            # SOLE author of root metadata.json (the PP-5 invariant, pinned by
            # tests/test_pp5_resume_ledger.py) and there is still exactly one
            # save. The field is absent -- not false -- when nothing is pending,
            # so an older index without it reads identically.
            #
            # Fail-closed: if that save raises, NOTHING is recorded, the outer
            # handler logs it, and the next run reprocesses the document.
            pending: dict = {}
            if any(getattr(p, "equation_lane_retry_pending", False) for p in state.pages.values()):
                pending["equation_lane_retry_pending"] = True
            if any(getattr(p, "table_judge_retry_pending", False) for p in state.pages.values()):
                pending["table_judge_retry_pending"] = True
                # Cold review round 3, finding 1: the UNION of the rung kinds
                # every latched page is waiting on. The document is worth
                # reopening as soon as any one of them is back.
                #
                # Cold review round 4, new finding 1: UNKNOWN is the top element
                # of that union, not the empty set. A latched page restored from
                # a record written before this field existed carries no kinds and
                # means "some rung, we cannot say which". Unioning it as nothing
                # narrowed the document to the kinds the OTHER pages happened to
                # name, and a recovery of the unnamed rung was then missed
                # forever. If any latched page is unknown, the document is
                # unknown: the key is omitted, and the gate widens to any rung.
                latched_pages = [
                    p
                    for p in state.pages.values()
                    if getattr(p, "table_judge_retry_pending", False)
                ]
                any_unknown = any(
                    not (getattr(p, "table_judge_retry_rungs", []) or []) for p in latched_pages
                )
                doc_rungs = sorted(
                    {
                        kind
                        for p in latched_pages
                        for kind in (getattr(p, "table_judge_retry_rungs", []) or [])
                    }
                )
                if doc_rungs and not any_unknown:
                    pending["table_judge_retry_rungs"] = doc_rungs
            index_meta = _LatchedDocMetadata(meta, pending) if pending else meta
            RootIndex(output_dir).record(rel_key, index_meta)
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
    def _structure_class_floor_note(state: DocumentState) -> str | None:
        """GH-317: document-level note for structure-class floor pages."""
        try:
            from socr.core.manifest import structure_class_floor_applies

            pages = sorted(
                page_num
                for page_num, page_state in state.pages.items()
                if structure_class_floor_applies(page_state)
            )
            if not pages:
                return None
            return (
                f"page(s) {', '.join(str(page_num) for page_num in pages)}: "
                "structure-class ladder exhausted; fail-closed floor shipped "
                "(marker plus page image, native geometry grid withheld)"
            )
        except Exception as exc:
            logger.warning("structure-class floor note derivation failed (non-fatal): %s", exc)
            return None

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
        from socr.core.audit_log import AuditEvent
        from socr.core.pdf import open_pdf

        if fig_info.bbox is None:
            return

        x0, y0, x1, y1 = fig_info.bbox
        try:
            with open_pdf(state.handle.path) as pdf:
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

    def _detect_and_crop_equation_page(
        self,
        state: DocumentState,
        page_num: int,
        output_dir: Path,
        *,
        page=None,
    ) -> list[EquationRegion]:
        """Detect display-equation regions on one page and save crop PNGs.

        If ``page`` is provided (a ``fitz.Page``), detection and crop rendering
        reuse it directly without opening the document again. If omitted or None,
        opens the PDF for the duration of this call.

        Returns the detected ``EquationRegion`` records in page-vertical order,
        each carrying geometry, ``source_text``, ``equation_label``, and the
        absolute ``crop_path`` (or None if crop render failed).
        """
        from ocr_output_contract import doc_dir_for, relative_key

        from socr.core.audit_log import AuditEvent
        from socr.core.pdf import open_pdf
        from socr.math.detect_equations import (
            EquationDetectionResult,
            detect_display_equations,
            save_equation_crops,
        )

        scan_root = self._scan_root or state.handle.path.parent
        doc_dir = doc_dir_for(output_dir, relative_key(state.handle.path, scan_root))
        equations_dir = doc_dir / "equations"

        if page is None:
            try:
                with open_pdf(state.handle.path) as pdf:
                    try:
                        p = pdf[page_num - 1]
                    except IndexError:
                        logger.warning(
                            "equation detection: page %d out of range (skipping)", page_num
                        )
                        return []
                    return self._detect_and_crop_equation_page(
                        state,
                        page_num,
                        output_dir,
                        page=p,
                    )
            except Exception as exc:
                logger.warning("equation detection: cannot open PDF for page %d: %s", page_num, exc)
                return []

        det: EquationDetectionResult = detect_display_equations(page, page_num)
        if not det.regions:
            return []

        save_equation_crops(det.regions, page, equations_dir, dpi=self.config.render_dpi)

        for region_index, region in enumerate(det.regions):
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
                        "source_text": region.source_text,
                        "equation_label": region.equation_label,
                        "region_index": region_index,
                    },
                )
            )

        return det.regions

    def _detect_and_crop_equations(
        self,
        state: DocumentState,
        page_nums: list[int],
        output_dir: Path,
    ) -> dict[int, list[EquationRegion]]:
        """GH-36a: detect display-equation regions and save crop PNGs.

        Runs the deterministic, model-free detector on each page in
        ``page_nums``.  Saves a crop PNG for every detected region to
        ``equations/`` beside the figures directory, and records provenance in
        ``state.events`` (AuditEvent kind ``equation_region_detected``).

        Returns detected ``EquationRegion`` records grouped by 1-indexed page_num.

        No text is modified, no model is called.  This is DETECTION + EVIDENCE
        only; the engine/validation/splice layer is GH-36b / P4-R.
        """
        from ocr_output_contract import doc_dir_for, relative_key

        from socr.core.pdf import open_pdf

        scan_root = self._scan_root or state.handle.path.parent
        doc_dir = doc_dir_for(output_dir, relative_key(state.handle.path, scan_root))
        equations_dir = doc_dir / "equations"

        results: dict[int, list[EquationRegion]] = {p: [] for p in page_nums}
        try:
            pdf = open_pdf(state.handle.path)
        except Exception as exc:
            logger.warning("equation detection: cannot open PDF: %s", exc)
            return results

        try:
            for page_num in page_nums:
                try:
                    page = pdf[page_num - 1]
                except IndexError:
                    logger.warning("equation detection: page %d out of range (skipping)", page_num)
                    continue

                regions = self._detect_and_crop_equation_page(
                    state,
                    page_num,
                    output_dir,
                    page=page,
                )
                results[page_num] = regions
        finally:
            pdf.close()

        total_regions = sum(len(regs) for regs in results.values())
        if total_regions and not self.config.quiet:
            console.print(
                f"  [dim]GH-36a: {total_regions} equation region(s) detected "
                f"and cropped to {equations_dir}[/dim]"
            )

        return results

    def _guard_equation_sidecar_block(
        self,
        state: DocumentState,
        page_num: int,
        page_out,
        result,
        native_text: str,
    ) -> tuple[str, bool]:
        """Gate the ONE additive post-verdict step (cold review rounds 2-3).

        "Judged bytes are shipped bytes" is scoped by ruling to mean that no
        post-verdict step may ADD or ALTER content. Two enumerated SUBTRACTIVE
        sanitizers are exceptions (see ``_phase_agentic``'s docstring); this
        legacy GH-36b sidecar is the only additive one on the routed page path.

        It is guarded by REUSING the two guards PR #518 built for the P4-R
        region lane, not by re-implementing them here. Round 3 found the local
        copies were strictly weaker, which is the whole argument against copies:

        * **Assembly delimiter** — owned by ``contract_delimiter_violation`` at
          the ``process_equation_region`` choke point, which keys on the
          contract's OWN ``PAGE_MARKER_RE`` (case-insensitive, leading
          whitespace, trailing text) and also refuses a code fence that would
          break out of the sidecar block. A violation lands here as
          ``validation_ok=False``, so the LaTeX is already unattached; nothing
          is left for this method to do about it.
        * **Numeric presence** — ``region_presence_verdict``, the same one-way
          containment guard the region lane uses. It deliberately has NO
          exponent exclusion: #518 removed one because ``x^9`` let an invented 9
          through with an empty candidate multiset. It ABSTAINS
          (``PRESENCE_UNVERIFIABLE``) when the page has no numeric oracle or its
          text layer shows decode damage, so notation-only LaTeX on a prose page
          is not convicted.

        On an INVENTED verdict the LaTeX is refused and the sidecar is rebuilt
        with the crop PNG and the native text kept, so refusing costs no
        content. The reason handed to the rebuilt sidecar is digit-free on
        purpose: it is shipped text, and quoting the invented number there would
        put it back in the corpus by the back door. The numbers live in the
        ``equation_sidecar_refused`` audit event only.

        Returns ``(block_to_append, latex_attached)``.
        """
        from socr.core.audit_log import AuditEvent
        from socr.tables.escalation_canary import (
            PRESENCE_INVENTED,
            PRESENCE_UNVERIFIABLE,
            region_presence_verdict,
            text_value_tokens,
        )

        block = result.sidecar_block or ""
        latex_attached = bool(result.latex_attached)
        if not block or not latex_attached:
            return block, latex_attached

        ps = state.pages.get(page_num)
        encoding_suspect = bool(getattr(ps, "has_encoding_hygiene_suspect", False))
        corrupt_math = bool(getattr(ps, "has_corrupt_math", False))
        text_layer_damaged = encoding_suspect or corrupt_math
        verdict = region_presence_verdict(
            native_text,
            result.raw_latex or "",
            encoding_suspect=encoding_suspect,
            corrupt_math=corrupt_math,
        )
        # GH-543: the legacy GH-36b path had the same hole GH-522 closed on the
        # region lane -- only PRESENCE_INVENTED refused, so an encoding-suspect
        # or corrupt-math page that ABSTAINS still attached a crop-backed LaTeX
        # sidecar carrying numbers nobody could check.
        #
        # Scoped twice over, and both narrowings matter:
        #
        #  - only when the TEXT LAYER IS DAMAGED. UNVERIFIABLE also covers "the
        #    page has no numeric oracle", and there a numeral in the reading is
        #    usually NOTATION -- the 2 in `E = mc^2`, an equation tag -- not a
        #    data value. Refusing on that convicts notation-only LaTeX on prose
        #    pages, which `test_a_page_with_no_numbers_is_unverifiable_not_invented`
        #    deliberately protects. The damaged-text case is the one #522 and
        #    #543 are actually about: an oracle exists but cannot be trusted.
        #  - only when the reading HAS numeric tokens. A pure-symbol equation
        #    carries nothing that can be invented, and dropping it would discard
        #    safe LaTeX on exactly the pages already worst served.
        #
        # Read with `text_value_tokens`, the extractor `region_presence_verdict`
        # uses on the candidate, so the two cannot disagree about what a number
        # is.
        if verdict.status == PRESENCE_INVENTED:
            refusal = "latex carried numbers absent from the page source"
        elif (
            verdict.status == PRESENCE_UNVERIFIABLE
            and text_layer_damaged
            and text_value_tokens(result.raw_latex or "")
        ):
            refusal = f"latex carried numbers the presence guard could not check ({verdict.reason})"
        else:
            return block, latex_attached

        from socr.math.equation_latex import build_equation_sidecar

        block, latex_attached = build_equation_sidecar(
            crop_path=(result.crop_ref or result.crop_path),
            native_text=native_text,
            raw_latex="",
            validation_ok=False,
            validation_reason=refusal,
        )
        state.events.append(
            AuditEvent(
                page_num=page_num,
                kind="equation_sidecar_refused",
                engine="equation_latex",
                detail=(
                    f"region {result.region_index} LaTeX refused by the numeric-presence "
                    f"guard ({refusal}); the crop and the native text are kept, and the "
                    "page is NOT demoted"
                ),
                data={
                    "region_index": result.region_index,
                    "crop_path": result.crop_path,
                    "presence_status": verdict.status,
                    "presence_reason": verdict.reason,
                    "invented": list(verdict.invented),
                    "oracle_size": verdict.oracle_size,
                    "raw_latex": result.raw_latex,
                },
            )
        )
        return block, latex_attached

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
                # Round 2, finding 2: this is the one ADDITIVE post-verdict step,
                # so it goes through the delimiter and numeric-presence guards
                # before any of it can reach shipped bytes.
                _block, _latex_attached = self._guard_equation_sidecar_block(
                    state, page_num, po, result, native_text
                )
                if _block:
                    if po.text:
                        po.text = po.text + "\n\n" + _block
                    else:
                        po.text = _block

                # Provenance: emit audit event.
                if _latex_attached:
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
                            "latex_attached": _latex_attached,
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

        console.print(f"\n{status_str} | {result.engine} | {result.processing_time:.1f}s")
        if state.pages_needing_repair:
            console.print(
                f"[yellow]{len(state.pages_needing_repair)} page(s) still failing[/yellow]"
            )
        if result.error:
            console.print(f"[dim]{result.error}[/dim]")
