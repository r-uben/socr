"""GH-36b: equation crop → LaTeX via local VLM + 1A structural validation + 1C sidecar.

This module implements the three-step GH-36b pipeline for each detected
display-equation crop (produced by GH-36a ``detect_equations.py``):

  1. **Engine call** — read the crop PNG with the local VLM
     (``qwen3-vl:30b-a3b-instruct``) to obtain a LaTeX string.  Reuses
     ``recover.latex_for_image`` (same Ollama endpoint, same keep_alive
     semantics, same empty-string-on-failure contract).  Mock-testable via
     the injectable ``ocr`` parameter.

  2. **1A structural gate** — validate the VLM output with
     ``validate_latex.validate_latex_structure`` (pylatexenc, offline,
     deterministic).  Only structurally well-formed LaTeX proceeds; failures
     keep the native text and record the reason.

  3. **1C non-destructive sidecar** — the faithful crop PNG is ALWAYS inlined
     in the document (as a Markdown image reference).  If LaTeX passed 1A it
     is attached ADJACENTLY in a fenced ```latex block preceded by a comment
     marking it as structurally-validated, non-authoritative.  If 1A failed
     the native linearised text is emitted below the crop reference instead.
     Neither the crop nor the native text is ever silently replaced.

The design was settled by the consilium panel (run 20260615T210537Z-6621):
  - 1A (structural validation) + 1C (non-destructive sidecar) policy, unanimous.
  - 1B (full render / image-compare) REJECTED.
  - Engine: reuse local ``qwen3-vl:30b-a3b-instruct``, NOT marker-pdf/Texify.

This path is gated by ``config.recover_clean_equations`` (default False).
Do NOT enable by default until throughput is measured on a real corpus.

Named constants
---------------
``EQUATION_SIDECAR_HEADER``
    Comment line prepended to every 1A-validated LaTeX block to mark it as a
    structurally-validated, non-authoritative candidate.  Plain HTML comment so
    it survives most Markdown renderers without display artefacts.

``EQUATION_SIDECAR_FAILED_HEADER``
    Inline note prepended to the native-text block when 1A fails, explaining
    why LaTeX is absent.  Keeps the failure visible in the document without
    being disruptive.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from socr.math.recover import DEFAULT_HOST, DEFAULT_MODEL
from socr.math.validate_latex import EQUATION_LATEX_PROMPT, validate_latex_structure

logger = logging.getLogger(__name__)

# ── Sidecar formatting constants ────────────────────────────────────────────

# HTML comment prepended to every 1A-validated LaTeX block.  It marks the
# block as a structurally-validated candidate (syntax only — NOT a fidelity
# guarantee) so downstream readers and human reviewers know the provenance.
# Kept as a named constant so tests and callers can assert its presence.
EQUATION_SIDECAR_HEADER = (
    "<!-- socr-equation: structurally-validated LaTeX candidate "
    "(1A syntax gate, non-authoritative — see crop for ground truth) -->"
)

# Note emitted in place of a LaTeX block when 1A validation fails.  It
# preserves the failure reason inline and signals that the native text is the
# authoritative content below.
EQUATION_SIDECAR_FAILED_HEADER = (
    "<!-- socr-equation: LaTeX validation failed — "
    "native linearised text retained (crop is the visual ground truth) -->"
)

# ── Data structures ──────────────────────────────────────────────────────────


@dataclass
class EquationLatexResult:
    """Result of processing one equation crop through the engine + 1A gate + 1C sidecar.

    Attributes
    ----------
    region_index:
        0-based index of the ``EquationRegion`` this result corresponds to.
    page_num:
        1-indexed page number.
    crop_path:
        Absolute or relative path to the saved crop PNG (from GH-36a).  May be
        None if the crop was not saved (e.g. render failure in GH-36a).
    raw_latex:
        The VLM output after ``clean_latex`` stripping, before validation.
        Empty string if the engine failed or returned nothing.
    validation_ok:
        True if the 1A structural gate passed; False otherwise.
    validation_reason:
        "ok" on pass; a one-liner explanation on failure.
    latex_attached:
        True when the LaTeX was attached adjacently (only possible when
        ``validation_ok=True``).
    model_id:
        The model identifier used for the engine call.
    sidecar_block:
        The formatted Markdown/HTML sidecar block to be inserted into the
        document body, or "" if this result has nothing to contribute (e.g.
        no crop path and validation failed).
    source_text:
        The region's EXACT native text slice (``EquationRegion.source_text``).
        P4-R attaches the sidecar immediately after this slice inside the page's
        own native prose, so the slice must be retained verbatim; an empty
        string means "no in-place anchor", and the attachment helper reports the
        record as unaligned rather than guessing a position.
    crop_ref:
        The Markdown-visible crop reference (e.g. ``equations/x.png``), which is
        relative to the document directory and therefore NOT the same string as
        ``crop_path`` (an absolute filesystem path handed to the model). "" when
        no crop was retained.
    """

    region_index: int
    page_num: int
    crop_path: str | None
    raw_latex: str
    validation_ok: bool
    validation_reason: str
    latex_attached: bool
    model_id: str
    sidecar_block: str = ""
    source_text: str = ""
    crop_ref: str = ""
    # P4-R only: the numeric-presence guard's verdict on this region's reading.
    # "" for the legacy GH-36b path, which has no such guard.
    presence_status: str = ""
    presence_reason: str = ""


# ── Engine call ──────────────────────────────────────────────────────────────


def latex_for_crop(
    crop_path: str | Path,
    *,
    ocr=None,
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
    timeout: float = 300.0,
    keep_alive: str = "30m",
) -> str:
    """Read a saved equation-crop PNG and return cleaned LaTeX from the local VLM.

    Reads the PNG from ``crop_path``, base64-encodes it, and sends it to the
    local Ollama API using ``recover.latex_for_image``.  The ``ocr`` parameter
    is injectable for tests (avoids a real network call).

    The prompt used is ``validate_latex.EQUATION_LATEX_PROMPT`` — a named
    constant, not an anonymous magic string.

    Parameters
    ----------
    crop_path:
        Path to the saved crop PNG (saved by GH-36a ``save_equation_crops``).
    ocr:
        Injectable engine callable ``(png_bytes, *, model, host, ...) -> str``.
        Defaults to ``recover.latex_for_image`` when None.
    model:
        Local Ollama model to use.  Must be ``qwen3-vl:30b-a3b-instruct``
        (the only sanctioned local vision model); NEVER ``:8b`` or ``:30b``.
    host, timeout, keep_alive:
        Forwarded to the Ollama call.

    Returns
    -------
    str
        Cleaned LaTeX string (possibly empty on engine failure).
    """
    path = Path(crop_path)
    try:
        png_bytes = path.read_bytes()
    except OSError as exc:
        logger.warning("latex_for_crop: cannot read crop %s: %s", crop_path, exc)
        return ""

    # Injectable stub path (for tests — avoids real network call).
    if ocr is not None:
        return ocr(png_bytes, model=model, host=host)

    # Real path: call the Ollama endpoint directly with EQUATION_LATEX_PROMPT.
    # We do NOT delegate to ``recover.latex_for_image`` because that function
    # uses the corrupt-math prompt (``_PROMPT``); the clean-equation path needs
    # EQUATION_LATEX_PROMPT.  The transport layer is identical.
    import base64
    import json
    import urllib.error
    import urllib.request

    from socr.math.recover import clean_latex

    payload = json.dumps(
        {
            "model": model,
            "prompt": EQUATION_LATEX_PROMPT,
            "images": [base64.b64encode(png_bytes).decode()],
            "stream": False,
            "keep_alive": keep_alive,
            "options": {"temperature": 0, "num_ctx": 8192},
        }
    ).encode()
    req = urllib.request.Request(
        f"{host}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
        logger.warning("equation LaTeX engine call failed: %s", exc)
        return ""

    return clean_latex(body.get("response", ""))


# ── 1C sidecar builder ───────────────────────────────────────────────────────


def build_equation_sidecar(
    crop_path: str | None,
    native_text: str,
    raw_latex: str,
    validation_ok: bool,
    validation_reason: str,
) -> tuple[str, bool]:
    """Build the 1C non-destructive sidecar block for one equation region.

    The sidecar is inserted ADJACENTLY to any existing body text; it never
    replaces the crop or the native linearised text.  The caller is responsible
    for positioning the block (e.g. after the relevant prose paragraph or as a
    standalone block after detection).

    Policy (1C, consilium 20260615T210537Z-6621):
      * The crop PNG image reference is ALWAYS emitted (it is the visual ground
        truth).
      * If ``validation_ok``: attach the LaTeX in a fenced ```latex block
        beneath the image reference, preceded by ``EQUATION_SIDECAR_HEADER``.
      * If not ``validation_ok``: emit ``EQUATION_SIDECAR_FAILED_HEADER`` +
        the native linearised text (so the failure is transparent).
      * The native text is NEVER omitted or replaced by the LaTeX.

    Parameters
    ----------
    crop_path:
        Path to the crop PNG.  If None, we emit a placeholder comment instead
        of an image reference (crop failed to save in GH-36a).
    native_text:
        The native linearised text for this equation region.  Kept on failure
        and always included for transparency.
    raw_latex:
        Cleaned VLM LaTeX.  Only used when ``validation_ok=True``.
    validation_ok:
        Result of the 1A gate.
    validation_reason:
        Reason string for the 1A result (logged in the sidecar on failure).

    Returns
    -------
    (sidecar_block, latex_attached)
        ``sidecar_block``: the formatted Markdown/HTML string to be appended
        to the page body.
        ``latex_attached``: True when ``raw_latex`` was included in the block.
    """
    lines: list[str] = []

    # Always inline the crop as the visual ground truth.
    if crop_path:
        lines.append(f"![equation crop]({crop_path})")
    else:
        lines.append("<!-- socr-equation: crop PNG unavailable -->")

    latex_attached = False

    if validation_ok and raw_latex.strip():
        # 1C: LaTeX passes 1A → attach adjacently, clearly marked non-authoritative.
        lines.append(EQUATION_SIDECAR_HEADER)
        lines.append("```latex")
        lines.append(raw_latex.strip())
        lines.append("```")
        latex_attached = True
    else:
        # 1A failed (or empty LaTeX) → keep native text, note the failure.
        reason_note = (
            f"validation failed: {validation_reason}"
            if not validation_ok
            else "engine returned empty output"
        )
        lines.append(
            f"<!-- socr-equation: LaTeX NOT attached ({reason_note}) — "
            f"native text is the authoritative content -->"
        )
        if native_text.strip():
            lines.append(native_text.strip())

    return "\n".join(lines), latex_attached


# ── Top-level per-region processing ─────────────────────────────────────────


# ── Assembly-contract safety ────────────────────────────────────────────────


def contract_delimiter_violation(text: str) -> str:
    """Reason why ``text`` must not be embedded in a page body, or "".

    Cold review round 1, finding 1. Model output reaches the page inside a
    fenced block, and the 1A gate is a LaTeX SYNTAX check -- it accepts
    ``y = 2x + 1\n## Page 3`` because that is well-formed LaTeX text. The
    output contract, however, keys page boundaries on ``## Page N`` lines
    anywhere in the body: ``assemble_pages`` writes them and
    ``split_native_pages`` reads them back, fenced or not. A model-authored
    marker therefore invents a page boundary, and
    ``_rewrite_all_fragments`` then writes the first half of a page's native
    prose to one fragment and the rest to the next -- whole-page native prose
    split and reassigned by a model reading, which ruling 3 forbids outright.
    It also corrupts replay, because the saved body no longer round-trips to
    the document's page count.

    Reproduced end to end before this guard existed: a three-page fixture
    assembled with FOUR page markers and split into four logical pages.

    There is no escape convention for these delimiters anywhere in the
    contract -- the only existing handling is a leading-marker STRIP in the
    provisional flush, which cannot help a marker in the middle of a body -- so
    a violating reading is REJECTED rather than escaped. Rejection costs one
    region's LaTeX; the alternative costs a page's prose.

    Checked:
      * ``PAGE_MARKER_RE`` -- the page-boundary convention itself.
      * a triple-backtick run -- it closes the sidecar's own fenced latex
        block, after which everything the model wrote is live Markdown and any
        marker it contains is a real heading rather than fenced text.
    """
    from ocr_output_contract import PAGE_MARKER_RE

    if not text:
        return ""
    if PAGE_MARKER_RE.search(text):
        return "output contains a '## Page N' boundary marker owned by the output contract"
    if "```" in text:
        return "output contains a code fence that would break out of the sidecar block"
    return ""


def attach_equation_sidecars_in_place(
    native_text: str,
    results: list[EquationLatexResult],
) -> tuple[str, list[int]]:
    """Splice each result's sidecar in immediately after its own native slice.

    P4-R ruling 3: the model reading is REGION-SCOPED and ADVISORY. The unit of
    replacement is the equation region with its crop attached, and whole-page
    native prose is never swapped for a whole-page model read. So this helper is
    strictly ADDITIVE: it finds each result's exact ``source_text`` inside
    ``native_text`` and inserts ``sidecar_block`` directly after it. The native
    bytes are never removed, reordered, or rewritten — every character of
    ``native_text`` survives, in order, in the returned string.

    Callers pass ONLY records they have already decided are attachable (crop
    retained, 1A-valid LaTeX, presence guard not FAIL). A record this function
    is not given contributes nothing, which is what makes a refused, rejected or
    provider-less page ship bytes identical to a run with the lane off.

    Determinism on repeated slices: results are consumed in the order given and
    each search starts after the previously consumed slice, so two regions whose
    native text is identical attach to the first and second occurrence
    respectively rather than piling onto the same one.

    Idempotence: a slice already followed by its own sidecar block is left
    alone, so re-running over this function's own output does not double-attach.
    Such a record is NOT reported as unaligned — it is already attached.

    Returns
    -------
    (text, unaligned)
        ``text``: the page text with sidecars spliced in (``native_text``
        unchanged when nothing attached).
        ``unaligned``: ``region_index`` of every record whose ``source_text``
        could not be located, in input order. Nothing is appended for those —
        a dangling block at the page end would put a reading somewhere it does
        not belong — so the caller records them as an audit event instead.
    """
    if not results:
        return native_text, []

    unaligned: list[int] = []
    # (insert_at, block) pairs collected against the ORIGINAL string, applied in
    # one pass at the end so no offset is invalidated mid-scan.
    inserts: list[tuple[int, str]] = []
    cursor = 0
    for result in results:
        slice_text = result.source_text or ""
        block = result.sidecar_block or ""
        if not slice_text or not block:
            unaligned.append(result.region_index)
            continue
        found = native_text.find(slice_text, cursor)
        if found < 0:
            unaligned.append(result.region_index)
            continue
        end = found + len(slice_text)
        cursor = end
        # Already attached (idempotence): the block follows this slice, modulo
        # the separator this function itself writes.
        if native_text[end:].lstrip("\n").startswith(block):
            continue
        inserts.append((end, f"\n{block}"))

    if not inserts:
        return native_text, unaligned

    out: list[str] = []
    prev = 0
    for at, block in inserts:
        out.append(native_text[prev:at])
        out.append(block)
        prev = at
    out.append(native_text[prev:])
    return "".join(out), unaligned


def process_equation_region(
    region_index: int,
    page_num: int,
    crop_path: str | None,
    native_text: str,
    *,
    source_text: str = "",
    crop_ref: str | None = None,
    ocr=None,
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
    timeout: float = 300.0,
    keep_alive: str = "30m",
) -> EquationLatexResult:
    """Full GH-36b pipeline for one equation region: engine → 1A gate → 1C sidecar.

    Parameters
    ----------
    region_index:
        0-based index (for provenance tracking).
    page_num:
        1-indexed page number.
    crop_path:
        Path to the saved crop PNG (may be None if GH-36a crop failed).
    native_text:
        Native linearised text for this region (faithful but flattened).
    source_text:
        The region's exact native slice, retained on the result so P4-R can
        attach the sidecar in place. Defaults to "" for the legacy GH-36b
        caller, which appends its block at page end and needs no anchor.
    crop_ref:
        Markdown-visible crop reference written into the sidecar. Defaults to
        ``crop_path`` (the legacy behaviour); P4-R passes the document-relative
        path so the shipped .md does not carry an absolute filesystem path.
    ocr:
        Injectable engine callable (for tests — avoids real model call).
    model:
        Local vision model.  Must be ``qwen3-vl:30b-a3b-instruct``.
    host, timeout, keep_alive:
        Forwarded to the Ollama endpoint.

    Returns
    -------
    EquationLatexResult
        Full provenance record including raw_latex, 1A result, attachment
        decision, model id, and the formatted sidecar block.
    """
    # Step 1 — engine call (skip if no crop)
    raw_latex = ""
    if crop_path:
        raw_latex = latex_for_crop(
            crop_path,
            ocr=ocr,
            model=model,
            host=host,
            timeout=timeout,
            keep_alive=keep_alive,
        )

    # Step 2 — 1A structural validation gate
    if raw_latex.strip():
        validation_ok, validation_reason = validate_latex_structure(raw_latex)
    else:
        validation_ok = False
        validation_reason = "engine returned empty output — no crop or call failed"

    # Step 2b — assembly-contract gate. The 1A check is LaTeX syntax only and
    # happily accepts a page-boundary marker; embedding one would let a model
    # reading split the document. Applied here so BOTH consumers of this
    # function (the legacy GH-36b sidecar and the P4-R region lane) are
    # covered by one choke point.
    if validation_ok:
        violation = contract_delimiter_violation(raw_latex)
        if violation:
            validation_ok = False
            validation_reason = f"assembly-contract violation: {violation}"

    # Step 3 — 1C non-destructive sidecar
    ref = crop_path if crop_ref is None else crop_ref
    sidecar_block, latex_attached = build_equation_sidecar(
        crop_path=ref,
        native_text=native_text,
        raw_latex=raw_latex,
        validation_ok=validation_ok,
        validation_reason=validation_reason,
    )

    return EquationLatexResult(
        region_index=region_index,
        page_num=page_num,
        crop_path=crop_path,
        raw_latex=raw_latex,
        validation_ok=validation_ok,
        validation_reason=validation_reason,
        latex_attached=latex_attached,
        model_id=model,
        sidecar_block=sidecar_block,
        source_text=source_text,
        crop_ref=ref or "",
    )
