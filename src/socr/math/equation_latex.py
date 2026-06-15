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


def process_equation_region(
    region_index: int,
    page_num: int,
    crop_path: str | None,
    native_text: str,
    *,
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

    # Step 3 — 1C non-destructive sidecar
    sidecar_block, latex_attached = build_equation_sidecar(
        crop_path=crop_path,
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
    )
