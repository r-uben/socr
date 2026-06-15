"""Structural LaTeX validation gate — GH-36b 1A policy.

Provides a single, deterministic, offline function ``validate_latex_structure``
that decides whether a VLM-produced LaTeX string is structurally well-formed
before it is attached adjacently to a detected equation crop.

Policy (per consilium decision 20260615T210537Z-6621, Hybrid 1A + 1C):
- 1A structural gate: pylatexenc pure-Python validation (offline, deterministic,
  replay-safe).  A string passes if it is:
    * non-empty (after stripping whitespace),
    * parseable by ``pylatexenc.latexwalker.LatexWalker`` without a
      ``LatexWalkerError`` being raised, AND
    * has at least one non-space character (i.e. not purely whitespace nodes).
- On failure: do NOT attach LaTeX; keep the faithful crop PNG + native text.
- 1B (full render / image-compare) REJECTED by the panel: no render toolchain.

The function is self-contained and has no side effects; it is safe to call from
any context (tests, orchestrator, future replay).

Validation semantics
---------------------
1A validates **syntax**, not **visual fidelity**.  A syntactically-valid LaTeX
string can still be a semantic hallucination (swapped subscripts/superscripts,
wrong operator, missed factor).  The crop PNG is always retained alongside any
attached LaTeX precisely because 1A does not and cannot guarantee fidelity.  The
crop is the authoritative visual ground truth; the attached LaTeX is a
structurally-validated candidate, non-authoritative.

This distinction MUST NOT be lost in future engineering: upgrading from 1A to 1B
(full render) or beyond requires a separate explicit decision.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ── Validation prompt constant (reused from recover.py / referenced by tests) ─

# The prompt that should be sent to the local VLM when asking it to transcribe
# a detected equation crop.  Kept here as a named constant so:
#   * tests can assert it is used / inspect it,
#   * it is never an anonymous magic string in call sites, and
#   * it can be updated in one place without hunting through the codebase.
#
# Design note: The prompt deliberately avoids giving the model numeric examples
# of what "correct" LaTeX looks like (per repo rule: no magic numbers/thresholds
# hardcoded in prompts).  It instead states the task and the required output
# format in functional terms.  The model reasons from context (the image) and
# its own training.
EQUATION_LATEX_PROMPT = (
    "Transcribe the mathematical equation shown in this image to LaTeX. "
    "Output ONLY the LaTeX source — no prose explanation, no markdown code "
    "fences, no surrounding $ or $$ delimiters. "
    "Be faithful to every symbol: subscripts, superscripts, fractions, "
    "integrals, Greek letters, operators, and parentheses. "
    "If any part of the equation is unreadable, omit it rather than guessing."
)


def validate_latex_structure(latex: str) -> tuple[bool, str]:
    """Deterministic 1A structural-validation gate for VLM-produced LaTeX.

    Accepts the LaTeX string returned by the local VLM (after ``clean_latex``
    stripping of fences/delimiters) and returns a ``(ok, reason)`` pair where:

    * ``ok=True`` means the string passed all structural checks and MAY be
      attached adjacently to the crop PNG (still non-authoritative).
    * ``ok=False`` means the string failed and MUST NOT be attached; the native
      linearised text and crop PNG are kept as the faithful fallback.

    Checks (in order, fail-fast):
    1. Non-empty: ``latex.strip()`` is not empty.
    2. Parseable: ``LatexWalker(latex).get_latex_nodes()`` does not raise
       ``LatexWalkerError`` (catches unmatched ``{}``, unclosed environments,
       etc.).

    Parameters
    ----------
    latex:
        The VLM output string after fence/delimiter stripping by ``clean_latex``.
        May be empty or contain only whitespace (→ fails check 1).

    Returns
    -------
    (True, "ok")
        String passed both checks.
    (False, "<reason>")
        String failed; ``reason`` is a one-liner suitable for audit records.

    Never raises; any unexpected error is treated as a structural failure.
    """
    try:
        from pylatexenc.latexwalker import LatexWalker, LatexWalkerError
    except ImportError as exc:  # pragma: no cover — pylatexenc is a declared dep
        logger.error("pylatexenc not available; treating all LaTeX as invalid: %s", exc)
        return False, "pylatexenc not available"

    stripped = latex.strip()

    # Check 1 — non-empty
    if not stripped:
        return False, "empty or whitespace-only LaTeX"

    # Check 2 — structural parse
    try:
        walker = LatexWalker(stripped, tolerant_parsing=False)
        nodelist, _pos, _len = walker.get_latex_nodes(pos=0)
    except LatexWalkerError as exc:
        # Surface the parse error in the reason (truncated to keep audit compact).
        reason = str(exc).split("\n")[0][:200]
        return False, f"parse error: {reason}"
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("validate_latex_structure: unexpected error: %s", exc)
        return False, f"unexpected validation error: {exc}"

    return True, "ok"
