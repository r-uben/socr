# Backlog fixes — tickets

One ticket per confirmed-still-valid defect. Dispatch one `socr-implementer` per READY ticket.

---

## GH-140 — math-font pages ship trusted-native with no audit of known-lossy math

**Status:** READY
**Branch:** `fix/140-math-font-audit`
**Write ownership:** `src/socr/math/accounting.py`, `src/socr/core/born_digital.py`,
`tests/` (a new test module for this ticket)

### Context — confirmed still real on `main@5478b42`

`born_digital.py:2991` still reads:

```python
needs_ocr_enhancement = has_corrupt_math
```

`_detect_math_fonts` (`born_digital.py:3276`) feeds `has_equations`, which is metadata and
deliberately does NOT request enhancement. So a page whose maths is typeset in math fonts —
detected, and known by that detector's own docstring to extract badly ("subscripts flatten,
Greek letters drop, reading order breaks around equations") — takes the free native lane and
ships SUCCESS.

Recovery exists only behind `--detect-equations` / `--recover-clean-equations`, both default
False, so by default nothing recovers it **and nothing records the loss**.

The adjacent PUA class is handled better: `socr/math/accounting.py` emits
`native_math_unrecovered` (`UNRESOLVED_MATH_KIND`) for unmapped-glyph damage. Note that
module's docstring already names #140 — the PUA half landed, this half did not. A math-font
page with no PUA codepoints emits nothing at all. Same class of loss, strictly less surface.

`born_digital.py` has had **0 commits** since the triage baseline, so this diagnosis is current.

### Scope — the audit event ONLY

The issue offers two remedies and settles neither:
1. emit a mandatory audit event when the recovery flags are off;
2. route math-font pages to whole-page OCR.

**(2) is explicitly OUT OF SCOPE.** The issue itself notes `born_digital.py:687` warns that
lane was *empirically worse* for the adjacent PUA class because it falls back to the same
broken native layer, and whether that transfers here is an unmeasured empirical question. Do
not change routing. Do not change `needs_ocr_enhancement`.

### Plan
Mirror the existing PUA accounting for the math-font case: when a page is trusted native, math
fonts were detected, and no retained recovery covers that maths, record it.

Follow `socr/math/accounting.py`'s existing discipline: it is **pure** — opens no PDF, renders
no crop, calls no provider, because callers run inside repeated page finalization. Keep it so.
Reuse that module rather than inventing a parallel one.

### Acceptance Criteria
1. A born-digital page with detected math fonts, no PUA damage, and recovery flags off emits a
   durable audit event naming the loss. The event survives into the page sidecar.
2. The signal is **outcome-based, not configuration-based**. Turning the two recovery flags on
   must not silence the event unless recovery actually covered the maths — that inversion is
   the exact bug #165 fixed for the PUA class; do not reintroduce it here.
3. No double-reporting: a page that already emits `native_math_unrecovered` for PUA damage must
   not also emit the new event for the same maths.
4. **Decide explicitly whether the new kind joins the status-demoting set at
   `orchestrator.py:2559`**, and justify the choice in the decision log. This repo's rule is
   that loss surfaces at every level — page status, document status, metadata, CLI — not just
   one. Silent-by-default is the failure mode this ticket exists to close, so argue the case
   either way rather than defaulting quietly.
5. No new magic threshold.

### Verification
- Run the **FULL** suite: `PYTHONPATH=$PWD/src ~/venvs/socr/bin/pytest -q`. **Not** a `-k`
  subset — a keyword filter is exactly how GH-249's regression reached CI.
- `uvx ruff@0.16.0 format --check .` (NOT the venv's older ruff).
- Every new test must be shown to FAIL without the change.
- Pin a DIFFERENCE, not an absolute outcome measured locally: CI has no ollama and no provider.
- Do not `Closes #140` unless criteria 1-4 all hold; say which remain otherwise.
