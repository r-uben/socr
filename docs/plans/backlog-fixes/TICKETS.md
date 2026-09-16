# Backlog fixes — tickets

One ticket per confirmed-still-valid defect. Dispatch one `socr-implementer` per READY ticket.

---

## GH-140 — math-font pages ship trusted-native with no audit of known-lossy math

**Status:** DONE as an interim observability patch (implemented, two REVISE
rounds applied, committed, awaiting review/CI/merge). **NOT fully resolved:**
see the second REVISE note below — the underlying demote-or-not question is
deferred, not closed.

**REVISE (2026-09-16):** criterion 4's original "yes, demote" answer was
reversed. First reversal argument (36.1% vs. the PUA precedent's 2.4% is
"~15x, unaffordable") was itself corrected in the third round below —
prevalence alone does not justify suppressing a signal, and the trigger-
rates ruling explicitly accepts the same 36% figure for routing. The
correct argument: that acceptance is conditioned on over-routing being a
cost, not a correctness risk (native prose ships regardless); status
demotion changes what every consumer sees and flips the exit code, so the
acceptance does not transfer. Separately, and regardless of the prevalence
question: an 8.0% slice (≤10 math-font chars) has no display equation to
ever recover, so it would be a permanent, unclearable `AUDIT_FAILED` under
the original design. Demotion withheld pending a separate-ticket
`trigger_rates.py` extension measuring the false-positive rate and the
clearable share; the event, sidecar persistence, page note and
CLI/document-note surfacing all still ship. Also added: a resume
round-trip test, and a docstring caveat on `regions_covered` ("not
invented", not "verified correct"). See `docs/log/2026-09-16_140.md`.

**Second REVISE (2026-09-16, design panel):** the document-level note moved
out of `final_result.error` into `final_result.audit_notes` — `error` is
load-bearing (`cli.py` greps it for `LOST_CONTENT_NOTE`; GH-177 documents it
as "already AUDIT_FAILED"), so a populated `error` on a `success=True`
result would smuggle the full failure blast radius back in through a field
a reasonable caller checks before `status`. Also recorded here: **this
ticket does not establish that the math-font signal is a confirmed defect**
— it is prevalence-only (the trigger-rates log), with no measured
false-positive rate, so every other note bucket in this block reports an
observed defect while this one reports a suspicion. Named follow-up (not
implemented, both panel models converged on it independently): demote only
when `equation_region_evidence` shows a region was FOUND
(`regions_total > 0`) and recovery FAILED to cover it
(`regions_covered < regions_total`) — a structural gate needing no invented
character-count threshold, under which the 8.0% no-region slice correctly
never confirms rather than being permanently stuck. See
`docs/log/2026-09-16_140.md`.
**Branch:** `fix/140-math-font-audit`
**Write ownership:** `src/socr/math/accounting.py`, `src/socr/core/born_digital.py`,
`tests/` (a new test module for this ticket). **Deviation:** also touched
`src/socr/core/state.py`, `src/socr/core/manifest.py`,
`src/socr/pipeline/orchestrator.py` — necessarily, the PUA precedent this
ticket mirrors spans exactly these same three files. See
`docs/log/2026-09-16_140.md`.

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

---

## GH-658a — a default install has no scanned-table witness, silently

**Status:** READY
**Branch:** `fix/658-scanned-witness`
**Write ownership:** `pyproject.toml`, `src/socr/tables/source_evidence.py` (warning only),
`tests/`

### Context — confirmed still real on `main@5478b42`
`pytesseract` is declared **nowhere** in `pyproject.toml` — not a dependency, not an extra.
So a fresh install always lands on `WITNESS_PACKAGE_MISSING` (`source_evidence.py:84-86`,
whose own comment says exactly this), `classical_ocr_pixmap` returns `""`, and a scanned
table page has no evidence source. The owner installed tesseract by hand on this Mac on
2026-09-08; nobody else gets that.

`source_evidence.py` has had **0 commits** since the triage baseline.

### Plan
1. Declare the witness as an optional extra (the issue suggests `socr[scanned]`).
2. Emit a **loud, once-per-run** warning when a scanned table page is assessed and no
   evidence source is available — naming the missing package and the extra that fixes it.

### Acceptance Criteria
1. `pyproject.toml` declares the extra; a plain install is unchanged in behaviour.
2. A scanned table page with no witness emits the warning exactly once per run, not once
   per page (log noise on a 200-page scan is its own defect).
3. The warning names `WITNESS_PACKAGE_MISSING` vs `WITNESS_BINARY_MISSING` distinctly —
   "install the extra" and "install the tesseract binary" are different user actions and
   `source_evidence.py` already distinguishes them.
4. **No behaviour change to fail-closed logic.** This ticket is packaging + visibility only.
5. No new magic threshold.

---

---

## GH-658b — a distrusted text layer is discarded instead of used as a witness

**Status:** READY — **BEHAVIOUR CHANGE, reviewer must scrutinise**
**Branch:** `fix/658-scanned-witness`
**Write ownership:** `src/socr/tables/source_evidence.py`, `tests/`
**Depends on:** GH-658a landing first is preferred but not required.

### Context
Measured (D3 re-measure, `docs/log/2026-09-07_D3-fed-table-lane-remeasure.md`): on Fed
1977-11-15 p3, 1982-11-16 p3 and 1990-11-13 p3, cached candidates carry **62/62, 67/67,
66/66** of the page's numbers with **0 extras**, and each is rejected
`source_evidence_table_reject: no local content evidence available for scanned table` ->
`table_ladder_unverified cause=no_witness` -> fail-closed marker. **Shipped 1-11% of the
numbers; an older heuristic run shipped 100%.**

Root cause: `build_scanned_evidence` excludes the page text layer when it is distrusted
(GH-163), so `has_content_evidence` is False and A1b row corroboration (#640) never runs.

### Plan
When the text layer exists but is distrusted, use it as a **corroboration** witness via
`corroborate_rows` — ordered row match tolerates the measured 6-7% corruption (the 1989
fixture had 295 usable native words) — and ship **flagged** `header_binding_unverified`
per the 2026-09-06 owner ruling, instead of shipping no witness at all.

### Acceptance Criteria — read criterion 1 twice
1. **This RELAXES a fail-closed path. It must not become a silent accept.** A page rescued
   this way ships FLAGGED, and that flag must surface at every level the repo requires —
   page status, document status, metadata, CLI — never as a clean SUCCESS. GH-249's
   lesson, one week old: in this pipeline an absent refusal reads downstream as consent.
2. A distrusted text layer is used only as *corroboration*, never promoted to trusted
   content, and never merged into the shipped text.
3. A page with **no** text layer at all still fails closed exactly as today. This ticket
   rescues the distrusted-layer case only.
4. A candidate that genuinely disagrees with the distrusted layer must still be rejected —
   demonstrate with a fixture where corroboration fails.
5. No new magic threshold: reuse `corroborate_rows`' existing tolerance, do not invent one.

### Verification (both tickets)
- FULL suite: `PYTHONPATH=$PWD/src ~/venvs/socr/bin/pytest -q`. Never a `-k` subset.
- `uvx ruff@0.16.0 format --check .` (not the venv's older ruff).
- Every new test demonstrated to FAIL without the change.
- Pin a DIFFERENCE, not a locally-measured absolute — CI has no provider and no tesseract.
- Do not `Closes #658` unless BOTH 658a and 658b are complete.
