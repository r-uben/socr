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

**Status:** DONE
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

**Status:** DONE — **BEHAVIOUR CHANGE, reviewer scrutinised one collision (see below)**
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

### Collision on implementation (flagged for reviewer)
Implementing the rescue as specified made exactly one pre-existing test fail out of the
full suite: `test_gh163_scanned_native_trust.py::TestTheSuspectLayerCannotCorroborate`
(cubic P1 on #512) hard-pinned a reject for the precise fixture this ticket's root cause
targets (a candidate whose only "evidence" is the untrusted layer itself). Per this
ticket's own text ("use it as a corroboration witness ... instead of shipping no witness
at all") and criterion 1, that test was rewritten to assert the new invariant — flagged
accept, not silent success — rather than left as a stale reject. See
`docs/log/2026-09-16_658.md` for the exact diff reasoning and both mutation-based
fails-without-fix demonstrations.

### Verification (both tickets)
- FULL suite: `PYTHONPATH=$PWD/src ~/venvs/socr/bin/pytest -q`. Never a `-k` subset.
- `uvx ruff@0.16.0 format --check .` (not the venv's older ruff).
- Every new test demonstrated to FAIL without the change.
- Pin a DIFFERENCE, not a locally-measured absolute — CI has no provider and no tesseract.
- Do not `Closes #658` unless BOTH 658a and 658b are complete.

---

## GH-221 — cascade-halt cannot fire: the liveness probe is blind to a wedged GPU

**Status:** DONE — see `docs/log/2026-09-16_221.md`
**Branch:** `fix/221-wedge-canary`
**Write ownership:** `src/socr/tables/extract.py`, `tests/` (a new module for this ticket)

### Context — confirmed still real on `main@3e04f1c`

`probe_ollama_idle` (`src/socr/tables/extract.py:189`) is a `GET /api/tags`. **Its own
docstring concedes the limitation:**

> This is a lightweight /api/tags ping — it does NOT check whether a generation is still
> running server-side (Ollama does not expose that). It only tells us whether the HTTP
> layer is healthy.

The PP-2 cascade-halt guard depends on it:

```python
_had_timeout = any("timeout" in (att.reason or "") for att in decision.attempts)
if _had_timeout and not probe_ollama_idle():
    backend_degraded = True
    halt_reason = "PARTIAL_SAVE_VLM_TIMEOUT"
```

A wedged VLM leaves Ollama's HTTP layer perfectly healthy — the issue measured `/api/tags`
returning **200 OK in 0.05-0.14s** with `qwen3-vl:30b-a3b-instruct` mid-generation at 100%
GPU. So `probe_ollama_idle()` returns True, the guard never arms, and the loop keeps firing
pages into a jammed GPU. **The safety mechanism cannot fire for the exact failure it was
built to catch.**

GH-222 fixed *which host* is probed, not the functional blindness. `extract.py` has had
**0 commits** since the triage baseline.

### Plan

Replace the liveness ping with a **functional canary**: a minimal generation request
against the *same model*, which is the only thing that distinguishes "HTTP alive" from
"GPU available". A tiny `num_predict` request that returns promptly means the backend can
actually serve; one that times out means it cannot.

Keep `/api/tags` as a cheap precondition if useful — an unreachable host is still a halt —
but it must no longer be the *sole* evidence.

### Acceptance Criteria

1. A backend whose HTTP layer answers but whose GPU is wedged is detected as **not idle**,
   so the cascade-halt arms.
2. A healthy backend is still detected as idle — no false halt. A false positive here stops
   a legitimate run, so this is as important as criterion 1.
3. **The canary runs only after a timeout has already been observed.** It must not add a
   generation call to the happy path — cost and latency on every page would be a
   regression, and the guard's call site already gates on `_had_timeout`.
4. **No new magic threshold.** Derive the canary's timeout from an existing constant or
   from the observed timeout that triggered the check. Do not invent a number.
5. The probe must remain safe when the backend is unreachable entirely (connection refused,
   DNS failure) — that is still "not idle", not an exception escaping into the pipeline.

### Verification — read this before writing a single test

**CI HAS NO OLLAMA AND NO PROVIDER.** This ticket is about a network service, so it is the
most exposed change in the queue to the repo's most-documented trap. Every test must be
**hermetic**: patch the HTTP/generation call, never contact a real endpoint. A test that
passes here because Ollama is running locally and fails in CI is worse than no test.

- Run the **FULL** suite: `PYTHONPATH=$PWD/src ~/venvs/socr/bin/pytest -q`. Never a `-k`
  subset — that filter let a regression reach CI on GH-249.
- **Pin a DIFFERENCE, not an absolute:** drive the same cascade-halt decision twice in one
  process, changing only the canary's simulated response (wedged vs healthy), and assert
  the halt decision differs exactly as intended.
- **Prove the guard is load-bearing by MUTATION, not deletion:** neuter the canary so it
  always reports idle, re-run, and show which tests fail and which correctly still pass. A
  test that fails only with an ImportError proves a symbol is new, not that behaviour changed.
- `uvx ruff@0.16.0 format --check .` — not the venv's older ruff.
- Do not `Closes #221` unless criteria 1-5 all hold; say which remain otherwise.
- Decision log at `docs/log/2026-09-16_221.md`.

---

## GH-64 — a tabular-looking page falls to native with no flag, silently

**Status:** READY
**Branch:** `fix/64-tabular-native-flag`
**Write ownership:** `src/socr/core/born_digital.py`, `tests/` (a new module for this ticket)

### Context — confirmed still real on `main@ba92c19`

PP-6 narrowed the table routing gate to `has_numeric_columns` (lane REUSE across data
rows — see the GH-248/GH-348 docstring at `born_digital.py:3215-3228`, and do not
paraphrase it as mere co-occupancy). Correct change, but it has a side effect: a
**2-column whitespace-aligned borderless table** (>=15 rows, label|value, one numeric
lane per row) that the old `_detect_columnar_numbers` heuristic would have routed now
falls to the native prose path.

Values are char-exact; the row x column **grid structure** is not reconstructed — and it
happens **silently**. No flag, no audit event. The triage confirms there is no
`possible_table_structure_not_reconstructed` hook anywhere in `src/`.

The corpus invariant is *no silent content loss*. This is structure loss, so it must be
**visible**.

### Hard scope limits — read both

1. **Do NOT re-widen the routing gate.** The issue says so explicitly and PP-6 narrowed it
   deliberately to stop over-routing born-digital pages. This ticket adds a SURFACE, it
   does not change which lane a page takes.
2. **Do NOT demote document status.** Emit the event, the page-level flag and the CLI
   surface. GH-140 (merged tonight, `c61fd58`) added an audit flag and its demotion was
   removed after review measured the trigger rate at 36.1% of the free lane versus 2.4%
   for the precedent it copied. Nobody has measured this signal's rate either. If you
   believe demotion is warranted, say so in the decision log and leave it unimplemented.

### The threshold hazard — the main design risk here

"Looks tabular" invites an invented number, and this repo forbids those. **Reuse the
existing `_detect_columnar_numbers` heuristic as the predicate** — the issue names it,
it already encodes what "would have been routed before PP-6" means, and reusing it makes
the flag exactly "PP-6 changed this page's routing". Do not write a new ratio, a new
row-count cut, or a new lane threshold. If reuse turns out to be impossible, STOP and
report it as a design fork rather than inventing a cut.

### Acceptance Criteria
1. A born-digital page that the pre-PP-6 heuristic would have routed, and which now falls
   to native with no table handling, emits a durable audit event naming the structural
   loss. It reaches the page sidecar.
2. A page that genuinely has no table structure does NOT emit it. Test this as hard as
   criterion 1 — a flag that fires everywhere is noise, not a signal.
3. A page that IS routed to table handling does not emit it (no double-reporting).
4. **No new magic threshold.** Reuse the existing predicate.
5. Routing behaviour is byte-identical to before this change. Demonstrate it.

### Verification
- FULL suite: `PYTHONPATH=$PWD/src ~/venvs/socr/bin/pytest -q`. Never a `-k` subset.
- **Prove the guard by MUTATION, not deletion**: neuter the new detector so it never
  fires, re-run, and report which tests fail and which correctly still pass. A test that
  fails only with an ImportError proves a symbol is new, not that behaviour changed.
- **Pin a DIFFERENCE**: same page, flag-on vs flag-off, outcomes differ exactly as intended.
- `uvx ruff@0.16.0 format --check .` — not the venv's older ruff.
- Quote only numbers you actually ran; every figure is verified independently.
- COMMIT before reporting, staged by name, never `git add -A`.
- Do not `Closes #64` unless criteria 1-5 all hold.
- Decision log at `docs/log/2026-09-16_64.md`.
