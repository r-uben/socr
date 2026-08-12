# TICKETS — GH-147 landscape pages rowized on the wrong axis

Status keys: `TODO` · `WIP` · `DONE` · `BLOCKED`. `depends-on` gates dispatch.

Context: measured across the corpus — landscape pages are **1.92% of 22,979 pages
(441 pages) but 50% of the pages below 80% recall**. The rowizer clusters rows by
y; on a page whose text runs at 90° the rows run along x, so every emitted row is
a transposed slice. Output is incoherent, and the page still ships SUCCESS.

Decision taken from the measurement: **refuse, do not transform.** At 1.9% of pages
a coordinate transform is not yet warranted; routing to OCR is correct, cheap, and
fails closed. Revisit only if the refusal rate proves painful.

## Stream A — refuse

### TICKET-A1 — page-level rotated-text detection · TODO · depends-on: none · wave 1
**Problem:** Nothing currently records that a page's text runs non-horizontally.
**Do:** Compute the page's dominant text direction from its own line dirs and expose
it on `PageAssessment`. Derive from the page, never assume horizontal.
**Files:** `src/socr/core/born_digital.py`
**Done when:** `PageAssessment` for Fama p392 reports a non-horizontal dominant direction and an upright page in the same document reports horizontal.

### TICKET-A2 — refuse the native lane for rotated pages · TODO · depends-on: A1 · wave 2
**Problem:** A rotated page must not ship a transposed grid as trusted native text.
**Do:** When the dominant direction is non-horizontal, do not emit a reconstructed
table; mark the page for OCR routing and emit an `AuditEvent` naming the reason.
Prose on such pages should still be retained.
**Files:** `src/socr/core/born_digital.py`, `src/socr/pipeline/orchestrator.py`
**Done when:** Fama p392 emits no markdown table separator and carries an audit event of kind `landscape_page_refused`.

## Stream B — evidence

### TICKET-B1 — corpus-level regression figure · TODO · depends-on: A2 · wave 3
**Problem:** The claim "half the catastrophic pages are landscape" must stay true
after the change, and the refusal rate must be known.
**Do:** Re-run the corpus measurement; report how many pages are refused and what
happens to the below-80% population.
**Files:** `tests/test_landscape_refusal_gh147.py`, `logs/`
**Done when:** a log records refused-page count and the new below-80% distribution.
