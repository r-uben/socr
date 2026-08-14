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

### TICKET-B1 — corpus regression on the refusal predicate, with a structural witness · TODO · depends-on: A2 · wave 3b
**Problem:** The claim "half the catastrophic pages are landscape" must stay true
after the change, and the refusal rate must be known.

⚠️ **RETARGETED 2026-08-13 by a wave-3 ruling. The original metric is invalidated by our own
fix.** The ticket measured landscape damage via below-80% **word recall**. But GH-147 A2
(merged `13033a3`) never calls `extract_structured` on a born-digital rotated table page — it
sets `native_text = raw_text.strip()` (`born_digital.py:915-931`) and flags
`native_table_lane_refused`. Word-multiset recall against the raw page is therefore **~1.0 by
construction**, and the 20/40 figure cannot stay true after a *correct* fix. The old criterion
would fail the ticket for succeeding.

**Do:** Two arms.

*Corpus arm.* After a metadata preflight, copy every non-zero corpus PDF to `/tmp` and open
only those copies through the installed `BornDigitalDetector`. Log:
- opened-PDF count, and a per-file path / size / sha256 manifest;
- the **45 zero-byte inputs enumerated by path as excluded-dead** — so 362/407 cannot
  masquerade as 407/407;
- refusal count and rate under `native_table_lane_refused`;
- the historical broad-landscape count beside it;
- the below-80% word-recall split, **labelled as native-lane detector output only**.

*Witness arm.* Ship `tests/fixtures/landscape_refusal_gh147/` (generator, paired
rotated/upright PDFs, `ground_truth.json`) and `tests/test_landscape_refusal_gh147.py`:
`extract_structured` on the rotated render must show a grid whose column count tracks
ground-truth **rows**; `detect_page` / `_assess_page` must then suppress that grid, set
`native_table_lane_refused`, and keep the word multiset — while the upright twin stays
unrefused and column-aligned. **Density is load-bearing:** use whatever authored geometry
actually emits a pre-refusal grid (12×14 is the starting candidate). A sparse grid that emits
none makes the before-arm vacuous.

**Binding constraints from the ruling:**
- A permanently dead 0-byte library file is **not** a hard veto on the ticket. Abort only on an
  evicted/absent input you would otherwise score.
- Do **not** copy the design pass's 263 / 91 / 172 / 0.9988 figures into the log as results.
  Re-run the measurement and publish only numbers from that run's manifest. Those are method,
  not evidence.
- Keep the below-80% table in the log, but it is **not** the pass/fail criterion. The criterion
  is refusal rate plus structural grid suppression. State plainly that refused-page word recall
  is ~1.0 by construction at `born_digital.py:924`.
- Label every recall figure as native-lane detector output. Do not claim OCR recovery or
  final-document fidelity — that needs a provider-backed `process()` run this ticket does not own.
- The before-arm may call `extract_structured` only as *the path A2 skips*, labelled as that
  rung. The after-arm must be `detect_page` / `detect`, never a standalone `reconstruct.py` import.
- Assert a **relational** shape (emitted columns nearer ground-truth rows than columns), not
  brittle exact markdown dimensions. Name GH-152 in the docstring — `reconstruct.py` moves
  under this ticket.
- Do not narrow the refusal predicate, and do not count routing-only cost as a code change.

**Files:** `tests/test_landscape_refusal_gh147.py`, `tests/fixtures/landscape_refusal_gh147/`,
`docs/plans/gh147-landscape-pages/logs/`
**Done when:** a dated log records the refused-page count and rate, the opened-PDF count, the
per-file manifest, the named zero-byte exclusions, and the below-80% distribution labelled as
detector output. **Acceptance is not a green suite:** revert A2 with
`git stash push -- src/socr/core/born_digital.py` and the rotated after-arm (no separator /
`native_table_lane_refused` / no shipped grid) must **FAIL**, while the upright control passes
in both states.
**Coordinator follow-up:** this plan's `STATUS.md` still carries the stale 20/40 headline.

**Corpus note:** the library exists in two places and neither is complete alone — iCloud
(`~/Library/Mobile Documents/com~apple~CloudDocs/Library/Papers/papers`, 407 PDFs, 45 evicted
to 0-byte placeholders) and ProtonDrive (`~/Library/CloudStorage/ProtonDrive-*/Papers`, 277
PDFs, essentially all real). **All 45 iCloud placeholders have a real copy in ProtonDrive**, so
the union covers all 407. Google Drive holds a third archive copy but must **not** be read
from: it is kept quit by design and streams rather than stores. Never open a PDF in place from
any of them — copy to `/tmp` first and verify the byte size.

