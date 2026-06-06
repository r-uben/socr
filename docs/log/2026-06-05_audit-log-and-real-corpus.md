# Dev log — Audit log + first real-corpus validation of dual-pass (2026-06-05)

Branch: `feat/audit-log` (off `main`, which now holds the merged unified +
qwen + dual-pass train, PR #29). 518 tests pass.

## What shipped
- **Durable per-run audit log** (`src/socr/core/audit_log.py`). Writes
  `audit_log.json` next to the output every run (non-fatal, only when there are
  events). Two sources merged: judge rejections and dual-pass reconciliations
  appended at the source with rich detail (named issues, exact changed cells), +
  escalations derived from per-page attempt `FailureMode` enums (RECITATION called
  out by its own kind). Makes batch runs inspectable. `DocumentState.events`
  stream; written in `_phase_assemble`. 6 tests.
- **Booktabs localizer fix** (`fix: reject out-of-page frame rules`). See findings.

## First real-corpus run (the point of the session)
Target: Fama-French 1997 "Industry Costs of Equity" (JFE), born-digital so the
native text is ground truth. 3 dense table pages (p7/p8/p13) excerpted; run with
`--no-native-first` to force the VLM path so dual-pass engages.

### Finding 1 — Fama-French tables are pure booktabs (design hypothesis confirmed)
Every table page: `find_tables(lines)` returns **0 tables**; the pages carry 7-12
horizontal rules. PyMuPDF's default detector would miss all of them; the rule-band
detector is the only thing that localizes them. This is exactly the case dual-pass
was built for.

### Finding 2 — Fail-open holds under real failure (validated twice)
- Run A (judge model = default ladder -> `qwen3.5:cloud`): Ollama cloud passthrough
  returned **502 Bad Gateway** / timeouts on every judge + crop call.
- Run B (judge model pinned to local `qwen3-vl:8b`): the judge got **empty output**
  (no JSON) and the crop reads **timed out (>120s)** on dense full pages.
- Both runs: every page kept, no crash, `Success` in ~20s, **no `audit_log.json`**
  (infra errors are not quality events — correct). The designed degradation works.

### Finding 3 — Booktabs localizer is not robust on dense real pages
Diagnosed on the real rule geometry:
- **Frame / crop-mark rules** drawn at `y < 0` and `y > page height` stretched the
  band to the whole page with negative coords. **Fixed**: filter rules to the page
  rect, clamp the band. Regression test added.
- **Column fragmentation**: with mixed-width rules, the x-overlap grouping splits
  one wide table into several column-bands.
- **Multi-table pages over-merge**: p13 is prose with two small coefficient tables;
  they merge into one band (the gap between them is prose). Rule gaps cannot
  distinguish this from a single tall table — p8 is one full-page table with a
  *larger* internal gap (238pt) than p13's between-table gap (160pt). So no
  gap-split heuristic is safe.
- All of this **fails safe**: an imprecise crop hits the reconciler's column-count
  check and is flagged, never patched. Cost is "no benefit", not corruption.

### Finding 4 — Crop-read VLM is the practical bottleneck on this hardware
`minicpm-v` read a real (whole-page) crop in 27s but collapsed the 3 factor
sub-columns into one cell (`<br>`-separated) -> column mismatch -> would flag, not
patch. `qwen3-vl:8b` times out on dense crops (consistent with the qwen-session
finding). `qwen3.5:cloud` is flaky (502). So even with a good bbox, getting a
faithful, structurally-matching crop reading on a dense real table is unreliable.

## Honest bottom line
Dual-pass's mechanism is validated on clean tables and **never corrupts output**
(safety rails hold on every real failure mode seen). But its **practical hit-rate
on the owner's dense real pages is currently low**, gated by two things the
real-corpus run exposed: (a) the rule-band localizer needs content-aware region
detection to produce precise crops on dense / multi-table pages, and (b) the
crop-read VLM is slow/unreliable on a 64GB Mac + flaky cloud. This echoes the
qwen-session panel verdict ("near ceiling for page-level OCR on this hardware").

## Image-based table localization (built this session)
Vector detectors read `get_drawings()`, empty on a true scan (confirmed: no
scanned page in the pre-1996 set exposes one vector rule), so dual-pass was
structurally blind on its real target. Added `image_locate.py`: connected-
component horizontal-rule detection on the rendered raster. The hard part the
vector case lacked is separating a drawn rule from a justified-prose text row;
three discriminators (thin / solid >=55% ink / wide >=40%) cleanly separate them
- validated: a dense table page yields its rules, three REAL scanned prose pages
yield zero false positives. Gated on the scanned signature (image + no vectors)
and only when vectors found nothing. Reuses `bands_from_rules`; fail-open without
opencv. 522 tests pass.

## Corpus-composition findings that reframe the whole feature
Investigating real scans surfaced two facts that matter more than any code:
1. **The owner's corpus is overwhelmingly born-digital.** Only ONE scanned paper
   in the entire pre-1996 set (Christiano-Eichenbaum 1992, table-less theory). No
   scanned paper has a rule-dense (ruled-table) page. So the scanned slice that
   image-localization serves is, on this corpus, nearly empty.
2. **The real defect is structure-loss, not VLM corruption.** On born-digital
   booktabs tables (the common case) `find_tables` returns 0, so `extract_
   structured` cannot build a grid, and PyMuPDF native extraction yields correct
   VALUES but a flat 1-D token stream (`Industry / b / b / s / h / FabPr / 0.253
   / ...`) - the numbers are char-exact, the table grid is gone.

**Implication / recommended pivot (not yet taken):** the highest-value table work
is not "catch a VLM corrupting a table" (rare here) but "restore grid structure to
native-extracted born-digital booktabs tables." The safe design: crop the table,
use a VLM for LAYOUT ONLY, and anchor every cell to the char-exact native value
(native = ground truth, so the VLM never supplies a number - no silent corruption,
which sidesteps codex's crop-fidelity objection entirely). Owner chose to finish
image-localization first; this pivot is the standout candidate for next.

## Auto-patch default — codex verdict (pending decision)
Codex (gpt-5.5) strongly recommends flipping AUTO-PATCH -> FLAG-ONLY by default:
column-count agreement is not real protection (a model can keep table shape and
still change 0.031->0.037), and a silent wrong patch to a research number is worse
than a missed correction. Make auto-patch opt-in (`--auto-patch-tables`) until the
crop reader is proven on held-out ground truth. The graceful-automatic path
(per-cell confidence via crop self-consistency / zoom-in tiebreak / native anchor)
all depends on a fast reliable crop reader first. NOT yet implemented.

## Next (owner's call)
1. **Content-aware table-region detection** — bound a table by its columnar-numeric
   content, not just rules; would fix fragmentation + multi-table + missing-bottom
   -rule. The real unlock for dual-pass on dense pages. Sizeable.
2. **Reliable crop model** — a fast, structure-faithful table VLM. Re-test
   `minicpm-v` with a structure-preserving prompt; consider a dedicated table model.
   Cheaper than #1, addresses Finding 4.
3. Ship audit-log + localizer fix as-is (PR `feat/audit-log` -> main); treat
   dual-pass as best-effort (helps on clean table pages, no-ops safely elsewhere).
4. A real scanned paper (no native text) is the other untested case — the excerpt
   trick forced VLM on born-digital; a genuinely scanned table page is the true
   target.
