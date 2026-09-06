# TICKETS — defect-census fixes (trustworthy output on institutional corpora)

Source: `docs/log/2026-09-06_fed-defect-census.md` (Fed fed-01, 766 docs; ECB 30-page
sample on `main@eb14c82`). Owner ruling 2026-09-06: when the header-attribution guard
abstains but the candidate's rows match the native text layer in order, ship the table
flagged "header binding unverified" instead of failing closed; only where a native layer
exists; the flag surfaces at page, document, metadata and CLI.

Every rule here is about documents, never about an institution. No magic thresholds:
any constant is named, documented, and sourced to a measurement in the census log.
Tests pin a DIFFERENCE (same run with/without the new path), never a locally measured
absolute (CLAUDE.md). CI has no provider — patch `_available_engines_for_agentic`.

Fixtures (do not copy into the repo; reference by path):
- ECB: `~/Data/socr/census-ecb-2026-09-06/in/*.pdf`, outputs + cached candidates in `out/`.
- Fed: `~/Data/socr/fed-sample-2026-09-05/in/fed-1989-11-14-minutes.pdf`;
  re-run on main at `~/Data/socr/census-591-recheck/`.
- Fed scan text page: `fed-01/.../fed-meetings-1969-1969-05-1969-05-27-minutes` p1.

Status keys: `TODO` · `WIP` · `DONE` · `BLOCKED`. `depends-on` gates dispatch.
Parallelizable = no shared files / no dep. Each ticket = one implementer agent, then one
reviewer pass, then the Astra (Codex pane) check, before commit. `pipeline/orchestrator.py`
is shared by most tickets — never two orchestrator tickets in one wave.

## Stream A — selection: ship a corroborated candidate instead of failing closed

### TICKET-A1 — native row corroboration reopens a guard-abstained candidate · TODO · depends-on: none · wave 1
**Problem:** 9/9 ECB statistical pages: `native_table_verifier_warn ambiguous_lane_count_mismatch`
→ `table_output_defect == header_unattributed` → structure-class floor → marker, while a
cached candidate reproduced 98–100% of the page's numbers with row order intact (three of
them ladder-ACCEPTED, #589). Shipped 1–4%.
**Do:** In the structure-class selection (orchestrator S1 branch + `structure_class_grid_winner`),
before the floor fires, score every cached grid candidate with an *ordered row match*
against the page's native words: a candidate row is bound iff its ordered numeric run occurs
contiguously on one native baseline band (reuse `tables/binding.py` `parse_grid`,
`_assign_bands`; do not write a second rowizer). If the page has a native layer and the best
candidate's bound-row share clears `ROW_CORROBORATION_MIN` (named constant; source: census
min 36/39 on a page with wrapped labels — document it, and record the measured share in the
sidecar so the constant can be revisited), ship that candidate with page status WARNING,
`failure_mode = header_binding_unverified`, a `table_row_corroborated` audit event carrying
`{bound, total, engine}`, and the flag in `tables_trust.json`, metadata, and the CLI
summary line. No native layer (scan) → unchanged fail-closed. A ladder-ACCEPTED candidate
that the floor still discards must emit `structure_floor_overrode_ladder` (closes #589 (c)).
**Files:** `src/socr/pipeline/orchestrator.py` (structure-class branch only),
`src/socr/core/manifest.py` (`structure_class_grid_winner`), new
`src/socr/tables/row_corroboration.py`, `src/socr/core/result.py`, `src/socr/core/tables_trust.py`,
`src/socr/core/audit_log.py`, `tests/tables/test_row_corroboration.py`,
`tests/pipeline/test_structure_floor_corroboration.py`.
**Done when:** `socr replay` is not applicable (new model path) — instead: run
`socr process` on `census-ecb-2026-09-06/in/ecb-reports-2003-report-p80-82.pdf` and
`…economic_bulletin-p127-129.pdf` on the branch; every one of the 6 pages ships ≥ 95% of
`pdftotext -layout` numbers (the census scorer in the log) with `failure_mode
header_binding_unverified` in `pages/NNNNN.json`, the page listed in `tables_trust.json`,
and the CLI summary naming the count. Unit test: same candidate set, flag on vs off,
asserts the winner differs exactly on the corroborated page and is identical on a scanned
page (no native words). `uvx ruff@0.16.0 format --check .` clean.

### TICKET-A2 — a truncated candidate never beats a complete one · TODO · depends-on: A1 · wave 2
**Problem:** bulletin p3: qwen's output ends mid-number (`| 2019 | 364.2 | 7,05`), 34/389
numbers + 55 wrong, shipped as WARNING over gemini's complete, ladder-ACCEPTED 389/389.
**Do:** Add `DEFECT_TABLE_TRUNCATED` to `table_output_defect`: last table row unterminated
(no closing `|`) or numeric-row count below the native effective row count by more than the
wrapped-label allowance already used by `value_guard_row_count_warning`. A truncated candidate
is ineligible as grid winner whenever any other candidate for the page is not truncated;
emit `candidate_truncated {engine}`. Never silently drop it if it is the only candidate.
**Files:** `src/socr/tables/structure_check.py`, `src/socr/core/manifest.py`,
`src/socr/core/audit_log.py`, `tests/tables/test_structure_check_truncated.py`.
**Done when:** bulletin p3 fixture ships the gemini candidate (389 numbers, 0 not-in-source);
unit test with a (truncated, complete) pair picks complete, and with (truncated) alone still
ships it flagged. ruff clean.

## Stream B — fail-closed marker scope (#591)

### TICKET-B1 — the marker replaces the table region, not the page · TODO · depends-on: A1 · wave 2
**Problem:** the `page_failed` ending ships the marker alone (ECB survey-2013 p1: question
prose lost, words 0.00; Fed 1989-11-14 p3 directive prose per #591); the floor ending keeps a
shredded native layer. Two endings, both wrong.
**Do:** On both endings, emit native prose lines whose bbox lies outside every
`detected_table_bboxes` entry, in reading order, with the marker + page image in the table's
place. One assembly function shared by both endings.
**Files:** `src/socr/pipeline/orchestrator.py` (marker assembly), `src/socr/core/manifest.py`,
`tests/pipeline/test_fail_closed_marker_scope.py`.
**Done when:** survey-2013 p1 and Fed 1989-11-14 p3 ship ≥ 0.9 word recall vs `pdftotext`
restricted to text outside the table bbox, marker present, table region still withheld;
golden byte-identity tests unchanged for pages with no marker.

## Stream C — native prose geometry (#592)

### TICKET-C1 — baseline-aligned adjacent blocks are one line · TODO · depends-on: none · wave 1
**Problem:** a tab-aligned two-column run (`Mr.` column, name column) is emitted block by
block: 12 bare honorifics, then 12 names; 174 Fed pages, all SUCCESS; reproduces on main
(`census-591-recheck` p1). Geometric, not lexical: no `PRESENT`/`Mr.` patterns.
**Do:** In `core/born_digital.py` line assembly from `get_text("dict")` blocks: when two
blocks are x-disjoint and a line in one shares its baseline band (y-overlap above the
existing band tolerance) with a line in the other, merge those lines left-to-right into one
output line. Applies to any page; must not merge true multi-column prose (columns whose
lines do not pairwise share baselines across the whole run — require the run length to
match, as a list does and a two-column article does not).
**Files:** `src/socr/core/born_digital.py`, `tests/core/test_born_digital_two_column_runs.py`.
**Done when:** Fed 1989-11-14 p1 emits `Mr. Greenspan, Chairman` as one line and 0 bare
honorific lines; a two-column journal-article fixture (existing test corpus) is byte-identical
before/after; all golden tests pass.

## Stream D — throughput

### TICKET-D1 — nougat leaves the automatic ladder · TODO · depends-on: none · wave 1
**Problem:** nougat (academic-paper model) sits in the free tier of `provider_ladder`; on a Mac
it burns 6+ CPU minutes per rejected table page and produced nothing on the census.
**Do:** `PROFILE_NOUGAT.auto_eligible = False` (reachable via `--primary nougat` like deepseek);
document the reason in the profile.
**Files:** `src/socr/core/providers.py`, `tests/core/test_providers.py`.
**Done when:** `provider_ladder()` default excludes nougat; `--primary nougat` still resolves;
test pins the difference.

### TICKET-D2 — measure the 5–7 min route phase · TODO · depends-on: none · wave 3
**Problem:** `timings_s.route` is 317–420 s on ECB statistical pages; nothing in route should
call a model. Unknown cost centre.
**Do:** Instrument route sub-stages (exclusive seconds) on the census fixtures; write
`docs/log/<date>_route-phase-cost.md` naming the top cost. Measurement only; no fix.
**Files:** `src/socr/pipeline/orchestrator.py` (route timers), `docs/log/`.
**Done when:** the log attributes ≥ 80% of route seconds to named sub-stages on the 6 fixture pages.

## Stream E — noise that erodes trust

### TICKET-E1 — a page-sized raster with a text layer is the scan, not a chart · TODO · depends-on: B1 · wave 3
**Problem:** #511 large half: `has_chart_marks` fires on 97% of scanned Fed pages; the "chart"
PNG is the page itself. No prose lost, but a phantom asset per page and `audit.json`
`chart_pages` disagrees with the markdown 20×.
**Do:** In `has_chart_marks` / `_is_chart_asset_page`: a raster whose placed area covers
≥ the page's text-layer bbox and whose page has native words inside that raster is the
scan itself → not a chart. Keep the small-raster gate (#510) as is.
**Files:** `src/socr/figures/extractor.py`, `src/socr/pipeline/orchestrator.py`,
`tests/figures/test_has_chart_marks_scan_raster.py`.
**Done when:** Fed 1969-05-27 p1 fixture emits no chart mark; an existing true-chart fixture
still does; test pins the difference.

### TICKET-E2 — `table_not_scorable` only on pages with a detected table · TODO · depends-on: E1 · wave 4
**Problem:** every prose page of a transcript is flagged "untrusted tables" (3/3 on ECB
transcript; 400 events / 68 docs on Fed). A false trust flag on pages with no table.
**Do:** Emit `table_not_scorable` only when `detected_table_count > 0`; otherwise no event.
**Files:** `src/socr/pipeline/orchestrator.py`, `src/socr/core/tables_trust.py`,
`tests/pipeline/test_table_not_scorable_scope.py`.
**Done when:** transcript excerpt yields `untrusted_page_count == 0`; a table page still flags.

## Stream F — one-table Fed items (last)

### TICKET-F1 — ditto marks resolve to a derived value, flagged · TODO · depends-on: A2 · wave 4
**Problem:** #625, 240 rows / 16 Fed docs; faithful but unparseable.
**Do:** Post-emission normaliser: a cell that is only a ditto glyph becomes the value above
with `derived: ditto` recorded per cell in the sidecar; never silent.
**Files:** `src/socr/core/normalizer.py`, `tests/core/test_ditto.py`.
**Done when:** 1979-11-20 fixture rows carry the value and the flag; test pins the difference.

### TICKET-F2 — `&nbsp;` indentation becomes hierarchy · TODO · depends-on: F1 · wave 4
**Problem:** #624, 104 cells / 11 docs.
**Do:** Normalise leading `&nbsp;` runs in a label cell to a documented hierarchy marker.
**Files:** `src/socr/core/normalizer.py`, `tests/core/test_nbsp_hierarchy.py`.
**Done when:** fixture rows carry the marker; no `&nbsp;` in output; test pins the difference.

Parked, not tickets: hyphenated line breaks (43,734 lines; owner remit ruling pending);
chart-axis ticks dropped on qwen-lane chart pages (gist-only figures).
