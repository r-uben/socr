# TICKETS — defect-census fixes (trustworthy output on institutional corpora)

Source: `docs/log/2026-09-06_fed-defect-census.md` (Fed fed-01, 766 docs; ECB 30-page
sample on `main@eb14c82`; #591/#592 re-verified on main). Owner ruling 2026-09-06: when the
header-attribution guard abstains but the candidate's rows match the native text layer in
order, ship the table flagged "header binding unverified" instead of failing closed; only
where a native layer exists; the flag surfaces at page, document, metadata and CLI.
Panel 2026-09-06 (GPT, DeepSeek, Kimi; Gemini and Grok seats failed at transport/quota):
findings folded in below; log in `logs/2026-09-06_panel.md`.

Every rule here is about documents, never about an institution. No magic thresholds: any
constant is named, documented, and sourced to a measurement recorded in the census log or
in the ticket's own log. Tests pin a DIFFERENCE (same inputs, new path on vs off), never a
locally measured absolute. CI has no provider: any test through `_phase_agentic` patches
`_available_engines_for_agentic` and sets `_resolve_judge_model` to `""`. Every new audit
event kind goes into both the `audit_log.py` rank dict and the `tables_trust.py` distrust
set (or it falls to default rank).

Fixtures (reference by path; never copied into the repo):
- ECB: `~/Data/socr/census-ecb-2026-09-06/in/*.pdf`; outputs and cached candidates in
  `out/<stem>/cache/`; scorer = the census log's numeric-multiset + ordered-row match.
- Fed: `~/Data/socr/fed-sample-2026-09-05/in/fed-1989-11-14-minutes.pdf`; re-run on main at
  `~/Data/socr/census-591-recheck/` (p1 = #592 shape, p3 = #591 shape).
- Fed scan text page: `fed-01/fed-meetings-1969-1969-05-1969-05-27-minutes` p1 (#511 shape).
- Two-column prose negative control: name one page from `tests/` fixtures in C1's log
  before coding (an existing born-digital journal page with balanced columns).

Status keys: `TODO` · `WIP` · `DONE` · `BLOCKED`. `depends-on` gates dispatch. Same wave ⇒
disjoint files. Each ticket = one implementer, one reviewer pass, then the Astra (Codex
pane) check, before commit. `pipeline/orchestrator.py` and `core/manifest.py` are shared by
most tickets: never two tickets touching either in one wave.

Out of scope, by name: #215 (header reject term), #270 (cell fabrication), #393 (rotated
pages fail closed by construction until the rotation lift lands), #249 (axis ticks on
qwen-lane chart pages), hyphenated line breaks (owner remit ruling pending), landscape
refusals (25 pp / 21 Fed docs; #263 closed — re-measure in D3 before reopening).

## Stream A — selection: ship a corroborated candidate instead of failing closed

### TICKET-A1a — `corroborate_rows()`: ordered row match against native words · TODO · depends-on: none · wave 1
**Problem:** the repo's `bind()` (`tables/binding.py:1323`) binds by multiset anchors and
interpolation; nothing checks that a candidate row's ordered numeric run sits contiguously
on one native baseline. That is the mechanical corroboration the census used and the
ruling relies on. `parse_grid()` handles only the first table (`binding.py:152`);
`_assign_bands` keys on `round(y0)` (#600) and is private.
**Do:** New `src/socr/tables/row_corroboration.py` with one public function
`corroborate_rows(words, markdown, region) -> RowCorroboration` returning
`bound`, `total` (numeric body rows only: header, delimiter and value-less rows excluded),
`extra_numbers` (candidate numbers absent from the region's native words), and
`native_numeric_rows`. Region-scoped: words outside `region` are dropped before banding;
every markdown table block is handled, not just the first. Bands are baseline bands from
word y-centres with a tolerance derived from the region's median word height (no
`round(y0)`; note #600). Constants: `ROW_CORROBORATION_MIN` — a share, inclusive, value
36/39 as measured on the census (bulletin p2, wrapped labels) — and
`EXTRA_NUMBERS_MAX_SHARE` (not-in-source numbers / candidate numbers), measured on the
six ECB fixture pages and recorded in this ticket's log before the constant is set. Gate
semantics documented in the module: `total == 0` → abstain (no evidence), never clears.
**Files:** `src/socr/tables/row_corroboration.py`, `tests/tables/test_row_corroboration.py`,
`src/socr/benchmark/replay_binding.py` (freeze inputs and re-run the scorer for bit-identity).
**Done when:** `~/venvs/socr/bin/pytest tests/tables/test_row_corroboration.py -q` exits 0
with cases: perfect page; wrapped-label page (36/39 shape) clears; zero-numeric native page
abstains; candidate with all rows bound plus fabricated rows fails on `EXTRA_NUMBERS_MAX_SHARE`;
two-table page scores each table against its own region. Running the function over the six
ECB fixture candidates (log the command) reproduces the census bound/total within ±1 row per
page; the log records each page's share. `uvx ruff@0.16.0 format --check .` clean.

### TICKET-A1b — selection consults corroboration before the floor · TODO · depends-on: A1a · wave 2
**Problem:** `structure_class_grid_winner` (`core/manifest.py:853`) is a pure function over
`PageState`, which carries no words (`core/state.py`); the S1 branch has nothing to score
against. 9/9 ECB statistical pages floored over a candidate that reproduced the page.
**Do:** Store the page's native words (and `detected_table_bboxes`) on `PageState` at
extraction time. In `structure_class_grid_winner`, when the structure guard abstained
(`header_unattributed`) and native numeric words exist inside a detected table bbox, score
every cached grid candidate with `corroborate_rows` on that region; a candidate clearing
both constants and whose numeric row count is not below the native effective row count by
more than the existing wrapped-label allowance (name the symbol from
`value_guard_row_count_warning`; this closes the A1→A2 window) becomes the winner. Tie-break
by ladder verdict, then share. A1a's `RowCorroboration` exposes only counts; A1b extends it
with `unbound_rows` (candidate row indices that did not bind) so that a table clearing the
share floor with one wrong row (a single altered digit still clears at 36/39 on a 39-row
table — measured, A1a log) ships with THAT row named in the sidecar and marked in the
markdown (a trailing `<!-- row unverified -->` comment on the row), never silently.
A1a's skipped-band gate is blind to a dropped FIRST or LAST row of a block (no straddling
bound pair; measured on bulletin p1 with the whole page as region). A1b scopes the region to
the detected table bbox and adds an edge check: native numeric bands inside the bbox before
the first matched band or after the last one count as skipped unless they are the
column-index line; measure on the six fixtures before setting anything. No native numeric
words in the region (scan, image-only table region) → unchanged fail-closed. A ladder-ACCEPTED candidate the floor still discards
emits `structure_floor_overrode_ladder` (closes #589 via option (c), generalised). Keep the
candidate's header row as emitted; the flag carries the doubt (owner may rule for neutral
`col N` stubs later — record the header text in the sidecar either way so the choice is
reversible).
**Files:** `src/socr/core/state.py`, `src/socr/core/manifest.py`,
`src/socr/pipeline/orchestrator.py` (extraction: populate words; nothing else),
`src/socr/core/audit_log.py`, `src/socr/core/tables_trust.py`,
`tests/test_s1_structure_class_winner_corroboration.py`.
**Done when:** unit test builds one `PageState` with a floored native branch and two cached
candidates (one corroborating, one not) and asserts the winner differs exactly when the
corroboration path is on vs off, and is identical on a state with no native words. The
existing floor fixtures (`tests/test_gh317_structure_class_floor.py`,
`tests/test_s1_structure_class_winner_gh_reachability.py`, title-only text layers) still
floor. `pytest tests/ -q` green.

### TICKET-A1c — the corroborated page surfaces at every level; resume contract · TODO · depends-on: A1b · wave 3
**Problem:** a corroborated table must ship as a doubt, visible everywhere; and a WARNING
page is non-terminal (`_load_terminal_page` requires SUCCESS + `audit_passed`,
`orchestrator.py:8939-8956`), so it re-runs the model on every resume.
**Do:** Page status WARNING, `failure_mode = header_binding_unverified`, sidecar carries
`{bound, total, extra_numbers, engine, share}`, audit event `table_row_corroborated`,
`tables_trust.json` entry, document status and metadata roll-up, CLI summary line naming the
count. Resume contract, stated in the sidecar and docs: corroborated pages are reprocessed on
resume (not skipped) and their bytes are not bit-stable across runs; `floor_shipped` keys
must not grant a skip to this failure mode.
**Files:** `src/socr/core/result.py`, `src/socr/core/tables_trust.py`,
`src/socr/core/audit_log.py`, `src/socr/pipeline/orchestrator.py` (summary + resume gate),
`src/socr/cli.py`, `tests/pipeline/test_header_binding_unverified_surfacing.py`.
**Done when:** hermetic test (ladder patched, judge model `""`) drives one page through
`process()` twice with the path on vs off and asserts: sidecar `failure_mode`, `tables_trust`
entry, metadata roll-up and captured CLI line all differ exactly as specified; resume test
asserts the corroborated sidecar is reprocessed. Then on the branch, with Ollama:
`PYTHONPATH=src ~/venvs/socr/bin/socr process ~/Data/socr/census-ecb-2026-09-06/in/ecb-reports-2003-report-p80-82.pdf -o /tmp/a1c/`
and the bulletin excerpt; the census scorer command (copied into the log) reports ≥ 95% numbers
on all 6 pages, each with `header_binding_unverified`.

### TICKET-A2 — a truncated candidate never beats a complete one · TODO · depends-on: A1c · wave 4
**Problem:** bulletin p3: qwen's output ends mid-number (`| 2019 | 364.2 | 7,05`), 34/389
numbers + 55 wrong, shipped WARNING over gemini's complete, ladder-ACCEPTED 389/389.
**Do:** `DEFECT_TABLE_TRUNCATED` in `table_output_defect`, evaluated on raw rows *before*
strict parsing discards malformed rows: final table row unterminated relative to the
candidate's own style (mixed terminated rows + unterminated last row), or numeric-row count
below native effective rows by more than the named wrapped-label allowance. A truncated
candidate is ineligible as grid winner whenever any other candidate for the page is not
truncated; audit event `candidate_truncated {engine}`. If it is the only candidate it still
ships, flagged.
**Files:** `src/socr/tables/structure_check.py`, `src/socr/core/manifest.py`,
`src/socr/core/audit_log.py`, `src/socr/core/tables_trust.py`,
`tests/tables/test_structure_check_truncated.py`.
**Done when:** deterministic test (no provider) with a (truncated, complete) candidate pair
picks complete and with (truncated) alone ships it flagged; a candidate with all rows
unterminated (its own style) is not truncated. On the branch the bulletin p3 fixture ships
the gemini candidate (389 numbers, 0 not-in-source, log the scorer command).

## Stream B — fail-closed marker scope (#591)

### TICKET-B1 — the `page_failed` ending keeps prose outside the table · TODO · depends-on: A2 · wave 5 · closes #591
**Problem:** the structure-class floor already splices prose around detected tables behind a
four-condition coverage guard (`core/manifest.py:938-1023`, GH-520). The `page_failed`
ending does not: ECB survey-2013 p1 and Fed 1989-11-14 p3 shipped the marker alone
(words 0.00). One ending is fixed, the other is #591.
**Do:** Route the `page_failed` ending through `table_floor_text_for_source` (extend, no
sibling) under the same four-condition guard; when the guard fails, whole-page marker as
today. Add the two bbox sanity checks the panel asked for: a bbox that leaves native numeric
rows outside every table (too small) or covers prose lines with no numeric tokens beyond the
table's own rows (too large) fails the guard.
**Files:** `src/socr/pipeline/orchestrator.py` (page_failed assembly), `src/socr/core/manifest.py`,
`tests/pipeline/test_page_failed_marker_scope.py`.
**Done when:** test drives one page_failed page with the guard satisfied vs violated and
asserts prose present vs whole-page marker. On the branch, survey-2013 p1 and
fed-1989-11-14 p3 ship ≥ 0.9 word recall vs `pdftotext` restricted to text outside the table
bbox, marker present, table region withheld. Golden byte-identity tests unchanged.

## Stream C — native prose geometry (#592)

### TICKET-C1 — baseline-aligned adjacent blocks on a prose page are one line · TODO · depends-on: none · wave 1 · closes #592
**Problem:** a tab-aligned two-column run (`Mr.` column, name column) is emitted block by
block: 12 bare honorifics, then 12 names; 174 Fed pages, all SUCCESS; reproduces on main.
The prose path returns `page.get_text("text")` before any block walk
(`core/born_digital.py:2195`), so the seam is the no-table path.
**Do:** A shared line assembler used by the no-table path (and available to the table path):
walk `get_text("dict")` lines; when two x-disjoint blocks have a bijection of lines sharing
baseline bands over the whole run, and the horizontal gap between them is comparable to a
word space rather than a column gutter (`ALIGNED_RUN_GAP_MAX_WORD_SPACES`, measured on the
Fed 1989 p1 list and on the named two-column journal page and recorded in the ticket's log
before the value is set), merge each pair left-to-right into one line. Balanced justified
prose columns have a gutter many word spaces wide and are never merged.
**Files:** `src/socr/core/born_digital.py`, `tests/core/test_born_digital_aligned_runs.py`.
**Done when:** differential test runs the Fed 1989 p1 text layer through the assembler on
vs off and asserts `Mr. Greenspan, Chairman` is one line with 0 bare honorific lines when on;
the named two-column journal fixture is byte-identical on vs off; all golden tests pass.

## Stream D — throughput and measurement

### TICKET-D1 — nougat leaves the automatic ladder · TODO · depends-on: none · wave 1
**Problem:** nougat sits in the free tier of `provider_ladder` (`core/providers.py:195`); on a
Mac it burns 6+ CPU minutes per rejected table page and won the Fed 1989 p3 page with a
hallucination.
**Do:** `PROFILE_NOUGAT.auto_eligible = False`, reachable via `--primary nougat`; reason in the
profile docstring.
**Files:** `src/socr/core/providers.py`, `tests/core/test_providers.py`.
**Done when:** `provider_ladder()` default excludes nougat; `--primary nougat` still resolves;
test pins the difference.

### TICKET-D2 — measure the 5–7 min route phase · TODO · depends-on: B1 · wave 6
**Problem:** `timings_s.route` is 317–420 s on ECB statistical pages; route should call no model.
**Do:** Instrument route sub-stages (exclusive seconds) on the six ECB fixture pages; write
`docs/log/<date>_route-phase-cost.md` naming the top cost. Measurement only.
**Files:** `src/socr/pipeline/orchestrator.py` (route timers), `docs/log/`.
**Done when:** the log attributes ≥ 80% of route seconds to named sub-stages.

### TICKET-D3 — re-measure the Fed table lane on main · TODO · depends-on: A2 · wave 5
**Problem:** the Fed table-lane counts in the census come from `6fa89d9` (349 commits behind,
heuristic judge); the plan's claim to fix institution 1 is unverified until re-measured.
**Do:** Sample 12 table-bearing fed-01 docs (stratified by decade), run on the wave-4 branch
with the pinned-tree protocol, score with the census scorer, and log landscape refusals
(#263 closed) in the same pass.
**Files:** `docs/log/<date>_fed-table-lane-remeasure.md` only (benchmark scripts under
`src/socr/benchmark/` if reused).
**Done when:** the log tabulates shipped vs best-candidate numbers per page, and either
reopens #263 with a count or records it as absent on main.

## Stream E — noise that erodes trust

### TICKET-E1 — a page-sized raster with native words inside it is the scan · TODO · depends-on: B1 · wave 6
**Problem:** #511 large half: `has_chart_marks` (`figures/extractor.py:1066`) fires on 97% of
scanned Fed pages; the "chart" PNG is the page itself. No prose lost; a phantom asset per page.
**Do:** In `has_chart_marks` / `_is_chart_asset_page`: a raster whose placed area covers
≥ `SCAN_RASTER_PAGE_COVERAGE_MIN` of the page (measured on the Fed scans and on a real
raster-chart fixture with OCR text, recorded before the value is set) *and* whose page has
native words inside the raster that are not all inside a chart-sized sub-box is the scan →
not a chart. Keep the small-raster gate (#510). Do not cite the old audit-counter
discrepancy: `chart_pages` no longer exists on main; instead assert the current audit
summary agrees with the markdown marks on the fixture.
**Files:** `src/socr/figures/extractor.py`, `src/socr/pipeline/orchestrator.py`,
`tests/figures/test_has_chart_marks_scan_raster.py`.
**Done when:** Fed 1969-05-27 p1 fixture emits no chart mark; a raster-chart-with-text fixture
still does; test pins the difference; audit summary == markdown marks on both.

### TICKET-E2 — `table_not_scorable` only on pages with a detected table · TODO · depends-on: E1 · wave 7
**Problem:** every prose page of a transcript is flagged "untrusted tables" (3/3 on ECB
transcript; 400 events / 68 docs on Fed). No GitHub issue yet — open one first and cite it.
**Do:** Emit `table_not_scorable` only when `detected_table_count > 0`.
**Files:** `src/socr/pipeline/orchestrator.py`, `src/socr/core/tables_trust.py`,
`tests/pipeline/test_table_not_scorable_scope.py`.
**Done when:** transcript excerpt yields `untrusted_page_count == 0`; a table page still flags;
test pins the difference.

## Stream F — one-table Fed items (last)

### TICKET-F1a — ditto cells resolve to the value above (text) · TODO · depends-on: A2 · wave 7 · #625
**Do:** `OutputNormalizer.normalize()` (`core/normalizer.py:115`, returns a string) replaces
a cell that is only a ditto glyph with the value above, and returns the list of
`(table, row, col)` it changed alongside the text.
**Files:** `src/socr/core/normalizer.py`, `tests/core/test_ditto.py`.
**Done when:** unit test pins the difference on a three-row fixture; multi-column ditto and a
ditto in the first row (no value above → unchanged, recorded) are covered.

### TICKET-F1b — per-cell provenance for derived values · TODO · depends-on: F1a · wave 8
**Do:** Carry F1a's change list into the sidecar as `derived_cells: [{table,row,col,kind:"ditto"}]`
and the assembly; never silent.
**Files:** `src/socr/core/result.py`, `src/socr/core/manifest.py`, `src/socr/pipeline/orchestrator.py`,
`tests/pipeline/test_derived_cells.py`.
**Done when:** the 1979-11-20 fixture sidecar lists every resolved ditto; test pins the difference.

### TICKET-F2 — `&nbsp;` indentation becomes hierarchy · TODO · depends-on: F1b · wave 9 · #624
**Files:** `src/socr/core/normalizer.py`, `tests/core/test_nbsp_hierarchy.py`.
**Done when:** fixture rows carry the documented marker; no `&nbsp;` in output; test pins the difference.
