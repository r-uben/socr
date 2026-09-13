# #747: score the SEP corpus against the Fed's published per-bin tables

## What changed

- `src/socr/figures/sep_ground_truth.py` (new): fetches, caches, and parses
  the Fed's accessible per-release page (`fomcprojtabl<date>.htm`, one date
  spelled `fomcprojtable`) into a `ReleaseTable` of `GroundTruthPanel`s. Parses
  `<thead>`/`<tbody>` structurally by each cell's own `id`/`headers`
  attribute, not by position or row-content sniffing, so a year panel with
  only one projection column (5 of 23 corpus releases: every September-quarter
  release) is not silently misread as a fixed 2-columns-per-year grid — the
  bug the first draft shipped with and was caught by re-validating against
  all 23 dates, not just the two the brief's own hand-check covered. A
  headers-id/table-position mismatch, or a `<thead>` missing either header
  row, raises `GroundTruthUnavailable` rather than guessing. `fetch_release_html`
  sends an explicit, identifying `User-Agent`
  (`socr-figures-ground-truth/1 (+...; issue #747)`, not a browser
  impersonation) and turns any fetch failure into `GroundTruthUnavailable`
  naming the URL, the release date, and that a `cache_dir` can be supplied
  instead — see Findings below for why this needed fixing.
- `src/socr/figures/score_sep_ground_truth.py` (new, `socr-score-sep-ground-truth`
  entry point): the measurement harness. Reader values come from a **fresh**
  `read_chart_page` run on this tree's own source, never from the stale
  `~/Data/socr/sep-dotplots/out/` run (see Findings). Model values come from
  that stale run's `<doc>/<doc>.md` model-authored grids, matched to a panel
  by the nearest preceding Markdown heading. Both sides are scored per cell
  against `sep_ground_truth`, classified into `exact` / `wrong_count` /
  `wrong_bin` (an adjacent paired swap of equal-and-opposite magnitude — a dot
  in the wrong bin, not a miscount) / `fabricated` (nonzero where the Fed
  publishes zero) / `missing` (nothing was read) / `no_ground_truth` (the
  reading names a panel or series ground truth does not have). Reader and
  model are reported and returned separately throughout — never collapsed.
- `tests/test_gh747_sep_ground_truth.py` (new, 14 tests): the parser's
  symmetric/asymmetric-column cases, the `GroundTruthUnavailable` refusals,
  every `_classify_series` outcome including the adjacent-vs-non-adjacent
  swap distinction, and `_model_readings`'s bold-only-heading and
  transposed-grid handling (see Findings below). Hermetic (inline HTML/
  Markdown fixtures, no network, no corpus PDF).
- `pyproject.toml`: registers `socr-score-sep-ground-truth`.

**Not wired as a gate or test**, per the brief: CI has no network (the fetch
cannot run there) and no provider (reader/model figures legitimately differ
there), and the corpus scores were not yet known before this ticket measured
them — a gate written first would encode an expectation, not a finding.

## Cold-cache fetch: two real bugs, found by an independent reproducer

team-lead's independent reproduction attempt used a fresh cache directory
and hit `urllib.error.HTTPError: 403 Forbidden` inside `fetch_release_html`
before a single byte landed. My own run never exercised this path — the
cache was already warm from building it, so every call in my session read
`raw/<date>.htm` off disk and `urlopen` never ran. That is a first-run-only
failure, invisible from a warm machine: the harness looked correct to me and
was broken for anyone starting clean, including me on a clean checkout. Two
separate bugs surfaced once this was chased down:

1. **federalreserve.gov 403s urllib's default `Python-urllib/x.y`
   User-Agent.** Fixed by sending an explicit, identifying User-Agent
   (`socr-figures-ground-truth/1 (+...; issue #747)`) rather than a browser
   impersonation. Verified live: the same date that 403'd now returns 200
   and the same Figure 3.E table byte-for-byte (modulo a per-request
   Cloudflare challenge nonce embedded in the page, which differs on every
   fetch by construction).
2. **A second, unrelated bug the first one was masking:** the one
   URL-spelling override (`_HTM_NAME_OVERRIDES["20220316"]`) was a bare
   filename (`"fomcprojtable{date}.htm"`), not a full URL — so a cold fetch
   of `2022-03-16`'s release would have raised `ValueError: unknown url
   type` even with the User-Agent fixed, on the one release this override
   exists for. This had never been exercised either: the override is read
   by `_htm_url`, and every prior run of this ticket's own validation
   against all 24 cached files only ever hit the `raw_path.exists()` branch,
   never `fetch_release_html` itself. Fixed by making the override a full
   URL like the base template.

Both fixes are covered by a new hermetic test
(`test_fetch_release_html_turns_a_cold_cache_403_into_a_legible_error`,
`urlopen` mocked via `monkeypatch` to raise `HTTPError(403)`, asserting the
call surfaces as `GroundTruthUnavailable` naming the release date) and by
running `fetch_release_html` live against the network for both a fake date
(confirms a 404 -- not a 403 -- now reaches past the User-Agent block and
comes back as `GroundTruthUnavailable`) and `20220316` itself (confirms the
override fix works end to end).

**What this does and does not verify.** team-lead's independent
reproduction, run against my already-cached HTML (`raw/*.htm`), verifies the
scoring and parsing logic downstream of the cache — the half of the
pipeline that reproduced exactly. It does not verify the fetch path itself,
which was 403ing for them at the time; that half is verified here only by
my own live fetch calls above, not by an independent second machine. State
this explicitly rather than letting the reproduction read as more complete
than it is.

## Mutation proof (five guards, per CLAUDE.md's discipline)

Run outside the repo: `/tmp/socr747-mutants/prove_guards.py` (not committed).
Loaded (`socr.__file__` under `socr-747`), applied uncapped
(`src.count(anchor) == 1` before mutating), load-bearing (mutant output
differs from real output), right-suite (each mutated line is guarded by one
of this ticket's own tests, confirmed collected):

- `_classify_series`'s adjacent-swap check (`da == -db`): disabling it turns
  `[wrong_bin, wrong_bin]` into `[fabricated, wrong_count]` — the real
  `test_classify_series_wrong_bin_is_an_adjacent_paired_swap` would go red.
- `parse_release_html`'s headers-id/position cross-check: disabling it makes
  a `<td>` with the wrong `headers` id parse silently instead of raising —
  the real `test_parse_release_html_raises_on_a_headers_id_mismatch` would go
  red.
- `_model_readings`'s bold-only-heading fallback match: disabling it makes a
  document whose panel headings are bold-only Markdown (`**2020**`, no `#`)
  fall back to the page's own `## Page 1` heading — the real
  `test_model_readings_understands_a_bold_only_panel_heading` would go red
  (`KeyError` on `readings["2020"]`).
- `_model_readings`'s transposed-grid skip (`_is_transposed(...)`): disabling
  it scores a transposed grid's single generic row under a bogus "series"
  named after a bin range — the real
  `test_model_readings_skips_a_transposed_grid` would go red
  (`readings != {}`).
- `fetch_release_html`'s cold-fetch-failure wrapping (`except
  urllib.error.URLError`): disabling it lets a raw `HTTPError` escape past
  `fetch_release_html` instead of becoming a `GroundTruthUnavailable` naming
  the URL and the cache-dir alternative — the real
  `test_fetch_release_html_turns_a_cold_cache_403_into_a_legible_error`
  would go red.

## Corrections to the brief (measured, not implemented around)

1. **No raw-value-to-bin summing is needed.** The brief assumed the Fed's
   public data was per-participant raw values requiring a many-to-one fold
   onto socr's 0.25-wide bins. Figure 3.E — the exact figure every corpus page
   renders — publishes its own histogram **already binned at socr's exact bin
   width**, with column headers that are already the bin labels
   (`0.13 - 0.37`, comparable via the *same* `chart_reconcile._bin_key` the
   reader/model reconciler already uses). No separate mapping rule exists to
   verify because none is needed.
2. **No second-release fetch is generally needed.** The brief's pairing step
   (fetch the current release AND the prior one) is unnecessary: every
   release's own table carries BOTH projection-month columns already,
   literally labelled ("September projections", "December projections").
   The one exception is also moot: September 2020 published no Figure 3
   series at all (confirmed: its accessible page has Figure 1 and Figure 2,
   no Figure 3.x), so there is nothing to fetch there even if a second fetch
   were wanted — and December 2020's own table already carries the September
   2020 counts, which is what `sep-20201216-p09` needs.
3. **The brief's own hand-checked numbers for `sep-20201216-p09`'s Longer-run
   panel are internally consistent with what this parser reads** (`2.13-2.37`
   → 3, `2.38-2.62` → 9, matching "Fed 2.125→ +2.500→ = 1 + 8 = 9" from the
   issue comment) — flagging this because an earlier draft of my own notes
   misremembered these as 6/10; the parser and the issue agree at 3/9.

## Corpus scores (23 documents, all releases 2020-12 through 2026-06)

**Reader vs. truth: 2221 exact / 2271 total readings (50 `no_ground_truth`,
zero `wrong_count`, zero `wrong_bin`, zero `fabricated`, zero `missing`).**
Denominator, stated plainly rather than left for the reader to trust: of the
2271 cells the reader produced a reading for, 2221 were actually compared
against a Fed value and matched exactly; the other 50 could not be compared
at all (no matching panel/column in the ground truth — see below) and are
counted as `no_ground_truth`, not folded into the exact count. Every cell
that WAS compared came back exact, corpus-wide — consistent with the
20-of-20 hand-check the issue reported on two documents, now confirmed
across all 23. A scorer that silently declines to compare would report the
same 100%-of-compared figure while comparing nothing; the fact that 50 of
2271 readings were declined (not silently matched) is the evidence this
harness can and does say "no" instead of defaulting to "yes" — see the two
reader defects below, which are real declined-to-match cases, not synthetic
ones.

The 50 `no_ground_truth` cells are a genuine reader defect, found by this
measurement, not a scoring-harness artifact:
- `sep-20250319-p09` and `sep-20250618-p09`: **every panel's `label` reads as
  the empty string** (`chart_reader._panel_label`'s x0/x1 containment check
  rejects the year heading because it starts a few points left of the plot
  frame on these two releases' layout) — 40 cells lose their year identity
  entirely and cannot be matched to a ground-truth panel. This is a reader
  bug, out of this ticket's scope to fix; team-lead is filing it as its own
  issue.
- `sep-20250917-p09`: the `2028` panel's `June projections` series (10 cells)
  names a column the Fed's own table does not carry for that panel — 2028 was
  first projected in September 2025, so June 2025 published no `2028` column
  at all. Whether the page itself draws a (necessarily-empty) dashed series
  here or the reader manufactures one is not resolved by this ticket; flagged
  as `no_ground_truth` rather than silently matched or silently dropped,
  which is the harness doing its job.

**Model vs. truth (from the stale prior run's `.md` grids, `--model-dir
~/Data/socr/sep-dotplots/out`): 202 exact / 232 classified (21 `wrong_count`,
9 `wrong_bin`) / 525 total cells, 293 `no_ground_truth`.** Denominator: of
525 model-authored cells, 232 were actually compared against a Fed value
(202 exact, 30 wrong); the other 293 could not be compared at all.

The `no_ground_truth` count is entirely attributable to one model defect,
not scattered noise. An earlier draft of this log mis-stated it as "5 of 23"
documents while listing seven stems in the parenthetical — one of those
seven numbers was wrong. Re-checked by directly inspecting each candidate
document's `.md` table headers (`grep -n "^| Percent" <doc>.md`), which
separated three distinct causes that had been conflated:

- **The genuine model defect — single-column collapse — is confirmed on
  exactly 5 of 23 documents**: `sep-20210317-p09`, `sep-20210922-p09`,
  `sep-20231213-p09`, `sep-20241218-p09`, `sep-20250319-p09`. Each one's
  model-authored grid carries a **single** `"Number of participants"` column
  instead of two separate projection-month columns, so none of that
  document's cells can be matched to either of the Fed's two columns. This
  is a **model** defect (visible in that stale run's output), not a reader
  defect and not a harness bug — the harness correctly declines to guess
  which of two Fed columns a collapsed cell belongs to. These 5 documents
  account for all 293 `no_ground_truth` cells on the model side
  (48+60+85+60+40 = 293, confirmed by a per-document breakdown).
- **`sep-20201216-p09` was wrongly included in the original list — it is a
  harness bug, not a model defect.** This document's model-authored grid
  correctly carries two columns ("December projections" / "September
  projections"), but every one of its panel headings is written as
  bold-only Markdown (`**2020**`) rather than a `#`/`##`/`###` heading. The
  original heading scanner only recognised `#` headings, so it fell back to
  the page's own `## Page 1` heading and merged every panel on the page into
  one bucket keyed `"Page 1"` — which then failed to match any ground-truth
  panel by year. Fixed by adding `_BOLD_HEADING_RE` as a fallback match in
  `_model_readings`'s heading scan. This alone recovered 120 cells that were
  previously misclassified as `no_ground_truth` for a harness reason.
- **`sep-20251210-p09` was also wrongly included — a second, distinct
  harness bug.** This document's grid is **transposed**: bin ranges appear
  as the column headers and the single body row is labelled with a generic
  string (`"Number of Participants"`) instead of one bin label per
  projection-month column. The original code read this as a "series"
  literally named for a bin range, which then failed to match any Fed
  column. This is not the same defect as the genuine collapse above — the
  grid has the right information, oriented the wrong way for this parser.
  Fixed by adding `_is_transposed()` and explicitly skipping such grids in
  `_model_readings` (contributing 0 cells, correctly, rather than silently
  mis-scoring them).

With both harness bugs fixed, the `no_ground_truth` total (293) now equals
exactly the sum of the 5 genuinely-collapsed documents' cells, with no
residual from either fixed cause — the load-bearing check that the
recount, not just the prose, was right.

**Where the 30 classified mistakes (21 `wrong_count` + 9 `wrong_bin`) fall,
and a correction to a claim made about this in reproduction.** team-lead's
first independent reproduction ran against the pre-fix numbers (88/112
classified, 24 mistakes) and reported all 24 concentrated on one document,
`sep-20220316-p09`. After the bold-heading fix that claim no longer holds:
`sep-20201216-p09` now contributes 6 of the 30 mistakes (2 `wrong_bin`, 4
`wrong_count`) that were previously invisible as `no_ground_truth` cells
under the harness bug, not previously-checked-and-passing cells. The
corrected breakdown is 24 mistakes on `sep-20220316-p09` and 6 on
`sep-20201216-p09` — concentrated on two documents, not one, and the second
document's mistakes only became checkable once its own harness bug was
fixed. `sep-20220316-p09` remains the standout: 24 of its cells are wrong,
against 6 spread across the other document and zero on the remaining 21.
That document was already this corpus's outlier before this ticket — 9 of
the 10 pre-existing #734b contradictions were on it too, through a
mechanism this ticket did not touch.

**Cross-validation on `sep-20220316-p09`'s Longer-run panel.** This parser's
`ReleaseTable` for `20220316` reads, for the December 2021 projections
column: `2.13-2.37` → 4, `2.38-2.62` → 10. team-lead derived exactly these
two numbers by hand from the Fed's December 2021 table before this harness
existed (summing `2.250`→4 for the first bin, and `2.375`→1 plus `2.500`→9
for the second). Two independent routes — a hand sum from Figure 2's raw
per-participant values, and this module's structural parse of Figure 3.E —
land on the same pair of merged-bin values. This is also the strongest
evidence yet for **Correction 1** above: Figure 2 (raw values, summed by
hand) and Figure 3.E (pre-binned, parsed structurally) describe the same
projections and agree, which was not established by this ticket's own
work and makes relying on Figure 3.E's pre-binned table safe, not merely
convenient.

## What was NOT done, and why

- The `~/Data/socr/sep-dotplots/out/` reader data was confirmed fabricated
  garbage on most non-refused pages (bin labels are literal footnote-sentence
  tokens, e.g. `"Definitions"`, `"of"`, `"variables"` — matches
  `docs/plans/chart-data/DIAGNOSIS-sep-2026-09-12.md`'s known root cause) and
  was **not used** for reader scoring; a fresh `read_chart_page` run against
  this tree's own source was used instead, per the module docstring's stated
  rationale.
- No fresh multi-engine run producing current model output was available in
  this worktree (would require network/provider access this ticket did not
  have); the model side is scored from the existing stale run's `.md` grids
  as the best available signal, with the caveat above stated plainly rather
  than presented as a clean number.
- Two findings are reported, not fixed, because fixing them is out of this
  ticket's scope (a measurement harness, not a reader/prompt fix): the
  reader's blank panel label on 2 documents (a reader defect), and the
  model's single-column collapse on 5 documents (a model defect — never the
  same thing as the reader defect above; the two are scored, and reported,
  on entirely separate sides of this harness on purpose). The two additional
  root causes found while re-checking the model count (`sep-20201216-p09`'s
  bold-heading collapse, `sep-20251210-p09`'s transposed-grid gap) were
  harness bugs, not findings about the reader or model, so those WERE fixed
  in this same commit rather than left as findings.

## Test result

`PYTHONPATH=$PWD/src ~/venvs/socr/bin/pytest tests/ -q` → **5360 passed, 4
xfailed** (pre-existing xfails, unrelated to this ticket).
`uvx ruff@0.16.0 format --check .` → clean (668 files).
