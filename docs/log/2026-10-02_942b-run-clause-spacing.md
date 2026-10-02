# GH-942b: should `header_band_missing`'s run clause use `_page_word_space`? Measured: no effect, not built

Branch `fix/942b-run-clause-spacing` from origin/main f63927e (ancestry verified). Follow-up to the
cubic P1 on #943 (table-dominated pages: gutters dominate `_median_word_gap`, so the run bound is too wide).

## Experiment (throwaway, reverted)

Monotone form: keep the all-words run count, OR in a second `_run_count` using
`_page_word_space(words, zones)` (zones = each block's reach plus extent, as in `prose_in_header`),
taking the larger of the two. No new constant. Census: 127 pages, frozen inputs copied to
`~/.local/state/socr-housekeeping/gh942b/` (base = `git archive origin/main`, branch = worktree,
`socr.__file__` asserted for both).

## Result

- `header_band_missing` fires: 46 -> 46. Removals 0 (asserted `base <= new`). Every other predicate's
  set identical on all 127 pages.
- New fires: 0. New SHIP to DEFER: 0 (so no render to view).
- Spacing evidence exists on 65 pages, abstains on 62.
- The 2 pages #942 still misses stay missed: Gurkaynak 46 abstains (no outside-table evidence; its
  all-words gap is 2.25); Boukus 39 has evidence but the outside spacing (2.98) is within 0.3% of the
  all-words one (2.99), so the bound barely moves.

## Decision

Not built (no code, no tests, no mutations; same discipline as #924). The cubic concern is real in
principle but the census contains no page where it bites. Reopen only if a page appears where the two
estimates differ materially AND the header is missed. The "Known limits" note in
`2026-10-01_942-header-band-runs.md` stays accurate; its "reuse that estimate" follow-up is measured
at zero gain on this census.
