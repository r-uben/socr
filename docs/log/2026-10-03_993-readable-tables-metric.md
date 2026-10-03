# GH-993: readable tables per paper

Branch `feat/993-readable-tables-metric`, cut from origin/main 54fe285a (ancestry verified).

## What changed

- `src/socr/core/table_counts.py` (new): `count_page_tables`, `count_document_tables`,
  `count_from_sidecars`, `TableCounts` (with the CLI line). Pure derivation, no detection,
  no thresholds.
  - `shipped_text`: `find_table_blocks` over the shipped page text.
  - `verified_text`: that count on a `success` page with no `tables_trust` entry.
  - `unverified_text`: that count on a page with `failure_mode == table_unverified` or a live
    `table_ladder_unverified` flag.
  - `withheld`: one per `[page N failed: unverifiable table` / `invalid table emission`
    marker in the shipped text. A prose-recovery page (banner
    `SCANNED_PROSE_RECOVERED_FLAG`) counts 1, because its markers are per withheld run and
    the producer says a run count is not a table count.
  - `flattened_to_prose` left out: socr does not record it (#994).
- `pipeline/orchestrator.py`: `_record_table_counts` runs after `pre_records` and again after
  `final_records` (the post-figure guard can turn a table into an invalid-emission marker);
  result on `DocumentState.table_counts` (new field, `core/state.py`). `_write_metadata`
  adds `tables` to the per-document `metadata.json` only, through the existing
  `_LatchedDocMetadata` wrapper; the root index keeps the contract shape. CLI prints
  `tables: N as text (V verified), W withheld` at the end of assemble (not in quiet mode).
- `library.py` / `cli.py`: per-paper `tables` in `manifest.json` (metadata block, else the
  same derivation from sidecars, else `null`), and `tables` + `tables_recorded_documents` in
  the `refresh_index` summary, printed on the "Index refreshed" line. The total is over the
  documents that have counts; the denominator is printed.
- `docs/OUTPUT.md`: `tables` block definitions.

## Validation on real output (no pipeline re-run)

Read from `~/.local/state/socr-housekeeping/archive-scan/redo-out/<stem>/pages/*.json` and
`tables_trust.json`, compared with `withheld/chain.json` (chain covers only pages where the
January output had more tables than the re-OCR, so it is a subset check, not equality).

| paper | shipped_text | verified | unverified | withheld | chain `failed_table_img` pages | ours |
| --- | --- | --- | --- | --- | --- | --- |
| Gleason_Lee 2003 | 4 | 0 | 4 | 3 | 16, 17 | 16, 17 (p17 carries two regional markers) |
| hansen 1995 | 3 | 0 | 3 | 2 | 8, 13 | 8, 13 |
| bernanke_kuttner 2005 | 8 | 0 | 8 | 1 | 6 | 6 |

Also bochkay (chain 9, 15, 22, 23, 26 all present; four more pages withheld that the chain
does not list because January had no table there either) and peersman (p8 withheld, matches
`tables_trust.json` `structure_class_ladder_exhausted_floor`; chain does not list it).
Every chain page is in our withheld set; every chain `table_withheld` page is in it.
Chain `table_unverified` pages with kept text are a subset of our unverified pages.

First pass overcounted bochkay p15 (6 markers on one prose-recovery page); that is what the
prose-recovery clamp fixes.

`verified_text` is 0 on these because every one is a halted `partial` run whose table pages
carry distrust flags. The corpus-wide verified rate needs a full library refresh, not done.

## Tests

`tests/test_gh993_readable_tables.py` (13 tests): process() on the GH-359 harness (scripted
ladder rungs, provider and judge patched) reaches a verified, a withheld and an unverified
table and asserts exact counts; difference pin (one reader verdict changes the counts by
exactly one); metadata equals an independent sidecar derivation; root index unchanged;
CLI line once; post-figure guard withholding; per-page classes; library entry and total.
`_surface_table_scoring` is stubbed in the e2e cases: the ruled fixture otherwise raises
`table_not_scorable` on every page, and no table could be verified.

Mutations in an external copy (src, tests and pyproject copied; canary asserts `socr.__file__`
is in the copy; anchor count asserted `== 1`): 12 of 12 killed. Mutants: withheld forced to
0, verified ignores trust, unverified ignores failure mode, shipped forced to 0,
prose-recovery clamp removed, metadata block not written, counts not stored on state, final
records not recounted, CLI line removed, library sidecar fallback removed, library total
zeroed, manifest entry dropped. (`final_records_not_recounted` survived the first test set
and prompted the post-figure test.)

## Caveats

- Blocks, not tables: a fragmented table counts per fragment (same unit as the chain).
- `verified_text` is conservative: a `success` page with any trust flag is not verified.
- `shipped_text - verified_text` includes rejected and flagged text, not only unverified.

## Full suite

One run, default OLLAMA_HOST, nohup: 6593 passed, 4 failed, 2 skipped, 4 xfailed. The 4 failures
were the P6 stage A/B/C difference oracles: their CLI capture (`tests/p6_corpus_fixture.py`)
compares the assemble output to a pre-change baseline and now also saw the new additive
`tables:` line. The capture drops that line (it is pinned in `test_gh993_readable_tables.py`);
the two P6 files then pass (61 passed together with the GH-993 file). The rest of the suite
was not re-run after that one-line fixture change.
