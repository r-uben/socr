# #1177: the document `.md` is the assembly of the final records

2026-10-16. Branch `fix/issue-1177` off `origin/main` (b4b59ade).

## Cause

`_phase_assemble` saves the document `.md` from `pre_records = finalized_page_records(state)`.
`_preserve_chart_regions` runs after that and is where `chart_region_placement_unresolved` /
`chart_region_render_failed` get set. The later `finalized_page_records(state, final_text)`
re-runs the #1145 verdict seam with the flag now set, adds
`> [page N unchecked: chart_region=FAIL; …]` to the record, and `_rewrite_all_fragments` writes
it into `pages/NNNNN.md`. The `.md` was re-saved only on the table-emission-failure branch,
so the document missed the line. Before #1145 the chart-region guard only changed status, so
the bodies never diverged. Measured on run 23319732: `document_not_assembly` = 7,537 pages in
152 documents, and `document_mark_not_first` = 311.

## Fix

In the final-body guard, after `final_records` is computed: if any record body (stripped, as
`assemble_pages` strips) differs from the corresponding page of `final_text`, rebuild
`final_text = assemble_pages(record texts, page_numbers=1..N)`, set
`final_result.pages[0].text` and `_save_markdown` it. This is an `elif` after the emission
branch, which already rebuilds from the same records.

- When the bodies match, the document is not rewritten, so documents whose final records
  match the saved body stay byte-identical. The golden and byte-identity suites pass
  unchanged.
- The metadata is not rewritten. None of its fields comes from the saved body
  (`blank_pages` reads `best_output`, and the counts are covered by the GH-993 rewrite).
  `output_path` names the same file, and the document status already counts the
  chart-region flags. GH-503's provisional record is not touched.
- The manifest, sidecars and fragments are written after this point from the same
  `final_records` and the new `final_text`, so all four copies agree.

**Rejected alternative: run the verdict marks after `_preserve_chart_regions`.**
`pre_records` feeds `canonical_page_texts`, which is the input to `_preserve_chart_regions`.
Moving the seam would need a second records pass over the preserved texts and would change
the notes, buckets and counts built from `pre_records`. It would also cover only the
chart-region flags. Re-saving from the final records covers any page state that changes
after `pre_records`, and touches one site.

**Gap this fix leaves open:** the first metadata write is already final, so if the run
crashes between it and this re-save, the document is recorded COMPLETED while its `.md` is
still stale. Fragments, sidecars and the manifest are written after that metadata write
too, so they already had this gap. It is not new with this fix, and closing it means
reordering the metadata writes, which is outside this ticket.

## Tests

`tests/test_gh1177_document_md_final_records.py` runs `process()` on the GH-189 synthetic
mixed chart+table page and reuses that file's fixture. `route_page` is stubbed with an
accepted decision, the ladder is patched and the judge model is `""`, so the test is
hermetic and CI takes the same path as a local run.

- Both arms (anchorless winner = unresolved placement; anchored winner = control): document
  `.md` == `assemble_pages(fragments)`, and `result.pages[0].text` == the `.md`.
- Unresolved arm: a spy on `_preserve_chart_regions` checks that the flag went from False to
  True inside it, so it was set after the first save. The fragment carries a
  `chart_region=` canonical mark, and the document's leading marks under `## Page 1` equal
  the fragment's.
- Control arm: the flag stays False and no `chart_region` mark appears in either copy.

Before the fix, the same `process()` run gave a document lacking the mark that the
fragment had at its top, and `doc != stitched`.

## Mutant

External copy at `/tmp/1177-mutant` (src, tests and `pyproject.toml`), plus a canary test
that asserts `socr.__file__` starts with `/tmp/1177-mutant/src/`. The unmutated copy gave
5 passed. The mutant replaced the `elif` condition with `elif False:`. Before editing, an
uncapped `src.count(anchor) == 1` check passed, and the anchor count was 0 after the edit.
Result: 2 failed (unresolved arm of the invariant test, and the late-flag test), 3 passed
(control arm, control test, canary).
