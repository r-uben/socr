# GH-928: native_first reads markdown tables with the verifier's helpers

## Change

`src/socr/tables/native_first.py`: `_markdown_table_tokens` skips a separator with `_MD_SEP_RE.match(line)` (was
`startswith("| ---")`) and splits cells with `_parse_output_row_cells`; `splice_cell_tokens` splits the row with
`_parse_output_row_cells`. Both helpers were already imported. No new parser.

## Measurement (127-page census, `inputs_now.pkl`, frozen words/markdown)

- Separator lines: old rule vs `_MD_SEP_RE` classify every line identically (0 differing lines, 0 pages).
- Cell splitting: old `strip("|").split("|")` vs `_parse_output_row_cells` identical on every line (the helper is the same
  expression; 0 differing lines).
- Full `plan_native_table` + `_markdown_table_tokens` + `retained_prose_lines_to_keep` + `splice_cell_tokens` (on the plan's
  CELLS), main vs branch: 110 defer / 17 ship on both; no page changes action, reason, predicates, cells, tokens, kept prose or
  spliced bytes. No page changed, so no render review applied.

So the corpus holds no aligned separator in a native-first markdown; the change is a no-op there and fixes only the synthetic case.

## Notes

- `_MD_SEP_RE` needs at least two columns. The single-column `|:---|` from the issue is NOT matched by the shared regex (old
  code does not match it either, so no change). Tests use `|:---|:---:|`.
- The shared cell split is not escape-aware: `a \| b` splits at the escaped pipe. Pinned, not changed.

## Tests (`tests/test_native_table_first.py::TestSharedMarkdownParsing`)

Aligned separator is not content (difference pin vs plain separator), aligned separator does not suppress retained prose,
escaped-pipe split behaviour pinned, splice over an aligned-separator table.

Mutations in an external copy (src + tests copied out, `socr.__file__` asserted inside the copy):
- M1 old `startswith("| ---")` in tokens: killed (1 failed).
- M3 escape-aware split in `splice_cell_tokens`: killed (1 failed).
- M2 old inline split in tokens: survives. It is an equivalent mutant: the old expression equals the helper byte for byte.
