# 2026-10-07 -- #988 row-shortfall term: confirm a row-count shortfall by content

Branch `fix/table-checks-nav-and-shortfall` from origin/main ed2550e2, first of two commits; the second (navigation
bar, #988 M2) is logged in `2026-10-07_988-navigation-furniture.md`. Issue r-uben/socr#988
(evidence comment: https://github.com/r-uben/socr/issues/988#issuecomment-6041020092).
Supersedes the approach of `fix/truncated-shortfall-symmetric` (blank-stub credit), which clears none
of the pages below.

## Why

Cluster job 687398 ran four Coca-Cola sustainability reports (2018-2021), with Qwen3-VL and then
gemini-3.8-flash. 73 pages shipped the fail-closed marker. One rejected answer per page is cached. An
audit judged all 73 against the text layer and every answer with content against the page image:

| verdict | answers |
|---|---|
| complete and correct | 56 |
| wrong | 11 |
| mixed | 4 |
| not verifiable | 2 |

The shortfall term fired on 58 of the 73 answers. No answer had lost a row. The 6 right
`table_truncated` refusals were column shifts or invented values.

Mechanism: the native side counted every page band with `row_shape_min` numerals (year caption, footer,
footnotes, prose figures, chart labels), and one-value rows make `row_shape_min` 1. 2021 p74 (Gemini,
every number on the text layer) read 17 rows against 24 bands.

## Change

- `structure_check._truncated_row_shortfall`: the row count runs first, unchanged (same bands, through
  `row_corroboration.is_table_shaped_band`, factored out of `table_shaped_native_row_count`). When it
  falls short, the same inequality (`_falls_short`) is applied to the number of bands the candidate
  ACCOUNTS for, and the term fires only if that falls short too.
- A band is accounted for iff its numeric tokens can be drawn from the multiset of numeric tokens the
  candidate wrote anywhere (`_candidate_numeric_supply`). Tokens are consumed, one written number per
  band. Candidate text is split on whitespace, `|`, `*` and `<br>`. Footnote superscripts (`³`, `$^3$`,
  `<sup>3</sup>`) are glued to the preceding token, as PyMuPDF glues them.
- The lane gate, `row_shape_min`, `_STRAY_HEADER_BAND_ALLOWANCE` and `ROW_CORROBORATION_MIN` are
  unchanged. The native count is not narrowed by geometry: region scoping, rejected in #988 rounds 1-4,
  was measured again here and is blind to a cut tail.
- Consequence of the AND: the term never refuses a reading main's row count accepts.

## Review (socr-reviewer, findings in the audit scratch dir): two required changes, both taken

1. The first draft skipped bands made only of bare years. That let a cut table of year values through
   (start/end years: 12 of 24 rows dropped, unflagged). The skip is removed, and the year test now pins
   that cut.
2. The first draft replaced the row count by the content count, so a formatting difference created
   refusals main never made:
   - a marker written `^3`, or `³` after a space;
   - a marker the model dropped;
   - `10,234` against a printed `10 234`.

   The AND fixes this: content only decides on a page the row count already doubts.

Recurring values and repeated blocks were checked by the reviewer and are not bypasses (consumption holds).

## Measured (56 complete answers; synthetic cuts = largest table, 50 answers have one)

`table_truncated` alone:

| | main | this branch |
|---|---|---|
| complete answers flagged | 47 / 56 | 6 / 56 |
| tail cut 10% / 25% / 50% (of 50) | 44 / 42 / 44 | 43 / 41 / 44 |
| middle cut 10% / 25% / 50% (of 50) | 45 / 46 / 44 | 26 / 41 / 44 |

Gate level (the production `NativeTableVerifierJudge` replayed on each cached answer, inner judge
stubbed to accept, same probe for both trees): complete answers passing go from 1 to 15 of 56, and no
answer main accepts is refused. Two answers that are not fully correct now pass; on main each was refused
only by this term misfiring: 2018 p60 (invented value) and 2019 p51 (mixed). With the navigation-bar fix
the figure is 40 of 56 (see that log).

## Tests

`tests/tables/test_gh988_shortfall_missing_content.py`: the real p74 fixture (words + Gemini answer)
plus one synthetic test per mechanism. On main 6 of the 9 fail; the other 3 pin behaviour main already
has and the fix must keep (never refuse what the row count accepts; refuse a cut table of year values;
refuse a reading that writes one of two identical panels).

Mutants (external copy, `socr.__file__` canary), each killed:

| mutant | killed by |
|---|---|
| no consumption | value-written-once test |
| no superscript glue | p74 test, glue test |
| no `*` split | chart-label test |
| supply = table rows only | 6 tests |
| no row-count gate | never-refuses test |
| content gate off | 6 tests |
| year-only bands credited | year-rows test |

## Known limits (disclosed in the docstring)

- A dropped row is missed when its numbers recur elsewhere in the candidate, or when it is the one row the
  allowance absorbs.
- Numbers restated in prose are credited (pinned by `test_known_limit_…`).
- A block written twice is judged by the row count alone, as on main, so a doubled half-table passes.
- The term no longer catches, by accident, column shifts or invented values (on the corpus: 2018 p60). A
  column-arity check and an invented-token check are follow-ups.
