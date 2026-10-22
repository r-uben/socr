# #1227 — a percent spaced from its number in LaTeX

## Diagnosis

The #1185 own-text-layer check (`text_layer_numbers_clause` → `compare_numbers`) dropped two
LaTeX percent writings. Both failed closed, so no false CHECKED, but a wrong number in either
writing was silent.

- `$\pi^* = 9 \%$` in a table row: `_number_or_nothing` kept a `=` span only if its right
  side was one number. It stripped `\,` `\;` `\:` `\!` `\ ` and `~`, but not a plain space, so
  `9 %` failed `_LONE_NUMBER_RE` and the span was dropped.
- `9~\%`, both in a cell and inside maths: `_strip_latex` turned `\%` into `%` and left `~`,
  so the token was `9~%`, which does not parse.
- The old `\,` test only asserted agreement. A dropped cell also "agrees", so reverting the
  `\\[,;:! ]` alternative did not fail it.

## Change

- `_number_or_nothing`: `\\(?=%)` → `[ \t]*\\(?=%)`. Only blanks directly before `\%` are
  stripped. Stripping every space would join `= 1 2` into a single number.
- `_strip_latex`: `~?\\%` → `%`. Only a tie directly before `\%` is stripped. A bare `~5`
  stays unparsed, so nothing beyond the three cases becomes a number (#1220 holds the rest).
  The reference side runs through `_strip_latex` too, but a PDF text layer has no `~\%`.

## Tests

`test_a_spaced_latex_percent_cell_is_counted`, parametrised over `$= v\,\%$`, `$= v \%$`,
`$= v~\%$` and `v~\%`. For each writing:
- the right value (2) adds exactly one compared number over the empty-cell row, and the clause
  stays silent;
- a planted wrong value (9) is the only dispute, and the clause FAILs.

Mutants, each in an external copy of `src` + `tests` + `pyproject.toml`, with `socr.__file__`
inside the copy and `count(anchor) == 1` asserted:
- reverting the space fix kills the `v \%` row;
- reverting the tie fix kills both `~` rows;
- removing `\\[,;:! ]|` kills the `\,` row.

## Advisor (Opus subagent)

- It claimed `$x_2 = 9\%$` leaks the subscript `2` as a compared number, and proposed keeping
  only the right side. **Rejected after measuring:** that `2` is tagged as a label digit, and
  `compared` stays 1. Changing what a kept span returns would also be out of scope.
- **Adopted:** assert the exact dispute rather than just a FAIL, and add `$= v~\%$` as a
  fourth writing (it was dropped too).
- Diff review: no defects. It compared HEAD and the new tokenizer on about 40 inputs (ranges,
  `~\%` with no digit, `| 9 | \% |`, a newline before `\%`); only the intended cases changed.
  **Adopted:** pin the written form each row disputes (`9` for the plain space, since the `%`
  splits off, and `9%` otherwise), not "either one".

## Not measured

No corpus is on this machine, so how many real pages the fix touches is unknown.

## Round 2 — Argon review of PR #1229 (MERGE-AFTER-FIX)

Each finding was probed through `compare_numbers` and `text_layer_numbers_clause`, on this
branch and on origin/main (6e7d04de):

| finding | probe | main | round 1 | verdict |
| --- | --- | --- | --- | --- |
| P1 bracket | `$\pi^* = (9 \%)$` vs layer `9` | dropped | **agrees (false)** | real, new in round 1 |
| P1 bracket | `(9 \%)`, `$(9 \%)$` vs layer `9` | agrees (false) | agrees (false) | real, pre-existing |
| P1 year guard | `1995 \%` vs layer `1995.5` | agrees (false) | agrees (false) | real, pre-existing |
| P1 prose | `$9 \%$` in prose | dropped | dropped | fail-closed, not a false agreement |
| P2 `\~\%` | `3\~\%` vs layer `3` | dropped | **counted as 3** | real, new in round 1 |
| P2 label | `A-8~\%` | dropped | label digit | matches main's `A-8\%`; a label cannot cause agreement; left as is |

Every one has the same cause: round 1 removed the spacing only inside the equation check,
so later steps still saw `9 %`.

**Change.** Round 1's two edits are reverted to main. In their place,
`_PERCENT_SPACING_RE` deletes the blanks and ties before a `\%` as the first step of
`_numeric_tokens`, on both sides. Every spaced writing then reads exactly as `9\%`
already does: the equation check, prose maths, brackets, the label lookahead and the year
guard. `(?<!\\)~` keeps the accent `\~` unparsed.

The owner suggested `[ \t]*~?\\%` → `%` in `_strip_latex`. It would close the bracket,
prose and year cases, but `_strip_latex` runs after the label step, so `A-8~\%` would differ
from `A-8\%`. The other early option, rewriting to `%` (not `\%`) at the top, would feed `9%`
to the currency lookahead in `$9 \%$`, where it used to see `9\%`.

Keeping the backslash avoids changing any downstream path.

**Tests.**
- The counted test now has 8 rows: 5 in a table cell and 3 in prose, the prose ones lone
  numbers only, since equations in prose are dropped by design. The plain-space row now
  reads `9%`.
- New: the spaced writing compares equal to the unspaced one (whole record), and that
  record is a dispute. Covers the bracket cases (plain, maths, equation) and the year guard.
- New: `3\~\%` is not counted.

**Mutants** (external copy, `socr.__file__` inside it, `count(anchor)==1`):

| mutant | failing tests |
| --- | --- |
| remove the blank alternative | 6 |
| remove the tie alternative | 5 |
| remove the accent lookbehind | 1 |
| remove `\\[,;:! ]` from the equation check | 1 |
| remove the normalisation line | 11 |
