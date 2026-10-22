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
