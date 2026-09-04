# GH-582: inline-math-wrapped tokens convicted as contradictions

## What

The issue's directly-measured shapes (ladder corpus run, main @ e830d9b):

- doc02 (Nakamura-Steinsson): 5 of 5 recorded cell contradictions are a
  `$`-wrapped native-equal value (`$-0.06$` vs native `−0.06`, `$-0.21$`
  vs `−0.21`, ...). All three doc02 tables were demoted `ACCEPTED` ->
  `UNVERIFIED` on this (held 0/5, 0/3, 0/4).
- doc03 (Pflueger-Rinaldi): 2 of its 3 recorded items are the same
  wrapping class (`Adjusted R2` native vs `Adjusted $\text{R}^2$` model,
  and an empty native cell vs `$\text{R}^2$`). The third, `S&P` (native)
  vs an empty model cell, is described in the issue only as "a lane shift
  adjacent to the wrapped header, likely the same root" — its cause is
  not independently established, and this fix makes no claim to have
  cleared it; there is no corpus replay or dedicated fixture for it here.

So: 3 tables demoted on wrapping alone (doc02), plus doc03 where wrapping
accounts for 2 of its 3 items and one demoted table — not "zero other
causes" across all four demotions, and not a claim that every doc03 item
disappears under this fix.

A later comment on the same issue reports a running tally after 6 of 8
documents: 14 recorded contradiction items — 8 the wrapping class this
fix addresses, 4 sibling LaTeX classes (Greek-letter commands vs Unicode,
`\log`, `\&` escapes — explicitly flagged there as separate follow-up
scope, not covered here), and 2 native row-label defects unrelated to
this issue — with 6 tables demoted on the full set.

`binding.py`'s cell comparison ran `is_numeric_token(model_value)` before
any comparison; `$-0.06$` fails that predicate (`_normalize_numeric_token`
left the trailing `$` on), so the cell was convicted as "not numeric at
all" without its value ever being compared. `adjudication.tokens_agree` used
the same predicate, so the blind-cell (raster-transcription) adjudicator
could never disprove the conviction either — held 0/5, 0/3, 0/4 in the
observed run. Row/parent labels had the parallel gap: `normalize_label`'s
`_NON_ALNUM_RE` strips `$`, `\`, `{`, `}`, `^` as punctuation but leaves the
LaTeX command name behind (`\text{R}^2` -> `textr2`, not `r2`), so a
wrapped label never matches its plain-text native counterpart.

## Why

Inline math is presentation, the same class GH-103/GH-206 already fold
away for bold/asterisk/dagger/currency decoration — not a value or label
difference. Convicting on it turns a correct table's own emitted markdown
against itself and burns the third-vendor disproof lane on typesetting.

## Fix

One shared function, `strip_math_presentation()` in
`src/socr/tables/native_verifier.py` (next to `strip_presentation`, the
existing numeric normalizer), with two modes rather than one blanket pass
— a cold review of the first draft (a single unconditional strip) found it
created false numeric matches (`4$3` == `43`, `43$` == `43`, `$10^2$` ==
`102`, `1_2` == `12`, none true on `origin/main`), so cell and label
comparisons now go through different, narrower rules:

- **Numeric path** (`label=False`, the default — what `is_numeric_token()`
  / `_normalize_numeric_token()` call, so the binder's cell comparison and
  `tokens_agree`'s numeric path both get it automatically): unwraps ONE
  balanced delimiter pair that encloses the WHOLE token — `$...$` or
  `\(...\)` — via `^\$([^$]+)\$$` / `^\\\(([^()]+)\\\)$`, and nothing
  else. A stray or embedded delimiter (`4$3`, `43$`) is not a wrapper and
  is left untouched, so the numeric regexes downstream still reject it
  exactly as they do on `origin/main`. The interior is disallowed from
  containing another instance of the same delimiter, so a malformed
  double-wrap (`$$43$`) also does not unwrap — a first-round review found
  the earlier `^\$(.+)\$$` unwrapped `$$43$` to `$43`, which the existing
  currency-prefix strip then turned into a false numeric match `43`.
  `^`/`_` script markers are never dropped here: an exponent/subscript
  changes what the token means (`$10^2$` is the expression "10 to the
  power 2", not the decimal `102`), so it stays non-numeric.
- **Label path** (`label=True`, passed explicitly at the row-label call
  sites in `binding.py` and `adjudication.tokens_agree`'s `row_label`
  branch): row/parent labels are folded through `normalize_label()`
  immediately after this call, which already discards
  `$`/`\`/`{`/`}`/`^`/`_` as punctuation — so it is safe (and, to avoid a
  footnote-marker-regex mismatch between the two sides, necessary) to
  unwrap `\text{}`/`\mathrm{}`/`\textbf{}` to their content (looped, so
  nested wraps fully unwrap), flatten every `^`/`_` script marker (braced
  or bare) while keeping the scripted content, and only then drop the
  delimiters — wherever in the label they occur, not just a whole-token
  wrap, so `Adjusted $\text{R}^2$` folds to `Adjusted R2`.

Column-header span confirmation (`binding._norm_header_text`) is a
separate, non-convicting path and does not use this helper — left
untouched, out of scope.

The "not a numeric token"/"empty normalized label" conviction and
`row_label_unverifiable` fail-closed path are unchanged for a cell or label
that is still non-numeric / non-matching after normalization (verified by
`test_symbolic_row_labels_fail_closed_as_unverifiable`, which still holds:
native `β` vs candidate `$\beta$` stays unverifiable, not falsely matched
or falsely convicted).

One deliberate deviation from the review's literal list: `$43$` (a clean,
unsigned wrap with no garbage) is pinned in a test to become numeric
(`"43"`), not to stay non-numeric. It takes the exact same balanced
whole-token-wrap path as the issue's own `$-0.06$`; there is no
principled reason to unwrap one and not the other, and the review's own
prescribed regex (`^\$(.+)\$$`) mechanically unwraps both. Flagged
explicitly rather than silently resolved either way.

No new thresholds.

## Files

- `src/socr/tables/native_verifier.py` — `strip_math_presentation()`
  (`label` keyword), wired into `strip_presentation()` for the numeric
  path.
- `src/socr/tables/binding.py` — both `normalize_label()` call sites, now
  passing `label=True`.
- `src/socr/tables/adjudication.py` — `tokens_agree()`'s `row_label`
  branch, `label=True`.
- `tests/test_binding_adjudication.py` — `tokens_agree` reproducers (cell
  and row_label), plus `not tokens_agree("$$43$", "43", kind="cell")`
  (round-2 review: a doubled delimiter is not a balanced wrap).
- `tests/test_gh103_tokenizer_presentation.py` — parity-with-origin
  regression for the numeric path: `4$3`, `43$`, `$10^2$`, `1_2`,
  `(/997)`, `/53`, `LoIic6`, a wrapped BMP-private-use-char token, `$$43$`,
  `$43$$`, `$$` all stay non-numeric; `$-0.06$`, `\(0.5\)`, `$43$`
  become numeric.
- `tests/test_binding.py` — three `bind()` tests: (1) identical native
  geometry/candidate values, differing only in `$...$` wrapping on every
  numeric cell, assert identical `matched_cells`/`contradicted_cells`/
  `row_label_contradictions`; (2) a two-lane row with one PLAIN numeric
  cell (so row anchoring binds on `origin/main` too) and one `$-0.06$`
  cell, so the walk reaches the actually-reported conviction branch
  (`binding.py` ~1572) rather than failing earlier at row anchoring; (3) a
  `\text{R}^2`-style label fixture asserting no row-label contradiction.

## Difference tests

- `tokens_agree("$-0.06$", "−0.06", kind="cell")` is True;
  `tokens_agree("$-0.06$", "−0.60", kind="cell")` stays False.
- `tokens_agree("Adjusted $\text{R}^2$", "Adjusted R2", kind="row_label")`
  is True; a genuinely different label stays False.
- `bind()` on one native fixture, plain vs every-numeric-cell-`$`-wrapped
  candidate: identical matched cells, identical (empty) contradicted/
  row-label-contradiction sets, zero contradictions on the wrapped run.
- `bind()` on a two-lane row (one plain numeric cell, one `$-0.06$` cell):
  `origin/main` produces exactly the issue's contradiction pair
  (`−0.06`, `$-0.06$`) with row binding otherwise verified; the fix
  produces two matches and zero contradictions.
- `bind()` on a native `Adjusted R2` row label vs candidate
  `Adjusted $\text{R}^2$`: no row-label contradiction.

All difference tests were confirmed to fail against `origin/main`'s
source (via `git show origin/main:<path>` into a temporary copy of
`src/socr` — never `git stash` in this shared checkout) and pass against
the fix.

Related: #330 (binder scoping), #352.
