# GH-206: `is_numeric_token` notation gaps

## What changed

`src/socr/tables/native_verifier.py` — `strip_presentation` (and therefore
`is_numeric_token` / `_normalize_numeric_token`, which both call it) now closes
two gaps:

1. **Leading-decimal values.** `.034` (no leading zero — a normal way to print
   a coefficient below 1) is inserted with the missing `0` via a new
   `_LEADING_DECIMAL_RE`, anchored at the start of the token so it cannot touch
   a decimal point elsewhere in the string. Handles the bracket/minus-wrapped
   forms too: `(.034)` → `(0.034)`, `-.034` → `-0.034`.
2. **Unicode significance-star / footnote-mark glyphs.** Added a named,
   documented `_PRESENTATION_MARKS` tuple: ASCII `*`/`_`/`**`/`__` (already
   handled) plus `∗` (U+2217 ASTERISK OPERATOR, what LaTeX `\ast` actually
   emits), `⁎` (U+204E LOW ASTERISK), `✱` (U+2731 HEAVY ASTERISK), `†`
   (U+2020 DAGGER), `‡` (U+2021 DOUBLE DAGGER), `§` (U+00A7 SECTION SIGN).
   Deliberately excludes superscript letter footnote refs (a, b, ...) — those
   decorate prose cells too, and stripping them would loosen the predicate
   into admitting label tokens, which the issue explicitly forbids.

## Probe (issue reproduction)

Before (on `main`, `ce2d84d`):
```
'.034'      False
'0.67∗∗∗'   False
```

After:
```
'.034'      True
'0.67∗∗∗'   True
```
All previously-True cases in the issue's probe (`0.67`, `0.67***`, `−0.45`,
`$1.2`, `45%`, `(0.014)`, `1,234`) remain True.

## Negative cases tried (must stay rejected)

`p<.05`, `e.g.`, `N/A`, `..034` (malformed), `.a`, `Revenue`, `**Revenue**`,
`of which:`, `""`, `---`, `—` — all still `False`. (`1.5.6` is `True` both
before and after this change — pre-existing looseness in `_NUM_TOKEN_RE`
unrelated to this ticket, not touched.)

## Tests

Extended `tests/test_gh103_tokenizer_presentation.py` (natural home — same
`strip_presentation` machinery, same GH-103 provenance) with parametrized
cases for both gaps, a normalization-symmetry test (`.034` vs `0.034`), and an
explicit "still rejects prose" test.

## Verification

- `~/venvs/socr/bin/pytest tests/test_gh103_tokenizer_presentation.py -q` → 49 passed.
- `~/venvs/socr/bin/pytest tests/ -q` → **1611 passed, 1 xfailed** (baseline on
  `main` was 1591 passed / 1 xfailed; the +20 are the new parametrized cases).
- `uvx ruff@0.16.0 format --check .` → 288 files already formatted, clean.

## Skipped

The issue asks, if cheap, to quantify the before/after TR-3 firing-rate change
on the 32-paper / 245-page corpus behind the 25.3% figure in #205. I could not
find that corpus or the tooling that produced the figure without a
non-trivial search (the source PDFs, per the ticket brief, are split across
an iCloud path with evicted files and a ProtonDrive path, neither complete
alone), so this is not the "cheap" case the issue anticipated. Skipped;
flagged for whoever picks up #205 to re-run with the corpus/tooling they used
originally.
