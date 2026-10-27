# #1237: differential sweep for reading_agreement number normalisation

## Why

#1227 (PR #1229) needed three rounds. Two of them made a wrong number agree when `main`
disputed it, and only a reviewer caught each one:
- round 1: `$\pi^* = (9 \%)$` agreed with a bare `9`;
- round 2: `(9) \%` and `$\pi^* = 9\ \%$` dropped and the page passed.

In round 3, a main-vs-branch sweep over a generated grid found both regressions in seconds.
This turns that sweep into a standing test.

## Design

- **Grid:** `socr.devtools.reading_agreement_sweep`. Every combination of:
  - 8 wrong and 5 right numbers;
  - 12 gaps;
  - 7 wrappers;
  - 3 contexts (pipe row, prose, `tabular`);
  - `\%` inside or after the wrapper.

  That is 3,672 wrong-value readings and 2,268 right-value readings.
- **Baseline:** a recorded file, `tests/fixtures/reading_agreement_sweep/baseline.json`,
  holding `[compared, disputed]` per reading. It is stamped with the commit and a sha256
  of `reading_agreement.py`. The other option was a frozen copy of the old module. It was
  rejected because the comparator imports much of socr, so a vendored copy would drift or
  break. The cost of a recorded file: any intended change regenerates it, and the diff
  shows every reading the change moved.
- **Invariants:**
  - wrong values: no reading compares fewer numbers or raises fewer disputes than the
    baseline;
  - right values: a reading that now disputes more must equal its unspaced control under
    the current code.

  This pins a difference, not an outcome measured on one machine, and `compare_numbers`
  is pure text, so the test is the same with and without a provider.
- **Regeneration:** `socr-regenerate-reading-agreement-sweep` (`--print-only` to preview).
  The devtool imports only the standard library, per `tests/test_package_layering.py`. It
  captures from `<root>/src` in an isolated child interpreter, like
  `socr-regenerate-p6-prechange`.
- **Baseline revision:** generated at `a52598117632` (`reading_agreement.py`
  sha256 `db0b99164d10`).

## Guard shown to fail

Each mutant ran in an external copy of `src`, `tests` and `pyproject.toml`, with
`socr.__file__` asserted inside the copy and `count(anchor) == 1`:

| mutant | failing test |
| --- | --- |
| remove the #1227 percent-spacing line | wrong-value invariant |
| remove the digit anchor (Argon's round-2 regression) | wrong-value invariant |
| replace the spacing with a digit (a spaced right value then disputes) | right-value invariant |

## Gates

- Full suite: 8,995 passed, 0 failed, 48 skipped, 9 xfailed.
- `uvx ruff@0.16.0 format --check .` is clean.

## Limits

- The grid covers percent spacing only. Other writings (#1220) are not covered until rows
  are added to the grid and the baseline is regenerated.
- The test checks invariants, not correctness. An input the baseline already gets wrong
  stays wrong without failing the test.
