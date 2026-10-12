# #1087: the distrusted-layer rescue (#658b) is removed

## Defects (both reproduced on origin/main 9ad04bb1)

1. `_corroborate_via_distrusted_layer` never looked at numbers in the model table's header
   row: header `| label | 999 |`, row `-5`, layer `-5`, silent OCR, `native_trusted=False`
   passed with `header_binding_unverified`, shipping `999`.
2. `RowCorroboration.clears` tolerates a small share of unsupported numbers when most rows
   match, so 49 correct rows plus one row with an unprinted number passed.

## Three review rounds, each beaten

1. Reuse `collect_table_tokens` + multiset: it sees only whole-cell numbers in recognised
   tables and folds `(5)`, `5`, `-5`. Escaped: `Year 999`, `888kg`, `Total: 999.`,
   `| lone | 999 |`, `(5)` vs `-5`.
2. Rescue-local regex over the whole output, sign presentation kept. Escaped: `1e3` (vs
   layer `1`,`3`), `1/2`, superscripts, `− 5`, `1,5` vs `15`, a layer word reported twice.
3. Decline unless every digit-bearing token is a plain number, dedupe layer words. Escaped:
   `½` (not `isdigit`), `1 / 2`, `( 5 )`, `5 %`, overlaid `-5` / `−5` layer words.

Every number grammar was bypassed by the next review, so parsing numbers was abandoned.
A fourth attempt (decline on any numeric character) made the rescue unreachable: `clears`
abstains without numeric rows, so a numeral-free table is never rescued either.

## Owner ruling (2026-10-12): remove the rescue

- With #1080's preflight a missing Tesseract stops the run.
- The rescue rescued 0 of the 181 stored pages.
- The pages it would handle are "could not check" tables that #1048's ruling will ship
  marked "unverified" once built; that, not a guarded rescue, is the intended lane.

## Fix

Deleted `_corroborate_via_distrusted_layer`, its call in `verify_scanned_table`, and the
helpers that became unused. When the evidence bundle is empty, `verify_scanned_table` returns
`verify_table_tokens(bundle, tokens)` as it did before 65551586 (#658b): `verifiable=False`,
`passed=False`, reason `no local content evidence available for scanned table`, no flag.

Kept: `corroborate_rows` / `clears` (core/manifest.py uses them; `HEADER_BINDING_UNVERIFIED`
is still produced there), and #1085's evidence-side rules (`_is_dash_led`,
`ambiguous_numeric`), which the main check still uses.

## Tests

- `test_gh658b_distrusted_layer_corroboration.py` rewritten: matching layer, disagreeing
  layer and no layer all fail closed identically.
- `test_gh163_scanned_native_trust.py` and `test_gh1085_evidence_leftovers.py`: rescue
  assertions became "fails closed". The `bool(ambiguous)` guard is checked on the bundle
  (a dash-led reading has `has_content_evidence`, silence does not), because both arms now
  reject and a verdict cannot tell them apart.
- `test_gh1087_rescue_numbers.py`: one test per repro (header 999, 49+1, mixed cell, prose)
  plus `clears` still unchanged.

## Measurement

`step-1087/measure.py` replays the 181 stored readings: 2 pages reach the old rescue point
(empty witness, table fails), both declined by the rescue before the removal; 179 have a
witness reading. The rescue rescued 0 pages, so removing it changes no stored page.

## Mutants (`step-1087/mutants.py`; external copy, canary asserted, anchors ==1)

- Rescue re-introduced (passes when the layer covers the rows): the #1087 repro tests and
  the 658b tests fail.
- `ambiguous` dropped from evidence presence (#1085): the bundle test fails.
