# GH-190 empty-table gate

## Accepting mechanism and pre-fix probes

Before the fix, `_parse_grid` dropped blank rows through its vacuous separator
test, so the issue fixture produced `_parse_grid(...) == []` and
`check_markdown(...) == []`. `table_emission_defect(...) == ""` for both the
blank-header issue fixture and the populated-header/empty-body fixture. The
final `table_output_defect(...)` therefore accepted a structurally valid table
whose body carried no content.

## Production changes and invariant

`src/socr/tables/reconcile.py` adds the raw-row `table_content_defect`
predicate and `TABLE_CONTENT_EMPTY`. It recognizes a valid header, delimiter,
and non-empty body, then treats whitespace-only cells and cells made only of
Unicode dash-property placeholders as empty while preserving symbols, zeros,
NA values, and sparse rows as content. `src/socr/tables/structure_check.py`
adds this predicate to `table_output_defect` after emission defects and before
grid-shape checks, preserving the existing `table_structure_failed`
propagation. This is a universal empty-body invariant, not a density score:
the gate fires only when every body cell is empty or a dash placeholder, so one
real value is sufficient regardless of table width or sparsity.

## Adjacent defects deliberately unchanged

`_parse_grid` still drops blank and separator-looking rows, so reconciliation
diffs cannot see them. The born-digital native lane calls
`table_emission_defect` directly and therefore does not run the new content
term. Manifest's final `_apply_table_emission_guard` and phase-major/
whole-document outputs also call `table_emission_defect` rather than
`table_output_defect`. Header+separator with no body remains outside GH-190;
label-kept/value-dropped rows are not an all-empty body and need a separate
per-row policy; and authored-grid predicates independently treat dash-only
bodies as content. None is fixed here because doing so widens the ticket or
requires an excluded file.

## Follow-up (cold review)

The content rule is width-independent: the gate fires only when every raw body
cell is empty or a Unicode dash placeholder, so a single real value preserves
the table regardless of parsed body width or sparsity. Strict GFM alignment
markers are recognized as placeholders, while symbols, zeros, `NA` values, and
sparse rows remain authored content.

The cold review deliberately reverses the earlier header+delimiter/no-body
position: a valid header and delimiter with no body is now a content defect.
The born-digital native lane also contributes this content defect to its
aggregate, and the paired pipeline e2e seam explicitly exercises OCR routing
even when the input PDF is genuinely born-digital; no-native-first remains an
explicit part of that seam.

Two adjacent cold-review gaps remain open and are intentionally unchanged:
`manifest.py`'s final guard remains emission-only, and
`has_strict_table_grid` still treats single-hyphen body cells as authored
content. Neither behavior is modified by this feature.
