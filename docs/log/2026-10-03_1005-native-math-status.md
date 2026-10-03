# 2026-10-03 #1005: native-math status follows the shipped output

**Symptom.** Hameed_Morck_Shen_Yeung p9 (#960 A/B, B989): qwen read ships (manifest ending
`model_output`, audit_passed true) yet status `warning`, note and `native_math_unrecovered` event
"native text shipped with unmapped math glyphs ... no recovery evidence was retained".

**Root cause.** `_apply_unresolved_math_guard` (core/manifest.py) and the per-page reduction in
`_phase_assemble` (pipeline/orchestrator.py) reduced `has_unmapped_math_glyphs` + `math_recovery_evidence`
without asking what ships. A model page has no recovery evidence, so the "no evidence" branch fired on a body
with no native bytes. Same shape at document level: `unresolved_math_pages` forced AUDIT_FAILED.

**Fix.** `native_math_damage_ships(output, provenance)` in manifest.py: False only when the selection
provenance is a model-reading ending, the engine is not native/chart_asset, and the body has no private-use
codepoint left. Unknown / possibly-native endings (hybrid, unverified/flagged attempts, whole-doc) keep the
warning. Used by the guard and by the assemble reduction (including the witness fallback). `audit_passed` untouched.

**Siblings.** `native_minus_as_digit`, `native_invisible_text_scan`, `native_garbled_math`,
`native_untrusted_judge_timeout` are set only at native-ship sites (synthetic fallback return, chart lane) and
their document lists require a native best_output or native fallback: not affected. Tests pin this.
`native_math_font_unrecovered` is note-only and left alone.

**Tests.** `tests/test_gh1005_native_status_follows_shipped.py` (6): native vs model on the same flagged page
for all four damage classes, document-level event/status, and a model body still holding PUA keeps the warning.
`test_math_reporting_never_changes_which_table_ships` encoded the bug (clean model table -> WARNING); updated.
Mutant (external copy, socr.__file__ canary, anchor count 1, guard forced True): 2 failures. Full suite 6759
passed; ruff format clean.
