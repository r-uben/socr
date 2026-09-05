# GH-585: binder still convicts sibling LaTeX presentation after GH-582 wrap fix

Three review rounds on `fix/585-sibling-latex`, each catching a new widening the
previous round's fix introduced. Seam: `strip_math_presentation(tok, *, label=)` in
`src/socr/tables/native_verifier.py`, shared by the binder's row-label compare
(`binding.py`) and `adjudication.tokens_agree`.

## Round 1 (`134f5c1`)

Added three deterministic mapping classes before the `normalize_label` compare:
Greek LaTeX commands → their Unicode letters (`\Delta` → `∆`), `\&`/`\%`/`\_`/`\$`/`\#`
unescaping, and dropping the backslash of alphabetic word commands (`\log` → `log`).
Kept the existing bare-symbolic-label fail-closed rule (`β` vs `$\beta$` stays
UNVERIFIABLE). C2b prediction artifact re-derived at `73818b0` (10 remaining items,
1 addressed).

## Round 2 (`ba5c6a0`)

Reviewer found four further gaps in round 1's fix:

1. Plain Greek-command→Unicode mapping alone reintroduced a new false agreement:
   `normalize_label`'s ASCII-only filter strips ALL non-ASCII, including any Greek
   letter, so `α Coefficient` and `$\beta$ Coefficient` (different letters) both
   folded to `"coefficient"`. Fix: transliterate every Greek Unicode letter to its own
   ASCII name, still exempting the bare-symbol-only case.
2. `strip_math_presentation(label=False)` (the cell/numeric path) never unescaped
   `\&`/`\%`/etc., so `12\%` was invisible to the numeric tokenizer. Fix: extend the
   escape unmap to `label=False` too.
3. The word-command regex was an unbounded `\\[A-Za-z]+`, so an unsupported/unverified
   command like `\logx` would also lose its backslash and gain an agreement it never
   earned. Fix: replace with an explicit allow-list of standard LaTeX operator names
   (`_MATH_OPERATOR_NAMES`).
4. `\varepsilon`/`\epsilon` (and other var-forms) were folded to the same base letter;
   this equivalence is not established in the corpus. Fix: each variant transliterates
   to its own distinct name (`greekvarepsilon` etc.), never the base letter's. Only
   `Δ`(U+0394)/`∆`(U+2206) may alias, since they render as the identical printed glyph.

## Round 3 (this commit)

Two further findings:

1. Round 2's plain-ASCII-word transliteration (`∆` → `"Delta"`) was itself a widening:
   a label that literally spells the prose word "Delta" would falsely agree with
   native `∆`, and `normalize_label`'s lowercasing folds `Δ`/`δ` (different letters)
   together. Fix: replace plain names with typed, case-preserving tokens that cannot
   collide with prose — `greek`-prefixed for lowercase (`greekalpha`), `greekcap`-
   prefixed for uppercase (`greekcapdelta`), covering every letter of the Greek
   alphabet programmatically (via the `+0x20` codepoint offset), not just the 11 with
   dedicated LaTeX commands.
2. Writing the required `\logx` end-to-end regression test (through both
   `tokens_agree` and `bind()`) surfaced a second, deeper gap: retaining `\logx`'s
   backslash at the `strip_math_presentation` helper was not actually sufficient
   evidence, because `normalize_label`'s `_NON_ALNUM_RE` strips ANY leftover backslash
   unconditionally, downstream of this helper — so `\logx` and the bare word `logx`
   still folded to the same key regardless of the round-2 operator allow-list. Fix:
   after the Greek and operator maps run, any remaining `\<word>` is folded to an
   `unmapped`-prefixed alphanumeric token (`\logx` → `unmappedlogx`) that cannot equal
   either the bare word or a real operator of similar spelling.

## Files changed (round 3)

- `src/socr/tables/native_verifier.py` — case-preserving Greek token table
  (`_GREEK_UNICODE_TOKEN`), `_MATH_UNMAPPED_COMMAND_RE` catch-all.
- `tests/test_gh103_tokenizer_presentation.py` — updated variant-command literal
  strings, prose-collision test, case-distinction test, unmapped-command test.
- `tests/test_binding_adjudication.py` — `\logx` end-to-end via `tokens_agree`.
- `tests/test_binding.py` — `\logx` end-to-end via `bind()`.

## Verification

- Targeted (`test_gh103_tokenizer_presentation.py` + `test_binding.py` +
  `test_binding_adjudication.py`): 164 passed, 1 xfailed (pre-existing).
- Full suite: see commit message / team-lead report for the run captured at commit
  time.
- Frozen-corpus harness (`socr.benchmark.replay_binding` on
  `~/Data/socr/ladder-run2-2026-09-04`): doc05 recorded 6 → fresh 4, doc07 recorded
  5 → fresh 3, the genuine `S&P` text-difference pair still contradicts on both docs,
  C2b frozen prediction PASS against the `73818b0` artifact.
- `uvx ruff@0.16.0 format --check .` clean.
