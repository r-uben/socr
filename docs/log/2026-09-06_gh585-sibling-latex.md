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

## Round 4 (this commit)

Codex seat reviewed `4d1b35a` (round 3) and returned NO: round 3's fix was still a
flat-STRING sentinel folded into `normalize_label`'s key, not a structural type
distinction. Two collisions follow directly from that:

1. `greekalpha`/`unmappedlogx` are ordinary characters once inside the flattened
   string key, so literal prose that happens to spell the sentinel collides with the
   real Greek letter or unmapped command: `α Coefficient` falsely agreed with typed
   `greekalpha Coefficient`, and `\logx y` falsely agreed with typed
   `unmappedlogx y`.
2. Six Greek variant glyphs that a PDF's text layer can emit directly — `ϑ` (U+03D1),
   `ϕ` (U+03D5), `ϖ` (U+03D6), `ϱ` (U+03F1), `ς` (U+03C2), `ϵ` (U+03F5) — were never
   mapped to their `\vartheta`/`\varphi`/`\varpi`/`\varrho`/`\varsigma`/`\varepsilon`
   command counterparts, so native text using these codepoints falsely contradicted
   labels written with the LaTeX command.

Fix: replace the flat-string key with a **structured** one. New
`label_key(text) -> tuple[tuple[str, str], ...]` in `native_verifier.py` tokenizes
presentation-stripped text into typed 2-tuples — `("lit", normalize_label(chunk))`
for literal runs, `("greek", tag)` for a recognized Greek letter (from LaTeX command
or Unicode glyph, either case, any variant, including the six codepoints above),
`("cmd", name)` for an unrecognized backslash command. Two labels agree iff their
tuples are equal; a type tag can never collide with a literal string because they
live in different tuple positions, not the same character stream.
`strip_math_presentation` reverts to ONLY its GH-582 wrap/escape/script duties — all
Greek/word-command substitution moves to `label_key`. `label_key_is_bare_symbolic`
replaces the old empty-string check for the bare-symbol UNVERIFIABLE rule (a key
that is exactly one `("greek", …)` token and nothing else).

`binding.py`'s row-label compare and `adjudication.tokens_agree(kind="row_label")`
now compare `label_key()` tuples. `binding.py`'s `_fallback_pair_allowed` row-pairing
helper was *also* switched to `label_key` equality, beyond the reviewer's literally
named call site: it used the same `normalize_label(strip_math_presentation(...))`
composition, and leaving it on the old helper (now stripped of all Greek/command
handling) would have silently regressed row-pairing to treat every Greek letter as
indistinguishable from every other — worse than any prior round's bug, not merely
unfixed by this one.

Round-3 correction: round 3 aliased `ς` (final sigma) to the same tag as base `σ`,
reasoning it was a positional glyph variant. Round 4's review named `ς` as one of the
six codepoints requiring its own `varsigma` tag, distinct from base sigma; this log
adopts that correction.

## Round 5 (this commit)

Codex seat reviewed `8603981` (round 4) and returned NO: `label_key`'s
`flush_literal()` ran `normalize_label` on every literal chunk it emitted,
including INTERNAL chunks — ones that end because a Greek/command token
follows, not because the label itself ends. `normalize_label`'s trailing-
footnote-suffix rule (a bare digit right after a letter/bracket at the end of
the string it is given) is a WHOLE-LABEL rule; applied to an internal chunk it
discards a real digit that is part of the chunk's own content. `"Model1 α"`
and `"Model2 α"` both keyed to `(("lit", "model"), ("greek", "alpha"))` — a
false agreement the digit should have prevented.

Fix: split `normalize_label` (`native_rows.py`) into a shared
`_fold_emphasis_and_markers` helper plus two public entry points —
`normalize_label` (unchanged behavior: emphasis/marker fold, suffix rule,
punctuation strip) and new `normalize_label_chunk` (same fold and punctuation
strip, no suffix rule). `label_key`'s `flush_literal` now takes a `final: bool`
flag: every call triggered mid-scan by an upcoming Greek/cmd/greekchar token
uses `normalize_label_chunk` (internal chunk, suffix rule withheld); only the
final flush after the token scan — the chunk that actually reaches the
label's end — uses `normalize_label` (suffix rule applies). No caller outside
`label_key` was touched; `normalize_label`'s existing behavior for every other
caller (GH-96 exactness, the escalation canary, `_fallback_pair_allowed`'s use
via `label_key`) is unchanged.

Negative control: `Model1 α` vs `Model2 α` no longer agree (through
`tokens_agree` and `bind()`). Positive control: a genuine trailing footnote
digit on the label's own end (`Coefficient1` vs `Coefficient`) still agrees,
unchanged from before this fix. Every earlier-round control still holds.

## Files changed (round 5)

- `src/socr/tables/native_rows.py` — `_fold_emphasis_and_markers` extracted;
  new `normalize_label_chunk` (no suffix rule); `normalize_label` unchanged
  in behavior, refactored to reuse the shared fold.
- `src/socr/tables/native_verifier.py` — `label_key`'s `flush_literal` takes
  `final: bool`; internal chunks use `normalize_label_chunk`, only the final
  chunk uses `normalize_label`.
- `tests/test_gh103_tokenizer_presentation.py` — internal-chunk-keeps-digit
  and final-chunk-still-folds unit tests on `label_key`.
- `tests/test_binding_adjudication.py` — same two controls via `tokens_agree`.
- `tests/test_binding.py` — same two controls end-to-end via `bind()`.

## Verification (round 5)

- Targeted (`test_gh103_tokenizer_presentation.py` + `test_binding.py` +
  `test_binding_adjudication.py`): 180 passed, 1 xfailed (pre-existing).
- Broader native/table-metric suite (`test_corpus_rescore_gate.py`,
  `test_gh331_stub_labels.py`, `test_gh367_adjudication_lift.py`,
  `test_metric_corruption_battery.py`, `test_native_table_verifier.py`,
  `test_package_layering.py`, `test_source_evidence_table_judge.py`,
  `test_table_header_gh146.py`, `test_tr1_rowizer.py`): 196 passed.
- Frozen-corpus harness: doc05 recorded 6 → fresh 4, doc07 recorded 5 → fresh
  3, unchanged from rounds 3–4; C2b frozen prediction PASS against the
  `73818b0` artifact.
- `uvx ruff@0.16.0 format --check .` clean (580 files).
- Full suite: see commit message / team-lead report for the run captured at
  commit time.

## Files changed (round 4)

- `src/socr/tables/native_verifier.py` — `label_key`, `label_key_is_bare_symbolic`,
  `_GREEK_COMMAND_TAG`, `_GREEK_UNICODE_TAG`, `_LABEL_TOKEN_RE`;
  `strip_math_presentation` reverted to wrap/escape/script only.
- `src/socr/tables/binding.py` — row-label compare and `_fallback_pair_allowed` use
  `label_key`/`label_key_is_bare_symbolic`.
- `src/socr/tables/adjudication.py` — `tokens_agree(kind="row_label")` uses
  `label_key`/`label_key_is_bare_symbolic`.
- `tests/test_gh103_tokenizer_presentation.py` — rewritten GH-585 section asserts on
  `label_key()` tuples instead of `strip_math_presentation` string output; adds the
  six-variant-glyph parametrized test.
- `tests/test_binding_adjudication.py` — sentinel-collision and variant-glyph
  end-to-end controls via `tokens_agree`.
- `tests/test_binding.py` — the same two controls end-to-end via `bind()`.

## Verification (round 4)

- Targeted (`test_gh103_tokenizer_presentation.py` + `test_binding.py` +
  `test_binding_adjudication.py`): 174 passed, 1 xfailed (pre-existing).
- Frozen-corpus harness: doc05 recorded 6 → fresh 4, doc07 recorded 5 → fresh 3,
  unchanged from round 3; C2b frozen prediction PASS against the `73818b0` artifact.
- `uvx ruff@0.16.0 format --check .` clean (580 files).
- Full suite: see commit message / team-lead report for the run captured at commit
  time.

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
