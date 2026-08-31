# 2026-08-31 — GH-367: adjudication lift path for the binding clamp

Follow-up to panel #3 on GH-353/TICKET-E1 and to GH-359 ruling 5. A
`bind()` contradiction caps a table at `TABLE_UNVERIFIED`; a judge PASS
must not lift that cap. Until this ticket, nothing else could either.

Flag `--table-judge-ladder` stays default OFF. GH-326 / GH-322 stay closed.
GH-359 ruling 5's clamp is not weakened: it still fires, and an ordinary
PASS still cannot accept a contradicted table.

## Position

Disproof is exact identity under bind()'s own normalizers: a native token
is overturned only if it contains an encoding-garbage codepoint, or if an
independent cell-raster transcription of that token's bbox (which never
sees markdown or the contradiction) equals the markdown token and differs
from native. Rejected: lifting on a judge PASS or on a model verdict that
"native is corrupt" — that is the fallible override the clamp exists to
prevent — and encoding-only, which cannot catch GH-334's well-formed
inverted OCR overlay. The deciding trade-off: extra off-family transcribe
calls fire only on already-clamped tables, in exchange for a third source
that actually looks at the paint bind() never sees.

## What counts as disproof

`bind()` compares two text layers (native extract vs emitted markdown) at
a geometric locus. It never looks at pixels. Overturning it requires
evidence *about that native token* that is stronger than the comparison
itself, not a second opinion about the table.

Two signals qualify. Both are per-contradiction. Both are exact; there is
no ratio, no confidence field, no fuzzy match.

### 1. Encoding garbage (deterministic, no model)

The native token contains a codepoint that is not decoded text. The
classes are the ones `BornDigitalDetector._garbage_ratio` already names as
garbage:

- control except TAB / LF / CR
- U+FFFD replacement
- BMP private-use (`U+E000`–`U+F8FF`)
- UTF-16 surrogates

Presence of any one such codepoint is enough. Bind() compared a
codebook/error string to markdown; the comparison is invalid, not a
content disagreement. This is stronger than bind() because bind() assumed
both sides were text.

Well-formed GH-273 labels (`RowA` / `RowB`) are not garbage. GH-334's
inverted OCR (`LoIic6` for *Police*) is not garbage either — it is valid
ASCII, which is why encoding-only cannot be the whole path.

### 2. Independent raster transcription (constrained model role)

A transcriber is shown ONLY a crop of the native word's bbox (same padding
and DPI as table witnesses). It never sees the markdown, the native
string, the contradiction list, or a PASS/FAIL schema. The prompt asks
for `{"token":"..."}` and nothing else. Empty / infra / missing key → not
a disproof.

Disproof is arithmetic on three strings, using the same
`_normalize_numeric_token` / `normalize_label` bind() used to convict:

- transcribed agrees with the markdown token, AND
- transcribed disagrees with the native token

Anything else (agrees with native, third string, empty) is not a
disproof. A model that emits `{"verdict":"PASS"}` or `"the native layer is
corrupt"` does not lift anything.

This is stronger than bind() because the raster *is* the painted evidence
both text layers claim to describe. Native is a text-layer claim about
the paint; the crop is the paint. Agreeing with the pixels and not with
native falsifies the text-layer side of that one comparison.

The transcriber is off-family from the qwen extractor (same ollama-cloud
seat as ladder CLI₁, different prompt). `strict_local` skips the POST;
encoding-garbage still runs.

## Per-contradiction, not partial

The panel constraint is "disproves EACH mechanical contradiction". A
partial disproof lifts nothing: one remaining conviction still withholds
acceptance. There is no per-cell SUCCESS.

## Terminals

| Adjudication | Ladder outcome | Terminal |
| --- | --- | --- |
| lifted | ACCEPTED | ACCEPTED (clamp not applied) |
| lifted | UNVERIFIED / REJECTED | unchanged (nothing to restore) |
| held / partial / infra | ACCEPTED | `TABLE_UNVERIFIED` (clamp stays) |
| not run | REJECTED | REJECTED (ceiling on accept, not a floor on reject) |

Adjudication runs only for tables the clamp would actually withhold
(ladder ACCEPTED + bind() contradiction). Empty rungs stay UNVERIFIED
without spending transcribe calls; resume with rungs available re-runs
the gate.

A failed lift is never `TABLE_REJECTED`. REJECTED means skip-on-resume,
and native may still be the bad side. UNVERIFIED retries.

Failures surface at page (`table_ladder_disposition`), document
(`AUDIT_FAILED` / `table_unverified`), sidecar, audit log, and CLI — the
existing UNVERIFIED surfaces. A successful lift emits
`table_ladder_accepted` plus a supporting `table_binding_adjudicated`
event. `table_binding_adjudicated` is not a fourth terminal.

## Resume

The lift is recorded on `PageState.binding_adjudication` (per `table_id`)
and in the page sidecar. A record applies on resume only when:

- `status == "lifted"`
- markdown SHA-256 matches
- the contradiction signature set is identical
  (`kind, row_path, col_path, native_token, model_token`)

Fingerprint+checksum match is required before the sidecar is consulted
(a prompt/rung change already forces reprocess). UNVERIFIED pages
reprocess; the restored record is what stops bind() from silently
re-clamping a sibling table on a still-unverified page. A lifted table
that is the page's only table ships SUCCESS and skips like any other
SUCCESS page.

## Flag

No new flag. Adjudication is part of the ladder gate and runs only when
`--table-judge-ladder` is on. The ladder stays default OFF.

## Why a judge PASS still cannot lift this

GH-273: two frontier judges blessed a value-multiset-correct, row-label-
shifted table. The clamp exists because that PASS is the wrong evidence
class. The transcriber is a different class: it does not see the table,
does not vote, and cannot accept. If it independently reads the native
glyph as the native string, the clamp stays — which is the GH-273 case,
where native is the visual truth.
