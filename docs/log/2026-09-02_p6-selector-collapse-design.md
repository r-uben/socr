# P6 — collapse the 15-ending selector to three (design)

Ticket: programme item P6 of `docs/log/2026-09-01_conceptual-revision.md`. Closes #176; the
real lever of #155. Worktree `/Users/rubenffuertes/repos/tools/socr-p6`, base `7e25e4a`
(P0, P2, P3, P4, P5 landed). READ-ONLY except this file.

## 0. Grounding canary

```
python3 -c "import ast;t=ast.parse(open('src/socr/core/manifest.py').read());f=[n for n in ast.walk(t) if isinstance(n,ast.FunctionDef) and n.name=='_select_page_output_tagged'][0];print(f.lineno, f.end_lineno, len([r for r in ast.walk(f) if isinstance(r,ast.Return)]))"
```

`def _select_page_output_tagged` is at **`src/socr/core/manifest.py:997`**, ends at 1472
(476 lines), with **15 `return` statements**. `tests/test_r7_winner_kind_tags.py:65` pins
that count, and `test_tag_order_matches_enum_declaration_order` pins source order against
`WinnerKind` declaration order.

## 1. The 15 endings

Order is precedence. "→" is the ruling's ending: **N** native prose, **M** accepted model
output, **F** fail-closed marker.

| # | WinnerKind | Reached when | → |
|---|---|---|---|
| 1 | `CORRUPT_MATH_HYBRID` (`:1036`) | `p.corrupt_math_hybrid` set, engine `native+math`, in `attempts`, not shredded, no table-distrust flag | M (flagged) |
| 2 | `PASSING_BEST_OUTPUT` (`:1085`) | `best_output.audit_passed` and not (native winner + structure-class/table-distrust) and not (native-lane winner + `native_rotated_text_shredded`) | M |
| 3 | `UNVERIFIABLE_TABLE_SCANNED` (`:1103`) | not born-digital, `scanned_table_evidence_failed`, `attempts` | F |
| 4 | `UNVERIFIABLE_TABLE_MODEL_KEPT` (`:1150`) | born-digital + D3 conjunction, and `d3_floor_kept_model_output(p)` is not None | M (flagged) |
| 5 | `UNVERIFIABLE_TABLE_NATIVE` (`:1204`) | born-digital, `native_table_structure_failed` and (`native_table_unverifiable` or `native_table_header_unattributed`), `attempts`, no kept model | F |
| 6 | `ROTATED_TEXT_SHREDDED` (`:1231`) | born-digital + `native_rotated_text_shredded` (no attempts gate) | F |
| 7 | `FLAGGED_MODEL_KEPT` (`:1252`) | born-digital, `flagged_model_page_output(p)` is not None | M (flagged) |
| 8 | `STRUCTURE_CLASS_GRID_PASSING` (`:1325`) | `_reaches_structure_class_branch(p)` and a grid winner with `audit_passed` | M |
| 9 | `STRUCTURE_CLASS_GRID_FLAGGED` (`:1330`) | same, grid winner not passing (soft-reject allowlist) | M (flagged) |
| 10 | `STRUCTURE_CLASS_FLOOR` (`:1349`) | same branch, no grid winner (P2's ship path) | F |
| 11 | `NATIVE_FALLBACK` (`:1399`, demoted arm) | born-digital + native text, `native_demoted` = (`needs_ocr_enhancement` or any native-table defect or `chart_asset_render_failed`) and `attempts`, or `text_grid_rejected` | **neither** — see §5 |
| 12 | `NATIVE_CLEAN` (`:1399`, undemoted arm) | same return, `native_demoted` false | N |
| 13 | `WHOLE_DOC_SECTION` (`:1429`) | non-empty section for this page in the whole-doc split | M (or F if audit failed) |
| 14 | `BEST_OUTPUT_UNVERIFIED` (`:1439`) | `best_output` present but not passing | M (flagged) |
| 15 | `BEST_ATTEMPT_FLAGGED` (`:1446`) | `best_output` cleared, `best_attempt` present | M (flagged) |
| 16 | `NO_TEXT_MARKER` (`:1467`) | nothing produced text | F |

Sixteen rows, fifteen returns: #11 and #12 share return `:1399` via a conditional tag,
which the bijection test explicitly allows (`_tag_names`, `tests/test_r7_winner_kind_tags.py:34`).

**A seventeenth disposition exists outside the cascade.** `_apply_table_emission_guard`
(`manifest.py:1475`) can rewrite ANY ending's text into
`[page N failed: invalid table emission — …]` after the tag is dropped. `shipped_winner_kind`'s
own docstring (`:984`) warns the tag names the ending, not the shipped bytes. Any
tag-derived bucket scheme must state which of the two it means.

**The one ending that does not map: `NATIVE_FALLBACK`.** Ruling step 5 says native "is never
the consolation prize when the ladder is exhausted". P2 enforced that only for
structure-class pages. `NATIVE_FALLBACK` still ships demoted native prose for
`needs_ocr_enhancement`, `chart_asset_render_failed`, `text_grid_rejected` and the residual
native-table defects. Under the ruling it is either N (it is prose, ship it clean, drop the
WARNING) or F (the ladder was exhausted, floor it). That is the first fork.

## 2. Assemble buckets

All in `_phase_assemble`, `src/socr/pipeline/orchestrator.py:7511-7872`.

| Bucket | Derived from today | Tag-derivable? |
|---|---|---|
| `failed_pages` `:7511` | `is_page_failed_marker(shipped text)` | shipped text, not tag — keep as-is; it is the class-level guard #293 added |
| `d3_model_table_pages` `:7533` | `d3_floor_kept_model_output(p)` | yes = `UNVERIFIABLE_TABLE_MODEL_KEPT` |
| `d3_floor_pages` `:7536` | 5-term flag predicate + exclusion | yes = `UNVERIFIABLE_TABLE_NATIVE` |
| `native_only_distrust_pages` `:7557` | 7-term predicate incl. `config.native_only` | no — needs config, not on the PageOutput |
| `flagged_model_pages` `:7576` | `flagged_model_page_output(p)` | yes = `FLAGGED_MODEL_KEPT` |
| `structure_class_model_pages` `:7591` | `structure_class_grid_winner(p)` | yes = the two `STRUCTURE_CLASS_GRID_*` tags |
| `structure_class_floor_pages` `:7602` | `structure_class_floor_applies(p)` | yes = `STRUCTURE_CLASS_FLOOR` |
| `value_drift_pages` `:7618` | audit events | no — event-derived, no cascade branch |
| `corrupt_math_hybrid_pages` `:7652` | **already `shipped_winner_kind`** (GH-292) | yes, done |
| `native_fallback_pages` `:7656` | 6-term include + 7 exclusions | yes = `NATIVE_FALLBACK`, and this is where the ~90 lines of exclusion commentary die |
| `fabricated_ref_pages` / `doc_fabrication` `:7825` | PageState counter / events | no |
| `text_grid_rejected_pages` `:7828` | `p.text_grid_rejected` | partly — it is one arm of `native_demoted`, so the tag alone cannot separate it |
| `chart_detection_failed_pages` `:7840` | `p.chart_asset_detection_failed` | no |
| `table_rejected_pages` / `table_unverified_pages` `:7862` | `_table_ladder_terminal(p)` | no — post-selection disposition |

Eight of fourteen collapse to a tag comparison. Six are orthogonal to selection (config,
events, post-selection dispositions) and must stay flag-derived. **So "assemble derives
buckets from the tag" is true for the selection-shaped buckets only**, and the note the
orchestrator needs is that the remaining six are not selection dispositions at all.

Critically: if the enum collapses to three members, eight buckets lose their key. The
audit events and CLI lines at `:7908-8154` name distinct dispositions each traceable to a
filed issue. Collapsing the tag without carrying a **reason** field would be exactly the
silent-loss failure CLAUDE.md forbids.

## 3. Resume / sidecar / replay contract

- `_flush_page_sidecar` (`:6385`) serialises the **winning** output (GH-56), not
  `best_output`, plus PageState decision flags. Ending identity is not stored; it is
  re-derived on read.
- `_load_terminal_page` (`:6663`) grants skip iff terminal + fingerprint + checksum match
  **and** status is `SUCCESS` **and** `audit_passed` is True and the body is not a failure
  marker — with one exception for `TABLE_REJECTED` without `table_ladder_incomplete`.
  Consequence per ending: only `PASSING_BEST_OUTPUT`, `NATIVE_CLEAN`,
  `STRUCTURE_CLASS_GRID_PASSING` and a passing `WHOLE_DOC_SECTION` are skippable. All F
  endings and all flagged-M endings re-OCR on resume. **Any merge that changes a page's
  status or `audit_passed` changes resume behaviour**, which is a correctness surface, not
  cosmetics.
- Replay: `build_manifest` freezes `_winning_page_output`; `canonical_page_texts` feeds both
  the saved `.md` and the blobs, so text changes are bit-visible in replay.
- Writing the tag into the sidecar would be new provenance and would change the sidecar
  schema; it does not affect `run_fingerprint` (which covers model/prompt/render/flags), so
  old sidecars stay valid.

## 4. Staged plan

**S1 — tag-derived buckets, zero behaviour change.** Replace the eight derivable predicates
with `shipped_winner_kind` comparisons. Pin: for each of the eight, a test computing old
predicate and new tag over the same states and asserting set equality (the GH-292 shape,
`tests/test_gh292_hybrid_bucket_matches_the_tag.py`). Plus one byte-identity run of the
golden corpus asserting `canonical_page_texts` unchanged. LOC:
`git diff --stat main -- src/socr/pipeline/orchestrator.py`. Expect ~150 deleted, mostly the
`native_fallback_pages` exclusion chain.

**S2 — carry a reason.** Add `WinnerKind` (or its successor) plus a `reason` to the sidecar
and to a new `PageDisposition` record returned alongside the output. No selection change.
Pin: sidecar round-trip; `_load_terminal_page` decisions identical with and without the new
field (a field that changes nothing must not invalidate — the GH-525 precedent,
`tests/test_gh525_inert_fields_do_not_invalidate.py`).

**S3 — merge the F family.** `UNVERIFIABLE_TABLE_SCANNED`, `UNVERIFIABLE_TABLE_NATIVE`,
`ROTATED_TEXT_SHREDDED`, `STRUCTURE_CLASS_FLOOR`, `NO_TEXT_MARKER` become one
`FAIL_CLOSED_MARKER` ending with a reason. The marker TEXT differs per reason and must stay
byte-identical. Pin: byte-identity on each marker's fixture, plus difference tests that only
the enum member moved.

**S4 — merge the M family.** The six flagged-kept endings become `MODEL_OUTPUT` with
`accepted: bool` and a reason. Highest risk: each carries its own `status`/`failure_mode`
demotion and its own in-body note (`kept_table_flag_note`, `d3_superseded_note`). Pin per
ending: same text, same status, same `failure_mode`, same resume decision.

**S5 — resolve `NATIVE_FALLBACK`** per the panel's answer to Q1. This is the only stage
that changes shipped bytes.

Total expected deletion 400-500 LOC, consistent with the programme estimate. Measure with
`git diff --stat main -- src/socr/core/manifest.py src/socr/pipeline/orchestrator.py`.

## 5. Risks — what each merge loses

- **F merge:** loses the distinction between "we had a reading and refused it" (D3, scanned)
  and "we never had one" (`NO_TEXT_MARKER`). Acceptable only if the reason field is
  surfaced in metadata and CLI; the marker text already differs.
- **M merge:** loses the ordering argument that `CORRUPT_MATH_HYBRID` outranks the
  model-kept endings, which `WinnerKind`'s docstring says nothing else records. Precedence
  must move into an explicit ordered reason list or it is invented anew later.
- **`BEST_OUTPUT_UNVERIFIED` vs `BEST_ATTEMPT_FLAGGED`:** #158's provider identity is
  forwarded only on the latter. Merging without carrying provider fields re-opens #158.
- **`NATIVE_FALLBACK` → floor:** deletes real prose from born-digital `needs_ocr_enhancement`
  pages that today ship readable text. That is content loss, defensible only if the ruling's
  "never a consolation prize" is read as covering prose too, which §1 of the ruling says it
  does not (native authors prose).
- **`native_only_distrust_pages`, ladder terminals, value drift, fabrication:** untouched by
  any merge; they are not selection dispositions and must not be folded in.

## 6. Sharp questions for the panel

**Q1.** Under the ruling, what happens to `NATIVE_FALLBACK` — the born-digital page whose
native prose is intact but whose enhancement/chart-render/grid-rejection flag fired, that
today ships native text demoted to `WARNING` / `audit_passed=False`?

- (a) **Ship it as native prose, clean.** The ruling says native authors prose; the flags
  concern structure, and demoting prose for a structure flag is what created the bucket
  sprawl. Against: it silently upgrades pages that today drive the document to
  `AUDIT_FAILED`, and it makes them resume-skippable, so a later correctness fix never
  re-runs them.
- (b) **Keep the demoted-native ending as a fourth ending.** It is honest about a real,
  distinct state. Against: the ruling said three, and this is the ending that carries the
  most bucket machinery.
- (c) **Floor it.** Consistent with P2. Against: it deletes readable prose that nothing has
  said is wrong, which is content loss to buy consistency.

**Q2.** Should the three endings be a bare enum, or an ending plus a required `reason`
carried into the sidecar and the manifest?

- (a) **Bare three.** Maximum deletion; assemble emits three audit kinds. Against: eight
  buckets and their per-issue CLI lines lose their key, and a reader can no longer tell a
  refused-reading floor from a nothing-was-produced floor.
- (b) **Three endings plus a reason enum.** Keeps every existing surface; the deletion comes
  from the predicates, not the vocabulary. Against: the reason enum is the old 15 members
  under a new name, so #176's "flag sprawl" is renamed rather than removed.

**Q3.** Does the tag name the ending SELECTION took, or the bytes that SHIPPED, given that
`_apply_table_emission_guard` can turn any ending into a failure marker after the tag is
fixed?

- (a) **Selection.** Status quo; `failed_pages` stays text-derived, and the two surfaces are
  documented as different questions. Against: it is the exact "counted under a disposition
  it does not have" bug class R7 exists to kill, only moved one seam later.
- (b) **Shipped.** Move the guards inside the cascade so the tag is computed after them.
  Against: the guards are deliberately the last backstop shared by whole-doc CLI paths that
  never reach the cascade's branches; folding them in changes what runs on those paths.

## 7. Acceptance-criteria readiness

#176's four criteria: "one authoritative selector" is already true (`_winning_page_output`)
and needs restating as "one selector with three endings". "`DocumentState.text` deleted or a
thin wrapper" is independent of P6 and implementable today. "Repair/routing policy moves out
of `PageState`" is NOT implementable as written until Q1 settles, because `needs_repair` and
the native-distrust flags are the inputs to the ending being merged. "Blackboard fields
documented as facts vs decisions" is implementable and should follow S2, where the
fact/decision line becomes concrete.

## 8. Panel (Codex gpt-5.6-sol, Gemini) and synthesis — 2026-09-02

**Corrections to this note.** Both panelists reject the "eight of fourteen buckets" count:
buckets and tags were conflated. Codex's inventory: six directly tag-derivable buckets,
six orthogonal ones (config, value-drift and fabrication events, text-grid rejection,
chart-detection failure, ladder terminals), and two leftovers — `failed_pages` (derived
from final text) and `native_fallback_pages` (which is not `NATIVE_FALLBACK`). Claim (ii),
that the emission guard rewrites text after the tag is fixed, is confirmed by both.

**Q1 — `NATIVE_FALLBACK`: both keep it, as a fourth ending, with a removal criterion.**
Neither promotion to clean nor blanket flooring is a refactor; both are policy changes
(silent upgrade that makes suspect pages resume-skippable, or deletion of readable prose).
Ruling: keep it, and attach the exit condition Codex names — enumerate the corpus pages
by trigger (`needs_ocr_enhancement`, `chart_asset_render_failed`, `text_grid_rejected`,
residual native-table defects), hand-check each trigger's fidelity, and assign each
trigger independently to N or F in a later ticket. The fourth ending is a measured
deviation from the three-ending ruling, recorded as such.

**Q2 — reason enum: both say three endings plus a typed primary reason** carried into the
sidecar and manifest, with orthogonal alerts kept outside it. Codex's test for whether it
is renaming rather than removing: merge only rows identical on all five surfaces (text,
status, failure mode, CLI/audit event, resume decision). Ruling: adopt, with that matrix
as the merge criterion.

**Q3 — selection vs shipped: split.** Gemini keeps the tag as selection and relies on the
text-derived `failed_pages`; Codex makes the public disposition name what SHIPPED, keeps
selection provenance as a separate internal field, and computes the final disposition
after the shared guard without moving the guard. Ruling: **Codex** — a "passing model"
tag beside a shipped failure marker is the exact misclassification class R7 exists to
kill; a final disposition computed after the guard closes it without touching the
whole-doc paths the guard protects.

**Staging, amended per Codex:** define the finalization-aware disposition + reason
contract FIRST (S2 before S1), then derive buckets from it with a difference test against
today's buckets over the golden corpus, then merge the fail-closed family, then the model
family. Every stage byte-identity-pinned and difference-pinned. Nothing in the first two
stages changes shipped bytes.
