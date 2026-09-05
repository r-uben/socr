# TICKETS — verifier-independence

Status keys: `TODO` · `WIP` · `DONE` · `BLOCKED`. `depends-on` gates dispatch.
Parallelizable = no shared files / no dep. Each ticket = one implementer agent,
then one reviewer pass before commit.

## Goal

Make the free native lane a *sound* witness of the model lane. Today the recovery
path that is supposed to overrule a broken native token is addressed by the native
layer's own bounding box (`orchestrator.py:_adjudicate_clamped_table` →
`adjudication._disprove_one(item.native_bbox)`). **Hypothesis to be established by
A1b, not assumed:** a shredded native row label may also truncate the recovery crop
(the crop is padded; whether the padding covers the loss is unmeasured).

Measured on the 2026-09-04 ladder re-run (`docs/log/2026-09-04_ladder-corpus-rerun.md`,
`main@f434019`): 7 tables adjudicated, **1 lifted, 6 held**; **3 of the 6** (doc02 p3,
doc02 p4, doc04 p3) are blocked by native row-label defects (#331 / #418 / #146); doc03
is held by lane contradictions, doc05/doc07 retain sibling-LaTeX and lane items (#585,
out of scope here). Cost is not the problem ($0.0020 / 20 pages). Latency is unmeasured — run-2's **8.0
min/page is confounded** (wall-clock spanned three `socr_source_digest` values; venv repoint
mid-run); no stage timing exists anywhere in sidecar, manifest, or CLI (B2 establishes a fresh
baseline under one digest with B1 in tree).

Three streams: **A** repair the native reference, **B** instrument latency, **C** make
the verifier independent of both lanes. `orchestrator.py` decomposition (#155) is not a
prerequisite — every ticket touches a bounded seam.

## Fixed inputs (verified 2026-09-04 by the Claude/Codex/agy/grok panel)

- **Frozen corpus + run-2 outputs:** `~/Data/socr/ladder-run2-2026-09-04/`
  (`in/doc0N.pdf`, `out/doc0N/doc0N/{manifest.json,pages/*.json,cache/}`,
  `manifest.json`, `run.sh`, `tabulate.py`, `SHA256SUMS`, 153 files). Copied from the
  run-2 scratchpad. `shasum -a 256 -c SHA256SUMS` must pass before any number is
  reported. **Absent in CI** — no test may open it; tests use fixtures under
  `tests/fixtures/`.
- **The binder builds its own native rows.** `src/socr/tables/binding.py:559
  _native_rows`, `:332 _assign_bands`, `:322 _row_label_and_bbox`, `:1276 bind()`.
  `binding.py:70-71` *deliberately* does not use `native_rows.py::LabeledRow`. So
  `native_rows.py` is **not** on the ladder path; row-label repair targets `binding.py`.
- **Adjudication seam:** `src/socr/tables/adjudication.py:227 adjudicate`,
  `:260 _disprove_one`; `src/socr/pipeline/orchestrator.py:5223` (call site of
  `_adjudicate_clamped_table`), `:5245` / `:5292` (interpret `lifted` / `held`),
  `:5522-5637` (`_adjudicate_clamped_table` … `_render_adjudication_crop`),
  `:5550` (call site of `adjudicate`), `:5643-5651 _apply_binding_adjudication_meta`
  (round-trips the record into the sidecar).
- **Contradiction types:** `ContradictionItem` is defined and built in
  `adjudication.py:70` / `:152-181` (`items_from_binding`, `_item_from_cell`,
  `_item_from_row_label`). `binding.py` never names it; it owns `ContradictedCell`
  (`:1002`) and `RowLabelContradiction` (`:1011`), each already carrying `native_bbox`.
- **Geometry that exists:** `src/socr/tables/locate.py:169 _horizontal_rules` → raw
  `(y, x0, x1)` rules. `:122 bands_from_rules` groups them into **whole-table** boxes
  and discards the intermediate rules — **no row-band helper exists yet**. Ruled tables
  carry per-row rules; booktabs tables carry top/mid/bottom only.
- **Resume gate:** skip checks (`orchestrator.py:8555-8571`) ignore extra JSON keys;
  `_run_fingerprint` (starts near `:976`; `socr_source_digest` at `:1251`) includes
  `socr_source_digest`, so **any `.py` edit invalidates every existing ledger** — run-2
  sidecars will never be skipped by a tree that contains any ticket here.
  `_restore_terminal_page_state` (`:8901-9112`) restores only named fields; a new sidecar field
  that is not restored is dropped on the assemble re-flush.
- **Byte-identity:** `_flush_page_sidecar` (`:8147-8455`) writes JSON;
  `_rewrite_all_fragments` (`:9238-9276`) writes `.md` from assembled bodies. New fields
  stay out of fragment markdown, `audit_events.detail`, and `canonical_page_texts`.
- **Hermeticity patches** (CI has no provider): `_available_engines_for_agentic` →
  `[PROFILE_QWEN_LOCAL]`, `_resolve_judge_model` → `""`, `_transcribe_cell_token` →
  stub. `tests/test_gh367_adjudication_lift.py:177-180` shows all three. **Pin a
  difference, never a value.**
- **`tests/` is flat** (`pyproject.toml:95`). No `tests/pipeline/`, `tests/benchmark/`,
  `tests/tables/`. Entry points go in `pyproject.toml [project.scripts]` (`:67-69`).

## Stream A — native reference repair

### TICKET-A1 — replay-binding harness · DONE · depends-on: none · wave 1
**Problem:** There is no way to re-bind a *frozen* model candidate against a *fresh*
native extraction, so a binder change can only be measured through a live OCR run.
**Do:** Add `socr.benchmark.replay_binding`: for each frozen page sidecar, take the model
candidate markdown and the recorded `binding_adjudication`, re-run `bind()` on the
frozen PDF with the *current* tree, and emit per table: `native_label_items`,
`items_disproved`, recorded `status`, and — separately, never folded into one score —
**label accuracy** (native label vs a hand-read label file), **crop coverage** (does the
recorded `native_bbox` cover the printed label; rendered via `_render_adjudication_crop`
*imported*, not modified), **final disposition**. Entry point
`socr-replay-binding <corpus-dir>` in `pyproject.toml [project.scripts]`.
**Files:** `src/socr/benchmark/replay_binding.py` (new), `pyproject.toml`
(`[project.scripts]` only), `tests/test_replay_binding.py` (new, hermetic: small
fixtures under `tests/fixtures/replay_binding/`, no corpus, no provider),
**read-only import** of `orchestrator._render_adjudication_crop` — B1 owns that file.
**Done when:** `~/venvs/socr/bin/socr-replay-binding ~/Data/socr/ladder-run2-2026-09-04`
prints 7 rows; on the **unchanged tree**, each row explicitly compares the **fresh** `bind()`
result against the frozen sidecar's recorded `binding_adjudication` as a **multiset**
(`native_token` / `model_token` / `kind`, duplicate counts preserved) — not merely echo of
recorded `status` / lifted-held counts (reproducing `1 lifted / 6 held` from recorded
statuses alone is vacuous); a hermetic fixture perturbs the native words and the test
asserts **the exact expected delta** is reported **and** the recorded sidecar bytes are
unchanged; `~/venvs/socr/bin/pytest tests/test_replay_binding.py -q`
exits 0 with `ollama` and `qwen-ocr` absent from `PATH`; `uvx ruff@0.16.0 format --check .`
clean.

### TICKET-A1b — failed-disproof autopsy · TODO · depends-on: A1 · wave 2 · agent: claude
**Problem:** The 8 failed disproofs and the 3 native-label-blocked tables have never been
looked at cell by cell, and the "native bbox truncates the crop" claim is a hypothesis.
**Do:** Run A1 on the frozen corpus. Census **all 12** class-(c) items the run-2 log counts
(8 on the three target tables, 2 on doc05 p1, 1 on doc07 p1, and one the log does not
allocate — find it). Render the 8 failed-disproof crops; hand-classify
each as `absent_text` / `shredded_label` / `bbox_truncated` / `neighbour_capture` /
`other`, with the rendered crop attached. State explicitly, per crop, whether the padded
crop did or did not cover the printed label. Write the hand-read label file A1 consumes.
**Files:** `docs/plans/verifier-independence/logs/YYYY-MM-DD_A1b-autopsy.md`,
`~/Data/socr/ladder-run2-2026-09-04/labels.json` (hand-read, outside git).
**Done when:** the log names a class for all 8 crops and locates all 12 class-(c) items,
identifies which of the 3 blocked tables are `shredded_label` vs `bbox_truncated`, states
**N = the count of native-side (`shredded_label` + `bbox_truncated`) target labels** (A2's
denominator), and answers the hypothesis in one line with a count (`M of 8 crops
truncated by native_bbox`).

### TICKET-A2 — binder row-label repair (#331 / #418 / #146) · TODO · depends-on: A1b · wave 3
**Problem:** The binder's own rowizer truncates, runs-on, or shreds row labels on the 3 target
tables (`Treasury inst. forward rate` missing `3Y`/`5Y`/`10Y` on doc02 p3/p4; shredded subscript
`1t 1t` on doc04 p3), so the free reference contradicts a correct model token and the table is
held.
**Do:** Fix only the classes A1b found native-side (`shredded_label`, `bbox_truncated`) in
`_native_rows` / `_assign_bands` / `_row_label_and_bbox`: stub-column retention (#331),
snap-radius drop (#418), first-row-as-header (#146) — whichever are live on the 3 target
tables. Derive any new bound from the page (font size, rule spacing), never a literal.
Add row-swap and neighbouring-label control fixtures so the repair cannot pass by
accepting more. Do **not** touch `native_rows.py`.
**Files:** `src/socr/tables/binding.py`, `tests/test_binding.py`,
`tests/test_tr1_rowizer.py`, `tests/test_gh331_stub_labels.py`,
`tests/test_table_header_gh146.py`, `tests/fixtures/replay_binding/` (controls).
**Done when:** on frozen replay, **3/3 target tables cleared of false native-label clamps**
(**revised 2026-09-05 before merge, PR pending: 2/3 — doc04 p3 is withdrawn from A2's
denominator.** A1b classed it as the math-font shape, not the rowizer shape; the curia build
found no page-derived geometry that separates a math subscript from a short annotation under
a label, and abstained rather than merge — a fold that admits `(a)` into a label is worse than
no fold. doc04 moves to a new ticket in the math-font lane, #219/#140 family. N becomes 7);
**N/N unresolved target labels reconstructed** against the rendered source, where **N =
A1b's native-side subset only** (`shredded_label`, `bbox_truncated`), with A1b's
classifications **frozen before repair starts** (A2 may not reclassify); all three target
tables appear in the report regardless; `absent_text`, `neighbour_capture`, and `other`
are logged on their own line, **out of that denominator**, and keep their justified
dispositions. If A1b's findings invalidate 3/3, the gate is **revised explicitly in
TICKETS.md before implementation** — never shrunk silently; **zero false accepts** on the row-swap and
neighbouring-label controls; A1 output for the 4 non-target tables **not worse** before/after — every
recorded item persists identically or disappears because the SAME root cause A2 fixed also
produced it (**amended 2026-09-05 before merge**: "byte-identical" became unachievable once A1b
classed doc01 p2's dropped `2` as the same 0.001 pt region-edge clip as doc02's stubs;
`_words_in_region` is one shared function and may not special-case pages; doc01 p2 goes 1 → 0
items and is recorded in the ticket log as a same-cause clear, not a regression);
full suite green; ruff format clean.

## Stream B — latency instrumentation

### TICKET-B1 — per-page, per-stage exclusive wall-clock · TODO · depends-on: none · wave 1
**Problem:** 8.0 min/page and nothing in the sidecar, manifest, or CLI says where it
goes.
**Do:** Time each stage of the page loop in `_phase_agentic` — `route`, `extract`,
`tables` (nested `ladder`, `adjudication` reported **exclusively**: children subtracted
from the parent so the exclusive keys sum to the page total), `figures`, `equations`,
`flush`. Write `timings_s` into the page sidecar (`_flush_page_sidecar`), restore it in
`_restore_terminal_page_state` (else the assemble re-flush drops it), roll up
per-document into the manifest and the CLI summary line. **Never** in fragment markdown,
`audit_events.detail`, `canonical_page_texts`, or `_run_fingerprint`. Measurement only:
no optimisation, no threshold, no new status.
**Files:** `src/socr/pipeline/orchestrator.py` (`_phase_agentic`, `_flush_page_sidecar`,
`_restore_terminal_page_state`), `src/socr/core/state.py` (`PageState.timings_s`),
`src/socr/core/manifest.py`, `src/socr/cli.py` (summary line), `tests/test_timings.py`
(new).
**Done when:** `~/venvs/socr/bin/pytest tests/test_timings.py -q` exits 0 in a shell with
`ollama` absent, and it proves (a) `timings_s.total` is an independently measured page
wall-clock and `total − Σ exclusive` is reported as unattributed time (**amended at merge**,
PR #594: the original "exclusive keys sum to total within 1 ms" was tautological once total
was defined as the sum),
(b) the resume skip decision is identical with and without `timings_s` present (pattern:
`tests/test_p6_disposition_persistence.py:191-228`), (c) the final `.md` is byte-identical
with timings on and off; existing golden/byte-identity tests unchanged. Patches:
`_available_engines_for_agentic`, `_resolve_judge_model`.

### TICKET-B2 — latency breakdown on the frozen corpus · TODO · depends-on: B1 · wave 2 · agent: claude
**Problem:** Run-2's 8.0 min/page is confounded (three `socr_source_digest` values mid-run) and
has no per-stage breakdown. B2 establishes a **fresh** timed baseline under **one digest** with
B1 in tree — not an explanation of the confounded 8.0.
**Do:** Before running: record the intended source digest from the checkout
(`_socr_source_digest()` on the B1 tree) and verify the resolved package path
(`~/venvs/socr/bin/python -c "import socr; print(socr.__file__)"`) points at that
checkout — a consistently wrong editable install passes a sidecars-agree check. Run from
a **dedicated checkout pinned to the intended B1 commit, kept unchanged throughout
measurement**; set `PYTHONPATH=<that checkout>/src` explicitly for every process
`run.sh` launches (the shared venv's editable pointer has been repointed mid-run twice); **assert all 20 sidecars are present and each
`socr_source_digest` equals the intended digest** before tabulating (else the run is
discarded, not reported); tabulate `timings_s` per stage per page, log the breakdown. No code change; no
claim about run 2's minutes.
**Files:** `docs/plans/verifier-independence/logs/YYYY-MM-DD_B2-latency.md`.
**Done when:** the log has a per-stage table (20 pages × stages) and one line naming the
stage that owns > 50 % of wall-clock, or stating no stage does.

## Stream C — verifier independence

### TICKET-C1 — design: geometry-addressed cell correspondence + abstain rule · TODO · depends-on: A1b · wave 3
**Problem:** The disproof crop is addressed by the native bbox (self-witness); the
model's structure would let the candidate choose where it is examined (mirror hole); a
union crop is unsafe because `tokens_agree` checks text, not location.
**Do:** Design note. Define **cell correspondence** = an independently established
(row band, column lane) address for a disputed cell from page geometry that **neither
lane owns** — not the binder's lanes, not the candidate's columns. Rows: a new
`row_bands_from_rules` over `_horizontal_rules` (pairs consecutive rules; `bands_from_rules`
discards them). Columns: state what geometry is available (vertical rules, if any;
whitespace gutters from the *page*, not from the binder). For booktabs tables state
precisely what is *not* available. Define the **abstain rule**: no correspondence ⇒
`_disprove_one` returns `None` (held, recorded `abstained`); it never falls back to
either lane's box. Read-only; sharp questions for the owner.
**Files:** `docs/plans/verifier-independence/logs/YYYY-MM-DD_C1-design.md`.
**Done when:** the note contains (a) the correspondence definition with the exact
`locate.py` symbols and the new helper's signature, (b) the abstain rule as a truth table
over {ruled, booktabs} × {row found, not} × {column found, not}, (c) the `binding.py` /
`adjudication.py` / `orchestrator.py:5200-5310` fields and branches that change, (d) ≤ 3
named decisions for the owner.

### TICKET-C2a — line-band / column-edge / origin helpers + `BindingResult` surface · TODO · depends-on: C1 · wave 4
**Problem:** C1 (rev 4, `logs/2026-09-05_C1-design.md` §(a)) measured that **no table on the
corpus has per-row or vertical rules**, so the address must come from PDF text-line geometry
inside the witness region, counted from an origin neither lane supplies. None of those helpers
exist, and `bind()` does not expose the two lists the ordinal chain needs.
**Do:** In `locate.py`, pure functions over `(page, region)` with no binder or candidate input:
`row_bands(page, region)` — text lines (`page.get_text("dict")`) inside the region grouped into
bands whose baselines differ by less than the smaller of their font sizes; where per-row rules
exist, `row_bands_from_rules` supplies the edges instead; `label_column_edge(bands, region)` —
`R₀ = min x0` over all non-leftmost lines, shrunk to a whitespace edge as C1 §(a) specifies
(fixed point in ≤ 2 passes), returning `None` when no `R > region.x0` exists;
`ordinal_origin(page, region)` — cluster the region's horizontal rules by y (rules closer than a
rule thickness are one border), return the y of the **second** group, `None` when there is no
second group (scanned pages). Every bound is derived from the page's own type sizes and rule
thickness — no literal. In `binding.py`, `BindingResult` gains two **read-only** fields,
`native_rows` and `row_binding` (the `_native_rows` output and the `_bind_rows` mapping already
computed) — nothing else in `binding.py` changes and no geometry enters it.
**Files:** `src/socr/tables/locate.py`, `src/socr/tables/binding.py` (`BindingResult` fields
only), `tests/test_locate_line_bands.py` (new; synthetic fixtures + the frozen-corpus check
below runs only when the corpus dir exists, else skips — CI has no corpus).
**Done when:** `~/venvs/socr/bin/pytest tests/test_locate_line_bands.py -q` exits 0 and proves:
(1) on a synthetic ruled fixture every printed row gets exactly one band and the bands equal
`row_bands_from_rules`; (2) on a synthetic booktabs fixture one band per printed row from text
lines, and `ordinal_origin` = the second rule group; (3) on a fixture with no rules
`ordinal_origin` is `None` (abstain input), never a guess; (4) `label_column_edge` returns `None`
on a one-column fixture. Corpus check (skipped in CI): on the frozen 7 tables the origins equal
C1's measured values (doc01 116.3, doc02 123.9, doc03 241.5, doc05/07 121.0, doc04 `None`) and
the band counts per table match C1 §(d)'s inputs. `BindingResult.native_rows` /
`.row_binding` are populated on every `bind()` call and the A1 harness output is byte-identical
before/after. Full suite green; ruff format clean.

### TICKET-C2b — geometry-addressed disproof + abstain semantics · TODO · depends-on: C2a, A2, B1 · wave 5
**Problem:** As C1 — `_disprove_one` transcribes `item.native_bbox`, so the recovery crop is
addressed by the lane it is checking. This is the architectural change the owner authorised.
**Do:** Exactly C1 §(c). `adjudication.py`: `ContradictionItem` gains `cell_bbox`,
`address_source`, `abstain_reason` (`native_bbox` stays, audit only, never transcribed);
`_disprove_one` transcribes `cell_bbox` and returns the new `"abstained"` `DisproofKind` when
it is `None` **without calling `transcribe`**; `adjudicate` treats `abstained` as not a
disproof (a table with any abstained item cannot be `lifted`); `to_record` emits both new
fields. `orchestrator.py:_adjudicate_clamped_table` builds the ordinal chain from
`BindingResult.native_rows` + `.row_binding` + C2a's bands/origin/edge, per C1 §(a)
conditions 1–4 (native chain **and** model chain from the same origin; `i = j = b`), attaches
`cell_bbox = (region.x0, band.y0, R, band.y1)` or `None` with the reason, and renders with
padding **clamped on both axes**: x within `[region.x0, R]`, y limited by the adjacent bands
(C1: 6 pt of padding against a 0.2 pt column clearance re-opens over-capture). The test
asserts the **rendered crop rectangle** respects both clamps — not merely that the
transcriber received a bbox. `:5259-5310` counts abstentions separately and the UNVERIFIED cause
gains `abstained`. `binding.py` gains **no** geometry fields. Keep the
`_transcribe_cell_token` patch seam.
**Prediction artifact (committed before C2b starts):** C1 §(d) at rev 4 — 3/22 addressed
(doc05 p1 items with triples (4,4,4), (6,6,6), (8,8,8)), 19 abstained with reasons — is the
artifact; reference its commit SHA in the ticket log. After A2 (merged, #602) the remaining set is
**14 = 22 − 7 doc02 items − 1 doc01 item**; doc04's item remains and abstains (no origin).
The prediction artifact must be **re-derived on the post-A2 tree** before C2b starts — C1
§(d) was computed pre-A2 and its remaining-set membership is stale; commit the updated
per-item table and cite that SHA.
**Files:** `src/socr/tables/adjudication.py`, `src/socr/pipeline/orchestrator.py`
(`:5200-5310` and `:5522-5637` only), `tests/test_binding_adjudication.py`,
`tests/test_gh367_adjudication_lift.py`, `tests/fixtures/replay_binding/controls/`.
**Done when:** three hermetic controls in one process, pinning the **difference**, with
`_transcribe_cell_token` call counts asserted: (1) *positive* — native bbox truncated to half
the label, both chains intact → the transcriber receives the geometry cell and the item is
disproved; (2) *wrong pointer* — the printed row above the disputed item has no native row →
`abstained`, call count 0, table `held`; (3) *shifted but order-preserving* — model table has
one row inserted above the disputed item, native unchanged → `abstained`, call count 0.
Any `process()` test patches `_available_engines_for_agentic`, `_resolve_judge_model`,
`_transcribe_cell_token`. Resume skip identical with and without the new fields.
**Frozen-replay gate:** `socr-replay-binding` asserts the implementation matches the committed
prediction item-for-item (address vs abstain, and the abstain reason class) on the 14 remaining
items; every table A2 cleared stays cleared. **Feasibility checkpoint:** if fewer than one
remaining item is geometry-addressed, C2b returns to the owner — correct abstention must never
force invented correspondence. Full suite green; ruff format clean.

### TICKET-C3 — ladder corpus re-run and report · TODO · depends-on: C2b, B2 · wave 6 · agent: claude
**Problem:** "The free lane witnesses the model" needs a number on the same 20 pages —
but a live OCR re-run is nondeterministic and stricter abstention can *lower* accepts
legitimately, so the live count is a **report**, not the gate.
**Do:** Re-run `run.sh` on the frozen corpus at the C2b tree under **B2's run discipline**
(pinned dedicated checkout, explicit `PYTHONPATH`, resolved-path check, all 20 sidecars on
the intended digest). Tabulate ACCEPTED /
WITHHELD / UNVERIFIED; lifted / held / abstained; cloud cost; per-stage minutes (B1).
Compare to run 2 line by line. Every held table carries a cause from the full taxonomy:
`absent_text`, `abstained`, `presentation` (#585), `lane_mismatch`, `infrastructure`,
`unresolved` — failure to disprove never becomes `model_wrong`. Log in the existing
ladder-run format.
**Files:** `docs/log/YYYY-MM-DD_ladder-corpus-run-3.md`.
**Done when:** the log exists with all of the above; **the hard gate is A2's frozen-replay
3/3 (already met by then), not this count.** The derivable expectation is 10/18 ACCEPTED
(7 baseline + 3 targets) *if* nothing else moves; the log reports the actual number and,
where it differs, names the class that moved it.

## Parked (deliberately)

- **doc05 / doc07 class-(c) native row labels (out of wave).** Run-2 log still has native
  row-label class-(c) items on doc05 (2) and doc07 (1), mixed with #585 / lane holds. A2's
  8-label census does not cover them — leave as follow-up (#585 / #331), not invisible.
- **#155 orchestrator decomposition.** Not a prerequisite; C2b touches two bounded hunks.
  Revisit only if C2b's reviewer finds the seam is not bounded.
- **#585 sibling-LaTeX normalisation** (`\Delta`, `\log`, `\&`). Buys no demonstrated
  table on this corpus (doc05/doc07 retain lane contradictions). Separate issue.
- **Latency optimisation.** B1/B2 measure. Optimising before the owner stage is known is
  guesswork.
- **Booktabs correspondence.** C1 says what is not available; C2b abstains there.
  Universal booktabs recovery is open-ended.

## Panel disagreement on record

agy wanted C3 as a strict `ACCEPTED ≥ 10/18` gate; Codex argued a live-run quota rewards
acceptance volume over justified acceptance and that the frozen replay is the only
deterministic gate. Plan follows Codex; agy's number is kept as the *expectation*.
