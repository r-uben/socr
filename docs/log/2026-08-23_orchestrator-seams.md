# D1 — where `_phase_agentic` and `_phase_assemble` divide

Date: 2026-08-23
Ticket: `docs/plans/orchestrator-decomposition/TICKETS.md` → D1
Base: `main`, `src/socr/pipeline/orchestrator.py` @ 7,520 lines
Scope: read-only. No source edits were made.

---

## 0. Headline

The two functions do not have the same shape, so they do not get the same treatment.

- **`_phase_agentic` is a dispatch table wearing a loop.** 294 lines of setup, then five
  *mutually exclusive* per-page lanes (564 lines) sharing one 134-line tail. The lanes are
  already disjoint; they are just inlined into an `if/elif` chain.
- **`_phase_assemble` is a page-disposition taxonomy.** Eleven page buckets, each derived,
  each turned into an audit event, each printed to the CLI — at three sites hundreds of
  lines apart, with the mutual-exclusion logic spread across all three.

Neither is "a long function that needs chopping into thirds". Both have one dominant
structure, and in both cases naming that structure is the whole refactor.

---

## 1. What are the phases inside `_phase_agentic`? (`orchestrator.py:2676-3777`)

**`CLAUDE.md` is wrong on two counts.** It documents the loop as
`route → extract → tables → figures → equations → flush`.

- **There is no figure step in the loop.** Figures are `_describe_and_embed_figures`
  (`orchestrator.py:6623`), called from `_phase_assemble`, after the body is saved.
  The loop's only image work is *stripping* VLM sentinel refs (`orchestrator.py:3721`).
- **`route` is not a universal first step.** It is one of five lanes, and four of the
  five never call `route_page` at all. Only the OCR lane routes.

The measured structure:

| Lines | n | Block |
|---|---|---|
| 2676–2714 | 39 | docstring |
| **2715–3008** | **294** | **doc-scoped setup** (8 labelled blocks, see §3) |
| 3009–3015 | 7 | loop banner comment |
| 3016–3064 | 49 | loop preamble: halt guard (`:3024`), per-page resume gate (`:3056`) |
| 3065–3153 | 89 | **lane 1** — corrupt-equation region recovery (`math_recovery_pages`) |
| 3154–3267 | 114 | **lane 2** — chart-asset page (`chart_winner_pages`) |
| 3268–3285 | 18 | **lane 3** — no-provider stamp (`no_ocr_provider_pages`) |
| 3286–3392 | 107 | **lane 4** — trusted native (`is_native`) |
| 3393–3628 | 236 | **lane 5** — OCR cost-ladder route (`else`) |
| 3629–3762 | 134 | **shared per-page tail** (runs for every lane) |
| 3763–3777 | 15 | post-loop summary + halt-reason handoff |

The shared tail, in order (`orchestrator.py:3629-3762`): table re-read (`:3660`) → table
escalation *or* scoring surface (`:3678`) → equation detect/crop/LaTeX (`:3714`) → image-ref
sanitize (`:3721`) → repetition guard (`:3731`) → blob cache put (`:3736`) → provisional
fragment + sidecar flush (`:3754`).

So the corrected one-liner is:
**`resume-gate → one of {math | chart | no-provider | native | route} → tail → flush`.**

---

## 2. What is `_phase_assemble` doing for 895 lines? (`orchestrator.py:5846-6741`)

It stitches fragments for about 55 of them. The other 840 are four things:

| Lines | n | Block |
|---|---|---|
| 5860–5866 | 7 | `_canonical_body` |
| 5867–5921 | 55 | fragment flush + **byte-identity stitch check** |
| **5922–6122** | **201** | **bucket derivation** — 11 page-disposition lists |
| 6123–6161 | 39 | phantom-image strip + document fabrication sweep (mutates `final_text`) |
| 6162–6210 | 49 | `pages_ok` conjunction → `DocumentStatus` (12 terms, one per bucket) |
| **6211–6482** | **272** | **audit-event emission + CLI summary**, one block per bucket |
| 6483–6516 | 34 | late sidecar flush (deliberately after the events — see `:6483` comment) |
| 6517–6600 | 84 | `EngineResult` build + five `error=` note concatenations |
| 6601–6640 | 40 | save markdown, write metadata, run figure phase |
| 6641–6700 | 60 | GH-226 final-body emission guard (re-saves, re-writes metadata) |
| 6701–6741 | 41 | authoritative fragment rewrite, final sidecar rewrite, manifest, audit log |

The eleven buckets: `failed_pages`, `d3_model_table_pages`, `d3_floor_pages`,
`native_only_distrust_pages`, `flagged_model_pages`, `structure_class_model_pages`,
`structure_class_native_fallback_pages`, `value_drift_pages`, `corrupt_math_hybrid_pages`,
`native_fallback_pages`, plus `fabricated_ref_pages` / `text_grid_rejected_pages` derived
later at `:6160` and `:6205`.

**The load-bearing invariant is mutual exclusion**, and it is not enforced anywhere — it is
maintained by hand, in prose. `native_fallback_pages` alone (`:6090-6122`) carries five
`and n not in …` clauses and ~60 lines of comment explaining why each exclusion has to match
`_winning_page_output` in `manifest.py` *exactly*. Three separate recorded bugs in those
comments (GH-151 B1 round 2, BLOCKING 2 on #269, #262) are all the same bug: a page counted
under two dispositions, or none.

Adding one disposition today costs **three edits, 300 lines apart**, plus an exclusion clause
in every sibling bucket. That is the cost the seam removes.

---

## 3. What state crosses a candidate boundary?

### `_phase_agentic` setup → loop: 26 names, but no lane needs more than six

The setup block binds 26 locals. Three are mutated inside the loop:
`_escalation_degraded` (`:3678`), `backend_degraded` (`:3603`), `halt_reason` (`:3604`).

Passing all 26 would not be a seam. **The point is that no lane needs them.** Per-lane
inputs, measured:

| Lane | Doc-scoped inputs it actually reads |
|---|---|
| **native (3286–3392)** | **none** — only `self.config`, `state`, `page_num`, `ps` |
| no-provider (3268–3285) | none |
| chart (3154–3267) | `_chart_figures_dir` |
| math (3065–3153) | `_chart_figures_dir`, `chart_winner_pages` |
| OCR route (3393–3628) | `ladder`, `judge`, `run_provider`, `provider_timeout`, `_chart_figures_dir`; writes `backend_degraded`, `halt_reason` |
| shared tail (3629–3762) | `_table_extractor`, `_escalation_profile`, `_escalation_degraded` (r/w), `run_provider`, `_detect_eq`, `_recover_eq`, `_agentic_doc_dir`, `_page_blob_store`, `is_native` |

Every lane's *output* is the same and is already a mutation, not a return: append to
`ps.attempts`, set `ps.best_output`, append to `state.events`, set flags on `ps`. So a lane
extracts as `def _agentic_lane_x(self, state, page_num, ps, …) -> None` with no new
plumbing at all.

The OCR lane is the one exception worth stating: `backend_degraded` / `halt_reason` are
loop-control, so that lane must **return** the halt signal rather than close over it.

### `_phase_assemble`: the buckets are pure functions of `state` + `self.config`

Every one of the eleven lists is a comprehension over `sorted(state.pages.items())` or
`state.events`, reading only `PageState` attributes, `self.config.native_only`, and five
predicates already imported from `socr.core.manifest` (`:5871-5879`). Nothing else crosses.
There is no hidden dependency on `final_text` — the strip mutation at `:6152` happens
*after* every bucket is derived except the two at `:6160`/`:6205`, and those read `state`,
not text.

---

## 4. Which live issues does each seam help?

51 issues are live. Mapping by the code each one has to edit:

| Seam | Issues it stops colliding on |
|---|---|
| **A1 native lane** (3286–3392) | #127, #223, #140, #64 — and the three born-digital audit events at `:3313/:3369/:3391` |
| **A2 OCR route lane** (3393–3628) | #227, #221, #159, #163, #249, #189 |
| **A3 shared tail** (3629–3762) | #160, #166, #157, #164 |
| **A4 chart lane** (3154–3267) | #167, #181, #189, #249, #150 |
| **A5 math lane** (3065–3153) | #271, #219, #165 |
| **A6 setup block** (2715–3008) | #154, #159, #139, #142, #168 |
| **B1 dispositions** (5922–6482) | **#176 directly**; plus #190, #270, #215+#245, #151, #146 — each of those *adds or changes a bucket*, i.e. each pays the three-edit tax today |
| **B2 finalization order** (6601–6741) | #171, #170, #238, #169 |

**Checked and confirmed out of scope: #162.** It is the first `W1` item and the obvious
candidate for "the seam that makes the accepting-gate tier fixable", but its defect is in
`socr/pipeline/agentic.py` (`SourceEvidenceTableJudge.assess` ~418, `NativeTableVerifierJudge.assess`
~524) — the orchestrator only *calls* the judge, at `:3406` (`route_page`, which owns the judge chain). No seam proposed here helps or
hinders it. This independently confirms the plan's 21-vs-30 partition on at least its
highest-priority member.

**Cosmetic seams, called out as required.** Three tempting cuts help no open issue and
should not be done: (i) hoisting the console/`--quiet` prints, (ii) splitting the setup
block by its eight existing `# --` comments — that yields eight functions sharing 26 names,
strictly worse, (iii) further decomposing `_canonical_body` / `_stitch_fragments`, which are
already 40-line single-purpose helpers.

Note also **#175** (inverted package layering) is scheduled in `W5` with a hold-until-D1
flag. It should stay held: A6 and B1 both touch `socr.core.manifest` ↔ orchestrator
coupling, and #175 would be re-deciding the same boundary from the other side.

---

## 5. The smallest first move

**A1 — extract the trusted-native lane, `orchestrator.py:3286-3392`, to a method.**

Why this one:

- **Zero doc-scoped inputs.** It is the only lane that reads nothing the setup block
  produced (§3). Signature is `(self, state, page_num, ps) -> None`; the diff is a cut,
  a paste, and a call.
- **Behaviour-identity is provable by inspection**, not by test archaeology: it is one
  branch of an `if/elif` chain, it returns nothing, and its only effects are mutations on
  objects already passed in.
- **It touches none of the four constraints.** It does not call `route_page`, so the empty
  ladder is irrelevant and CI's no-provider state cannot distinguish before from after. It
  does not touch `_run_fingerprint`, `_load_terminal_page`, the halt latch, or any fragment
  writer, so resume and byte-identity are untouched.
- **It is independently valuable**: four live issues (#127, #223, #140, #64) all edit
  exactly these 107 lines, and #223's own scope note splits it against #127 inside them.

Test posture, per this repo's rule about not pinning provider-dependent values: assert a
**difference of zero**. Run the same document through the pipeline twice in one process,
once with the lane inlined and once extracted — or, more practically, assert the
pre-existing golden/byte-identity fixtures are unchanged, and parametrise over both
provider states so the no-provider path is exercised too.

**A1 is 107 of 7,520 lines.** It is deliberately not the valuable move; it is the move that
proves the method and costs nothing if it is wrong. The valuable move is **B1**, and it is
roughly 470 lines — too large to be the first thing landed before the pattern is agreed.

---

## 6. The forks — for the owner, not decided here

### Fork 1 (the sharp one): methods on `UnifiedPipeline`, or modules?

Both A and B can land as *private methods on the same class*. That is the cheapest possible
behaviour-identical move — no imports change, no state has to be threaded, `self` carries
everything.

It also **does not reduce `orchestrator.py` by a single line.** The file stays 7,520 and
issue #155 stays open forever.

The alternative — lanes become a module (`pipeline/agentic_lanes.py`), dispositions become a
module (`pipeline/dispositions.py`) — actually moves the mass out, but every extracted unit
must then take its inputs explicitly, and `self.config` alone is read at dozens of points.

The plan's stated goal is that "module boundaries fall out of the seams". These seams do
fall out cleanly for **B1** (pure functions of `state` — a module is natural) and *less*
cleanly for the lanes (`self.config`-heavy — a method is natural). So the honest answer may
be **different answers for A and B**, and that is worth ratifying rather than assuming.

### Fork 2: order — A1 first, or B1 first?

A1 is cheap and proves nothing important. B1 is where the leverage is, where #176 lives, and
where six other live issues each pay a recurring tax. They are in different functions and do
not conflict.

### Fork 3 (raise, do not settle): does A2 want extracting at all?

The OCR lane is 236 lines and is the only lane that writes loop control. Extracting it means
returning a halt signal. That is still behaviour-identical, but it is the one lane where the
seam changes a control-flow shape rather than only a location — and cascade-halt already has
two open bugs against it (#227 fires when it should not; #221 cannot fire at all). Moving
that code while its semantics are disputed may be the wrong order.

---

## 7. Constraints — explicit clearance

| Constraint | Status for the proposals above |
|---|---|
| Byte-identical assembled `.md` | Guarded at `:5905` (stitch check) and `:6716` (`_rewrite_all_fragments`). A1 and B1 touch neither. **B2 does** — anything in 6601–6741 must be treated as byte-identity-critical. |
| `_run_fingerprint` / resume | `_load_terminal_page` is called at **two** sites, `:2757-2758` (pre-pass) and `:3056` (loop). Both are in the setup/preamble, outside every lane. A seam that relocates either invalidates every terminal page on resume. A1/B1 do not. |
| Cascade-halt latch | Read at `:3024`, written at `:3603`. Only the OCR lane writes it — see Fork 3. |
| CI has no provider | The empty-ladder early return is `:2873`, in the setup block. It is exactly CI's state, so **A6 is the highest-risk seam in the file** and should not be early. A1 never reaches it. |
