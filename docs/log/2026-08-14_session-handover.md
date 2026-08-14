# Handover — 2026-08-14, 02:00

Written at the end of a long session so the next one starts from state, not from memory.
Nothing here is a decision; decisions live in `TICKETS.md` and the dated notes. This is the
map, the open questions, and the things I am not confident about.

---

## 1. Where the repo is

`main` at `73a0243` + merged PRs below. Full suite on `feat/151-b1-structural-gate`:
**1610 passed / 1 xfailed**, `uvx ruff@0.16.0 format --check .` clean.

### Merged this session

| PR | What |
|---|---|
| #191 | Retargeted GH-144 A2 at the text-strategy grid, before it was implemented against the wrong function |
| #192 | GH-144 A2+A2b — numeric tokens no longer destroyed at grid construction |
| #193 | GH-147 A2 — rotated pages refuse the native table lane, audit event keyed on an explicit flag |
| #194 | GH-150 A2 — hermetic chart-detection tests |
| #196 | Wave 2 closeout on the coordinator board |
| #199 | GH-151 B1 design settled (the version now superseded — see §3) |

**Measured outcome of wave 2**, the one number worth remembering: decimal-token loss on the NS
QJE reference pages went **5 / 35 / 13 → 0 / 0 / 0** (pages 17, 42, 43), verified end to end
through `extract_structured`, not through an isolated rung.

### Open PRs

| PR | State | Blocking on |
|---|---|---|
| #200 | GH-151 B1 implementation, CI green, **3 review rounds + a 4th finding** | Direction — see §3 |
| #201 | Wave-3 retargets + board closeout | Deliberately held: it declares wave 3 CLOSED while #200 is open |
| #204 | The B1 reframe decision (§3) | Nothing |
| #179, #153 | Old drafts, untouched all session | — |

---

## 2. What wave 3 actually produced

One ticket built (GH-151 B1, PR #200). **Three retargeted** — sent back because their premises
no longer held. A premise check was added to the dispatch harness for this wave and fired on
three of four tickets on its first run.

- **GH-152 A1** — blamed the rowizer; the page-wide text-strategy `find_tables` at
  `reconstruct.py:141` merges the tables first and suppresses the fallback. Wiring only the
  rowizer is a measured end-to-end no-op. A1 is now detector-only; A2 recut to consume it at
  both rungs.
- **GH-150 B2** — defect reproduces (both PDFs copied, hashed, measured), but the fix is a
  merge inside `born_digital.py`, which B2 does not own. B2 becomes fixtures + a strict xfail;
  new TICKET-C1 owns the fix and is blocked on #200 for that file.
- **GH-147 B1** — its metric is invalidated by GH-147 A2's own fix. Refused pages get
  `native_text = raw_text.strip()` (`born_digital.py:915-931`), so word recall there is **~1.0
  by construction** and the 20/40 figure cannot survive a correct fix. Retargeted onto refusal
  rate plus a structural witness.

All three are written into their folders' `TICKETS.md` on #201.

**Assume waves 4 and 5 are stale too.** They were authored the same day as these. Check premises
before staffing.

---

## 3. The live question: what is GH-151 B1 for

Full reasoning in `docs/log/2026-08-14_gh151-b1-escalation-decision.md`; measurement in
`docs/log/2026-08-14_gh151-b1-predicate-design.md`. Short version:

B1's ticket exists to falsify **word recall as a routing gate** — p26 ships at 100% recall with an
unusable table. B1 was meant to supply the *structural half of an escalation signal*. PR #200
implements it as a **status label** instead, because the (correct) `--native-only` ruling barred
the flag from `needs_repair`, which is precisely the mechanism that would have made it routing.

**Owner has decided: rebuild it as an escalation signal.** Not yet designed.

### What #200 should become — NOT decided

1. Land the plumbing, replace the predicate (keep reviewed wiring, swap in TR-3, add winner-side check)
2. Narrow first (design note Option B: 26.9% → 14.3%), extend later
3. Close and rebuild

### The measurement that forced this (32 papers / 245 pages / 422 blocks)

- Gate as implemented fires on **26.9% of native table pages**
- `ragged` alone: **2/422**, and does **not** fire on B1's own acceptance fixture
- `_is_canonical_column_band`: suppresses **0/422** — dead on real data
- Geometry does **not** separate a split row from a header band: B1's fixture measures y-gaps
  `[18,18,18,18,18]`, a full row-pitch split identical to a heading band

---

## 4. Open questions for tomorrow

**Read §4b first — a review round landed after this section was written and answers several of
these, and changes the ranking.**

1. **#200's fate** — the three options above. My lean: (2) then (1), so nothing that fires on a
   quarter of correct pages ever reaches `main`, but the reviewed wiring survives.
   **Superseded — see §4b.1: both the reviewer and Fable reject (2), for different reasons.**
2. **What does an escalation signal DO when it fires?** Table pages already route to the model.
   So the signal cannot mean "send to OCR" — that already happens. Candidates: escalate the
   *ladder* (local → cloud); refuse to accept the winner and re-run; or mark for repair. Each has
   a different cost profile and none is designed. **First concrete answer in §4b.4.**
3. **Does the structural check run on the winner, including an accepted VLM grid?** If yes, that
   is a bigger change than B1 and touches the judge. If no, the silent-replacement gap stays open.
4. **Is the TR-3 gap (§5.2) worth more than B1?** I think it might be. It needs an hour of
   scoping, not a wave. **Sharpened in §4b.3 — it has an unmeasured precision problem of its own.**
5. **Should `/paper-lookup` be fixed?** It names only iCloud. The corpus spans two locations and
   neither is complete alone (see §6). Deferred by the owner; still true.

---

## 4b. Review round on #204 — reviewer proposals + Fable's independent judgement

Two inputs landed after §4 was written: a seven-point ranked review on #204 (comment
`5292582298`, plus an inline on the three options and a pointer on #200), and Fable's independent
read of that review. Fable verified claims by probe rather than agreeing; artifacts in
`/tmp/b1verify/`. Where they differ, both positions are recorded — neither is a decision.

**0. A hazard that was real and is now fixed.** #204 was cut from `feat/151-b1-structural-gate`
rather than `main`, so it carried #200's three implementation commits (1,963 insertions across
`orchestrator.py`, `manifest.py`, `state.py`, `tables_trust.py`) while its body claimed docs only.
Merging it would have landed the 26.9% shape gate. Rebuilt from `main` as `6ddcd62`; the PR now
contains exactly three doc files. The reviewer's "do not merge as it stands" was correct when
written and is stale now.

**1. Both reject "narrow 26.9% → 14.3%" — but for different reasons, and the difference matters.**
The reviewer's argument is principle: a tighter heuristic is still a heuristic, and B1 exists to
stop treating markdown shape as structure. Fable disagrees with that reasoning and agrees with the
conclusion on priority grounds: the design note §2.3 shows the pairs the narrowing *keeps* are
dominated by genuinely damaged pages (garbled `Sd. B` / `SILL Error` labels, real `R2` splits),
which is evidence of decent precision, not disqualification. And under the escalation reframe a
false positive costs **compute, not content** — a much weaker objection than it was for a status
label. Fable's defensible version: don't spend the next hour tightening shape when a
better-grounded signal is already computed and inert.

**2. The reviewer's counter-proposal (replace the predicate on #200's branch before merge, so
26.9% never touches `main`) is available — but it is not a swap.** Fable verified the load-bearing
claim independently: rebuilding B1's acceptance fixture and running `BornDigitalDetector.detect_page`
on the current tree gives `has_unverifiable_table_region=True`, hard-fail reason
`value_guard_multiset_mismatch: 1 paired row(s)`. So TR-3 does fire on B1's own fixture. Two
qualifications Fable found:
   - **No D3-floor collapse.** #200 uses a separate flag (`native_table_structure_defective`), so
     swapping its source leaves the D3 conjunction at `manifest.py:311-313` intact. Checked.
   - **But the swap creates a duplicate flag and rewrites the acceptance suite.** Under the swap,
     `native_table_structure_defective` becomes bit-identical to the existing
     `PageState.native_table_unverifiable` (`state.py:190` propagates the same field). The honest
     shape is "give `native_table_unverifiable` the wiring #200 built", not "keep both names".
     And the `GH151_P26_MD` seam test is **unimplementable** under TR-3 — it is a markdown string
     with no PDF, so no geometry can run — while the negative controls are all shape-written. Real
     move, but it is a test-suite rewrite plus a flag-dedup decision. Do not let it be scoped as
     an afternoon's work.

**3. The sharpest finding: TR-3's 25.3% is a firing rate, not a defect rate — nobody has measured
its precision either.** The reviewer calls surfacing TR-3 the better next hour and says its
"false-positive story is different". Fable: different in kind, yes; *measured*, no. The verifier is
conservative by design (hard-fail reserved for clean row alignment; a multiset mismatch under a
row-count discrepancy ships flagged instead — `native_verifier.py:826-846`), but a quarter of
native tables firing "certain dropped/shifted value" is implausibly high — if true this pipeline
would be known-broken in ways nothing else corroborates. TR-3 shares the `is_numeric_token`
machinery whose `.034` / U+2217 gaps are follow-up 7(b), and notation-driven false positives inside
the 25.3% were never separated. **Emitting an AuditEvent for the 62 pages costs nothing and honours
"no silent content loss" — do that. Keying escalation or any ERROR/PARTIAL status on it before
someone hand-judges a sample repeats the exact mistake that sank the 26.9% gate.**

**4. First concrete answer to "what should escalation mean" (Fable):** a **stop-condition veto on
the ladder**, with a fail-closed marker at the top rung. Structural soundness joins recall as an
acceptance criterion on the *winning* candidate at each rung. Fails at local → escalate local→cloud.
Fails at the top rung → do not silently ship the best-looking loser; ship it flagged with an
explicit marker, the precedent the D3 floor already sets (`manifest.py:303-333` — marker + PNG ref,
`audit_passed=False`). **Not** `needs_repair` (barred by the `--native-only` ruling), **not**
deletion. This shape also absorbs the winner-side check naturally: the same predicate runs on
whatever is about to ship, native or VLM, instead of only on the native attempt.
Gating dependency: point 3. At unmeasured precision, wiring TR-3 into ladder escalation means
escalating a quarter of table pages to cloud on unvalidated evidence.

**5. Coverage asymmetry — the signals are NOT nested, so "replace shape with TR-3" loses real
detections.** TR-3 misses **31 of the 66** shape-flagged pages (design note §2.4), and the narrowed
shape gate's kept set is dominated by genuine damage (§2.3). Garbled-label splits are *textual* and
plausibly invisible to a numeric-token multiset. The review treats Option C as strictly dominant;
the note's own numbers say otherwise. **The eventual escalation predicate may legitimately be a
disjunction of both signals.**

**Points both agree on, and that survive:** do not design escalation semantics inside #200 (file
it); the winner-side check is a separate, judge-touching ticket and must not be smuggled into a
"replace the predicate" patch; file the two follow-ups (TR-3 unsurfaced; `is_numeric_token`
notation gaps); `--native-only` stands; `orphan_rows` stays out; no majority vote; no new constants;
#201 correctly held.

**Fable's one-line conclusion:** the single most valuable next action is neither narrowing nor
swapping — it is **hand-judging the 62 TR-3 pages so the escalation gate is built on a defect rate,
not a firing rate.** That measurement is cheap and it gates everything else.

---


## 5. Follow-ups surfaced and not yet filed

1. **`is_numeric_token` rejects `.034` and U+2217 significance stars** (`native_verifier.py`).
   Makes any numeric-shape qualifier misfire on real econometrics notation.
2. **62 of 245 table pages carry a detected TR-3 geometry hard-fail that nothing surfaces.**
   `has_unverifiable_table_region` is computed on every table page and consumed only in
   conjunction with `native_table_structure_failed` (`manifest.py:321-325`,
   `orchestrator.py:4741-4748`). **This may be the most valuable thing found this session** — a
   correct signal, already computed, thrown away. Independent of B1.
3. **Nothing re-runs a structural check over an accepted VLM grid.** A broken native table is
   flagged; a broken model table ships clean.
4. **TICKET-B2** (correct GH-49's routing claim) is unblocked and should state what the
   structural half of the escalation gate actually is.
5. **A pre-existing D3 double-count** was fixed inside #200 rather than filed separately. Flagged
   by the implementer; noted here in case it should have been its own commit.
6. **GH-152 A2 may need `born_digital.py`** if left-to-right reading order stays in its
   `Done when` — `born_digital.py:1201` re-sorts by `y0` alone. Needs an explicit ownership grant.
7. Already filed by others during the session: **#195** (B1's rejection ships as a quiet warning —
   scoped too narrowly, the loss can be whole-table), **#197**, **#198**, **#189**.

---

## 6. Environment facts worth not rediscovering

- **The paper corpus spans two locations and neither is complete.** iCloud
  (`~/Library/Mobile Documents/com~apple~CloudDocs/Library/Papers/papers`) has 407 PDFs, **45
  evicted to 0-byte placeholders**. ProtonDrive (`~/Library/CloudStorage/ProtonDrive-*/Papers`)
  has 277, essentially all real — and covers **all 45** gaps. The union is complete. Google Drive
  holds a third archive copy that must **not** be read from (kept quit by design, streams rather
  than stores). Copy to `/tmp` and verify byte size; never open in place.
- **Fable was returning 529 for ~1 in 5 requests** for several hours. It is not down, it is
  degraded — a one-shot probe that hits a bad roll exhausts its retries and looks like an outage.
  Wave 3 was re-cast Rule→Grok, Ratify→Opus to get around it.
- **Grok can thrash.** In the #200 review panel it made 39 tool calls, 13 of them the *same*
  `Read` on `orchestrator.py`, with zero cache reads and flat input tokens. It narrates progress
  while looping. Check for repeated identical calls before trusting a long-running Grok agent.
- **Gemini Flash died on "Prompt is too long"** at ~2.9M accumulated cache-read tokens — a local
  ceiling, not the model's limit (the request never left the machine: `model: "<synthetic>"`,
  zero input tokens). Re-running with a lean brief (no full-suite run, per-file diff reads)
  completed fine. Worth checking that route's configured limit separately.
- The **wave-2 and wave-3 workflow scripts** live at `/tmp/socr-wave2/` and `/tmp/socr-wave3/`.
  `/tmp` is not durable. If they are worth keeping, commit them.

---

## 7. Process lessons — these earned their place

1. **Measure at the caller.** Wave 2's #200-sibling review returned a blocking "100% table loss"
   finding measured against `reconstruct_table_regions` in isolation. End to end the loss was
   **zero** — `extract_structured`'s `if not table_regions:` gate fires the chart-aware rowizer
   and recovers the table. Measuring one rung is not evidence about the ladder. Nearly cost a
   round of surgery on a non-bug.
2. **A green suite is not a guard.** Three consecutive review rounds on #200 each found a test
   that advertised a guarantee it did not enforce. Require proof that each new test *fails* when
   its production line is reverted. Round 3's fix was the right shape: extract the predicate into
   one shared function both production and tests call, so they cannot drift.
3. **Check the premise before staffing.** Three of four wave-3 tickets were stale. The check cost
   one stage and saved three wrong implementations.
4. **Write the retarget down before dispatching.** Agents read `TICKETS.md` from `main`. Twice
   this session the stale-instructions trap was only avoided by merging a docs PR first.
5. **Do not let a doer grade its own work** — held throughout, and it is what caught every one of
   the above.

---

## 8. Things I am not confident about

- **Whether B1 is worth the effort at all.** It has consumed a design pass, three review rounds,
  a fourth finding, a corpus measurement and a reframe. Its value lands on a fallback path and
  under `--native-only`. The reframe makes it more valuable *in principle*; whether the rebuilt
  version pays for itself is unproven. The TR-3 gap (§5.2) may be a better use of the same hours.
- **Whether the 26.9% firing rate would actually have hurt in practice.** It fires on pages that
  already route to the model. The harm is a wrong *label* on correct pages, not wrong output.
  That is real (it is your audit trail) but it is not content loss, and I may have escalated it
  harder than its severity warranted.
- **The premise-check retargets are one ruling each**, from one model, not independently verified.
  They are well-evidenced and I believe them, but they have not had the adversarial pass that
  code gets. A wrong retarget is expensive in a different way — it re-scopes work that was fine.
- **I do not know what the escalation signal should DO** (§4.2). The decision to rebuild is
  sound; the design is genuinely open, and I would not want an implementer guessing at it.
- **I called Fable "down" twice before checking properly**, and reported "no errors" on a stage
  that was failing because I counted starts and results but not failures. Both were caught by the
  owner looking at the screen. Where a claim in this document rests on my counting rather than on
  a file or a command output, treat it as provisional.
