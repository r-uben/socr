# GH-151 B1 — decision: rebuild as an escalation signal, not a status label

Date: 2026-08-14
Status: **direction settled by the owner.** Supersedes the shipping shape of PR #200.
Companion: `2026-08-14_gh151-b1-predicate-design.md` (the measurement this rests on),
`2026-08-13_gh151-b1-design.md` (the earlier design pass).

## What B1 was written to be

From `docs/plans/gh151-structural-gate/TICKETS.md`, the context line: `2024__bauer_pflueger_sunderam`
p26 ships at **100% word recall with 0 tokens missing and an unusable table** — spanning headers in
body cells, `R2` and its values on different rows, coefficients unbound from their standard errors.

That page exists in the ticket to **falsify word recall as a sufficient routing gate** — a claim made
on GH-49. Recall measures TOKEN loss and is blind to STRUCTURAL loss. TICKET-B2's whole job is to go
back to GH-49 and record the correction: *recall AND structure together gate escalation.*

**B1 was designed to supply the structural half of an escalation signal.**

## What it became, and why

PR #200 implements B1 as a **status label**: a `PageState` flag, an `AuditEvent`, manifest locks, and
membership in `TABLE_DISTRUST_KINDS`. It changes what a page is *called*. It does not change what the
pipeline *does*.

That happened because of a ruling taken on 2026-08-13 and recorded in TICKET-B1: the flag must never
override `--native-only`, and `native_table_structure_defective` must never enter
`PageState.needs_repair`. Three models converged on it and it is correct **for that flag** — but
`needs_repair` is precisely the mechanism that would have made B1 a routing signal. The ruling was
sound and severed B1 from its purpose at the same time.

The symptom was that the work kept costing more than it seemed to be worth: three review rounds, a
design pass and a corpus measurement, for a label on a fallback path.

## Why the label version is also weak on its own terms

Measured over 32 papers / 245 native-table pages / 422 blocks
(`2026-08-14_gh151-b1-predicate-design.md` §2):

- The gate as implemented fires on **26.9% of native table pages**. Reviewers demonstrated four
  legitimate shapes it flags (textual column headers, panel sub-headings, units rows, alternative
  numbering).
- `ragged` alone fires on **2 of 422 blocks (0.5%)** and does **not** fire on B1's own acceptance
  fixture. Falling back to `ragged` ships nothing.
- `_is_canonical_column_band` suppresses **0 of 422** blocks. The one hardcoded exception is dead on
  real data; adding siblings for `[1]`, `(i)`, `1a` would add more dead constants.
- Geometry does **not** separate the two cases. B1's own fixture measures y-gaps `[18,18,18,18,18]` —
  a full row-pitch split, geometrically identical to a heading band. A genuine sub-pitch shift does
  not produce the split at all; the extractor binds it correctly.

So the shape predicate is a heuristic that cannot be made precise, and the alternative in the same
family is inert.

## The decision

**Rebuild B1 as an escalation signal.** The question it must answer is not "what do we call this
page" but "**is the output about to ship structurally sound, and if not, is escalation warranted?**"

Two things make this the right shape:

1. **The signal already exists and is discarded.** `has_unverifiable_table_region` (TR-3) is computed
   on **every** table page and consumed only in conjunction with `native_table_structure_failed`
   (`manifest.py:321-325`, `orchestrator.py:4741-4748`). **62 of 245 pages (25.3%) carry a detected
   geometry hard-fail that nothing surfaces.** That is a verified geometric failure, not a shape
   heuristic — better evidence than anything B1 invented — and it is being thrown away. The design
   note flags this as "the same bug shape B1 exists to fix".

2. **The check must run on the winner, not only on the native attempt.** Raised independently during
   the wave-3 review (Grok F4): B1 marks native output defective, but nothing re-runs a structural
   check over an **accepted VLM grid**. A broken native table is flagged; a broken model table ships
   clean. The failure mode that matters in a citation corpus is silent *replacement* — plausible
   wrong numbers — not throughput.

Note the routing context this sits in: table pages already route to the model
(`_is_trusted_native_without_ocr` ends `return not self._page_has_tables(...)`). So the value is not
"send tables to OCR" — that already happens. It is deciding whether the **winning** output is good
enough to stop, or whether the ladder should escalate further.

## What this means for PR #200

Open question for the next session. The wiring (PageState field, audit event, manifest locks,
`TABLE_DISTRUST_KINDS` membership, the shared `structural_gate_fires` entry point) is reviewed and
sound, and an escalation version needs most of it. The **predicate** and the **purpose** are what
changed. Three options, in preference order pending review:

- **Land the plumbing, replace the predicate.** Keep #200's wiring; swap the shape heuristic for the
  TR-3 geometry signal; add the winner-side check. Risk: merges a gate that fires on 26.9% of correct
  pages, even briefly.
- **Narrow first, then extend** (the design note's Option B: two constant-free qualifiers, 26.9% →
  14.3%). Buys correctness now, still ships a heuristic.
- **Close #200 and rebuild.** Cleanest statement of intent, discards reviewed wiring.

## What does NOT change

- The `--native-only` ruling stands. Under `--native-only` the user has said do not OCR; a structural
  signal is recorded and surfaced, never acted on by routing. An escalation signal that fires in
  default mode does not contradict this — `--native-only` means the ladder is off.
- `orphan_rows` stays out of any gate: it fires on 69.9% of blocks and IS a legitimate standard-error
  row.
- No majority-vote predicate, ever (the reason GH-151 A2 / PR #184 was closed).
- No magic thresholds.

## Follow-ups this surfaces, to file separately

1. `is_numeric_token` rejects `.034` and U+2217 significance stars (`native_verifier.py`) — makes any
   numeric-shape qualifier misfire on real econometrics notation.
2. **62/245 table pages carry an unsurfaced TR-3 geometry hard-fail.** This is a live content-integrity
   gap independent of B1 and is arguably more valuable than B1 itself.
3. Nothing re-runs a structural check over an accepted VLM grid (Grok F4) — silent replacement.
4. TICKET-B2 (correct GH-49's routing claim) is unblocked by this decision and should now say what the
   structural half of the escalation gate actually is.
