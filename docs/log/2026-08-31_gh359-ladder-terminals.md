# 2026-08-31 — GH-359: pin the seven GH-353 ladder terminals

Pins the seven unpinned or self-contradictory terminals in
`docs/log/2026-08-30_table-judge-ladder.md` (GH-353 / #354). This note is
the spec those seven questions are coded against; the 2026-08-30 design
stands for everything else.

Flag `--table-judge-ladder` stays default OFF. GH-326 / GH-322 stay closed.
GH-189 stays open. Bake-off stays on GH-356.

## Position

Independent two-rung quorum, not a first-rung veto. CLI₁ is the cheaper,
weaker judge by construction (glm-flash vs gemini); a first-look FAIL must
not lock the stronger judge out, and a second look is not a second look if
it is handed the first look's complaint.

Rejected alternative: FAIL-immediately-REJECTED, findings-on-B, and
mechanical-as-REJECTED. That design makes CLI₁ a hard veto and CLI₂ a
rubber stamp of the complaint — the path that most needs an independent
witness becomes ornamental.

The deciding trade-off: veto power belongs at exhaustion of an *independent*
second look, not at first look. Independence is void if CLI₂ is primed with
CLI₁'s findings.

## The seven rulings

### 1. Last-rung PASS+low is `TABLE_UNVERIFIED`

A last-rung PASS with `confidence=low` is `TABLE_UNVERIFIED`, except when a
preceding rung already answered PASS (any confidence): two real witnesses
in agreement are a quorum, even if both are low. A lone low-confidence PASS
(no preceding PASS, or a preceding ¬S1/FAIL) is not verification.

Reason: the opening contract is "a table that cannot be verified does not
ship SUCCESS". PASS+low is the judge saying it is not sure. That is "cannot
verify", not "looked and said no", so the terminal is UNVERIFIED
(retryable), not REJECTED, and not a third status. Implicit SUCCESS is the
contract inverted; a third terminal would split retry semantics without
changing them.

### 2. CLI₂ may overrule CLI₁ FAIL. "FAIL trusted at any rung" is dropped.

A FAIL at a non-last rung is a tiebreak: the next rung is called. A
high-confidence PASS at any later rung accepts. A last-rung FAIL is
`TABLE_REJECTED`. CLI₁ FAIL + CLI₂ PASS+low is still UNVERIFIED (ruling 1:
one weak witness after a FAIL is not corroboration).

Reason: the A/B/C table is the mechanism the ladder was built around. "FAIL
trusted at any rung (never silently overridden downward)" is the sentence
that contradicts it, and it is the one we drop. Immediate REJECTED on CLI₁
FAIL would give the weaker judge a hard, non-retryable veto and make CLI₂
dead on the FAIL path — the path where a second opinion is worth the most.
"Never overridden downward" is true in a different sense we keep: a FAIL is
never silently discarded into SUCCESS without a later real PASS; last-rung
FAIL is REJECTED; mixed B then C is UNVERIFIED (ruling 3), not a quiet
accept.

### 3. Mixed B then C is `TABLE_UNVERIFIED`

CLI₁ = B (looked, said no) then CLI₂ = C (nobody could look) exhausts to
`TABLE_UNVERIFIED`, not `TABLE_REJECTED`.

Reason: REJECTED means the ladder exhausted on B — models looked and said
no at the end. CLI₂ never looked, so we do not have that exhaustion. The
CLI₁ FAIL is preserved on the audit trail / sidecar. Retry is correct
because the stronger judge never voted. Pinning this REJECTED would
collapse to ruling 2's rejected alternative (a single-rung FAIL is
terminal). The two terminals stay distinct: a FAIL a model produced is not
forgotten, but it is not a finished content verdict until the ladder
exhausts on B.

### 4. Judge input is crop + markdown. Nothing else.

B-escalation does not carry findings. The next rung is called with the same
crop and the same markdown, and with `prior_findings=None`. The prompt
never injects a prior verdict. Findings stay on the audit trail for humans
and for a future repair loop; they are not judge input.

Reason: handing CLI₂ the complaint anchors it toward FAIL, which makes
ruling 2's override a rubber stamp rather than a second look. The B≠C
distinction that remains is the terminal/retry split (ruling 2 vs 3) and
the `prior_was_pass` corroboration bit (ruling 1), not extra input. We ship
the independence sentence and drop the payload sentence.

### 5. Mechanical check is a GH-273 binding-shift detector. Fail → `TABLE_UNVERIFIED`. `fully_checked` is not a gate.

The remnant is `tables/binding.py:bind()` contradiction-only:
`contradicted_cells` or `row_label_contradictions` non-empty. That is the
GH-273 shape (identical numeric multisets, wrong cells / row labels).
Coverage gaps (`native_unbound`, `model_unbound`, `fully_checked is False`,
`no_known_contradiction`, `structural_agreement`) are NEUTRAL — they do
not demote. GH-330's 0/15 fully-checked pages is why `fully_checked` must
not be a SUCCESS gate; wiring it would re-block the wave this ladder
replaced. Do not reopen GH-326 / GH-322.

On a genuine contradiction: withhold acceptance. The table cannot finish
`ACCEPTED`; it becomes `TABLE_UNVERIFIED` (retryable). A judge `REJECTED`
is left untouched (the check is a ceiling on accept, not a floor on
reject). Mechanical evidence alone, including when every CLI rung is
unavailable, is UNVERIFIED not REJECTED: the native text layer itself can
be the culprit (GH-334), so a bookkeeping disagreement is not a content
verdict.

The mechanical check does not participate as a fake judge rung and does
not inject findings into the prompt (ruling 4). It runs at the agentic
gate, then clamps.

#### Overturn of panel #3's rung-0 composition (explicit)

Panel #3 (GH-353 E1, recorded verbatim in
`tests/test_ladder_binding_evidence.py`) said: "the mechanical FAIL is
prepended as rung 0 and tiebreaks into the real rung with the
contradiction's findings attached."

That composition is overturned on the merits, not silently rewritten:

1. Ruling 4 forbids findings injection. Prepending a FAIL rung exists to
   carry `prior_findings` into the next callable. Once that payload is
   gone, the fake rung is a lie about who looked: `bind()` is not a
   vision judge and must not occupy a ladder slot.
2. A fake last-rung FAIL would exhaust to REJECTED when no CLI is
   available. That treats a native-vs-markdown bookkeeping disagreement
   as a finished content verdict — the GH-334 case panel #3 itself
   refused. The clamp they actually ratified is withhold-accept
   (UNVERIFIED). REJECTED-via-fake-rung is leftover of the injection
   design, not the panel's content ruling.
3. The cap-on-ACCEPTED half of panel #3 stands: a later judge PASS
   cannot accept a contradicted table; a ladder REJECTED is left
   untouched.

Codex's "immediately TABLE_REJECTED" is a different overturn of the same
panel (the cap, not the composition). It is refused: `bind()` proves an
inconsistency between native geometry and emitted markdown, not that the
markdown is the wrong side. REJECTED means skip-on-resume. If native is
the culprit, that permanently skips a correct extraction. UNVERIFIED
withholds SUCCESS and retries.

### 6. Two wires: agentic produces content terminals; assemble backfills missing ones

Content verdicts (ACCEPTED / REJECTED / UNVERIFIED from a look) are
produced here:

```
# in UnifiedPipeline._phase_agentic, after the repetition guard,
# before the provisional fragment flush:
if self.config.table_judge_ladder:
    self._run_table_judge_gate(...)
```

That is not sufficient. These skip paths can emit a markdown table that
never produces a ladder event, and with the flag ON the document can
still ship SUCCESS:

- helper `engine == "chart_asset"` early return
- helper `if not witnesses: return` (no event) when shipped text later
  differs from `bo.text`
- `_load_terminal_page` resume `continue` (reuses a prior SUCCESS that
  itself skipped)
- the agentic `if` no-op'd / commented out (the old ruling-6 pin)

Cascade-halt and `bo is None` do NOT ship SUCCESS tables (remaining pages
become failure markers; `failed_pages` demotes the document).

Assemble therefore runs a completeness sweep
(`_backfill_missing_table_ladder_terminals`) on `canonical_page_texts`
before document status is built: every `find_table_blocks` table with no
accepted/rejected/unverified event becomes TABLE_UNVERIFIED. Assemble
does not re-judge and cannot turn a miss into REJECTED.

Difference pins (both load-bearing):

- Rejecting ladder, live helper: REJECTED (content still comes from
  agentic).
- Helper no-op'd, assemble live: UNVERIFIED, not SUCCESS (the skipped-
  table hole).
- Helper no-op'd, assemble no-op'd: SUCCESS — proves assemble is the
  last caller that can prevent an unjudged table shipping clean.

A helper-unit test going green while the agentic `if` is commented out
is not a content gate. An assemble completeness test going green while
the backfill is no-op'd is not a completeness gate. Native-lane pages go
through the same agentic `if`; `engine == "chart_asset"` still skips
inside the helper (B1); assemble catches residual markdown tables.

### 7. `NOT_A_TABLE` is `TABLE_REJECTED`. Not a figure reroute. GH-189 stays open.

A FAIL whose findings include `NOT_A_TABLE` is a content FAIL like the
other five codes. At B-exhaustion it is `TABLE_REJECTED`. The markdown is
not replaced with a figure asset. Crop unit is **per emitted table** (B0
witnesses), not per page: a mixed chart+table page is judged table-by-table,
and any REJECTED table makes the page REJECTED (A4 reducer).

Reason: the ladder's terminals are accept / REJECTED / UNVERIFIED. A region
the judge correctly flags as not a table is a content problem, not infra.
Rerouting to a figure asset is a new shipping path that collides with
GH-189 (chart silently dropped on mixed chart+table when the judge
*accepts*) and depends on crop-per-table vs crop-per-page, which this
ticket must not silently resolve. GH-189 stays open and is not subsumed.

## Transition table (2-rung, after these pins)

| CLI₁     | CLI₂        | terminal                         |
| -------- | ----------- | -------------------------------- |
| A high   | (not called)| ACCEPTED                         |
| A low    | A (any)     | ACCEPTED (quorum)                |
| A low    | B           | REJECTED                         |
| A low    | C           | UNVERIFIED                       |
| B        | A high      | ACCEPTED (override)              |
| B        | A low       | UNVERIFIED                       |
| B        | B           | REJECTED                         |
| B        | C           | UNVERIFIED                       |
| C        | A high      | ACCEPTED                         |
| C        | A low       | UNVERIFIED                       |
| C        | B           | REJECTED                         |
| C        | C           | UNVERIFIED                       |

Plus the mechanical clamp: any of the ACCEPTED rows above becomes
UNVERIFIED when `bind()` found a GH-273-class contradiction.

## What this does not do

- Does not flip `--table-judge-ladder` on.
- Does not reopen GH-326 / GH-322.
- Does not close or subsume GH-189.
- Does not change the GH-356 bake-off.
- Does not add a third terminal.
- Does not put a magic threshold anywhere.
