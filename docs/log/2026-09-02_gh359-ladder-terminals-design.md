# 2026-09-02 — GH-359 ladder terminals: design pass before the P1 flag flip

Read-only pass on `main` (worktree `socr-p1`, branch `docs/359-ladder-terminals-design`),
ordered by the conceptual revision's P1: "flip `table_judge_ladder` on, after #359 pins the
terminals" (`docs/log/2026-09-01_conceptual-revision.md`). No source or test was touched.

**Verdict up front:** the three terminals #359 names are no longer unspecified. They are
pinned by `docs/log/2026-08-31_gh359-ladder-terminals.md` (rulings 1–3), the code matches
the pins, #359 and its leftover #381 are both CLOSED, and every ruling has a `process()`-
level difference test (`tests/test_gh359_ladder_terminals.py`). What remains before the
flip is not design but evidence and mechanics, listed at the end.

One fact frames all three: **a ladder terminal never removes content.** The manifest guard
(`src/socr/core/manifest.py:1507-1542`) demotes the finalized page to
`WARNING`/`audit_passed=False` with the terminal as `failure_mode`; the table text ships.
Every pin below chooses a *label plus a retry policy*, not a content gate. The corpus rule
(a wrong number is worse than a missing one) is served by "never SUCCESS", which all
candidates except implicit-accept satisfy — the real forks are elsewhere.

## Terminal 1 — PASS + low confidence at the LAST rung

**Code today.** `src/socr/judge/table_ladder.py:150-178`. High-confidence PASS accepts at
any rung (`:156-162`). A low-confidence PASS at the last rung accepts only when the
immediately preceding rung was itself a real PASS of any confidence (`:163-170`,
`prior_was_pass` set at `:177`, reset by any FAIL or ¬S1 at `:145`/`:191`); otherwise it
exhausts to `TableLadderOutcome.UNVERIFIED` with `final_verdict=None` (`:171-176`). The
gate maps that to `FailureMode.TABLE_UNVERIFIED` (`orchestrator.py:2905-2908`).

**Design record.** Ruling 1: a lone low PASS is `TABLE_UNVERIFIED` (retryable), except
two real witnesses in agreement are a quorum even if both are low. The 2026-08-30 note's
contradiction (Row A "accept" vs the opening contract) is gone.

**Candidates.** (a) *Implicit SUCCESS* — the original hole. False-accept: the judge says
"I am not sure" and the corpus stamps it clean; the worst mode there is. (b) *REJECTED* —
false-reject: "not sure" is not "looked and said no"; REJECTED is skip-and-keep on resume
(`orchestrator.py:5305-5313`), so a diffident-but-correct table is mislabeled forever.
(c) *UNVERIFIED* (chosen) — no false-accept (never SUCCESS); the failure mode is cost,
not correctness: an UNVERIFIED page reprocesses on every resume, and deterministically
weak judges never converge. (d) *Third terminal* — splits retry semantics without
changing them.

**Recommendation: keep (c).** The soft spot is the quorum clause: two correlated weak
judges (same crop, both VLMs) can agree low-low on a hard table and accept. That is the
one sub-pin worth an owner look — sharp question 1.

## Terminal 2 — mixed B then C (CLI₁ FAIL, CLI₂ never looked)

**Code today.** `table_ladder.py:135-146`: a last-rung ¬S1 returns UNVERIFIED
unconditionally, whatever the earlier rungs said. The CLI₁ FAIL verdict is preserved in
`rung_results` for the audit trail but `final_verdict=None` — the failure is not
forgotten, and it is not a finished content verdict. D1b gives UNVERIFIED no resume
exception (`orchestrator.py:5319-5323`): the page reprocesses.

**Design record.** Ruling 3: REJECTED means the ladder exhausted on B; here the stronger
judge never voted, so the terminal is UNVERIFIED.

**Candidates.** (a) *REJECTED* — false-reject: CLI₁ is the weaker judge by construction
(glm-flash vs the gemini family); this hands it a terminal veto on a look CLI₂ never
took, and because REJECTED skips on resume, a wrong veto is permanent. (b) *UNVERIFIED*
(chosen) — false-accept: none at the SUCCESS surface. The real cost is availability:
while CLI₂ is persistently down (quota, missing binary), a genuinely bad table is
re-judged on every resume and never converges to REJECTED. There is no attempt cap
anywhere in `_load_terminal_page`.

**Recommendation: keep (b).** A mislabeled-forever good table (a) is worse than a
retried-forever bad table (b): both ship the text demoted, but (b) keeps re-examining it.
Name the missing piece rather than solve it here: UNVERIFIED has no retry-exhaustion
counter, so a persistent outage is an infinite per-resume ladder bill.

## Terminal 3 — FAIL versus the tiebreak

**Code today.** A high-confidence PASS at any rung accepts immediately, including after a
CLI₁ FAIL (`table_ladder.py:156-162`) — CLI₂ may overrule. A non-last FAIL escalates with
`prior_findings=None` (`:131`, `:180-182`); the prompt layer deletes the argument outright
(`src/socr/judge/table_prompt.py:90`). A last-rung FAIL is REJECTED
(`table_ladder.py:183-190`). CLI₁ FAIL + CLI₂ PASS-*low* is UNVERIFIED (ruling 1's
corroboration bit), not accepted.

**Design record.** Ruling 2: the A/B/C table is the mechanism; "FAIL trusted at any rung"
is the sentence that was dropped. What survives of it: a FAIL is never silently discarded
into SUCCESS without a later real PASS.

**Candidates.** (a) *FAIL at any rung is immediately REJECTED* — false-reject: glm-flash
false-FAILs (dense layout, unusual header) become permanent content verdicts and CLI₂ is
ornamental on exactly the path where a second opinion is worth most. (b) *Override only on
high-confidence PASS* (chosen) — false-accept: CLI₂ wrongly passes a table CLI₁ correctly
failed, and the ladder accepts on one judge's word; CLI₂ is the de facto single judge on
the FAIL path. The mitigation on record is the E1 mechanical clamp: a `bind()`
contradiction caps the table at UNVERIFIED even after a high PASS
(`orchestrator.py:2792-2843`), covering the measured correlated-miss class (GH-273 binding
shifts). But the clamp is blind where there are no native words
(`orchestrator.py:2613-2614`) — on scanned pages the override is bare.

**Recommendation: keep (b).** The bare-override residual on no-native-word pages is real
but narrow: those pages are the minority lane, and where no usable grid is authored at
all, P2's floor — not the ladder — already owns the ending.

## What else must be true before the flag flips

**1. The live smoke is still owed — and it is the blocking item.** All 16 tickets are
hermetic; every rung in every test is injected. The only live evidence on record is the
2026-08-30 pre-merge smoke (`docs/log/2026-08-30_gh353-ticket-a3.md`): one crop through
`agy` — schema-perfect, all six decoys caught — plus the discovery that the bare `gemini`
binary can no longer authenticate headlessly, which is why rung 2 defaults to `agy`
(`config.py:303-314`). Never done: rung 1 (`glm-5.3-flash:cloud` via `/api/chat`) against
a live ollama host; a full `process()` over a real document with both rungs live;
per-table latency against the 600 s `table_judge_timeout_sec` (`config.py:326`); `agy`
quota behaviour over a long document; and the real ¬S1 rate — the number that decides
whether a flag-on document ever reaches SUCCESS in practice. Also carried: `agy` has no
per-call model pin, so rung-2 model identity is unconfirmed; the fingerprint records the
binary, not the model.

**2. P2 fail-closed floor (on main, PR #490) — checked; no conflict, one cosmetic
overlap.** Three contact points, all handled. (i) *Selection:* the floor ending
(`manifest.py:1309-1324`) ships `ERROR`/`STRUCTURE_CLASS_LADDER_EXHAUSTED`; the ladder
guard only demotes `audit_passed=True` outputs or stamps a `NONE` failure mode, so the
floor's more specific mode always wins. (ii) *Resume:* the D1b REJECTED skip is forfeited
when the shipped winner is the floor (`orchestrator.py:5426-5433`) — P2 cold-review
finding 2's test pins exactly this. (iii) *Buckets:* a floored page that also carries a
REJECTED disposition counts in both `structure_class_floor_pages` and
`table_rejected_pages` (`orchestrator.py:6405-6415`) — the CLI summary names it twice;
cosmetic, not silent. The floor text contains no markdown table, so the assemble backfill
(`orchestrator.py:5903-5970`) does not double-flag floored pages. Ordering composes in
both directions: the gate runs in-loop (`orchestrator.py:3909-3910`), the floor is decided
at assemble; an ACCEPTED page can still floor (no grid authored ⇒ nothing witnessed), and
a REJECTED page can floor.

**3. Cost.** Flag-on adds up to two cloud calls per emitted table, plus GH-367
cell-transcription adjudication whenever the clamp fires. $0 marginal (subscriptions) but
quota-bound; the wall-clock bound is timeout × tables × rungs, and UNVERIFIED pages
re-pay the full ladder on every resume with no cap. `strict_local` + ladder makes every
table page UNVERIFIED by construction (`config.py:322-326`, `orchestrator.py:2551-2552`)
— surfaced correctly, but a strict-local user gets permanently PARTIAL documents; the
flip should say so at startup.

**4. Flip mechanics.** The fingerprint binds flag, rung identities, timeout and prompt
digest (`orchestrator.py:535-568`), so flipping the default reprocesses everything —
expected. The real work is the test audit: every golden/byte-identity test that
default-constructs `PipelineConfig` must pin the flag off explicitly, or the gate runs in
CI with unreachable rungs, every table page goes UNVERIFIED, and goldens change — worse,
they become machine-dependent (ollama up locally, absent in CI; the #253/#257 trap at
suite scale). P1's "~30 LOC" is the config line; this audit is the cost.

**5. #381 leftovers — verified closed.** The assemble writer of `table_ladder_incomplete`
is pinned at `process()` level including the resume reprocess difference
(`test_gh359_ladder_terminals.py:508-612`); the ruling-2 override test now asserts
not-UNVERIFIED on the override side (`:271-272`); the dead `{{PRIOR_FINDINGS}}` replace is
gone (`table_prompt.py:90-94`).

## THE SHARP QUESTIONS

**Q1 — Is two-low a quorum, or must acceptance require one high-confidence PASS?**
Today two low-confidence PASSes accept (ruling 1). *For the quorum:* UNVERIFIED never
converges — deterministically weak judges reprocess the page forever, and since the text
ships demoted either way, the practical stake is the label and an infinite retry bill, not
the bytes. *Against:* judge errors are correlated on hard tables (same crop, same model
class); SUCCESS is the only state that skips resume and stamps the corpus clean, and a
wrong number stamped SUCCESS is precisely the failure the corpus rule exists to prevent.
Requiring one high PASS fails closed into more UNVERIFIED — a cost problem, not a
correctness problem.

**Q2 — Should TABLE_REJECTED keep shipping the table text under WARNING, or withhold it
floor-style (marker + page PNG)?** *For withholding:* two judges looked and said no;
shipping the table anyway puts measured-bad numbers into the `.md` that downstream readers
quote without reading metadata — the D3/P2 floor already exists for exactly this shape.
*For shipping demoted:* both judges can be wrong about an unusual-but-correct table, and a
withheld table is unrecoverable while a labeled one is recoverable; decisively, a per-table
splice cannot be done safely today — P2 round 2 proved no detection-level region
enumeration exists to validate it against, so "withhold just the rejected table" would
reopen the exact circular-coverage hole P2 closed.

**Q3 — Is fail-closed the intended out-of-box posture for a machine with no reachable
rung?** Flag-on + both rungs down ⇒ every table page UNVERIFIED, no document ever clean.
*For:* that is the design's opening contract — "a table that cannot be verified does not
ship SUCCESS"; shipping unwitnessed tables as SUCCESS is the bug being fixed, and the
escape hatch exists (flag off). *Against:* default-on redefines the tool's contract from
best-effort to verified-or-labeled for every user who never heard of the ladder —
air-gapped and subscription-less users get a permanently PARTIAL corpus where today they
get SUCCESS-with-known-limits, and they will experience the flip as a regression, not a
gate.

## Grounding canary

```
$ grep -n "table_judge_ladder" src/socr/core/config.py
294:    table_judge_ladder: bool = False
323:    # ``strict_local and table_judge_ladder`` makes every rung unavailable before
$ wc -l src/socr/pipeline/orchestrator.py
7824 src/socr/pipeline/orchestrator.py
```
