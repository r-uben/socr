# Panel synthesis — what the critics found and what I took

2026-08-20. Draft: `logs/panel-raw/00-draft-attacked.md`.
Critics: Grok (gating-safety), GPT-5.6-sol (coverage), Gemini (ticket-size and
trust). Advisory mode — each attacked the same draft under a different lens, none
saw the others' output.

Two of the three initially failed to launch: this session was started as plain
`claude`, which cannot reach non-Claude models, and the plan skill's prescribed
`agent_ctl` path is blocked by the user's own no-direct-Python hook. Grok and
Gemini were run as background vendor-CLI jobs; GPT ran as a proper subagent after
the owner relaunched under `claudex`. Worth recording because it constrains how
the night itself can be dispatched.

## Taken — structural changes

**1. One code owner for the whole night, stacked branches.** *(Grok 2,4,5,13;
GPT 10)* The decisive finding. My lanes were fiction: `#161`, `#205`, `#225` and
the chart work all write `orchestrator.py::_phase_agentic` and
`manifest.py::_winning_page_output`; `#248` and the GH-144 trio all write
`reconstruct.py` and `born_digital.py`. Grok traced this to specific line ranges.
Parallel branches from one baseline would collide inside a 900-line function and
could undo each other — literally today's #250 shape. Grok's answer to my open
question 2 was unambiguous: serialization of a lane is not enough, one owning
agent is required. Parallelism now lives entirely in triage, which is read-only.

**2. Fix tickets pre-authored, not templated.** *(Grok 1)* My draft shipped
`TICKET-E<n>` as a shape to be filled in after adjudication. Nothing can author
tickets at 03:00, so wave 4 would never start and the morning would hold tracker
comments and zero PRs. E1–E7 are now written out; E0 only selects or skips them.

**3. Evidence is machine-checked, never model-checked.** *(Gemini 4; Grok 8)*
Both independently found the same hole: an LLM adjudicator asked to verify
another LLM's `file:line` citation will mark it verified without opening the
file, and `evidence_verified` was a boolean the adjudicator wrote about itself.
V1 is now a deterministic script that resolves every citation at the pinned SHA
and matches the snippet; no model may write that field.

**4. A citation proves code exists, never that a bug is fixed.** *(GPT 2)* The
sharpest coverage finding and the main guard against wrong closes. `ALREADY-FIXED`
now additionally requires `fixed_by_commit`, an ancestry proof, the issue's
acceptance criteria mapped one by one, and a passing reproducer. GPT named the
issues most at risk of a false "already fixed" — #220, #127, #215, #144/#151/#205,
#140, and #156/#163 whose fixes exist only on non-main branches.

**5. Verdict taxonomy extended.** *(GPT 3)* `STILL-VALID` wrongly implied "fixable
tonight", stranding umbrella issues (#39/#49/#56), proposals (#114/#202/#203),
architecture programmes (#155/#174–#178) and partial features (#127/#220). Added
`FIX-CANDIDATE`, `PARTIALLY-IMPLEMENTED`, `ROADMAP-UMBRELLA`, `BLOCKED-PROPOSAL`,
`CHORE-PLAN`, `DEFERRED`. Only `FIX-CANDIDATE` feeds stream E.

**6. Pinned baseline + issue snapshot.** *(GPT 1,5)* `origin/main` can move
overnight and issues can be edited by a human mid-run. Everything now refers to
one `main_sha`, and D0 skips any issue whose `updated_at` changed.

**7. Clusters adjudicate together.** *(GPT 6)* Independent batch assignment let
two batches reach contradictory verdicts on one mechanism. A2 now builds a
relationship map and no cluster may span batches.

**8. `depends-on` satisfied by DONE **or** SKIPPED.** *(Grok 6)* One skipped
predecessor would otherwise freeze a whole lane and the report with it.

**9. Two vendors plus escalation, wall-clock capped.** *(Grok 7)* My "all three
vendors" Done-when was a hang waiting for a dead CLI. A missing vendor is now a
split, not a blocker.

**10. Non-vacuity proven mechanically.** *(Gemini 6; Grok 14)* "CI green" is
cheatable — delete the assertion, or test a symbol the fix introduces so the
revert fails with ImportError. The new test must now be executed against the
pinned baseline and **fail there**, on an assertion about behaviour that already
exists, with both raw outputs pasted into the PR body. This is exactly the check
that caught PR #250's bad fix today.

**11. Coordinator between waves, with checkpoints.** *(GPT 13)* Nothing validated
counts or recovered from a dead vendor; coverage would shrink silently.

**12. Merges that remove ceremony.** *(Gemini 2,3)* A1/A2/A3 collapsed into one
preflight ticket; #195/#197/#198 merged into a single ticket — Grok independently
found the reason they must be one PR (they edit the same function and #198
creates an import cycle).

**13. Idempotent tracker writes with read-after-write.** *(GPT 14)* A timeout
after GitHub accepted a write would otherwise duplicate comments on retry.

**14. Terminal CI states.** *(GPT 12)* `GREEN`/`FAILED`/`TIMED-OUT`/`NO-CHECKS`
all release the successor; none blocks the morning report.

**15. F1 rewritten to GPT's section list.** *(GPT 15)* Including the row-per-issue
reconciliation, proposed-vs-executed mutations, and explicit reopen targets.

## Taken — scope narrowed on purpose

**#205 surfacing only, routing deferred.** *(Grok 3; Gemini 1)* Both flagged that
#205 forbids keying status or routing on the TR-3 hard-fail before the 62-page
set is hand-judged, and Grok noted the 25.3% firing rate shares notation gaps with
#198. Overnight we emit the AuditEvent so the signal stops being invisible; the
routing change waits for hand judgement. A 25% firing rate is not a 25% defect
rate, and acting on it unattended could delete good tables — the cardinal-rule
violation this whole plan exists to prevent.

**#248, #249, #189, #144 step 3 deferred entirely.** All need corpus measurement
or hand judgement. Gemini said so directly; GPT's feasibility classification made
it systematic.

**#221+#227 gated on #159.** *(Grok 5)* If the timeout stub cannot carry backend
identity, E7 is skipped with that reason rather than half-built.

## Disagreement surfaced, not resolved

**Autonomous issue closure.** Gemini argued the night must not run `gh issue
close` at all, and stage a one-click apply instead; GPT independently demanded a
second non-triager approval gate before any irreversible mutation. Three critics
under three different lenses all landed on the same danger. The owner has granted
autonomous closure. Rather than overrule either side, `tracker_mode` is a switch:
`staged` (default) preserves the grant while routing closes through a morning
apply; `direct` executes overnight. The owner sets it before launch.

## Rejected

**Gemini's suggestion to fold the preflight canary into a startup script only.**
Partially taken — A1/A2/A3 merged — but the isolation canary stays a named,
re-runnable script because every E worktree must re-prove isolation, not just the
first one (Grok 15 makes the same point).
