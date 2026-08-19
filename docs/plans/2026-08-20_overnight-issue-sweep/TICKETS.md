# TICKETS — overnight autonomous issue sweep (socr)

Status keys: `TODO` · `WIP` · `DONE` · `SKIPPED` · `BLOCKED`.
`depends-on` gates dispatch and is **satisfied by `DONE` OR `SKIPPED`** — a
skipped predecessor must never freeze a lane overnight.

Every agent receives `CONTRACT.md` verbatim. Every ticket writes its output to a
named path; a ticket that cannot produce that path exits and records why.

**Run mode** — `state/config.json`: `tracker_mode: "agent-gated"` (owner's
decision, 2026-08-20). Every tracker mutation is staged into
`actions/tracker_actions.json` first and nothing executes on a triager's word.
An independent two-agent review board then approves or rejects each staged
action, and approved ones execute overnight. Rejected or split actions do not
execute — they go to the morning report as the owner's decision queue. The
owner is not the reviewer; agents revise and decide.

---

## Stream A — Preflight (wave 0, serial, one agent)

### TICKET-A1 — Pinned baseline, isolation canary, contract · TODO · depends-on: none · wave 0
**Problem:** Three separate failure modes make every later result void: a moving
`origin/main`, a worktree that silently tests the main checkout's source, and a
dead `gh` token that turns tracker writes into no-ops or fabricated success.
**Do:** Resolve and record `main_sha` from `origin/main`. Create the base
worktree at that SHA. Write `bin/isolation_canary.sh` (exit 1 unless
`socr.__file__` is under the caller's worktree) and prove it by breaking a
sentinel symbol in the worktree, watching a test fail, restoring it. Run
`gh auth status`; on failure create `state/ABORT`. Copy `CONTRACT.md` into
`state/` with `main_sha` substituted.
**Files:** `baseline.json`, `bin/isolation_canary.sh`, `state/`
**Done when:** `baseline.json` contains `main_sha`, `created_at`, `gh_login`;
`bash bin/isolation_canary.sh` exits 0 from inside the worktree and exits 1 from
the main checkout; the sentinel break/restore transcript is in `logs/`.

### TICKET-A2 — Issue snapshot, cluster map, batches · TODO · depends-on: A1 · wave 0
**Problem:** Batching from memory triages issues that don't exist. Worse,
assigning related issues independently lets two batches reach contradictory
verdicts on one mechanism (header attribution #215/#245; table destruction
#144/#146/#151/#195/#197/#198/#205/#226; chart arbitration
#150/#167/#181/#189/#248/#249; cascade halt #221/#222/#227; structure loss
#127/#223; CLI semantics #139/#142/#168; cost identity #154/#159/#160; equations
#140/#157/#164/#165/#219; architecture #155/#174/#175/#176/#178).
**Do:** Snapshot every open issue at `main_sha` with `number`, `title`,
`updated_at`, `body_hash`. Build the relationship map (`duplicate_of`,
`overlaps`, `supersedes`, `umbrella_of`, `must_adjudicate_together`). Assign
batches so **a cluster is never split across batches**. Tag each issue with a
measurement-feasibility class: `PUBLIC-REPRO` / `SYNTHETIC-REPRO` /
`LOCAL-CORPUS` / `CORPUS-UNAVAILABLE` / `HUMAN-JUDGEMENT`.
**Files:** `issues_snapshot.json`, `clusters.json`, `batches.json`
**Done when:** the deduplicated union of batch membership equals the snapshot's
issue set exactly; no cluster spans two batches; every issue carries a
feasibility class.

---

## Stream B — Triage (wave 1, parallel, READ-ONLY)

Two vendors per batch plus escalation on disagreement — not three. A third seat
buys little once evidence is machine-checked, and a dead vendor CLI must never
block a batch.

### TICKET-B1..B4 — Triage batch 1..4 · TODO · depends-on: A1,A2 · wave 1
**Problem:** Much of the backlog is stale after #243/#246/#247/#250, and some
reports are wrong rather than unfixed — but a citation proves only that code
exists, never that a bug is resolved.
**Do:** For each issue, emit one verdict from this taxonomy:
`FIX-CANDIDATE` (still valid AND fixable in one night) · `ALREADY-FIXED` ·
`MISREPORTED` · `DUPLICATE` · `PARTIALLY-IMPLEMENTED` · `ROADMAP-UMBRELLA` ·
`BLOCKED-PROPOSAL` · `CHORE-PLAN` · `NEEDS-MEASUREMENT` · `DEFERRED`.
Each verdict carries citations `{path, line, snippet}` at `main_sha`.
`ALREADY-FIXED` additionally requires `fixed_by_commit`, its ancestry proof, the
issue's acceptance criteria mapped one by one, and a reproducer/regression
result. No repo writes of any kind.
**Files:** `triage/<batch>/<vendor>.json`
**Done when:** every issue in the batch has a verdict from at least two vendors
with a schema-valid record; a vendor that times out (wall-clock cap in
`state/config.json`) is recorded as `MISSING`, which C treats as a split rather
than a blocker.

---

## Stream V — Mechanical evidence check (wave 1.5, script, NO model)

### TICKET-V1 — Verify every citation without an LLM · TODO · depends-on: B1..B4 · wave 1.5
**Problem:** Asking a model to check another model's `file:line` evidence is the
one place fabrication survives — an adjudicator will mark a plausible-looking
citation verified without ever opening the file.
**Do:** A deterministic script. For each citation: `git show <main_sha>:<path>`
must succeed, the line must exist, and `snippet` must appear on that line. For
each `ALREADY-FIXED`: `git merge-base --is-ancestor <fixed_by_commit> <main_sha>`
must succeed, and `git log -L<line>,<line>:<path>` must show that commit actually
touched the cited region. Compute `evidence_verified` per verdict. No model may
write this field.
**Files:** `bin/verify_citations.py`, `triage/verified/<batch>.json`
**Done when:** every verdict carries a computed `evidence_verified` plus the
failure reason when false; the script's exit code is non-zero if any
`ALREADY-FIXED` passed without all three proofs.

---

## Stream C — Adjudication (wave 2, vendor ≠ any triager of that batch)

### TICKET-C1..C4 — Reconcile batch 1..4 · TODO · depends-on: V1 · wave 2
**Problem:** The doer must never grade its own work, and clustered issues must be
decided together or they contradict each other.
**Do:** Consider only verdicts with `evidence_verified: true`; anything else is
forced to `NEEDS-MEASUREMENT`. Reconcile per cluster, not per issue. Unanimous →
final. Split → escalate to a third vendor and record both readings verbatim.
Emit the final disposition ledger, one row per issue.
**Files:** `verdicts/<batch>.json`
**Done when:** every issue in the batch has exactly one final disposition; every
split records both readings; no row has `evidence_verified: false` and a
non-`NEEDS-MEASUREMENT` disposition.

---

## Stream D — Tracker actions (wave 3, SERIAL — one writer, never parallel)

### TICKET-D0 — Build the action manifest · TODO · depends-on: C1,C2,C3,C4,A1 · wave 3
**Do:** For every actionable disposition emit a record with a deterministic
`action_id`, the issue's `updated_at`/`body_hash` from the snapshot, the
disposition, its evidence, and the exact proposed comment text. Any issue whose
live `updated_at` differs from the snapshot is marked `SKIPPED-CHANGED` — a human
touched it while we worked, so we do not act on it.
**Files:** `actions/tracker_actions.json`, `bin/apply_tracker_actions.sh`
**Done when:** every actionable row has a unique `action_id` and a rendered
comment; the apply script is executable and idempotent (it re-reads live state
and skips any action whose marker comment already exists).

### TICKET-DR — Review board: agents approve or reject each staged action · TODO · depends-on: D0 · wave 3
**Problem:** A staged action is only as safe as its reviewer, and the owner is
asleep. All three panel critics independently warned that a unanimous triage
verdict can still be category-wrong — a closure that reads as well-evidenced and
is simply about the wrong thing. So the gate is agents, and it is deliberately
adversarial rather than confirmatory.
**Do:** Two reviewers per staged action, on different vendors, **neither of which
triaged or adjudicated that issue**. Each is prompted to REFUTE the action: for a
close, to find any acceptance criterion the `fixed_by_commit` does not actually
satisfy; for a correction or a new issue, to find the claim that is not
supported by its evidence. A reviewer that cannot open the cited code must
reject. Decision rule: **both approve → `APPROVED`**; anything else →
`HELD-FOR-OWNER`, with both readings recorded verbatim.
**Files:** `actions/review/<action_id>.json`, `actions/decisions.json`
**Done when:** every staged action carries exactly one of `APPROVED` or
`HELD-FOR-OWNER`; no reviewer appears as a triager or adjudicator of the same
issue; every `HELD-FOR-OWNER` records why, in a sentence a human can act on.

### TICKET-D1 — Execute approved closes · TODO · depends-on: DR · wave 3
**Do:** `ALREADY-FIXED` actions that are `APPROVED` by the review board and
survived D0's freshness check. Execute serially, then read the issue back and
confirm the state actually changed. Never close a `HELD-FOR-OWNER` action.
**Done when:** every closed issue carries a comment naming `fixed_by_commit`, the
acceptance criteria it satisfies, and the reproducer result — and the action
ledger records the returned URL and post-write verified state. Zero closes is a
valid outcome, recorded as such.

### TICKET-D2 — Corrections in place · TODO · depends-on: DR · wave 3
**Problem:** #249 is the template — a wrong report gets corrected, not closed.
**Done when:** each `MISREPORTED` issue has a comment stating what the original
claimed, what is actually true, and the measurement behind it; the issue remains
open; nothing was closed by this ticket.

### TICKET-D3 — File genuinely new defects · TODO · depends-on: DR · wave 3
**Problem:** Nothing in B or C produces new defects, so without an explicit input
this ticket either does nothing or invents duplicates.
**Do:** Its input is `discoveries/*.json` — candidates recorded by B/C/E agents
whenever they hit a defect no open issue covers. Each candidate needs a
reproduction, observed-vs-expected, a search across open AND closed issues, and
an independent validation. Then decide `NEW` vs `COMMENT-ON-EXISTING`.
**Done when:** every filed issue has a reproduction and an observed/expected, and
every candidate that was not filed records why.

---

## Stream E — Fixes as proposed PRs (wave 4)

**One code owner for the whole night.** The panel's decisive finding: the
candidate fixes do not separate into lanes. `#161`, `#205`, `#225` and the chart
work all write `orchestrator.py::_phase_agentic` and `manifest.py::_winning_page_output`;
`#248` and the GH-144 trio all write `reconstruct.py` and `born_digital.py`.
Parallel branches from the baseline would collide inside one 900-line function
and can silently undo each other — the exact shape of today's #250 defect.
So: **one agent, one stacked branch chain**, each PR based on its predecessor,
never on the baseline. Independent reviewers are still different agents.

### TICKET-E0 — Schedule the fix queue · TODO · depends-on: C1..C4 · wave 4
**Problem:** Fix tickets that do not exist at dispatch time cannot be invented at
03:00 — the night would end with tracker comments and zero PRs.
**Do:** The candidates below are pre-authored. E0 only selects: a candidate whose
issues are all `FIX-CANDIDATE` proceeds; otherwise it is marked `SKIPPED` with the
reason. Record the stack order and each PR's base branch. Every `FIX-CANDIDATE`
not selected gets a written defer reason.
**Files:** `fixes/queue.json`
**Done when:** every pre-authored candidate is `SELECTED` or `SKIPPED` with a
reason, and every `FIX-CANDIDATE` issue appears in exactly one of: a selected
candidate, or the deferred list with a reason.

Each selected candidate runs: **implement → independent review (different model)
→ revise → push → PR → CI to a terminal state → final verify.** Terminal CI
states are `GREEN`, `FAILED`, `TIMED-OUT`, `NO-CHECKS`; all four release the
successor, none blocks the report.

**Universal Done when for E1..E7:** branch pushed from its stack base; PR open
and NOT merged; `bin/isolation_canary.sh` passed in that worktree;
`uvx ruff@0.16.0 format --check .` clean; and the non-vacuity proof recorded in
the PR body — the new test executed against `main_sha` **fails**, executed on the
branch **passes**, with both raw outputs pasted. The reverted-fix test must fail
on an assertion about behaviour that already exists at `main_sha`, not on an
ImportError for a symbol the fix introduced.

### TICKET-E1 — #161 resume ledger trusts judge-rejected pages · TODO · wave 4 · stack base: baseline
`_load_terminal_page` gates on `status == SUCCESS` without checking
`audit_passed`, so resume restores text every judge rejected.

### TICKET-E2 — #225 fabricated image URLs ship under SUCCESS · TODO · depends-on: E1 · stack base: E1
No gate validates that an emitted URL is a local asset or present in the PDF's
own links. Touches `normalizer.py::strip_phantom_images` and the assemble path.

### TICKET-E3 — #205 surfacing ONLY (AuditEvent, no routing change) · TODO · depends-on: E2 · stack base: E2
**Deliberately narrowed.** #205 itself forbids keying status or routing on the
TR-3 hard-fail before the 62-page set is hand-judged. Overnight we emit the
unconditional AuditEvent so the signal stops being invisible. The routing change
(#144 step 3) is explicitly **deferred to the morning** — a 25% firing rate is not
a 25% defect rate, and acting on it unattended risks deleting good tables.

### TICKET-E4 — #147 landscape pages rowized along the wrong axis · TODO · depends-on: E3 · stack base: E3
Detection already exists (`dominant_text_direction`); the fix is to refuse the
native lane and route to the VLM, whose rendered image is upright.

### TICKET-E5 — #195 + #197 + #198 as ONE ticket · TODO · depends-on: E4 · stack base: E4
Merged on the panel's finding: all three edit
`reconstruct.py::_destroyed_numeric_tokens`, and #198 introduces an import that
`native_verifier` already imports from — three parallel PRs would conflict and
create a cycle.

### TICKET-E6 — #222 extract the backend probe behind an interface · TODO · depends-on: E5 · stack base: E5
`probe_ollama_idle` is hardcoded to localhost. Extracting the probe interface is
separable from any loop logic and lands cleanly on its own.

### TICKET-E7 — #221 + #227 cascade latch · TODO · depends-on: E6 · stack base: E6
Must ship together: #227 explicitly warns that fixing the probe alone makes
behaviour worse. **Gate:** if the timeout stub cannot carry backend identity
(the `#159` dependency), E7 is `SKIPPED` with that reason rather than half-built.

**Deferred by design, not omission:** #144 step 3, #248, #249, #189 — all need
corpus measurement or hand judgement and cannot be settled unattended.

---

## Stream W — Coordinator (between every wave, serial, Claude)

### TICKET-W1..W5 — Wave transition · TODO · wave: after A, B, V, C, D/E
**Problem:** Nothing else validates counts, applies zero-output semantics, or
recovers from a dead vendor; the graph would silently shrink its coverage.
**Do:** Validate the previous wave's schemas and row counts against
`batches.json`; write `state/checkpoint-<wave>.json` so a crashed run resumes
rather than restarts; mark failed inputs `SKIPPED` explicitly; dispatch only
successors whose deps are `DONE` or `SKIPPED`.
**Done when:** the checkpoint exists and its issue count reconciles to the
snapshot, with every discrepancy named.

---

## Stream F — Morning report (wave 5)

### TICKET-F1 — The artifact the owner reads in 20 minutes · TODO · depends-on: every D and E ticket terminal · wave 5
**Done when:** one document contains all of:
`Baseline` (pinned SHA, live drift since) ·
`Coverage reconciliation` (one row per snapshot issue → exactly one final
disposition; totals reconcile to the starting count plus newly filed) ·
`Autonomous mutations` (staged vs approved vs executed, with URLs and reopen
targets) ·
`Held for you` (every action the review board refused to approve, with both
reviewers' readings and a one-line ask) ·
`PR decision queue` (per PR: head SHA, CI conclusion, reviewer verdict,
non-vacuity proof, stack position) ·
`Deferred measurements` (what needs the corpus, and the exact command to run) ·
`Failures / abort` (anything that hit the latch, timed out, or was skipped) ·
`Human actions` (ranked, shortest first).
Zero closes, zero PRs, or an aborted night are all valid outcomes and must be
reported as such rather than dressed up.
