# DRAFT ticket graph — overnight autonomous issue sweep (socr)

Slug: `2026-08-20_overnight-issue-sweep`

## Goal

Run unattended overnight against socr's ~62 open issues. By morning the owner
should find: stale issues **closed with evidence**, misreported issues
**corrected in place**, genuinely new defects **filed**, and real fixes waiting
as **proposed PRs with CI green**. Nothing merged. Nothing force-pushed.

## Trust boundary (owner's grant, 2026-08-20)

- FULL autonomy on the issue tracker: close, comment, file new issues.
- FULL autonomy on branches and PRs — but PRs are **proposed, never merged**.
- The owner reviews in the morning.

## Hard repo facts every agent must obey (verified this session)

1. **Editable-install trap.** `import socr` resolves to the MAIN checkout's
   `src/socr`, so a git worktree does NOT isolate code under test. Verified fix:
   run everything as `PYTHONPATH=<worktree>/src ~/venvs/socr/bin/pytest …`, and
   assert the resolved module path before trusting any test result.
2. **The main checkout is owned by another session.** Never switch its branch,
   never write in it. All work happens in per-ticket worktrees.
3. **CI has no ollama and no provider.** Any test driving agentic mode must patch
   `_available_engines_for_agentic` or it passes locally and fails in CI.
4. **Lint gate is `uvx ruff@0.16.0 format --check .`** — NOT the venv ruff, which
   is older and reports clean on files CI rejects.
5. **No `Co-Authored-By`. No `git add -A`.** Stage by name. Branch from `main`,
   never from whatever branch is checked out.
6. **Forge is GitHub** (`git@github.com:r-uben/socr.git`) — `gh` is correct here.
7. **Cardinal rule for ranking:** no silent content loss. A wrong or dropped
   number is worse than a missing one; failures must surface at page, document,
   metadata and CLI level.

## Two failure modes this harness exists to prevent

- **Silent fabrication.** A headless agent denied a tool permission will invent
  confident output and exit 0. Every verdict must carry `file:line` evidence
  quoted from `git show origin/main:<path>`, and no issue is closed on one
  model's say-so.
- **Confident wrong measurement.** Issue #249 needed three owner revisions before
  its diagnosis held; PR #250's own fix reintroduced the bug it was fixing. So:
  the triager never adjudicates its own verdict, and the implementer never
  reviews its own diff.

---

## Stream A — Preflight (wave 0, serial, Claude inline)

### TICKET-A1 — Isolated worktree + import-isolation canary · TODO · depends-on: none · wave 0
**Problem:** A worktree silently tests the main checkout's source, so every
overnight test result would be meaningless.
**Do:** Create the overnight base worktree off `origin/main`. Prove isolation:
break a sentinel symbol in the worktree's `src/socr` and show a test fails there
while the main checkout is unaffected; restore the sentinel.
**Files:** worktree only (no repo files).
**Done when:** `PYTHONPATH=<wt>/src ~/venvs/socr/bin/python -c "import socr;print(socr.__file__)"`
prints a path under the worktree, and the sentinel demonstration is recorded in
the log.

### TICKET-A2 — Credential & forge preflight · TODO · depends-on: none · wave 0
**Problem:** The `gh` token has died mid-session before; overnight that would turn
tracker writes into silent no-ops or fabricated success.
**Do:** Verify `gh auth status`, `git remote -v`, and push capability. Define the
abort rule: on auth failure, all tracker writes stop, local branches continue,
and the condition is recorded for the morning report.
**Files:** none (log only).
**Done when:** `gh auth status` exits 0 and the log records the remote plus the
abort rule.

### TICKET-A3 — Operating contract handed to every agent · TODO · depends-on: none · wave 0
**Problem:** Repo-specific traps (ruff pin, CI-without-provider, PYTHONPATH,
no-merge) are the difference between usable and worthless overnight output.
**Do:** Write the single contract file every dispatched agent receives verbatim,
containing the seven hard facts above plus the no-merge / no-force-push rule.
**Files:** `docs/plans/2026-08-20_overnight-issue-sweep/CONTRACT.md`
**Done when:** the file exists and each of the seven facts appears in it.

### TICKET-A4 — Authoritative batch manifest · TODO · depends-on: A2 · wave 0
**Problem:** Batching from memory would triage issue numbers that don't exist or
miss ones that do.
**Do:** Enumerate every open issue live, and assign each to exactly one batch by
the cardinal rule (silent-loss first), recording the ranking reason per issue.
**Files:** `docs/plans/2026-08-20_overnight-issue-sweep/batches.json`
**Done when:** the file's issue numbers, deduplicated, exactly equal the live open
set from `gh issue list --limit 100 --state open --json number`.

## Stream B — Triage (wave 1, parallel, READ-ONLY)

Three independent agents on three different vendors per batch. Each verdict is
one of STILL-VALID / ALREADY-FIXED / MISREPORTED / DUPLICATE / NEEDS-MEASUREMENT,
each with quoted `file:line` evidence from current `main`.

### TICKET-B1..B4 — Triage batches 1..4 · TODO · depends-on: A1,A3,A4 · wave 1
**Problem:** Much of the backlog is stale after this week's merges (#243, #246,
#247, #250), and some reports are wrong rather than unfixed.
**Do:** For each issue in the batch, re-derive the claim against current `main`
source and state whether the premise still holds, with evidence. No repo writes.
**Files:** `…/triage/<batch>/<vendor>.json` (one file per agent — disjoint).
**Done when:** every issue in the batch has a verdict from all three vendors, and
every verdict carries at least one `path:line` citation.

## Stream C — Adjudication (wave 2, one per batch, vendor ≠ any triager)

### TICKET-C1..C4 — Reconcile each batch · TODO · depends-on: matching B · wave 2
**Problem:** The doer must never grade its own work, and a lone model's confident
verdict is exactly the fabrication risk.
**Do:** Reconcile the three verdict sets. Unanimous → actionable. Split → escalate
to a fourth model and record the disagreement verbatim. Verdicts whose evidence
does not check out are demoted to NEEDS-MEASUREMENT, never actioned.
**Files:** `…/verdicts/<batch>.json`
**Done when:** every issue carries a final verdict plus an `evidence_verified`
boolean, and any 2–1 split is recorded with both readings.

## Stream D — Tracker actions (wave 3, gated on adjudicated verdicts)

### TICKET-D1 — Close ALREADY-FIXED with evidence · TODO · depends-on: C · wave 3
**Done when:** each closed issue carries a comment naming the commit and the
`file:line` that fixed it; nothing closed on a 2–1 split.

### TICKET-D2 — Correct MISREPORTED in place · TODO · depends-on: C · wave 3
**Problem:** #249 shows the right move for a wrong report is correction, not
closure.
**Done when:** each such issue has a correction comment stating what the original
claimed, what is actually true, and the measurement; the issue stays open.

### TICKET-D3 — File genuinely new defects · TODO · depends-on: C · wave 3
**Done when:** each new issue states a reproduction and an observed-vs-expected,
and links the triage evidence.

## Stream E — Fixes as proposed PRs (wave 4)

Gated on STILL-VALID. One branch + one PR per ticket, each in its own worktree
with the PYTHONPATH discipline. Implementer and reviewer are different models.

**File-collision lanes (same-wave tickets must not share a file):**
- **Lane O — `pipeline/orchestrator.py`: strictly serial.** Candidates: the
  TR-3-on-native-lane fix (#144/#205), the cascade-halt trio (#221+#222+#227
  as ONE change — #227 warns that fixing the probe alone makes it worse), the
  chart/table arbitration fix (#249/#248/#189).
- **Lane T — `tables/*.py`:** GH-144 trio (#195/#197/#198) — small, independent.
- **Lane R — resume/ledger:** #161 accept-gate identity.
- **Lane F — figures/fabrication:** #225 URL-provenance gate.

### TICKET-E<n> — one per accepted fix · TODO · depends-on: C, lane predecessor · wave 4
**Done when:** branch pushed; PR open; `gh pr checks` all pass; the diff has a
regression test that FAILS with the fix reverted (non-vacuity stated in the PR);
`uvx ruff@0.16.0 format --check .` clean; NOT merged.

## Stream F — Morning report (wave 5)

### TICKET-F1 — Single morning artifact · TODO · depends-on: D,E · wave 5
**Done when:** one document lists every issue closed (with evidence link), every
issue corrected, every issue filed, every PR opened with its CI state, and every
item deliberately skipped with the reason — plus anything that hit the abort rule.

---

## Open questions for the panel

1. Is wave 1's 3-vendors-per-batch the right cost/benefit, or is 2 + escalation
   enough given the evidence-citation requirement already catches fabrication?
2. Is Lane O's strict serialization sufficient, or does `orchestrator.py` need a
   single owning agent for the whole night?
3. Should NEEDS-MEASUREMENT issues be attempted overnight at all (they need the
   copyrighted corpus and hand judgement), or deferred to the morning by design?
4. What is missing that would make the morning output untrustworthy?
