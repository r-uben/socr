# RUNBOOK — you are the night orchestrator

Owner authorised this run on 2026-08-20 and is now asleep. **Nobody will answer a
question until morning.** If you are blocked, record the blocker and route around
it; never guess, never invent a result, never wait on a human.

## Read these first, in order

1. `CONTRACT.md` — the seven hard repo facts. Hand this file **verbatim** to
   every agent you dispatch. It is not optional context; the editable-install
   trap alone will void every test result you collect if an agent misses it.
2. `TICKETS.md` — the ticket graph you are executing. This is the spec.
3. `STATUS.md` — the board. Keep it current as you go.
4. `logs/2026-08-20_panel-synthesis.md` — why the plan is shaped this way. Read
   it before you decide to deviate from anything; three critics already found
   the obvious shortcuts and explained what they break.

## What you are producing by morning

- Stale issues closed **with evidence**, but only through the review board.
- Misreported issues corrected in place and left open (#249 is the template).
- Genuinely new defects filed with a reproduction.
- Real fixes waiting as **proposed pull requests with CI green**.
- One morning report (TICKET-F1) the owner can act on in twenty minutes.

Zero closes, zero PRs, or an aborted night are all legitimate outcomes. Report
them plainly. A night that honestly reports doing little is worth far more than
one that dresses up work it did not do.

## You may

- **Open as many Herdr panes as you want** — explicit owner authorisation,
  2026-08-20. This overrides the standing "never spawn panes on your own
  initiative" rule for this run only.
- Use `/workflows` freely, and go beyond it wherever panes or subagents fit
  better.
- Spawn agents on any vendor. You are running under `claudex`, so GPT, Grok,
  Gemini and Kimi subagents all work. Use different vendors for triage,
  adjudication and review — the separation is the point, not decoration.

## You must not

- **Merge anything.** No `gh pr merge`, ever. PRs are proposed.
- **Force-push anything.**
- **Touch `/Users/rubenffuertes/repos/tools/socr` as a working tree.** Another
  session owns that checkout. Read-only git commands against it are fine; use
  your own worktrees for all work, and prove isolation with
  `bin/isolation_canary.sh` in every one of them.
- **Act on a `HELD-FOR-OWNER` action.** If the review board did not approve it,
  it waits for the morning. That is the whole point of the gate.
- **Close panes or agents you did not create.**
- Continue tracker writes once `state/ABORT` exists.

## The rule that matters most

`tracker_mode` is `agent-gated`. Nothing mutates the tracker on a triager's word.
Every staged action faces two reviewers on different vendors, neither of which
triaged or adjudicated that issue, each prompted to **refute** it rather than
confirm it. A reviewer that cannot open the code it is citing must reject. Both
approve, or the action is held.

This exists because of two things that actually happened here:

- A headless agent denied a tool permission produced confident invented output
  and exited 0.
- Issue #249 needed three owner revisions before its diagnosis held, and PR
  #250's own fix reintroduced the very bug it was meant to fix — with a test in
  that PR asserting the defective behaviour and passing.

Assume your first reading is wrong until something mechanical agrees with it.
Evidence is checked by `bin/verify_citations.py`, never by a model's say-so.

## Order of work

Waves 0 → 5 as laid out in `TICKETS.md`. Run the coordinator step (Stream W)
between waves: validate counts against `batches.json`, write
`state/checkpoint-<wave>.json`, mark failed inputs `SKIPPED` explicitly, and
dispatch only successors whose dependencies are `DONE` **or** `SKIPPED`. A
skipped predecessor must never freeze a lane.

Stream E is **one code owner, stacked branches** — not parallel. The panel traced
why: the candidate fixes all write the same two or three functions, and parallel
branches would collide inside a 900-line function and could silently undo each
other. That is exactly how PR #250 broke.

## When you finish

Write TICKET-F1's report, update `STATUS.md`, and stop. Leave every pane you
opened running — the owner decides what to close. Do not start a second pass on
your own initiative.
