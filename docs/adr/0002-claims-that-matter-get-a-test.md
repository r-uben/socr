# ADR 0002 — A claim that would cost us if it went stale gets a test, not a paragraph

Status: **Accepted** · 2026-08-25

## Decision

When a fact about this repo is load-bearing — something a future agent or a future you will
act on — encode it as a **test that fails when the fact stops being true**. Prose describing
it is commentary, not the record.

When commentary and a test disagree, **the test wins** and the commentary gets fixed.

## The evidence

On 2026-08-25, four recorded claims turned out to be false. Each had been true when written.
Nothing kept them true.

**1. "Worktrees test the main checkout, so implementation cannot fan out."**
In `CLAUDE.md` and `docs/plans/orchestrator-decomposition/`, stated as "a property of the
repo, not a preference". False under pytest: `pyproject.toml` sets `pythonpath = ["src"]`,
which puts the local tree ahead of the editable install's `.pth`. Someone had already fixed
the bug; the warning outlived it and serialised work that never needed serialising. There
were 13 live worktrees at the time — the guidance was being ignored in practice, which is
the loudest possible signal that it was wrong.
Cost: months of unnecessary serialisation.
Now pinned by `tests/test_worktree_source_canary.py`.

**2. Issue #155 was scoped against the wrong lever.**
It asked to split a "~5.5k LOC" file; the file was 7,593 lines, and the module split it
proposed removes ~470. The lines that actually die (1,153, plus two whole modules) die with
#174. The issue had been read as the architecture priority for months.
Cost: the backlog pointed at the wrong ticket.

**3. The #174 ruling omitted `--legacy-routing`.**
The ruling enumerated what dies and missed the flag that mattered most — because its author
(Claude) never checked whether the flag already existed. It did, at `cli.py:181`. The fork
was framed to the owner as "delete versus build a gate" when the gate was already built.
Found by the review panel, not the author.
Cost: caught before implementation, by review.

**4. The `/ocr` skill depends on flags we were about to delete.**
`~/repos/skills/ai-skills/ocr/SKILL.md` defines `/ocr multi` and `/ocr cloud` in terms of
`--multi-engine` and `--legacy-routing`. Nothing in socr recorded that dependency. A
different repo (`disputatio`) names `--multi-engine` too.
Cost: would have silently broken a named workflow in another repo.

## Why prose failed and a test would not have

All four were written in good faith and were accurate on the day. The failure mode is not
carelessness — it is that **prose has no failure state**. A stale paragraph reads exactly
like a true one. A stale test goes red.

This repo already has six places a fact can live: `CLAUDE.md`, `docs/plans/*/STATUS.md`,
`docs/log/`, `docs/ARCHITECTURE.md`, the Obsidian canvases, and GitHub issue bodies. Adding
a seventh does not help. Making one of them *checkable* does.

## What this looks like in practice

**Encode as a test:**
- Environment properties that other work depends on — "a worktree tests its own source"
- Architectural invariants — "tables and core must not import benchmark"
- Contracts a deletion would break — "help does not advertise the legacy routing entry"
- Cross-repo dependencies — if the `/ocr` skill needs a flag, a test should name that flag

**Leave as prose:**
- Why a decision was made (ADRs — this file included)
- What was measured on a given day, and by what method
- Ordering arguments and priorities

The test says *what must remain true*. The prose says *why we chose it*. Prose that asserts
a checkable fact without a test behind it is a liability with a shelf life.

## Rules that follow

1. **A number in prose carries its method.** "1,153 lines" is useless without "by call-graph
   reachability over `UnifiedPipeline`'s 88 methods". Failure #2 happened because a line
   count was quoted without the method that produced it, then reused for a year.
2. **A claim contradicted by observed behaviour is wrong, not ignored.** Thirteen worktrees
   existed while the docs said worktrees were impossible. Treat that gap as a bug report.
3. **A guard must cover the obvious way around itself.** The first layering test missed
   relative and dynamic imports — it would have stayed green through the exact regression it
   existed to prevent. Prove each evasion fails.
4. **Retract in place, loudly.** When a recorded claim is falsified, say so where it lived —
   do not quietly delete it. A reader who remembers the old claim needs to know it was
   wrong, not wonder if they imagined it.
5. **Cross-repo dependencies are recorded on the socr side.** socr cannot see the skills repo
   or `disputatio`. If they depend on a flag, socr's tests must name it.

## Consequences

- Deleting a documented flag or path requires first asking what tests name it — and what
  *outside* this repo does.
- Some existing prose should become tests. That is backlog, not a precondition.
- This ADR is itself prose. It explains a choice; it asserts no checkable fact. That is the
  distinction it exists to draw.

## Addendum, 2026-08-26 — a panel does not satisfy rule 3 on its own

GH-190 shipped a guard that could be evaded. Its cold review found the hole, and a second
multi-model run was launched specifically to close it. **That second run wrote an evasion
matrix covering the case its author had in mind — the same failure the first run made.**
Of four enumerated evasion routes, exactly one was tested.

The multi-agent process reproduced the human failure mode rather than correcting it.
Independent models do not automatically produce independent *imagination*: asked to "prove
the fix works", several will reach for the same obvious case. What broke the loop was an
external enumeration — naming the specific spellings (3-column borderless blanks, `" | |"`,
header-with-no-body, a GFM colon delimiter used as a body) — and then checking that a named
test existed for each.

Rule 3 therefore needs a sharper form:

> **Enumerate the evasions BEFORE the fix, in writing, and check each one off against a
> named test afterwards.** "Prove the guard covers its evasions" is not something a reviewer
> can verify. A list is.

This applies to our own work, not only to the models'. During R174b a restored guard in
`tests/test_dual_pass_tables.py` evaluated the production condition *inside the test body*,
so it asserted nothing — written by the same agent that had flagged that exact pattern in
someone else's diff an hour earlier. Reviewing for a failure mode does not immunise you
against it.
