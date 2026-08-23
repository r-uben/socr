# TICKETS — orchestrator decomposition

Status keys: `READY`, `NEEDS-DESIGN`, `BLOCKED`, `WIP`, `DONE`.

**Nothing here is dispatchable except D1.** This is deliberate. The seams are unknown, so
any implementation ticket written today would be guessing at its own write set.

## D1 — Propose the seams in the two large functions

Status: `READY`
Suggested agent: `socr-designer` (read-only; writes only `docs/log/`)
Depends on: nothing
Write ownership: `docs/log/2026-08-DD_orchestrator-seams.md` only. **No source edits.**

### Problem

`_phase_agentic` (1,101 lines) and `_phase_assemble` (895 lines) hold a quarter of
`orchestrator.py`. We do not know where they divide. Until we do, we cannot say whether the
result is 4 modules or 15, nor which open issues stop colliding.

### What D1 must answer

1. **What are the phases inside `_phase_agentic`?** `CLAUDE.md` documents the loop as
   route → extract → tables → figures → equations → flush. Confirm or correct that against
   the code, and give the line ranges for each.
2. **What is `_phase_assemble` actually doing for 895 lines?** It stitches fragments and
   calls `_rewrite_all_fragments`, which is documented as the sole authoritative fragment
   writer producing byte-identical output. Identify what else lives there.
3. **What state do the candidate pieces share?** Name the objects that would have to cross
   any proposed boundary (`DocumentState`, `PageState`, the ledger, the halt latch). A seam
   that requires passing eight mutable objects is not a seam.
4. **Which of the 51 live issues does each proposed seam help?** Explicitly. A seam that
   makes no open issue easier is a cosmetic seam and should be called out as such.
5. **What is the smallest first move** that is behaviour-identical and independently
   valuable? Not the whole decomposition — the first slice.

### Constraints D1 must respect

- The final assembled `.md` is **byte-identical** to whole-doc assembly and there are
  golden/byte-identity tests guarding it. Any proposal that cannot preserve this is
  rejected on sight.
- Resume semantics depend on `_run_fingerprint` and the per-page ledger gate
  (`_load_terminal_page`). A seam that changes fingerprint inputs silently invalidates
  every already-terminal page on resume.
- The cascade-halt latch is checked at the top of the page loop. Moving it changes failure
  behaviour on a wedged GPU.
- CI has no ollama and no provider. Proposals must not assume provider-dependent behaviour
  is observable in CI.

### Done when

A design note exists in `docs/log/` that answers all five questions with file:line
evidence, names the smallest first slice, and states explicitly which open issues that
slice unblocks. The note frames the remaining fork(s) as a decision for the owner — it does
not pick unilaterally.

### Explicitly out of scope for D1

- Writing any source change.
- Fixing any open issue.
- Choosing the final module layout. D1 proposes seams; the layout is a later decision
  informed by them.

## D2 — Order the 51 live issues against the decomposition

Status: `BLOCKED` (on D1)
Depends on: D1

Once the seams are known, group the 51 remaining open issues by which post-decomposition
component they touch, and identify which sets become parallelizable. Today they nearly all
serialize on `orchestrator.py`; the point of D1 is to break that. This ticket produces the
ordering that was originally asked for.

## Refactor tickets

Status: `BLOCKED` (on D1)

Not written yet, by design. They cannot have a defensible write set until D1 names the
seams. Each will carry the sequencing rule from `STATUS.md`: **behaviour-identical, no bug
fixes riding along.**
