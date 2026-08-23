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

---

# W — work available now (does not touch `orchestrator.py`)

These 21 issues do not cite `orchestrator.py` in their triage evidence, so they are
independent of D1 and can be picked up immediately. The other 30 wait for D2.

Ordering principle, taken from `CLAUDE.md`: **a wrong number is worse than a dropped one,
and a dropped one is worse than a missing feature.** Anything that ships incorrect content
under a clean status outranks everything else.

## W1 — the accepting-gate tier (do these first)

These decide whether every other table fix can be trusted, because they are the checks
themselves.

1. **#162 — table verifier exceptions fail open into the accepting inner judge.**
   Do this first. While a thrown verifier silently means "accept", every other gate in this
   list is unreliable and any fix you land here cannot be shown to work. It is a bounded
   bug, not a design problem.
2. **#215 + #245 — header attribution.** Treat as ONE piece of work, not two tickets: the
   reject term is parked (#215) and `EXACT_PASS` still ships at confidence 1.0 when the term
   abstains (#245). Same mechanism, opposite ends. Fixing one alone leaves the hole open.
3. **#190 — an all-empty but structurally valid table passes validation.** Shape-only
   checks cannot see content loss. Same family as the above: a gate that says yes when it
   should say no.

## W2 — wrong content shipped under a clean status

4. **#270 — the VLM fabricates coefficients into genuinely empty cells.** The single worst
   live defect by harm: invented numbers in a citation corpus, under `SUCCESS`.
   **NEEDS-DESIGN, not implementation.** Its whole evidence base is two judges agreeing on
   a page image, and this repo has a recorded case of two vendors agreeing and both missing
   a row-label shift. It needs a *mechanical* structural check specified before anyone
   writes code. Do not dispatch an implementer at it.
5. **#248 — a corrupt text layer makes prose pages look like borderless tables.**
6. **#213 — book indexes are routed to table reconstruction.**
7. **#152 — two side-by-side tables merged into one region and flattened.**
8. **#163 — any OCR text-layer word defers the scanned source-evidence gate.**
9. **#223 — heading loss is not native-lane-specific; the VLM lane ships 36 unrepresented
   headings.**

## W3 — measurement owed before any code

These three were rescoped during the 2026-08-23 triage. Both voters agreed the original
defect is fixed and disagreed on the residue. **A measurement settles them; a code change
does not.** Do not open a branch on these before re-measuring.

10. **#144** — its strict-xfail canary now passes. Needs a fresh corpus measurement to say
    what, if anything, remains.
11. **#146** — the two voters directly contradict each other on the header-band half. The
    D2 corpus measurement has never been re-run since the 53b0637 sweep. Run it first.
12. **#151** — the structural gate landed; what remains is `binding.py` existing as an
    unwired module with no runtime consumer (TICKET-A2R), plus TICKET-B2.

## W4 — user-visible correctness, lower harm

13. **#168** — `--config` / `--profile` values silently dropped or overwritten by CLI
    defaults.
14. **#127** — native path discards heading, emphasis, list and link structure. Scope was
    narrowed by its own night-sweep comment: the heading half is #223, this keeps
    lists/links/emphasis.
15. **#220** — side-by-side page-image ↔ extracted-markdown viewer. A tool, not a fix, but
    it is the thing that makes hand judgement cheap, and several items above are blocked on
    hand judgement.

## W5 — structural chores, no user-visible effect

16. **#175** — inverted package layering (tables↔benchmark, core→tables). Note: this is
    adjacent to the decomposition. Consider holding it until D1 reports, so the two do not
    fight over the same boundaries.
17. **#178** — ADR: stay Python, native kernels only after profiling.
18. **#156** — TODO.md / TICKETS.md drift. Partly discharged by this plan folder; re-read
    before working it.

## W6 — proposals, decide before scheduling

19. **#202** — measure Mistral OCR 4.1 before any routing change.
20. **#203** — consume Mistral OCR 4 block labels. Marked blocked in its own title.

These two are decisions, not tickets. Note the standing finding that models are not the
bottleneck — every measured loss so far has been socr-side detection and surfacing. Weigh
that before spending on a model bake-off.

## Not ordered here

**#154** — one voter says its cloud-egress defect does not reproduce on main and that it is
superseded by #159; the other says it is still valid. Unresolved. Read both ballots before
touching it.
