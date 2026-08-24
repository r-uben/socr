# TICKETS — orchestrator decomposition

Status keys: `READY`, `NEEDS-DESIGN`, `BLOCKED`, `WIP`, `DONE`.

**Dispatchable today: R1 and R11.** D1 has landed, so the refactor tickets below now have
defensible write sets. D2 remains blocked on nothing but scheduling.

## D1 — Propose the seams in the two large functions

Status: `DONE` (2026-08-23) — `docs/log/2026-08-23_orchestrator-seams.md`, PR #284.

Findings that changed the plan: `_phase_agentic` is five mutually exclusive lanes plus a
shared tail, not a linear pipeline; `_phase_assemble` is an eleven-bucket page-disposition
taxonomy written out at three sites. `CLAUDE.md`'s description of the agentic loop was
wrong on two counts (no figure step in the loop; `route` is one lane of five, not a
universal first step). #162 was confirmed to live in `pipeline/agentic.py`, outside every
proposed seam.

Original ticket text follows, unchanged.

Status when written: `READY`
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

Status: `READY` (D1 landed 2026-08-23)
Depends on: D1

Once the seams are known, group the 51 remaining open issues by which post-decomposition
component they touch, and identify which sets become parallelizable. Today they nearly all
serialize on `orchestrator.py`; the point of D1 is to break that. This ticket produces the
ordering that was originally asked for.

## Refactor tickets — R1..R11

Status: `READY` for R1; the rest gated as noted.
Source of truth for every line range: `docs/log/2026-08-23_orchestrator-seams.md` (D1).

**Two passes, decided 2026-08-23.** Pass one gives every piece a name and leaves it in
`orchestrator.py`. Pass two moves out only the pieces that move cleanly. The file will end
up roughly 470 lines smaller, not 2,000 — see R11.

Rationale for the order: you cannot safely move a 1,101-line function into a new module,
because the move and the split happen in one commit and a broken test cannot be attributed
to either. A 107-line one moves trivially. Pass one manufactures the units that pass two
can move.

Every R ticket carries the sequencing rule from `STATUS.md`: **behaviour-identical, no bug
fixes riding along.**

### Pass one — name the pieces in place (methods on `UnifiedPipeline`)

| # | Ticket | Lines | Doc-scoped inputs | Status |
|---|---|---|---|---|
| R1 | Extract the trusted-native lane | 3286–3392 (107) | **none** | `READY` |
| R2 | Extract the no-provider lane | 3268–3285 (18) | none | `BLOCKED` on R1 |
| R3 | Extract the chart-asset lane | 3154–3267 (114) | `_chart_figures_dir` | `BLOCKED` on R1 |
| R4 | Extract the corrupt-equation lane | 3065–3153 (89) | `_chart_figures_dir`, `chart_winner_pages` | `BLOCKED` on R1 |
| R5 | Extract the OCR route lane | 3393–3628 (236) | 5 in, 2 out | **`HELD`** — see below |
| R6 | Extract the shared per-page tail | 3629–3762 (134) | 8, one read-write | `BLOCKED` on R2–R4 |
| R7 | Extract the eleven bucket derivations | 5999–6300 (~300) | `state`, `page_texts`, `native_only` | **`NEEDS-DESIGN`** — fork open |
| R8 | Extract bucket → audit-event emission | 6211–6482 (272, part) | R7's return value | `BLOCKED` on R7 |
| R9 | Extract bucket → CLI summary | 6211–6482 (272, part) | R7's return value | `BLOCKED` on R7 |

**R1 is the first slice and the only ticket dispatchable today.** It is the one lane that
reads zero doc-scoped locals; signature `(self, state, page_num, ps) -> None`; effects are
mutations on objects already passed in. It clears all four constraints by inspection: it
never calls `route_page` (so CI's empty ladder cannot distinguish before from after), and
it touches no fingerprint, no ledger gate, no halt latch and no fragment writer.

**R5 is HELD, not blocked.** It is the only lane that writes loop control
(`backend_degraded`, `halt_reason`), so extracting it changes a control-flow shape rather
than only a location — and cascade-halt has two open bugs against it (#227 fires when it
should not, #221 cannot fire at all). Moving that code while its semantics are disputed is
the wrong order. Revisit after #227/#221 are settled.

**R7 is the valuable one.** Adding a page disposition today costs three edits 300 lines
apart plus an exclusion clause in every sibling bucket; three recorded bugs in that block's
own comments (GH-151 B1 r2, BLOCKING 2 on #269, #262) are all the same bug — a page counted
under two dispositions, or none.

**CORRECTION (2026-08-24, design panel).** This ticket previously said R7 "makes exclusivity
a property of one function", implying the twelve buckets partition the pages. **They do not.**
A three-model read (GPT / Fable / Kimi, all grounded) found the same counter-evidence
independently, and it was then verified directly:

- The **ship** buckets are exclusive, and already structurally so — `_select_page_output`
  (`manifest.py:763-1208`) is a 15-return, **zero-loop** cascade, so exactly one branch
  ships per page. Not a convention; a property of the code's shape.
- `value_drift`, `fabricated_ref` and `text_grid_rejected` are **not** ship buckets. They
  are orthogonal alerts derived from events/flags that co-occur with a page shipping fine.
- `d3_floor_pages` is a **deliberate strict subset** of `failed_pages`
  (`orchestrator.py:6004-6006`), double-surfaced on purpose: two events, two CLI lines, and
  the lost-content note.

So the shape is **one ship disposition per page + a set of alerts**, not a single partition.
An implementer who collapses that to one enum makes a D3-floor-only document report SUCCESS
while shipping a failure marker — the exact bug class this ticket exists to kill.

Full panel record, the open fork (re-derive vs tag the cascade), the convergent signature,
and the sidecar-event-order test gap: `docs/log/2026-08-24_r7-disposition-design-panel.md`.
**R7 is `NEEDS-DESIGN` until the owner rules on that fork.**

### Pass two — move out what moves cleanly

| # | Ticket | Status |
|---|---|---|
| R10 | Move R7/R8/R9 to `src/socr/pipeline/dispositions.py` | `BLOCKED` on R9 |
| R11 | Rescope issue #155 to the measured reality | `READY` (docs only) |

The five lanes stay methods permanently. They lean hard on `self.config`; moving them buys
an import and costs threading configuration through every call site. Only the disposition
group is a pure function of `DocumentState`, so only it becomes a module.

**R11 exists because #155 currently promises more than this plan delivers.** It says "split
the god-module (~5.5k LOC)". The file is 7,520 lines and this plan removes ~470 of them.
The issue should be rewritten to match what the measurement showed: the complexity is
inside functions, not between modules.

### Acceptance, every R ticket

- The assembled `.md` is byte-identical before and after. The existing golden/byte-identity
  fixtures are the gate; do not write new ones that pin a measured value.
- **Assert a difference of zero, never an absolute outcome.** Per `CLAUDE.md`: parametrise
  over both provider states; a tuple measured on a Mac is not a fact about CI.
- One commit per ticket. `uvx ruff@0.16.0 format --check .` clean.
- CI green on the exact head SHA before merge — confirm a run exists, do not merely check
  that nothing is red.

---

## How the multi-model fleet works on this

Roster: **Fable, GPT, Kimi, Composer**, with Opus adjudicating. Three lanes.

### The constraint that shapes all of it

`socr` is installed editable, so `import socr` resolves to **this** checkout's `src/socr`
regardless of which worktree a process runs in. Giving each model its own worktree does not
isolate them — all four would run their tests against the same source, and four independent
results would be quietly meaningless.

`CLAUDE.md` states this directly: *"Do code work in the main checkout, one branch at a time."*

The competing-implementations pattern (`agent-fanout`) is therefore **unavailable for this
work.** That is a property of the repo, not a preference.

### Lane 1 — design, PARALLEL, read-only

For each ticket, all four models independently propose the exact write set, the method
signature, and the argument list. They read; they do not edit. Opus reconciles into one
ratified spec before any implementation starts.

High value on R6 and R7, where the argument list is the whole design question. Near-zero
value on R1 and R2, where the extraction is mechanical — do not spend a panel on those.

### Lane 2 — implementation, STRICTLY SERIAL

One model, one ticket, one branch, in the main checkout. The next ticket does not start
until the previous is merged. This is the bottleneck and it cannot be parallelised.

### Lane 3 — review, PARALLEL, read-only

The three models that did **not** implement review the diff. Read-only, so parallel is safe.
The doer never grades its own work.

Specific things a reviewer must check, drawn from this repo's recorded failures:

- Does any new assertion pin an absolute outcome measured locally? (This reverted #253.)
- Does a negative assertion run against a `MagicMock`, making it vacuously true?
- Does the diff touch `_run_fingerprint` inputs, silently invalidating resume?
- Does a test drive agentic mode without patching `_available_engines_for_agentic`?

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
