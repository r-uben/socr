# STATUS — progressive page processing

Last updated: 2026-06-16

## Stage — ✅ COMPLETE

**The initiative is DONE. All 8 tickets are merged to `main`** and demonstrated live (a real run
flushes `pages/NNN.md` one at a time, writes the resume sidecars, and stitches a byte-identical
final `.md`; 1073 tests pass). socr's agentic mode now processes one page fully and saves it before
the next, with crash-recovery resume, a wedged-model halt, per-page tables, inline figures, a
chart→image-asset lane, and opt-in equation detection.

| Ticket | Issue | Merged commit | What it delivers |
|--------|-------|---------------|------------------|
| PP-0 | #55 | `e607066` | dual-pass crop-reread releasing deadline + cascade guard (the hang) |
| PP-6 | #54 | `dfbd317` | lane-cooccupancy routing gate (over-routing) + content-type vector |
| PP-1 | #65 | `065e5dd` | per-page fragment flush + atomic sidecar + byte-identical stitch |
| PP-3 | #67 | `e0fcb7f` | per-page `_reread_page_tables` helper (behavior-preserving) |
| PP-4 | #69 | `61ca221` | per-page figure extraction + inline embedding |
| PP-7 | #73 | `b6609e7` | chart→figure-asset lane (cluster-first vector detection + B1) |
| **PP-2** | **#71** | **`bca0ad3`** | **the fuse — page-major loop + per-page provisional flush + cascade-halt** |
| PP-5 | #76 | `9b393fa` | per-page resume ledger — skip terminal pages on re-run |

Each shipped as its own PR through implement → `socr-reviewer` → `/codex` merge gate. The `/codex`
gate caught ~10 real bugs the in-house reviewer passed (data-loss patches, half-saved states,
fragment-divergence holes, a native-page halt leak, fingerprint/invalidation gaps, CI-hermeticity).
Follow-up #64 (audit-flag for borderless 2-col tables that fall to native) is the one tracked residual.

### Historical planning record below

The sections that follow are the original plan (waves, dispatch, design forks). They are kept as the
design/decision record; the live status is the table above.

- **PP-0 (#55)** — crop-reread releasing deadline + timeout audit + wedged-GPU cascade guard + fitz-leak.
  Merged to `main` `e607066`. The `/codex` gate caught a real data-loss bug (a page with a crop timeout
  could still patch text) the in-house reviewer missed — fixed before merge.
- **PP-6 (#54)** — lane-cooccupancy routing gate + content-type vector on PageState. Merged `dfbd317`.
  Residual (rare 2-col borderless-table structure dropped to native) tracked as **follow-up #64** per
  `/codex` (Option B: audit-flag it; do NOT re-widen the gate).
- **PP-7** — chart→figure-asset lane. Design resolved + `/consilium`-ratified (A2 + B1). READY, Wave 2.
- **PP-1 (#65)** — fragment flush + atomic sidecar + byte-identical stitch. **WIP** (the sequential-write
  scaffold; PP-2 later makes the flush incremental).

The backlog (`TICKETS.md`) is 8 tickets, PP-0 … PP-7, in 5 waves.

The durable planning format mirrors `docs/plans/agentic-local-first/`:
- `TICKETS.md` — canonical backlog.
- `STATUS.md` (this file) — live execution, waves, assignments, next action.
- `README.md` — design rationale + consilium decisions + open questions.

This initiative is **disjoint from but downstream of** the top-level `docs/plans/` GH-issue board:
it owns #54/#55/#56 and the progressive-processing slice of #49/#47. The #49/#47/#36 feature work is
**now in `main`** (PRs #57/#61/#53/#59/#60 merged 2026-06-16) and this branch is rebased onto it, so
PP-2/3/4 can absorb that logic rather than re-derive it.

## The strategic bet (one line)

Make **only the agentic path** progressive (per-page lifecycle + `pages/NNN.md` flush + atomic
sidecar), leaving non-agentic/consensus/repair phase-major and bit-stable — because the agentic path
already owns the only per-page loop, write-through cache, and releasing deadline. Accepted cost: two
divergent code paths.

## Branch reconciliation — DONE (2026-06-16)

The five-way divergence off stale `main` (`11ebae9`) is **resolved**. All feature PRs were
rebase-merged to `main` in foundation-aware order, then this branch was rebased onto the result:

| PR | Carries | Landed on `main` |
|----|---------|------------------|
| #57 | GH-47B figure caption anti-fabrication | `c30f1e0` |
| #61 | GH-47C figure label audit (log-only) — re-cut from #58, which GitHub auto-closed when #57's branch was deleted | `125ab49` |
| #53 | #49 structured routing + GH-49A native-table verifier | `300c31b` / `fa3f8d2` / `7b1af30` |
| #59 | GH-36a equation region detect + crop + provenance | `1f86996` |
| #60 | GH-36b equation→LaTeX + pylatexenc 1A gate + 1C sidecar | `2a7728f` |

`feat/progressive-pages` is rebased onto `main@2a7728f` (docs-only → clean). The only code merge
conflict was the figure/equation seam in `orchestrator.py` (47C's `_record_figure_recoverable_labels`
vs 36a's `_detect_and_crop_equations`) — resolved by keeping **both** methods; 268 targeted tests green.

**Lesson (future stacked merges).** Do NOT `--delete-branch` a PR that is the base of another open PR
— GitHub auto-closes the child and it cannot be reopened once its base is gone (this happened to #58 →
re-cut as #61). Retarget the child to `main` *before* merging the parent, or re-cut it. For a
rebase-merged parent, drop the parent's now-duplicate commit from the child with
`git rebase --onto main <parent-old-tip> <child>` before merging the child.

**Still live — rebase reconciliation is a precondition to Wave 2/3.** The page-major restructure must
*absorb*, not duplicate, the now-merged logic:

- **GH-49A `NativeTableVerifierJudge`** (verifier-before-VLM-judge) is already per-page → confirm it
  sits at the routing gate inside the fused loop (**PP-2**); it is the table verify-layer PP-3 builds on.
- **GH-47B/47C figure caption + label logic** → folds into **PP-4**'s per-page figure step (inline
  embed), replacing the doc-tail figure phase.
- **GH-36 equation phases** (`_detect_and_crop_equations`, `_attach_equation_latex_sidecars`) → fold
  into a **per-page** splice at the figure seam, gated no-op in agentic mode. Owned by **PP-2 step 6a**;
  PP-4 shares the seam. The 1A pylatexenc gate + 1C non-destructive sidecar + crop-as-ground-truth is preserved.

## Waves & dependency graph

```
WAVE 0 (de-risk, parallel)     WAVE 1        WAVE 2 (per-page steps)      WAVE 3      WAVE 4
  PP-0  (#55, READY) ───────────────────────► PP-3 (tables) ──┐
  PP-6  (#54, READY) ──► PP-7 (chart lane) ──► PP-4 (figures)──┼──► PP-2 (fuse) ──► PP-5 (resume)
                          (NEEDS-DESIGN)        PP-7           ─┘     (NEEDS-DESIGN)  (NEEDS-DESIGN)
            PP-1 (flush primitive, READY) ─────────────────────────►
```

- **PP-2 depends on PP-0, PP-1, PP-3, PP-4** (it fuses them).
- **PP-3 depends on PP-0** (needs the deadlined reader).
- **PP-4 depends on PP-2** structurally but its per-page extractor refactor can be built in Wave 2;
  inline-embed wiring lands with/after PP-2. Blocked on README open question 1 before implementation.
- **PP-7 depends on PP-6** (the narrowed gate is what creates the chart gap PP-7 fills).
- **PP-5 depends on PP-1, PP-2.**

## Ready queue (no design gate)

| Ticket | Agent | GH | Notes |
|--------|-------|-----|-------|
| **PP-0** | `socr-implementer` | #55 | Crop-reread deadline + cascade guard + fitz-leak. Ship first — removes acute data-loss risk. |
| **PP-6** | `socr-implementer` | #54 | Lane-cooccupancy routing gate + content-type vector. Pairs with PP-7. |
| **PP-1** | `socr-implementer` | new | Fragment flush + atomic sidecar + stitch (byte-parity). Load-bearing primitive. |

## Design queue (NEEDS-DESIGN — `socr-designer` first, then orchestrator `/consilium` if a real fork)

| Ticket | Agent | Blocker / question to frame |
|--------|-------|------------------------------|
| **PP-7** | `socr-designer` first | Chart detection (vector + raster) + markdown representation of a chart-asset page. |
| **PP-3** | `socr-designer` first | Extracting the per-page reread helper; parity with global Phase 4c. |
| **PP-4** | `socr-designer` first | Inline vs doc-tail figure layout (README open question 1) before implementing. |
| **PP-2** | `socr-designer` first | The fused per-page driver + document-level cascade-halt policy. |
| **PP-5** | `socr-designer` first | Resume semantics against the finished lifecycle; fingerprint invalidation. |

## Active agents

| Ticket | Agent id/name | Started | Status | Owned files | Notes |
|--------|---------------|---------|--------|-------------|-------|
| — | — | — | none dispatched yet | — | Planning phase only. |

## Per-ticket workflow (orchestrator-driven, consilium-gated)

Same pipeline as the top-level board. The **orchestrator** (main Claude session) drives it; agents
do bounded work; `/consilium` is a main-thread tool only.

```
   READY ticket ────► socr-implementer ──► socr-reviewer ──► ACCEPT → next
                          ▲ (CONSILIUM-GATE on architectural fork)  │ (REVISE → re-dispatch)
   NEEDS-DESIGN ──► socr-designer ──► [orchestrator runs /consilium] ──► decision into ticket ──► implement
```

`/consilium` routing: design/architecture/trade-off forks → default panel (Codex + Gemini); skip for
mechanical tickets. PP-0/PP-1/PP-6 are mechanical-to-moderate and may skip the panel; PP-2/PP-7 are
the real design forks.

## Dispatch contract

Prompt each agent with exactly one ticket section from `TICKETS.md`, plus:
- You are not alone in the codebase; do not revert unrelated edits.
- Own only the files in the ticket's `Write ownership` (and the named **method region** for
  `orchestrator.py`) unless you first report why more scope is required.
- Use `uv run` (or `~/venvs/socr/bin/*`); never `python script.py`.
- Report changed files, tests run, failures, residual risks.
- Commit on the initiative branch (never `main`, never the figure-caption branch); stage by name;
  do not push; one commit per ticket.
- On an architectural fork you cannot resolve from the ticket: stop, return `CONSILIUM-GATE` + a
  one-sentence question.

## Agents

The existing repo agents cover this initiative — **no new agents needed** (adding redundant ones
would duplicate `.claude/agents/`):
- `socr-designer` — read-only design pass on the NEEDS-DESIGN tickets (PP-2/3/4/5/7).
- `socr-implementer` — one bounded code ticket.
- `socr-reviewer` — adversarial review before acceptance.

They previously hardcoded `docs/plans/TICKETS.md` and the stale `feat/001-issue-plans` branch.
As part of this initiative they were **parameterized**: each now takes the **plan folder** (default
`docs/plans/`; subfolder for a focused initiative) and the **initiative branch** as told-at-dispatch
inputs. So dispatch each with: the ticket id, `docs/plans/progressive-pages/` as the plan folder,
and the initiative branch.

## Next action

Reconciliation + rebase are **DONE** — the base is current `main@2a7728f` and Wave 2/3 is unblocked.

1. Open GitHub issues for the two `new` tickets (PP-1 fragment flush, PP-5 resume ledger), or fold
   them under #56 — orchestrator decision.
2. **Start Wave 0** (now fully unblocked): dispatch `socr-implementer` on **PP-0** (#55 crop-reread
   deadline + cascade guard) and **PP-6** (#54 routing) in parallel — confirm disjoint `orchestrator.py`
   method regions first; `socr-designer` on **PP-7** (chart lane, read-only). PP-0 is independent of
   everything and ships the acute #55 data-loss fix regardless.
3. Then **PP-1** (Wave 1, prove byte-parity), then Wave 2 (PP-3 tables / PP-4 figures, **absorbing**
   the now-merged GH-49A verifier + GH-47B/47C figure logic + GH-36 equation phases per the
   reconciliation section), then **PP-2** fuse (+ step 6a equation re-home), then **PP-5** resume.
4. Before pushing this branch / opening its PR: it is currently **local-only** in the worktree. Push +
   PR is the user's call.
