# socr — extraction defects, team session

You are orchestrating a team of subagents against five scaffolded plans in
`~/repos/tools/socr`. Do not re-derive the analysis: it is done, filed, and
written down. Your job is to dispatch, verify, and refuse bad work.

## Read first (in this order, do not skip)

1. `CLAUDE.md` — build/test/lint commands and the repo's traps
2. `docs/plans/extraction-defects/STATUS.md` — **the schedule.** Global waves and file
   ownership across all five plans. No folder below may be scheduled on its own.
3. `docs/plans/gh150-figures-as-tables/{TICKETS,STATUS}.md`
4. `docs/plans/gh151-structural-gate/{TICKETS,STATUS}.md`
5. `docs/plans/gh147-landscape-pages/{TICKETS,STATUS}.md`
6. `docs/plans/gh144-rowizer-destroys-values/{TICKETS,STATUS}.md`
7. `docs/plans/gh152-side-by-side-tables/{TICKETS,STATUS}.md`
8. `gh issue view 150 151 147 144 152` — each carries the measured evidence

Every ticket has an outsider-checkable **Done when**. That is the contract. A
ticket is done when the named command exits 0 or the named artifact exists —
never when an implementer says so.

## The team

| Seat | Agent | Job |
|---|---|---|
| Implementer | `gpt-sol` or `claude` (Opus) | one ticket, end to end: code + tests + ruff |
| Reviewer | **must differ in vendor from the implementer** | adversarial diff review before commit |
| Aux | `grok` | second lens when a reviewer and implementer disagree, or when a ticket's approach is genuinely uncertain |
| Verifier | `fable` | runs the **Done when** command itself and reports the raw output |

**Rotate the reviewer seat.** If `gpt-sol` implements, review with `fable` or
`claude`; if `claude` implements, review with `gpt-sol` or `grok`. A fixed
reviewer develops correlated blind spots with whatever it reviews most.

**Do not use `gemini`.** It failed three times in a row on 2026-08-11 (server
errors, two different models) and delivered nothing. Try it only if everything
else is unavailable.

## Protocol per ticket

1. Build the implementer prompt from **ticket fields only** — Problem / Do /
   Files / Done when. Do not paste conversation history or plan prose.
2. Implementer works on a branch off `main`: `fix/NN-<slug>`. Never on `main`,
   never on another session's branch.
3. Reviewer gets the diff and the ticket, and is asked to **find what is wrong**,
   not to approve. Require file:line evidence for every claim.
4. Verifier independently runs the **Done when** command and pastes raw output.
5. Only then: commit (stage by name, never `git add -A`), push, open a PR, wait
   for CI green, and ask the user before merging.

## Hard constraints — these have all bitten before

- **CI has no ollama and no provider.** Any test driving `_phase_agentic` or
  `process()` in agentic mode must patch `_available_engines_for_agentic`, and
  anything reaching `_resolve_crop_vlm_model` must stub it — otherwise the
  classification depends on whether the machine happens to have a model pulled,
  and the test passes locally and fails in CI.
- **Lint gate:** `uvx ruff@0.16.0 format --check .` — NOT the venv ruff, which is
  older and reports clean on files CI rejects.
- **Tests:** `~/venvs/socr/bin/pytest <paths> -q`. Full suite ~1500 tests.
- **Shared checkout.** Other sessions work in this same tree and switch branches
  under you. Check `git status` and the current branch before any write. If the
  branch is not yours, do not commit.
- **`reconstruct.py` may have in-flight work.** GH-152 A1 and GH-144 A2 both
  need it; both plans record this as a precondition. Verify it is merged before
  dispatching either.

## Traps that produced wrong work today — check for each of these

- **A test that passes against the unfixed code guards nothing.** Before
  accepting any regression test, revert the fix (`git stash push -- src/`) and
  confirm the test FAILS. Two tests written today passed both ways; one had a
  fixture with ruling lines that took a different code path entirely.
- **A passing suite is not evidence a classification is right.** A test can only
  fail if the classification disagrees with itself. One list written today was
  wrong in 10 of 42 entries and the suite was green throughout.
- **Character counts do not measure content loss.** Markdown scaffolding inflates
  output; four pages shipped MORE characters while losing 28–57% of their words.
  Use word-multiset comparison.
- **Word recall does not measure structural loss.** A page can score 100% recall
  with its table unusable (this is GH-151). Do not treat recall as sufficient.
- **Instrumentation cannot distinguish "dead" from "not exercised by this
  fixture."** If a measurement says something is unused, verify by reading the
  source before acting on it.

## Dispatch order

`docs/plans/extraction-defects/STATUS.md` is authoritative. Summary:

**Wave 0 — merge gate, not tickets.** PR #148 (`dominant_text_direction()`) and PR #149
(`reconstruct.py` header work). Both are implemented and open. Wave 1 does not need them;
waves 2+ do.

**Wave 1 — six in parallel, write sets disjoint:**

- **GH-150 A1** `figures/extractor.py`
- **GH-150 B1** `pipeline/orchestrator.py`
- **GH-151 A1** `tables/structure_check.py` (new)
- **GH-151 A2** `tables/native_verifier.py`
- **GH-147 A1** `core/born_digital.py` — after PR #148, else written twice
- **GH-144 A1** `logs/` only, read-only

Waves 2–5 are serialization, not parallelism: `reconstruct.py` (GH-144 A2 → GH-152 A1 →
GH-152 A2) and `born_digital.py`/`orchestrator.py` (GH-147 A2 → GH-151 B1) are each held by
one ticket at a time. **Never dispatch two tickets that name the same file**, whatever their
own folder's wave says.

## Priority if you can only do one

**GH-150.** The two lowest-recall pages in a 22,979-page corpus are figures the
table lane claimed. The fix is a precedence decision, not an algorithm: emitting
an image reference loses nothing recoverable, while emitting a pipe grid of axis
labels loses everything.

## Standing rules

- Never push or merge without explicit user approval, even on green CI.
- One commit per ticket. `Closes #NN` in the PR, not the commit, unless the
  ticket completes the whole issue.
- If a reviewer and implementer disagree substantively, bring in `grok` as the
  third lens and report the disagreement verbatim to the user rather than
  resolving it silently.
- Report what failed as plainly as what passed.
