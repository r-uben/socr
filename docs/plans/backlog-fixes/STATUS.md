# STATUS — backlog fixes

> **Current truth, 2026-09-16.** GH-249 MERGED (`5478b42`, PR #756). GH-140 MERGED
> (`c61fd58`, PR #757) as an **interim observability patch** — issue #140 deliberately
> left OPEN, the demote-or-not question is deferred. GH-658a/b implemented and
> reviewed (ACCEPT), rebased onto current main, awaiting CI-gated merge.
> The full ranked queue of remaining work lives at
> `~/.local/state/socr-housekeeping/QUEUE.md` (51 ready, 26 needing scoping, 8 to
> locate, 1 closure candidate).

## Live
- **GH-658a/658b** — implemented on `fix/658-scanned-witness`, reviewer verdict ACCEPT
  on the code. 658a: `scanned` extra + once-per-run witness warning. 658b:
  distrusted-layer row-corroboration rescue, ships **flagged** via the existing #659
  channel. Reviewer traced the flag to all four surfacing levels and could not construct
  a path to an unflagged SUCCESS; the no-text-layer case still fails closed. Full suite
  5399 passed / 4 xfailed; `ruff format --check` clean. Rebased onto `c61fd58` — the
  branch's own docs commit predated GH-140's merge and would otherwise have clobbered
  the #757 record. See `docs/log/2026-09-16_658.md`.

## Done
- **GH-140** — merged in PR #757 (`c61fd58`). Interim observability patch: the math-font
  lane now emits `MATH_FONT_UNRECOVERED_KIND`, persists it in the page sidecar,
  reconciles across resume, and prints a CLI line. **Issue left open** — it does not
  demote document status. The 36.1%-vs-2.4% prevalence comparison first cited as grounds
  was itself corrected (design panel, Astra): prevalence alone cannot justify suppressing
  a signal, and the trigger-rates ruling explicitly accepts 36% — for routing. The
  rejection survives on that ruling's actual stated condition: it accepted 36% because
  over-routing is a cost, not a correctness risk; status demotion changes what every
  consumer sees and flips the GH-177 exit code, so the acceptance does not transfer.
  Known gaps, documented in the PR: the document exits 0, and `metadata.json` does not
  name it (the note rides in `audit_notes`, which has no document-level consumer) — the
  durable machine-readable record is the per-page sidecar event.
  Named follow-up (deferred, separate ticket): demote only when a display-equation region
  was **found** AND recovery **failed** to cover it — a structural gate, no invented
  threshold, and the 8.0% no-region slice then reads as "never confirmed" rather than
  permanently stuck. See `docs/log/2026-09-16_140.md`.
- **GH-249** — merged in PR #756. Cost: 3 implementer passes, 1 adversarial review,
  1 CI catch. See `docs/log/2026-09-15_249.md`.

## Standing rules learned so far
- Run the FULL suite, never a `-k` subset — that filter let a regression reach CI on GH-249.
- Verify every reported number; two agents independently quoted a test count that was
  arithmetically impossible.
- Abstention is not neutral in this pipeline: downstream it reads as consent.
- Mutate, do not delete, to prove a guard is load-bearing. A test that fails only with an
  ImportError proves a symbol is new, not that behaviour changed.
- Never write a closing keyword (`Closes #NNN`) inside a sentence disclaiming it — the
  GitHub parser does not read English, and #140 auto-closed on exactly that.
- Rebase a long-lived branch before merge: this branch's docs commit predated GH-140's
  landing and would have silently reverted the #757 record.
- One full pytest suite at a time on this machine — two concurrent runs caused a memory
  kill that stranded an agent for hours.

## Next action
CI-gated merge for GH-658a/b, then pull the next band-A item from QUEUE.md.
