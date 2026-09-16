# STATUS — backlog fixes

> **Current truth, 2026-09-16.** GH-249 MERGED (`5478b42`, PR #756). GH-140 MERGED
> (`c61fd58`, PR #757) as an **interim observability patch** — issue #140 deliberately
> left OPEN, the demote-or-not question is deferred. GH-658a/b implemented and
> reviewed (ACCEPT), rebased onto current main, awaiting CI-gated merge.
> The full ranked queue of remaining work lives at
> `~/.local/state/socr-housekeeping/QUEUE.md` (51 ready, 26 needing scoping, 8 to
> locate, 1 closure candidate).

## Live
- **GH-64** — implemented on `fix/64-tabular-native-flag` (off `ba92c19`). Restored the
  pre-PP-6 `_detect_columnar_numbers` heuristic in `born_digital.py` as a private,
  audit-only predicate (never wired into routing), computing
  `possible_table_structure_not_reconstructed` whenever a page falls to native
  (`not has_tables`) but still has the pre-PP-6 borderless label|value shape.
  Write-ownership expanded (requested, verified against #136/#217/#140, granted) to
  `state.py` (field + propagation) and `orchestrator.py` (`_agentic_native_page`, one
  `AuditEvent` append) — the same seam GH-140 used the same night — since
  `PageAssessment.notes` alone reaches nothing the pipeline reads. Report-only, no status
  demotion (ticket's hard scope limit 2; no trigger-rate measured for this signal yet).
  Full suite 5444 passed / 4 xfailed (5431 main baseline + 13 new tests, exact);
  `ruff format --check` clean; mutation-tested (neutering the predicate fails exactly the
  4 tests that require it to fire, 9 others correctly unaffected, including both
  `_agentic_native_page` orchestrator-seam tests). Inherits the pre-PP-6 heuristic's known
  chart-axis false-positive class by construction (ticket forbids a new threshold to
  narrow it) — documented explicitly, not silently absorbed into criterion 2. See
  `docs/log/2026-09-16_64.md`. Not yet merged.
- **GH-221** — implemented on `fix/221-wedge-canary` (off `3e04f1c`). Replaced the
  `/api/tags`-only liveness probe with a functional generation canary (minimal
  `num_predict: 1` / `max_tokens: 1` request, run only after the existing precondition
  passes) so a wedged GPU with a healthy HTTP layer now reads as "not idle" and the
  cascade-halt guard actually arms. Timeout derived from the existing
  `_CROP_DEADLINE_FLOOR_S` constant, not a new number. Review round 2: the canary now
  carries an image (`images`/`image_url`) because the workload it guards
  (`TableCropExtractor`) is a vision call, not a text one — a text-only probe exercises
  a different code path than the one that wedges. Full suite 5431 passed / 4 xfailed;
  `ruff format --check` clean; mutation-tested (neutering the canary fails exactly the
  7 tests that assert its behaviour, 45 others correctly unaffected). See
  `docs/log/2026-09-16_221.md`. Not yet merged.
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
