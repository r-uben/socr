```markdown
# STATUS — GH-155 orchestrator split

Last updated: 2026-08-18

## Stage

Design packet complete; owner review pending. **No implementation has started.**

## Immutable review baseline

`0c5bbd11007d1aa038f040242c006a64d1a4483c` (merge of #224). All file:line citations in
README.md and TICKETS.md are pinned to it. At review time the working tree was verified
identical to the baseline for `src/socr` and `tests/` and the baseline is an ancestor
of HEAD. **Before starting any slice, re-verify this** (commands in the packet's
verification appendix / below); if `src/socr/pipeline/orchestrator.py`, `agentic.py`,
or the listed test files have drifted, re-derive the affected line citations and seam
counts before implementing.

## Approved decisions (settled during the review; re-open only with owner sign-off)

1. Grouped tree under `modes/`, `lanes/`, `persistence/` with `orchestrator.py` as
   facade + composition root; single `DocumentState`; no event bus / DI / phase graph /
   second run-state object.
2. `modes/legacy/extraction.py` dropped from the approved tree
   (`_run_engine_on_pages` is cross-mode; stays a facade method). Deviation flagged.
3. `modes/agentic/judging.py` kept as its own module; `legacy/quality -> agentic/judging`
   is the single legal cross-mode edge (call-time qualified lookup).
4. No compatibility shim for `socr.pipeline.agentic`; tests repoint atomically
   (seam-liveness over import compatibility; zero non-test src consumers).
5. Deps-protocol pattern (structural Protocols satisfied by `UnifiedPipeline`,
   production passes `self`); facade delegates keep frozen signatures; module globals
   move with their last production reader plus same-commit test repoint plus canary.
6. `DEFAULT_PROVIDER_TIMEOUTS` moves to `socr/core/providers.py` (avoids the forbidden
   `lanes -> modes` edge).
7. Chart/D3 render primitives live in `lanes/figures.py` for now; moving them into a
   new `socr/figures/` domain module is parked (P6) pending explicit approval.
8. Fingerprint stays facade-computed per call; persistence takes it as a string value.
9. `pipeline/__init__.py` becomes a lazy PEP 562 export (issue AC: unit tests that do
   not import the whole orchestrator).
10. Boundary enforcement via a stdlib-`ast` test, not a new import-linter dependency.
11. Issue LOC acceptance: < 1.5k hard gate retained (estimate ~950); < 800 reported as
    aspirational; any AC change (D4) needs owner approval.
12. Fingerprint under-invalidation fixes (D1a-e), `pp2_halt_reason` declaration (D2, DONE — #234/PR #236),
    and the blob-key canonicalization defect (D5) are separate tickets, never folded
    into extraction slices.

## Outstanding TODOs

- [x] Owner sign-off on the two tree deviations (2026-08-18): `modes/legacy/extraction.py`
      dropped, and `consensus.py` / `repair.py` / `reconciler.py` / `hpc_pipeline.py`
      named explicitly as unchanged siblings. Decisions 4, 7, 11 stand as written.
- [x] Filed the D1a-e / D2 / D5 issues on GitHub (2026-08-18): #229-#235.
- [ ] U2 (per-slice, empirical): triage test_orchestrator's 49 `get_engine` patches
      stays-vs-repoints during T10/T12 using the run-unrepointed-then-prove-non-vacuous
      protocol.
- [ ] U3 (per-slice): census capsys-based console assertions that witness moved prints
      (beyond the 3 string patches).
- [ ] Schedule the T12/T14 freeze window (no other orchestrator-touching PRs).

## Next action

Owner reviews this packet; on approval, open the T0 branch (`feat/155-lazy-export`)
and implement T0 + T1.

## Implementation sequencing warning

Work only in the main checkout, one branch at a time: the editable install resolves
`import socr` to this repo's `src/socr`, so a separate git worktree would test the main
tree's source. Wait for CI green before merging every slice (CI has no ollama; agentic
tests must patch `_available_engines_for_agentic`). Every slice changes
`socr_source_digest`, so cached corpora reprocess on next run — expected GH-214
fail-safe behavior, not a slice failure; byte-identity checks exclude the
`run_fingerprint`/`socr_source_digest` sidecar fields. Land D1 fingerprint tickets
outside the slice sequence and coordinate with running corpus jobs.
```
