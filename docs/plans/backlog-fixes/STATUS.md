# STATUS — backlog fixes

> **Current truth, 2026-09-16.** GH-249 is DONE and merged (`5478b42`, PR #756).
> GH-140 is READY and dispatched. The full ranked queue of remaining work lives at
> `~/.local/state/socr-housekeeping/QUEUE.md` (51 ready, 26 needing scoping, 8 to
> locate, 1 closure candidate).

## Live
- **GH-140** — READY, dispatched. Branch `fix/140-math-font-audit` off `5478b42`.

## Done
- **GH-249** — merged in PR #756. Cost: 3 implementer passes, 1 adversarial review,
  1 CI catch. See `docs/log/2026-09-15_249.md`.

## Standing rules learned from GH-249
- Run the FULL suite, never a `-k` subset — that filter let a regression reach CI.
- Verify every reported number; two agents independently quoted a test count that
  was impossible.
- Abstention is not neutral in this pipeline: downstream it reads as consent.

## Next action
Await the implementer's report on GH-140, then `socr-reviewer`, then CI-gated merge.
