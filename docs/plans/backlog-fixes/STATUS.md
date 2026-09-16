# STATUS — backlog fixes

> **Current truth, 2026-09-16.** GH-249 is DONE and merged (`5478b42`, PR #756).
> GH-140 is DONE (REVISE round applied), committed on `fix/140-math-font-audit`,
> awaiting review/CI/merge.
> The full ranked queue of remaining work lives at
> `~/.local/state/socr-housekeeping/QUEUE.md` (51 ready, 26 needing scoping, 8 to
> locate, 1 closure candidate).

## Live
(none — GH-140 moved to Done below)

## Done
- **GH-140** — implemented on `fix/140-math-font-audit`, REVISE round
  applied, not yet merged. Full suite 5405 passed / 4 xfailed, `ruff format
  --check` clean. See `docs/log/2026-09-16_140.md` for the criterion-4
  argument and a correction to the ticket's own stated context (the P4-R
  equation lane is already default-on; also had to narrow `has_equations` to
  a new `has_math_font_typesetting` field after the full suite caught a
  #269-shaped regression the ticket's literal framing would have
  reintroduced). REVISE: criterion 4 reversed after review measured the
  trigger rate against `docs/log/2026-09-02_p4m-trigger-rates.md` — the
  demotion's blast radius is ~15x the PUA precedent it copied and would make
  an unclearable 8.0% slice a permanent `AUDIT_FAILED`. The event/note/CLI
  surfacing stays; the document-status demotion was removed. Also added a
  resume round-trip test and a docstring caveat on `regions_covered`
  ("not invented" vs "verified correct").
- **GH-249** — merged in PR #756. Cost: 3 implementer passes, 1 adversarial review,
  1 CI catch. See `docs/log/2026-09-15_249.md`.

## Standing rules learned from GH-249
- Run the FULL suite, never a `-k` subset — that filter let a regression reach CI.
- Verify every reported number; two agents independently quoted a test count that
  was impossible.
- Abstention is not neutral in this pipeline: downstream it reads as consent.

## Next action
Await review (`socr-reviewer`) and CI-gated merge for GH-140.
