# STATUS — GH-152 side-by-side tables merged

Last updated: 2026-09-16

## Stage
A1 and A2 DONE (both merging rungs fixed and tested). B1 cannot be satisfied under this
plan's corpus-content constraint (see ticket note) — recorded as not-satisfiable rather
than left open as an apparent oversight.

## Base state
Both of this plan's original blockers (PR #149, GH-144 A2) landed on `main` before this
work started; `main@3f0e553` already carried them, so neither gated dispatch. See
`docs/log/2026-09-16_152.md` for the full account.

## Ticket board

| Ticket | Stream | Status | depends-on | Notes |
|--------|--------|--------|------------|-------|
| A1 | column segmentation (detector) | DONE | none | landed together with A2, same commit |
| A2 | column segmentation (integration, both rungs) | DONE | A1 | `rowize_from_word_list` + `reconstruct_table_regions`, both band-clipped |
| B1 | evidence on the real motivating page | NOT SATISFIABLE | A2 | corpus-content instruction for this dispatch forbids opening `2025__haim`; needs a corpus-machine run the owner authorises separately |

## Known remainders (not defects this plan closes; decided 2026-09-16 by team-lead — ship now, option b)

**GH-418** (a word beyond every lane's snap radius is silently dropped) is a separate, already
open, already-twice-rejected-fix issue in the same file (`_rowize_segment`), `related: GH-418`
from TICKET-A2. A2's band split avoids it whenever the gutter is correctly detected (each band
gets its own label zone and lane set), but a page where gutter detection itself fails to fire —
measured to happen, not hypothetical, hit twice while building this ticket's own fixtures — is
still exposed to #418's drop. **State this precisely: GH-152 closes the merge whenever
`_detect_column_gutter` fires; it does not make side-by-side tables safe unconditionally.**

**Duplicate content** — A2's narrower per-band regions can leave a text block that spans both
bands unsuppressed by `born_digital.py`'s per-region coverage check, so its lines survive as a
duplicate plain-text copy beneath the two correctly-split tables. This is introduced by A2, not
inherited (confirmed absent pre-fix on the identical fixture). Nothing surfaces it — no status,
note, or count — so it is silent-but-correct, not legible-by-construction. Accepted because
under this repo's cardinal rule duplication is a strictly better failure than the misattribution
this ticket exists to close (every value stays correct and correctly labelled). Team-lead is
filing this as its own tracked issue.

Both — see `docs/log/2026-09-16_152.md` for the full trace.

## Next action
None outstanding for this plan. #418 is tracked separately; the duplication finding will get
its own issue filed by team-lead.
