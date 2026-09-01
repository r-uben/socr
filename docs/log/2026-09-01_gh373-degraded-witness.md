# 2026-09-01 — GH-373: degraded-scope judgement for count-mismatch AMBIGUOUS

Follow-up to the first live GH-353 ladder run (Cochrane–Piazzesi 2002,
pages 10/12). When `prepare_table_witnesses` cannot map emitted blocks 1:1
to located boxes, the gate treated every table on the page as ¬S1 and
shipped `TABLE_UNVERIFIED` without a judge looking. Manual adjudication
with the same rung-1 judge, shown the full page plus "judge only the table
matching this markdown", resolved all three abstained tables: two false
alarms (PASS) and one real catch (p10-t1 FAIL/`HEADER_MANGLED`, spanning
"Standard Errors" header absent from the markdown).

Flag `--table-judge-ladder` stays default OFF. GH-359's seven rulings are
not reopened.

## Position

Count-mismatch AMBIGUOUS is judged against the full page, not the union of
located boxes, because the spanning header that motivated this ticket can
sit outside every located box and a union crop would keep hiding it. The
match-this-markdown instruction lives in `prompts/table_judge_scope_page.md`
and is spliced through a live `{{SCOPE_NOTE}}` slot — not a string in code,
and not a rewrite of the located prompt. Only count-mismatch AMBIGUOUS gets
this look; corroboration-contradicted AMBIGUOUS and MISSING stay ¬S1; a
high-confidence PASS on the page image may ACCEPTED because the judge voted
on real pixels, while a lone low-confidence PASS stays UNVERIFIED per
GH-359 ruling 1.

Rejected alternative: union-of-boxes crop plus prompt wording in Python.
The union is tighter and cheaper, but on a paired/spanning-header page it
may exclude the header band — which is exactly the defect the live run
caught. In-code wording is the GH-381 defect (a second copy of prompt text
that cannot take effect, or that drifts from the policy file).

The deciding trade-off: a larger, costlier image is the only crop that
cannot exclude the header band. Cost is bounded because this path fires
only on count mismatch (the exception, not the LOCATED common case) and
the ladder flag is default OFF. Wrong-table risk is the scope instruction
in the policy fragment, not a geometric guess about which box is whose.

## Fork 1 — the image is the full page

The issue said "the MERGED region crop (or the full page)". These are not
equivalent.

- Locator over-merge (2 blocks, 1 box): the single box *may* include a
  spanning header if a booktabs band captured the header rule. It also may
  not: a ruled-grid box is the interior of one table, and 6pt of
  `CROP_PADDING_PT` will not pull in a header sitting above the grid.
- Extra spurious box (2 blocks, 3 boxes, the GH-381 shape: two ruled grids
  plus a spanning booktabs band): the union of all three *would* include
  the header. That is one geometry, not the guarantee.
- One grid found, the other missed: the union is that one grid. The
  spanning header above both is outside it.

p10-t1's `HEADER_MANGLED` is a spanning header the locator did not pair
1:1. If that header were inside every located box, the page would not have
been AMBIGUOUS in the first place, *or* a LOCATED crop would already have
shown it. We cannot prove the union contains the header. The manual
adjudication that caught it used the full page. Full page is the image.

Cost: one extra full-page render per count-mismatch page, judged per
emitted block (rung-1, then possibly rung-2). AMBIGUOUS is the exception
path. The live run's noise was 9 UNVERIFIED on two pages, not a
corpus-wide cost explosion.

Wrong-table risk: the judge sees every table on the page. Fork 2's scope
instruction is the counter. The audit event records `witness_scope=page`
so a human can see the look was degraded. We do not invent a third
confidence channel ("degraded ⇒ structurally low") that would silently
rewrite GH-359 ruling 1's vocabulary.

## Fork 2 — scope instruction is a live placeholder, filled from a policy file

`prompts/table_judge.md` has one live slot, `{{EMITTED_MARKDOWN}}`. GH-381
deleted the dead `{{PRIOR_FINDINGS}}` slot because a second copy of prompt
wording lived in code where it could never take effect.

The new instruction is a second live slot, `{{SCOPE_NOTE}}`, filled from
`prompts/table_judge_scope_page.md` when the witness scope is `page`, and
from the empty string when the witness scope is `located`. No prompt
sentence lives in Python. `build_table_judge_prompt` is the only
substitution site. The rung callables still receive crop + markdown
(ruling 4); they keep calling `build_table_judge_prompt` with no extra
argument. The gate selects the fragment by setting
`table_judge_prompt_scope` around `run_table_ladder`, so a wording-only
edit to the fragment takes effect on the next call and is hashed into
`table_judge_prompt_digest` (resume invalidates).

Rejected: a second full prompt file (schema/codes would drift from
`table_judge.md`). Rejected: always-on wording in `table_judge.md` (that
rewrites the LOCATED prompt, which is the bake-off / live-run calibration
surface this ticket is not licensed to move). Rejected: a Python string
passed as a fourth rung argument (GH-381).

Relay (same day): the other panelist argued the scope rule is harmless on
a tight crop, so one file / always-on / no placeholder is simpler. Kept
the fragment. Their wording replaces "Judge ONLY the table region in the
crop" with "judge ONLY the table matching the emitted markdown." On a
LOCATED crop that includes a spanning header the markdown omitted, that
swap is a HEADER_MANGLED miss: the judge is told to grade the matching
subregion and ignore the header band as a neighbor. That is the defect
this ticket exists to catch, inverted onto the common path. The second
file is the minimum mechanism that adds the page instruction without
rewriting the located rule. Wording still lives only in the `.md` files.

## The two AMBIGUOUS causes

`_classify` is count-only. The builder below it produces AMBIGUOUS a
second way: pairing-corroboration contradiction on a count-matched page.

- **Count mismatch** gets the degraded look. The locator could not assign
  a box to a block; there is no 1:1 crop to show, but there *are* pixels
  of the page. Showing the page with "match this markdown" is not a
  guessed pairing.
- **Corroboration-contradicted** keeps abstaining. The module docstring is
  explicit: a demonstrated wrong pairing must never silently self-correct
  into another guess. Counts matched; boxes exist; the index pairing is
  known-wrong. Auto-swapping the crops would be that guess. Showing the
  full page would be asking the judge to pick a pairing we already proved
  we cannot trust. Both members stay `AMBIGUOUS`, `crop_path=None`,
  `witness_scope=none`.

## MISSING stays ¬S1

No box at all (borderless, locate failure, crop render failure) is not a
mapping problem, it is an absence of geometric evidence that a table
region exists. Full-page judgement of every borderless table is a
different, larger change and is not this ticket. Confirmed.

## Audit event

Every ladder terminal event from the gate carries `witness_scope`:

- `located` — 1:1 box crop (unchanged path)
- `page` — full-page degraded look (count-mismatch AMBIGUOUS)
- `none` — no image (MISSING, corroboration-contradicted, or page-render
  failure)

Assemble's completeness backfill does not invent a scope (it never saw a
witness); those events keep `rung_trail: []` only.

## Terminals on a degraded-scope look

The ladder transition table is unchanged. A degraded look is a real look
at real pixels, with an explicit match-this-markdown instruction.

| last look (page image)     | terminal                         |
| -------------------------- | -------------------------------- |
| PASS high                  | ACCEPTED                         |
| PASS low (lone)            | UNVERIFIED (GH-359 ruling 1)     |
| PASS low after a real PASS | ACCEPTED (quorum, ruling 1)      |
| FAIL at last rung          | REJECTED                         |
| ¬S1 at last rung           | UNVERIFIED                       |

Fail-closed-only (PASS on a page image can never ACCEPTED) is refused: it
would leave the two false-alarm PASSes as UNVERIFIED and fail the ticket's
noise-reduction goal. Treating degraded high PASS as structurally low is
also refused: that invents a confidence channel GH-359 did not pin and
would force CLI₂ on every count-mismatch PASS.

Mechanical `bind()` still requires a 1:1 box. Degraded witnesses have
`box=None`; the check is NEUTRAL (not a fake FAIL). A full-page bind
would mix two tables' native words.

## What this does not do

- Does not flip `--table-judge-ladder` on.
- Does not reopen GH-359, GH-326, GH-322, GH-189, GH-381.
- Does not auto-swap corroboration-contradicted pairs.
- Does not judge MISSING (borderless) tables.
- Does not put a magic threshold anywhere.
