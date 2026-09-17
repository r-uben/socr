# 2026-09-17 — GH-692: font equality alone must not merge a section heading

## The defect

`_wrapped_label_merge_plan`'s widened GH-624b branch (`binding.py`) merged a
label-only row onto its next row whenever (1) the merged text matched the
native row's full joined `row_path` and (2) both native rows carried equal,
unambiguous `label_font` signatures. Both conditions are satisfied by *every*
legitimate parent-heading-plus-child pair, not only by a genuinely wrapped
two-baseline label: `_native_rows` hands a data row whatever is on the
indent-prefix stack regardless of the data row's own indentation, so the
joined-`row_path` match proves nothing about which shape this is, and a
heading that happens to reuse its child's exact typeface satisfies font
agreement too. That is silent native-row loss under the live agentic path.

## Corpus measurement: not done

I do not have corpus access in this worktree. Per the ticket, I treated the
issue's step-1 ask ("do real headings disagree in font from their first
child on `fed-01`?") as **INCONCLUSIVE** and went straight to step 2 (add
the guard). I am not asserting any corpus fact about how often headings and
children share a typeface.

## Discriminator chosen: left-indent of the two rows' own label bboxes

A wrapped label's second printed baseline is the same stub cell continuing
onto a second line — it starts at the same or a shallower left edge as the
first line (a hanging wrap indents further right only via markdown parsing
elsewhere, never the native word geometry itself). A real child row nested
under a section heading starts strictly to the right of its heading — the
same indent relationship `_native_rows`'s own prefix-stack push/pop already
uses elsewhere in this module to decide ancestry.

So, in addition to the existing font-equality check, the widened merge now
also requires `child_bbox.x0 <= heading_bbox.x0` (native's own `label_bbox`
for each row — no new geometry parsing needed, both are already stored on
`_NativeRow`). Missing bbox on either row abstains (no merge), matching the
module's fail-closed rule for ambiguous/absent geometry evidence (same
posture as the existing missing-font-fields guard).

No magic threshold: the check is a pairwise comparison of two bboxes already
computed for other purposes, not an absolute distance or a tuned constant.

## Evidence — production caller, `spans` included

`tests/test_binding.py`:

- `test_gh692_same_font_heading_with_indented_child_does_not_merge` — same
  words/font/text-shape as the existing wrapped-label fixture, except the
  second row's own label is indented deeper than the first (`x0=80` vs
  `x0=50`). Runs through `bind()` with `spans=[...]`, the production path.
  Before the fix this merged; after the fix it must not.
- `test_gh692_same_font_wrapped_label_still_merges_when_not_indented` — same
  shape but the second row's label starts at the *same* left edge (the
  existing `_TWO_BASELINE_WORDS`/`_TWO_BASELINE_MARKDOWN` fixture). Pins the
  other direction: the guard must not degrade into "never merge" and re-open
  #624b.

Both go through `bind()` with `spans`, matching the production caller
(`flatten_page_spans` always supplies `spans`), not an isolated call into
`_wrapped_label_merge_plan`.

## Baseline / after, measured in this worktree

- `tests/test_binding.py` alone: 96 passed, 1 xfailed (baseline) ->
  98 passed, 1 xfailed (after; +2 new tests).
- Full suite (`PYTHONPATH=/tmp/wt-692/src ~/venvs/socr/bin/pytest -q`):
  5612 passed, 4 xfailed (baseline, measured before editing) ->
  5614 passed, 4 xfailed (after; +2, no regressions, no new xfails).

## Mutation round

Copied `src`, `tests`, and `pyproject.toml` to `/tmp/gh692-mutant` (outside
this repo — `pyproject.toml`'s `pythonpath = ["src"]` would otherwise shadow
an external `PYTHONPATH` and silently test the real source). Confirmed the
anchor block (`if wide_proven: ... font_widened = font_a is not None and
font_a == font_b` plus the new indent-bbox guard) appears exactly once in
the source before mutating. Reverted the added indent-bbox guard, restoring
bare font-equality widening. Ran a canary inside the mutant's own pytest
process (`os.path.realpath(socr.__file__)` starts with the mutant's
`realpath('src')` — needed because `/tmp` symlinks to `/private/tmp` on this
machine, so a bare prefix check on the unresolved path would false-fail on
correct mutant source).

Result: `tests/test_binding.py::test_gh692_same_font_heading_with_indented_child_does_not_merge`
FAILED (exactly the expected regression — the merge fires again on a
same-font, differently-indented heading+child pair).
`tests/test_binding.py::test_gh692_same_font_wrapped_label_still_merges_when_not_indented`
stayed green, and the rest of `test_binding.py` (97 passed, 1 xfailed
besides the one deliberate failure) was unaffected. Guard is load-bearing
and does not mask the wrapped-label direction.

## Files changed

- `src/socr/tables/binding.py` — indent-bbox guard on the widened font-based
  merge in `_wrapped_label_merge_plan`, plus docstring updates.
- `tests/test_binding.py` — two new GH-692 tests (`_INDENTED_CHILD_WORDS`/
  `_INDENTED_CHILD_MARKDOWN` fixtures).

## Scope note

No plan folder (`docs/plans/*/STATUS.md` / `TICKETS.md`) references GH-692,
so none was updated — this ticket was dispatched standalone via
`TICKET-692.md`, not through a tracked initiative plan.
