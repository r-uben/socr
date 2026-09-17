# 2026-09-17 — GH-692: font equality alone must not merge a section heading

**Round 2 correction below supersedes the strict `child_bbox.x0 <= heading_bbox.x0`
comparison this log originally described — see "Round 2" for why and what
replaced it.**

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

## Round 2 — the strict comparison was itself a regression

The owner measured what round 1 did not construct: varying the SECOND
LINE'S NATIVE WORDS (not `spans` — an earlier attempt on their side varied
only `spans` and merged every time, proving nothing, since `label_bbox`
comes from `words`/`lane_of`, not from `spans`). Result, on the round-1
strict `child_bbox[0] > heading_bbox[0]` comparison:

    delta=0      -> merges
    delta=0.01   -> does NOT merge
    delta=0.1    -> does NOT merge
    delta=2.0    -> does NOT merge   (ordinary hanging indent)
    delta=30     -> does NOT merge   (real nested child)

Exact x0 equality is far too strict. Real extracted continuation lines of
the same cell carry sub-point jitter (glyph left side bearing, kerning,
float rounding through the extraction path), and a small hanging indent is
a standard, common continuation-line convention — round 1's guard refused
both, which would have re-opened #624b far more broadly than the heading
case it closed. I could not show from the code that hanging indents cannot
occur here (`_native_rows`'s prefix-stack push/pop only constrains
relationships between successive `is_parent` rows, never a continuation
line's own offset), and I do not have corpus access to check whether this
specific corpus's tables use hanging indents. Reported this measurement
before changing anything, per instruction not to pick a fix silently.

### Fix: one-em tolerance, derived from the row's own font size

Replaced the strict inequality with a tolerance: the widened merge still
refuses only when `child_bbox.x0 - heading_bbox.x0` exceeds **one em** —
the label's own font point size (`font_a[1]`, the rounded size already
carried on `label_font`; guaranteed available here since `font_widened`
cannot be `True` without both fonts having resolved). One em is the
standard typographic unit for an indentation step: comfortably larger than
jitter/hanging-indent offsets, comfortably smaller than a genuine nesting
level (a distinct indentation column, not a sub-character shift). This is
page-derived data, not a picked constant — it varies with the document's
own label font size rather than a fixed point value. It explicitly does
NOT claim to bound a real nesting level that happens to be smaller than one
em; this module has no way to observe that from the data available to it.

### Evidence — boundary pinned both sides, plus the jitter case named up front

`tests/test_binding.py` (GH-692 section rewritten around a
`_indented_second_line_words/_spans(delta)` factory so every case shares
the same font/text shape and only the second line's indent varies):

- `test_gh692_same_font_heading_with_indented_child_does_not_merge` —
  `delta = 3 * one_em` (a full indentation column): still refuses.
- `test_gh692_same_font_wrapped_label_still_merges_when_not_indented` —
  `delta = 0`: still merges (unchanged from round 1).
- `test_gh692_wrapped_label_merges_despite_subpoint_extraction_jitter` —
  `delta = 0.01`: the exact case that would have bitten silently in
  production; now merges.
- `test_gh692_wrapped_label_merges_with_hanging_indent_at_exactly_one_em` —
  `delta = one_em` exactly: merges (inside-boundary pin).
- `test_gh692_heading_indent_just_over_one_em_does_not_merge` —
  `delta = one_em + 0.01`: does not merge (outside-boundary pin).

### Baseline / after, measured in this worktree (round 2)

- `tests/test_binding.py` alone: 98 passed, 1 xfailed (round-1 after) ->
  101 passed, 1 xfailed (round-2 after; +3 new tests, net of the 2 rewritten
  round-1 tests kept and 3 new ones added).
- Full suite: 5614 passed, 4 xfailed (round-1 after) -> 5617 passed,
  4 xfailed (round-2 after; +3, no regressions, no new xfails).

### Mutation round (round 2)

Copied `src`, `tests`, `pyproject.toml` to a second scratch dir outside the
repo (same reason as round 1). Confirmed the anchor line
(`if indent_delta > one_em:`) appears exactly once before mutating.
Mutated it to `if indent_delta > 0:` — i.e. reintroduced round 1's own bug
(effectively zero tolerance). Canary confirmed the mutant pytest process
resolved `socr.__file__` inside the mutant tree via `os.path.realpath`.

Result: exactly
`test_gh692_wrapped_label_merges_despite_subpoint_extraction_jitter` and
`test_gh692_wrapped_label_merges_with_hanging_indent_at_exactly_one_em`
FAILED (the two cases the tolerance exists for); the other 3 GH-692 tests
and the rest of `test_binding.py` (99 passed, 1 xfailed besides the 2
deliberate failures) were unaffected. The tolerance is load-bearing and its
own two pinning tests are what catch its removal — nothing else does.

Lint (`uvx ruff@0.16.0 format --check .`): clean, "732 files already
formatted".

## Parked for the owner — the one-em cutoff itself is an open fork

The owner measured both failure directions of this guard and found the
cardinal "a dropped row is worse than a missing one" rule does not
adjudicate between them: an over-merge consumes a heading (a row
disappears), an under-merge splits a wrapped label across two rows (the
value binds to the trailing half of the label instead of the whole).
Neither loses content outright — `row_label_contradictions` stays empty
either way — so which is worse is a corpus fact (do this corpus's tables
use hanging indents past one em, or nesting levels at or under one em?)
that is not obtainable from this worktree. Decision parked for the owner
with both measurements and two candidate next steps: (1) derive the
nesting quantum from another genuine parent->child indent step observed
elsewhere in the SAME table, falling back to today's one-em constant only
when no such reference exists in a single-section document; (2) find an
orthogonal second signal instead of tightening indent further.

Per the team lead's request, added
`test_gh692_open_fork_larger_hanging_indent_does_not_merge_pending_corpus_fact`
(delta = 2 × one em) as a plain (non-xfail) test pinning TODAY's actual
behaviour, with the fork spelled out in its docstring, so the next person
inherits the measurement instead of rediscovering it. It is explicitly not
a claim that today's behaviour is the correct final answer.


