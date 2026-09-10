# #714 round 2: a text table leaves the numeric-corroboration route

Branch `fix/714-a1b-text-table-blind-spot`, on top of round 1 (`0a10cd5`).
Round 1's diagnosis stands; its remedy is reversed.

## Why round 1 was wrong

Round 1 made A1b's boolean veto return True where the native page shows no
recurring numeric column lanes, reasoning that the predicate is a veto so "not
applicable" means "do not veto". At that call site True means ADMIT, and
the only evidence behind the admission is A1a's numeric-row corroboration.

Astra reproduced the consequence on the real page. Take the real BoE 2018 p1
cached qwen candidate, replace one text-only productivity row with

```
| The Bank guarantees permanent prosperity without any risk. | Unconditional guarantee. |
```

and change nothing else. Both candidates score:

```
bound=2, total=2, candidate_numbers=2, native_numeric_rows=2,
extra_numbers=(), skipped_native_rows=0, unbound_rows=((),)
```

Under round 1 the fabricated body SHIPPED, at `WARNING` /
`HEADER_BINDING_UNVERIFIED` via `STRUCTURE_CLASS_GRID_CORROBORATED`. The
fabricated row contributes nothing to the denominator, so the corroboration
cannot be the guard, and A1c is disclosure rather than verification. Two
matching numeric rows do not validate arbitrary prose cells.

## The rule

Where the lane gate is closed, the row-shape reconciliation is NOT APPLICABLE
and the numeric-row-corroborated ending is **not available** for that candidate.
A1b declines it, with its own reason.

The outcome is three-valued (`RowShapeOutcome`):

| outcome | meaning | admits? |
| --- | --- | --- |
| `RECONCILED` | counts reconcile, or nothing to compare | yes |
| `SHORTFALL` | page has lanes, candidate is short: rows were dropped | no |
| `NOT_RECONCILABLE_TEXT_TABLE` | no lanes: this route cannot speak for this candidate | no |

The invalid row-count comparison is NOT restored as an accidental defence: on a
lane-closed page it is never computed. Callers admit on `RECONCILED` and on
nothing else, so no caller's contract changed; the boolean face is gone (see the
follow-through below) because the two declines are not interchangeable.

The candidate falls through to the routes that CAN carry authority for text
cells: a completed page-judge acceptance, or #713's table-acceptance credential.

## The reason is legible at every level

Modelled on #713, which established that a floor must not lie about why.

* **Page.** `FailureMode.ROW_SHAPE_NOT_RECONCILABLE_TEXT_TABLE` and
  `SelectionProvenance.STRUCTURE_CLASS_TEXT_TABLE_FLOOR`. The public
  `PageDisposition` stays `(FAIL_CLOSED_MARKER, STRUCTURE_CLASS)`, deliberately
  and for #713's reason: every document surface keyed on that pair keeps
  counting the page, and the new fact rides on the mode and the tag.
* **Sidecar.** Both are persisted on the finalized record like any other mode.
* **Document.** Its own audit event
  (`row_shape_not_reconcilable_text_table_floor`) and its own sentence in
  `_structure_class_floor_note`, and the page is excluded from the "ladder
  exhausted" sentence -- it exhausted no ladder.
* **CLI.** Its own red line naming the pages and the two remedies.
* **Resume.** Added to `floor_shipped`, so the page forfeits the
  content-terminal exception exactly as the other two floor modes do. Missing
  this would have left a floored page restored verbatim and never re-OCR'd.

**Precedence, disclosed.** When the page judge also timed out, the floor reports
the timeout. Both facts hold; the timeout's remedy (re-run the judge) is what
produces the completed acceptance the declined text table needs, so it is both
the earlier cause and the subsuming one. Pinned by
`test_a_typed_judge_timeout_outranks_the_text_table_reason`.

Scoped to candidates that CLEARED A1a. A candidate with no bindable numeric rows
never reaches the row-shape check (`clears=None`), so its page floors under the
ordinary exhausted reason, which is truthful for it. Astra's zero-numeric control
pins that.

## What the real BoE page ships, re-measured

`finalized_page_record` over the real PDF, the run's own sidecar geometry and
page flags, and the run's own cached qwen `PageOutput`.

| cache state | gate | failure mode | provenance | body |
| --- | --- | --- | --- | --- |
| as cached (`audit_passed=False`, `judge raised: timed out`) | real | `row_shape_not_reconcilable_text_table` | `structure_class_text_table_floor` | 47-char marker |
| as cached | forced open | `structure_class_ladder_exhausted` | `structure_class_floor` | 47-char marker |
| counterfactual `audit_passed=True` | real | `none` | `passing_best_output` | 3539 chars, carries `fall to 4%` |
| counterfactual `audit_passed=True` | forced open | `none` | `passing_best_output` | 3539 chars |

**The page as cached ships the marker, and that is the honest outcome.** The
cache holds no acceptance credential and its timeout is untyped
(`page_judge_timeout_attempt` returns None), so neither route that can speak for
prose is available. Round 1's 3539-character recovery was an admission without
evidence, not a recovery. What the gate changes is the REASON: the same bytes,
under a mode that names what is missing.

The counterfactual row is the one #703 measured and is unchanged: on a page whose
candidate was accepted outright, A1b's branch is never reached.

## Tests

`tests/tables/test_gh714_a1b_text_table_gate.py` (14) and
`tests/tables/test_gh714_astra_review_reproducer.py` (5, Astra's own file
transcribed; the corpus tests skip where the census corpus is absent, and the
one round-1 assertion that no longer holds is re-pinned to the reason rather
than the bytes, with the adaptation named in the file).

Outcome pins, `(gate_open, gate_real)`:

| page / candidate | ungated | gated |
| --- | --- | --- |
| real BoE 2018 p1 cached qwen | SHORTFALL | NOT_RECONCILABLE_TEXT_TABLE |
| hermetic text table | SHORTFALL | NOT_RECONCILABLE_TEXT_TABLE |
| sparse-prefix truncated / complete | SHORTFALL / RECONCILED | SHORTFALL / RECONCILED |
| ECB bulletin p2 and p3, truncated / complete | SHORTFALL / RECONCILED | SHORTFALL / RECONCILED |
| 20-row grid, 2 rows deleted either end | SHORTFALL | SHORTFALL |

Every numeric outcome is unchanged. Selection pins: the declined page floors
under its own mode and tag; the fabricated variant never ships, hermetically and
on the real page; a completed page acceptance ships the text table in full; a
verified #713 credential ships it demoted under `JUDGE_TIMEOUT_LADDER_ACCEPTED`.

Guard tests updated for the 21st ending: `test_r7_winner_kind_tags` (its
`_tag_names` now flattens a nested conditional tag, since the floor carries three
reasons over one set of bytes) and `test_p6_disposition_contract`.

Full suite 5143 passed, 4 xfailed. `uvx ruff@0.16.0 format --check .` clean.

## Residuals

- **The census page is not recovered, and nothing here claims it is.** It needs a
  completed page-judge acceptance or a minted credential; the cache has neither,
  and #713's own credential path cannot help because the cached timeout was never
  typed. Whether a re-run produces either is unmeasured.
- **The CLI line is not exercised by a test.** The audit event and the document
  note are (through the real assemble phase); the console line is driven by the
  same derived page list and was verified by reading, not by executing.
- **A text table with NO numeric rows at all is unaffected** and floors under the
  ordinary exhausted reason. That is truthful but coarse: the page is withheld
  without the specific explanation the declined case now gets.
- **The decline is page-wide, inheriting every #703 lane-gate limitation**, the
  citation-rows scope limitation included. A text table sharing a page with three
  aligned numeric bands opens the gate and is exposed to the old comparison again.
- **`STRUCTURE_CLASS_GRID_CORROBORATED` is now unreachable for text tables** by
  construction. Any future route that admits text cells must bring its own
  evidence; this ticket removes one route, it does not add one.
- **Nothing here verifies prose cells.** The two available routes attest a page
  judge's acceptance and a table judge ladder's acceptance respectively. A
  fabricated prose cell inside a page the judge accepted is out of scope and
  remains so.

## Follow-through (Astra's two nonblocking nits at `cc8e4c8`)

- **`_row_shape_reconciliation_ok` is removed.** After round 2 no production
  code called it; only tests did, and a boolean face invites exactly round 1's
  mistake of reading "not False" as "admissible". The ticket tests now compare
  `_row_shape_reconciliation` against `RowShapeOutcome.RECONCILED` directly.
- **The "nothing refused the reading" claim is scoped.** In the CLI line and in
  the selection comment it now says that numeric-row reconciliation does not
  apply to this candidate, and states explicitly that this is not a claim that
  no judge refused it: a page-judge or table-judge rejection can coexist with
  this route diagnostic.

No behaviour change: same outcomes, same failure mode, same tag, same buckets.
