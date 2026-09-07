# 2026-09-07 — TICKET-B1 (#591): page_failed ending marker scope

Status: **DONE** — CONSILIUM-GATE raised and resolved by team-lead (Option 3 +
prose-corroboration guard). Both `NO_TEXT_MARKER` and `UNVERIFIABLE_TABLE_SCANNED`
now guarded; 16 tests passing; see the decision section below for the full
measurement and implementation record.

## What was implemented

`src/socr/core/manifest.py`:

- `_table_bbox_sane(p)` — the two bbox sanity checks the panel asked for (#591), built
  on `row_corroboration.baseline_bands` / `words_in_region`: "too small" (the claimed
  bbox's own bands contain no genuine numeric token — the box missed the table's rows)
  and "too large" (bands with no numeric token outnumber bands that have one — the box
  swallowed prose). Either failing floors the whole page, same as the four GH-520
  coverage conditions.
- `table_floor_text_for_source(p, page_num, source_text, *, fallback_marker=None)` —
  extended with a `fallback_marker` kwarg so a caller with its own whole-page marker
  (not the D3 "unverifiable table" string) gets ITS marker on a guard failure. Backward
  compatible: `structure_class_floor_text` and `_apply_ladder_disposition_guard`'s
  `TABLE_WITHHELD` branch call it with no `fallback_marker` and are unchanged.
- `_select_page_output_tagged`'s "nothing anywhere produced text" branch (provenance
  `NO_TEXT_MARKER`) now calls `table_floor_text_for_source(p, page_num, p.native_text,
  fallback_marker=page_failed_marker(page_num))` and ships the spliced result when it
  differs from the marker and native text is non-empty; otherwise the plain marker,
  unchanged from before B1.

`tests/pipeline/test_page_failed_marker_scope.py` (new, 12 tests, all passing):
synthetic guard-satisfied vs guard-violated pairs (region-count mismatch, missing bbox,
no markdown block, both new bbox checks), a backward-compatibility pin for the
`structure_class_floor_text`/`TABLE_WITHHELD` default marker, and real-fixture tests
against the two named PDFs (skipped if absent on the machine).

```
PYTHONPATH=/Users/rubenffuertes/repos/.worktrees/socr-b1/src ~/venvs/socr/bin/pytest \
  tests/pipeline/test_page_failed_marker_scope.py -q
12 passed
```

## The fork: `NO_TEXT_MARKER` structurally never has recoverable native text

Traced `_select_page_output_tagged` (`manifest.py:1944`) end to end. The branch this
ticket targets is reached only when `not (p.is_born_digital and p.native_text)` —
every path through the preceding `if p.is_born_digital and p.native_text:` block
(native-clean, native-fallback, the D3/TR-3 floor, the rotated-shredded floor, the
structure-class floor, the flagged-model keep) returns unconditionally; none fall
through.

Cross-checked every one of the 11 early returns in `born_digital._assess_page_signals`:
**every** `is_born_digital=False` return also sets `native_text=""`. There is no
production code path that sets `is_born_digital=False` while leaving `native_text`
populated.

Consequence: on a real page, `NO_TEXT_MARKER` is reached only with `p.native_text`
already empty. The `native_text = getattr(p, "native_text", "") or ""` line B1 added
is always `""` in production, so `table_floor_text_for_source` always takes its
`if not source_text.strip(): return whole_page` early exit — the routing fix is
correct and unit-tests green (`is_born_digital=False` + non-empty `native_text`
forces the branch and proves the splice mechanism works), but it **cannot change any
real page's shipped bytes** as scoped. Verified directly:

```python
# is_born_digital=True, native_text set (what a real born-digital page looks like):
_select_page_output_tagged(...) -> SelectionProvenance.NATIVE_CLEAN  # never reaches NO_TEXT_MARKER

# is_born_digital=False, native_text set (does not occur on any real page):
_select_page_output_tagged(...) -> SelectionProvenance.NO_TEXT_MARKER  # only way to exercise it
```

This is the exact fork flagged in the ticket's own STOP condition: "the page_failed
ending has no access to native text at assembly time."

## The two named fixtures don't reach `NO_TEXT_MARKER` either — measured

| Fixture | Real disposition | Why not `NO_TEXT_MARKER` |
|---|---|---|
| ECB survey-2013 p1 | `STRUCTURE_CLASS_FLOOR` | Already fixed pre-B1 (GH-520). Guard **fails** here (#639): `detected_table_count=3` vs `native_table_region_count=1` (the extra two bboxes are the chart-caption/units area, not the table body) — whole-page marker ships, correctly, per the guard's own contract. |
| Fed 1989-11-14 p3 | `UNVERIFIABLE_TABLE_SCANNED` | Not touched by B1. This branch (`manifest.py` ~2126) splices `best_output.text` (the nougat attempt) around every table region with **no GH-520 guard at all** — an existing gap, separate from this ticket. Also: `BornDigitalDetector.detect_page` returns `is_born_digital=False`, `native_text=""` for this page (6.6% encoding corruption routes it to OCR before any table logic runs) — even if this branch called the new guard, there is no native text to splice. |

Demonstrated the guard-satisfied mechanism works on REAL text regardless: feeding
ECB p1's own native text, its first (correct) detected bbox only, and a matching
region count of 1 through `table_floor_text_for_source` directly (bypassing the
unreachable-in-practice call site) produces the prose after the table intact —
**word recall 1.0** against pymupdf's own words outside that bbox. Pinned as
`test_ecb_survey_2013_p1_guard_satisfied_with_the_tables_own_bbox_achieves_recall`.

## Decision needed

B1 as ticketed (fix `NO_TEXT_MARKER` only) ships correct, tested, but practically dead
code — it cannot recover any real page's prose, and neither named fixture exercises it.
Options:

1. **Ship B1 as-is.** The fix is harmless and correct if the invariant
   (`is_born_digital=False ⟹ native_text=""`) ever changes or a code path I haven't
   found breaks it. Open a follow-up ticket for `UNVERIFIABLE_TABLE_SCANNED` (the
   branch that actually needed a guard on the Fed p3 fixture) separately.
2. **Redirect B1 to `UNVERIFIABLE_TABLE_SCANNED`** (`manifest.py` ~2126-2150): splice
   `best_output.text` through the same `table_floor_text_for_source` +
   `fallback_marker` mechanism instead of `p.native_text`, mirroring how the
   `TABLE_WITHHELD` ending already passes a non-native source into the same function.
   This is the branch the Fed p3 fixture (#591's own reproduction case) actually
   reaches, and it currently splices with no coverage guard at all.
3. **Both**, in one ticket: keep the `NO_TEXT_MARKER` fix already written (it is
   correct and passes review on its own terms) and extend the same call to
   `UNVERIFIABLE_TABLE_SCANNED`.

Not committed. `src/socr/core/manifest.py` and `tests/pipeline/test_page_failed_marker_scope.py`
are on disk in the worktree, uncommitted, pending this decision.

## Team-lead's decision: Option 3 + a mechanical prose-corroboration guard

team-lead chose Option 3. Keep the `NO_TEXT_MARKER` fix (correct, cheap, ships as
above). Extend the same fail-closed philosophy to `UNVERIFIABLE_TABLE_SCANNED`
(`manifest.py` ~2103): that branch splices `best_output.text` around the withheld
table region with **no guard at all** — on Fed p3, that attempt is nougat,
`failure_mode=hallucination`. Directive: add a geometric/mechanical-only
corroboration guard — keep the attempt's outside-table prose iff its own vocabulary
(tokens of 4+ letters) overlaps the page's native words at or above a named constant
`PROSE_CORROBORATION_MIN`, measured first against real fixtures.

### Measurement: word-overlap on the two named fixtures

Methodology: `re.findall(r"[a-z]{4,}", text.lower())` → `set()`; overlap = `|attempt
∩ native| / |attempt|` (fraction of the attempt's own vocabulary corroborated by the
page's real text).

| Fixture | Attempt | `failure_mode` | Overlap |
|---|---|---|---|
| Fed 1989-11-14 p3 | nougat (only cached attempt; no qwen/alternative exists for this page) | `hallucination` | **1.0** |
| ECB survey-2013 p1 | gemini-3-flash-preview | `none` (genuine) | **1.0** |

**Anomaly**: both measured 1.0 — no gap to set the constant "between", as directed.
Root cause, confirmed by reading the raw nougat attempt text
(`cache/ef/ef6b822...json`): nougat read the page's REAL vocabulary (the correct bank
names, dollar amounts, dates, and the FOMC directive paragraph verbatim) but emitted
the swap-arrangement table as one run per COLUMN instead of one row per line — a
**structural** defect (wrong row/column binding), not a vocabulary fabrication.
Token-set overlap is blind to this by construction: it cannot see word ORDER or
ATTRIBUTION, only word MEMBERSHIP. `failure_mode=hallucination` in this codebase
covers both senses (fabricated content AND structurally-scrambled-but-real content);
the census set has no example of the first sense to anchor the low side of the
constant. Recorded as a genuine gap, not a bug in the measurement.

Also measured (per the "any qwen/other cached attempt" instruction): Fed p3's cache
directory holds 6 files — one nougat attempt for page 3 (the one measured above), one
already-spliced/marker-only entry for page 3 (not a raw attempt), and four
`chart_asset`-engine entries for pages 1/2/4/5 (not page 3). **No alternative engine
attempt exists for this page** — the "no corroborating attempt exists" branch of
team-lead's directive is confirmed for this fixture; there is nothing else to try.

### `PROSE_CORROBORATION_MIN` — not calibrated from a discriminating anchor pair

Set to **0.5**, documented in `manifest.py` as a defensive floor rather than a
measured threshold (unlike `ROW_CORROBORATION_MIN` / `EXTRA_NUMBERS_MAX_SHARE` in
`row_corroboration.py`, both set strictly between two real anchors). Flagged as a
follow-up: calibrate against a genuine content-fabrication fixture when one turns up
in the census — this pair only demonstrates the guard cannot yet distinguish
"structurally garbled but real" from "fabricated," which is a real limitation of a
pure vocabulary-overlap check, not of the specific constant chosen.

### A second, load-bearing gap found while measuring: `native_words` was never cached for this page class

Before touching `manifest.py`, confirmed directly with pymupdf that Fed p3 DOES have
a real native text layer (295 words) — team-lead's "the page has a text layer" premise
holds. But `orchestrator.py`'s native-words caching (TICKET-A1b, #634) only runs for
pages with `detected_table_count > 0`; Fed p3 measures `detected_table_count = 0`
(no native table structure to detect on a page that reached OCR precisely because it
lacks one). Separately, `state.py`'s `apply_born_digital` only ever copies
`pa.native_text` onto `PageState` `if pa.is_born_digital` — so `p.native_text` is
unconditionally empty for every page in the `UNVERIFIABLE_TABLE_SCANNED` branch
(`not p.is_born_digital`), by construction, not just on this fixture.

Net effect: as scoped, the new guard would have had **zero witness** on the exact
fixture it exists to protect — not "insufficient corroboration," no data to check at
all. Widened the `orchestrator.py` caching filter to
`pa.detected_table_count > 0 or not pa.is_born_digital`, so every scanned page also
gets its native words cached (cheap: same `get_text("words")` call already made
best-effort for the A1b set). This is the change that makes
`_prose_corroboration_ok` able to fire at all on Fed p3 — confirmed:
`_prose_corroboration_ok(ps, nougat_text)` now returns `True` (correctly: it's not
fabricated, per the anomaly above) instead of `False` from a missing witness.

### Coverage guard: does NOT apply to `UNVERIFIABLE_TABLE_SCANNED`, and why

Team-lead asked for the same four-condition GH-520 coverage guard
(`table_floor_text_for_source`) or an explicit statement of why it cannot apply.
Its first condition is `detected_table_count > 0`. A page reaches
`UNVERIFIABLE_TABLE_SCANNED` precisely because native table DETECTION found nothing
on it (measured: Fed p3, `detected_table_count=0`, 0 bboxes) — there is no detected
table geometry for `find_table_blocks`'s parsed-block count to reconcile against.
The coverage guard's entire mechanism (detector count == parser count == bbox count)
has nothing to measure here. Documented in `manifest.py` at the call site; the
mechanical check this branch uses instead is `_prose_corroboration_ok`.

### What ships for Fed p3 after the fix — unchanged, for an orthogonal reason

`_prose_corroboration_ok` now passes (overlap 1.0). But `splice_all_table_regions`
on the raw nougat text still returns `None` — nougat never emitted markdown
pipe-table syntax for this page (it dumped the table as bare lines, one value per
line, no `|` characters), so `find_table_blocks` finds nothing to splice around. The
whole-page D3 marker fires regardless of the new guard, matching the fixture's real
recorded output (`~/Data/socr/census-591-recheck/out/fed-1989-11-14-minutes/pages/
00003.json`: `winning_output.text` is the bare marker + image ref,
`disposition.ending = "fail_closed_marker"`). **Follow-up, as directed**: recovering
this page's directive-paragraph prose needs a real OCR attempt of the prose region
specifically (not the whole page, which drags the table into the same attempt) —
out of scope for B1.

### Implementation

`src/socr/core/manifest.py`:
- `PROSE_CORROBORATION_MIN: float = 0.5` and `_prose_corroboration_ok(p, attempt_text)`
  — geometric/vocabulary-only guard, documented above and in the constant's own
  docstring (including the anomaly and the missing-anchor caveat).
- `UNVERIFIABLE_TABLE_SCANNED` branch: gates `splice_all_table_regions` on
  `_prose_corroboration_ok`; falls back to the whole-page D3 marker exactly as before
  when the guard fails (violated or no witness).

`src/socr/pipeline/orchestrator.py`:
- Widened the TICKET-A1b native-words caching filter to also cover
  `not pa.is_born_digital` pages, with a comment explaining why (see above).

`tests/pipeline/test_page_failed_marker_scope.py` (4 new tests, 16 total, all
passing): synthetic guard-satisfied vs guard-violated pair pinning the corroboration
difference directly, a no-witness-fails-closed test, and a real-fixture test against
Fed p3's actual cached nougat attempt and real pymupdf words, pinning both measured
findings (guard passes; marker still ships because there is no table block to
splice).

```
PYTHONPATH=/Users/rubenffuertes/repos/.worktrees/socr-b1/src ~/venvs/socr/bin/pytest \
  tests/pipeline/test_page_failed_marker_scope.py -q
16 passed
```

### Full suite: two pre-existing/incidental regressions found and fixed

A foreground full-suite run (`pytest -q`, ~4310 tests, 5:39) surfaced 6 failures,
none in the new test file:

1. **`tests/test_r7_winner_kind_tags.py`** (3 failures) — R7's structural proof that
   `_select_page_output_tagged` is single-return-per-ending (one `return` statement
   per `SelectionProvenance` tag, in enum declaration order) broke. Root cause:
   the ORIGINAL `NO_TEXT_MARKER` fix (from before this session, already on disk)
   used two separate `return` statements both tagged `SelectionProvenance.
   NO_TEXT_MARKER` — legal Python, but it breaks R7's bijection invariant (16
   returns / 17 tags become 17 returns / 18 tag-uses, one tag used twice) and
   its "one ending, one return" contract that other code relies on. Collapsed to
   a single `return` with the two outcomes distinguished by conditional
   expressions inside the `PageOutput(...)` call (`text=... if prose_kept else
   ...`), tag written once. This was latent before my Option-3 work — the
   original CONSILIUM-GATE session ran only the new test file, never the full
   suite, so it was never exercised.
2. **`tests/test_gh371_d3_region_splice.py::TestTask5_ScannedPageProsePreservation`**
   (3 failures) — these GH-90/GH-371 tests exercise `UNVERIFIABLE_TABLE_SCANNED`'s
   splice directly and never set `PageState.native_words`, so the new
   `_prose_corroboration_ok` guard correctly refused (no witness) what it used to
   splice unconditionally — exactly the behavior change team-lead asked for, now
   also hitting these fixtures' synthetic pages, which have no real PDF behind
   them to derive words from. Added a `_native_words_for(text)` test helper
   (one word-tuple per token, real coordinates irrelevant since these fixtures
   never set `detected_table_bboxes`) and set `ps.native_words =
   _native_words_for(model_text)` in the three affected tests — the fixture's own
   "native text layer" now says the same thing its OCR attempt does, which is
   the scenario these tests were written to prove works.

Re-ran the affected files plus the new B1 file after both fixes: 58 passed. Full
suite re-run clean:

```
PYTHONPATH=/Users/rubenffuertes/repos/.worktrees/socr-b1/src ~/venvs/socr/bin/pytest -q
4314 passed, 4 xfailed, 5 warnings in 310.23s (0:05:10)
```

```
uvx ruff@0.16.0 format --check .
597 files already formatted
```
