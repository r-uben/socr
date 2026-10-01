# Cleanup: ship gate, native-first lane, resume kinds (2026-10-01)

Branch `chore/cleanup-ship-gate` from origin/main 3cf5f87 (#926 merged; ancestry verified with
`git merge-base --is-ancestor`). Behaviour-preserving. The review items are inlined below
(T1-T25 from the tables review, P1/P8/P9/P10 from the pipeline review); both reviews were
written before #926, so every T item was re-located on current main.

Rule applied throughout: outputs, events, error and detail strings, routing and plan decisions
are identical. An item that could change any of them was skipped, with the reason below.

## Items

### ship_gate.py (tables review)

| Item | Status | What |
| --- | --- | --- |
| T1 module + predicate docstrings are wrong | DONE | Module docstring rewritten to the code as of #926: coverage is `extended_span` plus the interior between same-table blocks (inclusive both ends), `_PANEL_GAP_ROWS` and `_SAME_TEXT_DIRECTION_TOL_RAD` are the two MEASURED thresholds (the lane tolerances are the rowizer's own), predicate list now includes `header_band_missing`, `foreign_direction`, `direction_unavailable`; "each a separate function" and "nothing here is a new threshold" are gone. `data_row_missing_faults` docstring states the real span. Also answers cubic P3 on #926: the header describes the measured direction tolerance. |
| T2 split `order_faults` | DONE | `row_order_faults(pairs)` and `cell_order_faults(blocks, pairs, src_rows)`, called in the same order. The test patch in `test_gh916` now targets `row_order_faults`. |
| T3 lane counting 4x | DONE | `_lane_count(words, lanes)`; used by `_table_geometry`, `extended_span`, `table_spans`, `data_row_missing_faults`. |
| T4 token normalisation | DONE | `_out_seq(cells)` and `_src_seq(row_words)` replace the six inline comprehensions. `_key` keeps normalising raw tokens (it is not idempotent-safe to feed it normalised ones). The per-block key list is computed once in `_unique_pairs` instead of twice per row. |
| T5 `table_spans` index-mutated lists | DONE, with a deviation | `_Span` is a `NamedTuple(lanes, core, y_lo, y_hi)`, not a frozen dataclass: `benchmark/ship_gate_gaps.py` unpacks `for lanes, _core, y_lo, y_hi in g.table_spans(...)`, and a NamedTuple keeps that shape. The merge step uses `_replace`. |
| T6 `_table_geometry` repeated per block | DONE | `_block_geometries(pairs, src_rows)` computed once in `native_ship_gate` and passed as `geos=` to `table_spans`, `data_row_missing_faults`, `label_row_missing_faults`, `header_band_missing_faults`. The defaults (`geos=None`) keep every public signature callable as before. (It ran three times per block, not two: #926 added a third site.) |
| T7 split `label_row_missing_faults` | SKIPPED | Not in the requested list. Only a short docstring was added. |
| T8 `extended_span` mirrored walks | DONE (partial) | One inner `walk(edge, outward_ys)`. NOT done: replacing `min(core)`/`max(core)` with `ys_sorted[0]`/`[-1]`. On an empty `core` that swaps a `ValueError` for an `IndexError`, and `native_ship_gate` puts the exception text into the `gate_error` detail string. Unreachable today (`core` has >= 2 rows), but it is a string, so it stays. |
| T9 `_snap()` / `_lanes` | DONE | `_SNAP_PT` constant; `_lanes` inlined. The test that called `ship_gate._snap()` uses `_SNAP_PT`. |
| T10 `_PANEL_GAP_ROWS` comment | DONE | Kept the rule, the entry point `socr-measure-ship-gate-gaps`, and a pointer to `docs/log/2026-10-01_916-native-ship-gate.md`. |
| T11 fault dict literal 6x | DONE | `GateFault` TypedDict and `_fault(predicate, detail)`; `native_ship_gate` and `NativeTablePlan.faults` are typed with it. Key order unchanged. |
| T12 type hints | DONE (partial) | Aliases `Word`, `Block`, `SourceRows`, `BlockPairs`. `_CellText.take` stays a tri-state (True/False/None); it is documented in its docstring and the call site now says so, instead of an Enum. |
| T13 renames | DONE | `_is_num` -> `_is_source_number`, `_lead` -> `_leading_number`, `_USED` -> `_CONSUMED_MARK`, `_Anchors` -> `_unique_pairs(blocks, src_rows) -> list[BlockPairs]` (the `anchors` parameters became `pairs`; `out_count` was never read outside the class). `benchmark/ship_gate_gaps.py` and the tests that used `_Anchors` follow. |
| T14 stale comment / dict comprehensions | DONE | Core-membership comment says what core membership is; the two `lines` comprehensions in `sign_detached_faults` are one. |

### native_first.py / reconstruct.py

| Item | Status | What |
| --- | --- | --- |
| T15 rotation dead branch, wrong docstring | DONE | Removed the unreachable `else` (the function returns on empty `words` first) and the `_rotation_center_x/_y` pair; `_words_center(words)` is shared with `native_first.upright_words_for_page`. Docstring now says the centre comes from the words' bbox and `page_rect` only switches rotation on. |
| T16 `_same_line`; docstring | DONE | `_same_line(a, b)` replaces both `w[5:7] == s[5:7]` sites; "PR #888 review" dropped, the 145-sign measurement kept. |
| T17 stale `native_first` comment | DONE | Now says the GH-916/GH-917 gate runs inside `plan_native_table` and the quarantine stays until the full re-audit (#917). Removed "still returned for the grid-order tests". |
| T18 | DONE (safe parts only) | `words`/`markdown` normalised once at the top of `plan_native_table`; `NativeTablePlan.action` is `PlanAction = Literal[...]`. The `isinstance(drop, dict)` check is left alone, as instructed. |
| T19 separator parsing | SKIPPED | Changes behaviour (`startswith("| ---")` misses `\|:---`). Filed as an issue, see the end. |

### Tests

| Item | Status | What |
| --- | --- | --- |
| T20 hand-built markdown | DONE, with a deviation | One `_md(header, rows, *, sep_width=None)`. The review expected the three 5-cell-header-over-6-cell-separator sites to change by one empty header cell; measured, that is NOT harmless: padding the header made the verifier stop exact-passing the fixture (`test_difference_pin_bare_sign_cell_with_source_contact` failed with `defer`). So the shape is preserved exactly: the sites that had a 6-wide separator pass `sep_width=len(HEADER) + 1`, the padded-header sites pass a padded header, and the range-hyphen site (5 and 5) is the plain call. The inputs are byte-identical to before. |
| T21 repeated helpers | DONE | `_gate(words, md) -> set[str]` (23 sites), `_data_row_faults(words, md, bound)` (promoted from `TestBlockInteriorColumnCounts._faults`, used at 5 sites), `_sign_word(...)` (3 sites), `BLANK` (7 sites), `WORD_H`, `len(HEADER) - len(c)`. |
| T22 review-round names | DONE | `TestSignDetachedRound2` -> `TestSignDetachedBinding`, `TestDataRowRound2` -> `TestDataRowSpan`, `TestLabelRowRound2` -> `TestLabelRowMatching`; "round 2: wiring pins" header, the Astra / cubic / commit-0e52ddd attributions and "GH-916 round 5" removed from comments. |
| T23 dead / brittle test code | DONE | Removed the `BornDigitalDetector` import + `is not None` assert in the resume helper; dropped the redundant `and plan.action != SHIP`; `_two_blocks` always takes a per-lane list; `TestLane()._run` is the module function `_run_lane`. |
| T24 shared fixtures | DONE | New `tests/native_table_fixtures.py`: the forecast grid (`HEADER`/`ROWS`/`COL_XS`, which replaces `_FORECAST_*`), `native_first_config`, `routed_decision` (replaces 3 route_page fakes), `flush_and_restore`, `place`, `forecast_pdf`, `rotated_forecast_pdf`, `dense_pdf`. `test_gh916` no longer imports a fixture from the rotated test module. **Deliberate hermeticity change:** the unified config adds `local_engine=EngineType.QWEN`, which `test_rotated_native_table_first.py`'s old `_config` lacked; CLAUDE.md (#841) requires it for any `process()` test. |
| T25 rotated test | DONE | `_FOMC_PDF` constant and one `requires_fomc_fixture` marker (was 4 path copies and 2 skipif blocks); the duplicate, unsorted `socr.core.result` imports are merged. |

### orchestrator.py (pipeline review)

| Item | Status | What |
| --- | --- | --- |
| P1 repeated `retained_prose_survives` | DONE | Confirmed pure first: `retained_prose_survives` -> `retained_prose_lines_to_keep` -> `_markdown_table_tokens` read only their arguments (no I/O, no mutation, no global state). The call at the `if not ...: return None` guard and the second one feeding `clear_ocr_enhancement` take identical arguments, so the second can only be True. Now `clear_ocr_enhancement=True` with a comment. |
| P8 split `_plan_native_table_first` | DONE | `_plan_rotated_native_table(state, page_num, ps)`, `_record_rotated_quarantine(state, page_num)` (beside `_record_native_ship_gate`), `_read_page_words(pdf_path, page_num)`. Naming deviation: `_read_page_words` returns `(words, line_dirs)` because both call sites read the line directions in the same `open_pdf` block; each caller keeps its own `except` and log text. |
| P9 kind constants | DONE | `ROTATED_QUARANTINE_KIND` in `tables/native_first.py`; the emitter and `_RESUME_REPLAYED` import it, and `SHIP_GATE_KIND` is imported at every site. The tests use the constants too. No string literal for either kind remains in `src/` outside the two definitions. |
| P10 `resume_restore_kinds` notes | DONE | `_RESUME_REPLAYED: dict[str, str]` (kind -> one-line reason with issue number); the method is `TABLE_LADDER_EVENT_KINDS | EQUATION_LANE_EVENT_KINDS | _RESUME_REPLAYED.keys()`. The lazily imported kind constants are now imported at module level (no cycle; the modules do not import the orchestrator). The multi-paragraph notes were condensed to one line each; the shared rationale (a resumed page is not re-processed, so nothing re-emits the event) is stated once above the dict. New `tests/test_resume_restore_kinds.py` asserts the returned frozenset equals the 37-kind set hard-coded from main at 3cf5f87 (generated by running the unmodified code, then diffed against the refactored output: identical). |

Not requested, so untouched: clean-pipeline items 2-7, 11-20 (items 12 and 17 explicitly skipped
because they change behaviour or routing), and the orchestrator mixin split.

## Proof

- Full suite, default `OLLAMA_HOST`, one detached `nohup` run to a log: **6033 passed, 2 skipped, 4 xfailed in 427 s**
  (6039 collected).
- Collected tests: 6037 on main 3cf5f87 (`pytest --collect-only`) vs 6039 here: equal plus the two added P10 tests.
- Corpus re-measure with frozen sources (`measure917.py` adapted: WT and output paths from the environment, `line_dirs=`
  passed by keyword for the post-#926 signature, new-fire dumps redirected out of the shared state dir): main source
  (a `git archive` of origin/main, `socr.__file__` asserted inside it) vs this branch, 35 rotated + 92 upright pages.
  The per-page record (verdict, predicates, old/new split, `dir_fault`, details) was compared for all 127 pages:
  **0 differences, the two JSON results are identical.** Totals on both: rotated 22/35 fire (foreign_direction 4,
  label_row_missing 15, data_row_missing 2, sign_detached 2, header_band_missing 1), upright 40/92 fire
  (data_row_missing 6, header_band_missing 17, label_row_missing 20, foreign_direction 2).
- `uvx ruff@0.16.0 format --check .`: all files formatted. `ruff check` on the five touched src files: 17 findings
  vs 21 on main (no new rule or location).

## T19, filed as an issue (not done here)

`native_first.py` reads markdown tables with its own hand-rolled parsing in two places:
`_markdown_table_tokens` skips separator rows with `startswith("| ---")` and `splice_cell_tokens` splits cells with
`strip("|").split("|")`. The verifier and the ship gate use `_MD_SEP_RE` and `_parse_output_row_cells`. The hand-rolled form
misses an aligned separator such as `|:---|`, so it can treat that row as a content row. Reusing the shared helpers would unify the
readings, but it changes which lines count as separators and therefore which tokens the retained-prose splice and the cell splice
see. It needs its own corpus measurement and tests, not a cleanup commit.
