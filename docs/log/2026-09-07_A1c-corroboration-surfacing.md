# 2026-09-07 — TICKET-A1c: corroborated page surfaces at every level; resume contract

Issue: #641. Depends on TICKET-A1b (#634, merged to `main`). Branch:
`feat/641-corroboration-surfacing`, cut from `main@abb3d9b`. Worktree:
`/Users/rubenffuertes/repos/.worktrees/socr-a1c`.

## What changed

`src/socr/core/manifest.py`:

- New `SelectionProvenance.STRUCTURE_CLASS_GRID_CORROBORATED` member (R7: a new ending needs
  a new tag; reusing `STRUCTURE_CLASS_GRID_FLAGGED` for this ending failed the AST-verified
  R7 structural test — two returns must never share one tag, even when both map to the same
  `PageDisposition`). Mapped to `PageDisposition(MODEL_OUTPUT, STRUCTURE_CLASS)`, the same
  disposition as the ordinary passing/flagged grid endings — this is a distinct *ship reason*
  (a rescued fallback candidate whose header binding was never checked), not a distinct
  disposition.
- The corroboration-fallback return (A1b's `structure_class_grid_corroboration(p)` branch,
  gated by `_reaches_structure_class_branch(p)` and an empty `_strict_grid_authored_pool`) now
  ships `status=WARNING`, `audit_passed=False`,
  `failure_mode=FailureMode.HEADER_BINDING_UNVERIFIED` **unconditionally** — regardless of the
  winning candidate's own `audit_passed` — because A1a's row check only ever verifies ROW
  shape (ordered-number reproduction against native), never HEADER/column binding. A1b's own
  review flagged this gap: a candidate whose own `audit_passed` happened to be `True` shipped
  through A1b's code as an undemoted `SUCCESS`, indistinguishable from an ordinarily-verified
  S1 case (i) winner. A1c closes it.
- The demoted `PageOutput` carries a new `table_corroboration` dict:
  `{engine, bound, total, share, extra_numbers, skipped_native_rows, unbound_rows,
  corroboration_region, coverage_share, header_text}`, sourced from A1a's `RowCorroboration`
  plus the region/coverage values A1b's `structure_class_grid_corroboration` already computed.
- Demotion is by **status**, never by flipping `audit_passed` to silently discard the page in
  `_phase_assemble` (per #259's memoried trap — `audit_passed` selects the assemble winner,
  it is not a "flag this page" switch).

`src/socr/core/result.py`:

- `FailureMode.HEADER_BINDING_UNVERIFIED` — new member.
- `PageOutput.table_corroboration: dict | None = None`.
- `to_dict()` omits the `table_corroboration` key entirely when `None`, unlike every other
  field on this dataclass (all of which are serialized unconditionally, e.g. `rejection_class`
  at its `""` default). See "Root-cause correction" below for why this asymmetry is
  deliberate. `from_dict()` already tolerates the key's absence via `d.get(...)`.

`src/socr/pipeline/orchestrator.py`:

- `header_binding_unverified_pages` bucket, derived directly from `PageOutput.failure_mode`
  on `pre_records` (not from the disposition-bucket machinery — every corroborated winner
  already shares `structure_class_model_pages`'s disposition with an ordinary S1 case (i)
  winner, so a disposition-keyed bucket can't isolate it; this is a narrower,
  failure-mode-keyed subset for its own CLI line).
- CLI summary line: `"N page(s) shipped a corroborated table with header binding unverified:
  [...]"`.
- `src/socr/cli.py` needed no edit — confirmed by grep that all CLI summary printing for this
  family of buckets already lives in `orchestrator.py`; `cli.py` has no duplicate logic.

## Resume contract

`_load_terminal_page`'s per-page ledger gate requires terminal SUCCESS + `audit_passed`
before skipping a page on resume. A `HEADER_BINDING_UNVERIFIED` page ships `status=WARNING`,
`audit_passed=False` — it fails that gate unconditionally, so it is always reprocessed on
resume, never skipped. Pinned directly:
`test_a1c_header_binding_unverified_surfacing.py::...pin_the_difference` (positive case:
`_load_terminal_page` returns `None` for the corroborated sidecar) plus a reverse-regression
test in the same file (an ordinary clean SUCCESS page must still be resume-skippable, to catch
a fix that made the gate refuse everything).

Bytes are not bit-stable across runs for this failure mode by construction: the row
corroboration record is re-derived from a fresh route/judge each run, so a resumed run always
recomputes it rather than replaying a stale one — there is nothing cached to replay.

## Test-file edits required by the new enum member (R7 + one other hardcoded count)

- `tests/test_r7_winner_kind_tags.py` — the AST-derived structural invariant. Counts bumped
  15→16 returns, 16→17 unique tags (×2 occurrences), 16→17 total `SelectionProvenance`
  members; `STRUCTURE_CLASS_GRID_CORROBORATED` added to both `expected_multi_dispositions`
  and `expected_multi_reasons` for the `STRUCTURE_CLASS` group it joins. The per-disposition
  and per-reason bucket counts (`single_disposition_count == 12`, `single_reason_count == 9`,
  `len(by_disposition) == 14`, `len(by_reason) == 12`) are unchanged, since the new member
  joins an existing multi-member group rather than creating a new one.
- `tests/test_p6_disposition_contract.py` — a second, independent hardcoded
  `assert len(list(SelectionProvenance)) == 16` (a Stage A/B-era test, unrelated to R7's file)
  also needed bumping to 17, with a docstring noting the "16" in the test's own name is now
  historical and pointing at `test_r7_winner_kind_tags.py` as the live source of truth.
  Confirmed via grep this is the *only* other hardcoded `SelectionProvenance` count in
  `tests/`.

## Root-cause correction: the P6 golden-fixture breakage was NOT a corroboration-rescue defect

Adding the new field first broke 8 tests across `tests/test_p6_stage_ab_difference.py` and
`tests/test_p6_stage_c_difference.py` (the byte-identity/exact-delta oracle for the 12-page
synthetic P6 corpus). My first hypothesis to team-lead — that pages 8/9 were being genuinely
"rescued" by A1b's corroboration fallback and demoted by the new code — was **wrong**. Direct
verification (`structure_class_grid_corroboration(p)` called on all 12 fixture pages, on both
`main` and this branch) returns `None` for every page on both trees: **no page in the P6
corpus exercises the corroboration branch at all.** There is nothing to verify for per-page
rescue correctness (no `bound/total/extra_numbers/skipped_native_rows/unbound_rows` data to
report — the scenario team-lead's conditional approval asked me to check does not occur in
this corpus), so that precondition is moot.

The actual cause: `PageOutput.to_dict()` was unconditionally emitting
`"table_corroboration": self.table_corroboration` (`None` on every non-corroborated page,
following the pattern every other field in the dataclass uses). This dict is the exact input
to the content-addressed cache hash (`_page_blob_key` / `blob_hash`, `src/socr/core/cache.py`)
used as both `page_fingerprint` and manifest `blob_ref`. Adding *any* unconditional key to
that dict changes the hash of *every* page, corroborated or not — confirmed by diffing
`markdown` (byte-identical across the whole corpus) against `manifest`/`sidecars`
(`blob_ref`/`page_fingerprint` differ on all 12 pages) between `main` and this branch.

Team-lead's correction, which is what shipped: this is not cosmetically harmless. The same
hash feeds `_load_terminal_page`'s resume gate on *every real corpus*, not just the P6
fixture — the fed-01 corpus alone has 766 documents. Shipping the unconditional key would
have made every already-terminal page in every existing corpus look "changed" on the next
resume and be silently reprocessed. The fix is at the source, not the tests: `to_dict()` now
omits the `table_corroboration` key entirely when `None` (the one field on this dataclass that
does NOT follow the "serialize unconditionally at default" pattern, documented inline with the
reason). A non-corroborated page's serialized bytes are therefore byte-identical to
pre-#641 — confirmed: all 49 P6 tests in both files now pass **with zero edits** to
`tests/p6_stage_c_oracle.py`, `test_p6_stage_ab_difference.py`, or `test_p6_stage_c_difference.py`.
Touching `tests/p6_stage_c_oracle.py` as an ownership exception, which I had asked for and
team-lead had conditionally granted, turned out to be unnecessary — the file is untouched.

Checked whether any other field added by this ticket has the same effect: the only other new
surface is `SelectionProvenance.STRUCTURE_CLASS_GRID_CORROBORATED` and
`FailureMode.HEADER_BINDING_UNVERIFIED`, both string enum *values* slotted into existing
fields (`status`/`failure_mode`/manifest `disposition`) — these change a page's hash only when
that page's `status`/`failure_mode` actually changes, which is the correct, intended
resume-invalidation behaviour (a page that now ships differently *should* look changed).
Grepped `table_corroboration` across `src/`: it appears only in `result.py` and `manifest.py`
— no other unconditional new dict key was introduced anywhere else in this diff.

Two new tests pin this directly in `test_a1c_header_binding_unverified_surfacing.py`:
`test_to_dict_omits_table_corroboration_key_when_unset` (asserts the key is *absent*, not
`None` — the two are not byte-identical) and `test_to_dict_carries_table_corroboration_when_set`
(the corroborated round-trip).

## File-ownership note

`src/socr/core/manifest.py` was edited by both A1b and A1c (sequential, not concurrent —
A1b merged to `main` before this branch was cut). No conflict.

## Test results

- `PYTHONPATH=.../src ~/venvs/socr/bin/pytest tests/test_a1c_header_binding_unverified_surfacing.py tests/test_r7_winner_kind_tags.py tests/test_p6_disposition_contract.py tests/test_p6_stage_ab_difference.py tests/test_p6_stage_c_difference.py -q`
  → **86 passed**.
- Full suite, foreground: `PYTHONPATH=.../src ~/venvs/socr/bin/pytest tests/ -q`
  → **4279 passed, 4 xfailed**, 201.90s.
- `uvx ruff@0.16.0 format --check .` → `592 files already formatted`.

## Live verification (real Ollama, outside the synthetic P6 corpus)

Pending — run against `~/Data/socr/census-ecb-2026-09-06/in/ecb-reports-2003-report-p80-82.pdf`
and `ecb-meetings-2021-economic_bulletin-p127-129.pdf`, scored against `pdftotext -layout`
with the census scorer, to confirm ≥95% numbers shipped on any page tagged
`header_binding_unverified`. Results to be appended here once both runs complete.
