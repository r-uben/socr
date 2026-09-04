# TICKET-A1 — replay-binding harness

Branch `feat/vi-A1-replay-binding`, worktree
`/Users/rubenffuertes/repos/.worktrees/socr-vi-A1`, based on `main@0adf013`.

## Revision (second commit): candidate selection must not use bind() results

Review (cross-vendor reviewer, team-lead concurred) rejected the first cut
of the D3 fail-closed-marker fallback: it scored each cache candidate by
how well IT reproduced the recorded contradiction and picked the
best-scoring one. That lets the implementation under test (`bind()`)
choose its own input — a future `bind()` change (TICKET-A2's row-label
repair) could silently swap which candidate counts as ground truth,
invalidating the regression oracle A1 exists to provide.

Replaced with **provenance-only** selection
(`_select_candidate_for_table`): the winning candidate is identified from
the `table_binding_adjudicated` audit event recorded for that exact
`table_id` (its `engine` field), matched against the page's `cache/*.json`
entries by their own `engine` field. `bind()` is never called to decide
which candidate to use. Exactly one provenance match -> used. Zero or more
than one distinct candidate -> the row is `ReplayRow.unreplayable = True`
and `bind()` is never called for it at all (`replay_page` returns before
`replay_table`). Confirmed via a monkeypatch that raises if `bind` is
called (`test_ambiguous_provenance_is_unreplayable_and_never_calls_bind`).

Fixture extended with a third page (`doc00` p3) carrying two cache
candidates that both claim the same provenance engine — the ambiguous
case. On the real corpus, `doc01` p2 (the only fail-closed-marker case)
still has a single provenance-matching cache candidate, so it still
replays exactly as before.

## What changed

- `src/socr/benchmark/replay_binding.py` (new): given a frozen corpus
  directory (`in/<slug>.pdf` + `out/<slug>/<slug>/pages/*.json`), for every
  page sidecar carrying a `binding_adjudication`, re-derives the per-table
  witness (block markdown + located box) from the model's recorded
  candidate markdown using the CURRENT tree's `tables/witness.py` +
  `tables/locate.py`, re-runs `tables/binding.py:bind()` against the native
  words on the frozen PDF, and compares the resulting contradiction
  multiset (`kind`, `native_token`, `model_token`, duplicate counts
  preserved) against what the sidecar recorded. `socr-replay-binding
  <corpus-dir>` entry point.
- `pyproject.toml`: added the `[project.scripts]` entry.
- `tests/test_replay_binding.py` (new, 10 tests) + `tests/fixtures/replay_binding/`
  (new: `generate_fixture.py`, `corpus/in/doc00.pdf`, two sidecars, one
  cache entry).

## Two findings folded into the harness (not assumed from the ticket)

1. **`winning_output.text` is not always what `bind()` saw.** When a table
   later fails closed, `manifest.py` overwrites the shipped page text with
   the D3 fail-closed marker
   (`f"[page {page_num} failed: unverifiable table — see image]"`). This
   happened on the frozen corpus itself — `doc01` page 2 (`lifted`, 1 item)
   — so a naive "use `winning_output.text`" replay silently produces 0
   items on that row. The harness detects the marker
   (`_is_fail_closed_marker`) and falls back to the page's own
   `cache/*.json` route/extract entries, selected BY PROVENANCE (the
   `table_binding_adjudicated` audit event's `engine` field matched
   against each cache entry's own `engine` field) — never by scoring
   candidates against `bind()` (see the revision note above); the winning
   candidate is reported in `ReplayRow.note` so the substitution is never
   silent, and an ambiguous/missing provenance match makes the row
   `unreplayable` rather than guessing. Exercised in the fixture (pages 2
   and 3) and confirmed against the real corpus (`doc01` p2-t0: multiset
   still matches, 1/1, via its single provenance-matching candidate).
2. **The persisted `binding_adjudication[...].items[]` records do not
   carry `native_bbox`** (`adjudication.ContradictionItem.to_record`
   omits it) — confirmed by inspecting a frozen sidecar directly. Replay
   can therefore only compare `(kind, native_token, model_token)` as a
   multiset from the frozen side; label accuracy / crop coverage need a
   bbox and can only come from a FRESH re-bind (TICKET-A1b's job, once its
   hand-read label file exists). `replay_binding.py` reports these two
   fields as explicitly unavailable (`--labels` not supplied / no
   hand-read label for this table) rather than fabricating a score.

## Corpus output (unchanged tree, `~/venvs/socr/bin/python -m socr.benchmark.replay_binding ~/Data/socr/ladder-run2-2026-09-04`)

```
doc    page table_id  recorded  rec# fresh# match         disposition
---------------------------------------------------------------------
doc01  2    p2-t0     lifted    1    1      YES           ACCEPTED (binding lifted — GH-359 ruling 5 clamp released)
       note: winning_output.text is the D3 fail-closed marker; substituted cache/b4/b49dc7da2bd3019d7b4869e8e6f9e292229b5ae9913e9941241c36bc2a725500.json by provenance (engine='qwen')
doc02  3    p3-t0     held      3    3      YES           UNVERIFIED (binding held — GH-359 ruling 5 clamp applies)
doc02  4    p4-t0     held      4    4      YES           UNVERIFIED (binding held — GH-359 ruling 5 clamp applies)
doc03  1    p1-t0     held      2    2      YES           UNVERIFIED (binding held — GH-359 ruling 5 clamp applies)
doc04  3    p3-t0     held      1    1      YES           UNVERIFIED (binding held — GH-359 ruling 5 clamp applies)
doc05  1    p1-t0     held      6    6      YES           UNVERIFIED (binding held — GH-359 ruling 5 clamp applies)
doc07  1    p1-t0     held      5    5      YES           UNVERIFIED (binding held — GH-359 ruling 5 clamp applies)
```

7/7 rows, exact multiset match on every table (including the fallback
row), on the unchanged tree — satisfies the corpus-level Done-when. This is
NOT a pytest; it is reported here as requested.

Note: `shasum -a 256 -c SHA256SUMS` in the corpus directory reports one
`FAILED` line — `./SHA256SUMS` itself (a checksum file cannot correctly
list its own hash). All 152 data files under `in/` and `out/` pass.

## Tests

`PYTHONPATH=<worktree>/src ~/venvs/socr/bin/pytest tests/test_replay_binding.py -q`
— 11 passed (added `test_ambiguous_provenance_is_unreplayable_and_never_calls_bind`,
which monkeypatches `bind` to raise if called and asserts it is not, plus a
sidecar-bytes-unchanged check on the ambiguous page). Every test in the
file strips any PATH entry containing
`ollama` or `qwen-ocr` (autouse fixture) and asserts `shutil.which` returns
`None` for both, proving the module makes no provider call — it never
opens a network connection or shells out; the only I/O is the frozen PDF
and JSON sidecars.

`uvx ruff@0.16.0 format --check .` — clean (whole repo).

## Done-when line not fully satisfiable at A1

The ticket's Do section also asks for "label accuracy" and "crop
coverage" per table. TICKET-A1b (wave 2, depends on A1) is the ticket that
writes the hand-read label file these need
(`~/Data/socr/ladder-run2-2026-09-04/labels.json`, outside git); until
that exists there is nothing to score against. `replay_binding.py`'s
`ReplayRow.label_accuracy` / `.crop_coverage` fields are wired to consume
that file via `--labels` and currently report explicit unavailability.
This is not a corpus-level Done-when line (the actual A1 Done-when only
requires the 7-row multiset comparison, the hermetic delta test, and the
sidecar-bytes-unchanged assertion — all satisfied above) but is flagged
here since the Do section describes it as part of the contract.

## Files changed

- `src/socr/benchmark/replay_binding.py` (new)
- `pyproject.toml`
- `tests/test_replay_binding.py` (new)
- `tests/fixtures/replay_binding/generate_fixture.py` (new)
- `tests/fixtures/replay_binding/corpus/...` (new, generated fixture data)
- `docs/plans/verifier-independence/STATUS.md`, `TICKETS.md`
- this log
