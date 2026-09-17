# GH-819: resume drops three `_agentic_native_page` audit kinds

## Defect

Three kinds emitted by `_agentic_native_page` were absent from
`UnifiedPipeline.resume_restore_kinds()`:

- `native_encoding_hygiene_suspect` (#136)
- `native_unrecovered_symbol_glyphs` (#217)
- `possible_table_structure_not_reconstructed` (GH-64)

`_agentic_native_page` only runs for a page NOT skipped as terminal on resume
(`_phase_agentic`'s `if resumed is not None: ...; continue` gate at
`orchestrator.py:8701-8706` short-circuits it). A resumed terminal page
therefore never re-emits these events, and — since they were also missing
from the replay allowlist — `_restore_terminal_page_state`'s filter at
`orchestrator.py:12235` (`resume_restore_kinds()`) dropped them on restore.
The sidecar (`pages/NNN.json`) kept the record; `audit_log.json` and the CLI
audit line lost it.

## Category measurement (per kind)

All three were checked the same way: grepped their sole emission site in
`orchestrator.py` and confirmed it is inside `_agentic_native_page` (lines
9776, 9800, 9826), which is called only from the `elif is_native:` branch of
`_phase_agentic`'s per-page loop (line 8744) — downstream of the resume-skip
`continue` at line 8706. None of the three are emitted anywhere else. They
are therefore all in the "lost on resume, must be allowlisted" category, not
the "re-emitted every run" category `orphan_word_dropped` sits in
(`_phase_analyze`, called unconditionally at `orchestrator.py:1466`, before
the per-page resume loop at line 1468 — every run, resumed or not).

## Fix

Added the three kinds to `UnifiedPipeline.resume_restore_kinds()` with a
comment following the `equation_sidecar_skipped_no_page_output` (#157)
precedent's pattern: names the emitting phase, the resume-skip asymmetry,
and why each is a standing property of the page's source rather than of the
run that noticed it.

## Test

`tests/test_gh819_native_audit_resume.py` drives the real machinery: calls
`_agentic_native_page` to emit the events, `_flush_page_sidecar` to write a
real `pages/00001.json`, and `_restore_terminal_page_state` to read it back
on a fresh `DocumentState` — a genuine flush/restore cycle, not a hand-built
sidecar dict. Assertions pin the event COUNT after that cycle (1, not
membership) for all three kinds; a control kind
(`table_wrapped_label_merged`, already allowlisted) is carried alongside and
must also survive; `orphan_word_dropped` is carried alongside and must NOT
survive (it would double-count against `_phase_analyze`'s own re-emission,
which this resume path deliberately does not simulate); a clean page (no
flags set) must resume with none of these events; and a direct membership
check on `resume_restore_kinds()` is kept alongside the count assertions,
never instead of them.

## Mutation round

Removed the added `| {"native_encoding_hygiene_suspect", ...}` block (kept
the GH-819 comment in place to prove the harness isn't just matching the
comment), abort-if-anchor-missing guarded via an uncapped `str.count(...)
== 1` check before mutating. Reddened:

- `test_lost_kinds_replay_exactly_once_after_flush_and_restore` (0 replayed
  vs expected 1, for all three kinds)
- `test_all_three_kinds_are_members_of_resume_restore_kinds`

Stayed green: `test_control_kind_still_survives_resume`,
`test_analyze_phase_kind_is_not_double_counted`,
`test_page_carrying_none_of_these_kinds_resumes_unchanged`,
`test_run1_actually_emits_all_three`. Reverted via a second guarded
str-replace (same abort-if-missing pattern); re-ran the file, 6/6 green.

## Full suite

Baseline: `git archive 64c59b5` into `/tmp/gh819-baseline`, ran with
`PYTHONPATH=/tmp/gh819-baseline/src ... pytest tests -q -o pythonpath=src`.
Canary: `socr.__file__` resolved to the archive copy in both trees before
running. Result: **5660 passed, 1 skipped, 4 xfailed** (5665 total). The 1
skip is `test_gh592_scoped_positional_emission.py`, a documented
archive-artifact skip (CLAUDE.md), not a regression.

Worktree (with fix): **5667 passed, 0 skipped, 4 xfailed** (5671 total).
5671 − 5665 = 6, exactly the new test file's count; the skip→pass on
`test_gh592...` is the archive-vs-live-worktree difference already
documented, not caused by this change.

## Lint

`uvx ruff@0.16.0 format --check .` — clean after one `ruff format` pass on
the new test file (line-length wrap on `_resume`'s signature).

## Disagreement with the framing

None. The ticket's category test (does the emitting phase run on a resumed
terminal page?) applied cleanly to all three kinds with no ambiguity — all
three share one emission site, so there was no need to split them across
categories.
