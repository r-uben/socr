# 2026-09-20 — GH-798: make the crop-reread reader call killable

## What changed

`socr/tables/extract.py` — `OllamaTableReader.read` and `VllmTableReader.read`
now cross a `run_killable` process boundary instead of making a raw
`httpx.post` in-process, mirroring the GH-172/#796 fix already applied to
`OllamaVisionJudge.judge()` (`judge/ollama_judge.py`).

- Extracted the httpx call from each reader's `read()` into a new top-level
  function (`_ollama_read_crop`, `_vllm_read_crop`) so it can be targeted by
  a picklable `CallSpec`.
- `read()` now builds a `CallSpec` and calls `run_killable(spec,
  timeout=self.timeout)`.
- `_read_with_deadline`, `TableCropExtractor`, `_CropTimeoutError`
  (signature unchanged: `(deadline, page_num)`), `crop_wall_clock_deadline`,
  and the two GH-221 generation-canary probes
  (`_ollama_generation_canary`/`_openai_generation_canary`) were **not**
  touched. `KillableTimeoutError` is a `TimeoutError` subclass, and since
  Python 3.11 `concurrent.futures.TimeoutError is TimeoutError`, so
  `_read_with_deadline`'s existing `except concurrent.futures.TimeoutError:`
  clause already catches it — the fix required zero changes at that layer.

Tests:
- `tests/test_gh798_crop_reader_killable.py` (new) — the "pin a difference"
  proof: a loopback trickle server (one byte/0.2s, defeating httpx's
  per-chunk read timeout — same measurement shape as
  `docs/log/2026-09-17_172-design.md`). In one process: (1) a reconstruction
  of `OllamaTableReader.read`'s pre-fix body, run through the same
  `ThreadPoolExecutor` + `cancel` + `shutdown(wait=False)` wrapper
  `_read_with_deadline` uses — the abandoned worker thread is shown still
  blocked on the peer after the wrapper gives up; (2) the real, current
  `OllamaTableReader.read`, called directly — raises `TimeoutError` within a
  bounded time.

  Note on where the test is pinned: `_read_with_deadline`'s own
  `ThreadPoolExecutor` wrapper already raises `_CropTimeoutError` on ANY
  timeout, killable or not (it only *abandons* the thread — see its own
  docstring). Asserting that exception through that wrapper would pass
  identically before and after this fix and would not be a guard at all, so
  the test calls `OllamaTableReader.read` directly instead, which is the
  layer that actually changed.
- `tests/test_vllm_table_reader.py` — 2 existing tests
  (`test_posts_openai_multimodal_and_parses_choice`,
  `test_empty_choices_does_not_crash`) patched `socr.tables.extract.httpx.post`
  in-process and called `VllmTableReader(...).read(...)`; that patch cannot
  reach the spawned child `run_killable` now dispatches to. Rewrote both to
  call the extracted `_vllm_read_crop(...)` directly, the same pattern
  already used for `judge/ollama_judge.py:_post_generate`.

## Step 0 (scope), recorded for continuity across GH-172/GH-798

Four sites were named across this and the prior ticket as reachable through
the route/judge/escalate/reread machinery. Only one was genuinely
unkillable:

| Site | Verdict | Why |
|---|---|---|
| `route_page` (agentic.py) | already bounded | all 7 registered CLI engines go through `BaseEngine.process_pages`'s real `subprocess.run(..., timeout=...)`; the httpx-based engines (`VLLMEngine`/`DeepSeekVLLMEngine`) are `BaseHTTPEngine` subclasses never reachable from the agentic ladder (`engines/registry.py`'s `_ENGINES` dict has no entry for them) |
| `_TimeoutJudge.assess` (orchestrator.py) | already bounded | wraps `OllamaVisionJudge.judge()`, which was fixed in #796 to cross `run_killable` itself |
| `_escalate_table_page` (orchestrator.py:5719) | already bounded, but see below | wraps `run_provider`, which dispatches through the same CLI-subprocess `_run_engine_on_pages` path as `route_page` |
| `_read_with_deadline` (tables/extract.py) | **genuinely unkillable — fixed here** | `OllamaTableReader.read`/`VllmTableReader.read` made a raw `httpx.post` with no process boundary at all |

## Deferred: `_escalate_table_page`'s deadline mismatch — explicitly out of scope

Filed by the team lead as **#843** ("`_escalate_table_page`'s 120s deadline
i[s...]"). Not touched here, per instruction. What I found, and how I found
it (stated plainly — this was **read from source, not reproduced live**):

- `_escalate_table_page`'s own `ThreadPoolExecutor` wrapper waits up to
  `self.config.escalation_timeout_sec` (`core/config.py:321`, default
  `120.0`) before abandoning the future.
- The `run_provider` closure it calls passes
  `subprocess_timeout_sec=provider_timeout.get(profile.engine)` —
  `route_page`'s own **per-engine soft deadline**, not
  `escalation_timeout_sec` (`orchestrator.py:8558`).
- So the subprocess underneath can be killed well before the escalation
  wrapper's own 120s bound expires (or, depending on config, after — the two
  numbers are independent and nothing keeps them in sync).

This is a **timing mismatch between two already-bounded numbers**, not an
unkillable call — it doesn't defeat process-group kill the way GH-172/GH-798
did. I have not measured a live divergence (e.g. instrumented an actual run
where the mismatch changed observed behaviour); this is inferred from
reading the two call sites and `core/config.py`'s default. Left for #843.

## Test result

- `tests/test_gh798_crop_reader_killable.py`: 1 passed.
- `tests/test_vllm_table_reader.py`: 8 passed (2 rewritten).
- Full suite (`~/venvs/socr/bin/pytest tests/ -q`), reconciled by measuring
  the clean `origin/main` baseline in this exact worktree (`git stash` /
  `stash pop`, since this is a dedicated worktree, not the shared checkout):
  clean `origin/main` (205eeb4) = 5721 passed, 4 xfailed; this branch = 5722
  passed, 4 xfailed. Delta: **+1**, exactly the one new test added. (The
  "5718" figure quoted earlier in the ticket thread was stale — this
  worktree's own measured baseline is 5721.)
- `uvx ruff@0.16.0 format --check .`: clean (2 files reformatted before the
  check — the new test file and the edited `test_vllm_table_reader.py`).

## Mutation-guard exercise (GH-798 discipline: a guard must be shown to fail)

- Copied `src/`, `tests/`, `pyproject.toml` to `/tmp/mutant-798` (outside the
  repo).
- Added a canary test (`test_zzz_canary_mutant_source.py`) asserting
  `os.path.realpath(socr.__file__)` resolves under the scratch copy — passed
  (2 passed) before mutating, confirming the copy's own source is what runs
  under `PYTHONPATH=/tmp/mutant-798/src:/tmp/mutant-798/tests`.
- Asserted the anchor block (`OllamaTableReader.read`'s `run_killable` call)
  occurs **exactly once**, uncapped, before mutating — `count == 1`, per the
  guard-mutation discipline (an uncapped count catches an anchor that no
  longer exists or that duplicated).
- Reverted `OllamaTableReader.read` to call `_ollama_read_crop` directly (no
  `run_killable`) — the pre-fix body.
- Re-ran `test_gh798_crop_reader_killable.py` against the mutant: it **did
  not complete**. The process was killed after it exceeded any reasonable
  bound (exit 144, i.e. terminated), which is the strongest possible red
  signal — the mutated code reproduces the exact wedge the fix exists to
  prevent (the worker thread blocks on the trickling peer forever, with no
  process boundary to kill it). The guard is shown to guard.
