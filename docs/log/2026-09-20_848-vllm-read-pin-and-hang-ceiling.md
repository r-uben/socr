# GH-848: pin `VllmTableReader.read` wiring, hang ceiling on the gh798 pin, fix a truncated quote

Three post-merge leftovers from #844 (routing the crop readers through
`run_killable`), all test/doc debt. No production code touched.

## 1. `VllmTableReader.read`'s own body was unexercised

`tests/test_vllm_table_reader.py` only called `_vllm_read_crop` directly, so
`read()`'s `CallSpec` wiring — the exact func string, the six-element arg
order, and the b64 image payload — could silently rot: revert `read()` to an
in-process `httpx.post`, scramble the arg order, or point `CallSpec` at the
wrong function string, and the suite stayed green.

Added `TestVllmTableReaderReadWiring::test_read_wires_call_spec_to_vllm_read_crop`,
which patches `socr.tables.extract.run_killable` itself, captures the
`CallSpec`, and asserts:
- `spec.func == "socr.tables.extract:_vllm_read_crop"` (exact string),
- `spec.args` is the six-element tuple `(base_url, model, api_key, prompt,
  image_b64, timeout)`, in that order,
- `base64.b64decode(image_b64)` round-trips to the real PNG bytes written to
  the crop file (not merely "a non-empty string").

The nine pre-existing `_vllm_read_crop` parsing tests are untouched.

### Mutation guard proof

Copied `src`, `tests`, and `pyproject.toml` to `/tmp/mut848` (outside the
repo). Confirmed the anchor line was present exactly once, uncapped, before
editing:

```
args=(self.base_url, self.model, self._api_key, self._prompt, image_b64, self.timeout),
count: 1
```

Added the new pin test there; ran `PYTHONPATH=/tmp/mut848/src pytest
tests/test_vllm_table_reader.py -q` — 9 passed (pre-mutation baseline, new
test included). Used a canary asserting `socr.__file__` resolves under
`/tmp/mut848` to confirm the copy, not the editable install, was under test
(canary removed before any further step, never committed).

Mutated `VllmTableReader.read` — scrambled the arg order
(`self.model, self.base_url, ...` instead of `self.base_url, self.model,
...`) — and reran:

```
FAILED tests/test_vllm_table_reader.py::TestVllmTableReaderReadWiring::test_read_wires_call_spec_to_vllm_read_crop
AssertionError: assert 'Qwen/Qwen3-V...-A3B-Instruct' == 'http://h:8000/v1'
1 failed, 8 passed, 5 warnings in 0.14s
```

The new pin caught the mutation; the eight pre-existing parsing tests stayed
green (they exercise `_vllm_read_crop` directly and don't see `read()`'s
wiring). Deleted `/tmp/mut848` afterwards.

## 2. Hang ceiling on the gh798 difference-pin

`tests/test_gh798_crop_reader_killable.py`'s current-path half called
`OllamaTableReader.read` in-process against the trickle server. If that
body is ever reverted to a direct `httpx.post` (no `run_killable`
boundary), the call wedges forever against the trickling peer — CI hangs
instead of failing, the worst failure shape.

Wrapped that half in a child process with `subprocess.run(..., timeout=...)`,
matching the shape `tests/test_gh172_killable_boundary.py` already uses in
`test_plain_httpx_trickle_defeat_measured_in_a_child`. The child imports
`OllamaTableReader` directly (via `PYTHONPATH` pointed at `src/`), calls
`.read()` against the trickle server, and exits 0/1/2 depending on outcome;
the outer `subprocess.run` ceiling (`_READER_TIMEOUT_SEC + _KILLABLE_GRACE_SEC
+ 5.0`) turns an unbounded hang into a prompt `subprocess.TimeoutExpired`
test failure.

### Ceiling-fires proof

Built a standalone demo (outside the repo, not committed) that reproduces
the pre-#844 body — a direct `httpx.post` against the same trickle-server
shape, no `run_killable` boundary — and ran it through the identical
`subprocess.run(timeout=...)` wrapper:

```
PASS: subprocess.TimeoutExpired raised after 14.00s (ceiling=14.0s) --
ceiling fired, bounding the wedge
```

Confirms the ceiling actually bounds a wedge rather than being decorative.

## 3. Truncated #843 quote

`docs/log/2026-09-20_798-crop-reader-killable.md` quoted #843's title as
"`_escalate_table_page`'s 120s deadline i[s...]" — truncated placeholder
text left on `main`. Fetched the real title via `gh issue view 843` and
replaced it: "…is shorter than the subprocess bound it abandons the thread
to, so the reported timeout understates the real one".

## Test result

- `tests/test_vllm_table_reader.py` + `tests/test_gh798_crop_reader_killable.py`
  + `tests/test_gh172_killable_boundary.py`: 16 passed.
- Full suite (`PYTHONPATH=/tmp/wt-848/src pytest tests -q`): 5726 passed,
  4 xfailed (360.9s).
- Collect-only reconciliation against a clean `origin/main` worktree
  (`b35f110`): 5729 collected there vs. 5730 here — delta of +1, matching
  the one new pin test added. No other file changed collection count.

## Format gate

`uvx ruff@0.16.0 format --check .` initially flagged
`tests/test_gh798_crop_reader_killable.py` (one multi-line f-string
join); ran `uvx ruff@0.16.0 format tests/test_gh798_crop_reader_killable.py`
and re-checked — exit 0, 754 files already formatted.

## Files changed

- `tests/test_vllm_table_reader.py` — new `TestVllmTableReaderReadWiring` class.
- `tests/test_gh798_crop_reader_killable.py` — subprocess hang ceiling on the
  current-path half.
- `docs/log/2026-09-20_798-crop-reader-killable.md` — fixed the truncated
  #843 quote.
- This log.

No production code changed, per the ticket's scope.
