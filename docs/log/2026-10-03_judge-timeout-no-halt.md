# 2026-10-03 judge-timeout-no-halt (issue #987, branch fix/judge-timeout-no-halt)

## Problem

Measured in `~/.local/state/socr-housekeeping/withheld/` (11 re-OCR'd papers): 8 document halts
(`PARTIAL_SAVE_VLM_TIMEOUT`), every one preceded by a page-JUDGE timeout (`qwen3.8:27b`, 120 s),
never an OCR-rung timeout. About 57 table blocks lost to pages that got no model after the halt.

## Measurement (no models run)

- `_attempts_show_timeout` armed the halt on `judge_outcome == JUDGE_OUTCOME_TIMEOUT` (#713 r3) and
  on the substring "timeout", which a judge reason ("judge raised: page judge timeout ...") also
  contains. The canary (`probe_ollama_idle`) then ran on the OCR model.
- The judge (27B) and the OCR model (30B-A3B) take turns on one GPU. Ollama evicts a model when
  another needs the memory and the next request to the evicted one queues behind a reload.
  `docs/log/2026-09-16_221.md` already recorded a cold load of ~37.5 s against the 30 s canary
  default (`_CROP_DEADLINE_FLOOR_S`), and excused it because "the model is normally warm after a
  crop". That premise is false right after a judge call on a different model. So a 30 s canary
  cannot cover a cold load, and the failure follows a judge timeout by construction.

## Change

- `pipeline/orchestrator.py::_attempts_show_timeout`: attempts typed `JUDGE_OUTCOME_TIMEOUT` are
  skipped. A page whose OCR rung also timed out still arms the halt (that rung is its own attempt).
  The page itself still fails closed as before.
- `tables/extract.py`: `canary_deadline() = _CROP_DEADLINE_FLOOR_S + CANARY_LOAD_ALLOWANCE_S`, the
  allowance being `DEFAULT_READER_TIMEOUT_S` (120 s, the reader's existing read budget; no new
  number). Default `generation_timeout` of both canaries now uses it. A wedge never answers, so only
  the verdict on a real wedge is delayed, once.
- Halt surfacing unchanged.

## Tests

New `tests/test_gh987_judge_timeout_no_halt.py` (hermetic: ladder + `probe_ollama_idle` patched):
difference pin (judge timeout on p2 -> pages 3-4 run, no halt, canary not asked; provider timeout on
p2 -> halts after p2), mixed-attempt masking pin, cold-load default pin, slow-but-alive canary pin
(scaled constants). Updated to the new contract: `test_gh222_probe_host.py` (position test),
`test_gh713_round3_supersession_identity.py`, `test_gh221_generation_canary.py` (default timeout).

Mutations (external copy, `socr.__file__` canary asserted inside the copy):
- old predicate restored: 2 failed (difference pin, masking pin).
- `canary_deadline()` back to the floor: 2 failed (cold-load default, slow-but-alive).

## Notes

- #984's guard was not on this base; the full suite ran with `OLLAMA_HOST=http://127.0.0.1:1`.
- Not changed: a judge that is wedged still costs 120 s per page; this ticket only stops it halting
  the document.
