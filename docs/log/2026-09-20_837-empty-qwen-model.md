# GH-837 — an empty `qwen_model` must still load, and still pin

**Date:** 2026-09-20
**Branch:** `test/837-empty-qwen-model`
**Scope:** tests only; no production change.

## Why

`#836` (GH-834) made `PipelineConfig.from_file` reject a non-string `qwen_model`, and
deliberately exempted the empty string so a config file behaves like `--qwen-model ""`
on the command line — the channel parity GH-825 established. The production comment
stated that exemption. Nothing tested it.

An exemption with no test is a decision that can be reversed silently: tightening the
guard to `not isinstance(config.qwen_model, str) or config.qwen_model == ""` would have
load-rejected the empty string with the whole suite still green.

## What

Two controls in `TestGH834NonStringQwenModelIsRejectedAtLoad`, both pinning a difference
rather than an outcome:

- `test_an_empty_model_name_still_loads_and_still_pins` — `""` loads and
  `qwen_model_pinned` is `True`.
- `test_empty_and_non_string_are_treated_differently` — same loader, same key, two falsy
  values. `""` is a string and loads; `None` is not and raises. A guard written as a plain
  truthiness test (`if not config.qwen_model: raise`) satisfies every rejection case in the
  class and fails only this one, which is exactly the regression worth catching.

## Verification

- `tests/test_config_from_file.py` → 55 passed (53 before).
- Whole suite → 5718 passed, 4 xfailed. Reconciles: `main` collects 5716, this adds 2.
- `uvx ruff@0.16.0 format --check .` → 748 files already formatted.
- **Mutation**, in `/tmp/mut837` (a copy outside the repo, with a source canary asserting
  `socr.__file__` resolves inside that copy, and an uncapped `count(anchor) == 1` assertion
  before the edit): tighten the guard to also reject `""`, and exactly the two new tests
  fail — 54 passed, canary green, the six parametrised non-string cases untouched.

## Not done

Thinking-model (`qwen3-vl:30b`) refusal parity between the CLI and YAML channels remains
open, per the GH-834 discussion. It is a behaviour question, not a missing test.
