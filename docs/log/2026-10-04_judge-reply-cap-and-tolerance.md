# 2026-10-04 — page judge: bounded reply, and a tolerance clause in the prompt

Follows `2026-10-04_vllm-judge-image-order.md` (#1038).

## 1. Bounded reply (a bug)

Neither judge backend capped its reply length. On HPC job 682725 one vLLM
reply looped on a single issue for 27,201 tokens until the 32k context ran out
(`finish_reason: length`). The JSON was unparseable, so it failed, but only
after minutes of GPU. A shorter loop could stop inside the cap of a
fragment that `_extract_json`'s first-`{`-to-last-`}` fallback reads.

- `JUDGE_MAX_REPLY_TOKENS = 2048` (`socr/judge/judge.py`). Derivation: the
  longest complete verdict in 151 recorded vLLM replies was 645 tokens (a
  17-issue rejection); every accept was 34. 2048 is over 3x the maximum.
- vLLM sends it as `max_tokens`; Ollama as `options.num_predict`.
- A reply the server marks as cut by the limit (vLLM `finish_reason ==
  "length"`, Ollama `done_reason == "length"`) raises
  `JudgeReplyTruncatedError` before any parsing. `route_page` records that as
  `JUDGE_OUTCOME_EXCEPTION`, unaccepted, the existing judge-failure path. It is
  not a `TimeoutError`, so it cannot license #713's stand-in.

## 2. Tolerance clause in `prompts/judge_page.md`

Added: do not count text running across the page boundary, running heads and
feet, page numbers and download stamps, or a table with no printed row labels
as defects. Kept and made explicit: reject wrong or missing numbers and signs,
missing or extra rows and columns, shifted values, dropped body text, and
invented content.

The prompt is data, outside `_socr_source_digest`'s `.py` hash, and was in no
fingerprint field. `page_judge_prompt_digest` now enters `_run_fingerprint`
whenever a VLM judge runs, so this edit (and any later one) invalidates
resume.

## Measurement

Same 18 Forsythe–Lundholm pages and readings as #1038; 300 dpi, image first,
temperature 0. Old prompt = origin/main, new prompt = this branch. Six safety
readings carry one injected defect each:

| case | page | defect |
|------|------|--------|
| dropped_minus | 11 | `TE=1-(2n/N)` -> `TE=1(2n/N)` |
| changed_number_table | 6 | a table cell 160 -> 610 |
| changed_number_prose | 40 | a reference year 1967 -> 1976 |
| shifted_row | 22 | table values rotated one row down against labels |
| deleted_row | 8 | one table row removed |
| deleted_row_pair | 26 | one two-line row block removed |

### Results

Both stacks judged the identical 18 readings and 6 injected readings. HPC:
Qwen3-VL-30B-A3B-Instruct via vLLM (job 682753), 2 reps. Mac: `qwen3.8:27b`
via Ollama (`JUDGE_MODEL_DEFAULT`), 1 rep, one call at a time.

| stack | prompt | clean accepted | injected defects rejected |
|-------|--------|----------------|---------------------------|
| HPC vLLM | old | 11/18, 12/18 | 4/6 in both reps; ACCEPTED p40 changed_number_prose and p8 deleted_row; p6 truncated |
| HPC vLLM | new | 11/18, 12/18 | every non-truncated case rejected in both reps; p6 truncated both reps, p8 truncated rep 1 |
| Mac Ollama | old | 18/18 | 6/6 |
| Mac Ollama | new | 18/18 | 6/6, no truncation |

Reading the table:

- The new prompt does NOT raise clean-page acceptance on HPC (11/18 and 12/18,
  identical to the old prompt). Its value is safety. On HPC the old prompt let
  two of six real defects through (p40 `changed_number_prose`, a year 1967 ->
  1976, and p8 `deleted_row`, in both reps); the new prompt let none through.
- A truncated reply (p6 in both reps, p8 in rep 1 on HPC) counts as a judge
  failure and is never accepted, so it is fail-closed. The 2048 cap does
  truncate on long defect lists; those pages are rejected, not accepted, at the
  cost of a retry/escalation.
- The Mac judge already accepts all 18 clean pages and rejects all 6 defects
  under both prompts; the prompt change is neutral there.
- Caveat: one rep on the Mac, two on HPC, one document.

