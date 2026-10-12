# #1063: the maths readers reach a vLLM / OpenAI-compatible server

## Problem

Both maths readers POSTed to Ollama `/api/generate` only. On the Rorqual vLLM
deployment the corrupt-math recovery failed per call (21,398 `Connection refused`
lines, job 22889958) and the P4-R equation lane refused on purpose
(`EQUATION_LANE_BACKENDS`): 2,748 regions on 1,159 pages detected, none read
(run 22922695). #1080 made preflight report that as MISSING.

## Change

- `math/recover.py::latex_for_image_openai`: `/v1/chat/completions` counterpart of
  `latex_for_image`. Same prompt contract, same `clean_latex`, same failure contract
  (returns `""`; callers already record that). Image part first, as the table reader and
  the judge order it. `max_tokens` is `JUDGE_MAX_REPLY_TOKENS` and the timeout is
  `DEFAULT_READER_TIMEOUT_S`, both existing named constants; a reply with
  `finish_reason == "length"` is a cut fragment and is discarded (#1041's remedy, applied to
  this new call only). The Ollama calls are untouched, byte for byte.
- Orchestrator: `_maths_openai_served_model` is `resolve_qwen_intent`'s answer when its
  backend is OpenAI-compatible and `qwen_vllm_url` and the model id are set, else `None`
  (Ollama). It honours a pinned model (the table reader's #1073 does not).
  `_maths_openai_ocr(prompt)` returns an `ocr(png, *, model, host)` callable for the three
  existing injection points, or `None` so the default Ollama transport runs.
  - corrupt-math recovery (`_recover_corrupt_math_page`): corrupt-math prompt.
  - P4-R region lane (`_agentic_equation_region_page`): `EQUATION_LATEX_PROMPT`.
  - legacy clean-equation sidecar path: same.
  All three share the validation gates, presence guard, and sidecar attachment already
  downstream of the reader.
- `_equation_lane_backend_addressable` accepts an OpenAI-compatible backend for the QWEN
  rung when URL and model id are configured; everything else is refused with the existing
  reason text. `_equation_lane_serves` lets the served id match the lane model.
- Preflight `check_maths_readers`: on such a run it checks the lane models, then sends the
  existing 1-token generation canary (`probe_openai_server_idle`) on the served model.

## Decision: `math_model` / `clean_equation_model` on a vLLM run

Both default to the Ollama tag `qwen3-vl:30b-a3b-instruct`. On a vLLM run:

- the default tag, or the served id itself: call the served model (the only model that
  server has);
- any other value: refused, never replaced. Corrupt-math records
  `model_call_skipped` with `model call skipped: X is not served by the OpenAI-compatible
  server (it serves Y)...`; the region lane records `equation_region_reading_unvalidated`;
  preflight reports MISSING.

Cloud policy is evaluated first, unchanged: strict-local and a zero cost cap refuse a cloud
model before anything is sent, on either transport.

## Provenance

On a vLLM run the corrupt-math page records `provider_backend="openai-compatible"` and the
served id as `model_id` / `provider_model`. Ollama runs keep `ollama-compatible` and the tag.

## Failures

Server down, HTTP error, malformed or truncated reply: `""`, as on Ollama. Corrupt-math
regions become `[corrupt equation unresolved: ...]` with the crop retained; the region lane
records `equation_region_reading_unvalidated` and sets `equation_lane_retry_pending`. No new
fallback.

## Tests

`tests/test_gh1063_maths_readers_openai.py`: a real local HTTP server speaking both APIs.
Pins the difference (same page, Ollama fake vs vLLM fake, identical page output), the Ollama
request shape, the pinned model, strict-local, refusal of a foreign model name, server error,
truncation, and preflight both ways. Three tests in `test_equation_lane_pipeline_p4r.py`
that used `qwen_backend=vllm` as the example of an unaddressable backend now use `gemini`.
