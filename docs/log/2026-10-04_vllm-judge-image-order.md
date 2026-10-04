# 2026-10-04 — vLLM page judge sends the image after the text

## Symptom

HPC job 682581 (Qwen/Qwen3-VL-30B-A3B-Instruct on vLLM as both OCR engine and
page judge, `--judge-vllm-*`) accepted the qwen reading on 7 of 40 pages of the
Forsythe–Lundholm smoke paper. A Mac run of the same paper with the Ollama judge
accepted 23.

The "empty reason" in the report was a different field: `winning_output.judge_reason`
is empty because the winner on those pages is the native marker. Each qwen
attempt's `attempts_summary[].rejection_reason` carries the judge's issues.

## What was ruled out

- **Parsing.** Raw `/v1/chat/completions` bodies (job 682679) are bare JSON
  objects, `finish_reason: stop`, no fences, no thinking text. `parse_verdict`
  reads them correctly. Dropping `response_format` changes nothing.
- **The readings.** Cached HPC and Mac qwen readings are near-identical; page 35
  is byte-identical.
- **The model.** On the Mac, `qwen3-vl:30b-a3b-instruct` through Ollama (the
  same model as on HPC) accepted pages 35, 27 and 40 using the HPC readings
  verbatim. vLLM rejected all three.

## Cause

`_post_chat` built the user message as `[text, image_url]`. Ollama's qwen3-vl
renderer (`model/renderers/qwen3vl.go`) emits the vision tokens before the
message text. So the Ollama judge sees image→prompt and the vLLM judge sees
prompt+transcription→image.

## Measurement (job 682694, 18 pages, 300 dpi, same readings, temperature 0)

| order        | accepted |
|--------------|----------|
| text first   | 3 / 18   |
| image first  | 11 / 18  |

13 pages changed verdict: 12 rejected→accepted and 1 accepted→rejected (p14).
Image order does not explain the whole gap to Ollama. Image resolution is a
second factor: at 150 dpi, text-first accepted pages 27 and 40 as well. That
remainder is not addressed here.

## Not fixed here (noted)

- `math OCR call failed: Connection refused`: math recovery calls Ollama,
  which is absent on HPC.
- Page 36's judge call ran past the 120 s killable deadline.
