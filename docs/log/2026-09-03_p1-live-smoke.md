# P1 live two-rung smoke — the ladder works end to end on the Mac

2026-09-03, 05:53–06:18 local. The one pre-flip item every GH-353 ticket deferred: a
full `process()` over a real document with both judge rungs live. Not a corpus run;
a four-page extract (pages 1, 7, 10, 12 of one 2002 working paper, chosen from the
2026-08-20 lane-comparison manifest for their tables), socr at `3c76ca7` (main after
P1 prep #553 and stage C #558), flag on for this run only:

```
socr process cp4.pdf --table-judge-ladder --write-manifest --verbose -o <scratch>
```

Environment: local `qwen3-vl:30b-a3b-instruct` for OCR (ollama, Apple Silicon),
rung 1 `glm-5.3-flash:cloud` through the local ollama daemon's cloud route, rung 2
`agy` 1.1.25. Content-free: identifiers, counts and dispositions only.

## Result

| table | rung trail | terminal |
|---|---|---|
| p2-t0 | ollama:glm-5.3-flash:cloud ok → gemini (agy) ok | REJECTED (content, not retryable) |
| p3-t0 | — | ACCEPTED |
| p3-t1 | — | ACCEPTED |
| p4-t0 | empty; `witness_scope: none` | UNVERIFIED |

- **Both rungs ran live and answered** on p2-t0, and the ladder took the rejection to
  its terminal: `table_ladder_rejected`, page disposition
  `model_output / native_table_distrust`, `table_ladder_disposition: table_rejected`,
  document status PARTIAL, exit code 1 — the fail-closed contract, surfaced at every
  level. Rung-2 identity is recorded as `executing: agy` (the fingerprint caveat from
  the design note stands: the binary, not the model, is what is known).
- **Two tables accepted**, one page with no table shipped as ordinary model output.
- **p4-t0 never reached a rung**: the witness had no crop, so the terminal is
  UNVERIFIED with no latch (correct — nothing would change on resume). The CLI and the
  note nevertheless call it "infra problem, retryable on resume". Filed as #560.
- **Latch behaviour on a real run:** no rung was unavailable, so
  `table_judge_retry_pending` / `_rungs` are absent from every sidecar and from the
  root entry — the sparse-key contract holds live.
- **Cost and time:** total cost $0.0002 (rung 1 cloud route); 25 minutes wall clock
  for four pages, dominated by the local 30B OCR reads and the page-judge timeouts on
  the nougat/gemini OCR rungs, not by the table ladder. Per-table ladder latency is not
  printed; adding it to the audit event is a small follow-up if the flip needs a number.

## What this settles, and what it does not

Settles: the two rungs are reachable and produce schema-valid verdicts on a real
document; the REJECTED terminal, its surfaces and the exit policy behave as designed;
the P1 latch keys stay sparse when nothing is unavailable.

Does not settle: the ¬S1 rate and the UNVERIFIED rate on a corpus (four pages is a
smoke, not a measurement); `agy` quota behaviour over a long document; the no-crop
wording (#560). The flip still needs the owner's three rulings (design log §"Panel and
synthesis") and, ideally, one corpus run with the flag on to get shipped rates for the
NATIVE_FALLBACK triggers (2026-09-03 measurement) and the ladder terminals.
