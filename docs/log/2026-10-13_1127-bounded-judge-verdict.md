# #1127 — a schema-bounded page-judge verdict, so the reply stops running past the token cap

What shipped: the verdict JSON schema (Round 3), measured on the cluster below. The one-retry design in the first sections is HISTORY: it was parked in Round 2 (the retry never fired in production) and is not on this branch.

**Branch** `fix/issue-1127` · PR #1133 · HEAD `d878b85` plus this round.

## The defect

Corpus runs `t3_22922695` (280 pages) and `t3_23117187` (274 pages) lose the model
reading to `JudgeReplyTruncatedError` at `JUDGE_MAX_REPLY_TOKENS=2048`. Fail-closed
→ `page_judge_exception`. 104 pages truncate in both runs; 176 / 170 are one-off.
The cap stays: a complete 17-issue reject was 645 tokens; an uncapped reply looped
to 27,201.

`JudgeReplyTruncatedError` is a `RuntimeError`, not a timeout. Only a timeout may
license #713's credentialed stand-in.

## Measure (local Ollama; cluster vLLM not measured)

Seed 1127. Local judge `qwen3-vl:30b-a3b-instruct`. Probe: 1-token generation
available in 14.8s. Same page image (300 dpi) and the qwen cache candidate, two
calls each, 20 recurring + 20 one-off (80 calls). Observation timeout 240s
(production Ollama judge budget is 120s). Seconds/call: recurring mean 42.7
(min 1.1, max 121.3); one-off mean 44.3 (min 1.1, max 111.7). No call errors.

| group | first call truncated | second completed given first truncated | both truncated |
| --- | --- | --- | --- |
| recurring (20) | 6 / 20 | 3 / 6 | 3 |
| one-off (20) | 7 / 20 | 1 / 7 | 6 |

Recurring pages do **not** truncate on every local call (first-call rate 30%).
The three recurring both-truncates (ferrero appendix p26, hazell p57, forney
p14) and the one-off both-truncates loop the `issues` array: one complaint per
table cell / equation term / missing caption, restated until `eval_count=2048`.
That is a real runaway, not a fluke cut. Cluster vLLM (the backend that produced
the corpus losses) was not re-asked here.

## Why one retry

A second ask recovered 3 of 6 recurring first-call cuts and 1 of 7 one-off cuts.
The rest repeated the issues-list runaway. A third ask is not licensed by that
rate and would not bound GPU. A timeout is a different type and must not enter
this path (#713). No new named constant: the bound is the integer 1 in the
helper, documented here.

## What was built

`VLMPageJudge._judge_with_one_truncation_retry`: one extra `judge()` on
`JudgeReplyTruncatedError` only, inside `_TimeoutJudge`'s existing deadline.
A second cut re-raises (message tagged `(after 1 retry)`). A completed retry
prefixes `ProviderAttempt.reason` with `judge truncation retry;` so the sidecar
`attempts_summary` is not a silent second call. A page whose first call
completes is one call and today's reason bytes. No new `_run_fingerprint` key.

## Tests / mutants

`tests/test_gh1127_judge_truncation_retry.py`: stub truncates once then answers
→ judged; twice → today's exception; timeout → one call, `JUDGE_OUTCOME_TIMEOUT`.
Mutants (external copy under `scratch/routing-baseline/exec/1127/mutants/`):
drop the retry → first test fails; retry forever → bounded-calls test fails.

## Round 2: retry parked

The review said do not merge. The retry code stays in the worktree; it is not
deleted and it is not the thing to ship. Three reasons:

1. **The retry never fires in production (P1).** `run_killable` turns the child's
   `JudgeReplyTruncatedError` into a plain `RuntimeError`
   (`core/killable.py:285`). Real subprocess tests against fake HTTP servers
   made exactly one request for both Ollama and vLLM. Round-1 tests bypassed
   that boundary, so they did not show the production path.
2. **Latent P0s.** The retry can start after the deadline and after the breaker
   has opened, in an abandoned `_TimeoutJudge` worker (#1128). Retry provenance
   can also vanish from the journal.
3. **The evidence is thin.** 4 of 13 first-call cuts recovered on the local
   Ollama sample; cluster vLLM was not measured.

The alternative measured below is a *bounded verdict*: stop the issues-list
runaway so a complete accept/reject can finish inside 2048 tokens. A truncated
reply stays fail-closed — never a verdict.

## Round 2: bounded-verdict measure

**N = 17.** The issue count of the longest-by-tokens complete verdict in the
2026-10-04 record (151 vLLM replies; 645 tokens;
`docs/log/2026-10-04_judge-reply-cap-and-tolerance.md`). The dump is not on
disk. That is not "no complete verdict had more than 17 issues".

**How JSON is requested today.** Ollama `_post_generate` sends `format:
"json"`. vLLM `_post_chat` sends `response_format: {type: json_object}`.
Ollama's `/api/generate` also accepts a schema object as `format` (measured).
vLLM guided decoding accepts `response_format.json_schema` (docs only; no
local vLLM).

**Arms.** Same 40 pages as Round 1 plus 20 never-truncated pages (seed 1127,
qwen cache candidate, 300 dpi). One call per page per arm. Today's arm on the
40 reuses Round-1 first calls. Probe 9.5s. 180 records, 140 live Ollama calls.

| arm | truncated | rate | mean s (min–max) |
| --- | ---: | ---: | --- |
| today | 13 / 60 | 21.7% | 57.7 (14.1–121.3) |
| (a) schema, `maxItems` 17, fields first | 1 / 60 | 1.7% | 32.3 (1.0–111.3) |
| (b) prompt, "at most 17" | 3 / 60 | 5.0% | 13.3 (1.6–69.8) |

Never-trunc group: 0 truncates on every arm.

**Agreement** (pages where today and the bound both completed and today has
a readable `faithful`):

- schema: **46/46** same `faithful`; **38/38** same `(faithful, action)`
  where today also had an action. No never-trunc flips.
- prompt: **42/47** same `faithful`. Five accept→reject flips, one of them a
  never-trunc page (ozcan p27): today accept, 34 tokens, empty issues;
  prompt reject with 17 restated case-fold issues (`c` vs `C` in equations).

Schema still truncated one page (hong_stein p45): `maxItems` did not stop
very long per-item strings from hitting 2048 tokens. That reply is still
not a verdict.

**Recommend: build (a) only.** Smallest PR: one shared schema constant; Ollama
`format` becomes that object; vLLM `response_format` becomes `json_schema`
(unmeasured on cluster — say so). Do not edit `judge_page.md`. Keep
`JUDGE_MAX_REPLY_TOKENS` and `JudgeReplyTruncatedError` fail-closed. Do not
ship the prompt bound: it is a regression on clean accepts. Leave the parked
retry unmerged.

## Round 3: schema shipped; retry reverted

The retry diff is saved at `scratch/routing-baseline/exec/1127/retry-parked.patch`
and removed from the worktree. The branch does not ship it.

`JUDGE_VERDICT_SCHEMA` / `JUDGE_VERDICT_MAX_ISSUES = 17` live in `judge/judge.py`.
Ollama sends the schema as `format`; vLLM as `response_format.json_schema`.
`judge_page.md` is unchanged. The 2048 cap and `JudgeReplyTruncatedError`
stay fail-closed: a truncated reply is still never a verdict. `parse_verdict`
does not clip `issues` — the server is asked to cap, the client must not
trust that blindly.

Round 3 asked for a cluster measure. That ran (next section): vLLM 0.17.1
accepted `json_schema` (0 request rejects). A full-corpus compare against
**274 truncations per run** (`t3_23117187`) is still the next cluster job.

## Cluster check (Rorqual, vLLM 0.17.1 + xgrammar, 2026-10-13)

The review asked for the deployed vLLM path to be verified before merging. A one-GPU job served Qwen3-VL-30B-A3B-Instruct as the corpus runs do, then sent the same 60 roster pages (20 recurring truncations, 20 one-off, 20 never-truncating) through the real `VLLMVisionJudge.judge` path, killable child included: once with main's code (`response_format: json_object`) and once with this branch (`json_schema`).

| arm | truncated | other errors | mean s |
|---|---:|---:|---:|
| main | 15 / 60 (recurring 9/20, one-off 6/20, never 0/20) | 0 | 5 |
| branch | 2 / 60 (recurring 1/20, one-off 1/20, never 0/20) | 0 | 3 |

- No request was rejected, so the deployed vLLM accepts `json_schema`.
- 45 pages completed in both arms: `faithful` agreed on 45/45 and the suggested action on 44/45.
- The two branch truncations stay fail-closed, as before.

Not measured: a full corpus run (the next cluster run compares against 274 truncations per run), and verdict quality beyond agreement with today's verdicts.

## What happens to recovered pages

Agreement (46/46 local, 45/45 cluster) only covers pages where both arms
finished. The pages this change moves are the ones that were truncated on
today/main (`page_judge_exception` → ERROR + marker) and completed on the
schema arm (a verdict; an accept would ship SUCCESS).

Prefix `faithful` is read from the cut reply's first key for this report
only. It is never parsed in product code.

**Local Ollama** (`exec/1127/bounded/calls.jsonl`; today's `reply_head`):
13 recovered (today truncated, schema completed). Schema: **13/13
`faithful=false`, `retry_same`**. Today's cut prefix: **13/13
`faithful=false`**. **false→true flips: 0.**

**Cluster vLLM** (`exec/1127/cluster-smoke/calls-{main,branch}.jsonl`):
13 recovered (main truncated, branch completed). Branch: **13/13
`faithful=false`, `retry_same`**. The cluster smoke did not keep main's
raw reply. All 13 pages are in the local data: 10 local cut prefixes are
`false`; 3 (adamek p20, tabatabaei p58, Martens p22) completed locally
today as accept, so they have no local *cut* prefix. **false→true flips:
0** (no cut prefix was `false` with a branch `true`).

A recovered accept would have been the path that hides a bad reading.
There were none. Recovered pages stay fail-closed as rejects, just with a
finished verdict instead of an exception.

Renders for the recovered set live under `scratch/routing-baseline/exec/1127/renders/`
(`{group}_{doc[:40]}_pNNNNN.png`). This log does not judge those images.

## Round 4

Literal pins on the captured request: `required`, `additionalProperties is
False`, vLLM `strict: True`. External mutants under `exec/1127/mutants-r4/`
delete each; the pin fails.

Schema comment: `issues` last so `maxItems` forces `]` then `}`. A cut
reply is never parsed. `judge_page.md` notes that constrained decoding
overrides the prompt's key order (`git grep`: no test pins those prompt
bytes; `test_page_judge_prompt_fingerprint.py` only hashes the file).

Server floor: smallest option is a version note, not a schema-bearing
probe. Verified: vLLM 0.17.1 (cluster smoke, 0 schema rejects) and
Ollama 0.40.2 (local schema arm). A 400 is already loud fail-closed; a
probe generate would add a startup call the verified servers do not need.

`issues[].items` has no `maxLength`. The residual 1/60 local and 2/60
cluster truncations are that case (long per-item strings) and stay
fail-closed. Nothing added.
