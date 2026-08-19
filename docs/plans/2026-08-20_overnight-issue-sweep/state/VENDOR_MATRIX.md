# Vendor matrix — who may play which role

Owner directive, 2026-08-20 ~02:10: cursor is available as an extra non-Claude seat
(quota ~95%, healthiest of the pool); grok ~80% and kimi ~81%, so **their silence
tonight is not a quota problem**; GPT/codex is at zero until roughly 08:15 and
should be brought back in as a reviewer/adjudicator seat when it returns — it was
the sharpest of the three panel critics on coverage.

## The rule, unchanged

Whoever triages an issue never adjudicates or reviews it. Two reviewers per staged
action, on different vendors, neither of which triaged or adjudicated that issue.

## Live-vendor reality at 02:10

| vendor | status tonight |
|---|---|
| ollama-deepseek | fast, cleanest citations in the run (11/11 exact on batch-2) |
| ollama-minimax  | fast, but 9 of 12 verdicts lost to line drift on batch-3 |
| gemini-pro      | fast, 21 of 22 verified on batch-4 |
| gemini-flash    | untested, dispatched as batch-2 second seat |
| claude          | orchestrator + replacement triage seats |
| cursor          | reserved for adjudication — see below |
| grok            | **STALLED**: wrote a scaffold with a valid grounding token at 01:27, then nothing for 35+ min on two batches. Not quota. |
| kimi            | **STALLED**: no output file at all on two batches. Not quota. |
| gpt-*           | quota zero until ~08:15 |

Treating `gemini-pro` and `gemini-flash` as one vendor family for independence
purposes — different model seats, but not an independent house.

## Assignment (cursor makes the separation clean)

| batch | triage | adjudicate | review |
|---|---|---|---|
| batch-1 | claude + minimax | **cursor** | deepseek + gemini-pro |
| batch-2 | deepseek + gemini-flash | minimax | claude + **cursor** |
| batch-3 | minimax + deepseek | gemini-pro | **cursor** + claude |
| batch-4 | gemini-pro + claude | deepseek | minimax + **cursor** |

Every row satisfies the rule with no vendor-family overlap. Without cursor the pool
was four houses against five distinct roles, and one overlap per batch was
unavoidable — the owner's note removed a real defect in the plan rather than just
adding capacity.

## If GPT returns before the run ends

Add it as the escalation seat for any `SPLIT` the adjudicator records, and as a
third reviewer on any action whose two reviewers disagree. Do not re-run work that
is already terminal.
