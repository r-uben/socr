# REVIEW BOARD BRIEF — TICKET-DR, wave 3

You are one of **two reviewers** on a staged tracker action. The other is on a
different vendor. Neither of you triaged or adjudicated this issue.

**Decision rule: both reviewers approve, or the action does not execute.**
Anything else — one rejection, one abstention, one reviewer who could not open the
code — makes it `HELD-FOR-OWNER` and it waits until morning. A held action costs
the owner two minutes. A wrong autonomous close costs them a real bug.

## You are prompted to REFUTE, not to confirm

Do not ask "is this action reasonable?" Ask "what is wrong with it?" and go
looking. Specifically:

- **For a close (`ALREADY-FIXED`):** take the issue's acceptance criteria one at a
  time and try to find one that `fixed_by_commit` does **not** satisfy. Open the
  code at `main_sha` and check. A commit that touches the right file is not a fix.
  A test that passes is not a fix if it asserts the wrong behaviour — that is
  literally what PR #250 did in this repo today.
- **For a correction (`MISREPORTED`):** find the claim in the proposed comment
  that its evidence does not support. Corrections are posted publicly and stay on
  the record; a confidently wrong correction is worse than silence.
- **For a new issue:** find the open or closed issue that already covers it.

## The category error is the one to hunt

Every citation in the packet has already been resolved against the pinned tree by
`bin/verify_citations.py`, so evidence is real code. That is not the risk. The
risk is a well-evidenced verdict **about the wrong thing** — accurately describing
a mechanism that is not the one the issue reports. All three panel critics
independently flagged this as what survives every other gate. Read the issue's own
words first, then the action, then the code.

## Hard requirement

**You must open the cited code yourself.** Read-only, at
`/Users/rubenffuertes/repos/.worktrees/socr-night-base`, at `main_sha`. Never the
main checkout at `/Users/rubenffuertes/repos/tools/socr`.

If a tool call is denied, a file will not open, or `gh` fails — **you must
REJECT**, with `blocked` as the reason. Do not reason around a gap. An agent that
cannot see the code and approves anyway is the exact failure this board exists to
stop, and it has happened here before: a headless agent denied a tool permission
produced confident invented output and exited 0.

Verify your access before you judge, and paste the result:

    git -C /Users/rubenffuertes/repos/.worktrees/socr-night-base \
      show "53b0637b928c486e9ff3023fa9527905fec017b2:<a path from the packet>" | sed -n '<line>p'

## Output — one file per action

`actions/review/<action_id>.<your-vendor>.json`:

```json
{"action_id":"...","issue":123,"reviewer":"<vendor>",
 "access_proof":{"cmd":"...","output":"<exact stdout>"},
 "vote":"APPROVE|REJECT",
 "reason":"one sentence a human can act on",
 "refutation_attempted":"what you actually tried to break, and what happened",
 "unsatisfied_criteria":["criterion the fix does not meet"],
 "comment_corrections":"if the proposed text overstates anything, the fix"}
```

`APPROVE` means: I opened the code, I tried to refute this, and I failed to.
Nothing weaker earns it. If you are unsure, `REJECT` — the cost is asymmetric and
the owner is asleep, not gone.
