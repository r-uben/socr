# ADJUDICATION BRIEF — TICKET-C, wave 2 (READ-ONLY)

Two triagers on different vendors have each produced a verdict for every issue in
your batch. You triaged none of them — that is deliberate, and it is why you are
here. Your job is to reconcile, not to re-triage from scratch.

## Inputs

- `triage/verified/<batch>.json` — **read this, not the raw vendor files.** It
  carries every verdict with a machine-computed `evidence_verified`, produced by
  `bin/verify_citations.py` from git, not by any model.
- `triage/<batch>/*.json` — the raw vendor records, if you need a rationale in full.
- `clusters.json` — which issues share one mechanism.
- `issues_snapshot.json` — the pinned issue set.

## The rules that bind you

1. **Only `evidence_verified: true` verdicts count as evidence.** Anything else is
   forced to `NEEDS-MEASUREMENT`. You may not overrule that field; you did not
   compute it and neither did the triager.
2. Read `evidence_class` before you dismiss a verdict:
   - `CLEAN` — every citation resolved exactly.
   - `DRIFT` / `PARTIAL` — the snippet IS in the file, at a different line. The
     agent read the code; the line number moved. The verdict is still forced to
     `NEEDS-MEASUREMENT`, but you should say so in `recoverable: true` when the
     reasoning is otherwise sound and a corrected citation would settle it. That
     tells the morning which verdicts are one `grep` away from being usable.
   - `FABRICATED` — nothing resembling the snippet is in the file. Discard the
     verdict entirely and say so.
3. **Reconcile per cluster, not per issue.** Issues in one cluster share a
   mechanism. If your dispositions for two clustered issues cannot both be true,
   you have not finished. State explicitly how they fit together.
4. **Unanimous → final. Split → escalate**, and record BOTH readings verbatim; do
   not average them into a mush. A `MISSING` vendor is a split, not a blocker.
5. You may verify a claim yourself against the pinned tree — read-only, at
   `/Users/rubenffuertes/repos/.worktrees/socr-night-base`, never the main
   checkout. If you cite anything new, it must be `{path,line,snippet}` exact at
   `main_sha`; your citations go through the same machine check.
6. **You cannot close anything.** You produce a disposition. A separate two-agent
   review board, on vendors that neither triaged nor adjudicated the issue, then
   tries to REFUTE each staged action. Write for that reader.

## Bias you are being asked to resist

The cheap failure here is agreeing with a confident triager. Two things that
actually happened in this repo:

- Issue #249 needed three owner revisions before its diagnosis held.
- PR #250's fix reintroduced the bug it was meant to fix, with a test in that PR
  asserting the defective behaviour — and passing.

A unanimous, well-cited verdict can still be **category-wrong**: correctly
describing code that has nothing to do with what the issue reports. When both
triagers agree, spend your effort asking whether they are answering the issue's
actual question, not whether they cited real lines.

## Output

Write `verdicts/<batch>.json`:

```json
{"batch":"batch-N","adjudicator":"<vendor>","main_sha":"53b0637...",
 "grounding":{"canary":"<exact stdout of isolation_canary.sh>"},
 "clusters":[{"cluster":"table-destruction",
              "joint_reading":"one paragraph: what the mechanism actually does at main_sha",
              "issues":[144,146]}],
 "rows":[
  {"issue":144,
   "final_disposition":"FIX-CANDIDATE",
   "agreement":"UNANIMOUS|SPLIT|SINGLE-VENDOR|NO-EVIDENCE",
   "vendor_readings":{"grok":"...verbatim gist...","kimi":"...verbatim gist..."},
   "evidence_verified":true,
   "recoverable":false,
   "basis":"why this disposition and not the other reading",
   "citations":[{"path":"...","line":1,"snippet":"..."}],
   "action_recommended":"close|comment-correction|none|file-new|fix",
   "proposed_comment":"the exact text to post, or null",
   "risk_if_wrong":"one sentence — what breaks if this disposition is wrong"}]}
```

Every issue in the batch appears exactly once. No row may carry
`evidence_verified: false` together with a disposition other than
`NEEDS-MEASUREMENT`. `action_recommended: "close"` requires the three
ALREADY-FIXED proofs to have survived the machine check — if they did not, the
disposition is not a close, however obvious it looks.
