# TRIAGE BRIEF — TICKET-B, wave 1 (READ-ONLY)

You are one of two independent triagers on one batch of socr's open-issue backlog.
The other triager is on a different vendor and does not see your output. A third
agent adjudicates. Do not try to be agreeable — an honest split is more useful
than a false consensus.

## Hard rules

- **READ-ONLY.** You may not edit, commit, push, or touch the GitHub tracker.
  `gh issue view` / `gh issue list` are fine; `gh issue close|comment|edit` are
  forbidden. The ONLY file you write is your own output JSON.
- **Never touch `/Users/rubenffuertes/repos/tools/socr` as a working tree.** It is
  owned by another session. Read the code at the pinned baseline instead:
  `/Users/rubenffuertes/repos/.worktrees/socr-night-base` (detached at `main_sha`).
- Before ANY test or probe you run:
  `export PYTHONPATH=/Users/rubenffuertes/repos/.worktrees/socr-night-base/src`
  then `bash <plan>/bin/isolation_canary.sh /Users/rubenffuertes/repos/.worktrees/socr-night-base`.
  It must exit 0. A measurement taken without it is void.
  (`pytest` rooted inside that worktree is already isolated; the `socr` CLI and
  `python -c` are NOT — they silently run main-checkout code.)
- **CI has no ollama and no provider.** Any test driving `_phase_agentic` or
  `process()` in agentic mode must patch `_available_engines_for_agentic`.
- **Nobody will answer a question.** Blocked → record the blocker in the record and
  move on. Never guess. Never invent a citation, a line number, or a test result.
  An invented citation is caught mechanically and voids your whole verdict.

## Your job

For every issue in your batch, emit exactly one verdict:

| verdict | when |
|---|---|
| `FIX-CANDIDATE` | still valid AND a competent agent could fix it in one night |
| `ALREADY-FIXED` | the defect no longer exists at `main_sha` — highest bar, see below |
| `MISREPORTED` | the code does something, but not what the issue says (#249 is the template) |
| `DUPLICATE` | same defect as another issue; name it in `duplicate_of` |
| `PARTIALLY-IMPLEMENTED` | some acceptance criteria met, others not; list which |
| `ROADMAP-UMBRELLA` | a programme of work, not a single defect |
| `BLOCKED-PROPOSAL` | a proposal blocked on a decision or a measurement |
| `CHORE-PLAN` | housekeeping/doc drift, no runtime behaviour |
| `NEEDS-MEASUREMENT` | real question, cannot be settled without a corpus run or hand judgement |
| `DEFERRED` | valid but deliberately out of scope for one unattended night |

**`ALREADY-FIXED` is the dangerous one.** A citation proves code EXISTS. It never
proves a bug is FIXED. To claim it you must supply all three of:
1. `fixed_by_commit` — a real SHA, an ancestor of `main_sha`;
2. `acceptance_criteria` — the issue's criteria enumerated one by one, each marked
   met/unmet with its own citation;
3. `reproducer` — a test or command you actually ran, with its real output pasted.

Missing any one of those → the verdict is `NEEDS-MEASUREMENT`, not `ALREADY-FIXED`.
This is enforced by a script, not by a reviewer's goodwill.

## Citations

Every verdict carries citations. A citation is `{"path","line","snippet"}` and must
resolve at `main_sha`:

    git -C /Users/rubenffuertes/repos/.worktrees/socr-night-base show 53b0637b928c486e9ff3023fa9527905fec017b2:<path> | sed -n '<line>p'

`snippet` must be a substring of THAT line. Copy it from the command's output; do
not retype it from memory. Citations are verified mechanically in wave 1.5 and a
failing citation voids the verdict that carried it.

## Clusters

Your batch is grouped into clusters of issues that share one mechanism (see
`clusters.json`). Read a cluster's issues together and keep your verdicts mutually
consistent — if you call #197 already-fixed and #144 still-broken, say explicitly
why both can be true.

## New defects you stumble on

If you find a real defect that no open issue covers, write it to
`<plan>/discoveries/<vendor>-<slug>.json` with `{title, mechanism, citations,
observed, expected, searched_issues}`. Do NOT file it on the tracker.

## Output — one JSON file, exactly this shape

Path: `<plan>/triage/<batch>/<vendor>.json`

```json
{
  "batch": "batch-N", "vendor": "<you>", "main_sha": "53b0637...",
  "started_at": "...", "finished_at": "...",
  "verdicts": [
    {"issue": 144, "verdict": "FIX-CANDIDATE",
     "confidence": "high|medium|low",
     "rationale": "<= 6 sentences, concrete, naming the mechanism",
     "citations": [{"path":"src/socr/...","line":123,"snippet":"exact text on that line"}],
     "duplicate_of": null,
     "fixed_by_commit": null,
     "acceptance_criteria": [],
     "reproducer": null,
     "cluster_consistency": "how this sits with the other issues in its cluster",
     "blocked_by": null}
  ]
}
```

Every issue in your batch must appear exactly once. Coverage beats depth: a
truthful `NEEDS-MEASUREMENT` with one good citation is worth more than a confident
`ALREADY-FIXED` you cannot prove. Write the file incrementally as you go so a
timeout still leaves partial work on disk.
