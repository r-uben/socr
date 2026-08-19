# GPT (gpt-5.6-sol) — coverage lens

Panel critique of `00-draft-attacked.md`, 2026-08-20. Verbatim.

1. **NO TICKET — no immutable triage baseline.** A4 snapshots issue numbers but not the `origin/main` SHA, issue body/comment versions, or open-PR state. Agents can therefore judge different code or stale issue premises as the night progresses. This is especially dangerous for #215/#245 after PR #247, #140 after #243, #168 after #241, #161/#238 after the recent fingerprint fixes, and #220 after merged PR #224. **Minimum closure:** add a wave-0 baseline ticket recording one commit SHA and issue snapshots; require every B/C citation to be a permalink at that SHA, then add a pre-action freshness check for changed issues or `origin/main`.

2. **C1..C4 — source citations are being treated as proof of behavioral resolution.** `file:line` can establish that code exists; it cannot establish that an issue's reproducer or acceptance criteria pass. This makes false `ALREADY-FIXED` verdicts likely for:
   - #220: PR #224 merged the viewer, but closure requires checking the issue's full CLI/filter/read-only acceptance.
   - #127: PR #224 added a structure-loss detector; it did not make native extraction preserve structure.
   - #215: PR #247 added header attribution, while #245 reports a fail-open path inside that new mechanism.
   - #144/#151/#205: several guards landed, but the broader loss/surfacing claims remain distinct.
   - #140: glyph recovery in #243 is not equivalent to math-font detection/recovery.
   - #156 and #163: matching commits exist on non-main branches; they are not fixes on `origin/main`.
   **Minimum closure:** require an `ALREADY-FIXED` record to include `fixed_by_commit`, proof that it is an ancestor of the pinned `origin/main`, mapped acceptance criteria, and a passing regression/reproducer result. Otherwise demote to `NEEDS-MEASUREMENT`.

3. **C1..C4 / NO TICKET — the verdict taxonomy has no terminal state for non-defect backlog.** All 62 issues are guaranteed into B by A4, so none falls through triage numerically, but whole classes fall through after adjudication because `STILL-VALID` incorrectly implies eligibility for E:
   - umbrella/strategy: #39, #49, #56;
   - proposals or blocked experiments: #114, #202, #203;
   - architecture programmes/ADRs: #155, #174, #175, #176, #178;
   - broad audits/maintenance: #142, #156, #158;
   - features or partial implementations: #127, #220.
   These are neither automatically closable nor one-night "accepted fixes." **Minimum closure:** add dispositions `PARTIALLY-IMPLEMENTED`, `ROADMAP/UMBRELLA`, `BLOCKED-PROPOSAL`, `CHORE/PLAN`, and `DEFERRED`, each with an explicit morning action; E must consume only a separate `FIX-CANDIDATE` flag.

4. **NO TICKET — empirical claims have no validation lane.** B explicitly re-derives claims from source, while many issues turn on rendered output, corpus prevalence, or hand judgement: #64, #140, #144, #146, #147, #150, #151, #152, #190, #205, #213, #215, #223, #245, #248, and #249. `NEEDS-MEASUREMENT` merely parks them; nothing records whether the reproducer is public, synthetic, locally available, copyrighted, missing, or too costly. **Minimum closure:** add a measurement-feasibility ticket that classifies every such issue as `PUBLIC-REPRO`, `SYNTHETIC-REPRO`, `LOCAL-CORPUS-AVAILABLE`, `CORPUS-UNAVAILABLE`, or `HUMAN-JUDGEMENT`, and runs only the unattended-safe subset with saved commands/results.

5. **A4 — "four batches" is not yet an inspectable coverage scheme.** The draft contains no actual membership or balancing rule beyond a future `batches.json`. The equality check protects the 62 issues present at snapshot time, but not issues opened or reopened after A4, and not a failed/partial B worker that writes malformed output. **Minimum closure:** add `snapshot_sha`, `snapshot_at`, per-batch issue lists, expected vendor outputs, schema validation, and a wave-1 reconciliation asserting exactly `62 × 3` valid verdict records before C starts; separately report post-snapshot issue drift.

6. **NO TICKET — related-issue clustering happens too late, if at all.** A4 assigns each issue independently, so different batches can reach contradictory verdicts on the same mechanism:
   - header attribution: #215, #245, PR #247;
   - table destruction/surfacing: #144, #146, #151, #195, #197, #198, #205, #226;
   - chart/figure arbitration: #150, #167, #181, #189, #248, #249;
   - cascade halt: #221, #222, #227;
   - structure loss: #127, #223;
   - CLI/config semantics: #139, #142, #168;
   - cost/provider identity: #154, #159, #160;
   - equations: #140, #157, #164, #165, #219;
   - architecture: #155, #174, #175, #176, #178.
   **Minimum closure:** add a pre-triage relationship map with `duplicate_of`, `overlaps`, `supersedes`, `parent/umbrella`, and `must_adjudicate_together`; colocate each cluster in one C ticket.

7. **D1/D2 — no independent action gate separates adjudication from irreversible tracker mutation.** C's `evidence_verified=true` is too weak to authorize closure, and D1's "nothing closed on a 2–1 split" still allows a unanimous but category-wrong close. **Minimum closure:** add a machine-checked action manifest containing issue state, unchanged body/comment timestamp, pinned-main commit, acceptance proof, relationship-cluster disposition, and proposed comment; require a second non-triager approval specifically for every close. Changed issues are skipped, never force-actioned.

8. **D3 — no process produces candidate new defects.** B only evaluates existing issue claims, and C only adjudicates those verdicts. Therefore D3 has no defined input. Incidental observations can instead become poorly scoped duplicates, especially around #215's transposed-column observation, the #249 chart family, or unresolved findings from merged PR #250. **Minimum closure:** add a discovery-candidate schema and ticket requiring reproduction, observed/expected, search across open and closed issues, overlap links, independent validation, and a final `NEW` versus `COMMENT-ON-EXISTING` decision before filing.

9. **E&lt;n&gt; — no ticket converts 62 verdicts into a bounded fix queue.** The lane paragraph names selected candidates, but there is no owner, selection rule, severity/cost cutoff, dependency derivation, or generated list of actual E tickets. Most `STILL-VALID` issues therefore have no path to a PR, while agents may opportunistically choose different scopes. **Minimum closure:** add an E0 scheduling ticket that emits explicit fix tickets with issue set, scope, files, baseline SHA, priority, dependency, reviewer, and a documented defer reason for every remaining `STILL-VALID` issue.

10. **E&lt;n&gt; — serializing files does not prevent conflicting unmerged PRs.** Lane O proposes several PRs touching `orchestrator.py`, each branching from `main`, while nothing is merged overnight. Running them serially still yields sibling branches with overlapping diffs; later agents neither contain nor test against earlier proposed fixes. The same problem applies to clustered table changes. **Minimum closure:** choose one of: one owner/PR per collision lane; an explicit stacked-branch chain with PR bases pointing to predecessor branches; or defer later colliding fixes. Record the strategy per lane in E0.

11. **E&lt;n&gt; — implementer/reviewer separation is stated but not represented in the graph.** There is no reviewer sub-ticket, revision loop, rejection state, or rule for findings that expand scope. "PR open and checks green" can be reached without the promised independent review. **Minimum closure:** split each fix into `implement → review → revise if needed → push/PR → CI remediation → final verify`, with distinct agent identities and recorded review disposition.

12. **E&lt;n&gt; / F1 — CI has no terminal-failure or timeout path.** `gh pr checks all pass` is only a success condition. A failed check, missing workflow, merge conflict, rate limit, or check that remains queued can block F1 indefinitely. A wave producing zero E tickets is also undefined because F1 depends on `E`. **Minimum closure:** add terminal states `GREEN`, `FAILED`, `TIMED-OUT`, `NO-CHECKS`, and `NO-FIX-CANDIDATES`; define bounded retries and allow F1 after all E items reach any terminal state.

13. **NO TICKET — there is no coordinator work between waves.** The graph assumes completed files automatically trigger validation, dispatch, recovery from one missing vendor, lane scheduling, tracker actions, and report generation. It also lacks restart checkpoints after A, B, C, D, and each PR. **Minimum closure:** add one serial coordinator ticket per transition that validates schemas/counts, records a checkpoint, applies zero-output semantics, dispatches only ready successors, and marks failed inputs as skipped rather than silently shrinking coverage.

14. **A2/D1..D3 — tracker writes are not made idempotent or verified.** The auth abort rule stops future writes, but a timeout can occur after GitHub accepted a close/comment/create request. Retrying can duplicate comments or issues; assuming failure can leave an unreported mutation. **Minimum closure:** add an action ledger with deterministic action IDs, before/after issue state, returned URL, and read-after-write verification; retries must search for the action marker first.

15. **F1 — the morning artifact is not sufficient for a 20-minute trust review.** A flat list of closes, corrections, filings, PRs, and skips omits:
    - the pinned baseline and final live-state drift;
    - a 62-row reconciliation proving every starting issue received exactly one final disposition;
    - confidence, reproducer/test result, fixed-by commit, and relationship cluster;
    - proposed versus executed tracker actions;
    - PR head SHA, exact CI conclusion/time, reviewer result, conflicts, and failed branches without PRs;
    - a ranked human-decision queue and explicit rollback/reopen targets for autonomous closes;
    - counts whose totals reconcile to the starting 62 plus newly filed issues.
    **Minimum closure:** require F1 sections `Baseline`, `Coverage reconciliation`, `Autonomous mutations`, `PR decision queue`, `Deferred measurements`, `Failures/abort`, and `Human actions`, with one compact row per starting issue and links to all underlying evidence.
