# #259 stops at a design question, not a third patch

2026-08-20. PR #260 is open, CI green, and **should not be merged.**

## What is settled

The bug is real and owner-confirmed: on a table with spanning headers the VLM produced
the correct 26-row table, socr shipped the wrong 19-row native flattening, and both
carried all 112 decimals. Reproduced from the owner's run cache before any code changed.

The diagnosis in the issue pointed at the wrong line — `orchestrator.py:1322`, a
phase-major guard, while the repro ran `--agentic` and the model's output was
`status=success`, so that guard's own predicate was false for it. The real chain runs
through `manifest.py::_winning_page_output`, which accepts `best_output` only while
`audit_passed` is true; `orchestrator.py:3084` had set it false from `att.accepted`.

Also settled, and contradicting the issue: **the judge did rate this page**. All five
rungs recorded `accepted:false`. `judge_rejected: false` is a never-written field on the
agentic path, not evidence of acceptance.

`--no-native-first` is the **same** defect, not a second one. The flag governs routing
only, it did route the page, and nothing downstream reads it. Only its help text
(`cli.py:52`) misleads.

## Why it stopped

Three rounds, three holes, each found after the previous was closed:

| | hole | closed by the next patch? |
|---|---|---|
| R1 | the bound could not tell a hard rejection from a soft one | yes, by an allowlist |
| R2 | the soft mark is written on a refused decision, and the structural gate only inspects *accepting* ones — so a ragged grid wearing the soft label ships | closable at the keep site |
| R3 | **a DETECTED wrong value ships** | **no** |

Hole B, found by the code owner while checking the premise round 2 rests on:
`native_verifier.py:862-880` downgrades a numeric multiset mismatch to `AMBIGUOUS` when a
row-count discrepancy makes row pairing unreliable. `_value_guard` **detects** the wrong
number and returns it; `verify_native_table` then **discards** it — `drifted_rows` is
populated only on the `hard_fail` branch. Measured: output tokens `{4.40, 5.50, 9.99}`
against native `{4.40, 5.50, 6.60}`, drift detected = 1, drift surfaced = 0, and the keep
path ships the table containing `9.99`.

The requested structural check cannot close it: the gate is string-only grid shape and is
documented as blind to values by construction. The division of labour is the other way
round — TR-3 owns values, the gate owns structure — and on this path TR-3's finding is
thrown away before any consumer sees it.

## The structural reason, which is the actual finding

The keep site reconstructs, at **assemble** time, a quality judgement that lives in
**routing**-time components which do not persist their evidence. Round 2 had to add
`rejection_class` to `PageOutput` to recover one discarded fact. Round 3 would have to
propagate `drifted_rows` to recover another. Each round discovers one more thrown-away
datum. **The bound is not tightening toward a limit; it is chasing an information-loss
problem upstream of it.**

Green CI on #260 means the suite does not cover holes A and B, not that they are absent.

## The design fork — the owner's call

On a born-digital table page where the ladder accepted no rung **and** the native reading
also carries a table-distrust flag, socr has two candidates and trusts neither. It
**already has a settled answer**: the TR-3 D3 fail-closed floor — ship neither, emit an
explicit failed-table marker, route the region to the image-asset lane.

#259 asks for a *second, different* answer to the same question: keep the model's
reading, flagged. The two dispositions are reachable on pages that differ only in which
flags happened to fire.

1. Which disposition governs this page class?
2. If keep-with-flag is right, what is the **closed** set of conditions that make keeping
   safe — stated once, up front, rather than discovered one review round at a time?
   Current list: model output present and containing a grid; verifier deferred rather
   than refuted; no drift detected even on the downgraded path; grid well-formed by the
   gate's own predicate; neither fail-closed floor applies. Nobody can show that list is
   complete, and that is the whole problem.
3. Should the evidence be persisted at **routing** time — a disposition record on the
   attempt, written by the component that owns each check — rather than reconstructed at
   assemble time from whichever flags survived?

## A separate bug found on the way, present on main today

Detected numeric drift is silently discarded on the AMBIGUOUS path
(`native_verifier.py`, the `row_count_warn_info` return). `_value_guard` returns
`drifted` non-empty and it reaches **no** audit event, no page status, no metadata, no
CLI. That is a cardinal-rule violation independent of #259 and of this PR — it is not
introduced by anything here. **Worth its own issue; the owner files issues.**

## Recommendation

Do not merge #260. Rounds 1 and 2 are real gains — the corrected diagnosis, the
`rejection_class` plumbing, five reverse guards, and the confirmed fact that the judge did
reject — and they are worth keeping as the input to the design pass rather than as a
shipped predicate.
