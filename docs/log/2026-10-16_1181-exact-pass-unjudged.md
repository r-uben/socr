# #1181: a native-table EXACT_PASS ships the whole page unjudged and unmarked

Date: 2026-10-16. Branch `fix/issue-1181`, base `b354a6bd`.

## The defect

Corpus run 23349398 (main `b354a6bd`): acceptance meter = 3 of 17,155, all
`table_only_acceptance` (Bybee p15, Cook–Kazinnik p11, Eskildsen p67). On the native-table
`EXACT_PASS` accept (`pipeline/agentic.py` ~L1377) the page ships at `confidence=1.0` with no
page-judge call: SUCCESS, `audit_passed=true`, no verdict clause, no mark. Nothing checked the
caption, notes, panel headings or prose around the table.

The `EXACT_PASS` guard compares a per-row multiset of numeric tokens. It cannot see a value moved
to another column of the same row, or two swapped row labels: both measured on a synthetic page,
both pass. The binding oracle (`tables/binding.py`, GH-266/GH-273) is not called on this path.
5 of 17,091 page sidecars in run 23319732 shipped through this exit.

## Decision

Mark, not check. A judge call buys no measured correction on these pages, cannot run without a
provider, and escalates without a bound when it rejects. The mark is needed as the floor anyway.

A #1145 verdict clause (`core/verdict.py` `_exact_pass_unjudged_clause`): `page_judge=UNKNOWN`,
`scope=page`, demoting SUCCESS→WARNING with `FailureMode.MODEL_OUTPUT_UNJUDGED`. It fires on
every page accepted through that exit: the `native_table_verifier_exact_pass` event AND the
shipped output's accepting reason. A page the structural gate delegated to the judge carries a
replaced reason and is untouched.

The mark says what was compared and never that the table was verified: "no judge saw this page;
only the numbers within each table row were compared, not their columns or row labels, and
nothing outside the table".

**Rejected: firing only on pages with content outside the table.** That left a table-only
EXACT_PASS page SUCCESS and unmarked, which is a violation by the meter's own rule
(`rule_qwen`). It also guarded a case with zero corpus instances (5 of 5 pages had outside
content). Fable raised it; the orchestrator took it.

## Resume

The event joins `_RESUME_REPLAYED`, and `_resume_verdict_is_permanently_unverifiable` refuses a
`page_judge` clause, so a marked page is never ledger-skipped as SUCCESS. Consequence: it is
reprocessed on every resume (about 5 pages per corpus run).

## Evidence

`tests/test_gh1181_exact_pass_unjudged_clause.py`, 10 tests, hermetic. The difference test runs
the seam twice on the same bytes, with and without the EXACT_PASS accept. 6 external mutants
(canary, uncapped anchor counts), all killed; an independent review re-ran 2 of them plus one of
its own. Full suite 7,941 passed. Lead artefacts: `scratch/routing-baseline/leads/1181/`.

## Round 2 (review blocker: the accepting reason was not pinned)

The clause requires exact equality with `EXACT_PASS_ACCEPT_REASON`, but the exit wrote its own
literal. The reviewer changed that literal and the FULL suite passed (7,894): the clause could be
switched off in production with CI green, and only the meter on the next corpus run would have
noticed. The exit now uses the shared constant; the drift test asserts it appears once and that
neither literal spelling survives in `agentic.py`; and a new end-to-end test drives
`NativeTableVerifierJudge.assess` on an in-memory PyMuPDF page to its own accepting decision and
feeds THAT `reason` through the seam, so the boundary where the string is written is crossed by a
test for the first time. A mutant restoring the literal is killed by both.

`benchmark/acceptance.TABLE_ONLY_JUDGE_REASON_PREFIX` is deliberately NOT derived from the
product constant: the meter is the independent instrument, and an instrument sharing the
product's constant cannot catch the drift it is there to catch. A test pins the relationship the
other way round -- the product reason must start with the meter's prefix -- without coupling
them. The mutant that renames the shared constant on both sides therefore survives, and that is
the correct outcome: the string's TEXT is not load-bearing, its being shared is, and the meter's
prefix match still flags a renamed reason.

## Not measured

The meter on a fresh corpus run; the corpus rate of either binding shape.

## Follow-up

`tables/binding.py` is not consulted on the agentic EXACT_PASS accept path (GH-266 territory).
