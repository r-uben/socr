# TICKET-B1 review — 2026-08-01

Reviewed commit `402395c` on `feat/123-metric-blind-spots`.
Verdict: **ACCEPT-WITH-FOLLOWUP**. Landed as-is; findings 1 and 2 tracked below.

Reviewed by the orchestrating session rather than a `socr-reviewer` agent — see
"Process note" at the bottom.

## Sound

- **The `rows_establish_grid` extraction is a pure refactor.** The original inline
  block in `_table_page_needs_escalation` read
  `if modal_width < 2 or rows_at_modal < 2: return False`; the extracted function
  returns `modal_width >= 2 and rows_at_modal >= 2`. Same `Counter`, same
  `most_common(1)` tie-breaking (insertion order), same empty guard. No behavioural
  change to escalation triggering.
- **Consumer inventory is correct.** `table_exactness.score_page` has exactly two
  callers: `pipeline/orchestrator.py:1469` and `tables/escalation_decision.py:97-98`.
  (`benchmark/scorer.py:322` is a *different* `score_page` — the WER/NES scorer
  method — not this function.) Both handle `pct is None`; the orchestrator guards
  with `report.pct is not None and report.pct < 100.0`.
- **Test count anomaly resolved.** Not a behavioural flip. Only
  `tests/test_gh96_table_exactness.py` differs base→HEAD, by one non-parametrised
  test def (26→27). Base collected 1365, HEAD collects 1366 (= 1365 passed + 1
  xfailed). STATUS.md's documented base of 1363 passed + 1 xfailed = 1364 is **off
  by one** — a transcription error in the plan doc. The suite has exactly one
  conditional skip (`tests/test_dual_pass_tables.py:695`, gated on
  `judge._get_fitz_page is None`); that file runs 40 passed with zero skips, and
  B1's orchestrator diff does not touch `_build_page_judge`.

## Finding 1 — HIGH: `ceiling_note` reaches no surface

`grep -rn "ceiling_note" src/` outside `table_exactness.py` returns **zero hits**.
The not-scorable fact lives entirely inside the `ExactnessReport` object: no page
status, no document status, no metadata, no audit kind, no CLI surface.

CLAUDE.md requires failures to surface at *every* level, not just one. This is
arguably a **visibility regression**: OBR page 54 previously reported a loud, wrong
`0.0%`; it now reports `pct=None` and disappears without trace. Trading a wrong
number for no number and no trace is the shape the no-silent-content-loss rule
exists to prevent. Setting `ceiling_note` satisfies the ticket's literal wording but
not the repo rule behind it.

**Disposition:** folded into TICKET-C1, which already builds exactly this machinery
(named audit kind → `TABLE_DISTRUST_KINDS` → document-level surface). Not a
duplicate mechanism; the same one, given a second input.

## Finding 2 — MEDIUM: the ticket's headline benefit is not delivered by this commit

The stated motivation — *"reported aggregates count pages-with-no-table as engine
failures, understating every engine by an unknown amount"* — is not realised here.
There is **no aggregate helper in `src/`**; the only two consumers are per-page
decision paths. Corpus aggregation lives outside the repo in
`~/data/fiscal-ballast/_experiments/`. Engine numbers move only if those scripts
skip `pct is None`.

**Not yet validated.** The preserved runs
(`2026-07-31_gh96-engine-parity/`, `2026-08-01_gh96-corpus-rerun/`) have not been
re-scored against B1. Do not validate against a fresh OCR run — local-model
run-to-run variance exceeds the effect.

## Finding 3 — LOW: `gt_rows=0` is safe, but not for the reason given

The commit message claims `gt_rows=0` "closes `escalation_decision`'s own
`gt_rows>=2` gate". It does not close it — it **reroutes** to the `judge_escalation`
canary (`escalation_decision.py:121`), a more expensive path. This is harmless only
because `_table_page_needs_escalation` already returns `False` for non-grid pages,
so `decide_escalation` is never reached for them in production. That second guard is
doing the real work and is undocumented at the call site. `decide_escalation` is
public API; no test covers the routing.

## Finding 4 — LOW: one pre-existing ceiling is now unreachable via `score_page`

`score_rows`'s `len(gt) < 2` branch (`table_exactness.py:180`, the ceiling that
yields `pct=0.0` alongside `scorable=False`) can no longer fire through `score_page`
— anything with fewer than two rows fails the grid predicate first. It remains live
via direct `score_rows` calls in tests. The `pct=0.0`-vs-`pct=None` inconsistency
between the two ceilings is therefore mostly cosmetic now, but two ceilings with two
contracts is still a trap of the kind that produced the original seven defects.

## Constraint this imposes on TICKET-A1

Every A1 corruption-battery fixture **must clear the grid predicate** (at least two
value columns, at least two rows sharing that width). A fixture that does not becomes
silently not-scorable with `pct=None`, and every "score got strictly worse" assertion
then compares against `None`.

## Process note

The `socr-reviewer` agent dispatched for this ticket was stopped mid-run and its
findings discarded. It ran `git checkout` against the shared working tree twice
(→`9f9c7db`, →`18b3b64`), despite an explicit read-only brief and a follow-up
instruction naming `git checkout` specifically. During those windows the tree held
the *pre-B1* source, so anything it read off disk was void. Nothing was lost — the
worktree was clean throughout and `402395c` was always safe on the branch.

**Rule for the remaining tickets:** read other revisions with `git show <rev>:<path>`.
Never `git checkout` / `switch` / `stash` / `reset` in this repo while agents are
running. A throwaway `git clone` is *also* not isolation here — the editable install
resolves `import socr` to the main checkout, so tests run in a clone still exercise
the main tree's source.
