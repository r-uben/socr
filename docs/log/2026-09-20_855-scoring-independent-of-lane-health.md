# GH-855: a latched escalation lane also switched off table scoring

## What changed

- `src/socr/pipeline/orchestrator.py` (`_phase_agentic`, ~line 9126): the arm
  that gates `_surface_table_scoring` no longer reads `_escalation_degraded`.
  A new `_lane_configured = _escalation_profile is not None` variable drives
  the scoring UNION's second arm; `_lane_live` (`_escalation_profile is not
  None and not _escalation_degraded`) is unchanged and still gates
  `_escalate_table_page` alone at ~line 9203. One expression touched, exactly
  as scoped.
- `tests/test_gh855_scoring_independent_of_lane_health.py` (new, 2 tests).
  Deliberately two separate assertions rather than one comparison trying to
  carry both — the two halves are in tension for the same page and cannot
  both be pinned by a single run pair (see Review round below):
  1. `test_a_latched_page_the_detector_missed_now_gets_scored` — **presence**.
     With the latch forced SET, a page in the AFFECTED population (a page the
     detector's `_page_has_tables` misses, the lane configured, not a chart
     asset, latch set) still emits a real `table_unexplained_lanes`
     `AuditEvent`. Falsified by the mutation guard below.
  2. `test_the_latch_does_not_move_text_audit_passed_or_status` — **safety**.
     Same fixture, latch forced SET vs forced CLEAR in the same process,
     escalation forced to reject in both runs so `bo` never changes. Selected
     text, per-page `audit_passed` and per-page status are identical between
     the two runs.

  `_escalate_table_page` is a deterministic double in both tests (real GH-96
  timeout/networking behaviour is `test_gh96_escalation_lane.py`'s own
  concern); `_page_has_tables` is forced False so every page is in the
  affected population this ticket's expression change covers, isolated from
  the (unaffected) detector-flagged first arm.

## Verifying the issue's claims against source (per the ticket's ask)

The issue's line numbers, the coupling expression, and the docstring quote
from `_surface_table_scoring` all matched the source exactly (only the exact
line number shifted by one, from `~9115` to `9116`, from unrelated churn
above). No discrepancy found; nothing to report back.

## Trace the consequence (measured, not assumed)

- `_table_page_needs_escalation` (the only place scoring's `state.events`
  come from) was read line-by-line: every branch either returns a bool or
  appends an `AuditEvent`. It never mutates `bo.text`, `ps`, or
  `PageOutput.audit_passed`.
- The AFFECTED population — pages where `_page_has_tables` is False, the lane
  is configured, the page isn't a chart asset, and the latch is set — is
  exactly the set the old expression skipped and the new one scores. Neither
  downstream consumer of the score can reach those pages: the dual-pass
  crop-reread (`~9148`) additionally requires `_page_has_tables`, false by
  construction; the escalation call (`~9212`) additionally requires
  `_lane_live`, false while latched. So on that population the new score is
  provably observation-only — confirmed empirically in
  `test_the_latch_does_not_move_text_audit_passed_or_status`, not just argued.
- Separately traced `audit_passed` writers near this code (`_agentic_native_page`
  at line 9755, `native_table_distrusted`): that predicate reads
  `ps.native_table_unverifiable` / `native_table_structure_defective` /
  `native_table_emission_defect` / `native_table_header_unattributed` — none
  of which `_table_page_needs_escalation` touches. The two surfaces are
  disjoint.
- `_lane_live` at the escalation call site (`~9212`) is untouched: a latched
  document still cannot re-attempt escalation on a later page.
- **Document-status consequence — the repair, not a regression to bury:**
  `table_unexplained_lanes` and `table_not_scorable` are ordinary
  `TABLE_DISTRUST_KINDS`, resolvable only by a later `table_escalation_accepted`
  on the *same page* (`tables_trust.py`'s `WHOLE_PAGE_RESOLVING_KINDS`). A
  page scored while the lane is latched has no escalation attempt left to
  accept a better candidate and clear that distrust event, so a document an
  unlatched run would have resolved back to clean can now legitimately report
  degraded (`table_unexplained_lanes` is also a
  `CREDENTIAL_BLOCKING_EVENT_KIND`, `manifest.py:1797`, so this can gate the
  export credential too). That status movement is the fix doing its job —
  reporting a real, previously-silent defect — not a side effect.
- Cost: unaffected outside already-latched documents. The scoring call this
  ticket un-gates was already running on every page while the lane was
  healthy; latching a document is the failure path the issue itself
  documents as rare enough to be a "moment," not a steady-state cost driver.

## Review round: why the test is two assertions, not one paired run

An earlier draft tried to pin "more events under the latch" and "identical
text/audit_passed/status" as one difference between a latch-SET and
latch-CLEAR run of the same later page. That is unsatisfiable: the only way
for event COUNTS to differ between SET and CLEAR on the same page is for the
CLEAR run's escalation to be ACCEPTED (clearing the scoring event via
`table_escalation_accepted`) while the SET run's is latched off (event
stays) — but an accepted escalation necessarily changes the page's shipped
text, so "same text" cannot hold across that same comparison. The two claims
were split onto their own axes instead: presence is pinned against the
PRE-FIX GATE via mutation (assertion 1), and safety is pinned against the
LATCH via a paired run with escalation rejecting in both arms (assertion 2).

## Guard proof (outside the repo)

Copied `src` + `tests` + `pyproject.toml` to `/tmp/gh855-mutant`. Confirmed
the anchor `(_lane_configured and bo.engine != "chart_asset")` appears
exactly once (uncapped `str.count`), then mutated it back to
`(_lane_configured and not _escalation_degraded and bo.engine != "chart_asset")`
— restoring the old coupling. Both new tests failed as required:

```
tests/test_gh855_scoring_independent_of_lane_health.py::test_a_latched_page_the_detector_missed_now_gets_scored FAILED
tests/test_gh855_scoring_independent_of_lane_health.py::test_the_latch_does_not_move_text_audit_passed_or_status FAILED
...
E       assert 'table_unexplained_lanes' in []
```

(Assertion 1 fails because the event is never emitted once latched;
assertion 2 fails on its own trailing coverage check for the same reason —
both correctly detect the regression from the same root cause.)

## Verification

- `uvx ruff@0.16.0 format --check .` — clean (whole repo), unpiped, exit 0.
- Rebased onto `origin/main@7cd3752` (main moved twice during this ticket:
  `e207a8a` → `7cd3752`, #856 merged). Collection reconciled against a
  worktree freshly reset to `7cd3752`: main collects 5730; this branch
  collects 5732 (+2, exactly the two new tests).
- `PYTHONPATH=<worktree>/src ~/venvs/socr/bin/pytest tests -q` — full run:
  see the commit this log ships with for the exact pass/xfail counts at the
  final base; no failures at either base measured during this ticket.
- Targeted table/escalation suites also run in isolation before the full
  suite: `test_gh855_scoring_independent_of_lane_health.py`,
  `test_gh96_escalation_lane.py`, `test_gh190_empty_table_surfacing.py`,
  `test_table_not_scorable_scope.py`, `test_gh398_escalation_provenance.py`,
  `test_p35_cold_review_round1/2.py`, `test_gh205_tr3_unconditional_event.py`,
  `test_p5_reread_on_signal.py`, `test_native_only_table_status_gh211.py` —
  all green.

## Out of scope (left untouched, per the ticket)

`escalation_timeout_sec`'s value, the document scope of the
`_escalation_degraded` latch itself, and `ex.shutdown(wait=False)` — parked
for #843/#851.
