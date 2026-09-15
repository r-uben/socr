# STATUS — backlog fixes

> **Current truth, 2026-09-15.** First ticket in this folder. GH-249 dispatched, implemented,
> reviewed (one REVISE round), and fixed. Purpose: measure the real cycle time and failure rate
> of fixing one confirmed defect end-to-end, before deciding whether the remaining 58 are worth
> automating the same way.

## Live
- **GH-249** — DONE. Grid gate implemented on `fix/249-verifier-grid-gate-v2`
  (`native_verifier.py` + `test_native_table_verifier.py`, plus a required fixture update in
  `test_agentic.py`, `test_gh259_flagged_model_table_wins.py`, `test_source_evidence_table_judge.py`
  — single-native-row fixtures no longer establish a grid). All 4 acceptance criteria verified;
  see `docs/log/2026-09-15_249.md`.

## Active Agents
| Agent | Ticket | Scope | Status |
| --- | --- | --- | --- |
| socr-implementer | GH-249 | `src/socr/tables/native_verifier.py`, `tests/test_native_table_verifier.py` | DONE |

## Next action
Dispatch `socr-reviewer` on the diff, then wait for CI green before merging.
