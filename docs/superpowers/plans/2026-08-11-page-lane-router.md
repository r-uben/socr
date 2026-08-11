# Page-lane router Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract a first-class per-page router that chooses native PDF reading vs OCR LLM (vs chart-asset), wire the agentic loop through it, and make the decision auditable.

**Architecture:** Pure `decide_page_lane` in `pipeline/page_router.py`; orchestrator wrappers preserve existing predicates; agentic loop branches on `PageLane` and emits `page_lane` audit events. OCR provider escalation stays in `route_page` / `route_ocr_provider`.

**Tech Stack:** Python 3.11+, pytest, existing socr `PageState` / `PipelineConfig` / audit log.

## Global Constraints

- Behavior-preserving vs current `_is_trusted_native_without_ocr` + chart lane policy.
- Stage by name; one logical commit series; `uvx ruff@0.16.0 format --check .` for format gate.
- Tests: `~/venvs/socr/bin/pytest`.

---

## File map

| File | Responsibility |
|------|----------------|
| `src/socr/pipeline/page_router.py` | `PageLane`, `PageRouteDecision`, `decide_page_lane` |
| `src/socr/pipeline/orchestrator.py` | Delegate predicates; agentic branch + audit |
| `src/socr/pipeline/agentic.py` | Alias `route_ocr_provider`; docstring clarity |
| `tests/test_page_router.py` | Policy matrix unit tests |
| `docs/ARCHITECTURE.md`, `README.md` | Two-level routing docs |
| `docs/log/2026-08-11_page-lane-router.md` | Decision log |

---

### Task 1: Failing tests for `decide_page_lane`

- [ ] Write `tests/test_page_router.py` covering NATIVE / OCR / CHART / native_only / no native_first / tables / enhancement.
- [ ] Confirm import fails / tests fail before implementation.

### Task 2: Implement `page_router.py`

- [ ] Add module with enum, decision dataclass, reason constants, `decide_page_lane`.
- [ ] Run `tests/test_page_router.py` green.

### Task 3: Wire orchestrator + audit

- [ ] Refactor `_is_trusted_native_without_ocr` / `_is_chart_asset_page` to call the router.
- [ ] In `_phase_agentic`, decide lane once; branch; append `page_lane` AuditEvent.
- [ ] Keep chart PNG / native / OCR ladder behavior identical.

### Task 4: Alias + docs

- [ ] `route_ocr_provider = route_page` in `agentic.py`.
- [ ] Update ARCHITECTURE + README modality-vs-provider wording.
- [ ] Write `docs/log/2026-08-11_page-lane-router.md`.

### Task 5: Verify

- [ ] `~/venvs/socr/bin/pytest tests/test_page_router.py tests/test_chart_lane.py tests/test_pp2_agentic_fuse.py -q`
- [ ] Broader agentic/native subset if needed.
- [ ] `uvx ruff@0.16.0 format --check` on touched paths.
