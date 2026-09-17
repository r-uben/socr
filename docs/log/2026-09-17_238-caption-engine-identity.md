# 2026-09-17 — GH-238: fingerprint records the caption engine actually selected

## What changed

`_run_fingerprint` recorded `figure_caption_fallback_model` = `cfg.gemini_model` (pure config),
but `_get_vision_engine` returns three observably different caption producers depending on runtime
reachability: `LocalFirstFigureEngine(OllamaFigureEngine, gemini_fallback)` (Ollama up),
`GeminiAPIEngine` alone (Ollama down, an API key present), or `None` (neither reachable). Caption
bytes differ across all three, so a document OCR'd with Ollama and resumed without it could reuse
sidecars whose captions came from a different model.

The issue's stated blocker ("recording the resolved engine means probing availability from inside
an otherwise-pure fingerprint function") was already false: `_run_fingerprint` calls
`self._resolve_judge_model()` for exactly this reason, with the comment "Availability-dependent BY
DESIGN: a different judge is a different run, and the ledger must say so." This fix follows that
established pattern rather than opening a design question.

- `src/socr/pipeline/orchestrator.py`
  - Added `CAPTION_IDENTITY_NONE = "no-caption-engine"`, a literal sentinel (not `""`/`None`)
    matching `JUDGE_IDENTITY_HEURISTIC`'s shape: it must be distinguishable from the field's own
    `None`, which now means "`describe_figures` was off, the resolver was never called."
  - Added `_caption_engine_identity_cache: str | bool = False`, a class-level memoization slot
    mirroring `_judge_model_cache`.
  - Added `_resolve_caption_engine_identity()`: reproduces `_get_vision_engine`'s reachability
    checks (Ollama `/api/tags` probe, then `GEMINI_API_KEY`/`GOOGLE_API_KEY` presence +
    `strict_local`/zero-cap-pinned policy) without constructing engines or printing, and memoizes
    the result for the pipeline's lifetime. Returns `"ollama:<model>"`, `"gemini:<model>"`, or
    `CAPTION_IDENTITY_NONE`.
  - `_run_fingerprint`'s `figure_caption_fallback_model` now reads
    `self._resolve_caption_engine_identity() if cfg.describe_figures else None` — unchanged `None`
    when captions are off (no probe, matching the judge's `--judge-backend heuristic`
    short-circuit), the resolved identity otherwise.

- `tests/test_gh238_caption_engine_identity.py` (new): the three required evidence patterns —
  reachability pairs differ (Ollama-vs-Gemini, Ollama-vs-none, Gemini-vs-none), the converse (no
  engine reachable, `gemini_model` changed → fingerprints agree), and `describe_figures=False`
  never probes and the field stays the pre-existing `None`. Plus: the sentinel is not the bare
  `None` used by the off-switch, memoization (1 probe across 3 fingerprint calls), and the
  `object.__new__` lazy-attribute-default trap.

- `tests/test_fingerprint_flag_coverage.py`: `test_caption_fallback_model_invalidates_outside_
  enabled_engines` no longer holds unconditionally now that the field is the resolved identity, not
  raw `gemini_model` — updated to pin the reachability (Ollama down, API key present) that makes
  Gemini the resolved engine, with a docstring explaining why. The converse test
  (`describe_figures=False`) needed no change: the resolver is never called in that branch.

## Probe cost

At most one `OllamaFigureEngine.is_available()` HTTP call per `UnifiedPipeline` instance — on the
first `describe_figures=True` fingerprint call — never per page. `_run_fingerprint` runs once per
page via `_flush_page_sidecar`; the memoization cache (`_caption_engine_identity_cache`) makes every
subsequent call reuse the cached identity, mirroring `_resolve_judge_model`/`_judge_model_cache`.
No probe at all when `describe_figures=False` (the common case). The identity resolver never calls
`GeminiAPIEngine.initialize()` (which itself makes a live HTTP round trip) — it checks API-key
presence and cloud policy only, the same fidelity the field always had for the Gemini case.

## Tests

Hermetic — no ollama, no provider. `OllamaFigureEngine.is_available` is patched directly (not the
orchestrator's probe caller) so tests are deterministic regardless of whether the dev machine
actually has Ollama running. `GEMINI_API_KEY`/`GOOGLE_API_KEY` are cleared by an autouse fixture and
set only where a test needs Gemini reachable.

Mutation-proven, canary confirmed each mutant copy's `socr.__file__` resolved inside the mutant
(`os.path.realpath`), full archive copy (`git archive HEAD`) including `pyproject.toml` so its
`pythonpath = ["src"]` cannot shadow an external `PYTHONPATH` pointed at a bare `src/`+`tests/` copy:

1. Reverted `figure_caption_fallback_model` back to `cfg.gemini_model if cfg.describe_figures else
   None` (the original bug) → 6 of 9 new tests redden: all three reachability-pair tests, the
   converse (`no_engine_reachable_ignores_gemini_model`), the sentinel-not-None test, and the
   memoization test (this mutant never calls the resolver, so the probe-count assertion no longer
   applies the way it should — its failure mode differs but it correctly reddens). 19 passed
   (unaffected: `describe_figures=False` tests + pre-existing suite tests still describing the old
   coupling).
2. Removed the `_caption_engine_identity_cache` short-circuit (always re-resolve) →
   `test_resolution_is_memoized_across_fingerprint_calls` reddens: 3 calls instead of the asserted
   1. The other 8 tests in the file stay green — memoization is the only thing that guard covers.
3. Reverted the `describe_figures` gate to call the resolver unconditionally →
   `test_describe_figures_false_never_probes` and `test_describe_figures_false_field_is_none`
   redden (field becomes `CAPTION_IDENTITY_NONE` instead of `None`, and the probe fires). 7 of 9
   pass (unaffected: the reachability/converse/memoization tests, which all set
   `describe_figures=True`).

Full suite, measured directly (not from the teammate's independent re-run), `git archive HEAD`
snapshots at `/tmp/gh238-before` (main@7893044, unmodified) and `/tmp/gh238-after` (archive +
this change copied over):

- Before: 5626 passed, 1 skipped, 4 xfailed (272.98s)
- After: 5635 passed, 1 skipped, 4 xfailed (223.16s)
- Delta: +9 passed, 0 skipped/xfailed delta — exactly the 9 new tests in
  `test_gh238_caption_engine_identity.py`; no regression, no xfail/skip count shift.

Also ran the narrower `tests/test_orchestrator.py`, `tests/test_canon_remediation.py`,
`tests/test_cli_flag_agentic_status_gh142.py`, `tests/test_gh142_flag_audit.py` directly against
the worktree: 172 passed.

## Deviation from the ticket's framing

None found. The ticket's own claim that `_run_fingerprint` is "otherwise a pure function of config"
being false was independently confirmed by reading `orchestrator.py:1164` before writing any code.
