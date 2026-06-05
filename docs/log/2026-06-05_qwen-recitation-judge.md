# Dev log — Qwen engine, RECITATION safeguard, hard-page VLM judge (2026-06-05)

## Context
Owner reads a lot of academic papers (econ/finance), mostly born-digital PDFs with a
minority of scanned/canonical ones. Started from "I don't trust this OCR pipeline."

## What shipped this session (branch `feat/qwen-engine`, PR #28 → `refactor/unified-page-contract`)
1. **Qwen-VL engine** — new external sibling CLI `../ocr/qwen-ocr-cli` (own repo, gitdir
   `~/projects/qwen-ocr-cli.git`, venv `~/venvs/qwen-ocr-cli`, installed globally via
   `uv tool install`). 3 backends (ollama/vllm/api) + smart selector + `--model`. `EngineType.QWEN`
   in socr leads `_LOCAL_ENGINE_ORDER`, in providers ladder.
2. **Default OCR model = `qwen3.5:cloud`** (Ollama Cloud, ~0.57 socOCRbench, ~40–90s/pg, no extra
   key). Offline: `--qwen-model qwen3-vl:8b` (0.47, ~136s/pg, times out on dense pages). `--qwen-model` flag.
3. **RECITATION safeguard** — Gemini's copyright filter returns empty (`finish_reason=RECITATION`)
   on famous papers it memorized; socr detects it (`FailureMode.RECITATION`), auto-escalates that page
   to qwen (repair routes RECITATION → open model, never back to Gemini), and **surfaces it in output**
   ("Gemini refused (RECITATION) → recovering via qwen"). Fixed a latent bug: error PageOutputs
   defaulted `audit_passed=True` → skipped repair when whole backbone failed.
4. **DPI 200 → 300** default (small table digits; `--dpi` overrides).
5. **VLM judge on HARD pages (Phase 3b)** — on table/equation pages, a vision model (auto-resolved:
   qwen3.5:cloud → minicpm-v → qwen3-vl, prefers one ≠ OCR engine) checks OCR vs the page IMAGE for
   semantic corruption (wrong digits/signs/columns) heuristics can't see; rejects → repair. On by
   default, `--no-judge-hard-pages` to disable, fail-open. **Validated:** accepted faithful table OCR,
   rejected 12/4173-char corruption and named the exact misreads.

Commits: c3bcb6a, 036c11b, b5ec830, 1920f74, 3ad7125, b6d634c, 49e3b56.

## Key findings (evidence-driven; see [[reference-sococrbench]] memory)
- socOCRbench: socr's old local tier (GLM 0.37, DeepSeek 0.09) were the worst models; Qwen3.5-VL
  (~0.55–0.58) is the best open OCR. Gemini 3.x best overall (0.60–0.64) **but refuses famous papers**.
- "Newer ≠ better": Qwen3.6 (0.28–0.39) worse for OCR than Qwen3-VL 8B (0.47); Qwen3.7 cloud-only.
- On the owner's 64GB Mac: local qwen3.5:27b TIMED OUT (>300s/pg); local qwen3-vl:8b times out on dense
  pages. **Cloud-via-Ollama (`qwen3.5:cloud`) is the practical sweet spot.**
- Panel (Codex+Gemini) "is this the best?" verdict: near ceiling for page-level OCR, not for
  academic-paper reading. Their flashy recs (olmOCR, anchoring) DID NOT survive the owner's data:
  olmOCR Ollama package broken (garbage output); anchoring marginal on born-digital + 3× slower + can't
  help scans (no native text). The unglamorous **hard-page judge** was the real win.

## Gotchas / environment
- **Run socr via**: `export UV_PROJECT_ENVIRONMENT=~/venvs/socr` (direnv `.envrc` does this interactively).
  The in-iCloud `.venv` is a STALE BROKEN leftover — `uv run socr` 500s without the env var. In
  non-interactive shells use `PYTHONPATH=src .venv/bin/python -m pytest` after `uv pip install --python
  .venv/bin/python pytest`, OR the external env.
- iCloud repo: gitdir separated (`~/projects/socr.git`); never improvise iCloud git/venv (see `~/.claude/rules/icloud.md`).
- Tests: 491 pass. Lint: ruff.

## Open / next (all optional, owner's call)
1. **Dual-pass table extraction** (owner's idea, #3): judge now DETECTS bad tables; dual-pass would
   EXTRACT them precisely — crop table bbox (`find_tables()` gives bboxes; `extract_structured()` already
   crops) → table-specialized VLM pass → patch into page markdown; disagreement between whole-page and
   crop pass = built-in corruption flag. Natural next build if table *quality* (not just detection) matters.
2. **Merge PR #28** into `refactor/unified-page-contract` (qwen work depends on that unmerged branch;
   the refactor → main landing is a separate, bigger decision).
3. `qwen-ocr-cli` has **no GitHub remote** (local-only). Create one if backup wanted.
4. Document anchoring as opt-in `--anchor` flag (low priority — tested marginal).
5. Durable per-run audit log of RECITATION/judge rejections (currently runtime + manifest only).

## Branch state
- socr: `feat/qwen-engine` pushed, PR #28 open into `refactor/unified-page-contract`. Clean.
- qwen-ocr-cli: branch `main`, 4 commits, no remote.
