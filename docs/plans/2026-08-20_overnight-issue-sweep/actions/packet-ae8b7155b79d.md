# Review packet — action ae8b7155b79d

- **Issue**: #220 — https://github.com/r-uben/socr/issues/220
- **Kind**: **COMMENT**  (issue STAYS OPEN; comment only)
- **Disposition**: PARTIALLY-IMPLEMENTED  (UNANIMOUS)
- **Adjudicator (may NOT review this)**: grok  ·  batch batch-2
- **Triage seats for this batch (may NOT review this)**: ollama-deepseek, gemini-flash

## Per-seat triage verdicts (machine-checked)

| seat | verdict | evidence_verified | fixed_by_commit |
|---|---|---|---|
| gemini-flash | ALREADY-FIXED | NO | 0c5bbd11007d1aa038f040242c006a64d1a4483c |
| ollama-deepseek | ALREADY-FIXED | NO | 1216f2eb9f0c29e6cf6d0be66d17c13f2a7b6eed |

## Basis given by the adjudicator

Both seats claimed ALREADY-FIXED; the machine check already rejected that (deepseek's own table marks the optional gate-filter UNMET; gemini's fixed_by_commit never touched the cited ranges). Viewer exists (review(), build_review_html, _BOOL_SIGNAL_KEYS). CLI options are only doc_dir/pdf/output/scale/quality. `n` jumps to the next page with any signal — that is not 'filter to pages a given gate fired on'. 4 of 5 met. Do not close.

## Risk if this is wrong

Closing would drop the gate-filter the owner listed for calibration passes; leaving it open when they consider the optional item waived is only a tracker nit.

## Evidence cited (already resolved against main_sha by bin/verify_citations.py)

[
 {
  "path": "src/socr/cli.py",
  "line": 1037,
  "snippet": "def review("
 },
 {
  "path": "src/socr/cli.py",
  "line": 1044,
  "snippet": "\"\"\"Build a side-by-side page-image/markdown page for hand judgement (GH-220).\"\"\""
 },
 {
  "path": "src/socr/review/html.py",
  "line": 48,
  "snippet": "_BOOL_SIGNAL_KEYS = ("
 },
 {
  "path": "src/socr/review/html.py",
  "line": 246,
  "snippet": "def build_review_html(report: ReviewReport, *, title: str | None = None) -> str:"
 },
 {
  "path": "src/socr/review/html.py",
  "line": 514,
  "snippet": "  else if(k==='n'){ const nx = PAGES.findIndex((p,ix)=>ix>cur && p.signals.length);"
 }
]

## EXACT text proposed for posting

        Overnight adjudication at 53b0637: socr review exists (cli.py:1037, review/html.py) and covers side-by-side page image vs pages/NNN.md, sidecar header signals, and prev/next. The explicitly-optional acceptance item is still unmet: there is no filter for 'only pages a given gate fired on' (CLI is doc_dir/pdf/output/scale/quality only; `n` jumps to the next page with any recorded signal). Disposition is PARTIALLY-IMPLEMENTED, not a close. Stays OPEN.
