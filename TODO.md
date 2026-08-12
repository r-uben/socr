# socr — TODO

GitHub is the source of truth for open work. This file is a **pointer index** only.
Per-issue plans: [`docs/plans/issue-priority/TICKETS.md`](docs/plans/issue-priority/TICKETS.md)
Schedule graph: [`docs/plans/issue-priority-graph.md`](docs/plans/issue-priority-graph.md) · Obsidian canvas: [`docs/plans/issue-priority-graph.canvas`](docs/plans/issue-priority-graph.canvas)

Last updated: 2026-08-12 · 41 open issues · do **lower wave numbers first**; within a wave, top first.

**Criterion:** silent content loss → tables/figures → knob honesty → architecture → north-star.

---

## Now — Wave 1 (destroy content)

- [ ] **#150** charts extracted as tables (worst corpus pages) — start here
- [ ] **#144** rowizer drops numeric values
- [ ] **#147** landscape pages transposed
- [ ] **#146** data row emitted as header
- [ ] **#152** side-by-side tables merged
- [ ] **#145** one-point table overlap deletes prose

## Next — Wave 2 (fail closed)

- [ ] **#162** table verifier exceptions fail open
- [ ] **#166** all-failed crop rereads look clean
- [ ] **#161** resume skips judge-rejected SUCCESS
- [ ] **#140** math-font pages trusted native without audit

## Next — Wave 3 (routing identity)

- [ ] **#159** ProviderProfile identity discarded (cloud rung runs local)

## Then — Wave 4 (gates & honesty)

- [ ] **#151** recall ≠ structure gate *(after #144/#146)*
- [ ] **#167** any raster ≠ chart *(after #150)*
- [ ] **#163** OCR word must not defer scan gate *(after #162)*
- [ ] **#154** `--max-cost-per-page` vs qwen-cloud $0 *(after #159)*
- [ ] **#160** table escalation ignores budget *(after #154)*
- [ ] **#139** `--no-audit` inert on agentic
- [ ] **#168** `--config`/`--profile` values dropped
- [ ] **#172** soft timeouts leave workers that block exit
- [ ] **#177** single-file exit codes ≠ partial policy

## Then — Wave 5 (equations)

- [ ] **#157** `--recover-clean-equations` skips without PageOutput
- [ ] **#165** PUA-only math skips recovery *(after #140)*
- [ ] **#164** rejected recovery dumps full-page text *(after #157)*

## Then — Wave 6 (provenance)

- [ ] **#158** populate `model_version` in fingerprints
- [ ] **#173** fingerprint omits `auto_patch_tables` / equation models *(after #158)*
- [ ] **#171** terminal sidecars before figure provenance
- [ ] **#170** replay ignores visual assets *(after #171)*
- [ ] **#169** manifests drop non-empty reject reasons
- [ ] **#142** audit every CLI flag vs agentic *(after known liars)*
- [ ] **#64** audit tabular-looking native fallthrough

## Then — Wave 7 (architecture)

- [ ] **#178** ADR: stay Python / optional native kernels only after profiling
- [ ] **#174** quarantine legacy; agentic first-class only *(before #155)*
- [ ] **#175** break inverted package layering
- [ ] **#176** dumb DocumentState + one text selector *(soft after #174)*
- [ ] **#155** split `orchestrator.py` god-module *(after #174)*
- [ ] **#156** TODO/TICKETS drift policy *(this rewrite)*

## Later — Wave 8 (design / north-star)

- [ ] **#49** three-layer method ADR (extract / verify / escalate)
- [ ] **#39** quality-per-dollar calibration (`calibration.lock.json`) — needs GT
- [ ] **#114** `socr escalate` post-hoc pass (design)
- [ ] **#127** native headings/emphasis/lists/links
- [ ] **#56** CE umbrella tracker — not a unit of work; tracks Waves 1–5

---

## Parallel lanes (if multiple agents)

| Lane | Start | Order |
|------|-------|-------|
| A Content | Wave 1 | #150 → #144 → #147 → #146 → #152 → #145 → #151 → #167 |
| B Trust | Wave 2 | #162 → #166 → #161 → #163 · #140 → #165 |
| C Routing | Wave 3 | #159 → #154 → #160 → #142 |
| D CLI | Wave 4 | #139 · #168 · #172 · #177 |
| E Equations | Wave 5 | #157 → #164 |
| F Provenance | Wave 6 | #158 → #173 · #171 → #170 · #169 · #64 |
| G Arch | Wave 7 | #178 · #174 → #155 · #175 · #176 · #156 |

## Done recently (pointers only)

- See closed GitHub issues and `docs/log/`. Do not relist closed work here.

## Policy

1. Every checkbox above **must** link an open GitHub issue.
2. Implementation detail lives in `docs/plans/issue-priority/TICKETS.md`, not here.
3. When an issue closes, check it off here and mark the ticket DONE in the same PR.
