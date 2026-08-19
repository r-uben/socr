# STATUS — overnight autonomous issue sweep

Last updated: 2026-08-20

## Stage

Plan authored and panel-critiqued; **not yet launched**. The ticket graph in
`TICKETS.md` is the durable spec. Three independent critics (Grok on
gating-safety, GPT on coverage, Gemini on ticket-size and trust) attacked the
first draft; their raw output is in `logs/panel-raw/` and what was taken or
rejected is in `logs/2026-08-20_panel-synthesis.md`.

One decision is outstanding before launch: `tracker_mode` (see below).

## Base state (clean before tickets)

- Repo: socr, forge GitHub (`git@github.com:r-uben/socr.git`).
- `origin/main` at `53b0637` — includes today's #250 merge (glyph gate fix).
- This plan lives on branch `docs/overnight-issue-sweep`, cut from `origin/main`.
- The main checkout is on another session's branch and must not be touched.
- Open issues at authoring time: ~62.

## Outstanding decision

**`tracker_mode`** — `staged` (default) or `direct`, set in `state/config.json`.
All three critics independently argued against unattended issue closure; the
owner has granted it. Staged keeps the grant intact but routes closes through a
one-command morning apply. Nothing else in the graph changes either way.

## Ticket board

| Ticket | Stream | Status | depends-on | Wave |
|--------|--------|--------|------------|------|
| A1 | preflight | TODO | — | 0 |
| A2 | preflight | TODO | A1 | 0 |
| B1–B4 | triage | TODO | A1,A2 | 1 |
| V1 | evidence check | TODO | B1–B4 | 1.5 |
| C1–C4 | adjudication | TODO | V1 | 2 |
| D0 | tracker manifest | TODO | C1–C4,A1 | 3 |
| D1–D3 | tracker actions | TODO | D0 | 3 |
| E0 | fix scheduling | TODO | C1–C4 | 4 |
| E1–E7 | fixes (one owner, stacked) | TODO | E0 + predecessor | 4 |
| W1–W5 | coordinator | TODO | per wave | — |
| F1 | morning report | TODO | all D,E terminal | 5 |

## Dispatch waves

- Wave 0: A1 → A2 (serial, one agent)
- Wave 1: B1–B4 in parallel, two vendors each, wall-clock capped
- Wave 1.5: V1 (deterministic script, no model)
- Wave 2: C1–C4 in parallel, adjudicator ≠ triager, cluster-colocated
- Wave 3: D0 → D1,D2,D3 (serial writer)
- Wave 4: E0 → E1…E7 (single code owner, stacked branches)
- Wave 5: F1

## Next action

Owner sets `tracker_mode`, then dispatch wave 0 (A1).
