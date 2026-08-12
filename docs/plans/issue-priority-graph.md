# Open-issue priority graph

**Do lower wave numbers first.** Wave 1 before Wave 2 before Wave 3…  
Within a wave, top node = highest priority. Same-wave issues can run in parallel.

Criterion: silent content loss → tables/figures → knob honesty → architecture → north-star.

Open in Obsidian: [`issue-priority-graph.canvas`](issue-priority-graph.canvas)

Root index: [`../../TODO.md`](../../TODO.md) · Per-issue plans: [`issue-priority/TICKETS.md`](issue-priority/TICKETS.md)

---

## Start here (single-lane order)

If one person / one agent: work top to bottom. Stop for hard deps only.

1. **#150** charts extracted as tables
2. **#144** rowizer drops numbers
3. **#147** landscape transposed
4. **#146** data row as header
5. **#152** side-by-side merge
6. **#145** one-point overlap deletes prose
7. **#162** verifier fail-open
8. **#166** crop failures look clean
9. **#161** resume skips audit-failed
10. **#140** math-font trusted native
11. **#159** ProviderProfile identity
12. **#151** structure gate *(after 144/146 preferred)*
13. **#167** not every raster is a chart *(after 150)*
14. **#163** OCR deferral *(after 162)*
15. **#154** cloud priced $0 *(after 159)*
16. **#160** escalation budget *(after 154)*
17. **#139** `--no-audit` inert
18. **#168** config/profile dropped
19. **#172** soft timeout hang
20. **#177** exit-code mismatch
21. **#157** equation attach → then **#164**
22. **#165** PUA math *(after 140)*
23. **#158** model_version → then **#173**
24. **#171** figures before terminal → then **#170**
25. **#169** rejection reasons · **#64** fallthrough · **#142** flag audit last in this band
26. **#178** ADR · **#174** quarantine → **#155** split · **#175/#176/#156** alongside
27. Wave 8 is design only (**#49 #39 #114 #127 #56**) — not coding tickets

---

## Schedule graph (left = first)

Read **left → right**. Colored columns are waves. Solid edges = must finish source first. Dotted = nicer after, not blocking.

```mermaid
flowchart LR
  classDef w1 fill:#ef4444,stroke:#7f1d1d,color:#fff
  classDef w2 fill:#f97316,stroke:#9a3412,color:#fff
  classDef w3 fill:#eab308,stroke:#854d0e,color:#111
  classDef w4 fill:#22c55e,stroke:#166534,color:#fff
  classDef w5 fill:#14b8a6,stroke:#0f766e,color:#fff
  classDef w6 fill:#3b82f6,stroke:#1d4ed8,color:#fff
  classDef w7 fill:#a855f7,stroke:#6b21a8,color:#fff
  classDef w8 fill:#64748b,stroke:#334155,color:#fff
  classDef seq fill:#0f172a,stroke:#38bdf8,color:#e0f2fe

  W1((1)):::seq --> W2((2)):::seq --> W3((3)):::seq --> W4((4)):::seq --> W5((5)):::seq --> W6((6)):::seq --> W7((7)):::seq --> W8((8)):::seq

  subgraph WAVE1["Wave 1 FIRST"]
    direction TB
    I150((150)):::w1 --> I144((144)):::w1 --> I147((147)):::w1 --> I146((146)):::w1 --> I152((152)):::w1 --> I145((145)):::w1
  end

  subgraph WAVE2["Wave 2"]
    direction TB
    I162((162)):::w2 --> I166((166)):::w2 --> I161((161)):::w2 --> I140((140)):::w2
  end

  subgraph WAVE3["Wave 3"]
    direction TB
    I159((159)):::w3
  end

  subgraph WAVE4["Wave 4"]
    direction TB
    I151((151)):::w4
    I167((167)):::w4
    I163((163)):::w4
    I154((154)):::w4 --> I160((160)):::w4
    I139((139)):::w4
    I168((168)):::w4
    I172((172)):::w4
    I177((177)):::w4
  end

  subgraph WAVE5["Wave 5"]
    direction TB
    I157((157)):::w5 --> I164((164)):::w5
    I165((165)):::w5
  end

  subgraph WAVE6["Wave 6"]
    direction TB
    I158((158)):::w6 --> I173((173)):::w6
    I171((171)):::w6 --> I170((170)):::w6
    I169((169)):::w6
    I142((142)):::w6
    I64((64)):::w6
  end

  subgraph WAVE7["Wave 7"]
    direction TB
    I178((178)):::w7
    I174((174)):::w7 --> I155((155)):::w7
    I175((175)):::w7
    I176((176)):::w7
    I156((156)):::w7
  end

  subgraph WAVE8["Wave 8 LAST"]
    direction TB
    I49((49)):::w8
    I39((39)):::w8
    I114((114)):::w8
    I127((127)):::w8
    I56((56)):::w8
  end

  W1 --> I150
  W2 --> I162
  W3 --> I159
  W4 --> I151
  W5 --> I157
  W6 --> I158
  W7 --> I178
  W8 --> I49

  I144 -.-> I151
  I146 -.-> I151
  I150 --> I167
  I162 --> I163
  I151 -.-> I163
  I159 --> I154
  I139 -.-> I142
  I154 -.-> I142
  I168 -.-> I142
  I140 --> I165
  I140 -.-> I157
  I161 -.-> I169
  I178 -.-> I155
  I174 -.-> I176
  I175 -.-> I155
  I159 -.-> I114
  I154 -.-> I114
```

---

## Multi-lane (max parallelism)

| Lane | Starts at | Issues in order |
|---|---|---|
| **A Content** | Wave 1 | 150 → 144 → 147 → 146 → 152 → 151 → 167 |
| **B Trust** | Wave 2 | 162 → 166 → 161 → 163 · 140 → 165 |
| **C Routing** | Wave 3 | 159 → 154 → 160 → 142 |
| **D CLI** | Wave 4 | 139 · 168 · 172 · 177 |
| **E Equations** | Wave 5 | 157 → 164 |
| **F Provenance** | Wave 6 | 158 → 173 · 171 → 170 · 169 · 64 |
| **G Arch** | Wave 7 | 178 · 174 → 155 · 175 · 176 · 156 |

Lanes A/B/C can start together on day one. D waits only on people, not deps. E needs 140 for 165. F/G after the product-honesty spine is moving.

---

## Node key

| ID | Title | Wave | Rank in wave |
|---|---|---|---|
| 150 | charts extracted as tables | 1 | 1 |
| 144 | rowizer drops numeric values | 1 | 2 |
| 147 | landscape transposed | 1 | 3 |
| 146 | data row as header | 1 | 4 |
| 152 | side-by-side tables merged | 1 | 5 |
| 145 | one-point overlap deletes prose | 1 | 6 |
| 162 | verifier exceptions fail open | 2 | 1 |
| 166 | crop failures look clean | 2 | 2 |
| 161 | resume skips audit-failed | 2 | 3 |
| 140 | math-font trusted native | 2 | 4 |
| 159 | ProviderProfile identity discarded | 3 | 1 |
| 151 | recall is not a structure gate | 4 | 1 |
| 167 | any raster routes to chart | 4 | 2 |
| 163 | OCR word defers scan gate | 4 | 3 |
| 154 | cloud priced at $0 | 4 | 4 |
| 160 | escalation ignores budget | 4 | 5 |
| 139 | `--no-audit` inert | 4 | 6 |
| 168 | config/profile dropped | 4 | 7 |
| 172 | soft timeout hang | 4 | 8 |
| 177 | single-file exit codes | 4 | 9 |
| 157 | equation attach needs PageOutput | 5 | 1 |
| 165 | PUA math miss | 5 | 2 |
| 164 | full-page dump on reject | 5 | 3 |
| 158 | blank model_version | 6 | 1 |
| 173 | fingerprint incomplete | 6 | 2 |
| 171 | figures before terminal | 6 | 3 |
| 170 | replay ignores assets | 6 | 4 |
| 169 | drop rejection reasons | 6 | 5 |
| 142 | full flag audit | 6 | 6 |
| 64 | audit native fallthrough | 6 | 7 |
| 178 | Python-not-Rust ADR | 7 | 1 |
| 174 | quarantine legacy | 7 | 2 |
| 175 | package layering | 7 | 3 |
| 176 | dumb DocumentState | 7 | 4 |
| 155 | split orchestrator | 7 | 5 |
| 156 | TODO/TICKETS drift | 7 | 6 |
| 49 | three-layer method | 8 | design |
| 39 | quality-per-dollar | 8 | design |
| 114 | post-hoc escalate | 8 | design |
| 127 | native structure loss | 8 | design |
| 56 | CE umbrella tracker | 8 | tracker |
