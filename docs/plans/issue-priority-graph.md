# Open-issue priority graph

41 open issues ranked for parallel development.
Criterion: silent content loss → tables/figures → knob honesty → architecture → north-star.

**Legend**

- Solid arrow = hard dependency (do source first)
- Dotted arrow = soft dependency (easier / cleaner after)
- Waves are numbered 1–8; same-wave nodes can run in parallel unless an arrow says otherwise

Open in Obsidian: [`issue-priority-graph.canvas`](issue-priority-graph.canvas)

```mermaid
flowchart TB
  classDef w1 fill:#ef4444,stroke:#7f1d1d,color:#fff
  classDef w2 fill:#f97316,stroke:#9a3412,color:#fff
  classDef w3 fill:#eab308,stroke:#854d0e,color:#111
  classDef w4 fill:#22c55e,stroke:#166534,color:#fff
  classDef w5 fill:#14b8a6,stroke:#0f766e,color:#fff
  classDef w6 fill:#3b82f6,stroke:#1d4ed8,color:#fff
  classDef w7 fill:#a855f7,stroke:#6b21a8,color:#fff
  classDef w8 fill:#64748b,stroke:#334155,color:#fff

  NOW((NOW))

  subgraph WAVE1["Wave 1 - Destroy content"]
    direction LR
    I150((150))
    I144((144))
    I147((147))
    I146((146))
    I152((152))
  end

  subgraph WAVE2["Wave 2 - Fail closed"]
    direction LR
    I162((162))
    I166((166))
    I161((161))
    I140((140))
  end

  subgraph WAVE3["Wave 3 - Routing identity"]
    direction LR
    I159((159))
  end

  subgraph WAVE4["Wave 4 - Gates and honesty"]
    direction LR
    I151((151))
    I167((167))
    I163((163))
    I154((154))
    I160((160))
    I139((139))
    I168((168))
    I172((172))
    I177((177))
  end

  subgraph WAVE5["Wave 5 - Equations"]
    direction LR
    I157((157))
    I165((165))
    I164((164))
  end

  subgraph WAVE6["Wave 6 - Provenance"]
    direction LR
    I158((158))
    I173((173))
    I171((171))
    I170((170))
    I169((169))
    I142((142))
    I64((64))
  end

  subgraph WAVE7["Wave 7 - Architecture"]
    direction LR
    I178((178))
    I174((174))
    I175((175))
    I176((176))
    I155((155))
    I156((156))
  end

  subgraph WAVE8["Wave 8 - North star"]
    direction LR
    I49((49))
    I39((39))
    I114((114))
    I127((127))
    I56((56))
  end

  NOW --> I150
  NOW --> I144
  NOW --> I147
  NOW --> I146
  NOW --> I152
  NOW --> I162
  NOW --> I166
  NOW --> I161
  NOW --> I140
  NOW --> I159

  I144 -.-> I151
  I146 -.-> I151
  I150 --> I167
  I162 --> I163
  I151 -.-> I163

  I159 --> I154
  I154 --> I160
  I139 -.-> I142
  I154 -.-> I142
  I168 -.-> I142

  I140 --> I165
  I157 --> I164
  I140 -.-> I157

  I158 --> I173
  I161 -.-> I169
  I171 --> I170

  I178 -.-> I155
  I174 --> I155
  I174 -.-> I176
  I175 -.-> I155

  I159 -.-> I114
  I154 -.-> I114
  I49 -.-> I151
  I49 -.-> I162
  I56 -.-> I150
  I56 -.-> I144

  class I150,I144,I147,I146,I152 w1
  class I162,I166,I161,I140 w2
  class I159 w3
  class I151,I167,I163,I154,I160,I139,I168,I172,I177 w4
  class I157,I165,I164 w5
  class I158,I173,I171,I170,I169,I142,I64 w6
  class I178,I174,I175,I176,I155,I156 w7
  class I49,I39,I114,I127,I56 w8
```

## Node key

| ID | Title | Wave |
|---|---|---|
| 150 | charts extracted as tables | 1 |
| 144 | rowizer drops numeric values | 1 |
| 147 | landscape transposed | 1 |
| 146 | data row as header | 1 |
| 152 | side-by-side tables merged | 1 |
| 162 | verifier exceptions fail open | 2 |
| 166 | crop failures look clean | 2 |
| 161 | resume skips audit-failed | 2 |
| 140 | math-font trusted native | 2 |
| 159 | ProviderProfile identity discarded | 3 |
| 151 | recall is not a structure gate | 4 |
| 167 | any raster routes to chart | 4 |
| 163 | OCR word defers scan gate | 4 |
| 154 | cloud priced at $0 | 4 |
| 160 | escalation ignores budget | 4 |
| 139 | `--no-audit` inert | 4 |
| 168 | config/profile dropped | 4 |
| 172 | soft timeout hang | 4 |
| 177 | single-file exit codes | 4 |
| 157 | equation attach needs PageOutput | 5 |
| 165 | PUA math miss | 5 |
| 164 | full-page dump on reject | 5 |
| 158 | blank model_version | 6 |
| 173 | fingerprint incomplete | 6 |
| 171 | terminal before figures | 6 |
| 170 | replay ignores assets | 6 |
| 169 | drop rejection reasons | 6 |
| 142 | full flag audit | 6 |
| 64 | audit native fallthrough | 6 |
| 178 | Python-not-Rust ADR | 7 |
| 174 | quarantine legacy | 7 |
| 175 | package layering | 7 |
| 176 | dumb DocumentState | 7 |
| 155 | split orchestrator | 7 |
| 156 | TODO/TICKETS drift | 7 |
| 49 | three-layer method | 8 |
| 39 | quality-per-dollar | 8 |
| 114 | post-hoc escalate | 8 |
| 127 | native structure loss | 8 |
| 56 | CE umbrella tracker | 8 |

## Critical paths

1. Citation-safe tables: `(144|146|147|152) -> 151` and `(162|166) -> 163`, plus `150 -> 167`
2. Honest agentic product: `159 -> 154 -> 160 -> 142` and `174 -> 155`
3. Equations: `140 -> 165` and `157 -> 164`
