# 2026-08-30 — Table judge ladder: CLI-judged, raster-witnessed table acceptance (GH-353)

Design session (no code). Supersedes the #326/#330 raster-binder direction with a smaller
mechanism. Companion issue: [GH-353](https://github.com/r-uben/socr/issues/353).

## Problem

The ~20 open table issues collapse into four root causes:

1. **Acceptance is text-side, not image-side.** Every gate (recall %, word counts,
   delimiter shape) compares emitted text against other text; nothing looks at the page
   raster. Destroyed structure at 100% recall (#151), dropped values (#144), lost row
   labels (#331), wrecked headers (#146/#215) and fabricated cells (#270) all ship
   SUCCESS because the proxies cannot see them.
2. **Structure is judged, never verified.** Grid correctness is left to VLM judges, and
   judge agreement is not corroboration: in the GH-273 case two vendor models both
   blessed a table whose 50 values were all present with only the row/column binding
   shifted.
3. **Table-ness is decided too cheaply.** Figures (#150), charts (#189/#249) and book
   indexes (#213) are routed into table reconstruction because table-ness is inferred
   from text-layer geometry, not the image.
4. **Guards are per-bug patches at many exits, not one gate.** #301/#302/#303 are three
   different doors the same empty table walked through. Patching per-door does not
   converge; a single choke point does.

Additionally the **native lane ships unwitnessed**: native-trusted pages bypass every
model and judge, so content dies silently under SUCCESS.

## Decision

One quality gate at a single choke point. Every emitted table — **native lane included**
— is judged against its page crop by a ladder of subscription-backed CLI judges. A table
that cannot be verified does not ship SUCCESS; it demotes to an explicit terminal status.
This inverts the current default (unverifiable = trusted).

### Judge input

**Crop image + emitted markdown. Nothing else.**

*Why:* keeps the judge independent of extractor internals (word geometry, lane
metadata), so it works identically on native, qwen, or any future engine's output, and
cannot inherit an extractor's framing of the page.

### The ladder

```
table → CLI₁ (ollama cloud VLM, off-family) → CLI₂ (gemini CLI) → terminal
```

- **CLI₁** — ollama cloud vision model, **off-family from the qwen extractor** (e.g.
  glm-v class). *Why off-family:* a qwen judge shares the qwen extractor's blind spots
  and rubber-stamps its mistakes; error correlation between extractor and judge defeats
  the gate.
- **CLI₂** — gemini CLI on the Google One AI Pro subscription. *Why:* best available
  document-vision family, different from both qwen and CLI₁, scriptable, $0 marginal
  cost, quota otherwise idle. (Antigravity reaches the same models but is an IDE, not a
  headless per-page surface; the gemini CLI is the scriptable route to the same family.)
- Both judges are CLI subprocesses, matching socr's engines-are-CLIs architecture.

### Two successes per rung

- **S1 — the judge answered:** process ran, strict JSON parsed, in time.
  ¬S1 = timeout, crash, garbage output, quota exhaustion.
- **S2 — the judge approved:** verdict says the table matches the crop. Only meaningful
  when S1 holds.

| outcome | S1 | S2 | action |
| --- | --- | --- | --- |
| A | ✓ | ✓ | accept (record witnessing rung) — subject to confidence, below |
| B | ✓ | ✗ | escalate as **tiebreak**: next rung sees the findings |
| C | ✗ | — | escalate as **substitute**: next rung gets fresh eyes, no prior verdict |

*Why B ≠ C:* they exhaust to different terminals with different retry semantics, and a
B-escalation carries the complaint payload while a C-escalation must not bias the next
judge with a verdict that was never produced.

### Terminals — two distinct failure states

- **`TABLE_REJECTED`** — ladder exhausted on B: models looked and said no. Content
  problem. **Not retryable** (same table gets the same verdict).
- **`TABLE_UNVERIFIED`** — ladder exhausted on C: nobody could look. Infra problem.
  **Retryable on resume** (the ledger may re-attempt these pages; REJECTED pages skip).

*Why two:* collapsing them makes "bad table" indistinguishable from "ollama was down" in
every downstream artefact. Both must surface at page status, document status, metadata
and CLI (the no-silent-loss rule: failures surface at every level, not just one).

### Judge output schema (strict JSON)

```
verdict:    PASS | FAIL
confidence: high | low          # judge's own uncertainty; categorical, not a number
findings:   [ {code, where, detail} ]   # empty iff PASS
```

`code` is a **closed enum**, one value per measured failure family:

| code | catches | issue family |
| --- | --- | --- |
| `MISSING_VALUE` | value in crop, absent in output | #144, #331 |
| `FABRICATED_VALUE` | value in output, absent in crop | #270 |
| `WRONG_BINDING` | value present, wrong row/col | #151, GH-273 case |
| `HEADER_MANGLED` | header band wrong or absorbed into data | #146, #215 |
| `STRUCTURE_MERGED` | multiple tables flattened into one | #152 |
| `NOT_A_TABLE` | region is a chart / index / figure | #150, #213, #249 |

`where` = cell/row/col reference in the output's coordinates; `detail` = one evidence
sentence.

*Why this shape:*

- Codes map 1:1 onto the bug taxonomy already used in the issue tracker — findings are
  mechanically countable across a corpus.
- `NOT_A_TABLE` makes the judge double as the table-ness check: the routing family is
  corrected at the gate with no separate classifier.
- `where` + `code` is exactly the payload a future repair prompt needs, so repair can be
  bolted on later without a schema change.
- Strict JSON keeps ¬S1 honest: unparseable output *is* the S1 failure signal.
- Keep required fields minimal — one over-required schema field has previously voided an
  entire agent response.

### Confidence-gated quorum (no magic thresholds)

- **PASS + high confidence** → accept at the current rung.
- **PASS + low confidence** → confirm at the next rung; acceptance needs both.
- **FAIL** → trusted at any rung (tiebreak upward, never silently overridden downward).

*Why:* an escalate-on-fail ladder is only as good as the first rung's passes; a weak
judge's PASS must not gate the strong judge out. Rather than inventing an "easy vs hard
table" heuristic (a magic threshold), the judge self-labels its uncertainty and that
label routes the confirmation.

### Kept alongside: the mechanical binding check

A small mechanical check of value→(row, column) binding stays **in addition to** the
judges. *Why:* measured evidence (GH-273: identical 50-value multisets, only the binding
differed) shows even two frontier judges miss binding-only shifts — when every number is
present, "all values match" reads as PASS. Binding is a bookkeeping problem, not a
perception problem; it gets a bookkeeping check.

## What this subsumes

- Acceptance side of #144, #146, #151, #152, #215, #270, #301, #302, #303, #331.
- Routing family #150, #213, #249 (via `NOT_A_TABLE`).
- The native-lane-unwitnessed hole (native goes through the ladder like everything else).
- Replaces the #326/#330 binder direction: #330's own finding — the binder cannot fully
  check any real table — is the argument for judging the crop instead of binding every
  candidate.

## Non-goals (this design)

- **Repair loop** — deferred; the findings schema is repair-ready by construction.
- **Figures / formulas** — same ladder shape applies (the sketch routes them through the
  same quality-check box) but they are separate tickets with their own finding enums.

## Open questions

- Which ollama cloud model takes the CLI₁ seat (must have real vision support and be
  off-family from qwen; needs a short bake-off).
- Per-page call budget and latency ceiling for the ladder on long documents.
- Exact wording of the judge prompt and how the crop is produced for multi-table pages
  (one call per table region vs one per page).
