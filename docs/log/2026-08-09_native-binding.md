# GH-49B — Native label→value binding: design decision

**Ticket:** `GH-49B-DESIGN` (docs/plans/TICKETS.md)
**Issue:** #49 (later comment, from the GH-96 work)
**Written:** 2026-08-10 (design panel ran 2026-08-09/10)
**Status:** settled on approach; two owner decisions remain open

## The question

`native_rows_from_page` (`tables/native_rows.py:125`) reads reliable numeric data out of a
born-digital page's text layer. Today it feeds only grading (`benchmark/table_exactness.py`)
and the escalation trigger predicate (`orchestrator.py:1474-1477`). It never corrects VLM
output.

#49 asks that it *bind* label→value. The three sharp questions were: replace markdown or
only accept/reject? what makes native "trustworthy enough" to override? how do hierarchical
paths, empty parents and multi-table pages become markdown without inventing columns?

## Decision

**Numbers only, never layout — and only against proof.**

The VLM's emitted Markdown remains the sole authored structure. Headers, section rows,
value-less parent rows, text cells and table boundaries stay byte-identical. Native data may
overwrite **numeric cells only**, and only when the binding is provable:

1. one-to-one match between located native regions and Markdown table blocks;
2. row keys are the **normalized full hierarchy path**, not the leaf label;
3. bijective correspondence in document order — zero orphan, duplicate, surplus or missing
   data rows;
4. no value written into a `-1` (ambiguous) lane;
5. a monotone lane→existing-column map covering every native value.

All-or-nothing per table. Any ambiguity fails closed, changes nothing, and flags the page.
The rebound text is submitted as a **zero-cost candidate through the existing
`decide_escalation` gate** rather than written directly, so acceptance keeps one rule.

### Why not the two starting positions

Both panel models opened elsewhere and revised under evidence:

- **"Rebind by label key"** (opening position) is unsafe. `LabeledRow.key` is
  `normalize_label(self.label)` where `label` is `path[-1]` — the **leaf label only**
  (`native_rows.py:112-118`, verified). Two tables on one page each having a `Total` row
  can cross-bind, writing table A's number into table B with full confidence. On a citation
  corpus that is the worst available failure.
- **"Accept/reject only, never rewrite"** (opening position) delivers nothing: that is
  already what `decide_escalation` does. It leaves the canonical GH-96 failure — all 138
  values present, bindings wrong — unfixable at any cost.

### Why native cannot author the table

`native_rows_from_page` is an evidence projection, not a lossless table model. Verified at
`native_rows.py:527`:

```python
if label and values and not _MARKER_RE.match(label):
```

Value-less parent/section rows, marker rows and orphan numeric rows are all dropped.
Headers, text columns, blank-cell positions and units notes never enter the projection.
Serializing Markdown from it would necessarily invent presentation, and sometimes columns —
a direct violation of the no-silent-content-loss rule. This also matches settled prior art:
`docs/plans/table-repair/` and `docs/log/2026-06-17_gh56-table-repair-design.md` already
rejected deterministic full reconstruction on the real CE page ("VLM for structure, geometry
for values").

The full hierarchy is already available as `LabeledRow.path` / `.display` — it is simply not
used as the match key. Switching the key is cheap.

## Scope limit: born-digital only

**This ticket does nothing for scanned documents, and the ticket text must say so.**

`native_rows_from_page` reads the PDF text layer. A scan has none, so it returns zero rows,
`rows_establish_grid` fails, and the caller bails at `orchestrator.py:1477-1479` recording a
`table_not_scorable` audit event. Verified. The behaviour is correct — it fails closed and
surfaces at document level via `tables_trust` — but the coverage is zero.

Scanned tables remain on the weaker canary gate. Measured in
`escalation_decision.py:29-36` over 16 table pages (**percentages are stale post-#123 by the
file's own warning; the ordering is what the decision relies on**):

| rule | accuracy |
|---|---|
| no escalation | 45.0% |
| canary-only (what scans get) | 81.7% |
| exactness gate (born-digital) | 85.0% |

This is the same blind spot GH-127 had: both tickets silently assumed a text layer.

## Cost the panel understated

Region identity is **deliberately** discarded on both sides, not just the native one. From
the `native_rows_from_page` docstring (`native_rows.py:141-145`, verified):

> `_assign_lanes` clusters their value positions into anonymous column lanes **across the
> whole page**, not per region — `BenchmarkScorer._markdown_table_cells` flattens every pipe
> table on the page into one grid with no table boundaries, so a per-region lane space would
> not have anything to compare against on the markdown side.

So requirement (1) above — per-region one-to-one matching — cannot be met by extending the
native projection alone. **The Markdown-side parser must also learn table boundaries.** Both
panel proposals framed this as a native-side change; it is a two-sided one.

Additionally, `locate_tables` is known to over-merge rule bands
(`tables/locate.py:60-81,122-139`, agent-cited). Region identity can itself be ambiguous, and
must fail closed rather than be guessed.

## Acceptance test

Add and pass `~/venvs/socr/bin/pytest tests/test_native_binding.py -q` with fixtures proving:

- a wrong label→value binding is **rejected**, not written;
- two native regions map one-to-one to two Markdown tables and verify independently;
- same-leaf-label rows in different tables on one page do **not** cross-bind;
- ambiguous native parsing fails closed and leaves the incumbent text unchanged;
- headers and empty parent rows are byte-identical to the VLM candidate;
- **a scanned page is a no-op AND is flagged** — this guard exists so a later change cannot
  "fix" the no-op by loosening the gate.

## Open owner decisions

1. **Ship default-on, or flag-gated for one corpus-measurement cycle?** This deterministically
   mutates shipped Markdown on the primary path with no second engine in the loop, on a corpus
   where a wrong number is worse than a missing one. Panel lean: flag-gated for one cycle.
2. **Is the two-sided cost (Markdown parser learns table boundaries) acceptable in this
   ticket**, or does region-aware matching become its own ticket with GH-49B gated behind it?

## Provenance

Two models proposed independently, then exchanged one rebuttal round; both revised. Panel
agents were read-only leaf nodes (no sub-delegation, no inter-agent messaging) after the
2026-08-09 topology fix. Claims marked *verified* were re-checked against the working tree at
`main` @ `afd15f0` by the orchestrator, not accepted on an agent's word — specifically
`native_rows.py:112-118`, `:527`, `:141-145`, `orchestrator.py:1477-1479`, and
`escalation_decision.py:29-36`. Claims about `tables/locate.py` and `table_exactness.py`
matching behaviour are agent-cited and not independently re-verified.
