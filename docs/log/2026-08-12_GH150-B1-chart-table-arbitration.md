# GH-150 TICKET-B1 — chart-vs-table arbitration: design decision

**Ticket:** TICKET-B1 (`docs/plans/gh150-figures-as-tables/TICKETS.md`)
**Issue:** GH-150
**Written:** 2026-08-12
**Status:** implemented, wave-1, independent of A1/A2/B2

## The problem

`_is_chart_asset_page` unconditionally excluded any page where `_page_has_tables`
was true, so a figure whose axis labels parse as table-like rows (Heston p10, 932
vector drawing ops, `has_chart_marks=False` before A1, `has_tables=True`) could
never reach the chart lane — the table signal always won, and the figure shipped
as a pipe grid of axis labels instead of an image reference.

## Decision

> **Superseded during implementation — read this box first.** The decision below
> ("chart wins unconditionally") is the *original* one and is only half of what
> shipped. It holds for **chart-only** pages. For **mixed** chart+table pages the
> shipped behaviour is the opposite: they are held OUT of the page-level chart
> lane and keep their normal route.
>
> Why it changed: the page-level lane appends its PNG ref after the whole of
> `ps.native_text`. That is correct only when the chart IS the page. On a mixed
> page the chart sits between other regions, so appending sinks it below every
> table that followed it and breaks source order. Mixed pages instead rely on
> `rowize_from_words_chart_aware`, which emits an inline placeholder at the
> chart's own y-position — position-preserving by construction.
>
> So the arbitration is **region-level, not page-level**: the winner recorded in
> the `chart_table_arbitration` audit event is `chart_region`, not
> `chart_asset_page`. `_is_chart_asset_page` answers *eligibility*; the caller
> decides the route.

**Chart wins unconditionally when both signals fire.** No new threshold, no
tie-break, no `native_grid`-based gate. Rationale (encoded verbatim in the code
comment and the `AuditEvent.data.rationale` field): an image reference loses
nothing recoverable; a pipe grid of axis labels loses everything.

The surviving core of this rationale is that the chart must never be destroyed by
the table signal. What changed is *how* it is preserved — inline at its own
position, rather than by handing the whole page to the chart lane.

### Predicate split

`_is_trusted_native_without_ocr` (native-bypass gate, two callers: the non-agentic
branch at `orchestrator.py:765` and the agentic-loop alias
`_is_agentic_trusted_native`) keeps its exact prior semantics, including the table
exclusion — its second caller's behavior must not drift.

A new `_is_native_eligible_without_ocr` holds every gate EXCEPT the table
exclusion. `_is_trusted_native_without_ocr` is now `_is_native_eligible_without_ocr(...)`
followed by the `--native-only` short-circuit and `not self._page_has_tables(...)` —
observably byte-identical to before (the `native_only` early return still precedes
the table check, exactly as the pre-split predicate did).
`_is_chart_asset_page` calls the new eligibility predicate instead, so a
page with both a chart signal and a table signal can still be *detected* — and then
arbitrated by the caller, rather than unconditionally losing to the table signal.

### Precompute + prune, not an inline check

`_phase_agentic` precomputes the eligible set by scanning ALL of `state.pages`
(not just `ocr_pages` — a chart-only page with no table signal is already
trusted-native and never enters `ocr_pages`, so scanning only that list would
silently drop such pages back to plain native prose). Eligible pages split into
`chart_only_pages` and `chart_mixed_pages`; **only the chart-only set becomes
`chart_winner_pages`**.

The winner set is pruned from BOTH `ocr_pages` and `native_fallback_pages` before
the empty-ladder early return. **That prune is currently a no-op, deliberately
kept as a self-enforcing invariant** — say so plainly rather than implying it is
load-bearing:

1. winners are chart-only, hence have no table signal, hence
   `_is_trusted_native_without_ocr` returns True for them — so they never entered
   `ocr_pages` (built from `not _is_agentic_trusted_native`) and never entered
   `native_fallback_pages` (a subset of it). The prune removes nothing today. It
   is retained so that widening `chart_winner_pages` to include table-bearing
   pages stays safe: those pages *would* be in both lists, and the empty-ladder
   return (CI's exact no-provider state) stamps every `native_fallback_pages`
   entry `WARNING`/`MODEL_UNAVAILABLE` before the loop runs.

   Note what is deliberately *not* covered: **mixed** pages are not winners, so
   they stay in both lists and ARE stamped unavailable on a no-provider run —
   correctly, since they genuinely need the ladder for their table.

2. pruning the lists would not stop routing in any case — the loop's `else`
   branch calls `route_page` unconditionally for non-native, non-chart pages. The
   loop gate (`if page_num in chart_winner_pages:`, replacing the old
   `is_native and self._is_chart_asset_page(...)` check) is the load-bearing
   half.

### Two `bo.engine != "chart_asset"` guards

Two sites could overwrite the chart lane's `bo.text` with a re-scored table grid:

- the in-loop table re-read gate;
- the GH-96 escalation call site.

Both are guarded at the call site only (`bo.engine != "chart_asset"`); neither
`_reread_page_tables` nor `_escalate_table_page` was touched.

**Only the escalation guard is currently live.** The re-read gate already
short-circuits on `self._page_has_tables(page_num, ps)`, which is False for every
chart winner (winners are chart-only by construction), so its `chart_asset`
clause is unreachable today. It is kept for symmetry with the live guard and to
stay correct under the same widening described above. The escalation guard has no
such preceding table check, so it is the one that actually fires.

Chart winners fall into the `elif bo.text and self._page_has_tables(...)` scoring
branch (`_surface_table_scoring`) instead — intentional, chart pages surface as
not-scorable rather than silently skipped. Chart winners also newly reach
`_guard_agentic_page_table_repetition` (repetition-line audit only, no routing
effect).

### Audit event

`chart_table_arbitration` is appended immediately after the existing
`chart_asset_page` event, only when `self._page_has_tables(page_num, ps)` is
true for the chart-winning page (single-signal chart pages emit no arbitration
event). `page_num` is mandatory so the event joins the page record. `native_grid`
is computed in a narrow fail-open `try/except` (`fitz` + `native_rows_from_page`
+ `rows_establish_grid`) purely as audit data — it is never used to gate or
tie-break the routing decision, and a raising fitz open yields `native_grid=None`
rather than aborting the page.

## Open question — NOT resolved in B1

Recorded verbatim for the owner and the B2 implementer:

> a real grid that loses to chart ships as flattened text + PNG; possible
> both-representations path

## Must-nots observed

- Did not touch `src/socr/figures/extractor.py` (A1's file this wave).
- Did not touch `tests/test_chart_detection_gh150.py` (A2/B2 own it).
- Did not modify `_phase_assemble`, `_rewrite_all_fragments`, or any
  fragment-writing path; no golden was re-blessed.
- Did not add any routing tie-break or numeric threshold; `native_grid` stays
  audit-only.
- Did not change the existing `chart_asset_page` event, the
  `chart_asset_render_failed` fail-closed path, or the `_surface_table_scoring`
  `elif` at `:2580`.
- Did not emit `chart_table_arbitration` for single-signal pages.
- Did not change `_is_trusted_native_without_ocr`'s observable semantics (pinned
  by a new unit test: still False for a `has_tables=True` native page).
- Did not modify `_reread_page_tables` or `_escalate_table_page` internals — both
  guards are at the call sites only.
- Did not modify `test_table_page_routes_to_ocr_ladder` — it passes byte-for-byte
  unchanged (the dense-table fixture has no chart marks, so it never enters
  `chart_winner_pages`).

## Verification

`~/venvs/socr/bin/pytest tests/test_chart_lane.py -q` — 24 passed, including the
inverted both-signals unit test, the `_is_trusted_native_without_ocr` pin, the
single-signal no-arbitration assertion, and the new hermetic end-to-end test
(`route_page` never called; final `bo.text` keeps native prose + PNG ref, no
`|---|`; exactly one `chart_table_arbitration` + one `chart_asset_page` event;
`audit_log.json` round-trip). Full suite and CI status recorded in the PR.

---

## Revision (2026-08-12, same day) — "chart wins the whole page" was too coarse

The decision recorded above says a page carrying both signals routes to the
page-level chart-asset lane, because "an image reference loses nothing
recoverable; a pipe grid of axis labels loses everything". The premise about
*value* is right. The premise about *scope* was wrong, and the reading-order
parity guard caught it:

```
ParityError: READING-ORDER: 'chart' (pos 966) must appear before
             'historical_table' (pos 530) in the extracted markdown
```

### Mechanism

The page-level lane builds its output as

```python
chart_body = native_prose.rstrip() + "\n\n" + chart_png_ref
```

— the PNG reference is **appended after the entire page text**. That is correct
only when the chart *is* the page. On a mixed page the chart sits between other
regions, so appending sinks it below every table that followed it in the source.
The CE-like fixture has four regions (table → chart → table → prose) and the
chart landed last.

Routing mixed pages into that lane also pruned them from `ocr_pages`, so they
never reached the ladder and never reached the `not decision.accepted` path
where `_render_chart_region_pngs` runs.

### What changed

`chart_winner_pages` is now split into `chart_only_pages` (which take the
page-level lane, unchanged) and `chart_mixed_pages` (which keep their normal
route). Mixed pages rely on `rowize_from_words_chart_aware`, which already emits
an inline placeholder at the chart's own y-position and sorts regions by `y0` —
position-preserving by construction, and built for exactly this page shape under
TR-2.

The arbitration audit event survives with `winner="chart_region"`, emitted where
the decision is actually made rather than inside a lane mixed pages no longer
enter. The in-lane arbitration block (and its `native_grid` probe) became
unreachable and was removed.

### Residual gap — NOT fixed here

Region placeholders are resolved on the `not decision.accepted` path. If the
judge **accepts** a ladder output for a mixed page, that output governs and the
native placeholder never reaches the assembled markdown — the chart could still
be lost on a mixed page whose VLM extraction passes. That is a real hole, it
predates B1, and it needs its own ticket rather than being smuggled in here.
