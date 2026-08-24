# R7 — design panel on the page-disposition value object

Date: 2026-08-24
Ticket: `docs/plans/orchestrator-decomposition/TICKETS.md` → R7
Panel: GPT (gpt-sol), Fable, Kimi — parallel, read-only. Composer dropped (returned nothing twice).
Adjudicator: Opus.

---

## 0. Result

**The premise in `TICKETS.md` was wrong.** It said R7 "makes exclusivity a property of one
function", implying the twelve buckets partition the pages. They do not, and all three
models found the same counter-evidence independently.

The corrected shape is **two axes**:

- **Ship disposition** — exactly one per page. What text the page actually shipped.
- **Alerts / overlays** — zero or more per page. Orthogonal facts about a page that ships.

`value_drift`, `fabricated_ref` and `text_grid_rejected` are the second kind: they are
derived from events or flags, they do not select a winner, and they co-occur with a page
that ships perfectly well. Forcing them into a single enum would be a behaviour change.

---

## 1. The load-bearing claim, verified mechanically

The panel's central claim is that the ship buckets are exclusive. Two models asserting the
same thing is **not** evidence in this repo — there is a recorded case of two vendors
agreeing on a page and both missing the same defect. So it was checked directly.

`socr/core/manifest.py::_select_page_output` (lines 763–1208) is a straight-line cascade:

- **15 `return` statements**
- **0 loops**

Exactly one return executes per page, by construction. Exclusivity on the ship axis is
therefore structural, not conventional — it is a property of the code's shape, and no
comment maintains it.

*(Checked with an AST walk over the function, not by reading the comments.)*

## 2. The documented exception, and why it is not a counterexample

`d3_floor_pages` is a **deliberate strict subset** of `failed_pages`
(`orchestrator.py:6004-6006`). A D3-floor page emits **two** audit events (`page_failed`
and `table_region_unverifiable`), **two** CLI lines, and feeds `LOST_CONTENT_NOTE` through
`failed_pages`.

This does not break the cascade's exclusivity, because `failed_pages` is not a cascade
branch — it is re-derived orchestrator-side from the *shipped text* carrying a failure
marker (`orchestrator.py:5999`). Different axis, same page.

**The trap this creates:** an implementer who models `failed_pages` as "disposition ==
FAILED" makes a document whose only defect is a D3 floor page flip to SUCCESS — while
shipping a failure marker, and losing the event, the CLI line and the lost-content note.
That is precisely the bug class R7 exists to kill, reintroduced by the fix. Kimi named this
as the single biggest risk and it is the correct answer.

---

## 3. The open fork — for the owner

How does the new classifier learn which of the 15 endings a page took?

**Option A — re-derive.** The classifier evaluates its own predicates. One file, smaller
diff, no change to `manifest.py`.

**Option B — tag the cascade.** Refactor `_select_page_output` internally to
`_select_page_output_tagged(...) -> tuple[PageOutput, WinnerKind]`, one tag per existing
return; the public function drops the tag, so its output is byte-identical. The classifier
reads the tag.

**Recommendation: B.** Three arguments, in increasing order of weight:

1. A is a second copy of a decision. This repo has already paid for that once:
   `_reaches_structure_class_branch` (`manifest.py:627`) exists *only* because a mirror
   predicate drifted, and its fix was "walk the same branches in the same order".
2. The cascade being loop-free and single-return (§1) makes tagging mechanical — each
   return gains a constant. There is very little to get wrong.
3. **Kimi's caveat is decisive.** `corrupt_math_hybrid`'s disjointness from the model-kept
   buckets is only *implicit*, via native-engine prefix checks (`manifest.py:411`, `:608`).
   Nothing today has to rank them. A re-deriving classifier would force someone to **invent**
   that priority order — a new, unrecorded decision, made blind, inside a change whose whole
   contract is that it changes nothing. Option B inherits the cascade's own order, which is
   correct by definition. There is nothing to invent.

---

## 4. Convergent findings worth keeping regardless of the fork

- **Inputs.** All three independently proposed the same signature:
  `(state, page_texts, native_only)`. `page_texts` must be the **pre-strip** list computed
  once (`orchestrator.py:5954`) — recomputing risks deriving from post-strip text and
  breaking the byte-identity stitch check. `native_only` is the only config bit any bucket
  reads (`:6046`); pass the bool, not `self.config`.
- **Call-site ordering.** The classifier must run **after**
  `_guard_fabricated_image_refs_document` (`:6230`), which creates the page-0 evidence
  `doc_fabrication` reads (`:6238-6242`). Moving it earlier silently drops phase-major
  fabrication from `pages_ok`.
- **A fourth consumer the ticket omitted.** `final_result.error` reads `failed_pages`
  (`:6590`) and `corrupt_math_hybrid_pages` (`:6619`). The value object must serve it too.
- **An asymmetry to reproduce, not fix.** The event/CLI gate at `:6291-6302` omits
  `fabricated_ref_pages` and `text_grid_rejected_pages`, even though both ARE `pages_ok`
  terms (`:6271`, `:6280`). A document whose only defect is a fabricated ref prints nothing
  and appends nothing here, yet lands AUDIT_FAILED. That is current behaviour. R7 is a
  structure move; it must reproduce the asymmetry, not correct it.
- **Name collision.** `native_fallback_pages` is also an unrelated routing-time local in
  `_phase_agentic` (`:2737`, `:2768`, `:2854`). The new type must not reuse the bare name.

## 5. A test gap this refactor must close

`_flush_page_sidecar` filters `state.events` by page number **at write time**, so each
`pages/NNN.json`'s `audit_events` array captures the global append order for that page.
Audit-append order is therefore **observable output**.

The existing golden/byte-identity tests cover the assembled `.md`. They do **not** cover
sidecar event order. A page-major rewrite of the event loops would reorder every
multi-bucket page's sidecar and every current test would still pass.

So R7 needs a new hermetic characterization test — synthetic `PageState` flags, no
provider, no pipeline run — asserting at minimum:

- `set(d3_floor_pages) <= set(failed_pages)`
- the exact `(page_num, kind)` event sequence
- the sidecar `audit_events` order for a D3 page (`page_failed` before
  `table_region_unverifiable`)

Pinning equality between the old derivation and the new classifier on identical synthetic
states is a **difference pin**, not a locally-measured absolute pin, so it is safe under the
no-provider CI rule.

---

## 6. Panel hygiene

Each model had to quote the file's line count and the verbatim text of line 6119 before
answering, because a headless agent denied file access will otherwise produce a confident,
detailed, entirely invented design and exit successfully. All three passed (7,593 lines;
`        native_fallback_pages = [`). Composer returned nothing twice and was dropped rather
than waited on.
