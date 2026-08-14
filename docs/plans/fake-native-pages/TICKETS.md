# TICKETS — fake-native pages (baked-in OCR layers accepted as born-digital)

Status keys: `TODO` · `WIP` · `DONE` · `BLOCKED`. `depends-on` gates dispatch.
Same wave ⇒ disjoint files. Each ticket = one implementer agent, then one reviewer pass.

## Context — measured, not suspected

`1994__christiano_eichenbaum_evans__..._NBER.pdf` page 35 is a **scanned page with a bad OCR
text layer baked in**. socr classifies it `is_born_digital=True` and takes the free native path.
Its own text layer reads:

```
'Eftes of Fedal Fixbds'      <- "Effects of Federal Funds"
'Pof icy S1cls on:'          <- "Policy Shocks on:"
'l-2Ojwts'                   <- "1-2 Quarters"
'Sigrificace'                <- "Significance"
'0271'  '0276'  '-0711'      <- "0.271" "0.276" "-0.711"  (decimal points lost)
```

**Every existing quality check passes on this page** (measured):

| check | value | threshold | verdict |
|---|---|---|---|
| non-printable ratio | 0.000 | `MAX_GARBAGE_RATIO` 0.05 | pass |
| `(cid:` artifacts | 0 | — | pass |
| replacement chars | 0 | — | pass |
| word count | 334 | `MIN_WORDS_PER_PAGE` 15 | pass |
| mean word length | 5.22 | — | pass |

The existing detection is **character-class based**; this corruption is **lexical**. `Eftes`,
`Fedal`, `Sigrificace` are all printable, ASCII, and of ordinary length. No character-level test
can see them.

**But one existing signal does catch it, and is not being consulted.** The page's raster coverage
is **0.998** — a full-page scan. `RASTER_DOMINANCE_RATIO` (0.90) already exists and is documented
for exactly this case, but it is only consulted under the *clean-short-text exception*. This page
has 2077 chars, so the check never runs.

**Why this outranks the work it interrupts.** Numbers like `0271` (a lost decimal point) are
*already wrong before socr touches them*. Every downstream table check then argues about the
wreckage:
- TR-3 fires here — correctly — but the mismatch is corrupt source text, not a reconstruction
  defect. An unknown share of its measured firings (68/491, `#205`) may be these pages.
- The GH-151 B1 shape gate counts garbled labels (`Sat. Eucx`) as genuine textual damage.
- A page whose text reads `Eftes of Fedal Fixbds` should never reach the native path at all. It
  should go to OCR, where the model reads the image and gets it right.

This is a **silent content loss** breach at the routing layer: the free path is taken on a page
where it cannot possibly be correct, and nothing says so.

## Stream A — measure before fixing

### TICKET-A1 — quantify the fake-native population · TODO · depends-on: none · wave 1
**Problem:** The fix must be sized against the corpus, and tonight's table-defect numbers may be
contaminated by these pages. Nobody knows how many there are.
**Do:** Read-only measurement over the 40-paper list at `/tmp/b1probe/list.txt` (copy each PDF to
`/tmp` first, record byte size + sha256; substitute the ProtonDrive twin for an evicted iCloud
placeholder; name any file you skip). For every page report: raster coverage ratio, char count,
`is_born_digital`, `has_tables`, `has_unverifiable_table_region`. Then answer:
1. How many pages have raster coverage ≥ `RASTER_DOMINANCE_RATIO` **and** a text layer **and**
   `is_born_digital=True`? That is the fake-native population.
2. Of TR-3's firing pages, what share are in that population? **This number reframes `#205`.**
3. Of the GH-151 B1 shape gate's firing pages, what share are in it?
4. Sample 10 fake-native pages and quote their first 200 native chars, so the class is evidenced
   rather than asserted.
**Files:** `docs/plans/fake-native-pages/logs/` only. No `src/` changes.
**Done when:** a dated log in `logs/` states all four figures with the per-file manifest, and the
sampling rule at the top. State the opened-PDF count so a partial run cannot read as complete.

## Stream B — detect

### TICKET-B1 — consult raster dominance regardless of text length · TODO · depends-on: A1 · wave 2
**Problem:** `RASTER_DOMINANCE_RATIO` already encodes "a full-page raster means this is a scan",
but it is gated behind the clean-short-text exception, so a scan with a *long* baked-in OCR layer
bypasses it entirely.
**Do:** Make full-page raster coverage a first-class signal on `PageAssessment`, consulted for
every page rather than only short-text ones. A page with coverage ≥ `RASTER_DOMINANCE_RATIO` and
an embedded text layer is a scanned page with a baked-in OCR layer: it must **not** be
`is_born_digital`. Follow the GH-147 A2 precedent — set an explicit field at the moment the
condition is detected, never re-derive it downstream.
Do not invent a new threshold: `RASTER_DOMINANCE_RATIO` exists and is documented at
`born_digital.py:~440-462`. If A1's measurement shows 0.90 is wrong, change it in a commit that
says why, with the measurement attached.
**Files:** `src/socr/core/born_digital.py`
**Done when:** `_assess_page` on page 35 (fitz index 34) of
`1994__christiano_eichenbaum_evans__..._NBER.pdf` reports `is_born_digital=False`; a genuine
born-digital page from the same corpus still reports `True`; full suite passes.
**Regression guard, required:** a born-digital **figure page** (large chart, coverage 0.50–0.80)
must stay `is_born_digital=True`. Over-refusing costs an OCR call; under-refusing costs content.
Both directions need a test.

### TICKET-B2 — lexical quality signal · BLOCKED · depends-on: A1 · wave 2 (conditional)
**Problem:** Raster dominance catches scans. It does **not** catch a born-digital PDF whose
embedded text is corrupt for another reason (bad font subsetting, broken ToUnicode map), where
the page has no raster at all.
**Do:** Only dispatch this if A1 shows a material population of lexically-corrupt pages that
B1's raster check does **not** cover. If B1 covers the population, close this as unnecessary.
If dispatched: derive a lexical-plausibility signal from the text itself (e.g. share of tokens
that are not plausible words against a fixed reference vocabulary). **No tuned constant** — the
threshold must be derived from the measured distribution and named.
**Files:** `src/socr/core/born_digital.py`
**Done when:** the decision to build or drop it is recorded in `logs/` with A1's numbers as the
evidence.

## Stream C — consequence

### TICKET-C1 — route fake-native pages to OCR and surface the reason · TODO · depends-on: B1 · wave 3
**Problem:** Detection that changes no routing is the same "defect nothing consumes" mistake
GH-151 B1 was opened for.
**Do:** A page detected by B1 must route to OCR, and must emit an `AuditEvent` naming the reason.
Surface it at page status, document status, metadata **and** CLI — all four, per the house rule.
Under `--native-only` the ladder is off: record and surface, never reroute (this mirrors the
settled GH-147 A2 / GH-151 B1 ruling and must not be relitigated here).
**Files:** `src/socr/core/born_digital.py`, `src/socr/pipeline/orchestrator.py`,
`src/socr/core/state.py`
⚠️ **Contended files.** All three are claimed by open PR #200 (GH-151 B1, held) and by queued
GH-150 C1 / GH-152 A2. The coordinator must grant them explicitly before dispatch, and this
ticket serializes against those.
**Done when:** processing the NBER fixture emits an audit event of the new kind, the page is not
`audit_passed=True`, and it appears in the OCR-routed set; a genuine born-digital page is
unaffected.

### TICKET-C2 — re-measure the table defect rates · TODO · depends-on: C1 · wave 4
**Problem:** Every table-quality number measured before this fix may be contaminated by
fake-native pages. Decisions on `#205` and PR #200 are waiting on figures that may be wrong.
**Do:** Re-run the TR-3 firing measurement and the GH-151 B1 shape-gate measurement over the same
40-paper list, with fake-native pages now excluded from the native path. Report before/after for
both, using the same sampling rule so the comparison is like-for-like.
**Files:** `docs/plans/fake-native-pages/logs/`, and a comment on `#205`.
**Done when:** a dated log gives before/after for both signals, and `#205` carries the corrected
figures. If the corrected TR-3 rate differs materially from 13.8%, say so plainly — that number
is currently load-bearing in the PR #200 decision.
