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

**Why this was thought to outrank the work it interrupts — and what A1 found.** Numbers like `0271`
(a lost decimal point) are *already wrong before socr touches them*, so every downstream table check
then argues about the wreckage. The worry was that this contaminated the table-defect measurements.
**A1 measured it and the worry was overstated:**
- TR-3 fires here — correctly — but the mismatch is corrupt source text, not a reconstruction
  defect. ~~An unknown share of its measured firings (68/491, `#205`) may be these pages.~~
  **Measured: 6 of 68 (8.8%).**
- The GH-151 B1 shape gate counts garbled labels (`Sat. Eucx`) as genuine textual damage.
  **Measured: 6 of 71 (8.5%).**
- A page whose text reads `Eftes of Fedal Fixbds` should never reach the native path at all. It
  should go to OCR, where the model reads the image and gets it right. **This part stands
  unchanged** — it is a real routing defect and the reason B1 is still worth building.

This is a **silent content loss** breach at the routing layer: the free path is taken on a page
where it cannot possibly be correct, and nothing says so. That is true of 2.4% of pages, not of the
corpus at large.

## Stream A — measure before fixing

### TICKET-A1 — quantify the fake-native population · **DONE** (`817e593`) · wave 1
**Result:** 72/2972 pages (2.4%), 71 from two scanned documents, 0 from 37 of the 40 papers.
TR-3 overlap **6/68 (8.8%)**; shape-gate overlap **6/71 (8.5%)**. Full log:
`logs/2026-08-14_A1-fake-native-population.md`. The measurement **disproved this plan's premise**
that the table-defect numbers were materially contaminated — see STATUS.md.
**Original brief, kept for provenance:**
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

### TICKET-B1 — consult raster dominance regardless of text length · READY · depends-on: A1 ✓ · wave 2
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

**A1's findings — read these before implementing:**
- **Do not touch the threshold.** Coverage runs 0.516–0.789, then a **hard gap containing zero
  pages**, then 0.940–1.000. `RASTER_DOMINANCE_RATIO = 0.90` sits mid-gap; anything from ~0.80 to
  ~0.93 behaves identically on this corpus. The constant is confirmed by measurement.
- **A known false positive must be a fixture.** `2008__blinder_ehrmann_..._ECB.pdf` page 1 is a
  full-bleed title page: coverage **1.000**, a clean **34-word** text layer, genuinely born-digital.
  The naive rule refuses it. Decide deliberately how to handle it — a word-count floor is the
  obvious lever, but **derive it, do not tune it**, and say so in the commit. Build a look-alike
  fixture; **never commit the PDF** (public repo, copyrighted).
- **The regression guard has real specimens.** The 16 pages at coverage 0.52–0.79 in
  `2021__nagel__ml_ap.pdf` are genuine born-digital figure pages carrying 400+ words of native
  prose and `has_tables=False`. Model the guard fixture on that shape, not on an invented page.

### TICKET-B2 — lexical quality signal · **CLOSED, NOT BUILT** · decided by A1
**Decision:** dropped. A1 was the conditional gate on this ticket and it came back negative — no
material population of lexically-corrupt pages that B1's raster check misses. All 72 fake-native
pages are inside the raster band by construction; a sweep of the other 2771 born-digital pages
found the existing `has_encoding_hygiene_suspect` firing on 738 (26.6%), but spot-checking
`2003__woodford.pdf` p5 showed the trigger tokens were `Cataloging-in-Publication` and a URL —
**false positives of the ratio, not corruption.** Building a second lexical signal on that basis
would add noise, not coverage.

**Caveat, and the condition to reopen:** this is corpus-bound. 40 economics papers do not contain a
PDF with a broken ToUnicode map and no raster — the exact class B2 was written for. That page type
exists in the wild. **Reopen this ticket only with a measured specimen**, not on suspicion; the
whole point of A1 was that suspicion overstates populations.
**Evidence:** `logs/2026-08-14_A1-fake-native-population.md`, §"Are there fake-native pages raster
coverage does NOT catch?"

<details><summary>Original brief (kept for provenance)</summary>

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
</details>


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

### TICKET-C2 — re-measure the table defect rates · **DOWNGRADED to tidy-up** · depends-on: C1 · wave 4
**A1 removed this ticket's urgency.** It was written on "every table-quality number may be
contaminated". Measured overlap is 8.8% (TR-3) and 8.5% (shape gate), so a re-measure moves TR-3
from ~14.8% to roughly ~13.5% — a real correction, not a decision-changing one. **Do not hold
`#205` or PR #200 for it.**
**Problem (as amended):** the published figures include a small, now-quantified contamination.
Once C1 excludes fake-native pages from the native path, the numbers should be restated once so
the record is clean.
**Do:** Re-run the TR-3 firing measurement and the GH-151 B1 shape-gate measurement over the same
40-paper list, with fake-native pages now excluded from the native path. Report before/after for
both, using the same sampling rule so the comparison is like-for-like.
**Files:** `docs/plans/fake-native-pages/logs/`, and a comment on `#205`.
**Done when:** a dated log gives before/after for both signals, and `#205` carries the corrected
figures. State the corrected TR-3 rate plainly whether or not it differs materially — A1's finding
was that it will not, and a re-measure that quietly confirms an expectation is still worth writing
down.
