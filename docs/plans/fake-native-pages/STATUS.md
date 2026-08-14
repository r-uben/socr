# STATUS — fake-native pages

Last updated: 2026-08-14

## Stage

**A1 measured (`817e593`); B1 ready, B2 closes unbuilt.** The defect is real but **small and
concentrated**: 72/2972 pages (2.4%), 71 of them from two documents that are wholesale scans of
old typeset papers. 37 of the 40 papers contribute zero fake-native pages.

**A1 disproved this plan's own framing.** It was scaffolded on the suspicion that fake-native pages
were contaminating the table-defect numbers. They are not: only **6 of TR-3's 68 firings (8.8%)**
and **6 of the shape gate's 71 firings (8.5%)** are fake-native. ~91% of both signals fire on pages
that are genuinely born-digital by every check including raster coverage — those are real
reconstruction defects. `#205` and PR #200 are **not** blocked on this plan and never were; that
claim (struck through below) was an over-generalisation from one vivid page.

This plan is therefore **ordinary priority**, not an interrupt. B1 is still worth building — it is
a cheap, well-evidenced routing fix — but it does not unblock anything else.

Found while hand-judging TR-3 firings for `#205`: the first flagged page turned out not to be a
table-reconstruction defect at all. It is a 1994 NBER **scan** whose baked-in OCR text layer is
corrupt (`Eftes of Fedal Fixbds`, `Sigrificace`, `0271` for `0.271`), which socr classifies as
born-digital and reads natively.

## Base state (clean before tickets)

- Repo `~/repos/tools/socr`, `main` at `ce2d84d`. Full suite 1591 passed / 1 xfailed.
- Evidence for the diagnosis is in this plan's `TICKETS.md` context section — measured, not
  asserted.
- The reproducing page: `1994__christiano_eichenbaum_evans__..._NBER.pdf`, fitz index 34
  (printed page 35). Raster coverage **0.998**, 2077 chars, 0 non-printable, 0 CID artifacts,
  334 words, mean word length 5.22 — every existing quality check passes.

## Ticket board

| Ticket | Stream | Status | depends-on | Wave |
|--------|--------|--------|------------|------|
| A1 | measure | **DONE** (`817e593`) | — | 1 |
| B1 | detect | READY | A1 ✓ | 2 |
| B2 | detect | **CLOSED — not needed** | A1 ✓ | — |
| C1 | consequence | TODO | B1 | 3 |
| C2 | consequence | **DOWNGRADED** (see below) | C1 | 4 |

### A1's findings that change the board

- **B2 closes unbuilt.** A1 looked for the class B2 was written for — a born-digital page with
  corrupt text and *no* raster, which B1's gate would miss — and found no material population.
  The nearest existing signal (`has_encoding_hygiene_suspect`) fires on 26.6% of low-raster
  born-digital pages, but spot-checking showed false triggers on hyphenated words and URLs, not
  corruption. Caveat: this is corpus-bound. A different corpus (bad font subsetting, broken
  ToUnicode) could resurrect B2; reopen it with measurement, not suspicion.
- **`RASTER_DOMINANCE_RATIO = 0.90` is confirmed, not adjusted.** Coverage among born-digital
  pages runs 0.516–0.789 (16 genuine figure pages), then a **hard gap with zero pages**, then
  0.940–1.000 (the 72 fake-native pages). Any threshold from ~0.80 to ~0.93 separates them
  identically. B1 must not invent a new constant.
- **A real false positive exists and needs a fixture.** `2008__blinder_ehrmann_...ECB.pdf` page 1
  is a full-bleed title page: raster coverage 1.000, clean 34-word text layer, genuinely fine.
  B1 would wrongly refuse it. Add it as a regression fixture (look-alike, not the PDF — public repo).
- **B1's regression guard has real specimens.** The 16 pages at coverage 0.52–0.79 in
  `2021__nagel__ml_ap.pdf` are genuine born-digital figure pages with 400+ words of native prose.
  Use those shapes for the "must stay `is_born_digital=True`" test rather than a synthetic page.
- **C2 is downgraded.** It was written to re-measure table-defect rates on the assumption they were
  contaminated. At 8.8% overlap, a re-measure moves TR-3 from ~14.8% to ~13.5% — real but not
  decision-changing. Keep it as a tidy-up after C1, not as a gate on anything.

## Dispatch waves

- **Wave 1:** A1 — **done** (`817e593`).
- **Wave 2:** B1 on `born_digital.py`. **B2 is closed unbuilt** — A1 found no population it covers.
- **Wave 3:** C1 — `born_digital.py`, `orchestrator.py`, `state.py`. **Contended, see below.**
- **Wave 4:** C2 — `logs/` and an issue comment. Downgraded to tidy-up.

## File contention — the reason C1 cannot dispatch on demand

`src/socr/core/born_digital.py`, `orchestrator.py` and `state.py` are claimed by:

- **PR #200** (GH-151 B1) — open and held while its direction is decided.
- **GH-150 C1** (chart placeholder merge) — queued, blocked on #200.
- **GH-152 A2** — may need `born_digital.py` if reading order stays in its `Done when`.

C1 here joins that queue. B1 touches only `born_digital.py` and can go as soon as #200's fate is
settled. The cross-plan coordinator at `docs/plans/extraction-defects/STATUS.md` owns the global
order; this plan does not override it.

## Relationship to the work this interrupted — **resolved, it does not**

This plan was scaffolded on the fear that it would **invalidate the inputs** of GH-151 B1 and
`#205`. A1 settled that. It does not.

- `#205` measured TR-3 firing on 68 native table pages. A1 reproduced that count exactly (68, on
  461 native table pages) and found **6 of the 68 (8.8%)** are fake-native. The headline is
  contaminated by roughly 1 firing in 11 — worth a footnote, not a retraction.
- The GH-151 B1 shape gate fired on 71 pages; **6 (8.5%)** are fake-native, near-identically the
  same 6 pages. **91.5% of its firings are real structural damage in real born-digital tables.**
- ~~Nothing on #200 should be decided on the current figures until A1 lands.~~ **Struck.** This was
  wrong — a generalisation from a single dramatic page to a 2972-page corpus. A1 was still worth
  running: it cost one read-only pass to convert a blocking suspicion into a bounded 2.4% defect.
  But it never should have been described as blocking #200, and it did not block it in practice.

**Standing lesson for this plan and the coordinator:** a vivid single-page failure is a hypothesis
about a population, not a measurement of one. Size it before you let it stop other work.

## Next action

**Dispatch TICKET-B1** whenever `born_digital.py` frees up — it queues behind PR #200, which owns
that file. B1 is now fully specified: the threshold is confirmed, the false positive is named, and
the regression-guard specimens are identified. Nothing else in the repo waits on it.
