# Ladder corpus run — shipped rates with all three rungs live

2026-09-04, 12:17–14:25 local, socr `main` @ `e830d9b` (after #580). The follow-up every
GH-353/P1 note deferred: one run over real table pages with the table-judge ladder ON
(now the default), rung 1 `glm-5.3-flash:cloud` through the local ollama daemon, rung 2
`agy` 1.1.26, blind-cell adjudicator `kimi-k2.6:cloud`, OCR `qwen3-vl:30b-a3b-instruct`
on the Mac. Content-free: identifiers, counts and dispositions only.

Corpus: the 2026-08-20 lane-comparison manifest, the pages it tagged `table`, extracted
into one small PDF per paper. 8 of its 9 papers were in the library (Kekre–Lenel absent);
20 pages. One `socr process <pdf> --write-manifest --verbose` per paper, sequential.

## Headline

| | count |
|---|---|
| pages | 20 |
| tables that reached the ladder | 18 |
| ACCEPTED | 5 |
| WITHHELD (readers rejected, blind cell disagreed) | 1 |
| UNVERIFIED | 12 |
| wall clock | 2 h 07 min (6.4 min/page; first paper 12 min/page) |
| cloud cost | $0.0010 |

UNVERIFIED by cause (from the audit events):

| cause | tables | note |
|---|---|---|
| mechanical binding contradiction | 8 | 7 are LaTeX presentation, not content — see #582 |
| no table witness (no rung ran) | 2 | the #560 terminal, correctly labelled |
| a rung answered but was not accepted (`ok: false`) | 1 | reason text dropped — #581 |
| both rungs ok, default label "infra problem" | 1 | cause never recorded — #581 |

Page endings: 14 `model_output`, 6 `fail_closed_marker` (4 structure-class floor, 1
table withheld, 1 native table unverifiable). Status: 5 SUCCESS, 9 WARNING, 6 ERROR.
Exit code 1 on 7 of 8 papers — every one surfaced at document level, as designed.

## What the run settles

1. **The three rungs are reachable at corpus scale.** 17 of 18 witnessed tables got a
   rung-1 verdict; rung 2 ran on 5; the adjudicator ran on 7 clamped tables (1 lift,
   6 held) and once as the WITHHELD guard. No rung outage, so the P1 latch keys stayed
   absent from every sidecar — the sparse-key contract holds over 20 pages.
2. **The binder, not the readers, is the demotion engine.** 8 of 12 UNVERIFIED are the
   mechanical binding check clamping a rung-1 ACCEPTED. Of the recorded contradiction
   items, every one on doc02 (5/5), doc03 (3) and the header items on doc05/doc07 is a
   VLM cell or label wrapped in inline math (`$-0.06$` vs `−0.06`,
   `Adjusted $\text{R}^2$` vs `Adjusted R2`, `$\Delta \log$ …` vs `∆log …`). The
   binder tests `is_numeric_token` before comparing and convicts the wrapped token as
   "not a number"; `tokens_agree` has the same predicate, so the adjudicator is
   structurally unable to disprove it (`held` is guaranteed). Filed as **#582**; fix in
   flight (balanced-wrapper normaliser shared by binder and adjudicator). The sibling
   classes (`\Delta`→`∆`, `\log`, `\&`) are recorded on the issue as follow-up scope.
   The remaining contradictions are native-side: a shredded subscript header (`1t 1t`,
   doc04) and a run-on row label (`Sample 1988:1–…`, doc05/doc07).
3. **The UNVERIFIED label lies whenever no rung was unavailable.** "retryable on
   resume" is printed for the binding-contradiction and default branches, but the latch
   never fires for them, so resume skips the page. The guard chain's real reason
   (two-low-pass, geometry inconclusive, adjudicator abstained/suppressed) is kept in a
   local dict that only the ACCEPTED and WITHHELD messages read; `RungResult.error` is
   not written to the trail. Filed as **#581**.
4. **The structure-class floor fires on 4 of 20 pages** (P2, fail-closed: marker plus
   page image, native grid withheld). Correct by ruling, and the single largest source
   of pages shipping no text in this run. Not a ladder effect; recorded so the next
   P2 measurement has a baseline.
5. **The manifest's `table` tag is not reliable.** doc03 p2 is an appendix equation page
   (it went through the corrupt-math lane, on by default since #580, and shipped WARNING
   with the crop retained) and doc08 p1 is a figure page (routed as a chart asset,
   SUCCESS). Both routings are right; the August selector's tag was wrong.

## What it does not settle

- The ¬S1 rate: one `ok: false` rung answer in 23 rung calls, reason unknown (#581).
- Whether the 8 binding demotions would have been ACCEPTED after #582: replaying the
  recorded pairs through the fix branch's `tokens_agree` turns the wrapping class to
  agree and leaves the sibling and native classes as contradictions; the corpus needs a
  re-run after the fix lands to get a post-fix rate.
- `agy` quota behaviour: rung 2 ran only 5 times.

## Reproduce

Manifest, per-paper PDFs, logs, sidecars and `tabulate.py` are in the session scratchpad
(`ladder-run/`); the extraction is `pymupdf.insert_pdf` of the manifest's pages. The
tabulator reads `pages/NNN.json` (`status`, `disposition`, `table_ladder_disposition`,
`page_cost_usd`, `binding_adjudication`) and `audit_log.json` (`table_ladder_*` events
with `rung_trail`).
