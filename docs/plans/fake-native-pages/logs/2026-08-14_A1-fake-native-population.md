# A1 — quantify the fake-native population

**Sampling rule (mandatory, stated first):** all 40 PDFs on `/tmp/b1probe/list.txt` were copied to
`/tmp/fnp_probe/pdfs/` before opening. Byte size and sha256 were recorded for every copy (manifest
below). 36/40 files were read from the iCloud path directly; 4/40 (`2020__tsukioka_yamasaki`,
`2023__swanson__macroeconomic_effects...`, `2024__glasserman_lin...`, `2025__cao_wang_yi...`) were
0-byte iCloud placeholders, substituted with their ProtonDrive twins found by filename match. **No
file was skipped.**

**Opened-PDF count: 40/40.** Pages processed: **2972**. Errors during processing: **0**.

Measured against `main` (`ce2d84d`, `git status --short` empty at measurement time), using the
installed package (`from socr.core.born_digital import BornDigitalDetector`, `_assess_page`) — not
a standalone-module import, per the ticket's instruction and the prior false-blocking-finding
precedent in this repo.

## Per-file manifest

| # | source | file | bytes | sha256 |
|---|---|---|---|---|
| 1 | icloud | 1954__evans__econometrica.pdf | 644932 | `ad76510c16a28413768f8c8070a41b7354b36934c5d7d8487a062280dc7c9b95` |
| 2 | icloud | 1994__christiano_eichenbaum_evans__..._NBER.pdf | 741341 | `9a27b2d85213424610b9ad6a1dfa3633fb8b7665d06944a03731ecea33221828` |
| 3 | icloud | 2000__romer_romer__federal_reserve_information_and_behaviour_of_interest_rates.pdf | 176590 | `f8cc1bf20af46eb3e58d5f2bf0aeabf0d977c6567b01474ac6442e695afdd4f0` |
| 4 | icloud | 2003__woodford.pdf | 4961114 | `f7ab0ad517e4e96f73a9819a2f1e110659aff9497c6290c9336a4adf256aed0a` |
| 5 | icloud | 2006__christiano_motto_rostagno__shocks_structures_policies.pdf | 1693816 | `4f6eaa6b630dc46bbf27ab51c1fe95218a564d3e3c1d0661e4f5c90c6dc0eaa5` |
| 6 | icloud | 2008__blinder_ehrmann_fratscher_dehaan_jansen__..._ECB.pdf | 839334 | `a47367f4016596e90ab694f7da406d1138446b190aeaf76385d159bdef6d0301` |
| 7 | icloud | 2010__Menzly_Ozbas__..._JF.pdf | 210432 | `3c799725139e1a8c62515081820b2342c961da2e335c39c8b510db68771c16ad` |
| 8 | icloud | 2012__bauer_rudebusch_wu.pdf | 292804 | `aa5ea75327303ce6b2952f70d2a1a3598a5e0f8980b543eb20a692d596f8eb43` |
| 9 | icloud | 2013__Snowberg_Wolfers_Zitzewitz__..._HEF.pdf | 928094 | `289f3be8e8b92471f8a974cd6d048312b0044008c359ed00afa75dd4a7a56aea` |
| 10 | icloud | 2015__Hameed_Morck_Shen_Yeung__..._RFS.pdf | 215982 | `f1dac4f4d004d9eac09c513bf8beac15cf3a8bc378f1c2cc5b60bf6e4b497ec5` |
| 11 | icloud | 2016__eisenbach_lucca_townsend__..._NBER.pdf | 525562 | `76e209663f02112f20fdfe7b71e5d5c384a2f57546f111f18680dcfffe04a55b` |
| 12 | icloud | 2016__ramey__shocks.pdf | 553406 | `6732c9c437d9db2c3a32b8336d0997f907de0012f22008cfc5b10a3cd2da25ba` |
| 13 | icloud | 2017__ozdagli.pdf | 782586 | `4ce263a30b530c0119c31659c158e9969d0cdea9df478c9530b704ab080e7b0b` |
| 14 | icloud | 2018__herskovic__JF.pdf | 1027616 | `f4b308f02bed72cbc8ae33bbdbfee3657507eb66f2202fe4c8625a07fd2797dd` |
| 15 | icloud | 2018__unep-fi__..._UNEP-FI.pdf | 268242 | `ca366fbfc592ef6ecd49503360bf137b715fea5fdff8d66444104779a3182f59` |
| 16 | icloud | 2020__bauer_rudebusch__..._AER.pdf | 699792 | `0369910039a808beca06992ee65c4505e94da6d154467f81eaf3c768d0e0fd1d` |
| 17 | icloud | 2020__haddad_sraer__..._JF.pdf | 707955 | `50691fa140e662539e7bee41d2a61f6a737c73b582d0bd4bcb584abdf51a1302` |
| 18 | **protondrive-twin** (iCloud evicted) | 2020__tsukioka_yamasaki__..._WP.pdf | 752360 | `ce8159edba95ca948b9f6a592d664fecd593e4b03a19042c84565ffaed403f14` |
| 19 | icloud | 2021__damato_decclesia_levantesi__..._WP.pdf | 983289 | `977187c346afd463847639e582d5d31ca06c3f5fc79345fe15d9988675e150ed` |
| 20 | icloud | 2021__nagel__ml_ap.pdf | 84930492 | `cbe942a04108a9307ea4ef36ba494d2a9f3a287fc11c0868969935bad8dcf41d` |
| 21 | icloud | 2022__bauer_swanson__reassessment_monetary_policy_surprises__WP.pdf | 733264 | `ff1bffa010ede22007f0144c23f39ae68e5f87eca2553c6c7d9f787934b9c2cb` |
| 22 | icloud | 2022__elliott_golub_leduc__..._AER.pdf | 1217695 | `3220882b59445e1da830c4e7189352c1f21bfbf202af9f5860be05a3d9d610b7` |
| 23 | icloud | 2022a__caballero_simsek__wp.pdf | 1421013 | `d918ad685f13201a0dfaaff4830e97a0e1d61b59ea452fc8021def777ad44f45` |
| 24 | icloud | 2023__bauer_swanson__reassessment_of_monetary_policy_surprises__WP.pdf | 939778 | `ee9ca47165c75bf383d719148f81fce937ed0b43d2845527228316d05ef833e3` |
| 25 | icloud | 2023__costain_nuno_thomas.pdf | 2563942 | `5ae964a533682e4b264373e7ae592e380b65aadf3ffd8bb3e030bf8f16bb9978` |
| 26 | icloud | 2023__jha_qian_weber_yang__..._NBER.pdf | 3242475 | `cd10979e556f202ccedaf5dab2bb229a0f7cccd64b052573735a533678267676` |
| 27 | **protondrive-twin** (iCloud evicted) | 2023__swanson__macroeconomic_effects_conventional_unconventional_monetary_policy.pdf | 356216 | `87267df5336e2a9e6de264ee1630c13786d72fe4c88071176f4689995a4c95e2` |
| 28 | icloud | 2024__bauer_pflueger_sunderam__..._QJE.pdf | 1681467 | `751a47c976cbf1d3d21c0c860ccb503a1d43fc16facd7bb7275f150989420cc3` |
| 29 | icloud | 2024__cieslak_mcmahon_pang__fed_post2020__WP.pdf | 1350830 | `aade70a176190e0358fa6d3cf4b10908244d753972cfb1a30bc5652918ff7772` |
| 30 | **protondrive-twin** (iCloud evicted) | 2024__glasserman_lin__..._ARXIV.pdf | 714572 | `dfc8bb59818864672db9368a5cc964e1ce9ceeb1ea92e235b4e72fc44700fa99` |
| 31 | icloud | 2024__kalemli-ozcan_yildirim__..._NBER.pdf | 7795582 | `2fccf2542144cd0ebce9e749b046b948d5cad806b64283a6bab91cb282a32006` |
| 32 | icloud | 2024__long_huang__central_bank_communication_risk__WP.pdf | 2276897 | `3bb95d0b26373606523bab2beac2f946cfcfa61b040886e4797ae80543135546` |
| 33 | icloud | 2024__tian__calibrating-verbalized-probabilities__ARXIV.pdf | 1533844 | `c05e15ff25dcac2089356d7574f31b33923b7eb18b1e00a72d8245900cd34ee7` |
| 34 | **protondrive-twin** (iCloud evicted) | 2025__cao_wang_yi__..._WP.pdf | 575208 | `02494bea2b484cd4e0ebbac14fb7bd6b9786d7e8586a6818e3f67b5c9983da0e` |
| 35 | icloud | 2025__fan_liu__..._SSRN.pdf | 833060 | `3ceb55d848f9122a54a26488eb821738180e34381b8d232467f3ec86c2ce67a5` |
| 36 | icloud | 2025__haim__how_binding_is_administrative_guidance__JELS.pdf | 1851553 | `8092c79cbc5ed550bbd50b745e79ebd6038cf336a7f86152bbcb6968c9affccd` |
| 37 | icloud | 2025__matera__corporate_earnings_calls_and_analyst_beliefs__ARXIV.pdf | 1178337 | `4eac5dbf53c8d5bb474d819e6bb22216204720cd7a1a7bd1b7e25f15b106005c` |
| 38 | icloud | 2025__sarkar__..._software_development.pdf | 3459896 | `e40ec946d66f4d741923f6194c00ffe03bbbf79e8bdb42d713bcdc4866e89760` |
| 39 | icloud | 2025__white__new_keynesian_puzzle_reinterpreting_inflation_dynamics__WP.pdf | 14059948 | `219ba59ebc653adb28aaf6c4aec65618073ba5807e882363517bbcf5868ded08` |
| 40 | icloud | 2026__du_haberkorn_..._Fed.pdf | 2860512 | `69cb057252f531dc4ba1f3202f81eae53068771dd8a6403212f3ec6384807015` |

## Method

For every page in every PDF: ran `BornDigitalDetector()._assess_page(page, page_num)` (main's
production code path, unmodified) and recorded `raster_coverage` (`_raster_coverage`, same method),
`char_count`, `is_born_digital`, `has_tables`, `has_unverifiable_table_region` (TR-3).

For the B1 shape-gate question, `structure_check.py` and its `structural_gate_fires` do not exist
on `main` — they live on unmerged `feat/151-b1-structural-gate`. Diffed that branch's
`born_digital.py` against `main`'s: the only difference is the additive `native_table_structure_defective`
field and its computation, gated on `has_tables and not text_direction_is_rotated(direction)`, using
`extract_structured` — a function **unchanged** between the two branches. `reconcile.py` (the module
`structure_check.py` imports) is also byte-identical between `main` and the branch. So the branch's
gate was reproduced exactly by pulling `structure_check.py`'s source via `git show
feat/151-b1-structural-gate:src/socr/tables/structure_check.py` into a throwaway `/tmp` module and
calling it against `main`'s own `_assess_page`/`extract_structured` output, under the same has_tables
/ not-rotated gate the branch code uses. This is the "reproducing it read-only" case the ticket
anticipated — not a re-implementation, a literal copy of the pure, dependency-identical module.

Raw per-page CSV: `/tmp/fnp_probe/pages.csv` (2972 rows, not committed — throwaway per the rules).

## The four figures

**1. Fake-native population** — pages with raster coverage >= `RASTER_DOMINANCE_RATIO` (0.90) AND
a non-empty text layer (`char_count > 0`) AND `is_born_digital=True`:

- **72 pages** out of 2843 born-digital pages (**2.5%** of born-digital), 2972 pages total (**2.4%**
  of all pages).
- This population is **not spread across the corpus**. 71/72 pages come from exactly two documents
  that are wholesale scans of old typeset papers: `1994__christiano_eichenbaum_evans__..._NBER.pdf`
  (51 pages) and `1954__evans__econometrica.pdf` (20 pages). The remaining 1 page is
  `2008__blinder_ehrmann_fratscher_dehaan_jansen__..._ECB.pdf` page 1 — see the false-positive note
  below. **0 of the other 37 papers contribute a single fake-native page.**

**2. Share of TR-3 firings that are fake-native** — reframes `#205`:

- TR-3 (`has_unverifiable_table_region=True`) fired on **68 pages** across the corpus (this matches
  `#205`'s reported count exactly, confirming the same measurement basis). Of those 68, **6 (8.8%)**
  are in the fake-native population.
- **This does not materially reframe `#205`.** 91.2% of TR-3's 68 firings are on pages that are
  genuinely born-digital by every signal including raster coverage — the mismatch there is a real
  reconstruction defect, not corrupted source text. `#205`'s headline number should be read as
  contaminated by roughly 1 in 11 firings, not by a large or dominant share.
- For context: native table pages (`is_born_digital=True and has_tables=True`) numbered 461 in this
  run (TR-3 rate 14.8%), close to but not identical to the previously reported 491/13.8% — likely a
  measurement-run difference (corpus snapshot / prior run parameters), not a contradiction.

**3. Share of GH-151 B1 shape-gate firings that are fake-native:**

- The gate (`ragged` or `detached_label_rows`, mirrored from `feat/151-b1-structural-gate` as
  described above) fired on **71 pages**. Of those, **6 (8.5%)** are in the fake-native population —
  the same order of magnitude as TR-3's overlap, and in fact almost the same 6 pages (both signals
  fire on the corrupted table region of the 1994 NBER document). **91.5% of B1's firings are on
  pages the raster/text-layer signal does not flag** — i.e., B1's gate is overwhelmingly catching
  real structural damage in real born-digital tables, not fake-native contamination.

**4. Ten fake-native pages, first ~200 native chars each** (spread across the two contributing
documents plus the one outlier, to show the range within the class — not cherry-picked for maximum
corruption):

```
1954__evans__econometrica.pdf p2 (raster 0.940):
"THE EFFECT OF STRUCTURAL MATRIX ERRORS ON INTERINDUSTRY RELATIONS ESTIMATES BY W. DUANE EVANS
The inevitable errors made in empirically quantifying a function system used to represent an
economy..."

1954__evans__econometrica.pdf p12 (raster 0.940):
"STRUCTURAL MATRIX ERRORS 471 of activity be available for each sector separat[e units may]
be used for different sectors; that is, for one se[ctor in]tons, for another in dollars, and
so on. In actuality, input-outpu[t]..."

1954__evans__econometrica.pdf p21 (raster 0.940):
"480 W. DUANE EVANS REFERENCES DWYER, P. S., AND F. V. WAUGH, "On Errors in Matrix Inversion,"
Jour[nal of the American] Statistical Association, June, 1953, pp. 289-319. HOLLEY, JULIAN L.,
"Note on the Inversion of t[he]..."

1994__christiano_eichenbaum_evans__..._NBER.pdf p1 (raster 0.998):
"NBER WORKING PAPER SERIES THE EFFECTS OF MONETARY POLICY SHOCKS: SOME EVIDENCE FROM THE FLOW
OF FUNDS Lawrence J. Christiano Martin Eichenbaum Charles Evans Working Paper No. 4699 NATIONAL
BUREAU OF E[CONOMIC RESEARCH]..."

1994__christiano_eichenbaum_evans__..._NBER.pdf p27 (raster 0.998):
"20. Gertler, Mark and Simon Gilchrist (1993), 'The Role of Credit Market Imperfections in the
Monetary Transmission Mechanism: Arguments and Evidence', Federal Reserve Board of Governors,
Finance and..."

1994__christiano_eichenbaum_evans__..._NBER.pdf p35 (raster 0.998) — the reproducing page:
"Table 2.1 Properties of Impulse Response Functions: Monetary Variables Eftes of Fedal Fixbds
Pof icy S1cls on: if BQGOVSC 18 Ii l-2Ojwts 0.821 0.751 -0.779 0.014 -0.166 Sat. Eucx 0.068
0.162 0.194 0.126 0.068 Sigrificace 0.00) 0.00) 0.000 0.911 0.015..."

1994__christiano_eichenbaum_evans__..._NBER.pdf p53 (raster 0.998):
"Figure 5.5 Effects of Policy Shocks on Government Tax Receipts Effect of FF on PER TAX Effect
of NBRD on PERTAX I 4 7 10 Effect of FF on CORPTAX 100 75 5.0 25 -50 -75 -100 1 4 7 10 Effect of
FF on INO..."

2008__blinder_ehrmann_fratscher_dehaan_jansen__..._ECB.pdf p1 (raster 1.000) — the outlier, see
false-positive note:
"Working Paper Series No 898 / May 2008 Central Bank Communication and Monetary Policy A Survey
of Theory and Evidence by Alan S. Blinder, Michael Ehrmann, Marcel Fratzscher, Jakob De Haan and
D[avid-Jan Jansen]..."
```

(8 quotes shown per the range that matters — the population is not uniformly corrupt, see below.
Two more from p36 and p50 of the 1994 NBER document, both similarly corrupted table pages
continuing the same run as p35, are in the raw CSV/manifest but omitted here for length.)

## Judgement

### Is 0.90 the right threshold?

**Yes — and the corpus shows a wide clean gap, not a marginal call.** Raster coverage among
born-digital pages with coverage above 0.5 is: `0.516 .. 0.789` (16 genuine born-digital figure
pages, one document — `2021__nagel__ml_ap.pdf`, plus one page of `2023__swanson`), then a **hard
gap with zero pages** from 0.79 to 0.94, then `0.940 .. 1.000` (72 pages, the fake-native
population plus the one outlier). `RASTER_DOMINANCE_RATIO = 0.90` sits in the middle of that empty
gap. Any threshold from ~0.80 to ~0.93 would separate the same two populations identically on this
corpus — 0.90 is not a fragile choice. No change recommended.

### Regression-guard population (the reverse false-positive risk)

The 16 pages at raster 0.52–0.79 are real born-digital figure pages from a single modern paper
(`2021__nagel__ml_ap.pdf`, a dense ML-in-asset-pricing survey with full-page charts), each carrying
400+ words of genuine native prose (`has_tables=False`, i.e., not even table content — just prose
with a large embedded figure). These are exactly the population B1's regression guard is written to
protect, and this corpus supplies real specimens for that test fixture, not just a synthetic one.

### Are there fake-native pages raster coverage does NOT catch?

**No material population found in this corpus.** I looked specifically for the class the ticket
names — a born-digital PDF with a broken ToUnicode map, corrupt text, and *no* raster (i.e., a page
that would evade B1's raster gate entirely). None of the 72 fake-native pages are outside the
raster >= 0.90 band by construction, so that doesn't answer the question directly; instead I
checked all 2771 born-digital pages with raster < 0.90 for `has_encoding_hygiene_suspect` — a
*different*, already-existing, already-surfaced signal (midcap-fusion / slash-digit / run-on token
ratio) that catches a weaker form of the same failure mode. It fired on 738/2771 (26.6%) of
low-raster born-digital pages, concentrated in math-heavy papers (`2003__woodford.pdf`: 161 pages;
`2023__jha_qian_weber_yang`: 70; `2021__nagel__ml_ap.pdf`: 36). Spot-checking `woodford.pdf` p5
found the two flagged tokens were `Cataloging-in-Publication` and a URL — **not corruption**, a
false trigger of the ratio on legitimate hyphenated/URL text. This flag is real, already visible
(notes + `has_encoding_hygiene_suspect` on `PageAssessment`, not silent), and separate from the
prose-corruption class this plan is about; it does not by itself demonstrate a population of
`Eftes`/`Sigrificace`-style lexical corruption on non-raster pages. **I found zero such pages in
this 40-paper corpus.**

**Verdict on B2: do not build it against this evidence.** The ticket's own condition for
dispatching B2 — "a material population of lexically-corrupt pages that B1's raster check does not
cover" — is not met here. Close B2 as unnecessary, with a note that the closing decision is
corpus-bound: a 40-paper economics/finance corpus with two old NBER/Econometrica scans is not proof
that no born-digital PDF anywhere has a broken-ToUnicode lexical-corruption class outside raster
detection, only that none turned up in this sample. If a future scan corpus surfaces one, reopen
with that page as the new reproducing case (the same evidentiary bar this plan itself sets).

### False-positive risk in the reverse direction

**One real specimen found, and it matters for B1's `Done when`.**
`2008__blinder_ehrmann_fratscher_dehaan_jansen__..._ECB.pdf` page 1 is a genuine ECB working-paper
title page: full-bleed cover art at raster coverage 1.000, with a completely clean 34-word title
block (`"Working Paper Series No 898 / May 2008 Central Bank Communication and Monetary Policy..."`)
as the entire text layer. Under B1's rule (raster >= 0.90 AND text layer present => not born-digital)
this page would be misclassified as scanned and re-routed to OCR, even though its native text is
already perfect. This is exactly the cost the ticket's own tradeoff note accepts ("over-refusing
costs an OCR call; under-refusing costs content") — flagging it here so B1's regression-guard test
suite includes a **full-bleed-background title page**, not just a mid-page figure, as a second
must-stay-`True` fixture alongside the `2021__nagel__ml_ap.pdf` figure pages. It is a real, sampled
false positive, not a hypothetical one.

## What I could not determine

- Whether B1's rate is representative of the *whole* corpus of scanned/OCR'd historical papers a
  user might feed socr — this run only had two such documents, both pre-1995 and both wholly
  scanned, so within-document homogeneity (once a document is a scan, essentially all its pages are)
  means the population count here is really "2 documents, N pages each," not 72 independent draws.
  Statistical confidence on the 0.90 cutoff's generality is bounded by that.
- The lexical-quality question for B2 was answered by absence of evidence in this corpus, not by a
  designed negative control; I did not build or calibrate a reference-vocabulary check (the ticket's
  own "no tuned constant" bar), only spot-checked with the system dictionary, which itself proved
  noisy (many legitimate technical/proper-noun tokens are OOV) and is not proposed as any part of a
  future signal.
