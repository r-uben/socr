# GH-960: pages whose text layer garbled the maths re-route to a whole-page read

Branch `fix/960-garbled-math-reroute`, from origin/main aba13cd8 (after #989, #990, #1002).

## Problem
The native-lane audit (60 trusted-native pages, seed 20261002) found 8 pages whose
mathematics the text layer garbled while they shipped SUCCESS: woodford p387 and p673,
maggiori p27, Hameed p9, coibion_gorodnichenko p15, ramey p21, hansen_mcmahon_prat p8,
trammell_korinek p6. Symptoms: Σ extracted as `N`, Φ as `)`, fractions flattened, display
equations shredded one token per line, private-use glyphs, and math in plain Cambria
extracting as Syriac/Tamil codepoints. No existing check fired on any of them. Widening
`_MATH_FONT_RE` alone fixes none: it sends the page to the P4-R region lane, which keeps the
garbled native text as its floor.

## Detector (the measured union)
`born_digital.detect_garbled_math(page, text)` returns four counts. A non-zero count is a hit:
- `private_use`: private-use codepoints (`count_pua_chars`; catches Hameed);
- `math_alphanumeric`: characters whose Unicode name starts `MATHEMATICAL`;
- `misdecoded_script_letters`: letters of a script in `MISDECODED_MATH_SCRIPTS` (Syriac,
  Tamil, and further non-Latin scripts the English-language corpus is not written in; catches
  coibion and ramey). Latin, Greek, Cyrillic, CJK and modifier letters are excluded;
- `unlisted_math_font_chars`: characters in spans whose font matches `_MATH_FAMILY_FONT_RE`
  (`math`, anchored MathTime `R?MT(MI|SYN?|EX)B?`, `MnSymbol`) but not `_MATH_FONT_RE`
  (MathTime, UniMath, LibertinusT1Math, MathematicalPi, Fourier-Math, "Cambria Math").

There is no count threshold. Any one character is a hit, as in the measurement; the regexes
and the script set are named constants with their derivation documented in the code.
Population (8,398 trusted-native, non-chart pages): the union re-routes 1,849 pages (22%, 115
documents). The narrow private-use plus script set re-routes 254 pages and catches only 4 of
the 8.

## Change
Same pattern as #961 and #990:
- `core/born_digital.py`: detector; `PageAssessment.garbled_math_signals` (non-zero counts)
  and `garbled_math_scan_failed`. Any hit, or a raising scan (fail closed), sets
  `needs_ocr_enhancement`. The PUA comment that said whole-page OCR makes these pages worse
  is replaced: the A/B below measured the opposite.
- `core/state.py`: the two fields copied to `PageState`.
- `core/manifest.py`: `garbled_math_suspect(p)`; joins the `--native-only` distrust
  short-circuit and the fallback failure-mode chain.
- `core/result.py`: `FailureMode.NATIVE_GARBLED_MATH`.
- `pipeline/orchestrator.py`: audit kind `garbled_math_native` (`data.signals`, `data.error`,
  `data.native_only`), recomputed every run, not in `_RESUME_REPLAYED`. The chart-asset lane
  demotes to WARNING / `native_garbled_math`. The new `garbled_math_retained_pages`
  document bucket blocks SUCCESS and emits `native_garbled_math_retained` plus a CLI line.
  The resume ledger gate refuses a cached native/chart SUCCESS when this run's scan fires.
- Corrupt-math hybrid lane: `_is_corrupt_math_recovery_page` excludes a flagged page, so it
  goes whole-page. This needed one condition; Pastel p4/p5/p10 all fire, so the hybrid no
  longer owns them.
- `docs/OUTPUT.md`: failure mode and both audit kinds.
- `tests/test_native_clean_pua.py` pinned the old rule ("a PUA glyph must NOT force
  whole-page OCR"). #960 reverses it on measured evidence, so the pin now asserts the re-route.

## GPU A/B (qwen3-vl:30b-a3b-instruct, default ladder and judge)
Single-page PDFs, arm A = main, arm B = forced `needs_ocr_enhancement` (and the hybrid lane off),
two runs per arm. Each page was judged against a 200 dpi render.
Artefacts: `~/.local/state/socr-housekeeping/gh960/ab/` (`RESULT.md`, `RESULT-post989.md`,
`judgements.md`).
- WRONG pages: the model read fixed 8/8 in both runs. Residuals: maggiori's script-E read as
  `g`; running headers and page numbers dropped.
- Clean pages the union also fires on (caballero p7, shapiro p4, del_vitto p1, romer p12,
  trammell_korinek p25, theodoridis p830 and p1014): text worse on 0/7 in both runs.
  The only losses were running headers, page numbers, one figure link and one footnote marker.
  No number was invented.
- Pastel (corrupt-math hybrid): the hybrid output was unreadable, with invented syntax-only
  LaTeX. The whole-page reads were near-perfect on all 3 pages in both runs.
- Halts: 12/36 B page-runs halted on 3498cb27 (judge timeout, then canary, then
  PARTIAL_SAVE). On 3a43203b (after #989) 0/18 halted and Pastel shipped the model read
  6/6, SUCCESS. Two of the 18 page-runs still shipped native WARNING, because the judge timed
  out on the qwen read. That issue and the Hameed status residual (`native_math_unrecovered`
  stays on a model-replaced page) are filed separately.
- Cost: a forced read adds about 230–290 s per page, call it 250 s; native takes 5–11 s.
  For the 1,849 pages that is 1,849 × 250 s ≈ 128 GPU-hours, the "~120 GPU-h" order. On
  corrupt-math pages the hybrid itself took 122–1,258 s, so whole-page is no slower there.
  Caveat: other sessions shared the GPU during part of the A/B, which inflated times and
  halts.

## Verification
- Offline, with no model and the analysis phase only (`impl/measure_detector.py`): the implemented
  detector fires on 8/8 WRONG pages and on all 7 clean pages the union fired on. It also
  fires on Pastel p4/p5/p10, which leave the corrupt-math lane. Output:
  `~/.local/state/socr-housekeeping/gh960/impl/detector_on_audited_pages.json`.
- On all 60 audited pages (`impl/measure_sample.py`) the implemented detector fires on exactly
  the 15 pages the measured union fired on: none extra, none missing.
- `tests/test_gh960_garbled_math.py`: hermetic (`_available_engines_for_agentic` patched,
  `_resolve_judge_model` → ""). The pins are differences between the detector neutralised and
  live in the same process: routing, event, native-only demotion, no-provider demotion,
  chart lane, fail-closed scan, corrupt-math lane, and the resume refusal with its control.
- Mutants in an external copy, with a `socr.__file__` canary and an uncapped anchor count of 1.
  Each was killed: routing off (9 failed), corrupt-math exclusion off (2), resume refusal
  off (4), chart demotion off (3), manifest failure mode off (4).
- Full suite: 6698 passed, 1 failed (that old PUA pin, updated above and rerun green), 2 skipped,
  4 xfailed. `uvx ruff@0.16.0 format --check .` clean.
