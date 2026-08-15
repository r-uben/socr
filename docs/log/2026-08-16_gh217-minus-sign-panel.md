# GH-217 — minus-sign glyph forgery: three-model panel + reconciled plan

**Date:** 2026-08-16
**Issue:** #217 — bug(born-digital): text layer returns `2` for the minus sign, so negative
coefficients ship as large positives
**Status:** planning only. No code written, no branch, no tickets filed.

## How this was produced

Three heterogeneous models were given the same grounding (issue body + the located code paths
in `src/socr/core/born_digital.py`, `state.py`, `orchestrator.py`, `manifest.py`,
`tables/structure_check.py`) and each answered independently: (a) root cause, (b) fix plan with
files/functions, (c) testing + surfacing across the four required levels.

| panelist | model | tool calls | tokens |
|---|---|---|---|
| gpt-sol | `gpt-5.6-sol` | 71 | 177,586 |
| grok | `grok-4.6` | 42 | 98,712 |
| fable | `claude-fable-5` | 11 | 84,624 |
| synthesis | `claude-opus-5[1m]` | 5 | 86,990 |

All three independently dumped the real PDF and reached the same layer-1 mechanism, which is
why the root cause below is stated as confirmed rather than proposed.

Full per-agent transcripts:
`~/.claude/projects/-Users-rubenffuertes-repos-tools-socr/51aec9a7-a135-4121-9979-74e883cb6846/subagents/workflows/wf_70d2fa1b-a20/`

---

# GH-217 — reconciled plan (three-model panel, synthesized)

**Verdict up front: fable's analysis wins.** It is the only one that (i) identified the true blast radius, (ii) named the fault unit correctly (the *font*, not the page or the document), and (iii) *verified a deterministic repair end-to-end* rather than proposing one. gpt-sol's contribution that survives is the fail-closed discipline and the surfacing matrix. grok's contribution that survives is the "don't let TR-3 score the corrected read against the corrupt native truth" warning and the demand to fix two lying comments. grok's actual mechanism — span-gap surgery — is rejected below with reasons.

---

## 1. Root cause — consensus

All three panelists independently dumped the PDF and agree on the mechanism, at three layers:

**Layer 1 — the PDF (this is the defect).** The paper embeds subsetted Type 1 pi fonts: `SRQIVV+Universal-GreekwithMathPi` (xref 212) and `RKPFOV+MathematicalPi-One` (xref 213). Both have **no `/ToUnicode`** and an `/Encoding` whose `/Differences` array is **empty**. The font program's own built-in encoding maps character code `50` (`0x32`) to glyph `/H11002` — the Adobe/Linotype Universal-Pi name for **minus**. gpt-sol and fable *both* read `/H11002` at code 50 out of the cleartext PFA independently; that fact is confirmed twice, not asserted once.

**Layer 2 — PyMuPDF (not a bug).** With no ToUnicode and no usable Differences, MuPDF's high-level extractor falls back to code-as-Latin-1 and emits ASCII `'2'`. Unanimous: this is faithful-to-a-broken-encoding, not invention. gpt-sol adds a useful corroboration — `get_texttrace()` reports **U+FFFD** for the same glyph, i.e. MuPDF's low-level API *admits* the mapping is unknown while the high-level API guesses. That asymmetry is worth an upstream report but is not socr's fix.

**Layer 3 — socr (where responsibility begins).** socr treats `page.get_text()` as ground truth with no glyph-identity audit. I verified the detector coverage claims directly in the tree:

- `count_digit_corruption` (`src/socr/core/born_digital.py:219`) is `_DIGIT_CORRUPTION_RE = re.compile(r"(?<![0-9A-Za-z])/[0-9]")` — it knows exactly one *signature* (#136's eaten-leading-digit slash), not the *mechanism*. Zero hits here.
- `_encoding_corruption_ratio` scores 0.59% on p21, under the `ENCODING_CORRUPTION_FLAG` floor.
- TR-3 (`src/socr/tables/native_verifier.py:170,218,1077,1116`) compares emitted numbers against `page.get_text("words")` — the *same corrupt string*. Agreement is guaranteed and proves nothing.
- The comment at `src/socr/core/state.py:40` ("Digit corruption never sets this — that class is routed to OCR at detection") and its twin at `born_digital.py:927` and `orchestrator.py:2586` are **false for this class**. grok and fable both flagged this; both are right. Fix the comments in the same PR.

**Blast radius — the issue understates it, and this is the most important finding.** fable's document-wide rawdict scan by font and em-width found **295 forged digit glyphs across 20 pages**, and the forged codes are exactly `1`, `2`, `5` = **plus, minus, equals** (H11001 / H11002 / H11005). gpt-sol independently counted 170 uses of the `2` alone across 15 pages — consistent with fable's number once you add the `1` and `5` classes. So the paper does not merely ship `−0.12` as `20.12`; it ships `+x` as `1x` and `a = b` as `a 5 b` throughout the equations.

**Dissent, and who was right.** grok scoped the problem to "pi-font digit glued onto a roman mantissa" and explicitly declined to generalize ("do not invent a general map any pi-font letter to the right Unicode… out of scope"). **grok was wrong**, and it is the load-bearing error in its answer: the same font, the same missing ToUnicode, the same one-line repair covers the `1` and `5` classes for free. Scoping to the glued-mantissa shape ships a fix that leaves two-thirds of the corrupted glyphs in place and calls it done.

---

## 2. Recommended plan (numbered steps)

**Converged:** all three put the primary work in `born_digital.py`, with mirrored flags in `state.py`, an audit event + sidecar round-trip in `orchestrator.py`, a manifest rollup, and a `ground_truth.py` exclusion. That skeleton is uncontested; adopt it.

**Adjudicated:** the detection predicate (font audit, not statistical signature), the remedy (in-memory ToUnicode injection, not OCR and not span surgery), and the granularity (font-scoped). Basis stated per step.

1. **`src/socr/core/born_digital.py` — new document-level font audit.**
   Add `audit_digit_fonts(doc: fitz.Document) -> FontForgeryAudit`, called **once** from `detect()` (currently `born_digital.py:534`, a bare per-page loop with no preflight — I confirmed this) and lazily from `detect_page()` (`:555`). For every font xref that contributes a digit-decoded character anywhere in the document, produce a verdict of `TRUSTED` / `FORGED` / `SUSPECT`. Predicate in §3.
   *Why over the alternative:* gpt-sol proposed a page-walking locator plus rendered-glyph rasterization. That is strictly more expensive (raster crops per candidate), strictly less certain (geometry infers; the font's own encoding vector *states*), and — critically — it can only find glyphs that land in the `2\d\.\d{2}` shape. Font audit is O(#fonts), exact, and finds the `1` and `5` classes it wasn't looking for.

2. **Repair at the decode layer via synthesized ToUnicode, not string surgery.**
   For `FORGED` fonts, build a CMap from the resolved glyph names, inject with `update_stream` + `xref_set_key`, and re-derive text from the patched document.
   *Why over the alternatives:* fable **verified this end-to-end** on the real paper — after injection, p21 `get_text("words")` yields 22 true minuses and zero `20.xx` tokens. grok's span-gap rewrite re-implements MuPDF's word assembly, only works where span geometry survives, and by construction misses every non-glued occurrence. gpt-sol's route-to-OCR is the worst option for *this* content: the repo's own CLAUDE.md documents qwen3-vl collapsing on dense tables, and these are 160-cell coefficient grids. A proven deterministic decode beats a VLM re-read of 20 pages of numbers.

3. **Materialize the repaired document once, run-scoped — do not patch per call.** *(This is the trap no panelist caught; see §8.)* There are ~25 independent `fitz.open(state.handle.path)` sites across `orchestrator.py`, `tables/extract.py`, `tables/native_verifier.py`, `core/document.py`, `benchmark/`. An in-memory patch inside `born_digital` fixes *only born_digital's view*; TR-3 and the table extractor would still read the corrupt layer and would then hard-fail the corrected output. Write the patched bytes **once** to a run-scoped temp path and rebind the handle (`DocumentHandle.path` or a new `repaired_path` consulted by a single accessor) so every downstream `fitz.open` sees the same repaired document. **Never** write back to the source PDF.

4. **`SUSPECT` fonts (no ToUnicode, glyph names unresolvable) → existing #136 remedy.** At the existing gate site (`born_digital.py:~861`), return `is_born_digital=False`, `native_text=""`, note the font name. Here OCR genuinely is the only reader left. This closes the general hole for TeX pi fonts and old publisher symbol subsets that aren't in the name table.

5. **New `PageAssessment` fields** (`born_digital.py:291`), following the B1/GH-200 pattern — set once at the evidence, never re-derived: `native_glyphs_repaired: int`, `has_untrusted_digit_font: bool`. Plus a `DocumentAssessment` (`:359`) record naming the forged fonts and their resolved mappings.

6. **`src/socr/core/state.py`** — mirror both fields on `PageState`; propagate in `apply_born_digital()`. **Delete the false comment at L39–40** and its twins (`born_digital.py:927`, `orchestrator.py:2586`).

7. **`src/socr/pipeline/orchestrator.py`** — audit event `native_glyph_forgery_repaired` / `native_digit_font_untrusted` alongside the hygiene emission (~L2587); sidecar write/restore next to the B1/GH-200 flags (~L4445 / ~L4642); document rollup (~L4853).

8. **`src/socr/core/manifest.py:329–333`** — add `has_untrusted_digit_font` to the `native_distrusted` union (the existing `native_table_unverifiable` / `native_table_structure_defective` / `native_table_header_unattributed` tuple I read at L329–333). An unresolved page must not freeze as native SUCCESS. A *repaired* page is not distrusted — it ships SUCCESS with the count in the notes and the flag in the sidecar.

9. **`src/socr/benchmark/ground_truth.py:69`** — `_assess_usability` must exclude pages whose font audit is `FORGED`-unrepaired or `SUSPECT`. Same GH-39 contamination rationale that already gates `count_digit_corruption` there.

10. **`src/socr/tables/native_verifier.py`** — no code change *if* step 3 is done properly (the verifier reads the repaired document). If step 3 is deferred, the verifier must accept an injected word list. **Prefer step 3**; threading corrected evidence through the verifier is a second source of truth waiting to diverge.

11. **Keep `count_digit_corruption`.** Different mechanism: the slash class has a ToUnicode that lies subtly; this class has none. Retire nothing.

---

## 3. Detection predicate

Three stages, none of which is a count threshold.

**Stage 1 — trigger (exact, definitional).** For a font that contributed at least one digit-decoded character: `doc.xref_get_key(xref, "ToUnicode")` is null **and** `/Differences` is empty or absent. Under those conditions the decode is *definitionally* a guess — MuPDF has nothing to map from. No constant.

**Stage 2 — identification (exact, from the font's own encoding).** Extract the font (`doc.extract_font`), parse the cleartext (pre-`eexec`) section for `dup <code> /<glyphname> put` with a stdlib regex — no fontTools dependency. If the glyph name at a digit code is **not** the AGL digit name (`zero`…`nine`), the decode is **proven forged**. Resolve the true codepoint through a documented table `_PI_GLYPH_TO_UNICODE`: H11001→U+002B, H11002→U+2212, H11003→U+00D7, H11005→U+003D, H11021/H11022→`<`/`>`, H11032/H11033→primes, H11006→±, plus AGL names. This is an encoding table with a citable source (Adobe Universal-Pi glyph naming) — the same category as Unicode data, not a magic constant. Unresolvable name → `SUSPECT`, not a guess.

**Stage 3 — corroboration when the program can't be parsed (CFF/TrueType), self-calibrated.** Compare the `/Widths` entry at the digit code against the width of the same digit in the document's *trusted* digit-bearing fonts. These are exact integers from the PDF (833–910 vs 500 in this paper) — a **class inequality against the document's own baseline**, zero tolerance parameter. Outcome is `SUSPECT` (→ OCR), never `FORGED` (→ repair); width alone never authorizes a substitution.

**On glyph-geometry confirmation.** gpt-sol's rasterize-and-measure (aspect ratio ~12.3 for the stroke vs ~0.79 for a real Times `2`) is real evidence and a legitimate *diagnostic*, but it is **not needed** once the font's encoding vector names the glyph, and it should not become a predicate: it introduces a raster step, a foreground-threshold decision, and an aspect-ratio comparison that would have to be calibrated. gpt-sol was careful to say those figures "diagnose this specimen; they should not become thresholds" — correct, and the way to honor that is to not build the geometry path at all. Keep it as an optional debug utility if anyone wants it.

**Rejected outright:** the issue's own sketch — "≥5 tokens in [20,30) beside ≥5 parenthesized SEs". Two magic thresholds, banned by house rule; under-inclusive (blind to the `1` and `5` classes and to the ~12 corrupted non-table pages in this very paper); and it would fire on a legitimate table of values in the twenties. All three panelists independently rejected it. So do I.

---

## 4. Granularity: page vs document

**Decision: the fault unit is the FONT.** Audit fonts once per document (O(#fonts)); apply the verdict per page via font usage (`page.get_fonts()`). This is not a compromise between the two options — it dominates both.

- It gets **document-level coverage** for free: no sampling, no risk of a clean title page hiding the defect. The maintainer's "sample one page per document" idea is dead — gpt-sol is right that sampling creates an avoidable false-negative path for a correctness gate.
- It gets **page-level precision** for free: pages 1–4 of this paper never load the pi fonts and stay in the fast native lane, untouched, still `is_born_digital=True`.

**Losing argument:** gpt-sol's "the entire document remains untrusted until the all-page font-usage scan finishes" framing, which treats untrust as a document property that later narrows. That's an unnecessary intermediate state — fonts are document-level xref objects, so the scan resolves before any page assessment is finalized. grok reached the font-scoped answer too ("the unit of distrust is the embedded font"); fable stated it most cleanly as "page vs document is a false dichotomy."

---

## 5. Remedy

**Decision: repair-in-place at the decode layer, with loud accounting; refuse-and-route-to-OCR when proof is absent.**

Ranking of the four options for this defect class:

| option | verdict |
|---|---|
| flag-only | **Rejected, unanimously.** Ships known sign-inverted coefficients. Violates "a wrong number is worse than a missing one" in the most direct way available. |
| route-to-OCR as default | **Rejected as default, kept as fallback.** gpt-sol's position. VLM re-reads of 20 pages of dense coefficient grids can introduce *new* digit errors, and TR-3 would hard-fail a correct `−0.12` against native `20.12` and discard the good read. It is strictly worse than a proven decode. |
| span-gap string surgery | **Rejected.** grok's position. Re-implements MuPDF word assembly, misses the `1`/`5` classes and every non-glued occurrence, and depends on span geometry surviving. |
| **ToUnicode injection (repair-in-place)** | **Adopted.** Verified end-to-end by fable on the real paper. |

**Is sign repair acceptable under "a wrong number is worse than a missing one"?** Yes, and the rule *demands* it here, because the repair is:

1. **Proven, not inferred.** The font's own encoding vector names the glyph `/H11002`. We are not guessing what a dash-shaped mark means; we are reading the mapping the PDF failed to declare externally.
2. **Not silent.** Counted per page (`native_glyphs_repaired`), named per document (forged fonts + mappings), audit-logged, printed by the CLI. The #136 lesson — notes alone reach nothing that ships — is explicitly honored by carrying a *flag*, not a note.
3. **Non-destructive.** In-memory / run-scoped temp only; the source PDF is never rewritten.
4. **Fail-closed where proof runs out.** `SUSPECT` fonts are refused loudly and routed to OCR; if OCR is unavailable or unverified, ship the explicit failure marker, never the corrupt native text. gpt-sol's fail-closed discipline is adopted wholesale for this branch.

**`--native-only` interaction.** Repair still runs — it *is* native-side, no OCR involved — and repaired pages ship SUCCESS. `SUSPECT`-font pages under `--native-only` ship the explicit failure marker, not the corrupt layer. gpt-sol is right that policy ("no OCR") must not be reinterpreted as consent to ship wrong numbers. grok's proposal to demote *every* touched page to WARNING is rejected: once the decode is provably correct there is nothing left to warn about, and a WARNING that fires on 20 of 29 pages of a correctly-repaired document trains operators to ignore it.

---

## 6. Testing + surfacing

**Hermetic strategy — no provider, no ollama, no corpus bytes.**

*Unit layer (most of the coverage, all pure functions):*
- `parse_type1_encoding()` fed a synthetic PFA cleartext string → asserts `{49: "H11001", 50: "H11002", 53: "H11005"}`.
- CMap builder fed a mapping → asserts emitted CMap bytes.
- Width-class check fed synthetic `/Widths` arrays → asserts `SUSPECT` classification and that a normal 500-width digit font stays `TRUSTED`.
- Glyph-name table: unknown name → `SUSPECT`, never a guessed codepoint.

*Integration layer (one synthetic PDF, ~100 lines of fixture builder, zero copyrighted bytes):* a minimal Type 1 font whose cleartext encoding maps `0x32`→`/H11002`, no ToUnicode, empty Differences, content stream printing that glyph followed by Helvetica `0.12 (0.16)`. gpt-sol prototyped exactly this in memory and confirmed PyMuPDF reproduces the extraction failure. Build it multi-page, borrowing gpt-sol's control matrix (it is the best-designed part of its answer):

- p1: clean prose, no pi font → proves first-page sampling would miss the defect, and that the page stays in the native lane.
- p2: the forged coefficient → repaired to `−0.12`, `native_glyphs_repaired == 1`.
- p3: a genuine Helvetica `20.12` → **stays `20.12`**, no flag. This is the false-positive control and it must be in the suite.
- p4: forged `1` and `5` in an equation → proves the fix is not scoped to the mantissa shape (grok's blind spot, guarded by test).
- p5: same font, heading only → proves font-scoped propagation without a token count.
- variant: strip the encoding vector → `SUSPECT`, `is_born_digital=False`, OCR note.

*House rules:* the one orchestrator test (flag → sidecar → resume round-trip) must patch `_available_engines_for_agentic` to return `[PROFILE_QWEN_LOCAL]`. Lint with `uvx ruff@0.16.0 format --check .` — **not** the venv ruff. **Byte-identity golden tests:** a repaired document produces different (correct) bytes, so no forged-font document may enter the golden fixture set without a deliberate regeneration commit.

**Four surfacing levels (all four, or it didn't ship):**

| surface | content |
|---|---|
| **Page status** | `native_glyphs_repaired: int` and `has_untrusted_digit_font: bool` on `PageState`. Repaired → `SUCCESS` + note `"born-digital: N forged digit glyphs repaired (Universal-GreekwithMathPi: H11002→U+2212, …)"`. Untrusted font, unrecovered → `ERROR`, `audit_passed=False`, explicit marker + page image. |
| **Document status** | Manifest rollup lists forged/suspect fonts and total repaired glyphs. Any `SUSPECT` page OCR could not cover → the document must not report clean SUCCESS (PARTIAL / `AUDIT_FAILED` semantics as used elsewhere). A fully repaired document *may* return SUCCESS, with the repair recorded. |
| **Sidecar `NNN.json`** | Both fields round-trip (orchestrator ~L4445 / ~L4642), plus font names and the resolved mapping, so resume preserves the verdict. |
| **CLI** | Analysis: `glyph forgery: font Universal-GreekwithMathPi maps −/+/= to digits 2/1/5`. End of run: `295 glyphs repaired across 20 pages` / `N page(s) untrusted digit font, unresolved`. Unresolved single-file run exits nonzero, pointing at the page image and `audit_log.json`. Never print "completed with warnings" over sign-inverted values. |

---

## 7. Open disagreements

These are real splits, not rhetorical ones.

**7.1 — Remedy: repair vs OCR. (fable+grok repair / gpt-sol OCR.)** gpt-sol argues the raster contains the true stroke so a vision model can recover it, and that repair is "blind textual replacement." That characterization is wrong for the *ToUnicode* mechanism — the font names the glyph — but gpt-sol's underlying worry is legitimate: repair is only as good as the glyph-name table. **Why it matters:** if the table is wrong for some font, we silently ship a confidently-wrong character instead of a visibly-wrong one. **What settles it:** run the repair across the 40-paper corpus and diff repaired-vs-original text; any substitution whose glyph name is outside the documented Adobe Universal-Pi set should not have fired. Zero out-of-table substitutions ⇒ repair is safe to default. I adjudicated for repair on the strength of fable's end-to-end verification plus the `SUSPECT` escape hatch, but this measurement should run before merge.

**7.2 — Scope: general glyph-name repair vs the glued-mantissa case only.** grok explicitly declined to generalize; fable generalized and proved it works. **Why it matters:** 295 vs ~155 corrupted glyphs, and every equation in the paper. I ruled for fable — grok's scoping is the one clearly wrong call in an otherwise sharp answer.

**7.3 — Page status for repaired pages: SUCCESS (fable) vs WARNING/`audit_passed=False` (grok, for every page that touched the font).** **Why it matters:** flag fatigue vs conservatism. grok's rule would mark 20 of 29 pages of a *correctly repaired* document as failing audit. I ruled for SUCCESS-with-accounting, contingent on 7.1's corpus diff coming back clean. If it doesn't, grok's conservatism becomes correct and this flips.

**7.4 — Should the document-level preflight also run inside `detect_page()`?** gpt-sol says `detect_page()` must not be able to bypass the preflight; fable says run it lazily per call. Functionally equivalent, but the cost differs on per-page callers. Not load-bearing; implementer's call, with a test that `detect_page()` on a forged-font page cannot return unrepaired text.

**No disagreement on:** the root-cause mechanism, the rejection of the count-based signature, font-scoped granularity, TR-3 contamination, or the four surfacing levels.

---

## 8. Risks and non-obvious traps

1. **The multi-`fitz.open` trap — biggest implementation risk, missed by all three panelists.** I counted ~25 independent `fitz.open(...)` sites (`orchestrator.py` ×14, `tables/extract.py:261`, `native_verifier.py`, `core/document.py`, `core/chunker.py`, `benchmark/`). An in-memory patch confined to `born_digital` leaves TR-3 (`native_verifier.py:170,218,1077,1116`) reading the *corrupt* layer and hard-failing the corrected table output — the exact contamination grok warned about, arriving through a door grok didn't check. **Step 3 of the plan is not optional.** Patch once, materialize once, rebind the handle.

2. **Resume fingerprint omits source version (known, in memory).** Pages already terminal from an earlier run will **not** be re-assessed when this detection logic lands — they restore verbatim carrying `audit_passed=True`. This is exactly the #214 shape the manifest comment at `manifest.py:322–325` already documents. The PR must state that affected corpora need a forced re-run, and ideally this is the ticket that finally adds a source-version component to the fingerprint.

3. **False positives on legitimate `[20,30)` values.** Near-zero under the adopted predicate — a Times-Roman `20.12` has a ToUnicode or a standard encoding and never reaches the audit. This risk existed only for the rejected statistical signature. The p3 control test locks it in.

4. **#136 gate interaction.** `count_digit_corruption` stays. The two mechanisms are disjoint (lying ToUnicode vs absent ToUnicode) and the new audit runs *before* the existing gate site so a `SUSPECT` font short-circuits to the same OCR remedy without double-counting. Watch for a page hitting both paths.

5. **Benchmark contamination in both directions.** `ground_truth.py:69` currently excludes only `count_digit_corruption` hits. Until 7.1's corpus diff lands, exclude forged-font pages from ground truth **even after repair** — grading OCR against a repair we haven't validated corpus-wide is circular.

6. **Glyph-name table coverage.** TeX pi fonts, older Elsevier symbol subsets, and non-Adobe naming schemes will land in `SUSPECT` → OCR until added. That is the correct failure direction (loud refusal over guessed repair), but it means the fix's *coverage* is narrower than its *mechanism*. Say so in the PR.

7. **MuPDF version dependence.** The Latin-1 fallback behavior is a MuPDF implementation detail. A future PyMuPDF could start emitting U+FFFD from `get_text()` (matching its own `get_texttrace()`), which would change the extracted string and silently alter what the audit sees. Pin the observed behavior in a test that asserts the *unpatched* fixture extracts `2`, so a PyMuPDF bump that changes it fails loudly rather than quietly.

8. **The three lying comments** (`state.py:40`, `born_digital.py:927`, `orchestrator.py:2586`) claiming all digit corruption routes to OCR at detection. Verified false. They actively mislead the next reader into believing this class is already covered — which is plausibly part of why it shipped. Fix them in this PR.

---

# Addendum — the corpus-diff measurement (blocks the plan folder)

**Written 2026-08-16, after the panel.** No tickets should be cut until this runs: it decides
disagreement 7.1 (repair vs OCR as the default) and, through it, 7.3 (SUCCESS vs WARNING for a
repaired page). Both were adjudicated *contingent on this measurement*, so scaffolding a plan
folder first would bake in an unsettled call.

## The question it must answer

**Does the glyph-name table ever authorize a substitution it should not have?**

That is the only real objection to repair. gpt-sol's worry, restated precisely: repair converts a
*visibly* wrong character into a *confidently* wrong one, and the blast radius of a bad table entry
is silent. So the measurement's job is to **falsify** the repair, not to confirm it.

## Falsification criteria — decided in advance

The repair is safe to default **only if all four hold**:

| # | criterion | threshold |
|---|---|---|
| F1 | Substitutions whose glyph name is outside the documented Adobe Universal-Pi set | must be **0** (by construction — assert it, don't hope) |
| F2 | Fonts classified FORGED that carry a non-null `ToUnicode` | must be **0** (would mean the trigger is wrong) |
| F3 | Hand-verified substitutions that disagree with the rendered glyph | must be **0** out of the sample (see §Ground truth) |
| F4 | Text diff on documents with **no** FORGED font | must be **byte-identical** (negative control) |

Any F1/F2/F4 violation kills repair-as-default outright and promotes gpt-sol's route-to-OCR.
An F3 violation is worse — it means the table is wrong — and kills the approach entirely.

If F1–F4 all pass but the FORGED-font population is larger than expected (say >3 of 40 papers),
that is *not* a failure but it changes the ticket sizing, and 7.3 should be re-argued: a repair
firing across many documents deserves more surfacing than one that fires on a single outlier.

## Ground truth — the part that must not be circular

F3 is the load-bearing one and it cannot be checked against the glyph-name table, because that
table is the thing under test. The only independent evidence is **the rendered page**.

Procedure: for a stratified sample of substitutions — every distinct `(font, code, glyph-name)`
triple, plus ≥10 instances of the most common one — crop the glyph's bbox from a rendered page at
sufficient DPI and eyeball it against the substituted codepoint. Small sample, human judgement,
recorded per-instance in a table like the B1/TR-3 hand judgements
(`docs/log/2026-08-15_b1-hand-judgement.md` is the format precedent).

Do **not** automate F3 with an aspect-ratio rule. That is the same geometry predicate §3 rejected,
and using it as ground truth for the table would launder a heuristic into a verification.

## Shape of the run

Read-only, standalone script, no changes to `src/socr` — this is a measurement, not the
implementation. Output a JSON corpus record in the `/tmp/b1probe/corpus.json` style so it is
re-analysable without a re-run.

- **Input:** `/tmp/b1probe/list.txt` (40 papers, confirmed present 2026-08-16). Same list as the
  TR-3 and B1 judgements, so the three measurements stay comparable.
- **Stage A — font inventory.** Every font xref in every document: `ToUnicode` present?
  `/Differences` empty? which codes decode to digits? what glyph name sits at each digit code?
  Record all of it, including for TRUSTED fonts — the TRUSTED population is what calibrates the
  Stage-3 width check.
- **Stage B — classify** TRUSTED / FORGED / SUSPECT per §3. Assert F2 here.
- **Stage C — repair and diff.** For FORGED fonts only: inject the CMap in memory, re-extract, and
  diff per page against the original. Record every substitution as
  `(doc, page, font, code, glyph-name, old-char, new-char, count)`. Assert F1 and F4 here.
- **Stage D — hand judgement.** The F3 sample, per §Ground truth.

## What the output feeds

- **F1–F4 pass** → repair defaults, 7.3 resolves to SUCCESS-with-accounting, plan folder gets cut
  with steps 1–11 of §2 intact.
- **F1/F2/F4 fail** → route-to-OCR becomes the default, §5's ranking inverts, and the plan folder
  is a different shape (no CMap injection, no repaired-document materialization, so §8 trap 1
  evaporates and the ticket count drops).
- **F3 fail** → stop; the glyph-name table is not trustworthy and neither remedy is ready.
- Either way, the Stage-A inventory is reusable: it is the calibration data for the Stage-3 width
  check and it tells us how common missing-ToUnicode fonts are corpus-wide, which is a number
  nobody currently has.

## Cost note

Stage A is O(#fonts) over 40 PDFs — seconds. Stage C only touches FORGED documents, expected to be
1 (this paper). Stage D is the only human-time cost and it is deliberately small. This measurement
is cheap; the reason to run it before cutting tickets is not cost, it is that it changes the plan's
shape.
