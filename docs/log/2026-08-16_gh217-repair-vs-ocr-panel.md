# GH-217 round 2 — repair vs OCR, re-opened against corpus evidence

**Date:** 2026-08-16
**Issue:** #217 — born-digital text layer returns `2` for the minus sign
**Status:** planning only. No code, no branch, no tickets.

## Why this panel ran

Round 1 (`docs/log/2026-08-16_gh217-minus-sign-panel.md`) adjudicated **repair over OCR**,
contingent on a corpus measurement. The measurement ran: a read-only font audit over **407
PDFs** (`/tmp/gh217_corpus_font_audit.py`, detail in `/tmp/gh217_corpus_font_audit.json`).
It falsified round 1's proposed 16-name `_PI_GLYPH_TO_UNICODE` table — **zero** papers
resolved completely.

Same three heterogeneous models, same identical grounding, re-arguing the decision.
gpt-sol (which argued OCR in round 1 and was overruled) was told so, in the grounding all
three received.

| panelist | model |
|---|---|
| gpt-sol | `gpt-5.6-sol` |
| grok | `grok-4.6` |
| fable | `claude-fable-5` |
| synthesis | `claude-opus-5[1m]` |

### Two corrections the orchestrator verified before dispatch

1. **Harsher than the headline.** Not "resolved zero papers completely" — **zero papers
   classified `FORGED_RESOLVABLE` at all**. All 50 landed in `SUSPECT_UNKNOWN_NAME`.
2. **Oldstyle figures are not corruption.** `zerooldstyle`@48 … `nineoldstyle`@57,
   perfectly aligned, zero misalignments corpus-wide. Oldstyle figures *are* the digits;
   decoding them to `0`–`9` is correct. The scan's predicate (`name ∉ {zero..nine}`)
   false-positives on them. Impact: 1 of 50 papers is oldstyle-only, 49 carry genuine
   corruption, 0 mixed.

Both were given to all three panelists rather than silently corrected.

---

## 1. Verdict

**Hybrid, with repair as the primary route and fail-closed flagging — not OCR — as the residue.** Round 1 **survives in narrowed form**: its *route* (synthesize a ToUnicode CMap from the font's own encoding vector and re-decode natively) was right and the corpus strengthened it; its *implementation* (a 16-name private Linotype table) was falsified and is dead. The narrowing is real, not cosmetic: repair now fires on a **font-family encoding identification**, not a name-lookup, and the repair predicate must be complete over *every used code* in the font, not just the ten digit slots. gpt-sol's round-1 dissent for OCR is again overruled, and more decisively than in round 1 — the corpus shows the affected population is 46 Computer Modern math papers whose corruption garbles formulas, plus exactly **one** paper (Romer & Romer, present twice under different filenames) with the digits-in-coefficient-tables catastrophe. Routing 245 Helvetica/WinAnsi papers, or dense coefficient grids, through `qwen3-vl:30b-a3b-instruct` to fix that is a net increase in wrong numbers.

## 2. Why — the reasoning

The measurement did not falsify repair. It falsified the belief that the corpus defect was *Linotype Pi fonts*. It is not. I verified the distribution directly from `/tmp/gh217_corpus_font_audit.json`:

| class | papers | digit-slot content |
|---|---|---|
| `*+CMSY*` / `CMBSY8` / MathDesign-Symbol (OMS layout) | 46 | `prime` 57, `element` 61, `infinity` 48, `universal` 20, `negationslash` 19, `mapsto` 1, `triangle` 2 |
| `*+CMEX10` / MathDesign-Extension (OMX layout) | 24 | extensible bracket/paren/brace **pieces** |
| `*+CMMI*` oldstyle figures | 1 | genuinely the digits 0–9 |
| `LibertineMathMI*` | 1 | `u1D44E`, `u1D44F`, `u1D453`, `u1D456`, `u1D457` |
| `Universal-GreekwithMathPi` + `MathematicalPi-One/Five` | 2 files, **one paper** | the round-1 canonical case |

Every one of the 11 unresolved `H*` names lives in `2000__romer_romer__…pdf` and `2010__romer_romer__…pdf` — the same paper twice. The "unresolvable residue" round 1 was calibrated on is not a corpus class; it is one document's tail.

**I adjudicated three factual disputes by reading the script and the JSON.**

**(a) fable's structural false-negative is real and the other two missed it.** In `audit_pdf`, if `extract_font` returns a non-empty *binary* buffer (CFF/Type1C, TrueType), `parse_encoding`'s Type-1 cleartext regex returns `{}` → `digit_glyphs` is `{}` → `forged` is `{}` → `continue`, **emitting no finding at all**. Binary-programmed fonts without ToUnicode were silently classified *clean*, not `SUSPECT_NO_PROGRAM`. So the 245 are only the fonts with an *empty* buffer, and an unknown CFF/TT population passed invisibly. This matters because gpt-sol and grok both argued about the 245 as if that bucket contained the unparseable CFF fonts. It does not — the CFF fonts are not in the report at all. Any argument resting on "the scan found X clean" is unsound. fable wins this and it is the single most consequential finding of round 2.

**(b) grok's 245-bucket numbers are correct and decisive.** 4,784 of 5,006 findings carry `/WinAnsiEncoding`; the basefont histogram is Helvetica 1,179 / Times-Roman 543 / Times-Italic 430 / Courier variants ~740. These are non-embedded Base-14 fonts where an empty extracted buffer is the *expected* result and the named encoding defines digit semantics by construction. Declaring them untrusted is quarantine-by-scanner-ignorance.

**(c) On AGL-the-dictionary, grok is right and fable is loose.** fable's stated predicate is "exact match in a vendored, versioned AGL, or the `uXXXX` algorithm" — but fable simultaneously (correctly) says oldstyle names must decode to ASCII digits. Those two rules contradict: AGL maps `zerooldstyle` into the Adobe **PUA** (U+F730…), and `bracketleftex`/`parenlefttp` likewise, and `negationslash` is a combining overlay. Applying fable's own predicate to fable's own corpus writes PUA non-characters over 28 correct digits and over every OMX bracket piece — a confidently-wrong character, which the house rule ranks *worse* than the visible Latin-1 fallback. grok's R4 (target must be a non-PUA standalone scalar; oldstyle → ASCII; `uXXXX` → the literal codepoint) is the fix and I adopt it verbatim.

**Position changes.** gpt-sol moved from round-1 "OCR" to "strict hybrid with refusal as terminal fallback." That is genuinely better reasoned, not just responsive — its completeness argument is the one thing neither other panelist got right, and I adopt it: *installing a partial ToUnicode can destroy text that currently extracts fine*, so repair must cover every code the document actually uses from that font, not only codes 48–57. But gpt-sol loses on the two structural questions: it never saw the family-vector structure (it treats each glyph name as an independent semantic litigation, which is why it ends up unable to justify repairing anything), and its default-refusal fallback is disproportionate — refusing 46 CM papers because CMSY puts ∈ at slot 50 would fail documents whose *numbers are fine*.

fable's core insight is the one that carries: this is not 50 fonts each guessing a name; it is **four published font families whose entire ten-slot vector matches their documented encoding**. That converts a weak per-name convention argument into a strong joint-consistency identification. grok independently reached the same structure and stated the boundary more rigorously. I take fable's diagnosis, grok's boundary rule, and gpt-sol's completeness requirement.

## 3. Sub-question 1 — does AGL + uXXXX + oldstyle rescue repair?

**Partly, and only under a rewritten predicate. "Name ∈ AGL ⇒ write AGL's codepoint" is an assumption about the font author, not evidence about this document, and it is a footgun — reject it.**

Three distinct epistemic objects, which the orchestrator's observation conflated:

1. **Document-local fact:** this font program's encoding vector maps code 50 → `/element`. Measured. Certain.
2. **AGL dictionary lookup:** fonts that follow the convention spell U+2208 as `/element`. This is a claim about *conventions*, and a subsetter can name a snowman `/element`. On its own: assumption.
3. **Algorithmic `uXXXX`:** `u1D44E` *contains* U+1D44E. This is a specified algorithm (AGL §uniXXXX/uXXXX), not a dictionary guess. Materially stronger than (2) and should not be lumped with it.

**What converts (2) from assumption to evidence is not the list — it's joint vector consistency plus font identity.** For the 46 CMSY papers, the BaseFont behind the subset tag says `CMSY10`, *and* the complete ten-slot layout (`48→prime, 49→infinity, 50→element, 54→negationslash, 55→mapsto, 56→universal`) is the published Knuth/BlueSky OMS encoding. One matching name is a coincidence-prone guess; the whole digit-slot vector matching a named family's published table is an **identification**. That is the difference between "this name is in a phonebook" and "this font *is* CMSY."

Two supports, in descending strength:

- **Normative:** ISO 32000 §9.10.2 designates glyph-name→AGL as the conforming text-derivation path when ToUnicode is absent. Repair is *completing the standard extraction path MuPDF abandons* (it reads Differences arrays but not built-in Type-1 encodings), not a socr invention. Acrobat and pdfminer already do this. This is fable's strongest point and it stands.
- **Adversarial threat model:** a deliberately misnamed font defeats vector matching, but that is out of scope for an academic-papers library, and closable later by glyph render-verify if ever needed.

**What is not rescued:** the 11 `H*` names. AGL does not define `H11007`. It means ∓ only if you cite the Linotype Mathematical Pi encoding for that family. `H11006 = ±` therefore `H11007 = ∓` is pattern inference — exactly the class of guess that turns a visibly-wrong character into a confidently-wrong one. Do not repair by inference. The sting fable identifies is correct and must be stated in the ticket: **round 1's "verified end-to-end" canonical paper is itself only partially resolvable by any published standard.**

## 4. Sub-question 2 — the 245 unparsed-program papers

**Better scan. Unambiguously. Not OCR, not refusal.** The 245 are overwhelmingly non-embedded Base-14 with named encodings; "the regex could not read it" is a statement about the regex. And the scan's *false negatives* (§2a) mean the current output cannot support any untrusted verdict in either direction — the rescan is mandatory regardless of what the repair policy turns out to be.

What the replacement scan parses, in cost order:

1. **PDF-dictionary triage, no font parsing.** `/Encoding` ∈ {`/WinAnsiEncoding`, `/MacRomanEncoding`, `/StandardEncoding`, `/MacExpertEncoding`}, or BaseFont ∈ {Symbol, ZapfDingbats} → digit codes are digits by construction. **Clean, done.** This alone should clear most of the 4,784 + 66 findings. The current script ignores the base encoding name entirely — that is its second-largest defect.
2. **Type 1 / `FontFile`:** existing `dup N /name put` parse. Already sufficient for all 50 hits.
3. **CFF / `FontFile3` `/Type1C`:** `fontTools.cffLib` — built-in Encoding gives code→GID, charset gives GID→name (SIDs against the standard strings or the custom String INDEX). Then the same family rule as Type 1.
4. **TrueType / `FontFile2`:** a usable Unicode `cmap` subtable gives code→Unicode **directly** — no names, no convention, no assumption. That is the strongest evidence class in the whole problem. Failing that, `post` v2.0 glyph names, applied under the PDF symbolic/non-symbolic Encoding rules (never `post` names as an encoding on their own).
5. **Type 0 / CID:** `/Identity-H` without ToUnicode is the genuine risk class (3 findings, 2 papers here). Parse CIDToGIDMap + descendant cmap, or mark untrusted. Never dump into the WinAnsi bucket.
6. **Fix both audit flaws:** oldstyle names resolve to digits (kill the false positive); a non-empty-but-unparsed buffer classifies **SUSPECT**, never silently clean (kill the false negative).
7. **Actualization check:** a suspect font is only dangerous if content streams (including Form XObjects) actually emit those codes under that font. Cheap, and it correctly separates "garbles a formula" from "garbles a coefficient grid."

**Undecidable after all of that:** no ToUnicode, no Differences, no predefined named encoding, a program that parses but yields names in no published family (and no `uXXXX` form), and the codes are actually used. On the evidence, that population is small — currently 11 glyph names in one paper. Those go untrusted, not OCR'd.

Cost comparison is not close: one font parse per xref, O(#fonts), no raster, no GPU — versus rendering and VLM-reading 245 papers, most of which are Helvetica.

## 5. Sub-question 3 — hybrid, and the boundary rule

The hybrid survives because the second path is mostly **flag**, not a second transcription pipeline — so the "two code paths, two silent failure modes" objection doesn't land. socr is already route-and-fallback; this is a router decision feeding existing machinery, and the boundary is a mechanical predicate with no threshold and no confidence score.

**REPAIR** a font iff **all** of:

- **R1.** No ToUnicode stream, *or* a ToUnicode stream that does not cover every code the document actually uses from this font; **and** `/Encoding` is not a predefined named encoding; **and** `/Differences` does not define the codes in question.
- **R2.** The BaseFont name after the `+` subset tag matches a **documented family table vendored into the repo with its source cited next to it**. Initial set: `CMSY*`, `CMBSY*`, `CMEX*`, `CMMI*` (Knuth/BlueSky OML/OMS/OMX), `MathDesign-*-Symbol/-Extension`, `Universal-GreekwithMathPi`, `MathematicalPi-One`, `MathematicalPi-Five`. Adding a family is a reviewed data change with a citation — **never** an `if name in AGL` fallback.
- **R3.** The observed `{code → name}` vector is a **subset** of that family's published encoding. One mismatch ⇒ the identification is wrong ⇒ no repair on that font. (Vector match, not name match. This is the whole epistemic load.)
- **R4.** Every mapped target is a Unicode scalar that is **non-PUA, non-surrogate, and not a combining mark emitted as a standalone composite**. Oldstyle figure names → ASCII `'0'`–`'9'`, never AGL's U+F73x. `^u(ni)?[0-9A-F]{4,6}$` → the literal codepoint (independent of R2 — the algorithm is self-certifying).
- **R5 (completeness, from gpt-sol).** The synthesized ToUnicode must cover **every code the document actually uses from this font**, not just codes 48–57. A partial map turns currently-extractable characters into replacements. If any used code fails R2–R4, the font does not fully repair.

**Partial resolution is allowed, guessing is not.** Codes that resolve, resolve. Codes that don't → **U+FFFD**, a visible hole, per "a wrong number is worse than a missing one" — never a Latin-1 digit, never an inferred neighbour of a known `H*` name.

**OCR is opt-in only** (`--ocr-untrusted-encodings`), never the residue default, and **never on pages classified as dense numeric grids** — that is the documented `qwen3-vl:30b-a3b-instruct` collapse zone, and it is exactly where the one true catastrophe paper lives. Refusal (`--fail-on-untrusted-encoding`) is opt-in too. Default terminal state for the residue is **flag at page granularity**, not document refusal.

**Two implementation constraints that are load-bearing, not nice-to-have:**

- **Handle rebind.** There are 32 `fitz.open` sites in `src/socr`, and native `get_text("words")` is consumed by at least 10 modules including `tables/native_verifier.py`, `tables/header_attribution.py`, and `benchmark/ground_truth.py`. Repair must materialize **one run-scoped patched copy** that every opener sees — never the source PDF (cloud-storage rule). A repaired page scored by TR-3 against the *corrupt* native words will hard-fail the good read. This is the round-1 trap and it is still the likeliest way to ship this broken.
- **Byte-neutrality.** Injection must be proven not to perturb extraction on unaffected pages — the golden/byte-identity assembly tests are the gate.

**This is why repair beats OCR structurally, independent of glyph accuracy:** repair fixes the *native words themselves*, so TR-3 keeps working against corrected ground truth. OCR forces a TR-3 bypass, discarding the corpus's only numeric verifier precisely on the pages that need it most.

## 6. Surfacing

- **Page status / sidecar `NNN.json`:** `native_encoding_repaired` (font, xref, family table + citation, full synthesized `{code → unicode}`) or `native_encoding_untrusted` (font, xref, unresolved `{code → name}`, reason code). Repaired is never silent either.
- **Document status / manifest:** union of untrusted pages; a document with any untrusted used code must **not** freeze as native SUCCESS — `PARTIAL_UNTRUSTED_ENCODING`.
- **CLI:** one line per affected font at run end; non-zero exit under `--fail-on-untrusted-encoding`.
- **TR-3:** untrusted native words are excluded from ground truth rather than compared against.

## 7. Where the three still disagree

**(a) Residue default: flag vs refuse.** grok and fable say flag-only at page granularity with refusal behind a flag; gpt-sol makes refusal the *terminal* fallback whenever OCR cannot be independently accepted. Matters because it decides whether the Romer paper's MathPi-Five appendix fails the whole document. **Settles with:** the actualization check from §4.6 — if the unresolved codes are only used in prose formulas, flag; if they are used inside a numeric table cell, gpt-sol's refusal is correct for that page. Measure before choosing.

**(b) Is the CM class actually broken in practice?** grok flags that MuPDF may already emit ∈/∞ for CMSY via its own fallbacks, in which case 46 of the 50 are scan noise and repair shrinks to Pi + `uXXXX` + oldstyle-ASCII. fable assumes they are broken. **Settles with:** one command — extract native text from three CMSY pages and check whether ∈/∞ or Latin-1 digits come out. Do this **before** vendoring four family tables; it could cut the work by 90%.

**(c) How large is the invisible CFF/TrueType population?** Nobody knows, because the scan never reported it (§2a). fable expects small; gpt-sol treats it as the reason the whole decision is provisional. Matters because a large CFF class with non-standard names *actually used at digit codes* would shift cost-benefit toward a verified-OCR path. **Settles with:** the fontTools rescan. This is the one open number that could reverse the verdict, and it should gate the ticket.

**(d) Whether AGL membership can ever license repair on its own.** gpt-sol allows AGL as one pinned source among several; grok forbids it outright outside a family table; fable's stated predicate allows it and thereby contradicts its own oldstyle handling. I sided with grok. **Settles with:** nothing empirical — it is a rule choice, and grok's is the only one that doesn't emit PUA into a citation corpus.

## 8. What would change this verdict

- **A CMSY page whose native extraction already yields ∈/∞** ⇒ 46 papers are scan noise; narrow repair to Pi + `uXXXX` + oldstyle-ASCII (three families, one paper each). *Cheapest check available — run it first.*
- **Any corpus font where the family-matching vector is not that family** (e.g. `/element` at slot 50 in a `CMSY10` that renders something else) ⇒ vector identification dies; demote to flag-only for that class and require glyph render-verify as a repair precondition.
- **A cited, complete Mathematical Pi encoding that mismatches the embedded PFA on a digit slot** ⇒ stop repairing that family entirely; the Romer paper is flag-closed permanently.
- **fontTools rescan surfacing a large CFF/TT class with non-published names actually used at digit codes** ⇒ residue grows from one paper to dozens; reopen the verified-OCR investment.
- **Measured digit-error rate of `qwen3-vl:30b-a3b-instruct` on the actual untrusted table pages, below the native forgery rate, with TR-3 not comparing against corrupt native words** ⇒ OCR can become the residue default. Agreement with the native layer does **not** count as validation. Until that measurement exists, OCR is the higher silent-error path.
- **Synthesized-ToUnicode injection perturbing extraction on unaffected pages** (golden/byte-identity tests fail) ⇒ repair is not shippable in its current form regardless of glyph correctness.

Facts above verified directly against `/tmp/gh217_corpus_font_audit.py` (lines 82–92 for the false-negative path, 69–70 for the Differences skip, 59–63 for the presence-only ToUnicode check) and `/tmp/gh217_corpus_font_audit.json`.
---

## Round 2 follow-up: does CMSY corrupt numbers?

**Answer: no. Not one digit, anywhere. The 46 CMSY/CMEX papers are scan noise for GH-217.**

This settles residual disagreement **(b)** in §7 — the check the synthesis called "one command…
could cut the work by 90%".

### What was run

Read-only, `~/venvs/socr/bin/python` + PyMuPDF. For every paper in
`/tmp/gh217_corpus_font_audit_v2.json` carrying a `SUSPECT_UNKNOWN_NAME` verdict on a `CMSY*` or
`CMEX*` font, open each flagged page, walk `get_text("rawdict")`, and read the characters actually
emitted by spans whose font is CMSY/CMEX. Source PDFs copied to a temp path before opening
(cloud-storage rule); nothing written back.

### Three named papers, as asked

| paper | fonts | flagged pages | emitted from those spans |
|---|---|---|---|
| `2014__greenwood_vayanos__bond_supply_and_excess_bond_returns__RFS.pdf` | `RAKSVM+CMSY10`, CMSY8, CMSY7 | 7, … | `−` (U+2212), `·` (U+00B7) — **0 digits of 21 spans** |
| `2017__ozdagli_weber__monetary_policy_through_production_networks__NBER.pdf` | `AAQGQG+CMSY10`, CMSY8, CMSY6, `XPGLOV+CMEX10` | 8, 9 | `−`, `′` (U+2032), `∼` (U+223C) — **0 digits of 61 spans** |
| `2020__bauer_rudebusch__interest_rates_under_falling_stars__AER.pdf` | `QGUPKN+CMSY10`, CMSY8, CMSY7, `AYTHUI+CMEX10` | 2, 3, 8 | `∗` (U+2217), `−`, `≡` (U+2261), `→∞` — **0 digits of 26 spans** |

### Corpus-wide, all 43 CMSY/CMEX papers

Extended the same probe to every flagged page in all 43 papers (0 unreadable):

```
TOTAL spans in CMSY/CMEX fonts : 1302
spans emitting an ASCII digit  :    0
```

Top characters actually emitted: `−` ×339, `′` ×131, `∗` ×60, `{`/`}` ×49 each, `∈` ×47,
`×` ×31, `∞` ×21, `·` ×20, `≡` ×15, `∼` ×14, `≥` ×14, `∀` ×13.

### Why the audit and the reader disagree

The audit reads the font's **built-in Type-1 encoding vector** and finds `/element` at code 50,
then fails to resolve the name against its own table. MuPDF **already resolves these names itself**
— it applies the standard glyph-name→Unicode path for TeX text/math encodings. So the audit is
measuring *its own resolver's* coverage gap, not the reader's. `SUSPECT_UNKNOWN_NAME` on a CMSY
font is a statement about the audit script, not about the shipped text.

This is exactly the epistemic distinction §3 drew, arriving from the other direction: the encoding
vector is a document-local fact, but **what socr actually ships** depends on what MuPDF does with
it — and for the Computer Modern families it does the right thing.

### What this does to the plan

- The repair route narrows from **four font families to one paper**: the Romer & Romer
  `Universal-GreekwithMathPi` / `MathematicalPi-One` case (present twice under different
  filenames). That is the only confirmed **digit** corruption in 407 papers.
- **Do not vendor CMSY/CMBSY/CMEX/MathDesign family tables.** They would be dead code guarding a
  defect that does not occur.
- Round 2's verdict (repair, not OCR) **survives and gets cheaper** — the residue it was worried
  about is mostly imaginary.
- The `SUSPECT_UNKNOWN_NAME` predicate as written is **not** a usable routing signal: it fired on
  43 papers where the shipped text is correct. Any gate must be measured on *emitted characters*,
  not on the audit's ability to resolve a glyph name.

### Three real defects this surfaced, none of them GH-217

Visible in the emitted-character histogram, worth separate issues rather than folding in here:

1. **PUA bracket pieces** — `U+F8EB`–`U+F8FB` ×~150 from CMEX extensible delimiters. Already
   covered by `count_pua_chars` (`born_digital.py:270`); confirms that detector earns its place.
2. **Control characters** — `U+0000` ×34, `U+0001` ×35, `U+0010`/`U+0011` ×16 each. Should be
   caught by `_garbage_ratio` (`born_digital.py:1625`); worth checking it actually fires at these
   densities.
3. **Big operators decoding as letters** — `X` ×31 and `P` ×23 from CMEX10, i.e. `∑` and `∏`
   shipping as ASCII letters. A *silent symbol* corruption in the same family as GH-217 but
   without the sign-inversion consequence. Not covered by anything today.

**Scope note:** this probe only inspected pages the audit flagged, and only spans in CMSY/CMEX
fonts. It establishes that those fonts do not emit digits; it does not re-audit the other verdicts
(`SUSPECT_NO_PROGRAM`, `SUSPECT_UNPARSED_PROGRAM`, `SUSPECT_NO_DIGIT_VECTOR`).
