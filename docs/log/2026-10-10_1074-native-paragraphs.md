# #1074 — native paragraphs and split words: design

Worktree `fix/issue-1074` @ a257fd1d. Analysis plus an out-of-repo prototype (`socr.__file__` asserted). No code committed.

## 1. How native prose reaches `pages/NNN.md` and `<stem>.md`

Source, `src/socr/core/born_digital.py`: `raw_text = page.get_text("text")` + `clean_native_text`
(3220-3221). No-table page: `native_text = raw_text.strip()` (3737; rotated refusal 3678). These
pages carry no markdown links; links exist only on `has_tables` pages. Table page: `extract_structured`
(4116) → flat text + `_apply_links_to_flat_text` (4157, 4275, 4276) or
`interleave_table_regions_into_page` (4289): one line per dict line (4484), regions as `\n{md}\n`
(4379, 4489), joined by `"\n"` (4496). `apply_born_digital` copies to `ps.native_text` (state.py:724-725).

Lanes, `src/socr/pipeline/orchestrator.py`, all built from `ps.native_text`: native (11005);
chart_asset `native_prose + "\n\n" + png` (10720); native+math `splice_math` (3072) + markers (3092,
10560); equation lane `attach_equation_sidecars_in_place` (3764); cell repair rewrites
`ps.native_text` and re-emits native (11414); equation-sidecar fallback embeds region text (18537-18540).

Emit: every writer goes through `manifest._select_and_finalize_page` (manifest.py:4394): in-loop
flush (orchestrator.py:10467-10479), assemble records + fragment flush + stitch check (14952-14953,
15153-15154), `_rewrite_all_fragments` (14677), manifest blobs (manifest.py:4774), hence `replay`
(4893). Resume and the saved-body pass re-finalize already-shipped text.

## 2. Why native+math pages have blank lines

No paragraph code exists in `src` (no paragraph builder, reflow or dehyphenation). The math lane
appends evidence after `\n\n`: no-region marker (orchestrator.py:3092), unaligned-region suffix
(math/recover.py:464-466), chart PNG (10560); aligned regions splice inline with one `\n`
(recover.py:442, 448). Probe (60 random PDFs, 198 corrupt-math pages, model off): 118 got their only
blank line from the lane; 54 all-aligned pages got none, so the corpus's "0" is not fully reproduced
locally (the lane's selection is narrower than `has_corrupt_math`). The prose stays one printed line
per line. Nothing to reuse.

## 3. Consumers of native line structure

| Consumer | Depends on | (A) source reflow | (B) emit reflow |
|---|---|---|---|
| `splice_math` (recover.py:437) | exact `"\n".join(lines)` slice (110) | **breaks**: unaligned, evidence dumped at page end | safe |
| `attach_equation_sidecars_in_place` (equation_latex.py:421; slice detect_equations.py:413) | same | **breaks**: readings dropped | safe |
| `_word_char_spans` links (born_digital.py:1808) | token find | dehyphenated tokens unfindable → links skipped | safe |
| `_detect_flattened_table` (3934), `_detect_columnar_numbers` (3972) | `raw_text` / page lines | safe if `raw_text` untouched | safe |
| `_line_is_in_region_text` (1634) | dict lines in extraction | must run before joins | safe |
| `fence_spelled_runs` (manifest.py:4257) | runs of 1-char lines | **breaks** | breaks if hooked before the guard (prototype: `test_gh1030` failed); safe after |
| chart anchors (chart_regions.py:157, 639) | anchor row inside a line | not measured: anchor ending in a removed hyphen → unresolved placement (surfaced) | same |
| `native_text_value_counts` (escalation_canary.py:264) | tokens | alphabetic tokens change | safe |
| `detect_native_structure_loss` (2455) | — | no production caller | — |
| judges / table ladder | in-loop `best_output.text` | see reflowed | see unreflowed; shipped differs by whitespace + listed hyphens |
| #713 `finalized_sha256` (manifest.py:4434-4452; checked 13890) | digest of finalized text | — | hook must precede the stamp |
| resume, replay, saved-body re-finalize | finalize re-run on shipped text | — | reflow must be idempotent |
| stitch == final (15154) | all writers share bytes | holds | holds (one seam) |

## 4. Where: (B), the emit seam

`reflow_native_prose(text, paragraphs, vocabulary)` in `_select_and_finalize_page`, after
`_apply_scanned_figure_guard` (4432), before the #713 stamp (4434); only for
`engine.startswith(_NATIVE_TEXT_LANES)` and never on `is_page_failed_marker(text)`.
- `paragraphs`: per page, lists of line strings from `get_text("dict", flags=TEXTFLAGS_TEXT)` in
  `_assess_page_signals`. Those flags make dict lines equal `native_text` lines on 174 of 177 pages
  tested (173 with default flags; the 3 misses are NBER math lines with control bytes and stay as
  printed). On `PageState`, not persisted: `_phase_analyze` (call at 1886) runs on every run, resume included.
- `vocabulary`: document-wide, from the analyze-time native text, never from later-mutated `ps.native_text`.
- Match whole lines, exactly, with a forward cursor (the `splice_math` / `_word_char_spans` idiom). A
  paragraph not found contiguously stays as printed. Skip fences, HTML comments and table rows
  (reuse the `chart_regions` literal scanners and `find_table_blocks`). Insert a blank line only
  between two adjacent matched paragraphs; spliced or foreign lines keep their adjacency.
- Changes allowed: `\n`→` `, `\n`→`\n\n`, evidence-backed `-\n`→``. Idempotent: joined lines no
  longer match a multi-line paragraph.

Prototype (B) on Ayivodji, `--native-only`, both runs provider-less (with ollama live, 2 pages
routed to qwen, so compare like with like): 64/69 pages changed; non-space character multiset
identical on every page except **158 removed hyphens**; pages with a blank line 37 → 67;
fragments == saved `.md` 69/69; `replay` == saved 69/69; `--reprocess` resumed 61 terminal pages
with identical bytes.

## 5. Paragraph rule: a block is not a reliable paragraph

Rendered and checked: Ayivodji p5-8 (SSRN), Ozdagli-Weber p3 (NBER), Nakamura-Steinsson p2 (QJE), Koval p2 (ACL).
- Ayivodji: block = paragraph; headings are their own blocks. The p7→p8 paragraph split is cross-page (out of scope).
- NBER: every printed line is its own block (20.2pt pitch). Block rule: 782 mid-sentence breaks over 59 pages.
- QJE, ACL: a whole column is one block holding 4-6 indent-only paragraphs (QJE p2: 42 lines, 5 paragraphs).
- Corpus sample (60 random PDFs, first 12 text pages, 667 pages): 289 pages (43%) have median block =
  1 line; 244 (37%) hold ≥1 indented paragraph start inside a block (591 boundaries).

Hybrid (prototype): start from blocks; split before a line indented more than the page's median
word-space past the column's modal left edge when the previous line ends short of the modal right
edge; merge a block into the previous paragraph when its first baseline is one modal same-size pitch
below and starts at the column's left edge. Correct on all 5 inspected pages. Text proxies, block →
hybrid: NBER mid-sentence breaks 782 → 184 (rest are Σ/subscript math and tables), QJE likely merges
88 → 38, ACL 25 → 5, Ayivodji about the same. Column-break continuations stay split. No new constant:
baselines are exact to ±0.02pt, the smallest paragraph-gap increment seen is 5.1pt, the median
word-space is 2.7-4.2pt, so "± one word-space" separates them.

## 6. Hyphen rule (issue's buckets; vocabulary excludes the split occurrences)

| PDF | splits | rejoin | keep `-` | both | printed |
|---|---|---|---|---|---|
| Ayivodji | 206 | 169 | 18 | 1 (`non-linear`) | 18 |
| ACL (9 pp) | 175 | 94 | 2 | 0 | 79 |
| QJE | 237 | 198 | 7 | 0 | 32 |
| NBER | 10 | 1 | 4 | 0 | 5 |

Rejoin: `infla-tion`, `Pi-azzesi`, `over-all`. Keep: `non-experts`, `high-frequency`, `zero-coupon`.
Printed mixes syllable breaks (`de-spite`, `pur-poses`, `excel-lent`) with compounds
(`policy-induced`, `balance-sheet`, `hand-side`). A "both halves are words" test only moves items to
"keep", which prints identically; separating the rest needs a dictionary, so keep the issue's rule.
Ties keep the hyphen. Proposed, not prototyped: a hyphen or dash glued to its word before a
non-lowercase start (`Long-`/`Run`, `Cobb-`/`Douglas`, `1994–`/`2000`, `article—`/`is`) joins with
no space and removes nothing.

`TEXT_DEHYPHENATE`: no effect in `text`/`words`/`dict` in PyMuPDF 1.27.2; in `xhtml` it joins
unconditionally (`non-\nexperts` → `nonexperts`, synthetic page). Unusable. `pymupdf_layout` (the
advisory `find_tables` prints) adds a learned layout model to extraction: rejected.

## 7. Blast radius

Full suite on the prototype copy, final variant: **0 failures** (7,099 repo tests + canary passed).
Two earlier variants each broke one test, now design constraints: hook before the guards →
`test_gh1030` e2e; blank lines around foreign lines → the math lane's byte-exact additivity test
(`test_agentic_corrupt_math_default_flip_no_provider_is_additive_only`). Zero failures also means no
existing test covers the behaviour; the guard below is the coverage.

## 8. Smallest first PR and its guard

PR: `src/socr/core/native_paragraphs.py` (segmenter, vocabulary, `reflow_native_prose`);
`PageAssessment`/`PageState.native_paragraphs`; `DocumentState.native_vocabulary`; one call in
`_select_and_finalize_page`. No flag: the off-run in the test monkeypatches the call to identity.

Guard, two-run difference in one process, parametrised over ladder empty / `[PROFILE_QWEN_LOCAL]`,
`_resolve_judge_model` → `""`. Synthetic born-digital page: two indented paragraphs, a heading, a
split word whose joined form appears elsewhere, a compound whose hyphenated form appears elsewhere, a
split with neither, a table, a fenced spelled run. Assert reflow-on vs off: non-space multiset differs
by exactly the listed hyphens; paragraph count; table, fence and marker bytes identical; stitch ==
final in both; finalize(on) is a no-op. Mutants (copy outside the repo, `socr.__file__` asserted):
delete the call → fails; drop every line-end hyphen (MuPDF behaviour) → fails on the compound; move
the call before `_apply_scanned_figure_guard` → `test_gh1030` e2e fails (seen in the prototype).

## Consilium question

"socr will reflow native PDF prose at emit time (after all guards, native-lane text only,
content-keyed so tables/LaTeX/fences are untouched) and needs a deterministic, model-free
paragraph-boundary rule from PyMuPDF geometry. Measured on 667 sampled corpus pages: 43% put every
printed line in its own text block (double-spaced working papers) and 37% keep indent-only paragraphs
inside one block (journal styles); baselines are exact to ±0.02pt, paragraph gaps add ≥5pt, median
word-space is 2.7-4.2pt. Which rule ships?
(A) a PyMuPDF text block is a paragraph;
(B) hybrid: start from blocks, split before a line indented more than one word-space past the
column's modal left edge when the previous line ends short of the modal right edge, and merge a block
into the previous paragraph when its first baseline is one modal same-size pitch (± one word-space)
below and starts at the column's left edge;
(C) gap-only: ignore blocks; break only where the baseline gap exceeds the modal same-size pitch by
more than one word-space (indent-only paragraphs stay merged, as today's rendering already does)."

## Acceptance-criteria readiness

- Content multiset: implementable; state it as a two-run difference against reflow-off, same provider state.
- Byte identity: holds by construction; add idempotence (re-finalize is a no-op).
- One function, every lane: implementable; scope `_NATIVE_TEXT_LANES`, never markers.
- "A paragraph is a PDF text block": replace with the panel's rule (fails on 43% / 37% of sampled pages).
- Hyphen rule: add the tie rule and the hyphen/dash no-space join; define "elsewhere" as the
  document's analyze-time native text minus the split occurrences.
- Out of scope, add: paragraphs continuing across a column or page break stay split.

## Panel verdict (2026-10-10, Fable + Astra)

Both seats pick (B), amended. As written, (B) fails two shapes, measured by Fable on 60 corpus PDFs
(678 pages, 37,183 MuPDF lines). Raw blocks and a single page-wide right edge do not survive real
layouts. Proxy counts below are FS = false split, FM = false merge, SZ = join across a ≥2pt size
change.

| rule | FS | FM | SZ |
|---|---|---|---|
| A | 3923 | 574 | 521 |
| B as written | 1327 | 0 | 119 |
| C | 1262 | 968 | 0 |
| **B'** | **366** | **47** | **389** |

**B' ships.** It adds no constant beyond the word-space, the pitch and the font size.

1. **Printed line first.** MuPDF lines in one column whose baselines differ by less than the smaller
   font size are one printed line. Fragments join with a space and are never a boundary. Without
   this step, 659 of B's 1,327 false splits are same-baseline fragments: wide gaps and
   super/subscripts.
2. **Boundary before b, with previous line a.**
   - Same block: a boundary iff `a.x1 < R_env(a) − ws` and `b.x0 > a.x0 + ws`. Here `R_env(a)` is
     the modal x1 of the lines sharing a's left edge (within ws), or the column edge when a's left
     edge is unshared. Measuring against the local environment, not the column, stops an abstract or
     block quote splitting on every line (Sutskever p1: 13 paragraphs for one abstract under B).
   - Different block: the same test, OR a gap not within ws of the modal pitch of a's size class.
     Size is rounded to 2pt for jittered OCR layers; the "±0.02pt baselines" premise holds only for
     born-digital pages.
3. **Astra's safeguards.**
   - Estimate pitch and edges per column and per size class.
   - Never merge across protected content (math, tables, fences, markers) or across a font/style
     change.
   - Insufficient or tied evidence keeps the printed boundary.
   - Replay reads finalized text and needs no geometry.

**Residual failures, accepted.** All of these are visible and none loses content:

- display-equation rows split;
- a hanging-indent reference list with zero item spacing merges (FM 47);
- an equation row followed by prose at exactly one pitch merges;
- a paragraph continuing across a column or page break stays split.

A dedent split would fix the references, but it adds 52 false splits on ragged or jittered pages, so
it is left out.

**Hyphens: keep the issue's rule, plus the tie rule and the no-space join. No dictionary.**

Of 6,927 line-end splits, 80% are rejoined on a witness, 6% keep the hyphen, and 13.6% stay as
printed (`de-spite`). No other witness is safe:

- Code points: 7,214 of 7,218 line-end hyphens are U+002D.
- "Both halves absent from the document": wrong on 11 of 533, joining into non-words such as
  `crosssectional` and `peerreviewed`.
- A library-wide vocabulary makes one document's bytes depend on the other PDFs, which breaks
  replay.
- `/usr/share/dict` is not on CI.

A vendored static wordlist that confirms the joined form only would be a separate dependency
decision.

**Smallest guard.** One synthetic born-digital page with four parts:

- (a) two indent-only paragraphs in one block;
- (b) the same two with pitch above 1.5× the size, so MuPDF emits one block per line;
- (c) a three-line, full-width, narrow environment;
- (d) one printed line written as two same-baseline fragments.

Assert 2 + 2 + 1 paragraphs, with (d) inside its paragraph, the non-space multiset preserved, and a
re-run that changes nothing. Each mutant must fail:

- drop the fragment merge: (d) breaks;
- use column edges instead of `R_env`: (c) becomes 3;
- drop the pitch merge: (b) breaks;
- drop the indent split: (a) breaks.

Fable's scratch scripts are in the session scratchpad `panel1074/`.
