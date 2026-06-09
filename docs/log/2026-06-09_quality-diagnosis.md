# 2026-06-09 — Full quality diagnosis: why output content is poor

**Method:** 69-agent diagnostic sweep (6 area readers + adversarial verifier per major
claim + completeness critic + 3 gap readers). 84 findings raised, **82 confirmed, 2
refuted** (20 critical). Raw findings JSON: `2026-06-09_quality-diagnosis-findings.json`
(same dir, untracked). Every claim below was independently re-verified against code or
on-disk artifacts by a second agent instructed to refute it.

**Headline:** the ceiling is high — `test_run/test4_cloud` (Kuttner, full cloud) and the
escalated pages of `test_escalation2` are excellent (real LaTeX, the 42-row Table 2
perfectly structured). Typical output is far below that ceiling, and the gap is mostly
**self-inflicted**: the pipeline destroys good engine output post-hoc, ships failure
states silently, and routes against its own measurements.

---

## Root cause 1 — The normalizer destroys correct engine output (critical)

- `OutputNormalizer._clean_deepseek_glm` (normalizer.py:60,108-131) and
  `DeepSeekVLLMEngine._clean_ocr_output` strip ALL HTML tags with `<[^>]+>`.
  DeepSeek-OCR emits tables as HTML, so every table becomes a fused wordstream and
  **adjacent cell digits concatenate into fabricated numbers** ('4.4','79.1' → '4.479.1')
  in a research corpus. The papers library (DeepSeek-via-vLLM primary) went through
  exactly this path.
- The regex matches newlines and non-tags: it deletes inequality spans, `<EOS>` (visible
  in the shipped Sutskever output), anything between a stray `<` and the next `>`.
- Blanket NFKC corrupts math on every engine path, while native text — the one path that
  needs ligature fixing — bypasses `normalize()` entirely.
- Once flattened, the page has zero pipe lines → `find_tables` finds nothing → dual-pass
  can never patch it. The destruction is unrecoverable downstream.

## Root cause 2 — Failure paths ship garbage or nothing, then report SUCCESS (critical)

- **VLM-judge rejection is a dead end on every path.** Born-digital pages:
  `needs_repair` returns False once any attempt exists → silent flat-native fallback,
  document assembled as SUCCESS. Non-born-digital pages: repair router picks
  `EngineType.AUTO` (first member of default `enabled_engines`) → `get_engine(AUTO)`
  raises uncaught ValueError → run dies after cloud money was spent (reproduced).
- **Empty pages ship silently.** `_winning_page_output` never consults `p.attempts`
  despite its docstring; judge/audit-cleared pages assemble as "" and the CLI exits 0
  ("Completed with warnings"). Observed: flagship Kuttner run is missing page 10 — the
  paper's central Table 2 — invisibly.
- **Born-digital native text is exempt from every gate** and is the silent fallback for
  exactly the pages where it is known-lossy (tables/equations). Shipped consequences:
  Symbol-font mojibake ('=' → '¼', '+' → 'þ', minus → '@'), one-token-per-line table
  streams, ﬁ-ligature glyphs breaking search.
- **The figure phase can destroy a finished run**: PNGs → 50-min paid describe loop →
  only then write .md/metadata. Any exception loses the completed OCR text with zero
  record. Two real output dirs (bernanke_kuttner, test_paper) exhibit exactly this.
- DeepSeek-vLLM: `max_tokens=4096`, no `finish_reason` check — truncation returned as
  SUCCESS with hardcoded confidence 0.85. HPC "cloud fallback" is a stub returning `{}`.
- Raw deepseek-ocr timeout exceptions (full command line + prompt) spliced into the
  deliverable markdown.

## Root cause 3 — Routing never sees quality or price (the "ladder bug", critical)

- Three disjoint hardcoded ladders (`AUTO_ENGINE_ORDER`, `_LOCAL_ENGINE_ORDER` +
  `ENGINE_PRIORITY`, `RepairRouter` lists) that contradict each other AND the project's
  own benchmark numbers written as comments in the same files:
  `AUTO_ENGINE_ORDER` omits Qwen (best open, 0.47-0.58) entirely and ranks DeepSeek
  (0.09, near dead last) above GLM (0.37). Keyless/offline runs route whole documents to
  the worst engine. The repair router classifies DeepSeek as "capable" and "cloud".
- The agentic cost ladder is correct in spirit (cheapest-first) but capped at
  `max_retries+1 = 3` attempts — with 3+ local engines installed, **the paid rungs are
  unreachable**, so "escalate to cloud when needed" never happens.
- Mistral is the premium rung above Gemini despite being worse on the bench and ~5×
  pricier.
- **The internal benchmark scores every engine a perfect 0.0 WER** — CLI engines return
  one PageOutput with `page_num=0`, ground truth is 1-indexed `page_1.txt`…, the scoring
  loop matches nothing and defaults to 0.0. Calibration is a stable sort over all-zero
  ties. No benchmark artifact exists on disk; nothing in production imports
  `socr.benchmark`. Quality-to-price is computed nowhere.

## Root cause 4 — Gates measure the wrong things (major)

- Heuristics PASS: 12% mojibake, digit-varied hallucination loops, shredded flat tables,
  math-dense junk. Heuristics FAIL: a legitimate 24-word figure-caption page
  (min_word_count=50, severity=error) → deterministic paid escalation of good pages.
- Judge coverage is inverted: native and scanned pages — the highest-risk content — are
  exempt; the judge default model (`qwen2-vl:7b`) isn't in the maintained candidate
  list, so the agentic judge silently degrades to heuristics.
- Embedded-text-layer scans (Morris-Shin: 1990s JSTOR OCR layer) are treated as native
  → garbage math ships with no escalation; `test1_fixed` is byte-identical to
  `test1_auto`.

## Root cause 5 — The table/figure machinery is inert on default runs (critical)

- On a default `socr process`: dual-pass is flag-only (`--auto-patch-tables` off),
  native structure-restore never fires on table pages (has_tables forces VLM routing
  away from it — TICKET-20), and without Ollama both Phase 3b and 4c silently no-op.
  **No table machinery can modify output on a default run.**
- TICKET-19's numeric gate rejects unicode minus, en-dash, starred values, `coef(SE)`
  tokens — reproducing the exploded Kuttner Table 2 verbatim.
- Figures: descriptions spliced mid-sentence (page-boundary append), extracted PNGs
  never linked via `![...]()`, false-positive crops (mastheads, badges) each burn a paid
  Gemini call that bypasses cost accounting.

## Refuted (for the record)

- "Corrupt-math recovery is selective (eqs 1-5 unfixed in escalation2)" — quotes check
  out but the load-bearing interpretation was wrong.
- "Benchmark ground truth provably corrupt on math papers" — circularity is real, the
  "provably corrupt" claim did not survive verification.

---

## Ranked fix plan

**P0 — stop destroying content (small surgical diffs, transforms typical output):**
1. HTML-table→markdown conversion BEFORE tag stripping (port deepseek-ocr-cli's
   `_html_table_to_markdown`); tag-shaped, non-newline-spanning regex with `<EOS>`-style
   stoplist; cell-boundary separators so digits can never fuse. Both normalizer.py and
   deepseek_vllm.py. Regression test: numeric adjacency never concatenates.
2. `_winning_page_output`: fall back to best surviving attempt (flagged
   audit_passed=False/WARNING); never ship "" when attempts hold text; explicit
   `[PAGE N FAILED]` marker + non-zero exit when truly empty.
3. RepairRouter: filter chain to runnable engines (drop AUTO/VLLM sentinels); add an
   AUDIT_FAILED arm that switches engine family; try/except around `get_engine` in
   Phase 4.
4. Make judge rejection actually trigger repair on born-digital pages (explicit flag
   `needs_repair` honors, with attempt cap).
5. Write .md + metadata BEFORE the figure phase; metadata write in `finally`.
6. Mojibake/cmap sanity check on native text (Symbol-font chars between digits,
   ligature codepoints) → corrupt native pages escalate to VLM; scoped ligature
   normalization on the native path only.

**P1 — route by measured quality-to-price:**
7. Fix benchmark page-index off-by-one; score document-level; add a table-structure
   metric; re-run on the 10-paper set; persist results artifact.
8. Single ladder derived from ENGINE_PRIORITY + key availability (Qwen into auto order,
   DeepSeek demoted below GLM, Gemini before Mistral); kill the 3-attempt cap so the
   cost ladder can actually reach paid rungs; consistency test asserting ladders match
   the measured ranking.
9. Gate fixes: min_word_count sparse-page false positives; per-page truncation +
   `finish_reason` checks; judge samples scanned + native-table pages; embedded-OCR-
   layer scan detection (image-dominant pages with text → scanned routing).

**P2 — make tables/figures fire:**
10. TICKET-20: gate `needs_ocr_enhancement` on evidence of native deficiency
    (corruption detectors already exist), not mere has_tables/has_equations.
11. TICKET-19 token regex: unicode minus, stars, merged coef(SE), mojibake tolerance.
12. Figures: caption-anchored `![Figure N](figures/...)` links at paragraph boundaries;
    dedup/false-positive filter before paid describe calls; figures into cost
    accounting.

**Deliberately NOT in plan:** swapping models/engines wholesale, going full-agentic
(re-affirmed against, 2026-05-29), dual-pass on scans (deferred, documented).
