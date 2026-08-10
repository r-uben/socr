# Benchmark specification — the cost/quality claim (GH-39)

**Status:** specification only. No results. Nothing here has been run.

## 1. The claim under test

socr's agentic mode routes each page up a cost-ordered ladder, cheapest first,
escalating only when a judge rejects. The claim that justifies that design:

> Cheapest-first with judge-gated escalation reaches approximately the quality of
> always using the strongest engine, at approximately the cost of using the
> cheapest one.

Nothing in the repository measures this. Until it is measured, the ladder is a
plausible design, not a validated one, and #39 stays open.

### What a result must look like

| condition | cell exactness | WER / CER | cost (USD) | wall-clock |
|---|---|---|---|---|
| native-only (no OCR) | | 0 | 0 | |
| local-only (`--strict-local`) | | | 0 | |
| always-strongest (top rung every page) | | | | |
| **agentic (the shipped default)** | | | | |

The claim survives if agentic lands within a stated tolerance of
always-strongest on quality while landing near local-only on cost. If it does
not, the honest outcome is to change the claim, not the benchmark.

## 2. What already exists — reuse, do not rebuild

| module | role |
|---|---|
| `benchmark/dataset.py` | corpus definition + serialization |
| `benchmark/ground_truth.py` | per-page native text as GT |
| `benchmark/rasterize.py` | born-digital → synthetic image-only PDF |
| `benchmark/page_types.py` | the #39 page taxonomy |
| `benchmark/runner.py` | execute engines, collect runs |
| `benchmark/scorer.py` | WER, CER, NES, table-cell fidelity |
| `benchmark/table_exactness.py` | hierarchy-aware cell exactness (GH-96) |
| `benchmark/calibrate.py` | derive routing from measured results |

`rasterize.py` + `ground_truth.py` is the key mechanism: take a born-digital PDF,
keep its text layer as truth, rasterize it into an image-only PDF, and run OCR on
the raster. This yields paired (image, truth) data at corpus scale with no human
labelling.

## 3. Blocking defect: the ground truth is contaminated

`GroundTruthExtractor.extract` (`benchmark/ground_truth.py:41`) takes
`page.get_text("text")` verbatim, with no filtering.

GH-136 established that this text layer can be **wrong**: a broken font map
yields `(1997)` as `(/997)` and `155-84` as `/55-84`. Those pages are not rare
enough to ignore — they are the motivating case for an entire escalation tier.

The consequence is severe and it runs the wrong way:

- GT for such a page contains `(/997)`.
- An OCR engine that reads the raster correctly emits `1997`.
- The scorer counts that as an **error**.

So the benchmark penalises correct OCR precisely on the pages where OCR is most
valuable, and rewards engines that reproduce the corruption. Cheapest-first — the
lane that trusts native text — is flattered. The measured conclusion would be
biased toward the design under test.

**This must be fixed before any run.** Options, in preference order:

1. **Exclude contaminated pages from GT.** Reuse `count_digit_corruption` and the
   encoding-corruption ratio to drop any page whose text layer trips either gate.
   Cheap, uses shipped code, and is defensible: pages with a known-broken text
   layer cannot serve as truth.
2. **Hand-verify a subset.** Higher quality, does not scale, appropriate for a
   small table-heavy stratum where exactness matters most.
3. Do nothing and report scores. Not acceptable — the number would be wrong in a
   direction that favours us.

Whichever is chosen, the count of excluded pages is a reported figure, not a
footnote.

## 4. Corpus

- **Frozen.** A fixed manifest of paper identifiers + checksums, committed. A
  benchmark whose corpus drifts cannot compare runs across time.
- **Stratified by `page_types.py`**, and reported per stratum, not only in
  aggregate. A single mean hides the case that matters: dense financial tables.
  Prose pages will dominate any unweighted average and drown the signal.
- **Table-heavy pages are the stratum of record** for this corpus. Report them
  separately and prominently.

## 5. Denominators — the rule that decides honesty

**Every page attempted is in the denominator.** Failed pages, flagged pages,
pages that hit the cascade halt, pages that shipped a failure marker.

Excluding non-SUCCESS pages would let a condition score well by *failing more* —
routing its hard pages into exclusion and averaging over what remains. Since the
whole point of the fail-closed design is that hard pages surface rather than
disappear, a metric that drops them measures the opposite of what socr claims.

Report alongside every score: pages attempted, pages SUCCESS, pages failed, pages
flagged.

## 6. Provenance recorded per run

Without these a result is not reproducible and cannot be compared:

- resolved **provider** identity per page (id, model, backend) — already in the
  manifest journal
- resolved **judge** identity, including the literal `heuristic` (GH-133)
- run fingerprint
- `normalizer_version` / `assembly_version`
- corpus manifest checksum
- socr commit SHA

The judge identity matters especially: before GH-133 a run could be
heuristic-gated while reporting a VLM. Any benchmark from before that commit is
uninterpretable and must not be cited.

## 7. Repetition and nondeterminism

VLM output is not deterministic even at `temperature=0`. Single-shot numbers will
mislead.

- **≥3 repetitions** per condition on a fixed subset.
- Report **spread, not just mean** — a 2-point gap between conditions is noise if
  within-condition spread is 5 points.
- Judge disagreement across repetitions is itself a finding: a judge that accepts
  a page on run 1 and rejects it on run 2 makes the routing nondeterministic and
  the cost figure a distribution rather than a number.

## 8. Threats to validity — state these in any write-up

- **Synthetic rasters are not real scans.** No scanner noise, skew, JPEG
  artifacts, or bleed-through. Results are an upper bound on scanned-page
  performance. If real scans are available, run a smaller real-scan stratum.
- **GT is machine-extracted**, so it inherits PyMuPDF's own failure modes —
  flattened subscripts, dropped Greek, broken reading order around equations
  (see the math-font gap). Equation-heavy pages should be reported separately or
  excluded, and which was done must be stated.
- **Cost figures are estimates.** `providers.py` self-describes its prices as
  "rough estimates… to be tuned, not trusted". Cloud rungs priced at `$0.00`
  (subscription-billed) mean the cost column measures *marginal* cost, not total
  cost of ownership. Say so.
- **Ollama-cloud egress** is billed by subscription, so a "free" cloud rung is
  free only in the sense the ladder means it.

## 9. Acceptance criteria

Before this closes #39, the following must exist:

1. GT contamination fixed, with the exclusion count reported.
2. All four conditions run on the frozen corpus, ≥3 repetitions.
3. Per-stratum results with spread, all pages in the denominator.
4. Full provenance for every run.
5. A written verdict that names the tolerance and states plainly whether the
   claim holds — including the case where it does not.

## 10. Out of scope

- Calibrating the ladder order (`calibrate.py` consumes these results; it is the
  next step, not this one).
- Judge false-positive/false-negative rates. Related, separately measurable via
  the existing `judge-benchmark` command, and deserving its own spec.
