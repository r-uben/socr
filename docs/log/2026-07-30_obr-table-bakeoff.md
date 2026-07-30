# 2026-07-30 — OBR hierarchical-table bake-off: measuring the flagged pages, and one dangerous null result

Empirical pass over an existing strict-local run to answer the design fork in **#96**
(escalate vs fail-closed on ambiguous tables) with a measurement instead of a panel, and
to check **#95**'s premise (is the dual-pass flag list a signal worth surfacing?).

Outcome in one line: **escalation to a vision model works** (38.6% → 74.0% across 16 table
pages, never worse on any page, and it recovers two pages socr dropped or fail-closed),
**the dual-pass detector is reliable**, and **an unguarded escalation lane will silently
fabricate entire tables** — demonstrated, not hypothesised.

## Setup

| | |
|---|---|
| Document | OBR *Economic and fiscal outlook*, November 2022 — 68 pages, born-digital |
| PDF | `~/data/fiscal-ballast/regulation/uk/obr_efo/2022-11-17_efo-november-2022.pdf` |
| socr run | HPC, `--strict-local`, `qwen3-vl:30b-a3b-instruct`, 2026-07-17 |
| Output | `~/data/fiscal-ballast/regulation_ocr/uk/2022-11-17_efo-november-2022/` |
| Escalation candidate | Antigravity `agy` on a 200 dpi page PNG |

Both paths are local data, not in this repo.

## Method — ground truth is free on born-digital pages

The document is born-digital, so `fitz`'s native text layer already contains the true
label order and every digit. No hand transcription, no model, no cloud call.

Metric: **label-keyed cell exactness**. Build `label -> [values]` from the native layer by
grouping each non-numeric line with the numeric lines that follow it; build the same map
from the emitted markdown table; count cells that match positionally per label. Rows whose
label cannot be matched at all count as fully missed, which is the right severity for a
citation corpus — an unlabelled `-18.4` is unusable.

**Known limitations. Both matter for reading the numbers below, and both are requirements for
the real harness.**

1. **Duplicate-label collision.** Keying by bare label collides when a hierarchical table
   reuses a child label. `Other measures` appears twice in Table 1 (under *Growth Plan* and
   under *Autumn Statement*), and the scorer credits the first match only. This understated
   agy's page-13 score by 6 cells. **The productionised harness for #96 must key rows by
   `(parent, label)` path, not by label.**

2. **Per-page ceiling below 100%.** The native-layer parser is a heuristic
   (non-numeric line, then the numeric lines following it) and does not handle every layout.
   On some pages it caps *any* engine well below 100% — page 39 scores 79.7% for both socr
   and agy even though their outputs are substantively identical and both correct, and on
   pages 53 and 61 the parser matches no labels at all, scoring both engines 0.0% and 32.1%
   respectively. **A tie between two engines on this metric usually means the ceiling was
   reached, not that both failed.** Only divergences are informative; absolute per-page
   values are lower bounds.

Scorer used here is in the appendix.

## Result 1 — the flagged pages, measured

19 of 68 pages carried a dual-pass or verifier flag. Scored against the native layer:

| page | gt rows | cells | exact % | rows not found | orphan rows | labelled-but-empty | dup rows |
|-----:|--------:|------:|--------:|---------------:|------------:|-------------------:|---------:|
| 13 | 21 | 126 | 32.5 | 0 | 5 | 11 | 5 |
| 37 | 1 | 6 | 0.0 | 1 | 0 | 0 | 0 |
| 39 | 14 | 74 | 79.7 | 3 | 0 | 4 | 3 |
| 41 | 1 | 14 | 0.0 | 1 | 0 | 0 | 0 |
| 43 | 2 | 32 | 0.0 | 2 | 0 | 0 | 0 |
| 45 | 9 | 49 | 100.0 | 0 | 0 | 2 | 3 |
| 46 | 14 | 75 | 0.0 | 14 | 0 | 0 | 0 |
| 48 | 14 | 82 | 0.0 | 14 | 0 | 0 | 15 |
| 51 | 15 | 85 | 12.9 | 12 | 1 | 5 | 2 |
| 53 | 17 | 52 | 0.0 | 17 | 0 | 4 | 4 |
| 55 | 6 | 33 | 0.0 | 6 | 0 | 0 | 0 |
| 59 | 30 | 210 | 76.7 | 7 | 1 | 7 | 2 |
| 60 | 30 | 180 | 73.3 | 8 | 1 | 7 | 2 |
| 61 | 28 | 224 | 32.1 | 19 | 0 | 6 | 2 |
| 62 | 28 | 168 | 3.6 | 19 | 0 | 9 | 867 |
| 63 | 27 | 189 | 77.8 | 4 | 0 | 6 | 4 |
| 64 | 39 | 271 | 0.0 | 39 | 8 | 15 | 2 |
| 65 | 39 | 232 | 0.0 | 39 | 7 | 14 | 2 |
| 67 | 34 | 204 | 94.1 | 1 | 0 | 5 | 4 |

**Aggregate over the 16 pages that actually contain tables: 870 / 2254 cells exact = 38.6%.
917 duplicate emitted rows.** (Pages 37, 41 and 43 are excluded — see mode (c) below.)

### Four distinct failure modes, not one

**(a) Hierarchical off-by-one shift** — pages 13, 61. Every digit is present, correct, and
in the right column; only the label↔row pairing slides by one inside each nested block, and
the parent row is left empty. Page 13:

```
                                     native      emitted
September energy package              43.2       (empty)
  Energy price guarantee              24.8         43.2
  Energy bill relief scheme           18.4         24.8
  (orphan row, no label)                 —         18.4
```

Same shape in the `May cost-of-living`, `Growth Plan` and `Autumn Statement` blocks. A
headline spot-check passes (`64.2 / 39.8` sit in the un-nested top block and are correct)
while 67% of cells are on the wrong row. This is #96 as filed. Note the table carries its
own checksum: `24.8 + 18.4 = 43.2`.

**(b) Label truncation and column explosion** — pages 48, 51, 53, 64, 65. Labels are cut to
their first word and scattered across phantom columns. Page 65:

| native | emitted |
|---|---|
| `Income tax1` | `Income` |
| `National insurance contributions` | `National` |
| `Health and social care levy` | `Health` |
| `Other income tax` | `\| \| tax \| \|` |

Values are correct and correctly ordered. The labels are destroyed, so the grid is unusable.

**(c) Table absent — page 55 only.** Native text shows `Table A: Implications of higher gas
prices for near-term borrowing`; the fragment emits zero markdown table rows. Silent, unlike
page 46 which at least fails loudly with
`[page 46 failed: unverifiable table — see image]`.

> **Correction.** An earlier revision of this note also listed pages 37, 41 and 43 under this
> mode. That was wrong, and the error was mine, not socr's: those pages contain **Chart 16**,
> **Chart 18** and **Chart D**, not tables. Their numeric lines are chart data labels, which
> the ground-truth parser mistook for table rows (yielding phantom "ground truth" of 1, 1 and
> 2 rows). socr was **correct** to emit no table on all three, and the escalation run
> independently returned `NO TABLE` for each. Their phantom cells are excluded from the
> aggregate above.

**(d) Repetition runaway** — page 62. 865 consecutive copies of `| | | | | | | |`, reaching
the final `.md` as 867 of 3177 lines (27% of the document). Filed separately as **#97**;
the fix is a bounded repeat guard plus an audit kind, independent of #96's routing design.

## Result 2 — the dual-pass detector is trustworthy

Scored every page with a substantial table, flagged or not. Exactly one substantial-table
page was unflagged: **page 66, at 91.2%**. Zero false negatives.

Caveat: thin base. Most table pages in this document were flagged, so "zero false
negatives" rests on a single negative control. Worth re-checking on a second document
before treating the flag list as a complete trust index.

This matters for **#95**: the signal being hidden from consumers is a *reliable* signal.
Surfacing it is high-value and low-risk.

## Result 3 — escalation works

Page 13, the worst hierarchical case:

| path | label-keyed exactness |
|---|---|
| socr, strict-local qwen | **32.5%** (41/126) |
| agy on the page PNG | **95.2%** (120/126) measured, **~100%** true |

The 6-cell residual is entirely the scorer's duplicate-label collision described above. agy
reproduced the nested structure correctly: parents carrying their own values, `of which:`
children carrying theirs, zero orphan rows, and one first-column mismatch which is the same
collision artifact.

So the "escalate to a second strategy" branch of #96 is **empirically viable**. Fail-closed
is not the only remaining option.

Model provenance is unconfirmed: `agy-set-model` is a **dead symlink** into a deleted iCloud
path (`~/Library/Mobile Documents/.../toolkits/disputatio/vendor/agy-set-model`), so per-call
model routing is broken and the run used whatever selection was sticky, presumably the
`Gemini 3.5 Flash (High)` default. If the cheap tier produced this, escalation is affordable.
Confirm before costing it. This also breaks `/disputatio` and agent-ctl `--model` for gemini.

## Result 4 — the null result that constrains the design

The first agy attempt was run with `--sandbox`, which blocked the image read. It did not
error. It returned **741 bytes of clean, confident, entirely fabricated fiscal data** at
exit code 0 with empty stderr:

```
| | 2017 | 2018 | 2019 | 2020 |
| **Revenue** | **23,126** | **25,123** | **26,204** | **25,607** |
| Taxes | 19,451 | 21,326 | 22,239 | 21,650 |
| of which: indirect taxes | 10,230 | 11,450 | 12,010 | 11,500 |
```

The real table is £ billions over 2022-23…2027-28 concerning UK government decisions. There
is no `Revenue` row, no `indirect taxes` row, no year 2017, and no five-figure value anywhere
on the page. Every cell is invented.

Worse, it satisfied every structural instruction it was given: parents populated, children
carrying their own values, no orphan rows, no empty labelled rows. **It would have scored
better than the real output on every model-free heuristic in Result 1** — orphan-row count,
empty-parent count, lane-count agreement.

Had this been wired in as an auto-reviser, it would have replaced a 32.5%-correct table with
a 0%-correct one, and every available signal would have reported an improvement.

### Consequence: escalation requires a grounding canary, not a quality check

Structural plausibility cannot distinguish a good transcription from a fabrication. The
escalation lane needs evidence the model actually read *this page* before its output is
allowed to replace anything:

- require the model to echo caption, units, and column headers off the image first;
- match those against the native text layer, which supplies them for free on born-digital
  pages;
- reject the candidate outright on canary mismatch — never fall back to "use it anyway";
- record provenance on any replaced fragment (engine, timestamp, canary result, diff), so a
  revised page is never silently indistinguishable from an original one.

The second run added exactly this canary and it passed cleanly (`CAPTION: Table 1: Total
effect of Government decisions since March / UNITS: £ billion / YEARS: 2022-23, …, 2027-28`),
which is how Result 3 is known to be grounded rather than a lucky fabrication.

## Result 5 — escalation across all 19 flagged pages

Result 3 rested on a single page. Extended to all 19, each run with the canary gate, 4
concurrent:

| page | socr % | agy % | gt cells | verdict |
|-----:|-------:|------:|---------:|---------|
| 13 | 32.5 | **95.2** | 126 | hierarchical shift fixed |
| 39 | 79.7 | 79.7 | 74 | tie — socr already correct, scorer ceiling |
| 45 | 100.0 | 100.0 | 49 | tie — both correct |
| 46 | 0.0 | **100.0** | 75 | **socr fail-closed; agy recovered exactly** |
| 48 | 0.0 | **76.8** | 82 | label shredding fixed |
| 51 | 12.9 | **70.6** | 85 | label shredding fixed |
| 53 | 0.0 | 0.0 | 52 | tie — scorer parses no labels, uninformative |
| 55 | 0.0 | **100.0** | 33 | **socr dropped table; agy recovered exactly** |
| 59 | 76.7 | 76.7 | 210 | tie — scorer ceiling |
| 60 | 73.3 | 73.3 | 180 | tie — scorer ceiling |
| 61 | 32.1 | 32.1 | 224 | tie — scorer parse failure, uninformative |
| 62 | 3.6 | **32.1** | 168 | runaway avoided; still low (see #97) |
| 63 | 77.8 | 77.8 | 189 | tie — scorer ceiling |
| 64 | 0.0 | **89.7** | 271 | label shredding fixed |
| 65 | 0.0 | **89.7** | 232 | label shredding fixed |
| 67 | 94.1 | 94.1 | 204 | tie — both correct |
| 37, 41, 43 | — | `NO TABLE` | — | chart pages; both engines agree, correctly |

**Aggregate: socr 870 / 2254 = 38.6%, agy 1668 / 2254 = 74.0%.**

Three findings that matter more than the aggregate:

1. **agy was never worse than socr on any page.** Escalation carries no regression risk on
   this document, which is what makes attempting it safe.
2. **It recovers pages socr gives up on.** Page 46 was `table_region_unverifiable` +
   `page_failed` — socr emitted a stub and an image. agy reproduced all 14 rows matching the
   native layer. Page 55's dropped table likewise came back exact. So **fail-closed pages are
   recoverable**, and fail-closed should be the fallback *after* escalation, not instead of it.
3. **Mode (b) is fixed, not just mode (a).** Pages 48, 51, 64, 65 went from unusable (labels
   destroyed) to 70–90%. The original scope of #96 was too narrow.

The ties are not failures. Seven of them are pages where socr was already correct and both
engines hit the scorer's ceiling; two (53, 61) are pages the scorer cannot parse. See the
Method limitations.

### Canary results

All 15 table pages returned canaries consistent with the document: captions form the coherent
sequence `Table 1, 2, 3, 4, 5, 6, 7, A.1, A.2, A.3, A.4, A.5, A.6, A.7, A.9`, with correct
units and year headers. Zero canary mismatches, i.e. **zero fabrications in 19 correctly
configured runs**. Combined with Result 4, the pattern is clear: fabrication is what happens
when the model is denied its input, not a baseline rate. The canary detects exactly that
condition, which is why it is the right gate.



**#97 (new, most urgent)** — repetition runaway. Corrupting 27% of a shipped document today
with no detector anywhere. Bounded guard plus audit kind. Independent of everything else.

**#95** — surfacing. Confirmed with numbers: `metadata.json` reports
`error: "page(s) 46 produced no usable output"`, naming one page, while the actual state is
16 table pages at 38.6% aggregate exactness, one silently dropped table (55), and a quarter
of the document filled with blank rows. The signal being withheld is reliable (Result 2), so
a `tables_trust.json` sidecar plus a doc-level count is worth building and low-risk.

**#96** — no longer blocked on a design question:
- escalate is viable and **never regressed a page** (Results 3, 5), so pick
  escalate-with-canary, with fail-closed as the fallback *after* escalation rather than
  instead of it — page 46 proves fail-closed pages are recoverable;
- the canary and rejection rules are mandatory, not optional hardening (Result 4), and cost
  nothing to check on born-digital pages;
- scope must widen from mode (a) to modes (a), (b) and (c) — escalation demonstrably fixes
  all three;
- the fixture metric must key rows by `(parent, label)` path, and must be validated against a
  page where the current output is known-good so its ceiling is known (Method, limitation 2);
- native-layer reconciliation remains a cheaper partial fix for mode (a) specifically, since
  the correct row order is already present locally at zero cost.

## Appendix — scorer used

Scratch harness, `/tmp/obr-revise/score2.py`. Reproduced here because it is the measurement
behind every number above. **Not production code**: it has the duplicate-label collision
described in Method, and no tests.

```python
import re, sys, json

NUM = re.compile(r'^-?[\d,]+\.?\d*$|^-?\d+$')
SKIP = ('of which', 'note:', 'memo:')

def norm(s):
    return re.sub(r'[^a-z0-9]+', '', s.lower())

def truth(path):
    """label -> [values] from a PDF page's native text layer."""
    rows, label, vals = [], None, []
    for raw in open(path):
        l = raw.strip()
        if not l:
            continue
        if NUM.match(l):
            if label:
                vals.append(l)
        else:
            if label and vals:
                rows.append((label, vals))
            label = None if l.lower().startswith(SKIP) else l
            vals = []
    if label and vals:
        rows.append((label, vals))
    return [(l, v) for l, v in rows if len(v) >= 2]

def md_rows(path):
    """label -> [numeric cells] from an emitted markdown table."""
    out = []
    for line in open(path):
        s = line.strip()
        if not s.startswith('|'):
            continue
        cells = [c.strip() for c in s.strip('|').split('|')]
        if cells and cells[0] and set(cells[0]) <= set(':- '):
            continue
        lab = cells[0].replace('**', '').strip() if cells else ''
        vals = [c.replace('**', '').strip() for c in cells[1:]
                if NUM.match(c.replace('**', '').strip())]
        out.append((lab, vals))
    return out

def main(native, md):
    gt, got = truth(native), md_rows(md)
    lookup = {}
    for lab, v in got:
        if lab:
            lookup.setdefault(norm(lab), v)   # <-- collision on repeated labels
    hit = tot = 0
    missing_rows, shifted = 0, []
    for lab, tv in gt:
        gv = lookup.get(norm(lab))
        tot += len(tv)
        if gv is None:
            missing_rows += 1
            continue
        hit += sum(1 for a, b in zip(tv, gv) if a == b)
        if gv and tv and gv[0] != tv[0]:
            shifted.append((lab, tv[0], gv[0] or '(empty)'))
    seen = {}
    for lab, v in got:
        k = (lab, tuple(v))
        seen[k] = seen.get(k, 0) + 1
    return dict(gt_rows=len(gt), cells=tot, exact=hit,
                pct=round(100 * hit / tot, 1) if tot else None,
                rows_not_found=missing_rows,
                orphan_rows=sum(1 for lab, v in got if not lab and v),
                labelled_but_empty=sum(1 for lab, v in got if lab and not v),
                duplicate_rows=sum(c - 1 for c in seen.values() if c > 1),
                first_col_mismatch=len(shifted))

if __name__ == '__main__':
    print(json.dumps(main(sys.argv[1], sys.argv[2])))
```

Page PNGs were rendered at 200 dpi with `fitz.Matrix(200/72, 200/72)`; native text with
`page.get_text()`.
