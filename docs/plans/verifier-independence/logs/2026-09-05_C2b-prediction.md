# C2b prediction artifact — the 14 remaining items under C1 §(a) rev 4 + the #614 prefix rule

2026-09-05. Computed on the tree that merges as C2a (`feat/vi-C2a-line-bands@171b861`, PR
#616, on `main@2fe84d2`) with the **shipped helpers only** — `ordinal_origin`, `row_bands`,
`label_column_edge`, `band_index_for`, `BindingResult.native_rows` / `.row_binding` — over
the frozen corpus `~/Data/socr/ladder-run2-2026-09-04` (checksums verified). Fresh `bind()`
on each frozen page with its provenance-selected candidate (A1/A1c). This is the artifact
C2b's frozen-replay gate asserts against item-for-item; it is committed before C2b starts.

Rule applied per item, in order (first failure is the reason): origin exists; column edge
`R` exists; the disputed native row is unique by `row_path`; **native chain** — native rows
`0…i` sit on bands `0…i` one-for-one, each row anchored on its own word boxes (label box,
else lane boxes; never the binder's rounded `y`, which sits 0.4 pt above its band on doc05);
**model chain** — model row `j` is bound to `i`, `j = i`, and every model row `≤ j` is bound
with its partner on the band of its own index; **prefix rule (#614)** — no ambiguous band in
`0…b`; **column test** — the band's leftmost line ends before `R` and its second line starts
at or after `R`. Address = `(region.x0, band.y0, R, band.y1)`.

## Result

| | |
|---|---|
| remaining contradicted items (22 − 7 doc02 − 1 doc01) | **14** |
| geometry-addressed | **3** — doc05 p1: `∆log S&P` (4,4,4), `∆Slope (3m)` (6,6,6), `∆log Comm. price (3m)` (8,8,8) |
| abstained | **11** |
| feasibility checkpoint (≥ 1 addressable remaining item) | not tripped |

Identical to C1 rev 4 §(d): same three items, same triples. Coverage after A2 = 3/14.

## Per item

| doc | p | table | native \| model | verdict | reason |
|---|---|---|---|---|---|
| doc | p | table | native\|model | **verdict** | reason |
| doc03 | 1 | p1-t0 | S&P\| | **abstained** | column test: leftmost line crosses R or second line starts before R |
| doc03 | 1 | p1-t0 | \|R$^{2}$ | **abstained** | native chain breaks at native row 3 (band 2) |
| doc04 | 3 | p3-t0 | 1t 1t\|**ROTATED PCs** $\math | **abstained** | no origin (no second rule group) |
| doc05 | 1 | p1-t0 | ∆Slope (3m)\|$\Delta$ Slope (3m) | **addressed** | (i,j,b)=(6,6,6) cell=(72.0,235.4,203.3,247.3) |
| doc05 | 1 | p1-t0 | ∆log Comm. price (3m)\|$\Delta \log$ Comm. pr | **addressed** | (i,j,b)=(8,8,8) cell=(72.0,271.2,203.3,283.2) |
| doc05 | 1 | p1-t0 | \|$R^2$ | **abstained** | native chain breaks at native row 13 (band 12) |
| doc05 | 1 | p1-t0 | ∆log S&P\|$\Delta \log$ S&P 500  | **addressed** | (i,j,b)=(4,4,4) cell=(72.0,199.5,203.3,211.5) |
| doc05 | 1 | p1-t0 | Sample 1988:1–2019:12 \|Sample | **abstained** | native chain breaks at native row 13 (band 12) |
| doc05 | 1 | p1-t0 | Policy surprise mps mp\|Policy surprise | **abstained** | native chain breaks at native row 13 (band 12) |
| doc07 | 1 | p1-t0 | ∆Slope (3m)\|$\Delta \text{ Slope ( | **abstained** | no column edge |
| doc07 | 1 | p1-t0 | ∆log Comm. price (3m)\|$\Delta \log \text{ Co | **abstained** | no column edge |
| doc07 | 1 | p1-t0 | \|R$^2$ | **abstained** | no column edge |
| doc07 | 1 | p1-t0 | ∆log S&P\|$\Delta \log \text{ S\ | **abstained** | no column edge |
| doc07 | 1 | p1-t0 | Sample 1988:1–2019:12 \|Sample | **abstained** | no column edge |

## Abstention owners

| reason | items | owner |
|---|---|---|
| no origin (scanned page, no second rule group) | doc04 ×1 | #603; abstains by design |
| no column edge — label x0s are ragged (72.0 / 74.0 / 78.0: indented sub-labels), so `R` shrinks onto the label column and collapses | doc07 ×5 | C1 §(f); a per-band label edge is a design change, not C2b's |
| native chain breaks at native row 13 (band 12): the native rowizer has one more row than the page has printed rows above `R²` — a split/merged row upstream of the item | doc05 ×3 (`R²`, `Sample`, `Policy surprise`) | #600 (`round(y0)` split) family; A2-style native repair |
| native chain breaks at native row 3 (band 2) | doc03 ×1 (`R²`) | same family |
| column test — the leftmost line crosses `R` (lane-shift item; the label column is not clean on that band) | doc03 ×1 (`S&P`) | lane shift, class (d) |

## Reproduction

The computation is the C2b implementation's contract; until C2b lands it is reproduced by
the script below (run with `PYTHONPATH=<C2a tree>/src`, module-style). C2b's
`socr-replay-binding` extension must print the same 14 verdicts.

```python
"""C2b prediction artifact: apply C1 §(a) (rev 4) + the #614 prefix rule to every
remaining contradicted item on the post-A2/C2a tree, with the shipped helpers only.
Output: one row per item — geometry-addressed or abstained, with the first failing
condition — plus the counts the C2b gate asserts against."""

import sys
from pathlib import Path

from socr.benchmark.replay_binding import _select_candidate_for_table, _witness_for_table, discover_pages
from socr.core.pdf import open_pdf
from socr.tables.adjudication import items_from_binding
from socr.tables.binding import bind
from socr.tables.locate import _text_lines_in_region, band_index_for, label_column_edge, ordinal_origin, row_bands
from socr.tables.witness import prepare_table_witnesses

corpus = Path(sys.argv[1]).expanduser()
rows = []
for rec in discover_pages(corpus):
    for table_id in sorted(rec.binding_adjudication):
        md, why = _select_candidate_for_table(rec, table_id)
        if md is None:
            rows.append((rec.doc_slug, rec.page_num, table_id, "-", "-", "unreplayable", why)); continue
        with open_pdf(rec.pdf_path) as doc:
            page = doc[rec.page_num - 1]
            words = page.get_text("words")
            with prepare_table_witnesses(rec.pdf_path, rec.page_num, md) as ws:
                w = _witness_for_table(ws, table_id)
                region = w.box.bbox
            result = bind(words, md, region=region)
            items = items_from_binding(result)
            origin = ordinal_origin(page, region)
            bands_all = row_bands(page, region)
            R = label_column_edge(page, region)
            lines = _text_lines_in_region(page, region)
        bands = [b for b in bands_all if origin is not None and b.y0 >= origin]
        nrows = result.native_rows
        inv = {v: k for k, v in result.row_binding.items()}  # native idx -> model idx

        def band_of(native_idx):
            # C1 §(a): a native row's band is taken from its OWN words — the
            # label box, else its lane boxes — never from the binder's rounded y.
            r = nrows[native_idx]
            boxes = ([r.label_bbox] if r.label_bbox else []) + list(r.lane_bboxes.values())
            if not boxes:
                return None
            y_mid = (min(b[1] for b in boxes) + max(b[3] for b in boxes)) / 2.0
            return band_index_for(bands, y_mid)

        for it in items:
            tag = (rec.doc_slug, rec.page_num, table_id, it.kind, (it.native_token or "")[:22] + "|" + (it.model_token or "")[:22])
            if origin is None:
                rows.append((*tag, "abstained", "no origin (no second rule group)")); continue
            if R is None:
                rows.append((*tag, "abstained", "no column edge")); continue
            cands = [k for k, r in enumerate(nrows) if tuple(r.row_path) == tuple(it.row_path)]
            if len(cands) != 1:
                rows.append((*tag, "abstained", f"native row for row_path not unique ({len(cands)})")); continue
            i = cands[0]
            # condition 2: native chain 0..i one-for-one on bands 0..i
            ok = True
            for k in range(i + 1):
                if band_of(k) != k:
                    rows.append((*tag, "abstained", f"native chain breaks at native row {k} (band {band_of(k)})")); ok = False; break
            if not ok: continue
            b = i
            # condition 3: model chain — model row j bound to i, and every j' <= j bound with partner on band j'
            if i not in inv:
                rows.append((*tag, "abstained", "disputed native row not bound to a model row")); continue
            j = inv[i]
            if j != b:
                rows.append((*tag, "abstained", f"model index {j} != band {b}")); continue
            for jp in range(j + 1):
                if jp not in result.row_binding or band_of(result.row_binding[jp]) != jp:
                    rows.append((*tag, "abstained", f"model chain breaks at model row {jp}")); ok = False; break
            if not ok: continue
            # prefix rule (#614): no ambiguous band in 0..b
            amb = [k for k in range(b + 1) if bands[k].ambiguity]
            if amb:
                rows.append((*tag, "abstained", f"prefix crosses ambiguous band(s) {amb}")); continue
            # condition 4: column — band's leftmost line entirely left of R; second line at/after R
            band = bands[b]
            in_band = sorted((ln for ln in lines if band.y0 <= ln["baseline"] <= band.y1), key=lambda ln: ln["x0"])
            if not in_band or in_band[0]["x1"] > R or (len(in_band) > 1 and in_band[1]["x0"] < R):
                rows.append((*tag, "abstained", "column test: leftmost line crosses R or second line starts before R")); continue
            rows.append((*tag, "addressed", f"(i,j,b)=({i},{j},{b}) cell=({region[0]:.1f},{band.y0:.1f},{R:.1f},{band.y1:.1f})"))

print("doc\tp\ttable\tkind\tnative|model\tverdict\treason")
for r in rows: print("\t".join(str(x) for x in r))
n_items = sum(1 for r in rows if r[5] in ("addressed", "abstained"))
print(f"\nitems={n_items} addressed={sum(1 for r in rows if r[5]=='addressed')} abstained={sum(1 for r in rows if r[5]=='abstained')}")
```
