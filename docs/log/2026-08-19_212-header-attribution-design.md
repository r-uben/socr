# GH-212 — a sound header-attribution predicate

**Status:** specification, ratified. Not implemented.
**Verified against source at `ad92687`.** Every file:line below was re-read; every
frequency below was measured on the 367 non-empty PDFs of the paper library.

---

## The defect

On the `EXACT_PASS` path (`src/socr/pipeline/agentic.py:603-625`) socr returns
`accept=True, confidence=1.0` with no header-attribution term. The only gate is grid
*shape* (`src/socr/tables/structure_check.py:234-268`). A regression table whose
numerals are all correct but whose header band was destroyed therefore ships as
SUCCESS, and its coefficients cannot be attributed to a column.

Measured on 4 of 4 hand-judged damaged pages
(`docs/log/2026-08-15_200-open-decision-1-resolved.md`). Two of them — `021` and `028` —
have header tokens *entirely absent* from the emitted header row. This is the widest
blast radius of any open defect, on exactly the booktabs tables this corpus is made of.

## Why the four previous attempts were reverted

All four recovered header *role* from token **content**, or from a numeral-count proxy
for it, in `page.get_text("words")`. In this corpus that role is not a property of the
token. The docstring at `structure_check.py:243-258` records the parking decision.

| Attempt | Rule | Failure |
|---|---|---|
| T3 (`5a93b5f`) | `_is_table_header_row` requires `%` / `+/-` / `to\|more\|less` (`header_repair.py:146-161`, still current) | Abstains on `1997/2002` and `(1)(2)(3)` — i.e. on the 4/4 case |
| `062bdef` | `year_hits >= 2 or colnum_hits >= 2` | A body row with two four-digit numbers became the header band → HARD |
| Positional (`a767ee1`) | header = lane-aligned ∧ not in `_confirmed_data_ys`; data = numeric multiset with `_MIN_DATA_NUMERIC_CELLS = 3` (`header_repair.py:48`) | Star-only and `n.a.` rows have zero `_NUM_TOKEN_RE` hits, are never confirmed-data, sit above the first ≥3-numeral row, and are owed as header → HARD on a correct table |
| Normalized (`feba18b`) | cell-wide decoration strip, then split | `**Std** **Err**` → `{'std**','**err'}` → HARD on an intact header |

`header_attribution()` as it currently stands (`header_attribution.py:65-106`) must not
be unparked: it still calls `native_header_row` → T3's `_is_table_header_row`.

## The mechanism

**Do not ask whether a native row looks like a header.** Cut the page at the drawn rule
immediately above the numeric anchor. Significance-star and `n.a.` body rows sit below
that rule on booktabs tables and are therefore never owed. That is a geometric fact, not
a token classifier.

### Inputs

1. **Emitted grid** — `find_table_blocks(output_md) → block.grid`
   (`header_attribution.py:68-69`, `structure_check.py:266`).
2. **Native words** — `page.get_text("words")`, already taken at `agentic.py:522`.
3. **Horizontal rules** — a list precomputed by `locate._horizontal_rules(page)`
   (`locate.py:169-208`). This is the input none of the four attempts used.

Not used: the page image, the inner VLM judge, `_is_table_header_row`,
`_YEAR_HEADER_RE`, or `_MIN_DATA_NUMERIC_CELLS` as a header/body test.

### Algorithm — HARD only, pure

**1. Anchor, with a uniqueness check that must be written.**
`_best_anchor_y` (`header_repair.py:95-108`) returns the **first** native y-row whose
numeric multiset equals an emitted body row — it iterates `for y in sorted(rows_by_y)`
and returns on first match. **There is no uniqueness test in it, or anywhere in
`src/socr/tables/`.** Build one: collect *all* matching y-rows and abstain unless
exactly one matches.

This is not optional. Two panels on one page routinely share a rounded standard-error
row (`(0.01) (0.01) (0.01)`); without the check, table 2's grid anchors onto table 1's
copy, the cut lands on the wrong table's midrule, and a byte-perfect table gets HARD.

No match, or more than one → **UNVERIFIABLE**, never HARD. Star and `n.a.` rows have
empty numeric multisets, so they can neither be the anchor nor steal it from a
coefficient row.

**2. Lanes.** `_derive_lane_centers` (`header_repair.py:246`). Fewer than 2 lanes →
UNVERIFIABLE. Its internal filter at `header_repair.py:256` uses raw
`_NUM_TOKEN_RE.match(w[4])`; it must go through `is_numeric_token` /
`strip_presentation` (`native_verifier.py:445-491`) or leading-decimal columns (`.034`)
never form lanes and the owed set silently shrinks — the #206 blind spot.
Line 256 is *inside* `_derive_lane_centers`; this is a change to that function, not a
reuse of it unchanged.

**3. Cut.** Among rules whose x-span covers the anchor row's own `[min x, max x]`
(derived from that row — **not** `_RULE_X_OVERLAP = 0.6`), take the rule with the
largest y still `< anchor_y`, and only if it lies inside `_local_table_ys`
(`header_repair.py:118-125`). No such rule → UNVERIFIABLE. Never fall back to a content
test.

**4. Header y-range — no `min(local_ys)` fallback.**
The range is `(previous rule in that local set, cut_y)`. If there is no rule above the
cut, **abstain**. Do not fall back to `min(local_ys)`: `_local_table_ys`
(`header_repair.py:118-135`) admits every y-group overlapping the anchor's x-extent
±20pt, so on a full-width table that fallback owes the caption, the running head, and
any prose numeral within the 18pt snap radius (`_LANE_X_TOL_PT` 6.0 × `_LANE_SNAP_MULT`
3) — none of which appear in `grid[0]`, producing HARD on a correct table.

Words in the range that snap to a data lane are **owed**. Caption and title above the
toprule are out. Body, including every star and `n.a.` row, is below `cut_y` and out.

**5. Membership — fold brackets on this path only.**
HARD iff the owed-token `Counter` is not a sub-multiset of the emitted header's
`Counter`. Compare per whitespace-split token after `strip_presentation` + `casefold()`.
Never strip decoration from a cell as a whole — that was `feba18b`'s bug.

`strip_presentation` (`native_verifier.py:425-436`) deliberately does **not** fold
parentheses, because brackets denote negatives on the numeric path. On the header
comparison path they are presentation: native `(1)` must match emitted `1`. Fold
`()` and `[]` **here and nowhere else**. Same shape, same treatment: a trailing `%`
(native `Share %` vs emitted `Share`, standard when the unit moves to the caption).

This is not a vocabulary — no year lists, no `n.a.` lists. It is bracket and unit
punctuation, in the same class as the marks `_PRESENTATION_MARKS` already folds.

**6. The emitted side is not `grid[0]` alone.**
Markdown carries exactly one header row (`reconcile.py:87-110`), but a native header
band can have two tiers (`Model A | Model B` over `(1) | (2)`), and a *faithful*
emission puts tier 2 in `grid[1]`. Comparing against `grid[0]` alone makes the verdict
depend on the model's flattening style rather than on content.

Union `grid[0]` with any leading rows of `grid[1:]` that carry no numeric multiset of
their own, or abstain when `grid[1]` looks like a second header tier. Empty owed set →
OK.

**7. SOFT** stays advisory, unchanged
(`docs/log/2026-08-15_200-open-decision-1-resolved.md:37-44`).

### Why this cannot HARD on correct star / n.a. tables

Those rows have `y > cut_y`. They are never in the owed set, and the proof never
inspects their tokens, so it cannot confuse `***` or `n.a.` with header content. T3's
year and `(1)(2)(3)` bands sit *above* the midrule and are owed; if the emitted header
lost them, HARD fires — that is the 4/4 case (pages `021` and `028`, header tokens
ABSENT). Header tokens dumped into a body row (`021`) stay owed, because exclusion is by
y-cut, not by "this bag also appears in the body".

## Measured frequencies

Across 6,350 table-bearing pages in 367 non-empty library PDFs:

| Fact | Frequency | Consequence |
|---|---|---|
| Table page loses its **topmost** rule to `_RULE_FLATNESS_PT = 1.0` | **1.6%** | Abstain on these. Rare enough not to shape the design — but page `021` is one of them (toprule 1.494pt), so the damaged set is not representative. |
| Table page loses **any** wide rule to the same cutoff | 2.7% | As above. |
| Rules at 0.0pt (hairline) | 44,655 of ~56,600 | The midrule the cut needs is detected essentially always. |
| Table page carries a `(1)`…`(2)` column-number run | **11.1%** | ~1 in 9. Step 5's bracket fold is mandatory, not an edge case: without it this predicate false-rejects hundreds of correct tables and becomes the fifth revert. |

`_horizontal_rules` handles both stroked lines and filled rectangles
(`locate.py:191-201`), so filled-bar midrules are not a gap. `_RULE_FLATNESS_PT = 1.0`
and `_RULE_MIN_WIDTH_PT = 72.0` (`locate.py:35-36`) were tuned for `locate_tables`,
where a miss merely "forfeits the benefit" (`locate.py:138`); here a miss means
abstain, which is the safe direction.

## Failure modes

A false **ACCEPT** is the unacceptable direction for a citation corpus.

| Mode | Direction | Note |
|---|---|---|
| No midrule / borderless / `get_drawings()` empty | False accept, via abstain shipping on EXACT_PASS | Documented limit (`locate.py:17-19`). Scanned pages cannot EXACT_PASS anyway (`native_verifier.py:44-45`). Tracked as #245. |
| Topmost rule too thick (1.6%) | Abstain | Same exposure as above. |
| Anchor not unique | Abstain | Step 1 |
| No rule above the cut | Abstain | Step 4 — deliberately not a fallback |
| SOFT-only (tokens present, wrong column) | False accept of a mis-column | 1 of 3 classifiable damaged pages. Do not promote SOFT. |
| Spanning merged header | Documented miss | `200-shipping-spec.md:211-216` |
| Header tokenisation beyond the folds (`R^2` vs `R2`) | False reject | Residual. Keep the fold list to `_PRESENTATION_MARKS` + dash + bracket + `%`; do not grow a header vocabulary. |
| Rule narrower than `_RULE_MIN_WIDTH_PT` (72pt) | Abstain | Inherited named limit |

## Placement

`table_output_defect` (`structure_check.py:234`) is documented "Pure; no mutation, no
model calls" (`:264`). **Thread a precomputed rule list, never the `fitz` page** —
calling `get_drawings()` inside would break that purity and invert the layering.
`agentic.py` already holds the page at `:521`; compute `_horizontal_rules(page)` there
and pass the list.

The term belongs at the parked site (`structure_check.py:266-269`) so it runs on every
accepting path. `_apply_structural_gate` already covers warn-delegate, EXACT_PASS,
no-issue-delegate and (shape-only) verifier-exception exits (`agentic.py:636-639`), and
`test_structural_gate_covers_all_three_accept_paths` (`tests/test_agentic.py:359-363`)
exists so nobody patches only EXACT_PASS.

**Second caller.** `born_digital.py:1170` also calls
`structure_check.table_output_defect(native_text, words)`. A `rules=None` default keeps
it compiling but makes it silently abstain there. That is probably right — that path
flags the native layer, not a VLM table — but it must be a stated decision, not an
accident. The exception path likewise keeps `words=None`, skips this term, and still
runs shape (`agentic.py:531-537`).

## Tests

Fixtures that only `insert_text` and never draw a rule (`tests/test_agentic.py:282-295`)
will abstain under this predicate. **Every HARD test must draw a real booktabs midrule.**
Required cases: the 4/4 absent-header case (HARD); a correct table with star and `n.a.`
rows (OK); a correct table with `(1)(2)(3)` column numbers emitted as bare `1 2 3` (OK —
this is the 11.1% case); two panels sharing a standard-error row (abstain); a two-tier
header emitted across two markdown rows (OK); a page whose toprule is 1.5pt (abstain).

## Decision on abstain

Ship the HARD term alone. When the term abstains, the page continues to ship as it does
today; the EXACT_PASS short-circuit is **not** refused.

It closes the measured 4-of-4 booktabs case at the smallest blast radius. Borderless and
thick-toprule tables keep riding EXACT_PASS at confidence 1.0 — a pre-existing,
documented, unchanged hole, filed as **#245** and blocked on this landing. Escalating
every abstain to the VLM judge would add cost and latency on pages with no evidence of
damage.

## Provenance

Grok 4.6 produced the geometric mechanism and the first specification. Kimi K3 and
DeepSeek-v4 attacked it independently, read-only; Gemini 3.1 Pro was assigned and died
on an upstream error. Kimi returned NOT SOUND with six defects, DeepSeek IMPLEMENT WITH
NAMED CHANGES. The orchestrator re-read every load-bearing claim against source — two of
the original specification's assertions were false (the anchor uniqueness guard, and
"lanes unchanged") — then measured the two disputed frequencies across the library.
Steps 1, 2, 4, 5 and 6 above exist because of that pass.

This is the fifth attempt at this predicate. The four before it were reverted for
false HARD on correct tables, and the first adversarial read broke this one too. The
difference now is that the two constants it turns on have been measured rather than
assumed. Nothing here should be implemented without the tests listed above.
