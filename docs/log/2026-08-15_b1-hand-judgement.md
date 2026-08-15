# B1 shape-gate hand judgement — is 26.9% a firing rate or a defect rate?

Date: 2026-08-15
Question: for pages where **B1's shape gate fires and TR-3 does not**, is the reconstructed
table actually damaged, actually fine, or not a table at all?

This is the measurement named as the blocking next step in
`2026-08-15_tr3-hand-judgement.md` (open question 2) and in the #200 shipping spec
(OPEN DECISION 1). TR-3's own firings were judged earlier the same day; this measures the
*other* side of the disjunction, which nobody had looked at.

## Population

Same 40-paper list as the TR-3 judgement (`/tmp/b1probe/list.txt`), so the two measurements
are comparable. Builder: `~/.local/share/socr/tr3-judge/build_b1_review_set.py`, sampling
rule in its docstring.

| | pages |
|---|---|
| native table pages scanned | 491 |
| B1 fires (total) | 71 |
| — B1 **and** TR-3 | 38 |
| — **B1 but not TR-3** ← the population | **33** |
| TR-3 only (B1 misses) | 28 |
| staged for review (cap 25, binding) | 25 |

**The non-overlap claim holds on current code.** `2026-08-14_gh151-b1-predicate-design.md`
asserted TR-3 misses 31 shape-gate pages and catches 27 B1 misses; measured now it is **33
and 28**. Neither signal subsumes the other — re-confirmed, not merely cited.

## What was judged, and how

**5 pages judged carefully**, side by side against the rendered page. The remaining 20 were
**screened at a high level by the owner, not judged** — reported as showing the same picture.
That distinction is recorded deliberately: the counts below are out of 5, and the screen is
supporting context, not data. Two earlier measurements in this repo failed to state their
sampling rule and had to be corrected; this one states it.

| # | page | verdict | what was wrong |
|---|---|---|---|
| 0 | `2000__romer_romer` p21 | **damaged** | header band destroyed / detached from columns — *and see the separate finding below* |
| 1 | `2003__woodford` p798 | **not a table** | book back-matter index |
| 2 | `2010__Menzly_Ozbas` (JF) p20 | **damaged** | header lost |
| 3 | `2013__Snowberg_Wolfers_Zitzewitz` p15 | **damaged** | severe: columns scrambled, values under wrong headers, Notes prose absorbed into the grid |
| 4 | `2013__Snowberg_Wolfers_Zitzewitz` p16 | **damaged** | same shape as p15 |

**4 damaged, 1 not-a-table, 0 fine.**

## What this establishes

**B1's gate is not over-firing.** Zero "fine" verdicts. The concern that 26.9% was noise — the
reason the prior plan proposed swapping B1's predicate for TR-3 — is not supported. #200
shipping on B1's predicate alone is the right call.

**Header loss is the dominant defect on this side too.** Two of four damaged pages failed
specifically on the header band, independently volunteered by the judge. That is the *same*
failure mode found in the TR-3 judgement, on a **disjoint set of pages**. So the parked
header-attribution term (#215) targets the defect that actually breaks tables under *both*
signals — it is not a nice-to-have.

**A fifth defect class.** Snowberg p15/p16 are not header loss: whole columns are transposed,
values land under the wrong headers, and the Notes paragraph is absorbed as table rows. Not
in the five classes catalogued in the TR-3 judgement.

**#213 corroborated.** The one non-table is `2003__woodford` p798 — a book back-matter index,
the *same document* flagged in the TR-3 sample. Non-tables reaching table reconstruction is
now evidenced on two independent samples.

## What it does NOT establish

5 judged pages span only **4 independent documents** (p15/p16 are the same paper), so pages
cluster within documents and a rate computed from them would overstate its own precision.
0 "fine" out of 5 bounds the fine-rate below roughly 60% at 95% by the rule of three — enough
to kill "B1 is mostly noise", **not** enough to publish a defect rate. If a number is ever
needed for a decision, judge ~15 more, one page per unjudged paper (16 distinct papers are
present in the staged 25).

---

# Separate finding: the text layer silently inverts negative numbers

Found while reviewing page 0, **not** part of the judgement above. Filed separately.

`2000__romer_romer` p21: the paper prints `−0.12`; socr ships `20.12`. This is not a
reconstruction defect — **PyMuPDF's `get_text()` returns `20.12` from the PDF's own text
layer.** The minus glyph extracts as the character `2`. socr faithfully reproduced a corrupt
source.

**19 corrupted tokens on that page alone.** A signature scan across the 40 papers (`>=5`
tokens matching `2\d\.\d{2}` on a page that also carries `>=5` parenthesised standard errors)
finds **8 pages, all in that one paper** — roughly 155 corrupted tokens. So the fault is a
document-level font property: rare across documents, near-total within an affected one.

What fires on that page:

| signal | fires? |
|---|---|
| `has_encoding_hygiene_suspect` (#136) | **False** |
| `needs_ocr_enhancement` | **False** |
| `has_unverifiable_table_region` (TR-3) | **False** |
| `native_table_structure_defective` (B1) | **True** |

TR-3 cannot see it by construction: it compares the emitted numeric multiset against the
native one, and both carry the same corrupt values. `state.py`'s comment on
`has_encoding_hygiene_suspect` says digit corruption "is routed to OCR at detection" — on
this page it was not.

A negative coefficient reported as `+20.12` is the worst possible outcome for a citation
corpus: not a missing number, not an obviously broken one, but a plausible wrong one. The
only reason this page is flagged at all is B1's shape gate firing for an unrelated reason —
which is a second, unplanned argument for having shipped it.
