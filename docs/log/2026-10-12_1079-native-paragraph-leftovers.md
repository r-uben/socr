# #1079 — leftovers from the #1077 native paragraph reflow

Worktree `fix/issue-1079`. Measurements: `scratch/routing-baseline/exec/1079/measure_1079.py`
(gitignored). Caller is the native-text path the emit seam uses:
`get_text("text")` + `clean_native_text` + `page_paragraphs` + `reflow_native_prose`.
`extract_structured` was tried first and spent minutes per document inside table
reconstruction; it is not in these numbers. `socr.__file__` was the worktree
`src/socr/__init__.py`.

## Design note §3 vs code

The note was wrong. B' item 3 still said "never merge … across a font/style change".
Shipped `boundary()` only treats a font change as a boundary when `a` ends short of
`R_env` (a heading). Review round 1 of #1074 measured that: the any-font reading split
Ayivodji p7 mid-sentence italics. That is an owner-accepted refinement of B', not a
code bug. The design note now states the short-line form. First-slice item 5 (inline
`$` is not a marker) was also stale after round 4 and is updated.

## Hyphen witness: document-wide, pinned, not changed

`native_vocabulary` joins every born-digital page's analyze-time native text
(`DocumentState.native_vocabulary`). A joined form anywhere rejoins every line-end
split of that word.

Trade-off: a reference title or another section that spells the word unhyphenated
will rejoin a split even when that occurrence is a different sense. Scoping the
witness to the page would stop that, and would also stop the common case (the word
appears whole on the next page). Test
`test_hyphen_witness_is_document_wide_not_page_local` fails if `native_vocabulary`
is built from one page.

## Unmatched neighbour: no blank line, pinned

`_reflow_once` inserts `\n\n` only when two matched spans are adjacent
(`last_end == start`). An unmatched printed line keeps the adjacency it already had.
Test `test_unmatched_neighbour_gets_no_blank_line` fails if the merge branch
inserts a separator after every matched span.

## Measurement table at the native-text caller

All three PDFs opened (0 timeouts). Every page had enough native text to count.
These counts include pages `process()` would escalate off the native lane, so they
are not the first-slice "native-both" column.

| doc | pages | native pages | pages changed | paras/page before -> after | hyphens removed |
|---|---|---|---|---|---|
| Ayivodji (SSRN) | 69 | 69 | 69 | 1.0 -> 12.0 | 169 |
| Ozdagli-Weber (NBER) | 63 | 63 | 63 | 1.0 -> 15.1 | 1 |
| Nakamura-Steinsson (QJE) | 48 | 48 | 48 | 1.0 -> 13.5 | 201 |

Ozdagli-Weber's single removed hyphen and Nakamura-Steinsson's native-page /
pages-changed counts are the holes the issue named. They are filled here.

## Extra `get_text` cost (wall seconds, one walk each)

`page_paragraphs` adds `get_text("dict")` + `get_text("words")` per page.

| doc | pages | without reflow (`text` only) | extra geometry `get_text` | reflow itself | with reflow |
|---|---|---|---|---|---|
| Ayivodji | 69 | 0.155 | 0.651 | 0.131 | 0.937 |
| Ozdagli-Weber | 63 | 0.112 | 0.479 | 0.116 | 0.708 |
| Nakamura-Steinsson | 48 | 0.124 | 0.537 | 0.115 | 0.776 |

The extra geometry passes are about 4× the `text` extraction on these three.
Not measured: `process()` / analyze-time `detect()`, or a full-corpus wall clock.

## Currency `$` over-protection

30 documents that contain `$` on a native page (witness-pdfs first, then iCloud
`Papers/papers`; 62 PDFs scanned to find 30; 0 open timeouts). 1639 native pages
in those 30.

| | n | share of `$` pages |
|---|---|---|
| pages with any `$` | 204 | 1 |
| unbalanced `$` + digit-adjacent (currency shape) | 66 | 0.324 |
| of those, `_protected_lines` covers ≥50% of nonempty lines | 39 | 0.191 |
| reflow differs from a counterfactual that does not treat single `$` as a marker | 35 | 0.172 |

So 35/204 `$` pages (17%), 35/1639 native pages in the sample (2.1%), have most of
the page's reflow switched off by an unbalanced `$` that sits next to a digit.

Examples (currency, not `$$` / `\[`): Christiano–Eichenbaum–Evans 1994 pp. 40, 41,
46, 51 (one `$` each, `mathish=false`, prot_share 0.50–0.80);
Stephens–Donnelly 2000 pp. 1, 9, 18 (many `$`, `mathish=false`, prot_share
0.85–1.0). Forney 1988 p. 20 is over-protected but also `mathish=true` — counted,
not a clean currency example.

Not measured: whether those `$` are dollars of money vs a lone math delimiter;
the digit-adjacent test is the proxy. The rule is unchanged (fail-closed).
