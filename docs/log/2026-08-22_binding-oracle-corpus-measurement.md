# Binding oracle corpus measurement after GH-273

2026-08-22. Replay of the pure, still-unwired binding oracle against the saved
2026-08-21 lane-comparison campaign. The code under test is
`fix/273-row-label-binding@b183ed0`; its parent, `main@6bf59b9`, is the paired baseline.

**Decision: do not wire this oracle into winner selection.** The GH-273 change closes the
unparseable-input fail-open, but the oracle produced no fully checked binding on any saved
real table candidate. Its corpus contradictions therefore cannot yet be interpreted as
accurate accept/reject decisions. The known Cochrane–Piazzesi page 10 shifted-label case was
unverifiable for both row and column binding and produced no row-label contradiction.

The page content and candidate markdown are not committed. The papers are copyrighted.
This record contains only the method, identifiers, aggregate counts, and content-free
diagnostics.

## Inputs and isolation

- Same deterministic 21-page, 9-paper manifest as
  `2026-08-20_lane-comparison-manifest.json`.
- Saved campaign: the `main@7c7f174` rerun documented in
  `2026-08-21_lane-comparison-after-s1.md`.
- The campaign contained 25 candidates on 21 pages. The binding oracle applies to the 15
  table pages: 17 saved candidates (7 Gemini, 7 Qwen, 2 native, 1 Nougat).
- The replay made no model calls. It reused each excerpt PDF's native word layer and the
  candidate text already saved outside the repository.
- Parent and fixed source trees were frozen into separate temporary directories. Each run
  asserted that `socr.tables.binding.__file__` resolved inside the requested frozen tree.
- Baseline source: `6bf59b9`; fixed source: `b183ed0`.
- The replay was run twice byte-for-byte. Both baseline JSON outputs matched, and both fixed
  JSON outputs matched.
- Content-safe result hashes:
  - baseline: `8bbbfddf871e8a3e1b554773f2761abf080f771d6a81530e22bc91731d40f9dc`
  - fixed: `f739498320c9021ce5902c775279bef5bf495d8a7230fae8b297212ad2bf4c90`

## Literal whole-page results

The public `bind(words, markdown)` entry point was called with
`page.get_text("words")`, as its module contract states.

Fixed result over all 17 table candidates:

- 0 pass
- 14 contradiction
- 3 unverifiable

The split is mechanically reproducible but **is not an accuracy estimate**:

- 14 candidates contained a strict parseable grid.
- 0 of 14 had fully checked geometry.
- 0 of 14 had verifiable row binding.
- 0 of 14 had verifiable column binding.
- All 14 were labelled contradictions only because other unbound-content diagnostics fired
  while both binding dimensions were incomplete.
- 7 of 14 acquired one or more row-label-contradiction diagnostics, but every one also had
  incomplete row and column binding. Those seven diagnostics have not been classified as
  true or false and must not be used as a precision claim.
- 3 candidates were not strict parseable grids. The parent incorrectly classified all three
  as passes; `b183ed0` classifies all three as unverifiable. This is the only corpus-level
  disposition change caused by the patch.

## Known GH-273 page

Cochrane–Piazzesi page 10, Qwen candidate:

- strict grid parsed: yes
- structural agreement: false before and after
- row binding: unverifiable
- column binding: unverifiable
- matched cells: 0
- row-label contradictions: 0
- native-unbound cells: 61

Thus this replay does **not** demonstrate that the new label check catches the measured
real failure. The candidate already failed for broad page-scope geometry, and the row-label
comparison had no complete numeric row binding on which to operate. The synthetic paired
GH-273 regression remains load-bearing evidence for the narrow code fix; it is not a
substitute for corpus precision or coverage.

## Localization sensitivity check

A second, exploratory replay scoped native words to every region returned by the existing
`socr.tables.locate.locate_tables` vector locator. It deliberately evaluated every located
candidate-region pair rather than selecting a visually favorable region.

- 13 of 15 table pages had at least one located region.
- 14 regions yielded 15 candidate-region bindings.
- Fixed result: 0 pass, 10 contradiction, 5 unverifiable.
- Fully checked bindings: 0.
- The known GH-273 page remained row- and column-unverifiable and still produced no
  row-label contradiction.

These counts are a sensitivity check, not the primary measurement. The locator does not
pair a model grid with its corresponding native table, and its own documentation says
multi-table pages can over-merge. The zero-coverage result nevertheless shows that merely
putting the current locator in front of the binder does not make it ready to wire.

## Interpretation

The patch did what GH-273 required at the pure-oracle boundary:

- malformed or empty bindings no longer pass;
- a candidate label is checked only after numeric/order row anchoring;
- label mismatches can now prevent structural agreement;
- the oracle remains unwired.

The corpus measurement also found the next blocker: **candidate-to-native-table scoping and
coverage must be solved and remeasured before runtime disposition is designed.** Until at
least some known-good real grids become fully checked and the known shifted-label page is
specifically reached by the label check, pass/contradiction rates cannot estimate false
accepts or false rejects.

No winner-selection code changed, and no corpus content entered Git.
