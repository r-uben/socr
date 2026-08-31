# 2026-08-31 — GH-375: D3 regional splice vs header-unattributed sibling

## Decision

When `native_table_header_unattributed` is set there is no per-table header
identity, so ordinal-based regional splice is refused. The page ships the GH-90
all-table replacement (every `find_table_blocks` occurrence becomes a marker)
instead of a whole-page wipe.

Header-unattributed is a table defect. The flag is produced by walking
`find_table_blocks` on the same native text the floor later splices, so the
damaged grid that set the flag is in that list. Surrounding prose is not
implicated. Whole-page would re-lose the GH-371 prose for a table-level hole.

## Equal-count identity

Geometry-only splice (no header flag) still uses TR-3 ordinals. Count equality
cannot see a permutation of y0-sorted regions vs parsed blocks. `_verify_regions`
now records `table_grid_identity` per separator-bearing region; a mismatch
fails the splice closed to the whole-page marker. Absent identities (old
sidecars, constructed GH-371 fixtures) keep the count-only path.

## Not done

`process()` is not driven here: `--native-only` never sets
`native_table_structure_failed`, so D3 does not fire, and a non-native-only
`process()` needs a provider ladder. The ship seam is `_winning_page_output`
plus `_phase_assemble` / `canonical_page_texts`.
