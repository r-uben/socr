# Native-cell advisory quiet side effects (follow-up to #1206)

When `native_cell_gating` is **off** (default):

- `tables_trust.json` records native-cell outcomes under `native_cell_advisory` only; `untrusted_pages` / `counts_by_kind` exclude them.
- Document metadata must not gain the GH-95 `untrusted tables` note from advisory native-cell flags alone.
- GH-993 `verified_text` ignores advisory native-cell reasons (PLFSS p.38 stays verified).
- `review.html` does not treat `native_cell_check_*` audit events as suspect signals on SUCCESS pages.

Fail-closed gating (`--native-cell-gating`) is **not recommended** for French budget documents (RESF / PLFSS style) until the per-cell binding redesign lands; header-label ambiguity is expected there.
