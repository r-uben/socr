"""Build markdown table transcriptions from born-digital native word geometry."""

from __future__ import annotations

from socr.tables.binding import (
    _coalesce_numeric_word_fragments,
    _native_lane_geometry,
    _native_rows,
    _partition_words_by_region,
    _year_header_lane_centers,
)


def _data_column_count(words: list) -> int:
    year = _year_header_lane_centers(words)
    if year:
        return len(year)
    lc, _, _ = _native_lane_geometry(words, None)
    return max(lc, 1)


def markdown_from_native_words(
    words: list,
    region: tuple[float, float, float, float],
    *,
    spans: list | None = None,
) -> str:
    """Golden transcription: row labels and numeric cells from the PDF text layer."""
    scoped, _, _ = _partition_words_by_region(words, region)
    scoped = _coalesce_numeric_word_fragments(scoped)
    n_data_cols = _data_column_count(scoped)
    native_rows, _, _header_band_idxs = _native_rows(scoped, n_data_cols, spans)

    header = [""] + [""] * n_data_cols
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in range(n_data_cols + 1)) + " |",
    ]
    for nr in native_rows:
        if nr.is_parent or not nr.lane_tokens:
            continue
        label = nr.row_path[-1].strip() if nr.row_path else ""
        cells = [label]
        for lane in range(n_data_cols):
            tok, _amb = nr.lane_tokens.get(lane, ("", False))
            cells.append(tok if tok else "-")
        if not any(c.strip() and c.strip() != "-" for c in cells[1:]):
            continue
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
