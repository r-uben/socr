"""Shared column-gutter detection (GH-152 yardstick).

Used by table reconstruction and Gemma layout-aware crops so the two paths
cannot drift apart.
"""

from __future__ import annotations

import statistics
from collections import defaultdict

from socr.core.born_digital import ALIGNED_RUN_GAP_MAX_WORD_SPACES


def median_word_gap(words: list) -> float | None:
    """Median horizontal gap between words on the same PyMuPDF text line."""
    by_line: dict[tuple, list] = defaultdict(list)
    for w in words:
        by_line[(w[5], w[6])].append(w)

    gaps: list[float] = []
    for line_words in by_line.values():
        ordered = sorted(line_words, key=lambda w: w[0])
        for a, b in zip(ordered, ordered[1:]):
            gap = b[0] - a[2]
            if gap > 0:
                gaps.append(gap)

    if not gaps:
        return None
    return statistics.median(gaps)


def detect_column_gutter(words: list) -> float | None:
    """Return the x-coordinate of a single vertical gutter, or ``None``."""
    if len(words) < 2:
        return None
    word_gap = median_word_gap(words)
    if word_gap is None:
        return None
    min_gutter_pt = ALIGNED_RUN_GAP_MAX_WORD_SPACES * word_gap

    intervals = sorted((w[0], w[2]) for w in words)
    merged: list[list[float]] = []
    for x0, x1 in intervals:
        if merged and x0 <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], x1)
        else:
            merged.append([x0, x1])

    gutters = [
        (merged[i][1], merged[i + 1][0])
        for i in range(len(merged) - 1)
        if merged[i + 1][0] - merged[i][1] >= min_gutter_pt
    ]
    if len(gutters) != 1:
        return None
    gap_start, gap_end = gutters[0]
    return (gap_start + gap_end) / 2.0
