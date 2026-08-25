"""GH-181 regression for recursion-safe vector drawing clustering."""

from __future__ import annotations

import sys
from types import SimpleNamespace

from socr.figures.extractor import _cluster_drawings


def test_long_same_cluster_chain_does_not_recurse_in_find() -> None:
    """A union-find chain must remain usable beyond the active recursion limit."""
    fixture_box_width = 1.0
    fixture_box_gap = 1.0
    fixture_cluster_gap = 1.0
    fixture_x0 = 10.0
    fixture_y0 = 20.0
    fixture_y1 = 30.0
    drawing_count = 3 * sys.getrecursionlimit()
    fixture_page_width = fixture_x0 + drawing_count * (fixture_box_width + fixture_box_gap)
    fixture_page_height = fixture_y1 + 1.0

    drawings = [
        {
            "member": member,
            "rect": SimpleNamespace(
                x0=fixture_x0 + member * (fixture_box_width + fixture_box_gap),
                y0=fixture_y0,
                x1=fixture_x0 + member * (fixture_box_width + fixture_box_gap) + fixture_box_width,
                y1=fixture_y1,
            ),
        }
        for member in range(drawing_count)
    ]

    clusters = _cluster_drawings(
        drawings,
        fixture_page_width,
        fixture_page_height,
        fixture_cluster_gap,
    )

    assert len(clusters) == 1
    members, bbox = clusters[0]
    assert [drawing["member"] for drawing in members] == list(range(drawing_count))
    assert len(members) == drawing_count
    fixture_last_x1 = (
        fixture_x0 + (drawing_count - 1) * (fixture_box_width + fixture_box_gap) + fixture_box_width
    )
    assert bbox == (fixture_x0, fixture_y0, fixture_last_x1, fixture_y1)
