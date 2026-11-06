"""#1269: proximity clustering on a grid must equal the old pairwise loops, at far lower cost.

The two reference functions below are today's pairwise implementations copied
verbatim (from ``tables/reconstruct._union_find_clusters`` and
``figures/extractor._cluster_drawings``).
"""

from __future__ import annotations

import math
import random

import fitz
import pytest

from socr.core import box_clusters
from socr.figures.extractor import _cluster_drawings
from socr.tables import reconstruct as rc


def _reference_union_find_clusters(boxes, gap):
    if not boxes:
        return []
    n = len(boxes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        ax0, ay0, ax1, ay1 = boxes[i]
        for j in range(i + 1, n):
            bx0, by0, bx1, by1 = boxes[j]
            h_gap = max(0.0, bx0 - ax1) if ax1 < bx0 else max(0.0, ax0 - bx1)
            v_gap = max(0.0, by0 - ay1) if ay1 < by0 else max(0.0, ay0 - by1)
            if h_gap <= gap and v_gap <= gap:
                union(i, j)

    clusters = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)

    merged = []
    for indices in clusters.values():
        x0 = min(boxes[i][0] for i in indices)
        y0 = min(boxes[i][1] for i in indices)
        x1 = max(boxes[i][2] for i in indices)
        y1 = max(boxes[i][3] for i in indices)
        merged.append((x0, y0, x1, y1))
    return merged


def _reference_cluster_drawings(drawings, page_width, page_height, cluster_gap):
    if not drawings:
        return []

    boxes = []
    for d in drawings:
        rect = d.get("rect")
        boxes.append((rect.x0, rect.y0, rect.x1, rect.y1) if rect else None)

    valid = [(i, boxes[i]) for i in range(len(boxes)) if boxes[i] is not None]
    if not valid:
        return []

    parent = {i: i for i, _ in valid}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, (idx_i, box_i) in enumerate(valid):
        for j, (idx_j, box_j) in enumerate(valid):
            if i >= j:
                continue
            x0_i, y0_i, x1_i, y1_i = box_i
            x0_j, y0_j, x1_j, y1_j = box_j

            h_gap = (
                max(0, x0_j - x1_i) if x1_i < x0_j else max(0, x0_i - x1_j) if x1_j < x0_i else 0
            )
            v_gap = (
                max(0, y0_j - y1_i) if y1_i < y0_j else max(0, y0_i - y1_j) if y1_j < y0_i else 0
            )

            if h_gap <= cluster_gap and v_gap <= cluster_gap:
                union(idx_i, idx_j)

    clusters = {}
    for idx, _ in valid:
        root = find(idx)
        clusters.setdefault(root, []).append(idx)

    results = []
    for indices in clusters.values():
        cboxes = [boxes[i] for i in indices if boxes[i] is not None]
        if not cboxes:
            continue
        x0 = min(b[0] for b in cboxes)
        y0 = min(b[1] for b in cboxes)
        x1 = max(b[2] for b in cboxes)
        y1 = max(b[3] for b in cboxes)
        results.append(([drawings[i] for i in indices], (x0, y0, x1, y1)))

    results.sort(key=lambda r: (r[1][1], r[1][0]))
    return results


def _random_boxes(rng, gap):
    boxes = []
    for _ in range(rng.randint(0, 120)):
        x0, y0 = rng.uniform(-50, 650), rng.uniform(-50, 850)
        boxes.append(
            (
                x0,
                y0,
                x0 + rng.choice([0.0, rng.uniform(0, 5), rng.uniform(0, 200), 800.0]),
                y0 + rng.choice([0.0, rng.uniform(0, 5), rng.uniform(0, 200), 1000.0]),
            )
        )
    if boxes and rng.random() < 0.5:
        a = boxes[0]
        # neighbours at exactly the gap on each axis, and a duplicate
        boxes.append((a[2] + gap, a[1], a[2] + gap + 1, a[3]))
        boxes.append((a[0], a[3] + gap, a[2], a[3] + gap + 2))
        boxes.append((a[0] - gap - 1, a[1], a[0] - gap, a[3]))
        boxes.append(a)
    return boxes


GAPS = [30.0, 10.0, 3.0, 0.0]


@pytest.mark.parametrize("gap", GAPS)
def test_union_find_clusters_equals_pairwise_reference(gap):
    rng = random.Random(1269 + int(gap))
    for _ in range(150):
        boxes = _random_boxes(rng, gap)
        assert rc._union_find_clusters(boxes, gap) == _reference_union_find_clusters(boxes, gap)


@pytest.mark.parametrize("gap", GAPS)
def test_cluster_drawings_equals_pairwise_reference(gap):
    rng = random.Random(7 + int(gap))
    for _ in range(100):
        drawings = []
        for b in _random_boxes(rng, gap):
            drawings.append({"rect": fitz.Rect(*b)})
            if rng.random() < 0.1:
                drawings.append({"rect": None})
        got = _cluster_drawings(drawings, 612.0, 792.0, gap)
        want = _reference_cluster_drawings(drawings, 612.0, 792.0, gap)
        assert got == want


def _sliver_layout(n_slivers, bands=4):
    """Columns of 0.02 pt-wide, 7.5 pt-tall slivers (stacked bands), plus frames and axes."""
    per_band = n_slivers // bands
    boxes = []
    for b in range(bands):
        y0 = 100.0 + b * 7.5
        for k in range(per_band):
            x0 = 50.0 + k * 0.02
            boxes.append((x0, y0, x0 + 0.02, y0 + 7.5))
    width = per_band * 0.02
    boxes.append((50.0, 100.0, 50.0 + width, 100.0 + bands * 7.5))  # frame
    boxes.append((49.5, 100.0, 50.0, 100.0 + bands * 7.5))  # y axis
    boxes.append((50.0, 100.0 + bands * 7.5, 50.0 + width, 100.0 + bands * 7.5 + 0.5))  # x axis
    boxes.append((300.0, 500.0, 300.5, 520.0))  # an isolated mark
    return boxes


def _count_pair_checks(monkeypatch, fn, *args):
    calls = [0]
    real = box_clusters._within_gap

    def counting(a, b, gap):
        calls[0] += 1
        return real(a, b, gap)

    monkeypatch.setattr(box_clusters, "_within_gap", counting)
    out = fn(*args)
    return out, calls[0]


@pytest.mark.parametrize("gap", [30.0, 3.0])
def test_pair_checks_are_orders_of_magnitude_below_pairwise(monkeypatch, gap):
    big = _sliver_layout(20000)
    n = len(big)
    pairwise = n * (n - 1) // 2
    got, checks = _count_pair_checks(monkeypatch, box_clusters.cluster_box_indices, big, gap)
    assert checks * 100 < pairwise, (checks, pairwise)
    # The pairwise reference on 20k boxes is ~2e8 comparisons, too slow for a unit
    # test; equality is asserted on a smaller version of the same layout (n ~ 1.2k).
    small = _sliver_layout(1200)
    want = _reference_union_find_clusters(small, gap)
    assert rc._union_find_clusters(small, gap) == want
    # And the big layout resolves to the expected structure: one chart, one isolated mark.
    assert len(got) == 2
    assert sorted(len(g) for g in got) == [1, n - 1]


def test_large_box_is_checked_exactly_not_registered(monkeypatch):
    # gap 3 -> 1.5 pt cells; a page-sized frame spans far more cells than boxes.
    boxes = [(0.0, 0.0, 600.0, 800.0), (603.0, 10.0, 604.0, 11.0), (607.1, 10.0, 608.0, 11.0)]
    assert box_clusters.cluster_box_indices(boxes, 3.0) == [[0, 1], [2]]
    assert rc._union_find_clusters(boxes, 3.0) == _reference_union_find_clusters(boxes, 3.0)


def test_cell_boundary_neighbours_at_exactly_gap():
    # Boxes straddling cell edges, exactly gap apart on x, y and diagonally.
    for gap in (30.0, 10.0, 3.0):
        cell = gap / 2
        boxes = [
            (cell - 0.01, cell - 0.01, cell, cell),
            (cell - 0.01 + gap + 0.01, cell - 0.01, cell + gap + 0.01, cell),
            (cell - 0.01, cell + gap, cell, cell + gap + 0.01),
            (5 * cell, 5 * cell, 5 * cell, 5 * cell),
        ]
        assert box_clusters.cluster_box_indices(boxes, gap) == _groups(
            _reference_union_find_clusters, boxes, gap
        )


def _groups(_ref, boxes, gap):
    # Recover index groups from the reference by pairwise reachability.
    n = len(boxes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if box_clusters._within_gap(boxes[i], boxes[j], gap):
                parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=lambda g: g[0])


def _build_chart_page():
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    # Two separated area-chart blocks of thin filled slivers, plus a framed axis.
    for x0, y0 in ((60.0, 100.0), (60.0, 450.0)):
        for k in range(1600):
            x = x0 + k * 0.1
            page.draw_rect(fitz.Rect(x, y0, x + 0.05, y0 + 100.0), color=None, fill=(0.2, 0.4, 0.8))
        page.draw_rect(fitz.Rect(x0 - 2, y0 - 2, x0 + 162, y0 + 102), color=(0, 0, 0), width=0.5)
    page.insert_text((40, 95), "Percent", fontsize=8)
    page.insert_text((40, 445), "Index", fontsize=8)
    return page


def test_chart_region_bboxes_end_to_end_equals_reference(monkeypatch):
    page = _build_chart_page()
    new = rc.chart_region_bboxes(page)
    monkeypatch.setattr(rc, "_union_find_clusters", _reference_union_find_clusters)
    old = rc.chart_region_bboxes(page)
    assert new == old
    assert len(new) == 2


def test_float_division_at_cell_boundary_regression():
    # floor(14.999999999999998 / 15) == 0 but floor(45.0 / 15) == 3: three cells apart
    # although exactly gap apart, so a reach of 2 missed the pair.
    a = (9.999999999999998, 0.0, 14.999999999999998, 1.0)
    b = (45.0, 0.0, 50.0, 1.0)
    assert box_clusters._within_gap(a, b, 30.0)
    assert box_clusters.cluster_box_indices([a, b], 30.0) == [[0, 1]]
    assert rc._union_find_clusters([a, b], 30.0) == _reference_union_find_clusters([a, b], 30.0)


@pytest.mark.parametrize("gap", [30.0, 10.0, 3.0])
def test_exact_gap_pairs_just_below_cell_edges_match_reference(gap):
    rng = random.Random(int(gap) * 31)
    cell = gap / 2.0
    boxes_list = []
    for k in range(1, 200):
        edge = k * cell
        tiny = [math.nextafter(edge, -math.inf), edge - rng.uniform(0, 1e-9), edge - 1e-12]
        for x1 in tiny:
            for horizontal in (True, False):
                if horizontal:
                    a = (x1 - 5.0, 0.0, x1, 1.0)
                    b = (x1 + gap, 0.0, x1 + gap + 5.0, 1.0)
                else:
                    a = (0.0, x1 - 5.0, 1.0, x1)
                    b = (0.0, x1 + gap, 1.0, x1 + gap + 5.0)
                boxes_list.append([a, b])
    for boxes in boxes_list:
        want = _groups(_reference_union_find_clusters, boxes, gap)
        assert box_clusters.cluster_box_indices(boxes, gap) == want, (boxes, gap)


def _assert_equals_reference(boxes, gap):
    assert box_clusters.cluster_box_indices(boxes, gap) == _groups(
        _reference_union_find_clusters, boxes, gap
    ), (boxes, gap)
    assert rc._union_find_clusters(boxes, gap) == _reference_union_find_clusters(boxes, gap)


def test_inverted_box_matches_reference():
    _assert_equals_reference([(15.0, 0.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0)], 30.0)
    _assert_equals_reference([(0.0, 15.0, 1.0, 0.0), (0.0, 0.0, 1.0, 1.0)], 30.0)


def test_huge_magnitude_matches_reference():
    a, b = -7.205759403792794e16, -7.205759403792793e16
    _assert_equals_reference([(a, 0.0, a, 0.0), (b, 0.0, b, 0.0)], 3.0)
    a, b = 1.152921504606847e18, 1.1529215046068468e18
    _assert_equals_reference([(a, 0.0, a, 0.0), (b, 0.0, b, 0.0)], 30.0)


def test_overflow_and_underflow_inputs_match_reference():
    _assert_equals_reference([(0.0, 0.0, 1e21, 1.0), (1.0, 0.0, 2.0, 1.0)], 30.0)
    _assert_equals_reference([(0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0)], 5e-324)


def _cubic_layout(n):
    return [(0.0, 0.0, (n - 1) * 1.5, 0.1)] * (n // 2) + [(0.0, 4.6, (n - 1) * 1.5, 4.7)] * (n // 2)


def test_failed_cross_cell_products_are_budgeted(monkeypatch):
    n = 160
    boxes = _cubic_layout(n)
    got, checks = _count_pair_checks(monkeypatch, box_clusters.cluster_box_indices, boxes, 3.0)
    assert checks <= 2 * (n * (n - 1) // 2), checks
    assert got == _groups(_reference_union_find_clusters, boxes, 3.0)
    assert len(got) == 2
