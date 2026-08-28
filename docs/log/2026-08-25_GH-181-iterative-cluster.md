# 2026-08-25 — GH-181: iterative drawing clustering and audited chart detection

## Decision

GH-181 addresses two connected failure surfaces in the chart-asset pre-scan:

- [`GH-181`](https://github.com/r-uben/socr/issues/181) — recursive union-find
  lookup in `_cluster_drawings` could exhaust Python's recursion depth on a
  long same-cluster chain.
- The chart-eligibility predicate could catch that failure and silently turn it
  into `False`, making the page look like an ordinary non-chart page.

The fix is deliberately narrow. `_cluster_drawings.find()` now uses iterative
path halving:

```python
while parent[x] != x:
    parent[x] = parent[parent[x]]
    x = parent[x]
```

The existing union direction, valid-drawing filtering, member ordering, bbox
calculation, and final cluster ordering are unchanged. The chart detector
predicate no longer owns failure handling. `_phase_agentic`, the state-owning
document orchestration boundary, catches `Exception` around the eligibility
call, emits one `WARNING` with traceback context, appends one durable
`AuditEvent(kind="chart_asset_detection_failed", engine="chart_asset")`, and
continues through the page's existing non-chart route. The event stores the
exception type and message and survives run-audit serialization.

## Pre-fix canaries

The canaries were run before the source edits and failed for the intended
reasons, without a provider:

1. `tests/test_cluster_drawings_gh181.py` failed with
   `RecursionError: maximum recursion depth exceeded`. The traceback terminated
   in the nested recursive `find()` call, after the chain had already been
   constructed and the clustering assertions were reached.
2. The chart-lane failure canary preserved the real routed table
   `PageOutput` and called the normal route once, but its assertion found zero
   `WARNING` records. The old helper caught the detector exception at `DEBUG`
   level and returned `False`; it appended no durable audit event.

The first failure is the stack-depth defect. The second confirms that the
silent conversion at the helper boundary was independently observable from the
fallback route itself.

## Independent root-cause review

An independent vendor review confirmed that `union()` attaches the root of one
component to the root of the other without rank/size balancing, so a chain of
adjacent drawings can create a deep parent path. It found iterative root lookup
with path compression sufficient to remove the recursion ceiling; union by rank
or size was not required and would add unrelated behavior. It also confirmed
that `_is_chart_asset_page` was the wrong place for audit state: its caller owns
`DocumentState.events`, so the caller is the single boundary that can preserve
fail-safe routing while guaranteeing one warning and one durable event.

## Deliberate scope boundaries

- No parent-rank or parent-size heuristic was added.
- No drawing-count cap or other magic threshold was added.
- The existing `O(N^2)` pairwise clustering scan is unchanged; this fix removes
  recursion from root lookup, not the clustering algorithm's scan complexity.
- The broad `Exception` catch is retained only at the document orchestration
  boundary. Every caught detector failure is both logged at `WARNING` and
  appended as an `AuditEvent`; the predicate itself does not silently default.
- A detector failure alone does not demote an otherwise successful fallback
  page or document. The page continues through its established non-chart route;
  the audit event supplies observability without changing winner selection.

The detector work is related to the thin-stroke chart recognition in
[`GH-150`](https://github.com/r-uben/socr/issues/150), but it does not alter the
GH-150 detector thresholds or frame/data-mark policy. See the prior decision in
[`docs/log/2026-08-12_GH-150-A1.md`](2026-08-12_GH-150-A1.md).

## Post-fix verification

All requested commands were run on `fix/181-iterative-cluster` with no provider:

| Command | Observed result |
|---|---|
| `uv run pytest tests/test_cluster_drawings_gh181.py tests/test_chart_detection_gh150.py tests/test_chart_lane.py tests/test_figure_pass.py tests/test_agentic_figures.py tests/test_pp4_inline_figures.py -q` | **82 passed**, 5 warnings, 81.32 s |
| `uv run ruff format --check .` | **Passed**; 354 files already formatted |
| `uv run pytest -m "not integration" -q` | **2,222 passed**, 3 xfailed, 7 warnings, 130.67 s |

The targeted regression therefore covers both the former recursion failure and
the durable warning/audit boundary, while the non-integration repository gate
and formatting gate remain green.
