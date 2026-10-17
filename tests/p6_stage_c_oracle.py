"""Shared exact-difference oracle for P6 stage C.

This is a non-collected support module (named with a non-test prefix so pytest
does not collect it directly) containing:
1. Named corpus labels and constants from the 12-page P6 corpus fixture.
2. Canonical capture normalization (excising only VOLATILE_KEYS).
3. A recursive leaf-path difference utility and patch applicator.
4. The authoritative EXPECTED_STAGE_C_DIFFERENCES mapping and granular difference
   inventories.

Stage C migrates three assemble buckets (structure_class_model_pages,
structure_class_floor_pages, corrupt_math_hybrid_pages) from private
SelectionProvenance membership to exact public PageDisposition equality.
When the post-selection emission guard rewrites a candidate to a fail-closed marker:
- It leaves the migrated bucket (final disposition != migrated pair).
- It no longer emits the stale shipped audit kinds (structure_class_model_table_kept,
  corrupt_math_hybrid_shipped).
- It drops from the respective CLI bucket line and audit log count.
- It updates the corrupt-math error note.
- Derivatives: removing structure_class_model_table_kept on page 9 updates
  tables_trust.json (flags count, page 9 reasons/details, kind counts), the
  table-trust CLI summary line, and the table-trust error note in result_error
  and metadata.json.

All unaffected surfaces (document status, result status, result audit_passed,
all winning output bytes, markdown fragments, final .md, manifest entries,
flag-derived buckets, and orthogonal buckets) remain byte-identical.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from p6_corpus_fixture import (
    CLEAN_BORN_DIGITAL_PAGE,
    COLD_REVIEW_SHAPE_ONE_PAGE,
    COLD_REVIEW_SHAPE_TWO_PAGE,
    D3_FLOOR_PAGE,
    D3_MODEL_KEPT_PAGE,
    HYBRID_CLEAN_PAGE,
    HYBRID_REWRITTEN_PAGE,
    NO_TEXT_FAILURE_PAGE,
    PAGE_COUNT,
    PAGE_LABELS,
    PASSING_MODEL_PAGE,
    STAGE_C_PAGE_COUNT,
    STAGE_C_PAGE_LABELS,
    STRUCT_FLOOR_PAGE,
    STRUCT_MODEL_PASSING_PAGE,
    STRUCT_MODEL_REWRITTEN_PAGE,
)

# ---------------------------------------------------------------------------
# Re-exported Named Corpus Labels & Inventory
# ---------------------------------------------------------------------------

__all__ = [
    "CLEAN_BORN_DIGITAL_PAGE",
    "D3_FLOOR_PAGE",
    "D3_MODEL_KEPT_PAGE",
    "COLD_REVIEW_SHAPE_ONE_PAGE",
    "COLD_REVIEW_SHAPE_TWO_PAGE",
    "NO_TEXT_FAILURE_PAGE",
    "PASSING_MODEL_PAGE",
    "STRUCT_MODEL_PASSING_PAGE",
    "STRUCT_MODEL_REWRITTEN_PAGE",
    "STRUCT_FLOOR_PAGE",
    "HYBRID_CLEAN_PAGE",
    "HYBRID_REWRITTEN_PAGE",
    "PAGE_LABELS",
    "PAGE_COUNT",
    "STAGE_C_PAGE_LABELS",
    "STAGE_C_PAGE_COUNT",
    "VOLATILE_KEYS",
    "ADDITIVE_SIDECAR_KEYS",
    "LeafPath",
    "ExpectedDiff",
    "normalize_capture",
    "compute_leaf_diff",
    "apply_stage_c_patch",
    "assert_capture_diff_matches_oracle",
    "EXPECTED_STAGE_C_DIFFERENCES",
    "EXPECTED_STAGE_C_DIFFERENCE_ENTRIES",
    "BUCKET_DIFFERENCES",
    "STATE_EVENT_DIFFERENCES",
    "SIDECAR_EVENT_DIFFERENCES",
    "AUDIT_LOG_DIFFERENCES",
    "TABLES_TRUST_DIFFERENCES",
    "CLI_DIFFERENCES",
    "RESULT_ERROR_DIFFERENCES",
    "METADATA_DIFFERENCES",
    "PAGE_TOP_MARK_1145_DIFFERENCES",
    "VERDICT_SIDECAR_1145_DIFFERENCES",
    "CHART_FLOOR_BODY_SIDECAR_1155_DIFFERENCES",
]

#: Volatile keys excluded from canonical byte comparison.
#: Narrowed per cold review round 2: only fields that vary by environmental
#: source build or PDF generation timestamps are ignored.
#:
#: ``disposition`` is deliberately NOT excluded (cold review round 1, finding 1).
#: It is the public field this stage is about, and the stage-A/B HEAD already
#: persisted it on sidecars, manifest entries and page-contract records, so the
#: baseline capture carries it and the oracle compares it like anything else.
VOLATILE_KEYS: frozenset[str] = frozenset(
    {
        "socr_source_digest",
        "run_fingerprint",
        "input_checksum",
        "pdf_file_hash",
        # #1027: additive diagnostic record of which rungs ran (sidecar only); pinned by
        # tests/test_gh1027_invisible_scan_floor.py, not by the byte-identity oracles.
        "attempts_summary",
        # #1030: additive sidecar key (empty on every page without a scanned figure).
        "scanned_figure_png_ref",
        # Additive sidecar keys, None on every page of this corpus: the superseded table
        # verdict of a marker page (tests/test_marker_page_record_drops_stale_table_verdict.py)
        # and the --retry-failed-pages record (tests/test_retry_failed_pages.py).
        "table_ladder_superseded",
        "page_retry",
    }
)

#: Sidecar keys that did not exist at the frozen pre-change baseline and do now.
#:
#: Deliberately SEPARATE from ``VOLATILE_KEYS``: a volatile key is excised from the
#: comparison entirely, whereas each key here is compared in full, value by value, through
#: a real ``ExpectedDiff`` entry per page (``VERDICT_SIDECAR_1145_DIFFERENCES``). This set
#: exists only so the "no key moved" pin can say which additions are accounted for. A key
#: added without an entry here still fails that pin, and a key here without entries still
#: fails the oracle, so neither half can be satisfied alone.
ADDITIVE_SIDECAR_KEYS: frozenset[str] = frozenset(
    {
        # #1145: the per-page verdict. Pinned per page by
        # VERDICT_SIDECAR_1145_DIFFERENCES and by tests/test_gh1145_*.py.
        "verdict",
        # #1155: whether this page's body came from the chart floor, persisted so the
        # preservation pass still skips it after a resume (repo review of PR #1190).
        # Pinned per page by CHART_FLOOR_BODY_SIDECAR_1155_DIFFERENCES -- False on every
        # page of this corpus, including the floor page, whose chart reader resolves
        # nothing -- and by tests/test_gh1155_chart_witness.py for the page where it is
        # True.
        "chart_floor_body",
    }
)

LeafPath = tuple[str | int, ...]


@dataclass(frozen=True)
class ExpectedDiff:
    """An exact expected difference at a specific leaf path."""

    path: LeafPath
    old_value: Any
    new_value: Any
    description: str = ""

    def as_pair(self) -> tuple[Any, Any]:
        return (self.old_value, self.new_value)


# ---------------------------------------------------------------------------
# Canonical Capture Normalization
# ---------------------------------------------------------------------------


def normalize_capture(obj: Any) -> Any:
    """Recursively excise VOLATILE_KEYS from dictionaries and lists."""
    if isinstance(obj, dict):
        return {k: normalize_capture(v) for k, v in obj.items() if k not in VOLATILE_KEYS}
    if isinstance(obj, list):
        return [normalize_capture(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Leaf-Path Diff Utility & Patch Applicator
# ---------------------------------------------------------------------------


def _format_path(path: LeafPath) -> str:
    """Format a tuple path as a readable dotted/bracketed key path."""
    parts = []
    for p in path:
        if isinstance(p, int):
            parts.append(f"[{p}]")
        else:
            if parts and not parts[-1].endswith("]"):
                parts.append(f".{p}")
            else:
                parts.append(str(p))
    return "".join(parts)


def compute_leaf_diff(
    old_obj: Any,
    new_obj: Any,
    path: LeafPath = (),
) -> dict[LeafPath, tuple[Any, Any]]:
    """Compute exact leaf-level differences between two normalized objects.

    Recurses down dicts (and same-length lists of dicts), recording leaf-level
    differing values as ``{path: (old_value, new_value)}``.
    """
    diffs: dict[LeafPath, tuple[Any, Any]] = {}

    if old_obj == new_obj:
        return diffs

    if isinstance(old_obj, dict) and isinstance(new_obj, dict):
        all_keys = set(old_obj.keys()) | set(new_obj.keys())
        sorted_keys = sorted(all_keys, key=lambda k: (str(type(k)), str(k)))
        for k in sorted_keys:
            sub_path = path + (k,)
            if k not in old_obj:
                diffs[sub_path] = (None, new_obj[k])
            elif k not in new_obj:
                diffs[sub_path] = (old_obj[k], None)
            else:
                sub_diffs = compute_leaf_diff(old_obj[k], new_obj[k], sub_path)
                diffs.update(sub_diffs)
        return diffs

    if (
        isinstance(old_obj, list)
        and isinstance(new_obj, list)
        and len(old_obj) == len(new_obj)
        and all(isinstance(x, dict) for x in old_obj)
        and all(isinstance(y, dict) for y in new_obj)
    ):
        for idx, (sub_old, sub_new) in enumerate(zip(old_obj, new_obj)):
            sub_diffs = compute_leaf_diff(sub_old, sub_new, path + (idx,))
            diffs.update(sub_diffs)
        return diffs

    diffs[path] = (old_obj, new_obj)
    return diffs


def apply_stage_c_patch(
    base_capture: dict[str, Any],
    differences: dict[LeafPath, tuple[Any, Any]] | None = None,
) -> dict[str, Any]:
    """Apply the expected stage C patch to a stage-A/B capture.

    Returns a new deeply-patched capture where every old value at the
    enumerated leaf path is replaced with the corresponding new value.
    Fails if any target path does not match the expected old value.
    """
    patch_mapping = differences if differences is not None else EXPECTED_STAGE_C_DIFFERENCES
    patched = copy.deepcopy(base_capture)

    for path, (expected_old, new_val) in patch_mapping.items():
        curr = patched
        for step in path[:-1]:
            if isinstance(step, int):
                curr = curr[step]
            else:
                curr = curr[step]
        last_step = path[-1]
        # An entry whose old value is None may be an ADDED key: #1145's ``verdict`` does
        # not exist on the pre-change sidecars at all, and ``compute_leaf_diff`` already
        # reports an absent key as ``(None, new)``, so the two agree. Only an absent key
        # is tolerated here -- a key that is PRESENT with an unexpected value still
        # raises, so the patch cannot paper over a changed value.
        if expected_old is None and isinstance(curr, dict) and last_step not in curr:
            curr[last_step] = copy.deepcopy(new_val)
            continue
        current_val = curr[last_step]
        if current_val != expected_old:
            raise ValueError(
                f"Patch mismatch at leaf path {_format_path(path)}: "
                f"expected old {expected_old!r}, found {current_val!r}"
            )
        curr[last_step] = copy.deepcopy(new_val)

    return patched


def assert_capture_diff_matches_oracle(
    old_capture: dict[str, Any],
    new_capture: dict[str, Any],
    expected: dict[LeafPath, tuple[Any, Any]] | None = None,
) -> None:
    """Assert that the difference between old and new captures matches the oracle exactly.

    No wildcard ignored paths are allowed. Applying the expected patch to
    old_capture must equal new_capture exactly; any unenumerated addition,
    removal, or value discrepancy fails with its leaf path.
    """
    expected_map = expected if expected is not None else EXPECTED_STAGE_C_DIFFERENCES

    norm_old = normalize_capture(old_capture)
    norm_new = normalize_capture(new_capture)

    measured_diff = compute_leaf_diff(norm_old, norm_new)

    unexpected_keys = set(measured_diff.keys()) - set(expected_map.keys())
    missing_keys = set(expected_map.keys()) - set(measured_diff.keys())

    errors: list[str] = []
    if unexpected_keys:
        for k in sorted(unexpected_keys):
            old_v, new_v = measured_diff[k]
            errors.append(
                f"Unexpected diff at {_format_path(k)}:\n  old: {old_v!r}\n  new: {new_v!r}"
            )

    if missing_keys:
        for k in sorted(missing_keys):
            exp_old, exp_new = expected_map[k]
            errors.append(
                f"Missing expected diff at {_format_path(k)}:\n"
                f"  expected old: {exp_old!r}\n"
                f"  expected new: {exp_new!r}"
            )

    mismatched_values: list[str] = []
    for k in set(measured_diff.keys()) & set(expected_map.keys()):
        measured_pair = measured_diff[k]
        expected_pair = expected_map[k]
        if measured_pair != expected_pair:
            mismatched_values.append(
                f"Value mismatch at {_format_path(k)}:\n"
                f"  expected old: {expected_pair[0]!r}\n"
                f"  measured old: {measured_pair[0]!r}\n"
                f"  expected new: {expected_pair[1]!r}\n"
                f"  measured new: {measured_pair[1]!r}"
            )

    if errors or mismatched_values:
        full_msg = ["Stage C capture delta does not match expected oracle:"]
        full_msg.extend(errors)
        full_msg.extend(mismatched_values)
        raise AssertionError("\n".join(full_msg))

    patched_old = apply_stage_c_patch(norm_old, expected_map)
    if patched_old != norm_new:
        residual_diff = compute_leaf_diff(patched_old, norm_new)
        raise AssertionError(
            f"Applying stage C patch to normalized stage-A/B capture did not equal current "
            f"capture. Residual leaf diffs ({len(residual_diff)}): "
            f"{ {_format_path(k): v for k, v in residual_diff.items()} }"
        )


# ---------------------------------------------------------------------------
# Concrete Enumerated Stage C Difference Definitions
# ---------------------------------------------------------------------------

# 1. Bucket Memberships
_OLD_STRUCT_MODEL_BUCKET = [STRUCT_MODEL_PASSING_PAGE, STRUCT_MODEL_REWRITTEN_PAGE]
_NEW_STRUCT_MODEL_BUCKET = [STRUCT_MODEL_PASSING_PAGE]

_OLD_CORRUPT_MATH_BUCKET = [HYBRID_CLEAN_PAGE, HYBRID_REWRITTEN_PAGE]
_NEW_CORRUPT_MATH_BUCKET = [HYBRID_CLEAN_PAGE]

BUCKET_DIFFERENCES: list[ExpectedDiff] = [
    ExpectedDiff(
        path=("buckets", "structure_class_model_pages"),
        old_value=_OLD_STRUCT_MODEL_BUCKET,
        new_value=_NEW_STRUCT_MODEL_BUCKET,
        description=(
            f"Page {STRUCT_MODEL_REWRITTEN_PAGE} (rewritten by emission guard to fail-closed "
            "marker) is absent from structure_class_model_pages; only passing control "
            f"page {STRUCT_MODEL_PASSING_PAGE} remains."
        ),
    ),
    ExpectedDiff(
        path=("buckets", "corrupt_math_hybrid_pages"),
        old_value=_OLD_CORRUPT_MATH_BUCKET,
        new_value=_NEW_CORRUPT_MATH_BUCKET,
        description=(
            f"Page {HYBRID_REWRITTEN_PAGE} (rewritten by emission guard to fail-closed "
            "marker) is absent from corrupt_math_hybrid_pages; only clean control "
            f"page {HYBRID_CLEAN_PAGE} remains."
        ),
    ),
]

# 2. State Events (state.events)
_EVENT_STRUCT_MODEL_KEPT_P9 = [
    STRUCT_MODEL_REWRITTEN_PAGE,
    "structure_class_model_table_kept",
    "gemini",
    (
        "structure-class page (table); native may not author the grid (C1), "
        "and a model attempt did -- the model's reading ships instead of native, "
        "flagged per its own status"
    ),
]

_EVENT_CORRUPT_MATH_SHIPPED_P12 = [
    HYBRID_REWRITTEN_PAGE,
    "corrupt_math_hybrid_shipped",
    "native+math",
    (
        "native prose plus crop-backed equation candidate(s) shipped WARNING; "
        "LaTeX passed at most a syntax gate and mathematical fidelity remains unverified"
    ),
]

_OLD_STATE_EVENTS = [
    [2, "page_failed", "", "no usable OCR output; failure marker shipped"],
    [
        2,
        "table_region_unverifiable",
        "native",
        (
            "per-region geometry verifier hard-failed (geometry_impossible_collapse) and OCR ladder"
            " also failed; D3 fail-closed: explicit failed-table marker shipped — no"
            " collapsed/ragged table emitted"
        ),
    ],
    [
        3,
        "d3_floor_model_table_kept",
        "gemini",
        (
            "the native table region failed geometry/header verification and no OCR rung was"
            " accepted, but a model attempt authored a grid; the model's reading ships FLAGGED"
            " instead of the failed-table marker — verify it against the source image before citing"
        ),
    ],
    [
        4,
        "table_region_unverifiable",
        "native",
        (
            "per-region geometry verifier hard-failed (geometry_impossible_collapse) and OCR ladder"
            " also failed; D3 fail-closed: explicit failed-table marker shipped — no"
            " collapsed/ragged table emitted"
        ),
    ],
    [
        5,
        "d3_floor_model_table_kept",
        "gemini",
        (
            "the native table region failed geometry/header verification and no OCR rung was"
            " accepted, but a model attempt authored a grid; the model's reading ships FLAGGED"
            " instead of the failed-table marker — verify it against the source image before citing"
        ),
    ],
    [6, "page_failed", "", "no usable OCR output; failure marker shipped"],
    [
        8,
        "structure_class_model_table_kept",
        "gemini",
        (
            "structure-class page (table); native may not author the grid (C1), and a model attempt"
            " did -- the model's reading ships instead of native, flagged per its own status"
        ),
    ],
    [9, "page_failed", "", "no usable OCR output; failure marker shipped"],
    _EVENT_STRUCT_MODEL_KEPT_P9,
    [
        9,
        "table_structure_failed",
        "gemini",
        "table_latex_leak defect found in exact final page body",
    ],
    [10, "page_failed", "", "no usable OCR output; failure marker shipped"],
    [
        10,
        "structure_class_ladder_exhausted_floor",
        "native",
        (
            "every usable grid candidate was refused/absent; marker plus page image was selected,"
            " and the native geometry grid was withheld (fail-closed floor)"
        ),
    ],
    [
        11,
        "corrupt_math_hybrid_shipped",
        "native+math",
        (
            "native prose plus crop-backed equation candidate(s) shipped WARNING; LaTeX passed at"
            " most a syntax gate and mathematical fidelity remains unverified"
        ),
    ],
    _EVENT_CORRUPT_MATH_SHIPPED_P12,
    [12, "page_failed", "", "no usable OCR output; failure marker shipped"],
    [
        12,
        "table_structure_failed",
        "native+math",
        "table_width_mismatch defect found in exact final page body",
    ],
]

_NEW_STATE_EVENTS = [
    ev
    for ev in _OLD_STATE_EVENTS
    if ev not in (_EVENT_STRUCT_MODEL_KEPT_P9, _EVENT_CORRUPT_MATH_SHIPPED_P12)
]

STATE_EVENT_DIFFERENCES: list[ExpectedDiff] = [
    ExpectedDiff(
        path=("events",),
        old_value=_OLD_STATE_EVENTS,
        new_value=_NEW_STATE_EVENTS,
        description=(
            f"State events omit structure_class_model_table_kept for rewritten page "
            f"{STRUCT_MODEL_REWRITTEN_PAGE} and corrupt_math_hybrid_shipped for rewritten "
            f"page {HYBRID_REWRITTEN_PAGE}."
        ),
    ),
]

# 3. Sidecars (pages/00009.json and pages/00012.json audit_events)
_P9_SIDECAR_KEY = f"p6_corpus/pages/{STRUCT_MODEL_REWRITTEN_PAGE:05d}.json"
_P12_SIDECAR_KEY = f"p6_corpus/pages/{HYBRID_REWRITTEN_PAGE:05d}.json"

_OLD_P9_AUDIT_EVENTS = [
    {
        "data": {},
        "detail": "no usable OCR output; failure marker shipped",
        "engine": "",
        "kind": "page_failed",
    },
    {
        "data": {"structure_class_model_kept": True},
        "detail": (
            "structure-class page (table); native may not author the grid (C1), "
            "and a model attempt did -- the model's reading ships instead of native, "
            "flagged per its own status"
        ),
        "engine": "gemini",
        "kind": "structure_class_model_table_kept",
    },
    {
        "data": {"defect": "table_latex_leak", "site": "final_body"},
        "detail": "table_latex_leak defect found in exact final page body",
        "engine": "gemini",
        "kind": "table_structure_failed",
    },
]

_NEW_P9_AUDIT_EVENTS = [
    {
        "data": {},
        "detail": "no usable OCR output; failure marker shipped",
        "engine": "",
        "kind": "page_failed",
    },
    {
        "data": {"defect": "table_latex_leak", "site": "final_body"},
        "detail": "table_latex_leak defect found in exact final page body",
        "engine": "gemini",
        "kind": "table_structure_failed",
    },
]

_OLD_P12_AUDIT_EVENTS = [
    {
        "data": {},
        "detail": "no usable OCR output; failure marker shipped",
        "engine": "",
        "kind": "page_failed",
    },
    {
        "data": {
            "audit_passed": False,
            "cost_usd": 0.0,
            "crop_paths": [],
            "provider_backend": "",
            "provider_id": "",
            "provider_model": "",
        },
        "detail": (
            "native prose plus crop-backed equation candidate(s) shipped WARNING; "
            "LaTeX passed at most a syntax gate and mathematical fidelity remains unverified"
        ),
        "engine": "native+math",
        "kind": "corrupt_math_hybrid_shipped",
    },
    {
        "data": {"defect": "table_width_mismatch", "site": "final_body"},
        "detail": "table_width_mismatch defect found in exact final page body",
        "engine": "native+math",
        "kind": "table_structure_failed",
    },
]

_NEW_P12_AUDIT_EVENTS = [
    {
        "data": {},
        "detail": "no usable OCR output; failure marker shipped",
        "engine": "",
        "kind": "page_failed",
    },
    {
        "data": {"defect": "table_width_mismatch", "site": "final_body"},
        "detail": "table_width_mismatch defect found in exact final page body",
        "engine": "native+math",
        "kind": "table_structure_failed",
    },
]

SIDECAR_EVENT_DIFFERENCES: list[ExpectedDiff] = [
    ExpectedDiff(
        path=("sidecars", _P9_SIDECAR_KEY, "audit_events"),
        old_value=_OLD_P9_AUDIT_EVENTS,
        new_value=_NEW_P9_AUDIT_EVENTS,
        description=(
            f"Page {STRUCT_MODEL_REWRITTEN_PAGE} sidecar audit_events drops the "
            "structure_class_model_table_kept event; page_failed and table_structure_failed "
            "remain."
        ),
    ),
    ExpectedDiff(
        path=("sidecars", _P12_SIDECAR_KEY, "audit_events"),
        old_value=_OLD_P12_AUDIT_EVENTS,
        new_value=_NEW_P12_AUDIT_EVENTS,
        description=(
            f"Page {HYBRID_REWRITTEN_PAGE} sidecar audit_events drops the "
            "corrupt_math_hybrid_shipped event; page_failed and table_structure_failed "
            "remain."
        ),
    ),
]

# 4. Audit Log (audit_log.json)
_OLD_AUDIT_LOG_EVENTS = [
    {
        "data": {"d3_floor": True},
        "detail": (
            "per-region geometry verifier hard-failed (geometry_impossible_collapse) and OCR ladder"
            " also failed; D3 fail-closed: explicit failed-table marker shipped — no"
            " collapsed/ragged table emitted"
        ),
        "engine": "native",
        "kind": "table_region_unverifiable",
        "page_num": 2,
    },
    {
        "data": {},
        "detail": "no usable OCR output; failure marker shipped",
        "engine": "",
        "kind": "page_failed",
        "page_num": 2,
    },
    {
        "data": {"d3_floor_superseded": True},
        "detail": (
            "the native table region failed geometry/header verification and no OCR rung was"
            " accepted, but a model attempt authored a grid; the model's reading ships FLAGGED"
            " instead of the failed-table marker — verify it against the source image before citing"
        ),
        "engine": "gemini",
        "kind": "d3_floor_model_table_kept",
        "page_num": 3,
    },
    {
        "data": {"d3_floor": True},
        "detail": (
            "per-region geometry verifier hard-failed (geometry_impossible_collapse) and OCR ladder"
            " also failed; D3 fail-closed: explicit failed-table marker shipped — no"
            " collapsed/ragged table emitted"
        ),
        "engine": "native",
        "kind": "table_region_unverifiable",
        "page_num": 4,
    },
    {
        "data": {"d3_floor_superseded": True},
        "detail": (
            "the native table region failed geometry/header verification and no OCR rung was"
            " accepted, but a model attempt authored a grid; the model's reading ships FLAGGED"
            " instead of the failed-table marker — verify it against the source image before citing"
        ),
        "engine": "gemini",
        "kind": "d3_floor_model_table_kept",
        "page_num": 5,
    },
    {
        "data": {},
        "detail": "no usable OCR output; failure marker shipped",
        "engine": "",
        "kind": "page_failed",
        "page_num": 6,
    },
    {
        "data": {"structure_class_model_kept": True},
        "detail": (
            "structure-class page (table); native may not author the grid (C1), and a model attempt"
            " did -- the model's reading ships instead of native, flagged per its own status"
        ),
        "engine": "gemini",
        "kind": "structure_class_model_table_kept",
        "page_num": 8,
    },
    {
        "data": {"structure_class_model_kept": True},
        "detail": (
            "structure-class page (table); native may not author the grid (C1), and a model attempt"
            " did -- the model's reading ships instead of native, flagged per its own status"
        ),
        "engine": "gemini",
        "kind": "structure_class_model_table_kept",
        "page_num": 9,
    },
    {
        "data": {},
        "detail": "no usable OCR output; failure marker shipped",
        "engine": "",
        "kind": "page_failed",
        "page_num": 9,
    },
    {
        "data": {"defect": "table_latex_leak", "site": "final_body"},
        "detail": "table_latex_leak defect found in exact final page body",
        "engine": "gemini",
        "kind": "table_structure_failed",
        "page_num": 9,
    },
    {
        "data": {"structure_class_floor": True},
        "detail": (
            "every usable grid candidate was refused/absent; marker plus page image was selected,"
            " and the native geometry grid was withheld (fail-closed floor)"
        ),
        "engine": "native",
        "kind": "structure_class_ladder_exhausted_floor",
        "page_num": 10,
    },
    {
        "data": {},
        "detail": "no usable OCR output; failure marker shipped",
        "engine": "",
        "kind": "page_failed",
        "page_num": 10,
    },
    {
        "data": {
            "audit_passed": False,
            "cost_usd": 0.0,
            "crop_paths": [],
            "provider_backend": "",
            "provider_id": "",
            "provider_model": "",
        },
        "detail": (
            "native prose plus crop-backed equation candidate(s) shipped WARNING; LaTeX passed at"
            " most a syntax gate and mathematical fidelity remains unverified"
        ),
        "engine": "native+math",
        "kind": "corrupt_math_hybrid_shipped",
        "page_num": 11,
    },
    {
        "data": {
            "audit_passed": False,
            "cost_usd": 0.0,
            "crop_paths": [],
            "provider_backend": "",
            "provider_id": "",
            "provider_model": "",
        },
        "detail": (
            "native prose plus crop-backed equation candidate(s) shipped WARNING; LaTeX passed at"
            " most a syntax gate and mathematical fidelity remains unverified"
        ),
        "engine": "native+math",
        "kind": "corrupt_math_hybrid_shipped",
        "page_num": 12,
    },
    {
        "data": {},
        "detail": "no usable OCR output; failure marker shipped",
        "engine": "",
        "kind": "page_failed",
        "page_num": 12,
    },
    {
        "data": {"defect": "table_width_mismatch", "site": "final_body"},
        "detail": "table_width_mismatch defect found in exact final page body",
        "engine": "native+math",
        "kind": "table_structure_failed",
        "page_num": 12,
    },
]

_NEW_AUDIT_LOG_EVENTS = [
    ev
    for ev in _OLD_AUDIT_LOG_EVENTS
    if not (
        (ev["page_num"] == 9 and ev["kind"] == "structure_class_model_table_kept")
        or (ev["page_num"] == 12 and ev["kind"] == "corrupt_math_hybrid_shipped")
    )
]

AUDIT_LOG_DIFFERENCES: list[ExpectedDiff] = [
    ExpectedDiff(
        path=("audit_log", 0, "counts", "structure_class_model_table_kept"),
        old_value=2,
        new_value=1,
        description=(
            "audit_log.json structure_class_model_table_kept count drops from 2 to 1 (page 8 only)."
        ),
    ),
    ExpectedDiff(
        path=("audit_log", 0, "counts", "corrupt_math_hybrid_shipped"),
        old_value=2,
        new_value=1,
        description=(
            "audit_log.json corrupt_math_hybrid_shipped count drops from 2 to 1 (page 11 only)."
        ),
    ),
    ExpectedDiff(
        path=("audit_log", 0, "event_count"),
        old_value=16,
        new_value=14,
        description="audit_log.json total event_count drops from 16 to 14.",
    ),
    ExpectedDiff(
        path=("audit_log", 0, "events"),
        old_value=_OLD_AUDIT_LOG_EVENTS,
        new_value=_NEW_AUDIT_LOG_EVENTS,
        description=(
            "audit_log.json events list removes page 9 structure-class and page 12 corrupt-math"
            " events."
        ),
    ),
]

# 5. Tables Trust (tables_trust.json)
_OLD_P9_TRUST_DETAILS = [
    (
        "structure_class_model_table_kept: structure-class page (table); native may not author"
        " the grid (C1), and a model attempt did -- the model's reading ships instead of native,"
        " flagged per its own status"
    ),
    "table_structure_failed: table_latex_leak defect found in exact final page body",
]
_NEW_P9_TRUST_DETAILS = [
    "table_structure_failed: table_latex_leak defect found in exact final page body",
]

_OLD_P9_TRUST_REASONS = [
    "structure_class_model_table_kept",
    "table_structure_failed",
]
_NEW_P9_TRUST_REASONS = [
    "table_structure_failed",
]

TABLES_TRUST_DIFFERENCES: list[ExpectedDiff] = [
    ExpectedDiff(
        path=(
            "tables_trust",
            0,
            "counts_by_kind",
            "structure_class_model_table_kept",
        ),
        old_value=2,
        new_value=1,
        description=(
            "tables_trust.json structure_class_model_table_kept count drops from 2 to 1 "
            "(page 8 only)."
        ),
    ),
    ExpectedDiff(
        path=("tables_trust", 0, "table_flags_n"),
        old_value=9,
        new_value=8,
        description="tables_trust.json total table_flags_n drops from 9 to 8.",
    ),
    ExpectedDiff(
        path=(
            "tables_trust",
            0,
            "pages",
            str(STRUCT_MODEL_REWRITTEN_PAGE),
            "reasons",
        ),
        old_value=_OLD_P9_TRUST_REASONS,
        new_value=_NEW_P9_TRUST_REASONS,
        description=(
            f"tables_trust.json page {STRUCT_MODEL_REWRITTEN_PAGE} reasons removes "
            "structure_class_model_table_kept, leaving table_structure_failed."
        ),
    ),
    ExpectedDiff(
        path=(
            "tables_trust",
            0,
            "pages",
            str(STRUCT_MODEL_REWRITTEN_PAGE),
            "details",
        ),
        old_value=_OLD_P9_TRUST_DETAILS,
        new_value=_NEW_P9_TRUST_DETAILS,
        description=(
            f"tables_trust.json page {STRUCT_MODEL_REWRITTEN_PAGE} details removes "
            "the structure_class_model_table_kept detail line."
        ),
    ),
]

# 6. CLI Output
_OLD_CLI = (
    "\nAssemble:\n"
    "  5 page(s) produced no usable output: [2, 6, 9, 10, 12]\n"
    "  2 page(s) shipped crop-backed equation candidate(s); "
    "mathematical fidelity remains unverified: [11, 12]\n"
    "  2 structure-class page(s) shipped the model's grid reading over native "
    "(native may not author a grid): [8, 9]\n"
    "  1 structure-class page(s) hit the fail-closed floor "
    "(usable grid candidates refused/absent; marker plus page image selected; "
    "native geometry grid withheld): [10]\n"
    "  2 table page(s) shipped a MODEL reading over a failed-closed native table region "
    "(verify against the source image before citing): [3, 5]\n"
    "  2 table page(s) hit the D3 fail-closed floor "
    "(unverifiable region → explicit failure marker): [2, 4]\n"
    "  Output: <TMP>/out/p6_corpus/p6_corpus.md\n"
    "  Manifest: <TMP>/out/p6_corpus/manifest.json (replayable)\n"
    "  Audit log: <TMP>/out/p6_corpus/audit_log.json "
    "(2 corrupt_math_hybrid_shipped, 2 d3_floor_model_table_kept, 5 page_failed, "
    "1 structure_class_ladder_exhausted_floor, 2 structure_class_model_table_kept, "
    "2 table_region_unverifiable, 2 table_structure_failed)\n"
    "  Table trust: 8 page(s) with untrusted tables (9 flag(s)): 2, 3, 4, 5, 8, 9, 10, 12\n"
)

_NEW_CLI = (
    "\nAssemble:\n"
    "  5 page(s) produced no usable output: [2, 6, 9, 10, 12]\n"
    "  1 page(s) shipped crop-backed equation candidate(s); "
    "mathematical fidelity remains unverified: [11]\n"
    "  1 structure-class page(s) shipped the model's grid reading over native "
    "(native may not author a grid): [8]\n"
    "  1 structure-class page(s) hit the fail-closed floor "
    "(usable grid candidates refused/absent; marker plus page image selected; "
    "native geometry grid withheld): [10]\n"
    "  2 table page(s) shipped a MODEL reading over a failed-closed native table region "
    "(verify against the source image before citing): [3, 5]\n"
    "  2 table page(s) hit the D3 fail-closed floor "
    "(unverifiable region → explicit failure marker): [2, 4]\n"
    "  Output: <TMP>/out/p6_corpus/p6_corpus.md\n"
    "  Manifest: <TMP>/out/p6_corpus/manifest.json (replayable)\n"
    "  Audit log: <TMP>/out/p6_corpus/audit_log.json "
    "(1 corrupt_math_hybrid_shipped, 2 d3_floor_model_table_kept, 5 page_failed, "
    "1 structure_class_ladder_exhausted_floor, 1 structure_class_model_table_kept, "
    "2 table_region_unverifiable, 2 table_structure_failed)\n"
    "  Table trust: 8 page(s) with untrusted tables (8 flag(s)): 2, 3, 4, 5, 8, 9, 10, 12\n"
)

CLI_DIFFERENCES: list[ExpectedDiff] = [
    ExpectedDiff(
        path=("cli",),
        old_value=_OLD_CLI,
        new_value=_NEW_CLI,
        description=(
            "CLI summary lines reflect: equation candidates [11, 12]->[11], "
            "structure-class grid [8, 9]->[8], audit log counts 2->1 for both kinds, "
            "and table trust flags 9->8."
        ),
    ),
]

# 7. Result Error & Metadata Error Notes
_OLD_ERROR_NOTE = (
    "page(s) 2, 6, 9, 10, 12 produced no usable output; "
    "corrupt equation candidate unverified on page(s) 11, 12; "
    "untrusted tables on 7 page(s), 7 flag(s) (see tables_trust.json); "
    "page(s) 10: structure-class ladder exhausted; fail-closed floor shipped "
    "(marker plus page image, native geometry grid withheld); "
    "invalid final table emission on page(s) 9, 12"
)

_NEW_ERROR_NOTE = (
    "page(s) 2, 6, 9, 10, 12 produced no usable output; "
    "corrupt equation candidate unverified on page(s) 11; "
    "untrusted tables on 6 page(s), 6 flag(s) (see tables_trust.json); "
    "page(s) 10: structure-class ladder exhausted; fail-closed floor shipped "
    "(marker plus page image, native geometry grid withheld); "
    "invalid final table emission on page(s) 9, 12"
)

RESULT_ERROR_DIFFERENCES: list[ExpectedDiff] = [
    ExpectedDiff(
        path=("result_error",),
        old_value=_OLD_ERROR_NOTE,
        new_value=_NEW_ERROR_NOTE,
        description=(
            "Result error note drops page 12 from corrupt equations (now page 11 only) "
            "and updates pre-emission table trust from 7 pages/7 flags to 6 pages/6 flags."
        ),
    ),
]

METADATA_DIFFERENCES: list[ExpectedDiff] = [
    ExpectedDiff(
        path=("metadata", "metadata.json", "files", "p6_corpus.pdf", "error"),
        old_value=_OLD_ERROR_NOTE,
        new_value=_NEW_ERROR_NOTE,
        description="Top-level metadata.json error string matches result_error delta.",
    ),
    ExpectedDiff(
        path=("metadata", "p6_corpus/metadata.json", "error"),
        old_value=_OLD_ERROR_NOTE,
        new_value=_NEW_ERROR_NOTE,
        description="Per-document metadata.json error string matches result_error delta.",
    ),
]
#: #1129's eleven entries ALL moved into ``PAGE_TOP_MARK_1145_DIFFERENCES`` below.
#:
#: Not a deletion of #1129's ledger: every one of its leaves (pages 3 and 8, their
#: fragments, the assembled ``.md``, their sidecar text and fingerprint, and their manifest
#: blob_ref) is a leaf #1145 ALSO changes, because the canonical unchecked line lands in the
#: same leading block as the unaccepted banner. A leaf can hold one value, so it gets one
#: entry, and each moved entry's description names both tickets. Kept as an empty list, with
#: this note, so the #1129 provenance stays findable at the name a reader will grep for.
UNACCEPTED_OUTPUT_1129_DIFFERENCES: list[ExpectedDiff] = []


#: #1145 slice 1: the canonical unchecked line, and the leaves it moves.
#:
#: Every entry below is the measured delta from the frozen pre-change capture, not a
#: regenerated fixture. Eleven of them SUPERSEDE a #1129 entry at the same leaf path:
#: pages 3 and 8 carry #1129's unaccepted banner AND a canonical line in the same
#: leading block, so the leaf's value is the product of both changes and can only be
#: enumerated once. Those eleven moved here out of the #1129 list so that exactly one
#: entry owns each path and the granular per-entry check stays load-bearing.
PAGE_TOP_MARK_1145_DIFFERENCES: list[ExpectedDiff] = [
    ExpectedDiff(
        path=("manifest", 0, "entries", "11", "blob_ref"),
        old_value="598c3fdb3e3dc058f30918b66b553aa48db7c83ae000fcaeb0f94b17e76afd66",
        new_value="828f07f460a5428fa6ba3c3e8e36311ae13566dfe5f8bb25658bb87491f8011d",
        description="#1145: page 11 (hybrid_clean) gains the canonical unchecked line as its first rendered block. the corrupt-math hybrid winner (engine native+math, AUDIT_FAILED with no judge refusal evidence at all). equation_fidelity, NOT page_judge: see verdict._judge_refusal_clause. Its leading block is prose, so the line IS written.",
    ),
    ExpectedDiff(
        path=("markdown", "p6_corpus/pages/00011.md"),
        old_value="native prose plus crop-backed region candidate",
        new_value="> [page 11 unchecked: equation_fidelity=FAIL; scope=page; reason=a corrupt equation candidate ships on this page without a verified reading]\n\nnative prose plus crop-backed region candidate",
        description="#1145: page 11 (hybrid_clean) gains the canonical unchecked line as its first rendered block. the corrupt-math hybrid winner (engine native+math, AUDIT_FAILED with no judge refusal evidence at all). equation_fidelity, NOT page_judge: see verdict._judge_refusal_clause. Its leading block is prose, so the line IS written.",
    ),
    ExpectedDiff(
        path=("page_contract", 10, "winning_output", "text"),
        old_value="native prose plus crop-backed region candidate",
        new_value="> [page 11 unchecked: equation_fidelity=FAIL; scope=page; reason=a corrupt equation candidate ships on this page without a verified reading]\n\nnative prose plus crop-backed region candidate",
        description="#1145: page 11 (hybrid_clean) gains the canonical unchecked line as its first rendered block. the corrupt-math hybrid winner (engine native+math, AUDIT_FAILED with no judge refusal evidence at all). equation_fidelity, NOT page_judge: see verdict._judge_refusal_clause. Its leading block is prose, so the line IS written.",
    ),
    ExpectedDiff(
        path=("sidecars", "p6_corpus/pages/00011.json", "page_fingerprint"),
        old_value="sha256:598c3fdb3e3dc058f30918b66b553aa48db7c83ae000fcaeb0f94b17e76afd66",
        new_value="sha256:828f07f460a5428fa6ba3c3e8e36311ae13566dfe5f8bb25658bb87491f8011d",
        description="#1145: page 11 (hybrid_clean) gains the canonical unchecked line as its first rendered block. the corrupt-math hybrid winner (engine native+math, AUDIT_FAILED with no judge refusal evidence at all). equation_fidelity, NOT page_judge: see verdict._judge_refusal_clause. Its leading block is prose, so the line IS written.",
    ),
    ExpectedDiff(
        path=("sidecars", "p6_corpus/pages/00011.json", "winning_output", "text"),
        old_value="native prose plus crop-backed region candidate",
        new_value="> [page 11 unchecked: equation_fidelity=FAIL; scope=page; reason=a corrupt equation candidate ships on this page without a verified reading]\n\nnative prose plus crop-backed region candidate",
        description="#1145: page 11 (hybrid_clean) gains the canonical unchecked line as its first rendered block. the corrupt-math hybrid winner (engine native+math, AUDIT_FAILED with no judge refusal evidence at all). equation_fidelity, NOT page_judge: see verdict._judge_refusal_clause. Its leading block is prose, so the line IS written.",
    ),
    ExpectedDiff(
        path=("manifest", 0, "entries", "3", "blob_ref"),
        old_value="2464290d15a20872c3f227c9420de235b40d963bc1bd0787ee755802b39378a1",
        new_value="8e145ec962b4e7c4166aea9f0210504628c470ec06f40e3352590fe25facb1c8",
        description="#1129 + #1145: the assembled document carries both the unaccepted banner (pages 3 and 8) and the canonical unchecked line (pages 3, 8 and 11). This leaf is the product of both changes, so it is enumerated once, here, rather than twice at the same path.",
    ),
    ExpectedDiff(
        path=("manifest", 0, "entries", "8", "blob_ref"),
        old_value="9598aba8cc17ea63cb2aa7871908552f241931ef243524a3d0ff9ba1957e402f",
        new_value="597b18ef36a1a1eb1fc26e544f827b0c90f5e630a4d750cb0dff0f16ef6c21e0",
        description="#1129 + #1145: the assembled document carries both the unaccepted banner (pages 3 and 8) and the canonical unchecked line (pages 3, 8 and 11). This leaf is the product of both changes, so it is enumerated once, here, rather than twice at the same path.",
    ),
    ExpectedDiff(
        path=("markdown", "p6_corpus/p6_corpus.md"),
        old_value="## Page 1\n\nOrdinary prose with no table on it at all.\n\n## Page 2\n\n[page 2 failed: unverifiable table — see image]\n\n## Page 3\n\nThe regressions are reported below.\n\n$$y_{t+1} = \\alpha + \\beta x_t + \\varepsilon_{t+1}$$\n\n| | const. | slope | $R^2$ |\n|---|---|---|---|\n| one | -0.04 | 0.06 | 0.00 |\n| two | 1.96 | -0.06 | 0.23 |\n\nThe estimates are reported to two decimals.\n\n[page 3: unverifiable table — kept a model reading over the failed-table marker; the native table region failed verification and no OCR rung was accepted — verify against the page image before citing]\n\n## Page 4\n\nThe regressions are reported below.\n\n$$y_{t+1} = \\alpha + \\beta x_t + \\varepsilon_{t+1}$$\n\n| | const. | slope | $R^2$ |\n|---|---|---|---|\n| one | -0.04 | 0.06 | 0.00 |\n| two | 1.96 | -0.06 | 0.23 |\n\nThe estimates are reported to two decimals.\n\n## Page 5\n\nThe regressions are reported below.\n\n$$y_{t+1} = \\alpha + \\beta x_t + \\varepsilon_{t+1}$$\n\n| | const. | slope | $R^2$ |\n|---|---|---|---|\n| one | -0.04 | 0.06 | 0.00 |\n| two | 1.96 | -0.06 | 0.23 |\n\nThe estimates are reported to two decimals.\n\n## Page 6\n\n[page 6 failed: no usable OCR output]\n\n## Page 7\n\nPlain model prose for page seven.\n\n## Page 8\n\n| n | slope | R2 |\n|---|---|---|\n| 1 | 0.03 | 0.91 |\n\n## Page 9\n\n[page 9 failed: invalid table emission — table_latex_leak]\n\n## Page 10\n\n[page 10 failed: unverifiable table — see image]\n\n## Page 11\n\nnative prose plus crop-backed region candidate\n\n## Page 12\n\n[page 12 failed: invalid table emission — table_width_mismatch]\n",
        new_value="## Page 1\n\nOrdinary prose with no table on it at all.\n\n## Page 2\n\n[page 2 failed: unverifiable table — see image]\n\n## Page 3\n\n> [page unaccepted: socr kept this model reading but the page judge did not accept it (`judge_only`); check against the page image before citing]\n\n> [page 3 unchecked: native_table_verifier=FAIL; scope=page; reason=a model table ships here over a native floor that failed verification]\n\nThe regressions are reported below.\n\n$$y_{t+1} = \\alpha + \\beta x_t + \\varepsilon_{t+1}$$\n\n| | const. | slope | $R^2$ |\n|---|---|---|---|\n| one | -0.04 | 0.06 | 0.00 |\n| two | 1.96 | -0.06 | 0.23 |\n\nThe estimates are reported to two decimals.\n\n[page 3: unverifiable table — kept a model reading over the failed-table marker; the native table region failed verification and no OCR rung was accepted — verify against the page image before citing]\n\n## Page 4\n\nThe regressions are reported below.\n\n$$y_{t+1} = \\alpha + \\beta x_t + \\varepsilon_{t+1}$$\n\n| | const. | slope | $R^2$ |\n|---|---|---|---|\n| one | -0.04 | 0.06 | 0.00 |\n| two | 1.96 | -0.06 | 0.23 |\n\nThe estimates are reported to two decimals.\n\n## Page 5\n\nThe regressions are reported below.\n\n$$y_{t+1} = \\alpha + \\beta x_t + \\varepsilon_{t+1}$$\n\n| | const. | slope | $R^2$ |\n|---|---|---|---|\n| one | -0.04 | 0.06 | 0.00 |\n| two | 1.96 | -0.06 | 0.23 |\n\nThe estimates are reported to two decimals.\n\n## Page 6\n\n[page 6 failed: no usable OCR output]\n\n## Page 7\n\nPlain model prose for page seven.\n\n## Page 8\n\n> [page unaccepted: socr kept this model reading but the page judge did not accept it (`ambiguous_deferred`); check against the page image before citing]\n\n> [page 8 unchecked: native_table_verifier=FAIL; scope=page; reason=no table-ladder rung accepted this page's table and the model reading still ships]\n\n| n | slope | R2 |\n|---|---|---|\n| 1 | 0.03 | 0.91 |\n\n## Page 9\n\n[page 9 failed: invalid table emission — table_latex_leak]\n\n## Page 10\n\n[page 10 failed: unverifiable table — see image]\n\n## Page 11\n\n> [page 11 unchecked: equation_fidelity=FAIL; scope=page; reason=a corrupt equation candidate ships on this page without a verified reading]\n\nnative prose plus crop-backed region candidate\n\n## Page 12\n\n[page 12 failed: invalid table emission — table_width_mismatch]\n",
        description="#1129 + #1145: the assembled document carries both the unaccepted banner (pages 3 and 8) and the canonical unchecked line (pages 3, 8 and 11). This leaf is the product of both changes, so it is enumerated once, here, rather than twice at the same path.",
    ),
    ExpectedDiff(
        path=("markdown", "p6_corpus/pages/00003.md"),
        old_value="The regressions are reported below.\n\n$$y_{t+1} = \\alpha + \\beta x_t + \\varepsilon_{t+1}$$\n\n| | const. | slope | $R^2$ |\n|---|---|---|---|\n| one | -0.04 | 0.06 | 0.00 |\n| two | 1.96 | -0.06 | 0.23 |\n\nThe estimates are reported to two decimals.\n\n[page 3: unverifiable table — kept a model reading over the failed-table marker; the native table region failed verification and no OCR rung was accepted — verify against the page image before citing]",
        new_value="> [page unaccepted: socr kept this model reading but the page judge did not accept it (`judge_only`); check against the page image before citing]\n\n> [page 3 unchecked: native_table_verifier=FAIL; scope=page; reason=a model table ships here over a native floor that failed verification]\n\nThe regressions are reported below.\n\n$$y_{t+1} = \\alpha + \\beta x_t + \\varepsilon_{t+1}$$\n\n| | const. | slope | $R^2$ |\n|---|---|---|---|\n| one | -0.04 | 0.06 | 0.00 |\n| two | 1.96 | -0.06 | 0.23 |\n\nThe estimates are reported to two decimals.\n\n[page 3: unverifiable table — kept a model reading over the failed-table marker; the native table region failed verification and no OCR rung was accepted — verify against the page image before citing]",
        description="#1129 + #1145: page 3 (d3_model_kept) keeps #1129's unaccepted banner as its first line and gains the canonical unchecked line below it, in the same leading block. a model table ships over a failed native floor AND the judge refused it; #1129's banner covers page_judge, so only the native_table_verifier line is written. This leaf is the product of both changes, so it is enumerated once, here.",
    ),
    ExpectedDiff(
        path=("markdown", "p6_corpus/pages/00008.md"),
        old_value="| n | slope | R2 |\n|---|---|---|\n| 1 | 0.03 | 0.91 |",
        new_value="> [page unaccepted: socr kept this model reading but the page judge did not accept it (`ambiguous_deferred`); check against the page image before citing]\n\n> [page 8 unchecked: native_table_verifier=FAIL; scope=page; reason=no table-ladder rung accepted this page's table and the model reading still ships]\n\n| n | slope | R2 |\n|---|---|---|\n| 1 | 0.03 | 0.91 |",
        description="#1129 + #1145: page 8 (struct_model_passing) keeps #1129's unaccepted banner as its first line and gains the canonical unchecked line below it, in the same leading block. the structure-class ladder accepted no rung AND the judge refused; #1129's banner covers page_judge, so only the native_table_verifier line is written. This leaf is the product of both changes, so it is enumerated once, here.",
    ),
    ExpectedDiff(
        path=("page_contract", 2, "winning_output", "text"),
        old_value="The regressions are reported below.\n\n$$y_{t+1} = \\alpha + \\beta x_t + \\varepsilon_{t+1}$$\n\n| | const. | slope | $R^2$ |\n|---|---|---|---|\n| one | -0.04 | 0.06 | 0.00 |\n| two | 1.96 | -0.06 | 0.23 |\n\nThe estimates are reported to two decimals.\n\n[page 3: unverifiable table — kept a model reading over the failed-table marker; the native table region failed verification and no OCR rung was accepted — verify against the page image before citing]\n\n![p](figures/p.png)\n",
        new_value="> [page unaccepted: socr kept this model reading but the page judge did not accept it (`judge_only`); check against the page image before citing]\n\n> [page 3 unchecked: native_table_verifier=FAIL; scope=page; reason=a model table ships here over a native floor that failed verification]\n\nThe regressions are reported below.\n\n$$y_{t+1} = \\alpha + \\beta x_t + \\varepsilon_{t+1}$$\n\n| | const. | slope | $R^2$ |\n|---|---|---|---|\n| one | -0.04 | 0.06 | 0.00 |\n| two | 1.96 | -0.06 | 0.23 |\n\nThe estimates are reported to two decimals.\n\n[page 3: unverifiable table — kept a model reading over the failed-table marker; the native table region failed verification and no OCR rung was accepted — verify against the page image before citing]\n\n![p](figures/p.png)\n",
        description="#1129 + #1145: the assembled document carries both the unaccepted banner (pages 3 and 8) and the canonical unchecked line (pages 3, 8 and 11). This leaf is the product of both changes, so it is enumerated once, here, rather than twice at the same path.",
    ),
    ExpectedDiff(
        path=("page_contract", 7, "winning_output", "text"),
        old_value="| n | slope | R2 |\n|---|---|---|\n| 1 | 0.03 | 0.91 |",
        new_value="> [page unaccepted: socr kept this model reading but the page judge did not accept it (`ambiguous_deferred`); check against the page image before citing]\n\n> [page 8 unchecked: native_table_verifier=FAIL; scope=page; reason=no table-ladder rung accepted this page's table and the model reading still ships]\n\n| n | slope | R2 |\n|---|---|---|\n| 1 | 0.03 | 0.91 |",
        description="#1129 + #1145: the assembled document carries both the unaccepted banner (pages 3 and 8) and the canonical unchecked line (pages 3, 8 and 11). This leaf is the product of both changes, so it is enumerated once, here, rather than twice at the same path.",
    ),
    ExpectedDiff(
        path=("sidecars", "p6_corpus/pages/00003.json", "page_fingerprint"),
        old_value="sha256:2464290d15a20872c3f227c9420de235b40d963bc1bd0787ee755802b39378a1",
        new_value="sha256:8e145ec962b4e7c4166aea9f0210504628c470ec06f40e3352590fe25facb1c8",
        description="#1129 + #1145: page 3 (d3_model_kept) keeps #1129's unaccepted banner as its first line and gains the canonical unchecked line below it, in the same leading block. a model table ships over a failed native floor AND the judge refused it; #1129's banner covers page_judge, so only the native_table_verifier line is written. This leaf is the product of both changes, so it is enumerated once, here.",
    ),
    ExpectedDiff(
        path=("sidecars", "p6_corpus/pages/00003.json", "winning_output", "text"),
        old_value="The regressions are reported below.\n\n$$y_{t+1} = \\alpha + \\beta x_t + \\varepsilon_{t+1}$$\n\n| | const. | slope | $R^2$ |\n|---|---|---|---|\n| one | -0.04 | 0.06 | 0.00 |\n| two | 1.96 | -0.06 | 0.23 |\n\nThe estimates are reported to two decimals.\n\n[page 3: unverifiable table — kept a model reading over the failed-table marker; the native table region failed verification and no OCR rung was accepted — verify against the page image before citing]",
        new_value="> [page unaccepted: socr kept this model reading but the page judge did not accept it (`judge_only`); check against the page image before citing]\n\n> [page 3 unchecked: native_table_verifier=FAIL; scope=page; reason=a model table ships here over a native floor that failed verification]\n\nThe regressions are reported below.\n\n$$y_{t+1} = \\alpha + \\beta x_t + \\varepsilon_{t+1}$$\n\n| | const. | slope | $R^2$ |\n|---|---|---|---|\n| one | -0.04 | 0.06 | 0.00 |\n| two | 1.96 | -0.06 | 0.23 |\n\nThe estimates are reported to two decimals.\n\n[page 3: unverifiable table — kept a model reading over the failed-table marker; the native table region failed verification and no OCR rung was accepted — verify against the page image before citing]",
        description="#1129 + #1145: page 3 (d3_model_kept) keeps #1129's unaccepted banner as its first line and gains the canonical unchecked line below it, in the same leading block. a model table ships over a failed native floor AND the judge refused it; #1129's banner covers page_judge, so only the native_table_verifier line is written. This leaf is the product of both changes, so it is enumerated once, here.",
    ),
    ExpectedDiff(
        path=("sidecars", "p6_corpus/pages/00008.json", "page_fingerprint"),
        old_value="sha256:9598aba8cc17ea63cb2aa7871908552f241931ef243524a3d0ff9ba1957e402f",
        new_value="sha256:597b18ef36a1a1eb1fc26e544f827b0c90f5e630a4d750cb0dff0f16ef6c21e0",
        description="#1129 + #1145: page 8 (struct_model_passing) keeps #1129's unaccepted banner as its first line and gains the canonical unchecked line below it, in the same leading block. the structure-class ladder accepted no rung AND the judge refused; #1129's banner covers page_judge, so only the native_table_verifier line is written. This leaf is the product of both changes, so it is enumerated once, here.",
    ),
    ExpectedDiff(
        path=("sidecars", "p6_corpus/pages/00008.json", "winning_output", "text"),
        old_value="| n | slope | R2 |\n|---|---|---|\n| 1 | 0.03 | 0.91 |",
        new_value="> [page unaccepted: socr kept this model reading but the page judge did not accept it (`ambiguous_deferred`); check against the page image before citing]\n\n> [page 8 unchecked: native_table_verifier=FAIL; scope=page; reason=no table-ladder rung accepted this page's table and the model reading still ships]\n\n| n | slope | R2 |\n|---|---|---|\n| 1 | 0.03 | 0.91 |",
        description="#1129 + #1145: page 8 (struct_model_passing) keeps #1129's unaccepted banner as its first line and gains the canonical unchecked line below it, in the same leading block. the structure-class ladder accepted no rung AND the judge refused; #1129's banner covers page_judge, so only the native_table_verifier line is written. This leaf is the product of both changes, so it is enumerated once, here.",
    ),
]

#: #1145 slice 1: the additive ``verdict`` sidecar key, enumerated page by page.
#:
#: NOT added to ``VOLATILE_KEYS``. The #1027/#1030 precedent put additive sidecar keys
#: there, but ``verdict`` is deterministic, it is the field this ticket is ABOUT, and
#: excising it would make the oracle blind to a wrong verdict on any page. Enumerated,
#: it is the opposite: the table below is the whole per-page result of the clause
#: derivation on this corpus, and a clause that changes, appears or disappears fails
#: here with its page named.
VERDICT_SIDECAR_1145_DIFFERENCES: list[ExpectedDiff] = [
    ExpectedDiff(
        path=("sidecars", "p6_corpus/pages/00001.json", "verdict"),
        old_value=None,
        new_value={"page_num": 1, "clauses": []},
        description="#1145: page 1 (clean_born_digital) gains the additive ``verdict`` sidecar key. CHECKED: no clause, so no mark and byte-identical output.",
    ),
    ExpectedDiff(
        path=("sidecars", "p6_corpus/pages/00002.json", "verdict"),
        old_value=None,
        new_value={
            "page_num": 2,
            "clauses": [
                {
                    "check": "table_structure",
                    "result": "FAIL",
                    "scope": "page",
                    "reason": "this page's native table structure could not be read",
                }
            ],
        },
        description="#1145: page 2 (d3_floor) gains the additive ``verdict`` sidecar key. the D3-floor page withheld its table; its whole body is the [page 2 failed: ...] marker, so page_failed_coverage suppresses the line and the bytes do not move.",
    ),
    ExpectedDiff(
        path=("sidecars", "p6_corpus/pages/00003.json", "verdict"),
        old_value=None,
        new_value={
            "page_num": 3,
            "clauses": [
                {
                    "check": "native_table_verifier",
                    "result": "FAIL",
                    "scope": "page",
                    "reason": "a model table ships here over a native floor that failed verification",
                },
                {
                    "check": "page_judge",
                    "result": "FAIL",
                    "scope": "page",
                    "reason": "the page judge did not accept the bytes that ship",
                },
            ],
        },
        description="#1145: page 3 (d3_model_kept) gains the additive ``verdict`` sidecar key. a model table ships over a failed native floor AND the judge refused it; #1129's banner covers page_judge, so only the native_table_verifier line is written.",
    ),
    ExpectedDiff(
        path=("sidecars", "p6_corpus/pages/00004.json", "verdict"),
        old_value=None,
        new_value={"page_num": 4, "clauses": []},
        description="#1145: page 4 (cold_review_shape_one) gains the additive ``verdict`` sidecar key. CHECKED: the D3 flags are on the native candidate, but the model reading that ships was accepted by the judge, so no check failed on the shipped bytes.",
    ),
    ExpectedDiff(
        path=("sidecars", "p6_corpus/pages/00005.json", "verdict"),
        old_value=None,
        new_value={"page_num": 5, "clauses": []},
        description="#1145: page 5 (cold_review_shape_two) gains the additive ``verdict`` sidecar key. CHECKED, same shape as page 4.",
    ),
    ExpectedDiff(
        path=("sidecars", "p6_corpus/pages/00006.json", "verdict"),
        old_value=None,
        new_value={
            "page_num": 6,
            "clauses": [
                {
                    "check": "unknown",
                    "result": "UNKNOWN",
                    "scope": "page",
                    "reason": "socr did not confirm this page and did not record which check failed (status error, failure mode none)",
                }
            ],
        },
        description="#1145: page 6 (no_text_failure) gains the additive ``verdict`` sidecar key. ERROR with failure_mode=none: no field names the failed check, so the fail-closed fallback fires. The body is the [page 6 failed: ...] marker, so no line is written.",
    ),
    ExpectedDiff(
        path=("sidecars", "p6_corpus/pages/00007.json", "verdict"),
        old_value=None,
        new_value={"page_num": 7, "clauses": []},
        description="#1145: page 7 (passing_model) gains the additive ``verdict`` sidecar key. CHECKED: a passing model reading.",
    ),
    ExpectedDiff(
        path=("sidecars", "p6_corpus/pages/00008.json", "verdict"),
        old_value=None,
        new_value={
            "page_num": 8,
            "clauses": [
                {
                    "check": "native_table_verifier",
                    "result": "FAIL",
                    "scope": "page",
                    "reason": "no table-ladder rung accepted this page's table and the model reading still ships",
                },
                {
                    "check": "page_judge",
                    "result": "FAIL",
                    "scope": "page",
                    "reason": "the page judge did not accept the bytes that ship",
                },
            ],
        },
        description="#1145: page 8 (struct_model_passing) gains the additive ``verdict`` sidecar key. the structure-class ladder accepted no rung AND the judge refused; #1129's banner covers page_judge, so only the native_table_verifier line is written.",
    ),
    ExpectedDiff(
        path=("sidecars", "p6_corpus/pages/00009.json", "verdict"),
        old_value=None,
        new_value={
            "page_num": 9,
            "clauses": [
                {
                    "check": "table_structure",
                    "result": "FAIL",
                    "scope": "page",
                    "reason": "a table on this page cannot be represented faithfully and was withheld",
                },
                {
                    "check": "page_judge",
                    "result": "FAIL",
                    "scope": "page",
                    "reason": "the page judge did not accept the bytes that ship",
                },
            ],
        },
        description="#1145: page 9 (struct_model_rewritten) gains the additive ``verdict`` sidecar key. the table was withheld as an invalid emission and the judge refused; the body is the [page 9 failed: ...] marker, so neither clause writes a line.",
    ),
    ExpectedDiff(
        path=("sidecars", "p6_corpus/pages/00010.json", "verdict"),
        old_value=None,
        new_value={
            "page_num": 10,
            "clauses": [
                {
                    "check": "native_table_verifier",
                    "result": "FAIL",
                    "scope": "page",
                    "reason": "every rung of the table ladder was exhausted without accepting this page's grid",
                }
            ],
        },
        description="#1145: page 10 (struct_floor) gains the additive ``verdict`` sidecar key. the struct-floor page exhausted every ladder rung; its body is the [page 10 failed: ...] marker, so no line is written.",
    ),
    ExpectedDiff(
        path=("sidecars", "p6_corpus/pages/00011.json", "verdict"),
        old_value=None,
        new_value={
            "page_num": 11,
            "clauses": [
                {
                    "check": "equation_fidelity",
                    "result": "FAIL",
                    "scope": "page",
                    "reason": "a corrupt equation candidate ships on this page without a verified reading",
                }
            ],
        },
        description="#1145: page 11 (hybrid_clean) gains the additive ``verdict`` sidecar key. the corrupt-math hybrid winner (engine native+math, AUDIT_FAILED with no judge refusal evidence at all). equation_fidelity, NOT page_judge: see verdict._judge_refusal_clause. Its leading block is prose, so the line IS written.",
    ),
    ExpectedDiff(
        path=("sidecars", "p6_corpus/pages/00012.json", "verdict"),
        old_value=None,
        new_value={
            "page_num": 12,
            "clauses": [
                {
                    "check": "table_structure",
                    "result": "FAIL",
                    "scope": "page",
                    "reason": "a table on this page cannot be represented faithfully and was withheld",
                }
            ],
        },
        description="#1145: page 12 (hybrid_rewritten) gains the additive ``verdict`` sidecar key. the table was withheld as an invalid emission; the body is the [page 12 failed: ...] marker, so no line is written.",
    ),
]


# ---------------------------------------------------------------------------
# Complete Unified EXPECTED_STAGE_C_DIFFERENCES Mapping
# ---------------------------------------------------------------------------

#: #1155 (repo review of PR #1190): the additive ``chart_floor_body`` sidecar key.
#:
#: MEASURED, not assumed: the oracle's own unexpected-diff report named these twelve leaf
#: paths and no others, each ``None -> False``. The flag is only ever True on a page whose
#: bare whole-page marker the chart floor replaced, and this corpus has none -- page 10 is
#: a structure-class floor page whose chart reader resolves no panel, which is why its
#: markdown is unchanged here too.
CHART_FLOOR_BODY_SIDECAR_1155_DIFFERENCES: list[ExpectedDiff] = [
    ExpectedDiff(
        path=("sidecars", "p6_corpus/pages/00001.json", "chart_floor_body"),
        old_value=None,
        new_value=False,
        description=(
            "#1155: page 1 gains the additive ``chart_floor_body`` sidecar key. False: no "
            "page of this corpus reaches the chart floor, so no body and no mark moves, and "
            "the key records that fact instead of leaving it to memory."
        ),
    ),
    ExpectedDiff(
        path=("sidecars", "p6_corpus/pages/00002.json", "chart_floor_body"),
        old_value=None,
        new_value=False,
        description=(
            "#1155: page 2 gains the additive ``chart_floor_body`` sidecar key. False: no "
            "page of this corpus reaches the chart floor, so no body and no mark moves, and "
            "the key records that fact instead of leaving it to memory."
        ),
    ),
    ExpectedDiff(
        path=("sidecars", "p6_corpus/pages/00003.json", "chart_floor_body"),
        old_value=None,
        new_value=False,
        description=(
            "#1155: page 3 gains the additive ``chart_floor_body`` sidecar key. False: no "
            "page of this corpus reaches the chart floor, so no body and no mark moves, and "
            "the key records that fact instead of leaving it to memory."
        ),
    ),
    ExpectedDiff(
        path=("sidecars", "p6_corpus/pages/00004.json", "chart_floor_body"),
        old_value=None,
        new_value=False,
        description=(
            "#1155: page 4 gains the additive ``chart_floor_body`` sidecar key. False: no "
            "page of this corpus reaches the chart floor, so no body and no mark moves, and "
            "the key records that fact instead of leaving it to memory."
        ),
    ),
    ExpectedDiff(
        path=("sidecars", "p6_corpus/pages/00005.json", "chart_floor_body"),
        old_value=None,
        new_value=False,
        description=(
            "#1155: page 5 gains the additive ``chart_floor_body`` sidecar key. False: no "
            "page of this corpus reaches the chart floor, so no body and no mark moves, and "
            "the key records that fact instead of leaving it to memory."
        ),
    ),
    ExpectedDiff(
        path=("sidecars", "p6_corpus/pages/00006.json", "chart_floor_body"),
        old_value=None,
        new_value=False,
        description=(
            "#1155: page 6 gains the additive ``chart_floor_body`` sidecar key. False: no "
            "page of this corpus reaches the chart floor, so no body and no mark moves, and "
            "the key records that fact instead of leaving it to memory."
        ),
    ),
    ExpectedDiff(
        path=("sidecars", "p6_corpus/pages/00007.json", "chart_floor_body"),
        old_value=None,
        new_value=False,
        description=(
            "#1155: page 7 gains the additive ``chart_floor_body`` sidecar key. False: no "
            "page of this corpus reaches the chart floor, so no body and no mark moves, and "
            "the key records that fact instead of leaving it to memory."
        ),
    ),
    ExpectedDiff(
        path=("sidecars", "p6_corpus/pages/00008.json", "chart_floor_body"),
        old_value=None,
        new_value=False,
        description=(
            "#1155: page 8 gains the additive ``chart_floor_body`` sidecar key. False: no "
            "page of this corpus reaches the chart floor, so no body and no mark moves, and "
            "the key records that fact instead of leaving it to memory."
        ),
    ),
    ExpectedDiff(
        path=("sidecars", "p6_corpus/pages/00009.json", "chart_floor_body"),
        old_value=None,
        new_value=False,
        description=(
            "#1155: page 9 gains the additive ``chart_floor_body`` sidecar key. False: no "
            "page of this corpus reaches the chart floor, so no body and no mark moves, and "
            "the key records that fact instead of leaving it to memory."
        ),
    ),
    ExpectedDiff(
        path=("sidecars", "p6_corpus/pages/00010.json", "chart_floor_body"),
        old_value=None,
        new_value=False,
        description=(
            "#1155: page 10 gains the additive ``chart_floor_body`` sidecar key. False: no "
            "page of this corpus reaches the chart floor, so no body and no mark moves, and "
            "the key records that fact instead of leaving it to memory."
        ),
    ),
    ExpectedDiff(
        path=("sidecars", "p6_corpus/pages/00011.json", "chart_floor_body"),
        old_value=None,
        new_value=False,
        description=(
            "#1155: page 11 gains the additive ``chart_floor_body`` sidecar key. False: no "
            "page of this corpus reaches the chart floor, so no body and no mark moves, and "
            "the key records that fact instead of leaving it to memory."
        ),
    ),
    ExpectedDiff(
        path=("sidecars", "p6_corpus/pages/00012.json", "chart_floor_body"),
        old_value=None,
        new_value=False,
        description=(
            "#1155: page 12 gains the additive ``chart_floor_body`` sidecar key. False: no "
            "page of this corpus reaches the chart floor, so no body and no mark moves, and "
            "the key records that fact instead of leaving it to memory."
        ),
    ),
]


EXPECTED_STAGE_C_DIFFERENCE_ENTRIES: list[ExpectedDiff] = [
    *BUCKET_DIFFERENCES,
    *STATE_EVENT_DIFFERENCES,
    *SIDECAR_EVENT_DIFFERENCES,
    *AUDIT_LOG_DIFFERENCES,
    *TABLES_TRUST_DIFFERENCES,
    *CLI_DIFFERENCES,
    *RESULT_ERROR_DIFFERENCES,
    *METADATA_DIFFERENCES,
    *UNACCEPTED_OUTPUT_1129_DIFFERENCES,
    *PAGE_TOP_MARK_1145_DIFFERENCES,
    *VERDICT_SIDECAR_1145_DIFFERENCES,
    *CHART_FLOOR_BODY_SIDECAR_1155_DIFFERENCES,
]

#: Exactly one entry per leaf path. The dict below would silently keep only the last entry
#: at a duplicated path, and ``test_granular_differences_match_enumerated_surfaces`` would
#: then check a stale value and fail -- which is how a superseded #1129 entry would be
#: caught. Asserted here so the failure names the cause instead.
_PATHS = [entry.path for entry in EXPECTED_STAGE_C_DIFFERENCE_ENTRIES]
assert len(_PATHS) == len(set(_PATHS)), sorted(
    path for path in set(_PATHS) if _PATHS.count(path) > 1
)

EXPECTED_STAGE_C_DIFFERENCES: dict[LeafPath, tuple[Any, Any]] = {
    entry.path: entry.as_pair() for entry in EXPECTED_STAGE_C_DIFFERENCE_ENTRIES
}


# ---------------------------------------------------------------------------
# The Disposition Surface (cold review round 1, finding 1)
# ---------------------------------------------------------------------------
#
# ``disposition`` is captured, not stripped. The stage-A/B HEAD already
# computed and persisted it, including on the two guard-rewritten pages: the
# post-selection emission guard has always rewritten their ending to
# ``FAIL_CLOSED_MARKER / INVALID_TABLE_EMISSION``.
#
# So the EXPECTED disposition delta of stage C is EMPTY. Stage C changes which
# bucket reads that disposition, never the disposition itself. That is the
# claim, and it is only worth anything if the values are actually compared,
# which is why the pinned table below is asserted on all three persisted
# surfaces of both captures rather than merely diffed.

#: Every page's public disposition, on both sides of stage C. Guard-rewritten
#: pages 9 and 12 are ``(FAIL_CLOSED_MARKER, INVALID_TABLE_EMISSION)`` BEFORE
#: stage C as well as after -- the migration removes them from the three
#: buckets precisely because this pair already said what they shipped.
EXPECTED_PAGE_DISPOSITIONS: dict[int, tuple[str, str]] = {
    CLEAN_BORN_DIGITAL_PAGE: ("native_prose", "clean_native_prose"),
    D3_FLOOR_PAGE: ("fail_closed_marker", "native_table_unverifiable"),
    D3_MODEL_KEPT_PAGE: ("model_output", "native_table_unverifiable"),
    COLD_REVIEW_SHAPE_ONE_PAGE: ("model_output", "accepted_output"),
    COLD_REVIEW_SHAPE_TWO_PAGE: ("model_output", "accepted_output"),
    NO_TEXT_FAILURE_PAGE: ("fail_closed_marker", "no_usable_output"),
    PASSING_MODEL_PAGE: ("model_output", "accepted_output"),
    STRUCT_MODEL_PASSING_PAGE: ("model_output", "structure_class"),
    STRUCT_MODEL_REWRITTEN_PAGE: ("fail_closed_marker", "invalid_table_emission"),
    STRUCT_FLOOR_PAGE: ("fail_closed_marker", "structure_class"),
    HYBRID_CLEAN_PAGE: ("model_output", "corrupt_math_hybrid"),
    HYBRID_REWRITTEN_PAGE: ("fail_closed_marker", "invalid_table_emission"),
}

#: The three persisted surfaces that must each carry all 12 dispositions.
DISPOSITION_SURFACES: tuple[str, ...] = ("sidecars", "manifest", "page_contract")

#: Pages whose bucket membership stage C changes, and whose disposition it
#: does not. Both facts are asserted together so the pair cannot drift apart.
GUARD_REWRITTEN_PAGES: tuple[int, ...] = (
    STRUCT_MODEL_REWRITTEN_PAGE,
    HYBRID_REWRITTEN_PAGE,
)


def _as_pair(value: Any) -> tuple[str, str] | None:
    """Normalize either persisted disposition shape to an ``(ending, reason)`` pair."""
    if isinstance(value, dict) and {"ending", "primary_reason"} <= set(value):
        return (value["ending"], value["primary_reason"])
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (value[0], value[1])
    return None


def collect_disposition_leaves(obj: Any, path: LeafPath = ()) -> dict[LeafPath, Any]:
    """Every ``disposition`` leaf in a capture, keyed by its full leaf path."""
    found: dict[LeafPath, Any] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "disposition":
                found[path + (key,)] = value
            else:
                found.update(collect_disposition_leaves(value, path + (key,)))
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            found.update(collect_disposition_leaves(value, path + (idx,)))
    return found


def page_dispositions_by_surface(capture: dict[str, Any]) -> dict[str, dict[int, tuple[str, str]]]:
    """``{surface: {page_num: (ending, primary_reason)}}`` for the three persisted surfaces."""
    sidecars = {
        int(key.rsplit("/", 1)[-1].removesuffix(".json")): _as_pair(value["disposition"])
        for key, value in capture["sidecars"].items()
        if key.endswith(".json") and "disposition" in value
    }
    manifest = {
        int(key): _as_pair(entry["disposition"])
        for key, entry in capture["manifest"][0]["entries"].items()
        if "disposition" in entry
    }
    contract = {
        record["page_num"]: _as_pair(record["disposition"])
        for record in capture["page_contract"]
        if "disposition" in record
    }
    return {"sidecars": sidecars, "manifest": manifest, "page_contract": contract}


__all__ += [
    "EXPECTED_PAGE_DISPOSITIONS",
    "DISPOSITION_SURFACES",
    "GUARD_REWRITTEN_PAGES",
    "collect_disposition_leaves",
    "page_dispositions_by_surface",
]
