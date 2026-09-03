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

# ---------------------------------------------------------------------------
# Complete Unified EXPECTED_STAGE_C_DIFFERENCES Mapping
# ---------------------------------------------------------------------------

EXPECTED_STAGE_C_DIFFERENCE_ENTRIES: list[ExpectedDiff] = [
    *BUCKET_DIFFERENCES,
    *STATE_EVENT_DIFFERENCES,
    *SIDECAR_EVENT_DIFFERENCES,
    *AUDIT_LOG_DIFFERENCES,
    *TABLES_TRUST_DIFFERENCES,
    *CLI_DIFFERENCES,
    *RESULT_ERROR_DIFFERENCES,
    *METADATA_DIFFERENCES,
]

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
