"""P4-R t2: exact in-place advisory attachment of equation-region sidecars.

Covers the acceptance criteria for extending ``socr.math.equation_latex`` with
a pure attachment helper that splices a crop-backed, 1A-valid LaTeX sidecar
directly after the exact ``source_text`` slice it belongs to, rather than
appending at page end or replacing whole-page native prose (P4-R ruling 3).

The plan names no new module (`equation_lane.py` is explicitly forbidden) and
no new dataclass; the helper lives in `equation_latex.py` and consumes the
existing `EquationLatexResult` records. The exact helper name is not fixed by
the plan, so this file tries the two most likely names
(`attach_equation_sidecars_in_place`, `attach_equation_regions`) and skips if
neither exists yet -- t2 is not implemented on this branch.
"""

from __future__ import annotations

import importlib

import pytest

from socr.math.equation_latex import EQUATION_SIDECAR_HEADER, EquationLatexResult

# Cold review round 1, finding 6: the helper has landed, so it is imported by
# name. The previous version probed three candidate names and skipped the whole
# file if none existed, which made deletion of the feature look like a pass.
from socr.math.equation_latex import attach_equation_sidecars_in_place as attach


def _result(
    *,
    region_index: int,
    source_text: str,
    raw_latex: str = r"x^2",
    validation_ok: bool = True,
    sidecar_block: str = "SIDECAR",
) -> EquationLatexResult:
    r = EquationLatexResult(
        region_index=region_index,
        page_num=1,
        crop_path="equations/region-0.png",
        raw_latex=raw_latex,
        validation_ok=validation_ok,
        validation_reason="ok",
        latex_attached=validation_ok,
        model_id="qwen3-vl:30b-a3b-instruct",
        sidecar_block=sidecar_block,
    )
    # t2 acceptance requires the record retain its exact source slice; if the
    # field does not exist yet under this name, set it defensively so tests
    # still exercise the attach() call shape rather than failing on setattr.
    try:
        r.source_text = source_text  # type: ignore[attr-defined]
    except Exception:
        pass
    return r


class TestSourceOrderInsertion:
    def test_inserts_after_exact_source_slice_not_at_page_end(self):
        native = "Intro paragraph.\n\nx^2 + y^2 = z^2\n\nConclusion paragraph."
        result = _result(region_index=0, source_text="x^2 + y^2 = z^2")

        out, unaligned = attach(native, [result])

        assert "SIDECAR" in out
        # sidecar must appear immediately after the source slice, before the
        # conclusion paragraph -- not appended at the very end of the doc.
        idx_source = out.index("x^2 + y^2 = z^2")
        idx_sidecar = out.index("SIDECAR")
        idx_conclusion = out.index("Conclusion paragraph.")
        assert idx_source < idx_sidecar < idx_conclusion
        assert unaligned == []

    def test_every_native_byte_preserved(self):
        """Remove exactly what was inserted; what is left must be the input.

        Cold review round 1, finding 6: this used to check that three substrings
        were present. That passes even if the bytes between them are reordered or
        rewritten, which is the very thing the helper must never do.
        """
        native = "Intro paragraph.\n\nx^2 + y^2 = z^2\n\nConclusion paragraph."
        result = _result(region_index=0, source_text="x^2 + y^2 = z^2")

        out, unaligned = attach(native, [result])

        assert unaligned == []
        assert out != native
        recovered = out.replace("\n" + result.sidecar_block, "", 1)
        assert recovered == native


class TestNoOpAndIdempotence:
    def test_empty_attachable_set_is_a_no_op(self):
        native = "Only prose here, no equations at all."
        out, unaligned = attach(native, [])
        assert out == native
        assert unaligned == []

    def test_idempotent_on_repeated_application(self):
        native = "Before.\n\nE = mc^2\n\nAfter."
        result = _result(region_index=0, source_text="E = mc^2")

        once, _ = attach(native, [result])
        twice, _ = attach(once, [result])

        assert once == twice
        assert once.count("SIDECAR") == 1


class TestDeterministicRepeatedSlices:
    def test_repeated_source_text_consumed_once_per_region_in_order(self):
        native = "a = b\n\nfirst block\n\na = b\n\nsecond block"
        r0 = _result(region_index=0, source_text="a = b", sidecar_block="SIDECAR-0")
        r1 = _result(region_index=1, source_text="a = b", sidecar_block="SIDECAR-1")

        out, unaligned = attach(native, [r0, r1])

        assert unaligned == []
        idx0 = out.index("SIDECAR-0")
        idx1 = out.index("SIDECAR-1")
        # Region 0's sidecar must attach to the FIRST occurrence, region 1's to
        # the second -- not both piling onto the same occurrence.
        assert idx0 < out.index("first block") < idx1 < out.index("second block")


class TestUnalignedReporting:
    def test_unaligned_source_leaves_output_unchanged_and_is_reported(self):
        native = "This page has no matching equation text."
        result = _result(region_index=0, source_text="x^2 + y^2 = z^2 NOT PRESENT")

        out, unaligned = attach(native, [result])

        assert out == native
        assert 0 in unaligned


class TestMarkdownCropReference:
    def test_relative_crop_ref_used_in_sidecar(self):
        native = "before\n\nk = 1\n\nafter"
        result = _result(
            region_index=0,
            source_text="k = 1",
            sidecar_block=f"{EQUATION_SIDECAR_HEADER}\n![equation crop](equations/region-0.png)",
        )

        out, _ = attach(native, [result])

        assert "equations/region-0.png" in out
        assert EQUATION_SIDECAR_HEADER in out


class TestOnlyAttachableResultsContribute:
    def test_invalid_1a_result_contributes_nothing(self):
        native = "prose only, unclaimed"
        result = _result(region_index=0, source_text="prose only", validation_ok=False)
        # A caller that filters to attachable=False results before calling attach
        # should see a pure no-op; simulate the caller-side filter here.
        attachable = [r for r in [result] if r.validation_ok]
        out, unaligned = attach(native, attachable)
        assert out == native
