"""GH-310: ``upright_rotation_for`` must be fail-open all the way through.

Its guard wrapped only ``get_text``. ``dominant_text_direction`` and
``upright_rotation_degrees`` sat outside it, so a malformed ``dir`` -- or any
raise in either -- escaped a helper documented as fail-open. Two callers break
on that:

- the crop ``extract()``, whose call precedes ``_render_crop``'s own render-error
  boundary, so its never-raises contract failed
- the OCR ``process_pages`` render loop, now a one-line delegate to this helper

#309 hoisted the helper, so ONE wrap covers every caller. A second ``try`` at
each call site would reintroduce the duplication that hoist removed, which is
why this pins the helper rather than the call sites.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

fitz = pytest.importorskip("fitz")

from socr.core.born_digital import upright_rotation_for  # noqa: E402

_MALFORMED = {"blocks": [{"lines": [{"dir": ("x", "y"), "spans": [{"text": "hi"}]}]}]}
_WELL_FORMED = {"blocks": [{"lines": [{"dir": (0.0, -1.0), "spans": [{"text": "hi"}]}]}]}


def _page():
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), "some text", fontsize=11)
    return doc, page


class TestTheHelperIsFailOpenThroughout:
    def test_a_malformed_dir_returns_zero_instead_of_raising(self) -> None:
        doc, page = _page()
        try:
            with patch.object(fitz.Page, "get_text", lambda self, *a, **k: _MALFORMED):
                assert upright_rotation_for(page) == 0
        finally:
            doc.close()

    def test_a_well_formed_dir_still_computes_a_rotation(self) -> None:
        """Difference pin: the SAME call, differing only in whether ``dir`` is
        usable. Without this, a helper that returned 0 unconditionally would
        satisfy the test above while doing nothing."""
        doc, page = _page()
        try:
            with patch.object(fitz.Page, "get_text", lambda self, *a, **k: _WELL_FORMED):
                assert upright_rotation_for(page) == 90
        finally:
            doc.close()

    def test_the_swallowed_failure_is_logged(self, caplog) -> None:
        """GH-424 review: fail-open must not mean invisible.

        If either helper regresses, every page renders unrotated. Without a log
        line there is nothing anywhere saying why -- which is the silent-failure
        shape this repo rejects everywhere else.
        """
        import logging

        doc, page = _page()
        try:
            with (
                caplog.at_level(logging.DEBUG, logger="socr.core.born_digital"),
                patch.object(fitz.Page, "get_text", lambda self, *a, **k: _MALFORMED),
            ):
                assert upright_rotation_for(page) == 0
        finally:
            doc.close()

        assert any("direction inspection failed" in r.message for r in caplog.records), (
            "the failure was swallowed with no trace"
        )

    def test_a_raise_inside_the_direction_helper_is_contained(self) -> None:
        """The other half of the ticket: not just malformed data, but any raise
        from the two functions that used to sit outside the guard."""
        import socr.core.born_digital as bd

        doc, page = _page()

        def _boom(_blocks):
            raise RuntimeError("simulated direction failure")

        try:
            with (
                patch.object(fitz.Page, "get_text", lambda self, *a, **k: _WELL_FORMED),
                patch.object(bd, "dominant_text_direction", _boom),
            ):
                assert upright_rotation_for(page) == 0
        finally:
            doc.close()


# NOTE (GH-310): no caller-level test here, deliberately.
#
# The ticket names two victims: the crop ``extract()`` in ``tables/extract.py``
# (called before ``_render_crop``'s error boundary) and the OCR ``process_pages``
# render loop. A first attempt pinned ``BornDigitalDetector.extract_structured``
# instead -- a different function -- and it passed with the guard REMOVED,
# because that path catches the raise elsewhere. A test that passes either way
# guards nothing, so it was dropped rather than left looking like coverage.
#
# The helper-level pins above are the real guard: #309 hoisted this helper, so
# one wrap protects every caller, and reverting it reddens two of them.
