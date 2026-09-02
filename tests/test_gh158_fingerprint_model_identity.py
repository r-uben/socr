"""GH-158: a page's fingerprint must carry the model that actually read it.

Replay and resume identity are supposed to key off engine AND model, so that
swapping an ollama or Gemini model tag invalidates the cached page. That only
works if the fingerprint knows the model.

It did not. `build_manifest` took `model_version` from the caller's
`fingerprint_inputs` or from a doc-level `EngineResult.model_version` -- neither
of which exists on a per-page provider run. The page itself carried
`provider_model` all along (the journal two blocks below already wrote it out),
and nothing consulted it. So a correctly-recorded model page fingerprinted with
`model_version=""`, and a model swap left `replay` believing the cache valid.

Pinned as a DIFFERENCE between two pages that are identical except for the
model tag: the fingerprints must not be equal. Asserting a particular hash
would pin the hash function; asserting the difference pins the identity.

Native pages keep an EMPTY model, deliberately. There is no model, and a
sentinel like "n/a" would make "no model ran" indistinguishable from "the model
is named n/a" -- the distinction the provenance record exists to preserve.
`engine="native"` is what identifies them, and that is asserted too.
"""

from __future__ import annotations

from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

from socr.core.cache import BlobStore  # noqa: E402
from socr.core.document import DocumentHandle  # noqa: E402
from socr.core.manifest import build_manifest  # noqa: E402
from socr.core.result import PageOutput, PageStatus  # noqa: E402
from socr.core.state import DocumentState  # noqa: E402


def _pdf(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    pdf = tmp_path / "d.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "a text layer long enough to be a real one.")
    doc.save(str(pdf))
    doc.close()
    return pdf


def _fingerprint(tmp_path: Path, *, engine: str, model: str):
    """Fingerprint one page read by *engine* running *model*."""
    pdf = _pdf(tmp_path)
    state = DocumentState(handle=DocumentHandle.from_path(pdf))
    out = PageOutput(
        page_num=1,
        text="the page body",
        status=PageStatus.SUCCESS,
        engine=engine,
        audit_passed=True,
        provider_id="qwen-local" if model else "",
        provider_model=model,
        provider_backend="ollama" if model else "",
    )
    state.pages[1].attempts.append(out)
    state.pages[1].best_output = out
    manifest = build_manifest(state, BlobStore(tmp_path / "cache"))
    return manifest.entries[1].fingerprint


def test_a_model_page_records_the_model_that_read_it(tmp_path: Path) -> None:
    fp = _fingerprint(tmp_path / "a", engine="qwen", model="qwen3-vl:30b-a3b-instruct")

    assert fp.model_version == "qwen3-vl:30b-a3b-instruct", (
        f"the page knew its model but the fingerprint did not: {fp.model_version!r}. "
        "Replay cannot invalidate a cached page on a model swap it cannot see."
    )


def test_swapping_only_the_model_tag_changes_the_fingerprint(tmp_path: Path) -> None:
    """The property that actually matters, as a difference.

    Two pages identical in every other respect -- same PDF, same engine, same
    text, same DPI -- must not fingerprint alike when the model differs.
    """
    first = _fingerprint(tmp_path / "one", engine="qwen", model="qwen3-vl:30b-a3b-instruct")
    second = _fingerprint(tmp_path / "two", engine="qwen", model="qwen3-vl:8b")

    assert first.engine == second.engine, "the control failed: these differ by more than the model"
    assert first.model_version != second.model_version, (
        "a model swap left the fingerprint identical, so replay would reuse a "
        "page read by a different model"
    )


def test_a_native_page_is_identified_by_engine_not_by_a_fake_model(tmp_path: Path) -> None:
    """Control, and the deliberate asymmetry.

    Without this, populating `model_version` from anything at hand -- a default,
    a sentinel, the engine name -- would satisfy the tests above while erasing
    the difference between "no model ran" and "a model ran".
    """
    fp = _fingerprint(tmp_path / "native", engine="native", model="")

    assert fp.engine == "native", f"a native page must say so: {fp.engine!r}"
    assert fp.model_version == "", (
        f"a native page was given a model identity it never had: {fp.model_version!r}"
    )
