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


def _fingerprint(
    tmp_path: Path,
    *,
    engine: str,
    model: str,
    pdf: Path | None = None,
    reject: bool = False,
    determinant_model: str | None = None,
):
    """Fingerprint one page read by *engine* running *model*.

    ``pdf`` reuses an existing document, so two calls can differ ONLY by the
    model (cubic P3 on #507: building two PDFs separately left the file hash
    free to vary, which would have made the difference test pass for the wrong
    reason).

    ``reject`` routes the page through the rejected-but-shipped selection
    branch, which rebuilds the output instead of shipping the attempt.
    ``determinant_model`` supplies an engine-level configured model, which the
    page's own model must outrank.
    """
    pdf = pdf if pdf is not None else _pdf(tmp_path)
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
    if not reject:
        state.pages[1].best_output = out

    inputs = None
    if determinant_model is not None:
        inputs = {engine: (determinant_model, "ollama", "ocr", "prompt")}
    manifest = build_manifest(state, BlobStore(tmp_path / "cache"), fingerprint_inputs=inputs)
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
    pdf = _pdf(tmp_path / "doc")
    first = _fingerprint(
        tmp_path / "one", engine="qwen", model="qwen3-vl:30b-a3b-instruct", pdf=pdf
    )
    second = _fingerprint(tmp_path / "two", engine="qwen", model="qwen3-vl:8b", pdf=pdf)

    # Everything the cache identity is built from, except the model, must match
    # -- otherwise the inequality below could come from anywhere.
    assert (first.pdf_file_hash, first.page_num, first.render_dpi, first.engine) == (
        second.pdf_file_hash,
        second.page_num,
        second.render_dpi,
        second.engine,
    ), "the control failed: these two pages differ by more than the model"

    assert first != second, (
        "a model swap left the whole fingerprint identical, so replay would "
        "reuse a page read by a different model"
    )
    assert first.model_version != second.model_version


@pytest.mark.parametrize("determinant", [None, "qwen3-vl:30b-a3b-instruct"])
def test_a_native_page_is_identified_by_engine_not_by_a_fake_model(
    tmp_path: Path, determinant: str | None
) -> None:
    """Control, and the deliberate asymmetry.

    Without this, populating `model_version` from anything at hand -- a default,
    a sentinel, the engine name -- would satisfy the tests above while erasing
    the difference between "no model ran" and "a model ran".

    Parametrised over an engine-level determinant (cubic P2 on #507): a MIXED
    document populates `EngineResult.model_version` for its OCR engine, and the
    native pages were then stamped with a model that never read them. The
    unparametrised version of this test could not see that -- it had no
    configured model for the fallback chain to find.
    """
    fp = _fingerprint(
        tmp_path / f"native_{determinant}",
        engine="native",
        model="",
        determinant_model=determinant,
    )

    assert fp.engine == "native", f"a native page must say so: {fp.engine!r}"
    assert fp.model_version == "", (
        f"a native page was given a model identity it never had: {fp.model_version!r}"
    )


def test_a_rejected_page_keeps_the_model_that_produced_its_text(tmp_path: Path) -> None:
    """cubic P2 on #507: the rejected-but-shipped branch REBUILDS the output.

    When scoring cleared `best_output`, selection ships the rejected attempt's
    text wrapped in a fresh `PageOutput`. That rebuild dropped every provider
    field, so the page shipped with no model identity at all -- and a model swap
    could not invalidate it. The engine name alone does not separate two tags of
    the same engine.
    """
    pdf = _pdf(tmp_path / "doc")
    first = _fingerprint(
        tmp_path / "r1", engine="qwen", model="qwen3-vl:30b-a3b-instruct", pdf=pdf, reject=True
    )
    second = _fingerprint(tmp_path / "r2", engine="qwen", model="qwen3-vl:8b", pdf=pdf, reject=True)

    assert first.model_version == "qwen3-vl:30b-a3b-instruct", (
        f"a rejected-but-shipped page lost its model identity: {first.model_version!r}"
    )
    assert first != second, "two rejected pages read by different models fingerprint alike"


def test_the_page_model_outranks_the_configured_engine_model(tmp_path: Path) -> None:
    """cubic P2 on #507: what RAN beats what was CONFIGURED.

    `fingerprint_inputs` and `EngineResult.model_version` describe an engine's
    configured model. An agentic run can escalate one page to a different rung,
    and taking the configured value there would fingerprint that page under a
    model which never read it -- the precise failure this ticket is named for.
    """
    fp = _fingerprint(
        tmp_path / "esc",
        engine="qwen",
        model="qwen3-vl:30b-a3b-instruct",
        determinant_model="qwen3-vl:8b",
    )

    assert fp.model_version == "qwen3-vl:30b-a3b-instruct", (
        f"the fingerprint recorded the configured model, not the one that read "
        f"the page: {fp.model_version!r}"
    )
