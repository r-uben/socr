"""Step-1 foundation tests: content-addressed cache + manifest + replay.

These prove the corrected reproducibility design from the go-team panel:
record the winning OCR output once, then `replay` reconstructs a bit-identical
document purely from cached blobs — invoking NO engine. If this plumbing is
sound, the agentic layer can be built on top of it; if not, everything above is
built on sand.
"""

from __future__ import annotations

import pytest

fitz = pytest.importorskip("fitz")  # PyMuPDF; skip whole module if unavailable

from socr.core.cache import BlobStore, blob_hash  # noqa: E402
from socr.core.document import DocumentHandle  # noqa: E402
from socr.core.manifest import (  # noqa: E402
    Manifest,
    PageFingerprint,
    build_manifest,
    replay,
    stale_pages,
)
from socr.core.result import (  # noqa: E402
    DocumentStatus,
    EngineResult,
    FigureInfo,
    PageOutput,
    PageStatus,
)
from socr.core.state import DocumentState  # noqa: E402

# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


def _make_pdf(path, n_pages=2):
    doc = fitz.open()
    for i in range(n_pages):
        page = doc.new_page()
        page.insert_text((72, 72), f"native text page {i + 1}")
    doc.save(str(path))
    doc.close()
    return path


def _ocr_state(pdf_path, n_pages=2, engine="gemini"):
    """A DocumentState as if `engine` OCR'd every page successfully."""
    handle = DocumentHandle.from_path(pdf_path)
    state = DocumentState(handle=handle)
    pages = [
        PageOutput(
            page_num=i,
            text=f"OCR output for page {i}",
            status=PageStatus.SUCCESS,
            engine=engine,
            audit_passed=True,
        )
        for i in range(1, n_pages + 1)
    ]
    state.apply_result(
        EngineResult(
            document_path=pdf_path,
            engine=engine,
            status=DocumentStatus.SUCCESS,
            pages=pages,
        )
    )
    return state


# --------------------------------------------------------------------------
# serialization
# --------------------------------------------------------------------------


def test_page_output_roundtrip():
    page = PageOutput(
        page_num=3,
        text="some **markdown**",
        status=PageStatus.WARNING,
        engine="marker",
        confidence=0.7,
        figures=[FigureInfo(figure_num=1, page_num=3, figure_type="table", description="t")],
        audit_passed=False,
        audit_notes=["low words"],
        escalated_from="glm",
    )
    back = PageOutput.from_dict(page.to_dict())
    assert back == page  # dataclass __eq__ covers every field incl. nested figures


# --------------------------------------------------------------------------
# content-addressed store
# --------------------------------------------------------------------------


def test_blobstore_is_content_addressed(tmp_path):
    store = BlobStore(tmp_path / "cache")
    page = PageOutput(page_num=1, text="hello", engine="gemini")
    h1 = store.put_page(page)
    h2 = store.put_page(PageOutput.from_dict(page.to_dict()))  # identical content
    assert h1 == h2 == blob_hash(page.to_dict())
    assert store.get_page(h1) == page


def test_blobstore_missing_blob_raises(tmp_path):
    store = BlobStore(tmp_path / "cache")
    with pytest.raises(KeyError):
        store.get("0" * 64)


# --------------------------------------------------------------------------
# the core proof: build → replay is identical, and replay needs no engine
# --------------------------------------------------------------------------


def test_build_and_replay_matches_canonical_body(tmp_path):
    from ocr_output_contract import assemble_pages, split_native_pages

    pdf = _make_pdf(tmp_path / "paper.pdf", n_pages=2)
    state = _ocr_state(pdf, n_pages=2)
    store = BlobStore(tmp_path / "cache")

    manifest = build_manifest(state, store, dpi=120)
    out = replay(manifest, store)

    # Replay now emits the canonical '## Page N' body (assemble_pages), the same
    # assembler socr uses for the saved .md — bit-consistent with disk.
    assert out == assemble_pages(["OCR output for page 1", "OCR output for page 2"])
    assert split_native_pages(out) == [
        "OCR output for page 1",
        "OCR output for page 2",
    ]


def test_replay_from_disk_invokes_no_engine(tmp_path):
    """The reproducibility guarantee: a fresh process with only the on-disk
    manifest + cache (no DocumentState, no engine, no PDF re-render) rebuilds the
    exact document. This is what `socr replay` does on HPC."""
    pdf = _make_pdf(tmp_path / "paper.pdf", n_pages=3)
    state = _ocr_state(pdf, n_pages=3)
    cache_root = tmp_path / "cache"
    store = BlobStore(cache_root)
    manifest = build_manifest(state, store, dpi=120)
    manifest.save(tmp_path / "manifest.json")

    # Simulate a cold process: reload from disk only.
    reloaded = Manifest.load(tmp_path / "manifest.json")
    cold_store = BlobStore(cache_root)
    out = replay(reloaded, cold_store)

    from ocr_output_contract import assemble_pages

    expected = assemble_pages([f"OCR output for page {i}" for i in range(1, 4)])
    assert out == expected


def _whole_doc_state(pdf_path, n_pages=2, engine="gemini"):
    """A DocumentState as if a CLI engine OCR'd the whole PDF in one call.

    This is the path that triggered the historical replay/manifest empty-doc
    bug: the engine returns a single ``page_num=0`` PageOutput (stored in
    ``whole_doc_attempts``) holding the full ``## Page N`` markdown, and the
    per-page ``best_output`` slots are never populated.
    """
    handle = DocumentHandle.from_path(pdf_path)
    state = DocumentState(handle=handle)
    body = "\n\n".join(
        f"## Page {i}\n\nWhole-doc OCR text for page {i}" for i in range(1, n_pages + 1)
    )
    state.apply_result(
        EngineResult(
            document_path=pdf_path,
            engine=engine,
            status=DocumentStatus.SUCCESS,
            model_version="flash-2.0",
            pages=[
                PageOutput(
                    page_num=0,
                    text=body,
                    status=PageStatus.SUCCESS,
                    engine=engine,
                    audit_passed=True,
                )
            ],
        )
    )
    # Mirror the orchestrator: a passing whole-doc attempt sets audit_passed.
    state.whole_doc_attempts[-1].audit_passed = True
    return state, body


def test_whole_doc_replay_is_not_empty(tmp_path):
    """Regression: a whole-document CLI run must replay its full text, not an
    empty document. Before the fix, build_manifest froze empty per-page blobs
    because it only read per-page best_output (never populated on this path)."""
    pdf = _make_pdf(tmp_path / "paper.pdf", n_pages=3)
    state, body = _whole_doc_state(pdf, n_pages=3)
    store = BlobStore(tmp_path / "cache")

    manifest = build_manifest(state, store, dpi=120)
    out = replay(manifest, store)

    # Every page's text was recovered from the '## Page N' split — no empties.
    assert out.strip() != ""
    for i in range(1, 4):
        assert f"Whole-doc OCR text for page {i}" in out
    # One blob per page, each non-empty.
    for pn in range(1, 4):
        page = store.get_page(manifest.entries[pn].blob_ref)
        assert page.text.strip() != ""


def test_whole_doc_fingerprint_carries_model_version(tmp_path):
    """The fingerprint must record the engine's model_version so a model swap
    invalidates cached pages (previously it was always '')."""
    pdf = _make_pdf(tmp_path / "paper.pdf", n_pages=2)
    state, _ = _whole_doc_state(pdf, n_pages=2, engine="gemini")
    store = BlobStore(tmp_path / "cache")

    manifest = build_manifest(state, store, dpi=120)

    for pn in (1, 2):
        assert manifest.entries[pn].fingerprint.model_version == "flash-2.0"


def test_per_page_fingerprint_carries_model_version(tmp_path):
    """Per-page path also records model_version from its EngineResult."""
    pdf = _make_pdf(tmp_path / "paper.pdf", n_pages=2)
    handle = DocumentHandle.from_path(pdf)
    state = DocumentState(handle=handle)
    state.apply_result(
        EngineResult(
            document_path=pdf,
            engine="gemini",
            status=DocumentStatus.SUCCESS,
            model_version="flash-lite-9",
            pages=[
                PageOutput(
                    page_num=i,
                    text=f"ocr p{i}",
                    engine="gemini",
                    status=PageStatus.SUCCESS,
                    audit_passed=True,
                )
                for i in (1, 2)
            ],
        )
    )
    store = BlobStore(tmp_path / "cache")
    manifest = build_manifest(state, store, dpi=120)
    assert manifest.entries[1].fingerprint.model_version == "flash-lite-9"


def test_replay_detects_broken_cache(tmp_path):
    pdf = _make_pdf(tmp_path / "paper.pdf", n_pages=2)
    state = _ocr_state(pdf, n_pages=2)
    store = BlobStore(tmp_path / "cache")
    manifest = build_manifest(state, store, dpi=120)

    # An empty cache → every page is stale, and replay refuses to emit holes.
    empty = BlobStore(tmp_path / "empty_cache")
    assert stale_pages(manifest, empty) == [1, 2]
    with pytest.raises(KeyError):
        replay(manifest, empty)


# --------------------------------------------------------------------------
# fingerprint / invalidation
# --------------------------------------------------------------------------


def test_fingerprint_changes_invalidate(tmp_path):
    base = PageFingerprint(
        pdf_file_hash="abc",
        page_num=1,
        render_dpi=200,
        engine="gemini",
        model_version="flash-lite",
        image_hash="img1",
    )
    same = PageFingerprint(
        pdf_file_hash="abc",
        page_num=1,
        render_dpi=200,
        engine="gemini",
        model_version="flash-lite",
        image_hash="img1",
    )
    assert base.key() == same.key()

    # Any component change must change the identity (model drift, renderer drift,
    # engine swap, DPI change).
    for field_, new in [
        ("model_version", "flash"),
        ("image_hash", "img2"),
        ("engine", "marker"),
        ("render_dpi", 300),
    ]:
        d = base.__dict__.copy()
        d[field_] = new
        assert PageFingerprint(**d).key() != base.key(), field_


def test_ocr_pages_get_image_hash_native_pages_do_not(tmp_path):
    pdf = _make_pdf(tmp_path / "paper.pdf", n_pages=2)
    handle = DocumentHandle.from_path(pdf)
    state = DocumentState(handle=handle)
    # page 1: OCR'd; page 2: born-digital native fallback (no engine output)
    state.apply_result(
        EngineResult(
            document_path=pdf,
            engine="gemini",
            status=DocumentStatus.SUCCESS,
            pages=[
                PageOutput(
                    page_num=1,
                    text="ocr p1",
                    engine="gemini",
                    status=PageStatus.SUCCESS,
                    audit_passed=True,
                )
            ],
        )
    )
    state.pages[2].is_born_digital = True
    state.pages[2].native_text = "native p2"

    store = BlobStore(tmp_path / "cache")
    manifest = build_manifest(state, store, dpi=120)

    assert manifest.entries[1].fingerprint.engine == "gemini"
    assert manifest.entries[1].fingerprint.image_hash != ""  # rasterized → hashed
    assert manifest.entries[2].fingerprint.engine == "native"
    assert manifest.entries[2].fingerprint.image_hash == ""  # no rasterization

    from ocr_output_contract import assemble_pages

    assert replay(manifest, store) == assemble_pages(["ocr p1", "native p2"])
