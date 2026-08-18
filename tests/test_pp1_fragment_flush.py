"""PP-1 tests: per-page fragment flush, atomic sidecar, and end-of-run stitch.

Covers the acceptance criteria from the PP-1 ticket:

* Byte-identity: for a processed doc, ``_stitch_fragments`` reproduces the
  same body that ``_canonical_body`` / ``assemble_pages(texts)`` produces.
  Proved by building a DocumentState with known per-page texts, flushing
  fragments via the new helpers, stitching, and asserting equality.

* ``pages/NNN.md`` + ``pages/NNN.json`` exist for every page after a
  successful flush pass; sidecar writes are atomic (tmp + rename).

* No second ``metadata.json`` writer is introduced; only ``pages/`` artefacts
  are created by the new helpers.

* ``PAGE_MARKER_RE`` round-trip: ``split_native_pages(assemble_pages(texts))``
  recovers the same per-page texts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from ocr_output_contract import PAGE_MARKER_RE, assemble_pages, split_native_pages

from socr.core.cache import BlobStore, blob_hash
from socr.core.config import EngineType, PipelineConfig
from socr.core.document import DocumentHandle
from socr.core.result import PageOutput, PageStatus
from socr.core.state import DocumentState, PageState
from socr.pipeline.orchestrator import UnifiedPipeline, _page_blob_key

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_handle(tmp_path: Path, page_count: int = 3) -> DocumentHandle:
    """DocumentHandle without touching a real PDF."""
    with patch.object(DocumentHandle, "__post_init__", lambda self: None):
        h = DocumentHandle(path=tmp_path / "fake.pdf", page_count=page_count)
    return h


def _make_pipeline() -> UnifiedPipeline:
    cfg = PipelineConfig(
        primary_engine=EngineType.DEEPSEEK,
        quiet=True,
        audit_enabled=False,
        save_figures=False,
        write_manifest=False,
        agentic=False,
        dual_pass_tables=False,
        judge_hard_pages=False,
    )
    return UnifiedPipeline(cfg)


def _make_state_with_texts(
    handle: DocumentHandle,
    texts: list[str],
) -> DocumentState:
    """Build a DocumentState with ``best_output`` set on every page."""
    state = DocumentState(handle=handle)
    for page_num, text in enumerate(texts, start=1):
        ps = state.pages[page_num]
        po = PageOutput(
            page_num=page_num,
            text=text,
            status=PageStatus.SUCCESS,
            engine="deepseek",
            audit_passed=True,
        )
        ps.attempts.append(po)
        ps.best_output = po
    return state


_PAGE_TEXTS = [
    "First page body with several words for the OCR test.",
    "Second page body with different content and some tables here.",
    "Third page body ending the document cleanly.",
]


# ---------------------------------------------------------------------------
# Byte-identity invariant
# ---------------------------------------------------------------------------


class TestByteIdentity:
    """_stitch_fragments must reproduce assemble_pages(canonical_page_texts(state))."""

    def test_stitch_equals_canonical_body(self, tmp_path: Path) -> None:
        handle = _make_handle(tmp_path, page_count=3)
        state = _make_state_with_texts(handle, _PAGE_TEXTS)
        pipeline = _make_pipeline()
        pipeline._scan_root = tmp_path

        # Compute the in-memory canonical body.
        in_memory_body, has_text = pipeline._canonical_body(state)
        assert has_text

        # Flush fragments.
        page_nums = list(range(1, handle.page_count + 1))
        from socr.core.manifest import canonical_page_texts

        page_texts = canonical_page_texts(state)
        for pnum, ptext in zip(page_nums, page_texts):
            pipeline._flush_page_fragment(state, pnum, ptext, tmp_path)

        # Stitch and compare byte-for-byte.
        stitched = pipeline._stitch_fragments(state, tmp_path)
        assert stitched == in_memory_body, (
            f"Stitched body ({len(stitched)} bytes) != in-memory body ({len(in_memory_body)} bytes)"
        )

    def test_assemble_pages_split_round_trip(self) -> None:
        """split_native_pages(assemble_pages(texts)) == texts (stripped)."""
        texts = _PAGE_TEXTS
        body = assemble_pages(texts)
        recovered = split_native_pages(body)
        # assemble_pages strips leading/trailing whitespace from each text;
        # split_native_pages strips the bodies it returns.  Compare stripped.
        assert [t.strip() for t in recovered] == [t.strip() for t in texts]

    def test_page_marker_re_count(self) -> None:
        """assemble_pages produces exactly one PAGE_MARKER per page."""
        texts = _PAGE_TEXTS
        body = assemble_pages(texts)
        markers = PAGE_MARKER_RE.findall(body)
        assert len(markers) == len(texts)

    def test_stitch_with_explicit_page_numbers(self, tmp_path: Path) -> None:
        """Explicit page_numbers=[1,2,3] produce the same output as positional."""
        texts = _PAGE_TEXTS
        body_positional = assemble_pages(texts)
        body_explicit = assemble_pages(texts, page_numbers=[1, 2, 3])
        assert body_positional == body_explicit

    def test_stitch_single_page(self, tmp_path: Path) -> None:
        handle = _make_handle(tmp_path, page_count=1)
        state = _make_state_with_texts(handle, ["Only page body."])
        pipeline = _make_pipeline()
        pipeline._scan_root = tmp_path

        in_memory_body, has_text = pipeline._canonical_body(state)
        assert has_text

        from socr.core.manifest import canonical_page_texts

        for pnum, ptext in zip([1], canonical_page_texts(state)):
            pipeline._flush_page_fragment(state, pnum, ptext, tmp_path)

        stitched = pipeline._stitch_fragments(state, tmp_path)
        assert stitched == in_memory_body


# ---------------------------------------------------------------------------
# Fragment file existence and content
# ---------------------------------------------------------------------------


class TestFragmentFiles:
    def test_fragment_files_created(self, tmp_path: Path) -> None:
        handle = _make_handle(tmp_path, page_count=3)
        state = _make_state_with_texts(handle, _PAGE_TEXTS)
        pipeline = _make_pipeline()
        pipeline._scan_root = tmp_path

        from socr.core.manifest import canonical_page_texts

        for pnum, ptext in zip(range(1, 4), canonical_page_texts(state)):
            pipeline._flush_page_fragment(state, pnum, ptext, tmp_path)

        # Locate the pages/ directory.
        from ocr_output_contract import doc_dir_for, relative_key

        scan_root = tmp_path
        doc_dir = doc_dir_for(tmp_path, relative_key(handle.path, scan_root))
        pages_dir = doc_dir / "pages"

        for pnum in range(1, 4):
            frag = pages_dir / f"{pnum:05d}.md"
            assert frag.exists(), f"Fragment missing: {frag}"

    def test_fragment_body_no_header(self, tmp_path: Path) -> None:
        """Fragment files contain body text only — no '## Page N' header."""
        handle = _make_handle(tmp_path, page_count=2)
        texts = ["Alpha body.", "Beta body."]
        state = _make_state_with_texts(handle, texts)
        pipeline = _make_pipeline()
        pipeline._scan_root = tmp_path

        from socr.core.manifest import canonical_page_texts

        for pnum, ptext in zip(range(1, 3), canonical_page_texts(state)):
            pipeline._flush_page_fragment(state, pnum, ptext, tmp_path)

        from ocr_output_contract import doc_dir_for, relative_key

        doc_dir = doc_dir_for(tmp_path, relative_key(handle.path, tmp_path))
        for pnum, expected in zip(range(1, 3), texts):
            content = (doc_dir / "pages" / f"{pnum:05d}.md").read_text(encoding="utf-8")
            # Body-only: no ## Page marker present.
            assert not PAGE_MARKER_RE.match(content.strip().split("\n")[0])
            assert expected.strip() in content

    def test_fragment_atomic_rename(self, tmp_path: Path) -> None:
        """No .md.tmp file lingers after a successful flush."""
        handle = _make_handle(tmp_path, page_count=1)
        state = _make_state_with_texts(handle, ["Some text."])
        pipeline = _make_pipeline()
        pipeline._scan_root = tmp_path

        from socr.core.manifest import canonical_page_texts

        pipeline._flush_page_fragment(state, 1, canonical_page_texts(state)[0], tmp_path)

        from ocr_output_contract import doc_dir_for, relative_key

        doc_dir = doc_dir_for(tmp_path, relative_key(handle.path, tmp_path))
        pages_dir = doc_dir / "pages"
        # No tmp file should remain.
        tmp_files = list(pages_dir.glob("*.tmp"))
        assert not tmp_files, f"Lingering tmp files: {tmp_files}"


# ---------------------------------------------------------------------------
# Sidecar file existence, content, and atomicity
# ---------------------------------------------------------------------------


class TestSidecarFiles:
    def test_sidecar_files_created(self, tmp_path: Path) -> None:
        handle = _make_handle(tmp_path, page_count=3)
        state = _make_state_with_texts(handle, _PAGE_TEXTS)
        pipeline = _make_pipeline()
        pipeline._scan_root = tmp_path

        for pnum in range(1, 4):
            pipeline._flush_page_sidecar(state, pnum, tmp_path)

        from ocr_output_contract import doc_dir_for, relative_key

        doc_dir = doc_dir_for(tmp_path, relative_key(handle.path, tmp_path))
        for pnum in range(1, 4):
            sc = doc_dir / "pages" / f"{pnum:05d}.json"
            assert sc.exists(), f"Sidecar missing: {sc}"

    def test_sidecar_valid_json(self, tmp_path: Path) -> None:
        handle = _make_handle(tmp_path, page_count=1)
        state = _make_state_with_texts(handle, ["Body text."])
        pipeline = _make_pipeline()
        pipeline._scan_root = tmp_path

        pipeline._flush_page_sidecar(state, 1, tmp_path)

        from ocr_output_contract import doc_dir_for, relative_key

        doc_dir = doc_dir_for(tmp_path, relative_key(handle.path, tmp_path))
        sc_path = doc_dir / "pages" / "00001.json"
        payload = json.loads(sc_path.read_text(encoding="utf-8"))
        # Required top-level keys.
        for key in ("page_num", "status", "terminal", "engine", "run_fingerprint"):
            assert key in payload, f"Sidecar missing key: {key}"
        assert payload["page_num"] == 1
        assert payload["terminal"] is True

    def test_sidecar_decision_flags(self, tmp_path: Path) -> None:
        """PageState decision flags appear in the sidecar."""
        handle = _make_handle(tmp_path, page_count=1)
        state = _make_state_with_texts(handle, ["Body text."])
        # Simulate a judge-rejected page.
        ps: PageState = state.pages[1]
        ps.judge_rejected = True
        ps.native_table_structure_failed = True
        pipeline = _make_pipeline()
        pipeline._scan_root = tmp_path

        pipeline._flush_page_sidecar(state, 1, tmp_path)

        from ocr_output_contract import doc_dir_for, relative_key

        doc_dir = doc_dir_for(tmp_path, relative_key(handle.path, tmp_path))
        payload = json.loads((doc_dir / "pages" / "00001.json").read_text(encoding="utf-8"))
        assert payload["judge_rejected"] is True
        assert payload["native_table_structure_failed"] is True

    def test_sidecar_atomic_rename(self, tmp_path: Path) -> None:
        """No .json.tmp file lingers after a successful sidecar flush."""
        handle = _make_handle(tmp_path, page_count=1)
        state = _make_state_with_texts(handle, ["Text."])
        pipeline = _make_pipeline()
        pipeline._scan_root = tmp_path

        pipeline._flush_page_sidecar(state, 1, tmp_path)

        from ocr_output_contract import doc_dir_for, relative_key

        doc_dir = doc_dir_for(tmp_path, relative_key(handle.path, tmp_path))
        tmp_files = list((doc_dir / "pages").glob("*.tmp"))
        assert not tmp_files, f"Lingering tmp files: {tmp_files}"

    def test_sidecar_no_extra_metadata_json(self, tmp_path: Path) -> None:
        """Fragment flush must NOT create a second metadata.json."""
        handle = _make_handle(tmp_path, page_count=2)
        state = _make_state_with_texts(handle, ["Page one.", "Page two."])
        pipeline = _make_pipeline()
        pipeline._scan_root = tmp_path

        from socr.core.manifest import canonical_page_texts

        for pnum, ptext in zip(range(1, 3), canonical_page_texts(state)):
            pipeline._flush_page_fragment(state, pnum, ptext, tmp_path)
            pipeline._flush_page_sidecar(state, pnum, tmp_path)

        from ocr_output_contract import doc_dir_for, relative_key

        doc_dir = doc_dir_for(tmp_path, relative_key(handle.path, tmp_path))
        # Only pages/ sub-artefacts should exist — no metadata.json.
        assert not (doc_dir / "metadata.json").exists(), (
            "Fragment flush must not create metadata.json"
        )

    def test_sidecar_page_fingerprint_empty_when_no_best_output(self, tmp_path: Path) -> None:
        """page_fingerprint must be "" when a page has no best_output (failed/missing page).

        Regression guard for the `and/or` bug where `winning_dict = {}` (no
        best_output) caused `_page_blob_key({})` to run and return a non-empty
        sha256 hash instead of an empty string.
        """
        handle = _make_handle(tmp_path, page_count=1)
        state = DocumentState(handle=handle)
        # Deliberately leave best_output as None — simulates a page where OCR
        # produced no usable output.
        assert state.pages[1].best_output is None

        pipeline = _make_pipeline()
        pipeline._scan_root = tmp_path

        pipeline._flush_page_sidecar(state, 1, tmp_path)

        from ocr_output_contract import doc_dir_for, relative_key

        doc_dir = doc_dir_for(tmp_path, relative_key(handle.path, tmp_path))
        payload = json.loads((doc_dir / "pages" / "00001.json").read_text(encoding="utf-8"))
        got = payload["page_fingerprint"]
        assert got == "", f"Expected empty page_fingerprint for missing best_output, got: {got!r}"


# ---------------------------------------------------------------------------
# _stitch_fragments edge cases
# ---------------------------------------------------------------------------


class TestStitchFragments:
    def test_empty_pages_dir_returns_empty_string(self, tmp_path: Path) -> None:
        handle = _make_handle(tmp_path, page_count=1)
        state = _make_state_with_texts(handle, ["Text."])
        pipeline = _make_pipeline()
        pipeline._scan_root = tmp_path
        # Do NOT flush any fragments — pages/ dir doesn't exist.
        assert pipeline._stitch_fragments(state, tmp_path) == ""

    def test_stitch_respects_page_order(self, tmp_path: Path) -> None:
        """Pages assembled in ascending numeric order regardless of glob order."""
        handle = _make_handle(tmp_path, page_count=4)
        texts = [f"Page {i} body." for i in range(1, 5)]
        state = _make_state_with_texts(handle, texts)
        pipeline = _make_pipeline()
        pipeline._scan_root = tmp_path

        from socr.core.manifest import canonical_page_texts

        for pnum, ptext in zip(range(1, 5), canonical_page_texts(state)):
            pipeline._flush_page_fragment(state, pnum, ptext, tmp_path)

        stitched = pipeline._stitch_fragments(state, tmp_path)
        # Verify page order by checking that markers appear in ascending order.
        found_nums = [int(m) for m in PAGE_MARKER_RE.findall(stitched)]
        assert found_nums == sorted(found_nums)
        assert found_nums == list(range(1, 5))


# ---------------------------------------------------------------------------
# _page_blob_key
# ---------------------------------------------------------------------------


class TestPageBlobKey:
    def test_deterministic(self) -> None:
        d = {"page_num": 1, "text": "hello", "status": "success"}
        assert _page_blob_key(d) == _page_blob_key(d)

    def test_changes_on_text_change(self) -> None:
        d1 = {"page_num": 1, "text": "hello"}
        d2 = {"page_num": 1, "text": "world"}
        assert _page_blob_key(d1) != _page_blob_key(d2)

    def test_sha256_prefix(self) -> None:
        key = _page_blob_key({"x": 1})
        assert key.startswith("sha256:")

    def test_non_ascii_matches_blobstore_canonicalisation(self) -> None:
        """GH-235: the sidecar key and the BlobStore key must agree on non-ASCII.

        ``_page_blob_key`` used to serialise with ``ensure_ascii=True`` while
        ``core.cache`` uses ``ensure_ascii=False``, so identical logical content
        hashed to two different digests as soon as a page contained an accent, a
        CJK glyph, or a typographic dash — i.e. most of the citation corpus.
        The sidecar's ``sha256:`` prefix is a value-shape difference and is
        stripped before comparing; the DIGEST is what must match.
        """
        payload = {
            "page_num": 1,
            "text": "Rendimiento anual — 中文 — Fernández-Fuertes, café ±0.5%",
            "status": "success",
        }
        assert _page_blob_key(payload) == "sha256:" + blob_hash(payload)

    def test_non_ascii_agrees_with_stored_blob(self) -> None:
        """The digest must also address the bytes BlobStore actually wrote.

        Guards the fix from the other side: matching ``blob_hash`` is only
        useful if ``blob_hash`` is what the store keys by, so round-trip a
        non-ASCII payload through a real store and compare.
        """
        import tempfile

        payload = {"page_num": 2, "text": "café — 中文"}
        with tempfile.TemporaryDirectory() as tmp:
            stored = BlobStore(Path(tmp)).put(payload)
        assert _page_blob_key(payload) == f"sha256:{stored}"

    def test_ascii_payload_unchanged_by_the_fix(self) -> None:
        """Pure-ASCII content is unaffected: ensure_ascii is a no-op there.

        Pins the digest of an ASCII payload so the delegation cannot silently
        change keys for the (overwhelming) ASCII majority of pages.
        """
        payload = {"page_num": 1, "text": "hello", "status": "success"}
        expected = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        assert _page_blob_key(payload) == f"sha256:{expected}"
