"""GH-170: replay must validate visual assets, not just page text blobs.

Before this fix, ``stale_pages``/``replay`` checked ONLY the text ``BlobStore``.
A page's saved markdown can reference a figure/chart/equation-crop PNG on disk
that ``replay`` never looks at, so a missing, corrupted, or relocated asset
produced a document that LOOKED complete (exit 0, every page present) while a
figure silently resolved to nothing -- the exact silent-loss shape this repo's
cardinal rule forbids.

Hermetic: no PDF rendering needed (assets are plain PNG bytes written directly
to the doc dir), no engine/provider involved.
"""

from __future__ import annotations

import pytest

fitz = pytest.importorskip("fitz")  # PyMuPDF; skip whole module if unavailable

from click.testing import CliRunner  # noqa: E402

from socr.cli import replay as replay_cmd  # noqa: E402
from socr.core.cache import BlobStore  # noqa: E402
from socr.core.document import DocumentHandle  # noqa: E402
from socr.core.manifest import (  # noqa: E402
    Manifest,
    build_manifest,
    copy_page_assets,
    stale_assets,
)
from socr.core.result import DocumentStatus, EngineResult, PageOutput, PageStatus  # noqa: E402
from socr.core.state import DocumentState  # noqa: E402

PNG_BYTES = b"\x89PNG\r\n\x1a\nnot a real png, just distinguishable bytes v1"
PNG_BYTES_MODIFIED = b"\x89PNG\r\n\x1a\nnot a real png, just distinguishable bytes v2"


def _make_pdf(path, n_pages=2):
    doc = fitz.open()
    for i in range(n_pages):
        page = doc.new_page()
        page.insert_text((72, 72), f"native text page {i + 1}")
    doc.save(str(path))
    doc.close()
    return path


def _state_with_figure(pdf_path, doc_dir, filename="figure_1_page1.png", png_bytes=PNG_BYTES):
    """A one-page DocumentState whose saved text links a real on-disk PNG."""
    (doc_dir / "figures").mkdir(parents=True, exist_ok=True)
    (doc_dir / "figures" / filename).write_bytes(png_bytes)

    handle = DocumentHandle.from_path(pdf_path)
    state = DocumentState(handle=handle)
    state.apply_result(
        EngineResult(
            document_path=pdf_path,
            engine="gemini",
            status=DocumentStatus.SUCCESS,
            pages=[
                PageOutput(
                    page_num=1,
                    text=f"OCR text\n\n![Figure 1](figures/{filename})",
                    status=PageStatus.SUCCESS,
                    engine="gemini",
                    audit_passed=True,
                )
            ],
        )
    )
    return state


# ---------------------------------------------------------------------------
# build_manifest records asset provenance
# ---------------------------------------------------------------------------


def test_build_manifest_records_asset_hash_and_logical_path(tmp_path):
    pdf = _make_pdf(tmp_path / "paper.pdf", n_pages=1)
    doc_dir = tmp_path / "doc"
    doc_dir.mkdir()
    state = _state_with_figure(pdf, doc_dir)
    store = BlobStore(doc_dir / "cache")

    manifest = build_manifest(state, store, dpi=120, doc_dir=doc_dir)

    assets = manifest.entries[1].assets
    assert len(assets) == 1
    assert assets[0].logical_path == "figures/figure_1_page1.png"
    import hashlib

    assert assets[0].content_hash == hashlib.sha256(PNG_BYTES).hexdigest()


def test_build_manifest_without_doc_dir_records_no_assets(tmp_path):
    """Omitting doc_dir keeps the pre-GH-170 behaviour for callers that don't
    have one (mirrors an old manifest on load)."""
    pdf = _make_pdf(tmp_path / "paper.pdf", n_pages=1)
    doc_dir = tmp_path / "doc"
    doc_dir.mkdir()
    state = _state_with_figure(pdf, doc_dir)
    store = BlobStore(doc_dir / "cache")

    manifest = build_manifest(state, store, dpi=120)  # no doc_dir

    assert manifest.entries[1].assets == []


def test_asset_ref_roundtrips_through_manifest_json(tmp_path):
    pdf = _make_pdf(tmp_path / "paper.pdf", n_pages=1)
    doc_dir = tmp_path / "doc"
    doc_dir.mkdir()
    state = _state_with_figure(pdf, doc_dir)
    store = BlobStore(doc_dir / "cache")
    manifest = build_manifest(state, store, dpi=120, doc_dir=doc_dir)
    manifest.save(doc_dir / "manifest.json")

    reloaded = Manifest.load(doc_dir / "manifest.json")
    assert reloaded.entries[1].assets == manifest.entries[1].assets


# ---------------------------------------------------------------------------
# stale_assets: missing / modified / intact
# ---------------------------------------------------------------------------


def test_stale_assets_empty_when_everything_intact(tmp_path):
    pdf = _make_pdf(tmp_path / "paper.pdf", n_pages=1)
    doc_dir = tmp_path / "doc"
    doc_dir.mkdir()
    state = _state_with_figure(pdf, doc_dir)
    store = BlobStore(doc_dir / "cache")
    manifest = build_manifest(state, store, dpi=120, doc_dir=doc_dir)

    assert stale_assets(manifest, doc_dir) == []


def test_stale_assets_reports_missing(tmp_path):
    pdf = _make_pdf(tmp_path / "paper.pdf", n_pages=1)
    doc_dir = tmp_path / "doc"
    doc_dir.mkdir()
    state = _state_with_figure(pdf, doc_dir)
    store = BlobStore(doc_dir / "cache")
    manifest = build_manifest(state, store, dpi=120, doc_dir=doc_dir)

    (doc_dir / "figures" / "figure_1_page1.png").unlink()

    issues = stale_assets(manifest, doc_dir)
    assert len(issues) == 1
    assert issues[0].kind == "missing"
    assert issues[0].logical_path == "figures/figure_1_page1.png"
    assert issues[0].page_num == 1


def test_stale_assets_reports_modified_not_missing(tmp_path):
    """A corrupted/overwritten asset must be distinguished from a missing one --
    the operator action differs (investigate vs. restore/regenerate)."""
    pdf = _make_pdf(tmp_path / "paper.pdf", n_pages=1)
    doc_dir = tmp_path / "doc"
    doc_dir.mkdir()
    state = _state_with_figure(pdf, doc_dir)
    store = BlobStore(doc_dir / "cache")
    manifest = build_manifest(state, store, dpi=120, doc_dir=doc_dir)

    (doc_dir / "figures" / "figure_1_page1.png").write_bytes(PNG_BYTES_MODIFIED)

    issues = stale_assets(manifest, doc_dir)
    assert len(issues) == 1
    assert issues[0].kind == "modified"
    assert issues[0].logical_path == "figures/figure_1_page1.png"


def test_stale_assets_empty_on_legacy_manifest_with_no_asset_records(tmp_path):
    """A manifest with no recorded assets reports NO issues -- that is 'nothing
    to check', not 'verified intact' (the back-compat decision, GH-170 log)."""
    pdf = _make_pdf(tmp_path / "paper.pdf", n_pages=1)
    doc_dir = tmp_path / "doc"
    doc_dir.mkdir()
    state = _state_with_figure(pdf, doc_dir)
    store = BlobStore(doc_dir / "cache")
    manifest = build_manifest(state, store, dpi=120)  # no doc_dir -> assets == []

    # Even though the PNG on disk was never recorded, deleting it raises no issue.
    (doc_dir / "figures" / "figure_1_page1.png").unlink()
    assert stale_assets(manifest, doc_dir) == []


# ---------------------------------------------------------------------------
# copy_page_assets: relocation
# ---------------------------------------------------------------------------


def test_copy_page_assets_relocates_to_new_directory(tmp_path):
    pdf = _make_pdf(tmp_path / "paper.pdf", n_pages=1)
    doc_dir = tmp_path / "doc"
    doc_dir.mkdir()
    state = _state_with_figure(pdf, doc_dir)
    store = BlobStore(doc_dir / "cache")
    manifest = build_manifest(state, store, dpi=120, doc_dir=doc_dir)

    dest = tmp_path / "elsewhere"
    dest.mkdir()
    copied = copy_page_assets(manifest, doc_dir, dest)

    assert len(copied) == 1
    dst_file = dest / "figures" / "figure_1_page1.png"
    assert dst_file in copied
    assert dst_file.read_bytes() == PNG_BYTES


def test_copy_page_assets_is_noop_for_same_directory(tmp_path):
    pdf = _make_pdf(tmp_path / "paper.pdf", n_pages=1)
    doc_dir = tmp_path / "doc"
    doc_dir.mkdir()
    state = _state_with_figure(pdf, doc_dir)
    store = BlobStore(doc_dir / "cache")
    manifest = build_manifest(state, store, dpi=120, doc_dir=doc_dir)

    assert copy_page_assets(manifest, doc_dir, doc_dir) == []


# ---------------------------------------------------------------------------
# CLI-level: `socr replay` end to end
# ---------------------------------------------------------------------------


def _write_manifest_for_cli(pdf_path, doc_dir):
    state = _state_with_figure(pdf_path, doc_dir)
    store = BlobStore(doc_dir / "cache")
    manifest = build_manifest(state, store, dpi=120, doc_dir=doc_dir)
    manifest.save(doc_dir / "manifest.json")
    return manifest


def test_cli_replay_intact_succeeds_and_copies_nothing_in_place(tmp_path):
    """Negative control: an intact replay in the SAME directory still succeeds
    and rewrites nothing it should not (no asset copy, since source == dest)."""
    pdf = _make_pdf(tmp_path / "paper.pdf", n_pages=1)
    doc_dir = tmp_path / "doc"
    doc_dir.mkdir()
    _write_manifest_for_cli(pdf, doc_dir)

    runner = CliRunner()
    result = runner.invoke(replay_cmd, [str(doc_dir / "manifest.json")])
    assert result.exit_code == 0, result.output
    assert "OCR text" in result.output
    # The only copy of the asset is still the original -- no stray copy created.
    assert list(doc_dir.rglob("*.png")) == [doc_dir / "figures" / "figure_1_page1.png"]


def test_cli_replay_fails_explicitly_on_missing_asset(tmp_path):
    pdf = _make_pdf(tmp_path / "paper.pdf", n_pages=1)
    doc_dir = tmp_path / "doc"
    doc_dir.mkdir()
    _write_manifest_for_cli(pdf, doc_dir)
    (doc_dir / "figures" / "figure_1_page1.png").unlink()

    runner = CliRunner()
    result = runner.invoke(replay_cmd, [str(doc_dir / "manifest.json")])
    assert result.exit_code != 0
    assert "missing" in result.output
    assert "figure_1_page1.png" in result.output


def test_cli_replay_fails_explicitly_on_modified_asset(tmp_path):
    pdf = _make_pdf(tmp_path / "paper.pdf", n_pages=1)
    doc_dir = tmp_path / "doc"
    doc_dir.mkdir()
    _write_manifest_for_cli(pdf, doc_dir)
    (doc_dir / "figures" / "figure_1_page1.png").write_bytes(PNG_BYTES_MODIFIED)

    runner = CliRunner()
    result = runner.invoke(replay_cmd, [str(doc_dir / "manifest.json")])
    assert result.exit_code != 0
    assert "modified" in result.output
    assert "figure_1_page1.png" in result.output


def test_cli_replay_to_another_directory_copies_relocated_asset(tmp_path):
    pdf = _make_pdf(tmp_path / "paper.pdf", n_pages=1)
    doc_dir = tmp_path / "doc"
    doc_dir.mkdir()
    _write_manifest_for_cli(pdf, doc_dir)

    out_dir = tmp_path / "replayed_elsewhere"
    out_md = out_dir / "paper.md"

    runner = CliRunner()
    result = runner.invoke(replay_cmd, [str(doc_dir / "manifest.json"), "-o", str(out_md)])
    assert result.exit_code == 0, result.output
    assert out_md.exists()
    relocated = out_dir / "figures" / "figure_1_page1.png"
    assert relocated.exists()
    assert relocated.read_bytes() == PNG_BYTES
    assert "asset(s) copied" in result.output


def test_cli_replay_to_another_directory_still_fails_on_missing_asset(tmp_path):
    """Relocation must not paper over a broken source -- fail BEFORE copying."""
    pdf = _make_pdf(tmp_path / "paper.pdf", n_pages=1)
    doc_dir = tmp_path / "doc"
    doc_dir.mkdir()
    _write_manifest_for_cli(pdf, doc_dir)
    (doc_dir / "figures" / "figure_1_page1.png").unlink()

    out_dir = tmp_path / "replayed_elsewhere"
    out_md = out_dir / "paper.md"

    runner = CliRunner()
    result = runner.invoke(replay_cmd, [str(doc_dir / "manifest.json"), "-o", str(out_md)])
    assert result.exit_code != 0
    assert not out_dir.exists() or not (out_dir / "figures").exists()
