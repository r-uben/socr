"""GH-964: `socr library` reads the papers library's own config.

Every test builds its own library under ``tmp_path``. The real ``~/papers`` is
never read or written. The pipeline-driving tests pin the engine and patch the
provider ladder / judge resolver (CI has no ollama); library mechanics that do not
need OCR use a fake ``process`` callable that writes what the pipeline writes.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

fitz = pytest.importorskip("fitz")

from socr import cli as socr_cli  # noqa: E402
from socr import library as lib  # noqa: E402

CONFIG = {
    "root": None,  # filled per test
    "input": {"pdf": "pdf"},
    "output": {
        "text": "text",
        "document": {"markdown": "{stem}.md", "figures": "figures", "metadata": "metadata.json"},
    },
    "index": {
        "dir": "index",
        "manifest": "manifest.json",
        "missing_text": "missing_text.txt",
        "documents": "documents.txt",
        "unverified": "unverified.txt",
    },
    "archive": {"dir": "archive"},
    "backup": {"rclone_remote": "gdrive:papers"},
}


def _cfg_dict(root: Path) -> dict:
    d = json.loads(json.dumps(CONFIG))
    d["root"] = str(root)
    return d


def _write_cfg(tmp_path: Path, mutate=None) -> Path:
    root = tmp_path / "lib"
    root.mkdir(exist_ok=True)
    d = _cfg_dict(root)
    if mutate:
        mutate(d)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(d))
    return path


def _pdf(path: Path, text: str = "Estimated coefficient 0.082 significant at 1 percent") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page()
    y = 80
    for _ in range(14):
        page.insert_text((60, y), text, fontsize=9)
        y += 16
    doc.save(str(path))
    doc.close()
    return path


def _fake_doc(out_root: Path, stem: str, *, status="completed", pages=("success",), body="x"):
    d = out_root / stem
    (d / "pages").mkdir(parents=True, exist_ok=True)
    (d / f"{stem}.md").write_text(body)
    (d / "metadata.json").write_text(json.dumps({"status": status}))
    for i, ps in enumerate(pages):
        (d / "pages" / f"{i:05d}.json").write_text(json.dumps({"status": ps}))
    return d


def _fake_process(calls=None, **kw):
    def process(pdf: Path, out_root: Path):
        if calls is not None:
            calls.append((pdf.stem, out_root))
        _fake_doc(out_root, pdf.stem, **kw)

        class R:
            success = True
            status = None

        return R()

    return process


def _snapshot(d: Path) -> dict:
    return {
        str(p.relative_to(d)): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in sorted(d.rglob("*"))
        if p.is_file()
    }


# --- config loading -----------------------------------------------------


def test_load_resolves_relative_to_root_and_expands_tilde(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    path = _write_cfg(tmp_path, lambda d: d.update(root="~/lib"))
    cfg = lib.load_library_config(path)
    assert cfg.root == tmp_path / "lib"
    assert cfg.pdf_dir == tmp_path / "lib" / "pdf"
    assert cfg.text_dir == tmp_path / "lib" / "text"
    assert cfg.manifest == tmp_path / "lib" / "index" / "manifest.json"
    assert cfg.archive_dir == tmp_path / "lib" / "archive"
    assert cfg.staging_dir == tmp_path / "lib" / "index" / lib.DEFAULT_STAGING_NAME
    assert cfg.rclone_remote == "gdrive:papers"


def test_configured_staging_key_wins(tmp_path):
    path = _write_cfg(tmp_path, lambda d: d.update(staging="stage"))
    assert lib.load_library_config(path).staging_dir == tmp_path / "lib" / "stage"


@pytest.mark.parametrize(
    "dotted",
    [
        "root",
        "input.pdf",
        "output.text",
        "output.document.markdown",
        "output.document.figures",
        "output.document.metadata",
        "index.dir",
        "index.manifest",
        "index.missing_text",
        "index.documents",
        "index.unverified",
        "archive.dir",
        "backup.rclone_remote",
    ],
)
def test_missing_key_is_an_error(tmp_path, dotted):
    def drop(d):
        node = d
        *parents, leaf = dotted.split(".")
        for p in parents:
            node = node[p]
        del node[leaf]

    path = _write_cfg(tmp_path, drop)
    with pytest.raises(lib.LibraryConfigError, match=dotted.replace(".", r"\.")):
        lib.load_library_config(path)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["input"].update(pdf="../elsewhere"),
        lambda d: d["output"].update(text="/etc/text"),
        lambda d: d["archive"].update(dir="../../archive"),
        lambda d: d["index"].update(manifest="../manifest.json"),
    ],
)
def test_path_escaping_root_is_an_error(tmp_path, mutate):
    with pytest.raises(lib.LibraryConfigError):
        lib.load_library_config(_write_cfg(tmp_path, mutate))


def test_symlink_escape_is_an_error(tmp_path):
    root = tmp_path / "lib"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "pdf").symlink_to(outside)
    with pytest.raises(lib.LibraryConfigError, match="symlink"):
        lib.load_library_config(_write_cfg(tmp_path))


def test_layout_the_pipeline_cannot_write_is_an_error(tmp_path):
    path = _write_cfg(tmp_path, lambda d: d["output"]["document"].update(markdown="text.md"))
    with pytest.raises(lib.LibraryConfigError, match="cannot be reconciled"):
        lib.load_library_config(path)


def test_missing_config_file_is_an_error(tmp_path):
    with pytest.raises(lib.LibraryConfigError, match="not found"):
        lib.load_library_config(tmp_path / "nope.yaml")


# --- processing mechanics (fake pipeline) --------------------------------


def test_new_pdf_is_processed_into_text(tmp_path):
    cfg = lib.load_library_config(_write_cfg(tmp_path))
    _pdf(cfg.pdf_dir / "a.pdf")
    todo, blocked = lib.pending_pdfs(cfg)
    assert [p.stem for p in todo] == ["a"] and blocked == []
    calls: list = []
    lib.process_new(cfg, _fake_process(calls), todo)
    assert calls == [("a", cfg.text_dir)]
    assert (cfg.text_dir / "a" / "a.md").is_file()


def test_existing_text_dir_is_not_even_listed(tmp_path):
    cfg = lib.load_library_config(_write_cfg(tmp_path))
    _pdf(cfg.pdf_dir / "a.pdf")
    _fake_doc(cfg.text_dir, "a", body="precious")
    assert lib.pending_pdfs(cfg) == ([], [])


def test_incomplete_text_dir_is_blocked_not_processed(tmp_path):
    cfg = lib.load_library_config(_write_cfg(tmp_path))
    _pdf(cfg.pdf_dir / "a.pdf")
    (cfg.text_dir / "a").mkdir(parents=True)
    (cfg.text_dir / "a" / "notes.txt").write_text("hand edited")
    todo, blocked = lib.pending_pdfs(cfg)
    assert todo == [] and [p.stem for p in blocked] == ["a"]


def test_process_new_refuses_an_existing_text_dir(tmp_path):
    """Guard test: process_new must refuse even if the work list was wrong."""
    cfg = lib.load_library_config(_write_cfg(tmp_path))
    pdf = _pdf(cfg.pdf_dir / "a.pdf")
    _fake_doc(cfg.text_dir, "a", body="precious")
    before = _snapshot(cfg.text_dir)
    calls: list = []
    with pytest.raises(lib.LibraryError, match="existing text directory"):
        lib.process_new(cfg, _fake_process(calls), [pdf])
    assert calls == []
    assert _snapshot(cfg.text_dir) == before


def test_rerun_goes_to_staging_and_text_is_untouched(tmp_path):
    cfg = lib.load_library_config(_write_cfg(tmp_path))
    _pdf(cfg.pdf_dir / "a.pdf")
    _fake_doc(cfg.text_dir, "a", body="old")
    before = _snapshot(cfg.text_dir)
    calls: list = []
    lib.rerun(cfg, _fake_process(calls, body="new"), "a")
    assert calls == [("a", cfg.staging_dir)]
    assert (cfg.staging_dir / "a" / "a.md").read_text() == "new"
    assert _snapshot(cfg.text_dir) == before
    assert lib.staged_stems(cfg) == ["a"]
    lib.refresh_index(cfg)
    manifest = json.loads(cfg.manifest.read_text())
    assert manifest["documents"]["a"]["awaiting_approval"] is True


def test_rerun_refuses_to_overwrite_a_staged_run_and_unknown_stem(tmp_path):
    cfg = lib.load_library_config(_write_cfg(tmp_path))
    _pdf(cfg.pdf_dir / "a.pdf")
    _fake_doc(cfg.staging_dir, "a", body="staged")
    with pytest.raises(lib.LibraryError, match="already exists"):
        lib.rerun(cfg, _fake_process(), "a")
    with pytest.raises(lib.LibraryError, match="no PDF"):
        lib.rerun(cfg, _fake_process(), "zzz")
    assert (cfg.staging_dir / "a" / "a.md").read_text() == "staged"


def test_promote_archives_old_dir_and_installs_staged(tmp_path):
    cfg = lib.load_library_config(_write_cfg(tmp_path))
    _pdf(cfg.pdf_dir / "a.pdf")
    _fake_doc(cfg.text_dir, "a", body="old")
    (cfg.text_dir / "a" / "extra.bin").write_bytes(b"\x00\x01")
    _fake_doc(cfg.staging_dir, "a", body="new")
    old = _snapshot(cfg.text_dir / "a")
    from datetime import datetime, timezone

    archived, installed = lib.promote(cfg, "a", now=datetime(2026, 10, 2, tzinfo=timezone.utc))
    assert archived == cfg.archive_dir / "a.2026-10-02"
    assert installed == cfg.text_dir / "a"
    # nothing deleted: the archive holds every old file, byte for byte
    assert {k: v[0] for k, v in _snapshot(archived).items()} == {k: v[0] for k, v in old.items()}
    assert (installed / "a.md").read_text() == "new"
    assert not (cfg.staging_dir / "a").exists()


def test_promote_twice_same_day_never_clobbers_the_archive(tmp_path):
    cfg = lib.load_library_config(_write_cfg(tmp_path))
    from datetime import datetime, timezone

    now = datetime(2026, 10, 2, tzinfo=timezone.utc)
    _fake_doc(cfg.text_dir, "a", body="v1")
    _fake_doc(cfg.staging_dir, "a", body="v2")
    first, _ = lib.promote(cfg, "a", now=now)
    _fake_doc(cfg.staging_dir, "a", body="v3")
    second, _ = lib.promote(cfg, "a", now=now)
    assert first != second
    assert (first / "a.md").read_text() == "v1"
    assert (second / "a.md").read_text() == "v2"
    assert (cfg.text_dir / "a" / "a.md").read_text() == "v3"


def test_promote_refuses_when_nothing_is_staged(tmp_path):
    cfg = lib.load_library_config(_write_cfg(tmp_path))
    _fake_doc(cfg.text_dir, "a", body="old")
    before = _snapshot(cfg.text_dir)
    with pytest.raises(lib.LibraryError, match="nothing staged"):
        lib.promote(cfg, "a")
    assert _snapshot(cfg.text_dir) == before
    assert not cfg.archive_dir.exists()


def test_promote_without_existing_text_just_installs(tmp_path):
    cfg = lib.load_library_config(_write_cfg(tmp_path))
    _fake_doc(cfg.staging_dir, "a", body="new")
    archived, installed = lib.promote(cfg, "a")
    assert archived is None and (installed / "a.md").read_text() == "new"


# --- index --------------------------------------------------------------


def test_index_files_are_correct(tmp_path):
    cfg = lib.load_library_config(_write_cfg(tmp_path))
    for stem in ("clean", "warned", "partial", "absent"):
        _pdf(cfg.pdf_dir / f"{stem}.pdf")
    _fake_doc(cfg.text_dir, "clean")
    _fake_doc(cfg.text_dir, "warned", pages=("success", "warning"))
    _fake_doc(cfg.text_dir, "partial", status="partial")
    info = lib.refresh_index(cfg)
    assert info["missing_text"] == ["absent"]
    assert cfg.missing_text.read_text() == "absent\n"
    assert cfg.unverified.read_text() == "partial\nwarned\n"
    assert cfg.documents.read_text().splitlines() == [
        str((cfg.pdf_dir / f"{s}.pdf").resolve()) for s in ("absent", "clean", "partial", "warned")
    ]
    docs = json.loads(cfg.manifest.read_text())["documents"]
    assert docs["clean"]["status"] == "completed" and docs["clean"]["verified"] is True
    assert docs["warned"]["bad_pages"] == ["00001"]
    assert docs["partial"]["status"] == "partial"
    assert docs["absent"]["status"] == "missing_text"
    assert not list(cfg.index_dir.glob("*.tmp"))


def test_page_error_and_missing_metadata_are_unverified(tmp_path):
    cfg = lib.load_library_config(_write_cfg(tmp_path))
    for stem in ("err", "nometa"):
        _pdf(cfg.pdf_dir / f"{stem}.pdf")
    _fake_doc(cfg.text_dir, "err", pages=("error",))
    _fake_doc(cfg.text_dir, "nometa")
    (cfg.text_dir / "nometa" / "metadata.json").unlink()
    lib.refresh_index(cfg)
    assert cfg.unverified.read_text() == "err\nnometa\n"


def test_index_write_is_atomic(tmp_path, monkeypatch):
    """A failure between temp write and rename leaves the old file intact."""
    cfg = lib.load_library_config(_write_cfg(tmp_path))
    _pdf(cfg.pdf_dir / "a.pdf")
    cfg.index_dir.mkdir(parents=True)
    cfg.missing_text.write_text("old\n")
    real_replace = os.replace
    seen = []

    def boom(src, dst):
        seen.append((Path(src).name, Path(dst).name))
        if Path(dst) == cfg.missing_text:
            raise OSError("simulated crash")
        return real_replace(src, dst)

    monkeypatch.setattr(lib.os, "replace", boom)
    with pytest.raises(OSError):
        lib.refresh_index(cfg)
    assert cfg.missing_text.read_text() == "old\n"
    assert ("missing_text.txt.tmp", "missing_text.txt") in seen


# --- CLI, real pipeline with a pinned engine -----------------------------


@pytest.fixture
def hermetic(monkeypatch):
    from socr.core import providers
    from socr.core.result import PageOutput, PageStatus
    from socr.pipeline.orchestrator import UnifiedPipeline

    monkeypatch.setattr(
        UnifiedPipeline,
        "_available_engines_for_agentic",
        lambda self: [providers.PROFILE_QWEN_LOCAL],
    )
    monkeypatch.setattr(UnifiedPipeline, "_resolve_judge_model", lambda self: "")
    monkeypatch.setattr(UnifiedPipeline, "_resolve_crop_vlm_model", lambda self: None)
    monkeypatch.setattr(
        UnifiedPipeline,
        "_run_engine_on_pages",
        lambda self, state, nums, nat, eng, phase, profile=None: [
            PageOutput(
                page_num=p,
                text=f"text {p}",
                status=PageStatus.SUCCESS,
                engine=str(getattr(eng, "value", eng)),
            )
            for p in nums
        ],
    )
    monkeypatch.setattr(socr_cli, "_report_strict_local_ladder_diagnostic", lambda c: None)
    monkeypatch.setattr(socr_cli, "_report_no_reader_ladder_diagnostic", lambda c: None)


def _run(*args):
    return CliRunner().invoke(socr_cli.cli, ["library", "--primary", "qwen", *args])


def test_socr_is_loaded_from_this_tree():
    import socr

    assert Path(socr.__file__).is_relative_to(Path(__file__).resolve().parents[1] / "src")


def test_cli_processes_new_pdf_and_leaves_existing_alone(tmp_path, hermetic):
    cfg_path = _write_cfg(tmp_path)
    cfg = lib.load_library_config(cfg_path)
    _pdf(cfg.pdf_dir / "new.pdf")
    _pdf(cfg.pdf_dir / "old.pdf")
    _fake_doc(cfg.text_dir, "old", body="precious")
    (cfg.text_dir / "old" / "notes.txt").write_text("hand edited")
    before = _snapshot(cfg.text_dir / "old")

    res = _run("--config", str(cfg_path))
    assert res.exit_code == 0, res.output
    assert (cfg.text_dir / "new" / "new.md").is_file()
    assert (cfg.text_dir / "new" / "metadata.json").is_file()
    assert _snapshot(cfg.text_dir / "old") == before  # bytes AND mtimes
    assert "backup-gdrive" in res.output
    assert "new" not in cfg.missing_text.read_text().split()
    docs = json.loads(cfg.manifest.read_text())["documents"]
    assert set(docs) == {"new", "old"}
    assert docs["new"]["status"] == "completed"


def test_cli_second_run_is_a_noop(tmp_path, hermetic):
    cfg_path = _write_cfg(tmp_path)
    cfg = lib.load_library_config(cfg_path)
    _pdf(cfg.pdf_dir / "a.pdf")
    assert _run("--config", str(cfg_path)).exit_code == 0
    before = _snapshot(cfg.text_dir)
    assert _run("--config", str(cfg_path)).exit_code == 0
    assert _snapshot(cfg.text_dir) == before


def test_cli_rerun_then_promote_with_real_pipeline(tmp_path, hermetic):
    cfg_path = _write_cfg(tmp_path)
    cfg = lib.load_library_config(cfg_path)
    _pdf(cfg.pdf_dir / "a.pdf")
    assert _run("--config", str(cfg_path)).exit_code == 0
    first_md = (cfg.text_dir / "a" / "a.md").read_bytes()
    before = _snapshot(cfg.text_dir / "a")

    res = _run("--config", str(cfg_path), "--rerun", "a")
    assert res.exit_code == 0, res.output
    assert "awaiting approval" in res.output
    assert (cfg.staging_dir / "a" / "a.md").is_file()
    assert _snapshot(cfg.text_dir / "a") == before
    assert json.loads(cfg.manifest.read_text())["documents"]["a"]["awaiting_approval"] is True

    res = _run("--config", str(cfg_path), "--promote", "a")
    assert res.exit_code == 0, res.output
    archived = list(cfg.archive_dir.iterdir())
    assert len(archived) == 1 and (archived[0] / "a.md").read_bytes() == first_md
    assert (cfg.text_dir / "a" / "a.md").is_file()
    assert not (cfg.staging_dir / "a").exists()
    assert json.loads(cfg.manifest.read_text())["documents"]["a"]["awaiting_approval"] is False


def test_cli_promote_with_nothing_staged_fails(tmp_path, hermetic):
    cfg_path = _write_cfg(tmp_path)
    cfg = lib.load_library_config(cfg_path)
    _fake_doc(cfg.text_dir, "a", body="old")
    before = _snapshot(cfg.text_dir)
    res = _run("--config", str(cfg_path), "--promote", "a")
    assert res.exit_code != 0 and "nothing staged" in res.output
    assert _snapshot(cfg.text_dir) == before
    assert not cfg.archive_dir.exists()


def test_cli_config_error_is_loud(tmp_path, hermetic):
    cfg_path = _write_cfg(tmp_path, lambda d: d["archive"].pop("dir"))
    res = _run("--config", str(cfg_path))
    assert res.exit_code != 0 and "archive.dir" in res.output


def test_cli_dry_run_writes_nothing(tmp_path, hermetic):
    cfg_path = _write_cfg(tmp_path)
    cfg = lib.load_library_config(cfg_path)
    _pdf(cfg.pdf_dir / "a.pdf")
    root = tmp_path / "lib"
    before = {p: p.stat().st_mtime_ns for p in root.rglob("*")}
    for extra in ([], ["--rerun", "a"], ["--promote", "a"]):
        res = _run("--config", str(cfg_path), "--dry-run", *extra)
        assert res.exit_code == 0, res.output
    assert "would process" in _run("--config", str(cfg_path), "--dry-run").output
    assert {p: p.stat().st_mtime_ns for p in root.rglob("*")} == before
    assert not cfg.text_dir.exists() and not cfg.index_dir.exists()


def test_cli_uses_the_batch_pipeline_entry(tmp_path, hermetic, monkeypatch):
    """The library calls UnifiedPipeline.process, the call `batch` makes per file."""
    from socr.pipeline.orchestrator import UnifiedPipeline

    cfg_path = _write_cfg(tmp_path)
    cfg = lib.load_library_config(cfg_path)
    _pdf(cfg.pdf_dir / "a.pdf")
    seen = []
    real = UnifiedPipeline.process

    def spy(self, pdf_path, output_dir=None, scan_root=None):
        seen.append((Path(pdf_path).name, Path(output_dir), scan_root))
        return real(self, pdf_path, output_dir, scan_root)

    monkeypatch.setattr(UnifiedPipeline, "process", spy)
    assert _run("--config", str(cfg_path)).exit_code == 0
    assert seen == [("a.pdf", cfg.text_dir, cfg.pdf_dir)]
