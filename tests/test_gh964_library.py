"""GH-964: `socr library` reads the papers library's own config.

Every test builds its own library under ``tmp_path``. The real ``~/papers`` is
never read or written. The pipeline-driving tests pin the engine and patch the
provider ladder / judge resolver (CI has no ollama); library mechanics that do not
need OCR use a fake ``process`` callable that writes what the pipeline writes.
"""

from __future__ import annotations

import json
import os
import shutil
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
    assert cfg.staging_dir == tmp_path / "lib" / lib.DEFAULT_STAGING_NAME
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
    out = lib.process_new(cfg, _fake_process(calls), todo)
    assert calls == [("a", cfg.staging_dir)]  # processed in staging, then moved
    assert [(s, st) for s, st, _ in out] == [("a", lib.INSTALLED)]
    assert (cfg.text_dir / "a" / "a.md").is_file()
    assert not (cfg.staging_dir / "a").exists()


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
    assert docs["clean"]["status"] == "completed" and docs["clean"]["state"] == "verified"
    assert docs["warned"]["bad_pages"] == ["00001"]
    assert docs["partial"]["status"] == "partial"
    assert docs["absent"]["status"] == "missing_text"
    assert not list(cfg.index_dir.glob("*.tmp"))


def test_page_error_is_unverified_missing_metadata_is_unknown(tmp_path):
    cfg = lib.load_library_config(_write_cfg(tmp_path))
    for stem in ("err", "nometa"):
        _pdf(cfg.pdf_dir / f"{stem}.pdf")
    _fake_doc(cfg.text_dir, "err", pages=("error",))
    _fake_doc(cfg.text_dir, "nometa")
    (cfg.text_dir / "nometa" / "metadata.json").unlink()
    lib.refresh_index(cfg)
    assert cfg.unverified.read_text() == "err\n"
    assert json.loads(cfg.manifest.read_text())["documents"]["nometa"]["state"] == "unknown"


def _legacy_library(tmp_path):
    """Legacy metadata (no status), a curated unverified.txt entry, a marker dir."""
    cfg = lib.load_library_config(_write_cfg(tmp_path))
    for stem in ("legacy", "curated", "marked", "fresh", "gone"):
        _pdf(cfg.pdf_dir / f"{stem}.pdf")
    for stem in ("legacy", "curated", "marked"):
        _fake_doc(cfg.text_dir, stem)
        (cfg.text_dir / stem / "metadata.json").write_text(json.dumps({"model": "old"}))
    (cfg.text_dir / "marked" / "UNVERIFIED.txt").write_text("hand placed")
    cfg.index_dir.mkdir(parents=True)
    cfg.unverified.write_text("curated\nnot-in-library\n")
    return cfg


def test_legacy_status_is_unknown_and_curated_entries_survive(tmp_path):
    cfg = _legacy_library(tmp_path)
    marker_before = _snapshot(cfg.text_dir / "marked")
    lib.refresh_index(cfg)
    entries = cfg.unverified.read_text().split()
    assert entries == ["curated", "marked", "not-in-library"]  # legacy NOT listed
    docs = json.loads(cfg.manifest.read_text())["documents"]
    assert docs["legacy"]["state"] == "unknown" and docs["legacy"]["status"] == "unknown"
    assert docs["curated"]["state"] == "unknown"
    assert docs["marked"]["state"] == "unverified"
    assert _snapshot(cfg.text_dir / "marked") == marker_before  # marker never touched


def test_curated_entry_leaves_only_when_socr_processed_it_clean(tmp_path):
    cfg = _legacy_library(tmp_path)
    lib.refresh_index(cfg, frozenset({"legacy"}))  # processed, but was never listed
    assert "curated" in cfg.unverified.read_text().split()
    # socr re-processed 'curated' and it came out completed: it leaves the list
    (cfg.text_dir / "curated" / "metadata.json").write_text(json.dumps({"status": "completed"}))
    lib.refresh_index(cfg)  # not processed this run: curated entry stays
    assert "curated" in cfg.unverified.read_text().split()
    lib.refresh_index(cfg, frozenset({"curated"}))
    assert "curated" not in cfg.unverified.read_text().split()
    assert "not-in-library" in cfg.unverified.read_text().split()


def test_processed_but_still_bad_stays_listed(tmp_path):
    cfg = _legacy_library(tmp_path)
    (cfg.text_dir / "curated" / "metadata.json").write_text(json.dumps({"status": "partial"}))
    lib.refresh_index(cfg, frozenset({"curated"}))
    assert "curated" in cfg.unverified.read_text().split()


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
    assert seen and not list(cfg.index_dir.glob("*.tmp"))  # no stray temp files


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
    _fake_doc(cfg.staging_dir, "b", body="staged")  # so "--promote b" has something to preview
    root = tmp_path / "lib"
    before = {p: p.stat().st_mtime_ns for p in root.rglob("*")}
    for extra in ([], ["--rerun", "a"], ["--promote", "b"]):
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
    assert seen == [("a.pdf", cfg.staging_dir, cfg.pdf_dir)]


# --- GH-964 review: data-safety guards ------------------------------------


def test_second_instance_refuses(tmp_path, hermetic):
    cfg_path = _write_cfg(tmp_path)
    cfg = lib.load_library_config(cfg_path)
    _pdf(cfg.pdf_dir / "a.pdf")
    with lib.library_lock(cfg):
        with pytest.raises(lib.LibraryError, match="another"):
            with lib.library_lock(cfg):
                pass
        res = _run("--config", str(cfg_path))
        assert res.exit_code != 0 and "another" in res.output
        assert not cfg.text_dir.exists()
    assert _run("--config", str(cfg_path)).exit_code == 0  # released on exit


def test_unreadable_curated_list_aborts_the_refresh(tmp_path):
    cfg = lib.load_library_config(_write_cfg(tmp_path))
    _pdf(cfg.pdf_dir / "a.pdf")
    cfg.index_dir.mkdir(parents=True)
    cfg.unverified.write_bytes(b"\xff\xfe not utf-8 \x80")
    with pytest.raises(lib.LibraryError, match="curated list"):
        lib.refresh_index(cfg)
    assert not cfg.documents.exists() and not cfg.manifest.exists()
    assert cfg.unverified.read_bytes() == b"\xff\xfe not utf-8 \x80"


def test_unreadable_curated_list_aborts_via_cli(tmp_path, hermetic):
    cfg_path = _write_cfg(tmp_path)
    cfg = lib.load_library_config(cfg_path)
    _pdf(cfg.pdf_dir / "a.pdf")
    cfg.unverified.parent.mkdir(parents=True)
    cfg.unverified.mkdir()  # a directory where the file should be: unreadable
    res = _run("--config", str(cfg_path))
    assert res.exit_code != 0 and "curated list" in res.output


def test_index_write_refuses_a_symlink_target(tmp_path):
    cfg = lib.load_library_config(_write_cfg(tmp_path))
    _pdf(cfg.pdf_dir / "a.pdf")
    cfg.index_dir.mkdir(parents=True)
    victim = tmp_path / "victim.txt"
    victim.write_text("keep")
    cfg.missing_text.symlink_to(victim)
    with pytest.raises(lib.LibraryError, match="symlink"):
        lib.refresh_index(cfg)
    assert victim.read_text() == "keep"


def test_temp_names_are_unique(tmp_path, monkeypatch):
    cfg = lib.load_library_config(_write_cfg(tmp_path))
    _pdf(cfg.pdf_dir / "a.pdf")
    names = []
    real = os.replace
    monkeypatch.setattr(lib.os, "replace", lambda a, b: (names.append(Path(a).name), real(a, b)))
    lib.refresh_index(cfg)
    lib.refresh_index(cfg)
    assert len(names) == len(set(names)) == 8


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["index"].update(manifest="Same.txt", documents="same.TXT"),
        lambda d: d["index"].update(documents="a.txt", missing_text="a.txt"),
        lambda d: d["index"].update(manifest=lib.LOCK_NAME.upper()),
        lambda d: d["index"].update(unverified=lib.JOURNAL_NAME),
    ],
)
def test_duplicate_or_aliased_index_names_are_rejected(tmp_path, mutate):
    with pytest.raises(lib.LibraryConfigError, match="same file name"):
        lib.load_library_config(_write_cfg(tmp_path, mutate))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["archive"].update(dir="text/archive"),  # archive inside text
        lambda d: d["output"].update(text="index/text"),  # text inside index
        lambda d: d.update(staging="index/stage"),  # staging inside index
        lambda d: d.update(staging="text"),  # staging IS text
        lambda d: d["index"].update(dir=lib.DEFAULT_STAGING_NAME),  # default staging == index
        lambda d: d["input"].update(pdf="text/pdf"),
        lambda d: d["archive"].update(dir="."),  # archive is root: contains everything
    ],
)
def test_overlapping_dirs_are_rejected(tmp_path, mutate):
    with pytest.raises(lib.LibraryConfigError, match="same directory or nest"):
        lib.load_library_config(_write_cfg(tmp_path, mutate))


def test_symlink_alias_overlap_is_rejected(tmp_path):
    root = tmp_path / "lib"
    (root / "text").mkdir(parents=True)
    (root / "alias").symlink_to(root / "text")
    with pytest.raises(lib.LibraryConfigError, match="same directory or nest"):
        lib.load_library_config(_write_cfg(tmp_path, lambda d: d["archive"].update(dir="alias")))


def test_crash_mid_process_leaves_text_untouched(tmp_path):
    cfg = lib.load_library_config(_write_cfg(tmp_path))
    for stem in ("boom", "empty", "ok", "old"):
        _pdf(cfg.pdf_dir / f"{stem}.pdf")
    _fake_doc(cfg.text_dir, "old", body="precious")
    old = _snapshot(cfg.text_dir / "old")
    good = _fake_process()

    def process(pdf, out_root):
        if pdf.stem == "boom":
            _fake_doc(out_root, "boom", body="half")
            raise RuntimeError("pipeline crashed")
        if pdf.stem == "empty":  # finished without producing markdown
            (out_root / "empty").mkdir(parents=True)
            return good(pdf, out_root / "elsewhere")
        return good(pdf, out_root)

    todo, _ = lib.pending_pdfs(cfg)
    out = {s: st for s, st, _ in lib.process_new(cfg, process, todo)}
    assert out == {"boom": lib.FAILED, "empty": lib.FAILED, "ok": lib.INSTALLED}
    assert not (cfg.text_dir / "boom").exists() and not (cfg.text_dir / "empty").exists()
    assert (cfg.staging_dir / "boom" / "boom.md").read_text() == "half"  # leftovers kept
    assert _snapshot(cfg.text_dir / "old") == old
    # a retry reports the leftovers instead of overwriting them
    again = {s: st for s, st, _ in lib.process_new(cfg, process, [cfg.pdf_dir / "boom.pdf"])}
    assert again == {"boom": lib.BLOCKED}


def test_partial_run_is_installed_and_listed_unverified(tmp_path):
    cfg = lib.load_library_config(_write_cfg(tmp_path))
    _pdf(cfg.pdf_dir / "p.pdf")
    out = lib.process_new(cfg, _fake_process(status="partial"), [cfg.pdf_dir / "p.pdf"])
    assert out[0][1] == lib.INSTALLED
    lib.refresh_index(cfg, frozenset({"p"}))
    assert cfg.unverified.read_text() == "p\n"


def test_cli_crash_mid_process_exits_nonzero_text_untouched(tmp_path, hermetic, monkeypatch):
    from socr.pipeline.orchestrator import UnifiedPipeline

    cfg_path = _write_cfg(tmp_path)
    cfg = lib.load_library_config(cfg_path)
    _pdf(cfg.pdf_dir / "a.pdf")

    def crash(self, *a, **k):
        raise RuntimeError("model wedged")

    monkeypatch.setattr(UnifiedPipeline, "process", crash)
    res = _run("--config", str(cfg_path))
    assert res.exit_code != 0 and "failed a" in res.output
    assert not (cfg.text_dir / "a").exists()
    assert cfg.missing_text.read_text() == "a\n"


def _staged_promotion(tmp_path):
    cfg = lib.load_library_config(_write_cfg(tmp_path))
    _pdf(cfg.pdf_dir / "a.pdf")
    _fake_doc(cfg.text_dir, "a", body="old")
    _fake_doc(cfg.staging_dir, "a", body="new")
    return cfg


def _crash_on_nth_rename(monkeypatch, n):
    real = lib._rename_noreplace
    count = {"n": 0}

    def flaky(src, dst):
        count["n"] += 1
        if count["n"] == n:
            raise RuntimeError("simulated crash")
        return real(src, dst)

    monkeypatch.setattr(lib, "_rename_noreplace", flaky)


def test_crash_between_the_two_renames_recovers_from_the_journal(tmp_path, monkeypatch):
    cfg = _staged_promotion(tmp_path)
    _crash_on_nth_rename(monkeypatch, 2)
    with pytest.raises(RuntimeError):
        lib.promote(cfg, "a")
    assert not (cfg.text_dir / "a").exists()  # the dangerous window
    assert (cfg.index_dir / lib.JOURNAL_NAME).exists()
    monkeypatch.undo()
    msg = lib.recover_promotion(cfg)
    assert msg and "recovered" in msg
    assert (cfg.text_dir / "a" / "a.md").read_text() == "new"
    assert len(list(cfg.archive_dir.iterdir())) == 1
    assert (next(cfg.archive_dir.iterdir()) / "a.md").read_text() == "old"
    assert not (cfg.index_dir / lib.JOURNAL_NAME).exists()
    assert lib.recover_promotion(cfg) is None


def test_crash_before_any_rename_discards_the_journal(tmp_path, monkeypatch):
    cfg = _staged_promotion(tmp_path)
    _crash_on_nth_rename(monkeypatch, 1)
    with pytest.raises(RuntimeError):
        lib.promote(cfg, "a")
    monkeypatch.undo()
    assert lib.recover_promotion(cfg)
    assert (cfg.text_dir / "a" / "a.md").read_text() == "old"
    assert (cfg.staging_dir / "a" / "a.md").read_text() == "new"
    assert not (cfg.index_dir / lib.JOURNAL_NAME).exists()


def test_cli_recovers_a_journal_before_doing_anything_else(tmp_path, hermetic, monkeypatch):
    cfg = _staged_promotion(tmp_path)
    cfg_path = tmp_path / "config.yaml"
    _crash_on_nth_rename(monkeypatch, 2)
    with pytest.raises(RuntimeError):
        lib.promote(cfg, "a")
    monkeypatch.undo()
    res = _run("--config", str(cfg_path))
    assert res.exit_code == 0, res.output
    assert "recovered" in res.output
    assert (cfg.text_dir / "a" / "a.md").read_text() == "new"


def test_unreadable_journal_aborts(tmp_path):
    cfg = lib.load_library_config(_write_cfg(tmp_path))
    cfg.index_dir.mkdir(parents=True)
    (cfg.index_dir / lib.JOURNAL_NAME).write_text("{not json")
    with pytest.raises(lib.LibraryError, match="journal"):
        lib.recover_promotion(cfg)


def test_rename_refuses_an_existing_target(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    with pytest.raises(lib.LibraryError, match="refusing to replace"):
        lib._rename_noreplace(a, b)
    assert a.exists() and b.exists()


def test_stem_collisions_are_refused(tmp_path, hermetic, monkeypatch):
    cfg_path = _write_cfg(tmp_path)
    cfg = lib.load_library_config(cfg_path)
    # A case-insensitive filesystem cannot hold both files; fake the listing.
    fake = [cfg.pdf_dir / "Paper.pdf", cfg.pdf_dir / "paper.PDF", cfg.pdf_dir / "z.pdf"]
    monkeypatch.setattr(lib, "list_pdfs", lambda c: fake)
    with pytest.raises(lib.LibraryError, match="Paper.pdf / paper.PDF"):
        lib.check_stem_collisions(cfg)
    res = _run("--config", str(cfg_path))
    assert res.exit_code != 0 and "collide" in res.output
    assert not cfg.text_dir.exists() and not cfg.staging_dir.exists()
    assert not cfg.index_dir.exists()


# --- GH-964 review round 3: atomic rename, journal validation, durability ---


def test_noreplace_rename_refuses_an_empty_target_and_keeps_the_source(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    (src / "f.txt").write_text("data")
    dst.mkdir()  # plain os.rename() would silently replace an EMPTY directory
    with pytest.raises(lib.LibraryError, match="refusing to replace"):
        lib._rename_noreplace(src, dst)
    assert (src / "f.txt").read_text() == "data" and dst.exists() and not any(dst.iterdir())


@pytest.mark.skipif(lib._native_noreplace() is None, reason="no kernel no-replace rename here")
def test_noreplace_is_atomic_not_check_then_rename(tmp_path, monkeypatch):
    """Simulate the race: the pre-check sees 'absent', the target appears anyway."""
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    (src / "f.txt").write_text("data")
    dst.mkdir()
    monkeypatch.setattr(lib.os.path, "lexists", lambda p: False)
    with pytest.raises(lib.LibraryError, match="refusing to replace"):
        lib._rename_noreplace(src, dst)
    assert (src / "f.txt").read_text() == "data" and not any(dst.iterdir())


def test_install_never_replaces_an_empty_text_dir(tmp_path):
    cfg = lib.load_library_config(_write_cfg(tmp_path))
    _fake_doc(cfg.staging_dir, "a", body="new")
    (cfg.text_dir / "a").mkdir(parents=True)  # empty dir appeared meanwhile
    with pytest.raises(lib.LibraryError, match="refusing to replace"):
        lib.install_staged(cfg, "a")
    assert (cfg.staging_dir / "a" / "a.md").read_text() == "new"


def _write_journal(cfg, **fields):
    cfg.index_dir.mkdir(parents=True, exist_ok=True)
    j = {"stem": "a", "target": None, "staged": None, "archived": None}
    j.update(fields)
    (cfg.index_dir / lib.JOURNAL_NAME).write_text(json.dumps(j))


def test_journal_naming_paths_outside_the_library_is_refused(tmp_path):
    cfg = lib.load_library_config(_write_cfg(tmp_path))
    outside = tmp_path / "outside"
    _fake_doc(outside, "a", body="not yours")
    _fake_doc(cfg.staging_dir, "a", body="new")
    before = _snapshot(outside)
    for fields in (
        {"target": str(outside / "a"), "staged": str(cfg.staging_dir / "a")},
        {"target": str(cfg.text_dir / "a"), "staged": str(outside / "a")},
        {
            "target": str(cfg.text_dir / "a"),
            "staged": str(cfg.staging_dir / "a"),
            "archived": str(outside / "a.2026-10-02"),
        },
        {"target": str(cfg.text_dir / ".." / ".." / "a"), "staged": str(cfg.staging_dir / "a")},
    ):
        _write_journal(cfg, **fields)
        with pytest.raises(lib.LibraryError, match=lib.JOURNAL_NAME):
            lib.recover_promotion(cfg)
        assert (cfg.index_dir / lib.JOURNAL_NAME).exists()  # left for a human
    assert _snapshot(outside) == before
    assert (cfg.staging_dir / "a" / "a.md").read_text() == "new"
    assert not (cfg.text_dir / "a").exists()


def test_journal_naming_a_symlink_is_refused(tmp_path):
    cfg = lib.load_library_config(_write_cfg(tmp_path))
    outside = tmp_path / "outside"
    _fake_doc(outside, "a", body="not yours")
    cfg.staging_dir.mkdir(parents=True)
    (cfg.staging_dir / "a").symlink_to(outside / "a")
    _write_journal(cfg, target=str(cfg.text_dir / "a"), staged=str(cfg.staging_dir / "a"))
    with pytest.raises(lib.LibraryError, match="which is a symlink"):
        lib.recover_promotion(cfg)
    assert not (cfg.text_dir / "a").exists()
    assert (outside / "a" / "a.md").read_text() == "not yours"


def test_journal_with_a_staged_dir_missing_its_markdown_is_refused(tmp_path):
    cfg = lib.load_library_config(_write_cfg(tmp_path))
    (cfg.staging_dir / "a").mkdir(parents=True)
    (cfg.staging_dir / "a" / "junk.txt").write_text("x")
    _write_journal(cfg, target=str(cfg.text_dir / "a"), staged=str(cfg.staging_dir / "a"))
    with pytest.raises(lib.LibraryError, match="a.md"):
        lib.recover_promotion(cfg)
    assert not (cfg.text_dir / "a").exists()


def test_promote_fsyncs_in_the_durable_order(tmp_path, monkeypatch):
    cfg = _staged_promotion(tmp_path)
    jp = cfg.index_dir / lib.JOURNAL_NAME
    events = []
    real_fsync_dir, real_rename = lib._fsync_dir, lib._rename_noreplace
    monkeypatch.setattr(
        lib,
        "_fsync_dir",
        lambda p: (events.append(("fsync", Path(p), jp.exists())), real_fsync_dir(p))[1],
    )
    monkeypatch.setattr(
        lib,
        "_rename_noreplace",
        lambda s, d: (events.append(("rename", Path(d), jp.exists())), real_rename(s, d))[1],
    )
    lib.promote(cfg, "a")
    kinds = [e[0] for e in events]
    first_rename = kinds.index("rename")
    # journal directory fsynced (with the journal present) before the first rename
    assert ("fsync", cfg.index_dir, True) in events[:first_rename]
    # after each rename the parent dirs are fsynced before the next rename
    renames = [i for i, k in enumerate(kinds) if k == "rename"]
    assert len(renames) == 2
    between = events[renames[0] + 1 : renames[1]]
    assert {e[1] for e in between} >= {cfg.archive_dir, cfg.text_dir}
    after = events[renames[1] + 1 :]
    assert {e[1] for e in after} >= {cfg.text_dir, cfg.staging_dir}
    # the journal is deleted only after that, then its directory is fsynced
    assert events[-1] == ("fsync", cfg.index_dir, False)
    start = events.index(("fsync", cfg.index_dir, True))
    assert all(e[2] for e in events[start:-1])  # journal present until the last step


def test_journal_naming_a_symlink_inside_the_library_is_refused(tmp_path):
    cfg = lib.load_library_config(_write_cfg(tmp_path))
    _fake_doc(cfg.staging_dir, "b", body="other paper")
    (cfg.staging_dir / "a").symlink_to(cfg.staging_dir / "b")  # resolves inside staging
    _write_journal(cfg, target=str(cfg.text_dir / "a"), staged=str(cfg.staging_dir / "a"))
    with pytest.raises(lib.LibraryError, match="which is a symlink"):
        lib.recover_promotion(cfg)
    assert not (cfg.text_dir / "a").exists()


def test_recovery_rename_goes_through_the_no_replace_primitive(tmp_path, monkeypatch):
    cfg = _staged_promotion(tmp_path)
    _crash_on_nth_rename(monkeypatch, 2)
    with pytest.raises(RuntimeError):
        lib.promote(cfg, "a")
    monkeypatch.undo()
    calls = []
    real = lib._rename_noreplace
    monkeypatch.setattr(lib, "_rename_noreplace", lambda s, d: (calls.append(Path(d)), real(s, d)))
    assert lib.recover_promotion(cfg)
    assert calls == [cfg.text_dir / "a"]


# --- GH-964 review round 4: remaining durability gaps ----------------------


def _spy_fsync(monkeypatch, cfg):
    jp = cfg.index_dir / lib.JOURNAL_NAME
    events = []
    real = lib._fsync_dir
    monkeypatch.setattr(
        lib, "_fsync_dir", lambda p: (events.append((Path(p), jp.exists())), real(p))[1]
    )
    return events


def test_completed_promotion_recovery_fsyncs_dirs_before_dropping_the_journal(
    tmp_path, monkeypatch
):
    cfg = _staged_promotion(tmp_path)
    archived, _ = lib.promote(cfg, "a")
    # crash after both renames, before the journal was removed
    _write_journal(
        cfg,
        target=str(cfg.text_dir / "a"),
        staged=str(cfg.staging_dir / "a"),
        archived=str(archived),
    )
    events = _spy_fsync(monkeypatch, cfg)
    assert "completed" in lib.recover_promotion(cfg)
    with_journal = {p for p, present in events if present}
    assert {cfg.text_dir, cfg.staging_dir, cfg.archive_dir} <= with_journal
    assert events[-1] == (cfg.index_dir, False)  # journal's dir fsynced after the unlink


def test_creating_archive_and_text_dirs_fsyncs_their_parent(tmp_path, monkeypatch):
    cfg = lib.load_library_config(_write_cfg(tmp_path))
    _fake_doc(cfg.staging_dir, "a", body="new")
    _fake_doc(cfg.staging_dir, "b", body="new")
    events = _spy_fsync(monkeypatch, cfg)
    lib.install_staged(cfg, "a")  # creates text_dir under root
    assert cfg.root in [p for p, _ in events]
    # promote of a stem with an existing text dir creates archive_dir under root
    events.clear()
    _fake_doc(cfg.staging_dir, "a", body="newer")
    lib.promote(cfg, "a")
    assert cfg.root in [p for p, _ in events]
    assert cfg.archive_dir.is_dir()


def _fake_native(fail_errno, probe_works):
    """A kernel-primitive stand-in: fails the real call with fail_errno."""

    def native(s, d):
        if probe_works and b".noreplace-probe." in s:
            os.rename(s, d)
            return 0
        lib.ctypes.set_errno(fail_errno)
        return -1

    return native


@pytest.mark.parametrize("err", [lib.errno.ENOTSUP, lib.errno.EOPNOTSUPP, lib.errno.ENOSYS])
def test_unsupported_errnos_fall_back_to_the_guarded_rename(tmp_path, monkeypatch, err):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    monkeypatch.setattr(lib, "_native_noreplace", lambda: _fake_native(err, probe_works=False))
    lib._rename_noreplace(src, dst)
    assert dst.is_dir() and not src.exists()
    # the fallback still refuses an existing target
    src.mkdir()
    with pytest.raises(lib.LibraryError, match="refusing to replace"):
        lib._rename_noreplace(src, dst)


def test_einval_is_raised_when_the_probe_shows_the_primitive_works(tmp_path, monkeypatch):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    (src / "f").write_text("keep")
    monkeypatch.setattr(
        lib, "_native_noreplace", lambda: _fake_native(lib.errno.EINVAL, probe_works=True)
    )
    with pytest.raises(OSError) as ei:
        lib._rename_noreplace(src, dst)
    assert not isinstance(ei.value, lib.LibraryError) and ei.value.errno == lib.errno.EINVAL
    assert (src / "f").read_text() == "keep" and not dst.exists()
    assert not list(tmp_path.glob(".noreplace-probe.*"))  # scratch dirs cleaned up


def test_einval_falls_back_only_when_the_probe_proves_it_unsupported(tmp_path, monkeypatch):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    monkeypatch.setattr(
        lib, "_native_noreplace", lambda: _fake_native(lib.errno.EINVAL, probe_works=False)
    )
    lib._rename_noreplace(src, dst)
    assert dst.is_dir() and not src.exists()


def test_nested_archive_dir_creation_fsyncs_each_new_parent(tmp_path, monkeypatch):
    cfg = lib.load_library_config(
        _write_cfg(tmp_path, lambda d: d["archive"].update(dir="arch/old"))
    )
    _fake_doc(cfg.text_dir, "a", body="old")
    _fake_doc(cfg.staging_dir, "a", body="new")
    events = _spy_fsync(monkeypatch, cfg)
    lib.promote(cfg, "a")
    synced = [p for p, _ in events]
    assert cfg.root / "arch" in synced  # parent of the new archive dir
    assert cfg.root in synced  # parent of the new 'arch'


@pytest.mark.parametrize("probe_err", [lib.errno.EACCES, lib.errno.EIO])
def test_a_probe_failing_for_another_reason_does_not_fall_back(tmp_path, monkeypatch, probe_err):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    (src / "f").write_text("keep")

    def native(s, d):
        # the real call and the probe both fail; the probe with EACCES/EIO
        lib.ctypes.set_errno(probe_err if b".noreplace-probe." in s else lib.errno.EINVAL)
        return -1

    monkeypatch.setattr(lib, "_native_noreplace", lambda: native)
    with pytest.raises(OSError) as ei:
        lib._rename_noreplace(src, dst)
    assert ei.value.errno == probe_err and not isinstance(ei.value, lib.LibraryError)
    assert (src / "f").read_text() == "keep" and not dst.exists()
    assert not list(tmp_path.glob(".noreplace-probe.*"))


def test_recovery_creates_a_missing_text_dir_before_rolling_forward(tmp_path):
    # cubic on #966: promote writes its journal before creating text_dir, so a crash in
    # between leaves a journal whose target parent does not exist yet. Recovery must
    # create it (durably) rather than fail the roll-forward.
    cfg = lib.load_library_config(_write_cfg(tmp_path))
    _pdf(cfg.pdf_dir / "a.pdf")
    _fake_doc(cfg.staging_dir, "a", body="new")
    if cfg.text_dir.exists():
        shutil.rmtree(cfg.text_dir)
    _write_journal(
        cfg, target=str(cfg.text_dir / "a"), staged=str(cfg.staging_dir / "a"), archived=None
    )
    msg = lib.recover_promotion(cfg)
    assert msg and "recovered" in msg
    assert (cfg.text_dir / "a" / "a.md").read_text() == "new"
    assert not (cfg.index_dir / lib.JOURNAL_NAME).exists()


# --- #969-#973: library hardening -------------------------------------------


@pytest.mark.parametrize("bad", ["../x", "a/b", ".", "..", "/etc/passwd", "a\\b", ""])
def test_969_pathy_stems_are_refused_before_any_rename(tmp_path, bad):
    cfg = lib.load_library_config(_write_cfg(tmp_path))
    _pdf(cfg.pdf_dir / "a.pdf")
    _fake_doc(cfg.text_dir, "a", body="old")
    _fake_doc(cfg.staging_dir, "a", body="new")
    snap = (_snapshot(cfg.text_dir), _snapshot(cfg.staging_dir), set(tmp_path.rglob("*")))
    with pytest.raises(lib.LibraryError, match="invalid document stem"):
        lib.promote(cfg, bad)
    with pytest.raises(lib.LibraryError, match="invalid document stem"):
        lib.rerun(cfg, _fake_process(), bad)
    assert snap == (_snapshot(cfg.text_dir), _snapshot(cfg.staging_dir), set(tmp_path.rglob("*")))


def test_969_a_bare_basename_still_promotes(tmp_path):
    cfg = lib.load_library_config(_write_cfg(tmp_path))
    _fake_doc(cfg.staging_dir, "a.b-c", body="new")
    lib.promote(cfg, "a.b-c")
    assert (cfg.text_dir / "a.b-c" / "a.b-c.md").read_text() == "new"


def test_970_non_string_page_status_does_not_abort_the_refresh(tmp_path):
    cfg = lib.load_library_config(_write_cfg(tmp_path))
    _pdf(cfg.pdf_dir / "a.pdf")
    d = _fake_doc(cfg.text_dir, "a", pages=("success",))
    (d / "pages" / "00001.json").write_text(json.dumps({"status": ["warning"]}))
    (d / "pages" / "00002.json").write_text(json.dumps({"status": {"x": "error"}}))
    summary = lib.refresh_index(cfg)
    assert summary["documents"] == 1
    assert lib.doc_status(cfg, d)["bad_pages"] == []
    (d / "pages" / "00003.json").write_text(json.dumps({"status": "warning"}))
    assert lib.doc_status(cfg, d)["bad_pages"] == ["00003"]


def test_971_dry_run_promote_with_nothing_staged_fails(tmp_path, hermetic):
    cfg_path = _write_cfg(tmp_path)
    cfg = lib.load_library_config(cfg_path)
    _fake_doc(cfg.text_dir, "a", body="old")
    res = _run("--config", str(cfg_path), "--dry-run", "--promote", "a")
    assert res.exit_code != 0 and "nothing staged" in res.output
    assert "would promote" not in res.output


def test_971_dry_run_rerun_refuses_missing_pdf_and_staging_leftovers(tmp_path, hermetic):
    cfg_path = _write_cfg(tmp_path)
    cfg = lib.load_library_config(cfg_path)
    res = _run("--config", str(cfg_path), "--dry-run", "--rerun", "nope")
    assert res.exit_code != 0 and "no PDF" in res.output
    _pdf(cfg.pdf_dir / "a.pdf")
    _fake_doc(cfg.staging_dir, "a")
    res = _run("--config", str(cfg_path), "--dry-run", "--rerun", "a")
    assert res.exit_code != 0 and "already exists" in res.output
    assert "would re-process" not in res.output


def test_971_dry_run_rejects_a_pathy_stem(tmp_path, hermetic):
    cfg_path = _write_cfg(tmp_path)
    res = _run("--config", str(cfg_path), "--dry-run", "--promote", "../x")
    assert res.exit_code != 0 and "invalid document stem" in res.output


def test_972_symlinked_unverified_refuses_before_any_index_write(tmp_path):
    cfg = lib.load_library_config(_write_cfg(tmp_path))
    _pdf(cfg.pdf_dir / "a.pdf")
    _fake_doc(cfg.text_dir, "a")
    lib.refresh_index(cfg)
    curated = tmp_path / "curated.txt"
    curated.write_text("a\n")
    cfg.unverified.unlink()
    cfg.unverified.symlink_to(curated)
    _pdf(cfg.pdf_dir / "c.pdf")  # the next refresh WOULD change documents/missing_text
    before = {p: p.read_bytes() for p in (cfg.documents, cfg.missing_text, cfg.manifest)}
    with pytest.raises(lib.LibraryError, match="symlink"):
        lib.refresh_index(cfg)
    assert {p: p.read_bytes() for p in before} == before
    assert curated.read_text() == "a\n"


def test_972_absent_curated_file_is_still_the_empty_set(tmp_path):
    cfg = lib.load_library_config(_write_cfg(tmp_path))
    _pdf(cfg.pdf_dir / "a.pdf")
    lib.refresh_index(cfg)
    assert cfg.unverified.read_text() == ""


def test_973_roll_forward_fsyncs_the_archive_dir_before_clearing_the_journal(tmp_path, monkeypatch):
    cfg = _staged_promotion(tmp_path)
    _crash_on_nth_rename(monkeypatch, 2)  # old text archived, staged not yet installed
    with pytest.raises(RuntimeError):
        lib.promote(cfg, "a")
    monkeypatch.undo()
    assert cfg.archive_dir.is_dir() and not (cfg.text_dir / "a").exists()
    events = _spy_fsync(monkeypatch, cfg)
    assert "recovered" in lib.recover_promotion(cfg)
    assert (cfg.archive_dir, True) in events  # synced while the journal still existed
