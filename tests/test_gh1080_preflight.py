"""#1080: a preflight at the start of every run stops it when a needed tool is missing.

Hermetic: CI has no model provider, no judge and no Tesseract. Every probe is
faked at the name ``socr.preflight`` (or ``socr.cli``) imports it under, so
nothing here reaches a network or a binary.

The CLI cases pin a DIFFERENCE, not a value measured on one machine: the same
invocation in the same process with only the missing tool opted out (or only the
flag under test changed), asserting what differs.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner
from test_gh359_ladder_terminals import _make_config, _pass, _process, _QueueRung, _ruled_pdf

from socr import preflight
from socr.cli import cli
from socr.core.config import EngineType, PipelineConfig
from socr.pipeline.orchestrator import UnifiedPipeline
from socr.tables import source_evidence as se

pytestmark = pytest.mark.real_preflight

_REAL_CHECK_ENGINE_CLI = preflight.check_engine_cli

fitz = pytest.importorskip("fitz")


class _Pipe:
    """The two pipeline methods the checks call, with scripted answers."""

    def __init__(self, judge: str | None = "judge-model", maths: bool = True, config=None) -> None:
        self._judge = judge
        self._maths = maths
        self._stub = types.SimpleNamespace(config=config)
        self._judge_unavailable_reason = "judge: 410 Gone"

    def _resolve_judge_model(self):
        return self._judge

    def _equation_lane_backend_addressable(self, profile) -> bool:
        return self._maths

    # #1063: scripted OpenAI-compatible state; None = the Ollama transport.
    served: str | None = None

    def _maths_openai_served_model(self, backend=None):
        return self.served

    def _maths_model_refusal(self, lane_model: str) -> str:
        return ""

    # The run-level lane predicates are the REAL pipeline's, applied to this config.
    def _equation_region_lane_enabled(self) -> bool:
        return UnifiedPipeline._equation_region_lane_enabled(self._stub)

    def _corrupt_math_recovery_enabled(self) -> bool:
        return UnifiedPipeline._corrupt_math_recovery_enabled(self._stub)

    def _clean_equation_recovery_enabled(self) -> bool:
        return UnifiedPipeline._clean_equation_recovery_enabled(self._stub)


def _cfg(**kw) -> PipelineConfig:
    return PipelineConfig(quiet=True, **kw)


# --- the Tesseract state probe ---------------------------------------------------------


def _fake_pytesseract(
    version=None, raises: Exception | None = None, not_found: bool = False
) -> types.ModuleType:
    mod = types.ModuleType("pytesseract")

    class TesseractNotFoundError(Exception):
        pass

    mod.TesseractNotFoundError = TesseractNotFoundError

    def get_tesseract_version():
        if not_found:
            raise TesseractNotFoundError()
        if raises is not None:
            raise raises
        return version

    mod.get_tesseract_version = get_tesseract_version
    return mod


def test_witness_state_package_missing(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "pytesseract", None)  # import raises ImportError
    state, detail = se.classical_ocr_backend_state()
    assert state == se.WITNESS_PACKAGE_MISSING
    assert detail == se.WITNESS_STATE_MESSAGES[se.WITNESS_PACKAGE_MISSING]


def test_witness_state_binary_missing(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "pytesseract", _fake_pytesseract(not_found=True))
    assert se.classical_ocr_backend_state()[0] == se.WITNESS_BINARY_MISSING


def test_witness_state_exec_error_is_not_an_install_problem(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "pytesseract", _fake_pytesseract(raises=RuntimeError("boom")))
    state, detail = se.classical_ocr_backend_state()
    assert state == se.WITNESS_EXEC_ERROR
    assert "boom" in detail


def test_witness_state_present(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "pytesseract", _fake_pytesseract(version="5.0"))
    assert se.classical_ocr_backend_state() == (se.WITNESS_READING, "")


# --- per-check units -------------------------------------------------------------------


@pytest.mark.parametrize(
    "state", [se.WITNESS_PACKAGE_MISSING, se.WITNESS_BINARY_MISSING, se.WITNESS_EXEC_ERROR]
)
def test_tesseract_absent_is_a_needed_miss_named_by_its_state(state) -> None:
    with patch.object(preflight, "classical_ocr_backend_state", return_value=(state, "d")):
        check = preflight.check_tesseract(_cfg())
    assert check.missing and check.name == preflight.CHECK_TESSERACT
    assert check.detail.startswith(state)


def test_tesseract_present_is_ok() -> None:
    with patch.object(
        preflight, "classical_ocr_backend_state", return_value=(se.WITNESS_READING, "")
    ):
        assert not preflight.check_tesseract(_cfg()).missing


def test_native_only_still_needs_the_model_tools() -> None:
    """Preflight cannot know an input has no scans, and genuine scans still route to
    OCR under ``--native-only`` (``_agentic_no_provider_page``): the model tools are
    needed there too, so a missing one is a miss rather than a silent pass."""
    config = _cfg(native_only=True)
    with (
        patch.object(
            preflight, "classical_ocr_backend_state", return_value=(se.WITNESS_PACKAGE_MISSING, "")
        ),
        patch.object(preflight, "probe_model_generation", return_value=(False, "down")) as probe,
    ):
        server = preflight.check_model_server(config)
        tess = preflight.check_tesseract(config)
        judge = preflight.check_page_judge(config, _Pipe(judge=None))
    assert server.missing and tess.missing and judge.missing
    probe.assert_called_once()


def test_model_server_probes_a_generation_on_the_resolved_ollama_host() -> None:
    config = _cfg(qwen_backend="ollama")
    with patch.object(preflight, "probe_model_generation", return_value=(True, "")) as probe:
        ok = preflight.check_model_server(config)
    host, model, _timeout = probe.call_args.args
    assert ok.ok and ok.needed and model in ok.detail and host in ok.detail
    with patch.object(preflight, "probe_model_generation", return_value=(False, "wedged")):
        bad = preflight.check_model_server(config)
    assert bad.missing and "wedged" in bad.detail


def test_model_server_not_needed_when_qwen_is_not_an_enabled_engine() -> None:
    config = _cfg(enabled_engines=[EngineType.GEMINI])
    with patch.object(preflight, "probe_model_generation") as probe:
        assert not preflight.check_model_server(config).needed
    probe.assert_not_called()


def test_model_server_on_a_vllm_backend_asks_the_vllm_server_not_ollama() -> None:
    config = _cfg(qwen_backend="vllm", qwen_vllm_url="http://gpu:8000/v1", qwen_vllm_model="m/x")
    with (
        patch.object(preflight, "probe_openai_server_idle", return_value=False) as vllm,
        patch.object(preflight, "probe_model_generation") as probe,
    ):
        check = preflight.check_model_server(config)
    probe.assert_not_called()
    # A generation canary on the served model, not a /models listing.
    assert vllm.call_args.args == ("http://gpu:8000/v1",)
    assert vllm.call_args.kwargs["model"] == "m/x"
    assert check.missing and "http://gpu:8000/v1" in check.detail


def test_page_judge_missing_carries_the_judge_reason() -> None:
    check = preflight.check_page_judge(_cfg(), _Pipe(judge=None))
    assert check.missing and "410 Gone" in check.detail
    assert not preflight.check_page_judge(_cfg(), _Pipe(judge="m")).missing


def test_heuristic_judge_is_not_a_needed_tool() -> None:
    check = preflight.check_page_judge(_cfg(judge_backend="heuristic"), _Pipe(judge=None))
    assert not check.needed


def test_maths_readers_unreachable_backend_is_a_needed_miss() -> None:
    with patch.object(preflight, "probe_model_generation", return_value=(True, "")) as probe:
        check = preflight.check_maths_readers(_cfg(), _Pipe(maths=False, config=_cfg()))
        assert check.missing and "1063" in check.detail
        probe.assert_not_called()  # nothing can be posted to that transport
        assert not preflight.check_maths_readers(_cfg(), _Pipe(maths=True, config=_cfg())).missing


def test_maths_readers_present_means_their_models_answer() -> None:
    """The transport matching is not enough: each model a lane will call is probed."""
    config = _cfg(
        math_model="math-m",
        clean_equation_model="clean-m",
        equation_region_lane=True,
        recover_corrupt_math=True,
    )

    def answer(host, model, timeout):
        return (model != "math-m", "wedged" if model == "math-m" else "")

    with patch.object(preflight, "probe_model_generation", side_effect=answer) as probe:
        check = preflight.check_maths_readers(config, _Pipe(maths=True, config=config))
    assert {c.args[1] for c in probe.call_args_list} == {"math-m", "clean-m"}
    assert check.missing and "math-m" in check.detail and "wedged" in check.detail


def test_cloud_models_forbidden_by_policy_are_never_probed() -> None:
    config = _cfg(
        strict_local=True, math_model="m:cloud", qwen_model="q:cloud", qwen_model_pinned=True
    )
    with patch.object(preflight, "probe_model_generation", return_value=(True, "")) as probe:
        maths = preflight.check_maths_readers(config, _Pipe(maths=True, config=config))
        server = preflight.check_model_server(config)
    assert not [c for c in probe.call_args_list if "cloud" in c.args[1]]
    assert maths.missing and "forbidden by --strict-local" in maths.detail
    assert server.missing and "forbidden by --strict-local" in server.detail


def test_page_judge_never_probes_a_cloud_model_under_strict_local(monkeypatch) -> None:
    probed: list[str] = []

    def record(model):
        probed.append(model)
        return False, f"{model}: down"

    monkeypatch.setattr(UnifiedPipeline, "_probe_judge_candidate", staticmethod(record))
    config = _cfg(strict_local=True, judge_model="judge:cloud")
    preflight.check_page_judge(config, UnifiedPipeline(config))
    assert probed and not any("cloud" in m for m in probed)


def test_engine_cli_missing_is_a_miss_even_when_the_server_is_up() -> None:
    engine = types.SimpleNamespace(cli_command="qwen-ocr", name="qwen")
    config = _cfg(enabled_engines=[EngineType.QWEN])
    with patch.object(preflight, "get_engine", return_value=engine):
        with patch("socr.engines.base.BaseEngine.is_available", return_value=False):
            missing = preflight.check_engine_cli(config)
        with patch("socr.engines.base.BaseEngine.is_available", return_value=True):
            present = preflight.check_engine_cli(config)
    assert missing.missing and "qwen-ocr" in missing.detail
    assert present.ok and present.needed


def test_engine_cli_not_needed_when_no_cli_engine_is_in_the_run() -> None:
    config = _cfg(enabled_engines=[EngineType.GEMINI], primary_engine=EngineType.AUTO)
    assert not preflight.check_engine_cli(config).needed


def test_maths_readers_not_needed_when_every_maths_lane_is_off() -> None:
    config = _cfg(equation_region_lane=False, recover_corrupt_math=False)
    assert not preflight.check_maths_readers(config, _Pipe(maths=False, config=config)).needed


def test_strict_local_makes_the_table_judge_not_needed_and_keeps_the_notice() -> None:
    check = preflight.check_table_judge(_cfg(strict_local=True, table_judge_ladder=True))
    assert not check.needed and "TABLE_UNVERIFIED" in check.detail


def test_allowing_a_miss_unblocks_it_and_names_it() -> None:
    checks = [
        preflight.Check("a", True, False, "x"),
        preflight.Check("b", True, True),
        preflight.Check("c", False, False),
    ]
    assert [c.name for c in preflight.missing_checks(checks)] == ["a"]
    assert preflight.missing_checks(checks, ["a"]) == []
    assert preflight.ran_without(checks, ["a", "b", "c"]) == ["a"]
    assert preflight.ran_without(checks) == []


# --- the CLI gate ----------------------------------------------------------------------


@pytest.fixture
def others_present(monkeypatch):
    """Everything present except Tesseract, whose probe the test scripts."""
    for name, check in (
        ("check_model_server", preflight.CHECK_MODEL_SERVER),
        ("check_engine_cli", preflight.CHECK_ENGINE_CLI),
        ("check_page_judge", preflight.CHECK_PAGE_JUDGE),
        ("check_maths_readers", preflight.CHECK_MATHS_READERS),
        ("check_table_judge", preflight.CHECK_TABLE_JUDGE),
    ):
        monkeypatch.setattr(
            preflight, name, lambda *a, _c=check, **k: preflight.Check(_c, True, True, "present")
        )


class _RecordingPipeline:
    instances: list = []

    def __init__(self, config) -> None:
        self.config = config
        self.processed: list = []
        self.batched: list = []
        _RecordingPipeline.instances.append(self)

    def process(self, pdf, output_dir, **kwargs):
        self.processed.append(pdf)
        from socr.core.result import DocumentStatus, EngineResult

        return EngineResult(document_path=pdf, engine="x", status=DocumentStatus.SUCCESS)

    def process_batch(self, input_dir, output_dir=None, **kwargs):
        self.batched.append(input_dir)

    def _resolve_output_root(self, path, out):
        return out or path

    @property
    def last_outcome(self):
        return types.SimpleNamespace(exit_code=0)


@pytest.fixture
def pdf(tmp_path: Path) -> Path:
    path = tmp_path / "in" / "doc.pdf"
    path.parent.mkdir()
    doc = fitz.open()
    doc.new_page().insert_text((72, 100), "One page of prose.", fontsize=11)
    doc.save(path)
    doc.close()
    return path


def _invoke(args: list[str]):
    _RecordingPipeline.instances = []
    with patch("socr.pipeline.orchestrator.UnifiedPipeline", _RecordingPipeline):
        result = CliRunner().invoke(cli, args)
    return result, (_RecordingPipeline.instances[-1] if _RecordingPipeline.instances else None)


def _no_tesseract():
    return patch.object(
        preflight,
        "classical_ocr_backend_state",
        return_value=(
            se.WITNESS_BINARY_MISSING,
            se.WITNESS_STATE_MESSAGES[se.WITNESS_BINARY_MISSING],
        ),
    )


def test_process_stops_before_any_page_unless_the_miss_is_allowed(
    pdf: Path, tmp_path: Path, others_present
) -> None:
    base = ["process", str(pdf), "-o", str(tmp_path / "out"), "--primary", "qwen"]
    with _no_tesseract():
        stopped, stopped_pipe = _invoke(base)
        allowed, allowed_pipe = _invoke([*base, "--allow-missing", "tesseract"])

    # The DIFFERENCE between the two runs is exactly the opt-out.
    assert stopped.exit_code != 0 and allowed.exit_code == 0, (stopped.output, allowed.output)
    assert stopped_pipe.processed == [] and len(allowed_pipe.processed) == 1
    assert "binary_missing" in stopped.output and "--allow-missing" in stopped.output
    assert stopped_pipe.ran_without == []
    assert allowed_pipe.ran_without == [preflight.CHECK_TESSERACT]
    assert "MISSING (allowed)" in allowed.output


def test_batch_stops_before_any_file_unless_the_miss_is_allowed(
    pdf: Path, tmp_path: Path, others_present
) -> None:
    base = ["batch", str(pdf.parent), "-o", str(tmp_path / "out"), "--primary", "qwen"]
    with _no_tesseract():
        stopped, stopped_pipe = _invoke(base)
        allowed, allowed_pipe = _invoke([*base, "--allow-missing", "tesseract"])
    assert stopped.exit_code != 0 and allowed.exit_code == 0, (stopped.output, allowed.output)
    assert stopped_pipe.batched == [] and len(allowed_pipe.batched) == 1
    assert allowed_pipe.ran_without == [preflight.CHECK_TESSERACT]


def test_an_unknown_check_name_is_refused(pdf: Path, tmp_path: Path) -> None:
    result, _ = _invoke(["process", str(pdf), "--allow-missing", "tesseractt"])
    assert result.exit_code == 2 and "tesseractt" in result.output


def test_dry_run_never_probes(pdf: Path, tmp_path: Path) -> None:
    def boom(*a, **k):
        raise AssertionError("a dry run probed the machine")

    with patch.object(preflight, "run_preflight", boom):
        proc, _ = _invoke(["process", str(pdf), "--dry-run"])
        bat, _ = _invoke(["batch", str(pdf.parent), "--dry-run", "--primary", "qwen"])
    assert proc.exit_code == 0 and bat.exit_code == 0, (proc.output, bat.output)


def test_preflight_command_exit_code_follows_the_missing_set(others_present) -> None:
    runner = CliRunner()
    with patch("socr.pipeline.orchestrator.UnifiedPipeline", _RecordingPipeline):
        with _no_tesseract():
            missing = runner.invoke(cli, ["preflight", "--primary", "qwen"])
            allowed = runner.invoke(
                cli, ["preflight", "--primary", "qwen", "--allow-missing", "tesseract"]
            )
        with patch.object(
            preflight, "classical_ocr_backend_state", return_value=(se.WITNESS_READING, "")
        ):
            present = runner.invoke(cli, ["preflight", "--primary", "qwen"])
    assert (missing.exit_code, allowed.exit_code, present.exit_code) == (1, 0, 0)
    assert "tesseract" in missing.output and "MISSING" in missing.output
    assert "MISSING" not in present.output


# --- the metadata record ---------------------------------------------------------------


def _run_doc(tmp_path: Path, ran_without: list[str]) -> Path:
    pdf_path = _ruled_pdf(tmp_path, "doc.pdf")
    pipeline = UnifiedPipeline(_make_config(quiet=True))
    pipeline.ran_without = ran_without
    pipeline._surface_table_scoring = lambda *args, **kwargs: None
    out = tmp_path / "out"
    _process(pipeline, pdf_path, out, [_QueueRung([_pass("high")])])
    return out / "doc"


def test_the_run_records_what_it_went_without(tmp_path: Path) -> None:
    """Difference pin: same document, only ``ran_without`` changes."""
    plain = _run_doc(tmp_path / "plain", [])
    without = _run_doc(tmp_path / "without", ["tesseract"])

    plain_doc = json.loads((plain / "metadata.json").read_text(encoding="utf-8"))
    without_doc = json.loads((without / "metadata.json").read_text(encoding="utf-8"))
    assert "ran_without" not in plain_doc
    assert without_doc["ran_without"] == ["tesseract"]

    plain_root = json.loads((plain.parent / "metadata.json").read_text(encoding="utf-8"))
    without_root = json.loads((without.parent / "metadata.json").read_text(encoding="utf-8"))
    assert "ran_without" not in json.dumps(plain_root)
    assert "ran_without" in json.dumps(without_root)
    # Nothing else about the document moved.
    assert {k: v for k, v in without_doc.items() if k != "ran_without"}.keys() == plain_doc.keys()


# --- every CLI path that processes pages is gated -----------------------------------------


def test_hpc_lane_is_gated_and_records_what_it_went_without(
    pdf: Path, tmp_path: Path, others_present, monkeypatch
) -> None:
    monkeypatch.setattr(preflight, "check_engine_cli", _REAL_CHECK_ENGINE_CLI)
    nougat = types.SimpleNamespace(cli_command="nougat", name="nougat", is_available=lambda: False)
    built: list = []

    class _Hpc:
        def __init__(self, config) -> None:
            self.processed: list = []
            built.append(self)

        def process(self, pdf_path, output_dir=None):
            from socr.core.result import DocumentStatus, EngineResult

            self.processed.append(pdf_path)
            return EngineResult(document_path=pdf_path, engine="hpc", status=DocumentStatus.SUCCESS)

    base = ["process", str(pdf), "-o", str(tmp_path / "out"), "--hpc-sequential"]
    runner = CliRunner()

    def tesseract_probed(*a, **k):
        raise AssertionError("the HPC lane probed Tesseract, a tool it does not use")

    with (
        patch("socr.pipeline.hpc_pipeline.HPCPipeline", _Hpc),
        patch.object(preflight, "get_engine", return_value=nougat),
        patch.object(preflight, "classical_ocr_backend_state", side_effect=tesseract_probed),
    ):
        stopped = runner.invoke(cli, base)
        allowed = runner.invoke(cli, [*base, "--allow-missing", "engine-cli"])
    assert stopped.exit_code != 0 and allowed.exit_code == 0, (stopped.output, allowed.output)
    assert built[0].processed == [] and len(built[1].processed) == 1
    assert built[1].ran_without == [preflight.CHECK_ENGINE_CLI]
    # The lane uses no judge, Tesseract or maths reader, so none of them is needed.
    assert "tesseract      not needed" in stopped.output


def test_library_is_gated_and_records_what_it_went_without(tmp_path: Path, monkeypatch) -> None:
    from test_gh964_library import _pdf as lib_pdf
    from test_gh964_library import _write_cfg

    from socr import library as lib

    cfg_path = _write_cfg(tmp_path)
    cfg = lib.load_library_config(cfg_path)
    lib_pdf(cfg.pdf_dir / "new.pdf")
    _hermetic_library(monkeypatch)
    runner = CliRunner()
    base = ["library", "--primary", "qwen", "--config", str(cfg_path)]
    with _no_tesseract():
        stopped = runner.invoke(cli, base)
        assert not (cfg.text_dir / "new").exists(), "a page was processed before the stop"
        allowed = runner.invoke(cli, [*base, "--allow-missing", "tesseract"])
    assert stopped.exit_code != 0 and allowed.exit_code == 0, (stopped.output, allowed.output)
    meta = json.loads((cfg.text_dir / "new" / "metadata.json").read_text(encoding="utf-8"))
    assert meta["ran_without"] == ["tesseract"]


def _hermetic_library(monkeypatch) -> None:
    """Pipeline patches of test_gh964_library's `hermetic`, plus the others-present gate."""
    from socr.core import providers
    from socr.core.result import PageOutput, PageStatus

    for name, check in (
        ("check_model_server", preflight.CHECK_MODEL_SERVER),
        ("check_engine_cli", preflight.CHECK_ENGINE_CLI),
        ("check_page_judge", preflight.CHECK_PAGE_JUDGE),
        ("check_maths_readers", preflight.CHECK_MATHS_READERS),
        ("check_table_judge", preflight.CHECK_TABLE_JUDGE),
    ):
        monkeypatch.setattr(
            preflight, name, lambda *a, _c=check, **k: preflight.Check(_c, True, True, "present")
        )
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


# --- ran_without reaches the manifest and the end-of-run summary --------------------------


def _run_doc_full(tmp_path: Path, ran_without: list[str], *, quiet: bool) -> Path:
    pipeline = UnifiedPipeline(_make_config(quiet=quiet, write_manifest=True))
    pipeline.ran_without = ran_without
    pipeline._surface_table_scoring = lambda *args, **kwargs: None
    out = tmp_path / "out"
    _process(pipeline, _ruled_pdf(tmp_path, "doc.pdf"), out, [_QueueRung([_pass("high")])])
    return out / "doc"


def test_ran_without_reaches_the_manifest(tmp_path: Path) -> None:
    plain = json.loads(
        (_run_doc_full(tmp_path / "p", [], quiet=True) / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    without = json.loads(
        (_run_doc_full(tmp_path / "w", ["tesseract"], quiet=True) / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert "ran_without" not in plain
    assert without["ran_without"] == ["tesseract"]


def test_ran_without_is_named_in_the_end_of_run_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    capsys.readouterr()
    _run_doc_full(tmp_path / "p", [], quiet=False)
    plain = capsys.readouterr().out
    _run_doc_full(tmp_path / "w", ["tesseract"], quiet=False)
    without = capsys.readouterr().out
    assert "Ran without" not in plain
    assert "Ran without: tesseract" in without


def test_a_quiet_run_with_an_allowed_miss_still_prints_the_report(
    pdf: Path, tmp_path: Path, others_present
) -> None:
    base = ["process", str(pdf), "-o", str(tmp_path / "out"), "--primary", "qwen", "-q"]
    with _no_tesseract():
        quiet_plain, _ = _invoke(base)
        quiet_allowed, _ = _invoke([*base, "--allow-missing", "tesseract"])
    assert quiet_plain.exit_code != 0 and quiet_allowed.exit_code == 0
    assert "MISSING" in quiet_plain.output  # a stop is never hidden by --quiet
    assert "MISSING (allowed)" in quiet_allowed.output


def test_ran_without_is_named_in_the_batch_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from socr.core.result import DocumentStatus, EngineResult

    src = tmp_path / "in"
    _ruled_pdf(src, "doc.pdf")

    def batch_output(ran_without: list[str]) -> str:
        pipeline = UnifiedPipeline(_make_config(quiet=False))
        pipeline.ran_without = ran_without
        capsys.readouterr()
        with patch.object(
            pipeline,
            "process",
            side_effect=lambda pdf, *a, **k: EngineResult(
                document_path=pdf, engine="x", status=DocumentStatus.SUCCESS
            ),
        ):
            pipeline.process_batch(src, tmp_path / f"out{len(ran_without)}")
        return capsys.readouterr().out

    plain, without = batch_output([]), batch_output(["tesseract"])
    assert "Completed:" in plain and "Ran without" not in plain
    assert "Ran without: tesseract" in without


def _hpc_cfg(**kw) -> PipelineConfig:
    config = _cfg(**kw)
    config.hpc.enabled = True
    config.hpc.manage_server = False
    return config


def test_hpc_server_with_a_forbidden_cloud_model_is_not_probed() -> None:
    config = _hpc_cfg(strict_local=True)
    config.hpc.ocr_model = "m:cloud"
    with patch.object(preflight, "probe_openai_server_idle", return_value=True) as probe:
        check = preflight.check_model_server(config)
        config.strict_local = False
        local = preflight.check_model_server(config)
    assert check.missing and "forbidden by --strict-local" in check.detail
    assert local.ok  # control: the same model is probed when policy allows it
    assert probe.call_count == 1


def test_native_only_does_not_require_the_models_of_lanes_it_switches_off() -> None:
    """Needed uses the pipeline's own lane predicate: native-only turns both off."""
    native = _cfg(native_only=True)
    normal = _cfg()
    with patch.object(preflight, "probe_model_generation", return_value=(True, "")) as probe:
        off = preflight.check_maths_readers(native, _Pipe(maths=True, config=native))
        on = preflight.check_maths_readers(normal, _Pipe(maths=True, config=normal))
    assert not off.needed and on.needed
    assert probe.call_count == 1  # only the normal run probed (both lanes share one model)


def test_zero_cap_policy_exempts_the_table_judge_like_strict_local() -> None:
    config = _cfg(max_cost_per_page=0.0, max_cost_per_page_pinned=True, table_judge_ladder=True)
    check = preflight.check_table_judge(config)
    assert not check.needed and "--max-cost-per-page 0" in check.detail


def test_tesseract_is_not_probed_when_the_hpc_lane_does_not_need_it() -> None:
    with patch.object(preflight, "classical_ocr_backend_state", side_effect=AssertionError):
        check = preflight.check_tesseract(_hpc_cfg())
    assert not check.needed


def test_preflight_does_not_change_the_resume_fingerprint(monkeypatch) -> None:
    """Same run, gate not run vs gate run with an allowed miss: identical fingerprint,
    or a preflight would invalidate every resumable page."""
    from socr.cli import _preflight_gate

    def fingerprint(gate: bool) -> str:
        config = _make_config(quiet=True)
        pipeline = UnifiedPipeline(config)
        if gate:
            missing = [preflight.Check(preflight.CHECK_TESSERACT, True, False, "binary_missing")]
            with patch.object(preflight, "run_preflight", return_value=missing):
                _preflight_gate(config, pipeline, (preflight.CHECK_TESSERACT,))
            assert pipeline.ran_without == [preflight.CHECK_TESSERACT]
        return pipeline._run_fingerprint()

    assert fingerprint(False) == fingerprint(True)
