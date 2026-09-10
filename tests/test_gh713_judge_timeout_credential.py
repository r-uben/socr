"""#713: a ladder-accepted candidate whose PAGE judge timed out.

Measured on the third-institution census (``docs/log/2026-09-10_703-text-table-
dominance-gate.md``, BoE ``boe-meetings-2018-scan-p28-30.pdf`` p1): the audit log
records ``table_ladder_accepted`` for the qwen candidate, the cached ``PageOutput``
carries ``audit_passed=False`` with ``judge_reason='judge raised: timed out'``,
``_grid_authored_attempt`` refuses it, the strict pool is empty and the page ships
the fail-closed marker under ``structure_class_ladder_exhausted``. A 3,539-character
verified reading became total loss, under a reason that says every candidate was
refused -- when none was even judged.

Astra's ruling (2026-09-10) is implemented as option (a), a narrowly typed and
flagged exception, with (c) whenever its evidence is missing. Every test here pins
a DIFFERENCE between two runs in the same process that vary exactly one thing, per
this repo's CI rule: the provider ladder and the D3/native machinery behave
differently on a host with no ollama, so an absolute outcome measured locally is
not a fact about the code.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from socr.core.document import DocumentHandle
from socr.core.manifest import (
    CREDENTIAL_BLOCKING_EVENT_KINDS,
    SelectionProvenance,
    _select_page_output_tagged,
    _winning_page_output,
)
from socr.core.page_credential import (
    CREDENTIAL_SCHEMA,
    TableAcceptance,
    TableAcceptanceCredential,
    sha256_text,
)
from socr.core.result import (
    JUDGE_OUTCOME_EXCEPTION,
    JUDGE_OUTCOME_TIMEOUT,
    FailureMode,
    PageOutput,
    PageStatus,
)
from socr.core.state import DocumentState

PROSE_BEFORE = "The Committee reviewed the maturity-sorted swap-line balances."
PROSE_AFTER = "Further detail on the counterparties appears in the annex."
MODEL_TABLE = (
    "| counterparty | 2017 | 2018 |\n"
    "|---|---|---|\n"
    "| Bank A | 12.4 | 13.9 |\n"
    "| Bank B | 8.1 | 9.7 |"
)
SECOND_TABLE = "| facility | drawn |\n|---|---|\n| standing | 4.2 |\n| emergency | 0.0 |"
UNIQUE_MODEL_CELL = "| Bank B | 8.1 | 9.7 |"
NATIVE_TEXT = f"{PROSE_BEFORE}\n\n| counterparty | 2017 |\n|---|---|\n| Bank A | 12.4 |\n"

MODEL_TEXT = f"{PROSE_BEFORE}\n\n{MODEL_TABLE}\n\n{PROSE_AFTER}\n"
MODEL_TEXT_TWO_TABLES = f"{PROSE_BEFORE}\n\n{MODEL_TABLE}\n\n{SECOND_TABLE}\n\n{PROSE_AFTER}\n"


def _pdf(tmp_path: Path, name: str = "gh713.pdf") -> Path:
    fitz = pytest.importorskip("fitz")
    path = tmp_path / name
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "Swap-line balances by counterparty")
    doc.new_page().insert_text((72, 72), "Page two carries ordinary prose only.")
    doc.save(str(path))
    doc.close()
    return path


def _credential(
    state: DocumentState,
    *,
    candidate_text: str,
    tables: list[str],
    page_num: int = 1,
    run_fingerprint: str | None = None,
) -> dict:
    """A credential minted over ``candidate_text``, covering ``tables`` by index.

    The run fingerprint defaults to this test module's own pipeline fingerprint,
    matching what the real mint site records. Selection deliberately does not
    check it (``core.manifest`` cannot see the run's identity); the RESUME gate
    does, which is where a changed rung/prompt/DPI config must invalidate the
    credential rather than restore a page judged under different rules.
    """
    from socr.tables.reconcile import find_table_blocks

    lines = candidate_text.splitlines()
    blocks = find_table_blocks(candidate_text)
    entries = []
    for idx, block in enumerate(blocks):
        block_md = "\n".join(lines[block.start : block.end + 1])
        if block_md not in tables:
            continue
        entries.append(
            TableAcceptance(
                table_id=f"p{page_num}-t{idx}",
                markdown_sha256=sha256_text(block_md),
                witness_sha256=sha256_text(f"witness-bytes-{idx}"),
                witness_scope="located",
                rungs=("glm-5.3-flash:cloud",),
            )
        )
    return TableAcceptanceCredential(
        page_num=page_num,
        document_checksum=state.handle.file_hash or "",
        candidate_sha256=sha256_text(candidate_text),
        attempt_engine="qwen",
        attempt_provider_id="qwen-local",
        attempt_provider_model="qwen3-vl:30b-a3b-instruct",
        attempt_provider_backend="ollama",
        judge_model="qwen3-vl:30b-a3b-instruct",
        judge_outcome=JUDGE_OUTCOME_TIMEOUT,
        run_fingerprint=(
            _pipeline()._run_fingerprint() if run_fingerprint is None else run_fingerprint
        ),
        tables=tuple(entries),
    ).to_dict()


def _state(
    pdf_path: Path,
    *,
    model_text: str = MODEL_TEXT,
    judge_outcome: str = JUDGE_OUTCOME_TIMEOUT,
    credential: dict | None = None,
) -> DocumentState:
    """A born-digital structure-class page whose one grid candidate is unjudged.

    ``best_output`` is the native attempt, exactly as the agentic scorer leaves a
    page whose ladder accepted nothing -- the shape the BoE cache has.
    """
    from socr.tables.reconcile import find_table_blocks, table_grid_identity

    state = DocumentState(handle=DocumentHandle.from_path(pdf_path))
    ps = state.pages[1]
    ps.is_born_digital = True
    ps.has_tables = True
    ps.native_text = NATIVE_TEXT
    regions = find_table_blocks(NATIVE_TEXT)
    ps.native_table_region_count = len(regions)
    ps.native_table_region_identities = [table_grid_identity(b.grid) for b in regions]

    native_attempt = PageOutput(
        page_num=1,
        text=NATIVE_TEXT,
        status=PageStatus.SUCCESS,
        engine="native",
        audit_passed=True,
    )
    model_attempt = PageOutput(
        page_num=1,
        text=model_text,
        status=PageStatus.SUCCESS,
        engine="qwen",
        provider_id="qwen-local",
        provider_model="qwen3-vl:30b-a3b-instruct",
        provider_backend="ollama",
        audit_passed=False,
        judge_reason="judge raised: timed out",
        judge_outcome=judge_outcome,
        table_acceptance_credential=credential,
    )
    ps.attempts.extend([native_attempt, model_attempt])
    ps.best_output = native_attempt
    return state


# ---------------------------------------------------------------------------
# 1. The typed outcome. A substring is not a type.
# ---------------------------------------------------------------------------


class _Prof:
    engine = type("E", (), {"value": "qwen"})()
    id = "qwen-local"
    model = "qwen3-vl:30b-a3b-instruct"
    backend = "ollama"
    cost_per_page_usd = 0.0
    timeout_sec = 5


def test_judge_exception_type_decides_the_outcome_not_the_message() -> None:
    """A timeout, a defect and a completed rejection must be three outcomes.

    The DIFFERENCE pinned: the same reason text ("timed out") reaches all three
    paths, and only the exception's TYPE separates them. A gate keyed on the
    message would see one case where there are three -- and the message is built
    from an arbitrary ``str(exc)``, so any upstream can write it.
    """
    from socr.judge.judge import is_page_judge_timeout

    assert is_page_judge_timeout(httpx.ReadTimeout("timed out")) is True
    assert is_page_judge_timeout(TimeoutError("timed out")) is True
    # Same words, not a timeout: a defect in our own code, and a transport
    # failure that is not a timeout.
    assert is_page_judge_timeout(ValueError("timed out")) is False
    assert is_page_judge_timeout(httpx.ConnectError("timed out")) is False


def test_route_page_records_the_typed_outcome_per_exception(monkeypatch) -> None:
    """One process, three judges, three outcomes on the attempt itself."""
    from socr.pipeline import agentic

    def _run(_prof, page_num):
        return PageOutput(page_num=page_num, text=MODEL_TEXT, status=PageStatus.SUCCESS)

    class _Raising:
        def __init__(self, exc):
            self._exc = exc

        def assess(self, output, prof):
            raise self._exc

    class _Rejecting:
        def assess(self, output, prof):
            return agentic.AcceptDecision(accept=False, reason="judge rejected: timed out rows")

    outcomes = {}
    for label, judge in (
        ("timeout", _Raising(httpx.ReadTimeout("timed out"))),
        ("defect", _Raising(ValueError("timed out"))),
        ("rejection", _Rejecting()),
    ):
        decision = agentic.route_page(1, [_Prof()], _run, judge)
        outcomes[label] = decision.attempts[-1].output.judge_outcome

    assert outcomes["timeout"] == JUDGE_OUTCOME_TIMEOUT
    assert outcomes["defect"] == JUDGE_OUTCOME_EXCEPTION
    # A completed rejection carries NO typed outcome: the judge answered.
    assert outcomes["rejection"] == ""
    assert len({outcomes["timeout"], outcomes["defect"], outcomes["rejection"]}) == 3


# ---------------------------------------------------------------------------
# 2-4. Selection: the credential, and every way it fails to hold.
# ---------------------------------------------------------------------------


def test_matching_credential_ships_the_page_and_a_one_byte_change_does_not(
    tmp_path: Path,
) -> None:
    """The whole ticket, as one paired difference in one process.

    Both runs are identical except for ONE byte in the credential's bound
    candidate digest. With the credential intact the reading ships demoted; with
    one byte different the page fails closed and says the judge timed out.
    """
    pdf = _pdf(tmp_path)

    good = _state(pdf, credential=None)
    good.pages[1].attempts[1].table_acceptance_credential = _credential(
        good, candidate_text=MODEL_TEXT, tables=[MODEL_TABLE]
    )
    kept, kept_tag = _select_page_output_tagged(good, 1)

    bad = _state(pdf, credential=None)
    cred = _credential(bad, candidate_text=MODEL_TEXT, tables=[MODEL_TABLE])
    cred["candidate_sha256"] = cred["candidate_sha256"][:-1] + (
        "0" if cred["candidate_sha256"][-1] != "0" else "1"
    )
    bad.pages[1].attempts[1].table_acceptance_credential = cred
    floored, floored_tag = _select_page_output_tagged(bad, 1)

    # The admitted page ships the model's own cells, demoted, never promoted.
    assert kept_tag is SelectionProvenance.STRUCTURE_CLASS_JUDGE_TIMEOUT_CREDENTIALED
    assert UNIQUE_MODEL_CELL in kept.text
    assert kept.status is PageStatus.WARNING
    assert kept.audit_passed is False
    assert kept.failure_mode is FailureMode.JUDGE_TIMEOUT_LADDER_ACCEPTED
    # ... and it says so in the body, not only in the metadata.
    assert "TIMED OUT" in kept.text
    assert "unverified" in kept.text

    # One byte different: no stand-in, and the reason names the timeout.
    assert floored_tag is SelectionProvenance.STRUCTURE_CLASS_PAGE_JUDGE_TIMEOUT_FLOOR
    assert floored.failure_mode is FailureMode.PAGE_JUDGE_TIMEOUT
    assert UNIQUE_MODEL_CELL not in floored.text
    assert kept.text != floored.text

    # The stored attempt is never mutated by selection (the #252 defect).
    assert good.pages[1].attempts[1].audit_passed is False
    assert good.pages[1].attempts[1].status is PageStatus.SUCCESS


def test_a_completed_rejection_is_never_admitted_even_holding_a_credential(
    tmp_path: Path,
) -> None:
    """DIFFERENCE: the typed outcome alone, credential byte-identical.

    A judge that looked and said no is not a judge that never answered, and the
    credential must not launder one into the other.
    """
    pdf = _pdf(tmp_path)

    timed_out = _state(pdf)
    timed_out.pages[1].attempts[1].table_acceptance_credential = _credential(
        timed_out, candidate_text=MODEL_TEXT, tables=[MODEL_TABLE]
    )
    kept, kept_tag = _select_page_output_tagged(timed_out, 1)

    rejected = _state(pdf, judge_outcome="")
    rejected.pages[1].attempts[1].table_acceptance_credential = _credential(
        rejected, candidate_text=MODEL_TEXT, tables=[MODEL_TABLE]
    )
    floored, floored_tag = _select_page_output_tagged(rejected, 1)

    assert kept_tag is SelectionProvenance.STRUCTURE_CLASS_JUDGE_TIMEOUT_CREDENTIALED
    assert floored_tag is not SelectionProvenance.STRUCTURE_CLASS_JUDGE_TIMEOUT_CREDENTIALED
    assert UNIQUE_MODEL_CELL not in floored.text
    # A completed rejection is NOT a timeout, so the floor keeps its old reason.
    assert floored.failure_mode is FailureMode.STRUCTURE_CLASS_LADDER_EXHAUSTED


def test_an_uncovered_second_table_voids_the_whole_credential(tmp_path: Path) -> None:
    """DIFFERENCE: whether the credential covers BOTH emitted tables.

    A page-sized acceptance is not a per-table one. A credential that vouches for
    two of three tables is not a partial credential; it is an invalid one.
    """
    pdf = _pdf(tmp_path)

    complete = _state(pdf, model_text=MODEL_TEXT_TWO_TABLES)
    complete.pages[1].attempts[1].table_acceptance_credential = _credential(
        complete, candidate_text=MODEL_TEXT_TWO_TABLES, tables=[MODEL_TABLE, SECOND_TABLE]
    )
    kept, kept_tag = _select_page_output_tagged(complete, 1)

    partial = _state(pdf, model_text=MODEL_TEXT_TWO_TABLES)
    partial.pages[1].attempts[1].table_acceptance_credential = _credential(
        partial, candidate_text=MODEL_TEXT_TWO_TABLES, tables=[MODEL_TABLE]
    )
    floored, floored_tag = _select_page_output_tagged(partial, 1)

    assert kept_tag is SelectionProvenance.STRUCTURE_CLASS_JUDGE_TIMEOUT_CREDENTIALED
    assert floored_tag is SelectionProvenance.STRUCTURE_CLASS_PAGE_JUDGE_TIMEOUT_FLOOR
    assert floored.failure_mode is FailureMode.PAGE_JUDGE_TIMEOUT
    assert UNIQUE_MODEL_CELL not in floored.text


def test_a_witnessless_table_voids_the_credential(tmp_path: Path) -> None:
    """DIFFERENCE: whether the credential names the image the ladder looked at."""
    pdf = _pdf(tmp_path)

    witnessed = _state(pdf)
    witnessed.pages[1].attempts[1].table_acceptance_credential = _credential(
        witnessed, candidate_text=MODEL_TEXT, tables=[MODEL_TABLE]
    )
    kept_tag = _select_page_output_tagged(witnessed, 1)[1]

    blind = _state(pdf)
    cred = _credential(blind, candidate_text=MODEL_TEXT, tables=[MODEL_TABLE])
    cred["tables"][0]["witness_sha256"] = ""
    cred["tables"][0]["witness_scope"] = "none"
    blind.pages[1].attempts[1].table_acceptance_credential = cred
    floored, floored_tag = _select_page_output_tagged(blind, 1)

    assert kept_tag is SelectionProvenance.STRUCTURE_CLASS_JUDGE_TIMEOUT_CREDENTIALED
    assert floored_tag is SelectionProvenance.STRUCTURE_CLASS_PAGE_JUDGE_TIMEOUT_FLOOR
    assert UNIQUE_MODEL_CELL not in floored.text


@pytest.mark.parametrize(
    "kind",
    [
        "table_ladder_rejected",
        "table_ladder_unverified",
        "table_ladder_withheld",
        "table_binding_boundary_unresolved",
        "source_evidence_table_reject",
        "native_table_verifier_hard_fail",
    ],
)
def test_an_outstanding_hard_contradiction_withholds_the_stand_in(
    tmp_path: Path, kind: str
) -> None:
    """DIFFERENCE: one adverse event on the page, nothing else changed."""
    from socr.core.audit_log import AuditEvent

    pdf = _pdf(tmp_path)

    clean = _state(pdf)
    clean.pages[1].attempts[1].table_acceptance_credential = _credential(
        clean, candidate_text=MODEL_TEXT, tables=[MODEL_TABLE]
    )
    kept_tag = _select_page_output_tagged(clean, 1)[1]

    contradicted = _state(pdf)
    contradicted.pages[1].attempts[1].table_acceptance_credential = _credential(
        contradicted, candidate_text=MODEL_TEXT, tables=[MODEL_TABLE]
    )
    contradicted.events.append(AuditEvent(page_num=1, kind=kind, engine="qwen", detail="", data={}))
    floored, floored_tag = _select_page_output_tagged(contradicted, 1)

    assert kept_tag is SelectionProvenance.STRUCTURE_CLASS_JUDGE_TIMEOUT_CREDENTIALED
    assert floored_tag is SelectionProvenance.STRUCTURE_CLASS_PAGE_JUDGE_TIMEOUT_FLOOR
    assert UNIQUE_MODEL_CELL not in floored.text


def test_an_adverse_ladder_disposition_withholds_the_stand_in(tmp_path: Path) -> None:
    """DIFFERENCE: the page-level ladder reduction, credential unchanged."""
    pdf = _pdf(tmp_path)

    clean = _state(pdf)
    clean.pages[1].attempts[1].table_acceptance_credential = _credential(
        clean, candidate_text=MODEL_TEXT, tables=[MODEL_TABLE]
    )
    kept_tag = _select_page_output_tagged(clean, 1)[1]

    withheld = _state(pdf)
    withheld.pages[1].attempts[1].table_acceptance_credential = _credential(
        withheld, candidate_text=MODEL_TEXT, tables=[MODEL_TABLE]
    )
    withheld.pages[1].table_ladder_disposition = FailureMode.TABLE_WITHHELD
    floored_tag = _select_page_output_tagged(withheld, 1)[1]

    assert kept_tag is SelectionProvenance.STRUCTURE_CLASS_JUDGE_TIMEOUT_CREDENTIALED
    assert floored_tag is not SelectionProvenance.STRUCTURE_CLASS_JUDGE_TIMEOUT_CREDENTIALED


def test_the_blocking_kinds_stay_a_subset_of_the_trust_index(tmp_path: Path) -> None:
    """Drift guard: every kind that withholds the stand-in is a REAL distrust kind.

    ``CREDENTIAL_BLOCKING_EVENT_KINDS`` is a deliberate narrowing of
    ``tables_trust.TABLE_DISTRUST_KINDS``; a kind that appears only here would be
    an invented signal, and one that drifts out of the trust index would silently
    stop blocking.
    """
    from socr.core.tables_trust import TABLE_DISTRUST_KINDS

    missing = CREDENTIAL_BLOCKING_EVENT_KINDS - TABLE_DISTRUST_KINDS
    assert not missing, f"invented blocking kinds not in the trust index: {sorted(missing)}"


# ---------------------------------------------------------------------------
# 7. The existing BoE cache. A log event is not authority.
# ---------------------------------------------------------------------------


def test_the_cached_boe_candidate_cannot_be_promoted_from_the_log_event_alone(
    tmp_path: Path,
) -> None:
    """#713's own cache: ``table_ladder_accepted`` in the log, no credential.

    Replaying the cached candidate must yield the marker, never the accepted
    body. The cache carries the reason STRING ("judge raised: timed out") and the
    audit event, and neither is admissible: the event names a table id and a rung
    trail, which bind no prose and no bytes, and the string is model/transport
    text. The DIFFERENCE pinned is against the same page with a real credential.
    """
    from socr.core.audit_log import AuditEvent

    pdf = _pdf(tmp_path)

    cached = _state(pdf, judge_outcome="", credential=None)
    cached.events.append(
        AuditEvent(
            page_num=1,
            kind="table_ladder_accepted",
            engine="qwen",
            detail="table p1-t0 accepted by the judge ladder",
            data={"table_id": "p1-t0", "witness_scope": "page"},
        )
    )
    replayed = _winning_page_output(cached, 1)

    credentialed = _state(pdf)
    credentialed.pages[1].attempts[1].table_acceptance_credential = _credential(
        credentialed, candidate_text=MODEL_TEXT, tables=[MODEL_TABLE]
    )
    with_proof = _winning_page_output(credentialed, 1)

    assert UNIQUE_MODEL_CELL not in replayed.text, (
        "the cached BoE candidate was promoted from its audit event alone"
    )
    assert replayed.audit_passed is False
    assert UNIQUE_MODEL_CELL in with_proof.text


# ---------------------------------------------------------------------------
# 8. Document status, metadata and CLI.
# ---------------------------------------------------------------------------


def _pipeline():
    from socr.core.config import EngineType, PipelineConfig
    from socr.pipeline.orchestrator import UnifiedPipeline

    return UnifiedPipeline(
        PipelineConfig(
            primary_engine=EngineType.QWEN,
            agentic=True,
            enabled_engines=[EngineType.QWEN],
            quiet=True,
            save_figures=False,
            write_manifest=False,
            table_judge_ladder=False,
        )
    )


def test_assemble_surfaces_both_timeout_reasons_at_document_level(tmp_path: Path) -> None:
    """DIFFERENCE: credential present vs absent, through the real assemble phase.

    Each side must emit its OWN event kind, and the credentialed page must keep
    the structure-class surfacing it already had rather than trade one for the
    other.
    """
    from socr.core.result import DocumentStatus

    kept_state = _state(_pdf(tmp_path, "kept.pdf"))
    kept_state.pages[1].attempts[1].table_acceptance_credential = _credential(
        kept_state, candidate_text=MODEL_TEXT, tables=[MODEL_TABLE]
    )
    kept_result = _pipeline()._phase_assemble(kept_state, tmp_path / "kept_out")
    kept_kinds = {getattr(e, "kind", "") for e in kept_state.events}

    floor_state = _state(_pdf(tmp_path, "floor.pdf"))
    floor_result = _pipeline()._phase_assemble(floor_state, tmp_path / "floor_out")
    floor_kinds = {getattr(e, "kind", "") for e in floor_state.events}

    assert "judge_timeout_ladder_accepted" in kept_kinds
    assert "page_judge_timeout_floor" not in kept_kinds
    # The credentialed page is still a structure-class model page: #713 ADDS a
    # reason, it does not remove a surface a consumer already reads.
    assert "structure_class_model_table_kept" in kept_kinds

    assert "page_judge_timeout_floor" in floor_kinds
    assert "judge_timeout_ladder_accepted" not in floor_kinds
    assert "structure_class_ladder_exhausted_floor" in floor_kinds

    # Neither page may leave the document looking clean.
    assert kept_result.status is not DocumentStatus.SUCCESS
    assert floor_result.status is not DocumentStatus.SUCCESS


def test_the_sidecar_carries_the_credential_the_outcome_and_the_reason(
    tmp_path: Path,
) -> None:
    """Persistence: credential, typed outcome, finalized body and reason."""
    state = _state(_pdf(tmp_path))
    state.pages[1].attempts[1].table_acceptance_credential = _credential(
        state, candidate_text=MODEL_TEXT, tables=[MODEL_TABLE]
    )
    out_dir = tmp_path / "out"
    _pipeline()._phase_assemble(state, out_dir)

    sidecar = next(out_dir.rglob("pages/00001.json"))
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    winning = meta["winning_output"]

    assert meta["failure_mode"] == FailureMode.JUDGE_TIMEOUT_LADDER_ACCEPTED.value
    assert meta["status"] == PageStatus.WARNING.value
    assert meta["audit_passed"] is False
    assert winning["judge_outcome"] == JUDGE_OUTCOME_TIMEOUT
    cred = winning["table_acceptance_credential"]
    assert cred["schema"] == CREDENTIAL_SCHEMA
    # The finalized digest is stamped at finalization and describes the SHIPPED
    # body, which the credential's candidate digest deliberately predates.
    assert cred["finalized_sha256"] == sha256_text(winning["text"])
    assert cred["candidate_sha256"] != cred["finalized_sha256"]
    assert [t["table_id"] for t in cred["tables"]] == ["p1-t0"]


# ---------------------------------------------------------------------------
# 9. Resume.
# ---------------------------------------------------------------------------


def test_resume_reuses_the_page_byte_identically_and_a_mutation_revalidates(
    tmp_path: Path,
) -> None:
    """DIFFERENCE: one edited byte in the fragment on disk.

    Same process, same sidecar, same fingerprint. Untouched, the page is restored
    with its warning intact; with one character changed the ledger refuses and the
    page is reprocessed. Authority comes from the credential over the bytes, never
    from the sidecar's say-so.
    """
    # ONE pdf, reused: ``_pdf`` rewrites the file and a fresh PDF is not
    # byte-identical (its own metadata differs), which would fail the ledger's
    # input-checksum gate for a reason that has nothing to do with #713.
    pdf = _pdf(tmp_path)
    state = _state(pdf)
    state.pages[1].attempts[1].table_acceptance_credential = _credential(
        state, candidate_text=MODEL_TEXT, tables=[MODEL_TABLE]
    )
    pipeline = _pipeline()
    out_dir = tmp_path / "out"
    pipeline._phase_assemble(state, out_dir)

    frag = next(out_dir.rglob("pages/00001.md"))
    original = frag.read_text(encoding="utf-8")

    fresh = _state(pdf)
    restored = pipeline._load_terminal_page(fresh, 1, out_dir)

    frag.write_text(original.replace("8.1", "8.2"), encoding="utf-8")
    mutated = pipeline._load_terminal_page(fresh, 1, out_dir)

    assert restored is not None, "a credentialed timeout page must survive resume"
    assert restored.text == original
    # The warning is PRESERVED, never upgraded by the act of resuming.
    assert restored.status is PageStatus.WARNING
    assert restored.audit_passed is False
    assert restored.failure_mode is FailureMode.JUDGE_TIMEOUT_LADDER_ACCEPTED
    assert mutated is None, "an edited body must revalidate, not restore"


def test_resume_refuses_a_page_whose_credential_is_missing(tmp_path: Path) -> None:
    """DIFFERENCE: the credential, with everything else on disk identical."""
    pdf = _pdf(tmp_path)
    state = _state(pdf)
    state.pages[1].attempts[1].table_acceptance_credential = _credential(
        state, candidate_text=MODEL_TEXT, tables=[MODEL_TABLE]
    )
    pipeline = _pipeline()
    out_dir = tmp_path / "out"
    pipeline._phase_assemble(state, out_dir)

    sidecar = next(out_dir.rglob("pages/00001.json"))
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    fresh = _state(pdf)
    with_credential = pipeline._load_terminal_page(fresh, 1, out_dir)

    meta["winning_output"].pop("table_acceptance_credential", None)
    sidecar.write_text(json.dumps(meta), encoding="utf-8")
    without_credential = pipeline._load_terminal_page(fresh, 1, out_dir)

    assert with_credential is not None
    assert without_credential is None


# ---------------------------------------------------------------------------
# 10. The mint site itself.
# ---------------------------------------------------------------------------


class _FakeWitness:
    """Enough of ``tables.witness.TableWitness`` for the gate's own loop."""

    def __init__(self, table_id: str, markdown: str, crop_path: Path):
        from socr.tables.witness import WitnessScope, WitnessStatus

        self.table_id = table_id
        self.markdown = markdown
        self.crop_path = crop_path
        self.scope = WitnessScope.LOCATED
        self.status = WitnessStatus.LOCATED
        self.box = None
        self.page_num = 1
        self.block_index = 0
        self.note = ""
        self.boxes_found_on_page = 1


def _drive_gate(monkeypatch, tmp_path: Path, *, outcome, judge_outcome: str):
    """Run ``_run_table_judge_gate`` over one accepted/unaccepted table."""
    from socr.judge import table_ladder as tl
    from socr.judge.table_verdict import RungResult
    from socr.tables import witness as witness_mod

    tmp_path.mkdir(parents=True, exist_ok=True)
    pdf = _pdf(tmp_path)
    state = _state(pdf, judge_outcome=judge_outcome)
    ps = state.pages[1]
    bo = ps.attempts[1]
    ps.best_output = bo

    crop = tmp_path / "crop.png"
    crop.write_bytes(b"a witness image")

    import contextlib

    @contextlib.contextmanager
    def _fake_witnesses(path, page_num, markdown, **kw):
        yield [_FakeWitness("p1-t0", MODEL_TABLE, crop)]

    def _fake_ladder(rungs, crop_path, markdown, table_id):
        return tl.TableLadderResult(
            table_id=table_id,
            outcome=outcome,
            rung_results=[RungResult(rung="glm", ok=True, verdict=None)],
        )

    monkeypatch.setattr(witness_mod, "prepare_table_witnesses", _fake_witnesses)
    monkeypatch.setattr(tl, "run_table_ladder", _fake_ladder)

    pipeline = _pipeline()
    monkeypatch.setattr(pipeline, "_binding_evidence_for_witness", lambda *a, **k: (None, None))
    monkeypatch.setattr(pipeline, "_record_unresolved_binding_boundary", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "_record_candidate_row_normalization", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "_live_table_judge_rungs", lambda rungs: list(rungs))
    monkeypatch.setattr(pipeline, "_record_table_rung_refusals", lambda *a, **k: None)
    monkeypatch.setattr(
        pipeline, "_resolve_table_guard_chain", lambda *a, **k: a[3] if len(a) > 3 else None
    )
    pipeline._run_table_judge_gate(state, 1, ps, bo, [lambda **kw: None])
    return bo


def test_the_gate_mints_a_credential_only_on_a_full_acceptance(monkeypatch, tmp_path: Path) -> None:
    """DIFFERENCE: the ladder's own terminal, everything else held fixed.

    ACCEPTED mints a credential bound to the candidate; UNVERIFIED mints none.
    Without this the credential's producer is never executed by any test and the
    whole feature could be wired to a mint that never fires.
    """
    from socr.judge.table_ladder import TableLadderOutcome

    accepted = _drive_gate(
        monkeypatch,
        tmp_path / "acc",
        outcome=TableLadderOutcome.ACCEPTED,
        judge_outcome=JUDGE_OUTCOME_TIMEOUT,
    )
    unverified = _drive_gate(
        monkeypatch,
        tmp_path / "unv",
        outcome=TableLadderOutcome.UNVERIFIED,
        judge_outcome=JUDGE_OUTCOME_TIMEOUT,
    )
    # Same acceptance, but the judge produced a verdict rather than timing out.
    judged = _drive_gate(
        monkeypatch,
        tmp_path / "jud",
        outcome=TableLadderOutcome.ACCEPTED,
        judge_outcome="",
    )

    assert accepted.table_acceptance_credential is not None
    cred = TableAcceptanceCredential.from_dict(accepted.table_acceptance_credential)
    assert cred is not None
    assert cred.candidate_sha256 == sha256_text(accepted.text)
    assert cred.judge_outcome == JUDGE_OUTCOME_TIMEOUT
    assert [t.table_id for t in cred.tables] == ["p1-t0"]
    # The witness image is named by its own bytes, not merely asserted to exist.
    from socr.core.page_credential import sha256_file

    assert cred.tables[0].witness_sha256 == sha256_file(tmp_path / "acc" / "crop.png")
    assert cred.tables[0].witness_scope == "located"
    assert cred.tables[0].rungs

    assert unverified.table_acceptance_credential is None
    assert judged.table_acceptance_credential is None
