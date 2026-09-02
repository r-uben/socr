"""Cold review round 7 on the P3+P5 branch (`fix/p3-p5-judged-bytes-ship`).

Round 6 routed every journal site through ``DocumentState.record_engine_run``
and defended the contract with an AST guard. The round-6 review defeated that
guard four ways: an alias then append, a list-rebuild assignment, a ``getattr``
hop, and a subclass method that simply took the exempt name.

Ruling: a static guard cannot win that game. Enforce it at RUNTIME by
encapsulation. The journal lives in a private list; the public ``engine_runs``
is a read-only view, so every bypass shape raises where it is written rather
than being noticed by a pattern-matcher. The AST guard survives only in the
small role it can actually do: nothing outside ``state.py`` may name the private
list, and inside ``state.py`` only the recorder and the read-only view may.

The probe matrix below is the reviewer's, and each probe must now either raise
at runtime or be caught by the scoped guard.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from socr.core.document import DocumentHandle
from socr.core.result import DocumentStatus, EngineResult
from socr.core.state import DocumentState

fitz = pytest.importorskip("fitz")

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "socr"
_STATE_MODULE = _SRC / "core" / "state.py"

#: The private journal, and the only two members allowed to name it.
_PRIVATE_JOURNAL = "_engine_runs"
_OWNER_CLASS = "DocumentState"
_OWNER_MEMBERS = frozenset({"record_engine_run", "engine_runs"})


def _state(tmp_path):
    pdf = tmp_path / "doc.pdf"
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "source")
    doc.save(pdf)
    doc.close()
    return DocumentState(DocumentHandle(pdf))


def _run(state) -> EngineResult:
    return EngineResult(
        document_path=state.handle.path,
        engine="gemini",
        status=DocumentStatus.SUCCESS,
        cost=0.0002,
    )


# ---------------------------------------------------------------------------
# The scoped guard: only state.py may name the private journal, and only in the
# recorder and the read-only view.
# ---------------------------------------------------------------------------


def private_journal_references(path: pathlib.Path) -> list[str]:
    """Every reference to the private journal in *path*, outside its owners.

    Scoped by ENCLOSING CLASS AND MEMBER, not by a bare function name: round 6's
    guard exempted anything called ``record_engine_run``, so a subclass override
    with that name was waved straight through.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    parents: dict = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    def _owner(node) -> tuple[str | None, str | None]:
        member = None
        while node in parents:
            node = parents[node]
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and member is None:
                member = node.name
            if isinstance(node, ast.ClassDef):
                return node.name, member
        return None, member

    found: list[str] = []
    for node in ast.walk(tree):
        named = (isinstance(node, ast.Attribute) and node.attr == _PRIVATE_JOURNAL) or (
            isinstance(node, ast.Name) and node.id == _PRIVATE_JOURNAL
        )
        if not named:
            continue
        cls, member = _owner(node)
        if path == _STATE_MODULE and cls == _OWNER_CLASS and member in _OWNER_MEMBERS | {None}:
            continue  # the field declaration, the recorder, the view
        found.append(f"{path.name}:{node.lineno} in {cls}.{member}")
    return found


class TestTheJournalIsEncapsulated:
    def test_the_public_name_is_a_read_only_view(self, tmp_path) -> None:
        state = _state(tmp_path)
        state.record_engine_run(_run(state), page_nums=[1])
        assert isinstance(state.engine_runs, tuple)
        assert len(state.engine_runs) == 1
        assert state.total_cost == 0.0002

    # -- the reviewer's five probes ------------------------------------------

    def test_probe_alias_then_append_raises(self, tmp_path) -> None:
        state = _state(tmp_path)
        alias = state.engine_runs
        with pytest.raises(AttributeError):
            alias.append(_run(state))
        assert state.engine_runs == ()

    def test_probe_list_rebuild_assignment_raises(self, tmp_path) -> None:
        state = _state(tmp_path)
        with pytest.raises(AttributeError):
            state.engine_runs = list(state.engine_runs) + [_run(state)]

    def test_probe_augmented_assignment_raises(self, tmp_path) -> None:
        state = _state(tmp_path)
        with pytest.raises((AttributeError, TypeError)):
            state.engine_runs += [_run(state)]

    def test_probe_getattr_hop_raises(self, tmp_path) -> None:
        state = _state(tmp_path)
        with pytest.raises(AttributeError):
            getattr(state, "engine_runs").append(_run(state))

    def test_probe_constructor_cannot_install_a_journal(self, tmp_path) -> None:
        """Round 7: the private list is not a constructor input."""
        state = _state(tmp_path)
        with pytest.raises(TypeError):
            type(state)(handle=state.handle, _engine_runs=[_run(state)])

    def test_probe_dataclasses_replace_cannot_install_a_journal(self, tmp_path) -> None:
        """Round 7: nor a ``dataclasses.replace`` input -- a fresh list installed
        that way would carry runs no page was charged for."""
        import dataclasses

        state = _state(tmp_path)
        with pytest.raises((TypeError, ValueError)):
            dataclasses.replace(state, _engine_runs=[_run(state)])

    def test_probe_subclass_helper_is_caught_by_the_guard(self, tmp_path) -> None:
        sample = tmp_path / "subclass_helper.py"
        sample.write_text(
            "class Sneaky(DocumentState):\n"
            "    def stash(self, result):\n"
            "        self._engine_runs.append(result)\n",
            encoding="utf-8",
        )
        assert private_journal_references(sample) == ["subclass_helper.py:3 in Sneaky.stash"]

    def test_probe_subclass_override_taking_the_exempt_name_is_caught(self, tmp_path) -> None:
        """Round 6's guard trusted the bare function name, so this shape walked
        through. The scope check is by class AND member, and the class here is
        not the owner."""
        sample = tmp_path / "subclass_override.py"
        sample.write_text(
            "class Sneaky(DocumentState):\n"
            "    def record_engine_run(self, result, page_nums=None):\n"
            "        self._engine_runs.append(result)\n",
            encoding="utf-8",
        )
        assert private_journal_references(sample) == [
            "subclass_override.py:3 in Sneaky.record_engine_run"
        ]

    # -- the production tree obeys it ----------------------------------------

    def test_no_module_outside_state_names_the_private_journal(self) -> None:
        offenders: list[str] = []
        for path in sorted(_SRC.rglob("*.py")):
            offenders.extend(private_journal_references(path))
        assert not offenders, (
            "the journal is private to DocumentState; go through "
            f"record_engine_run instead: {offenders}"
        )

    def test_the_recorder_is_where_the_private_append_lives(self) -> None:
        """The owners are exempt from the guard, so prove they are really the
        ones doing the work rather than the exemption hiding an empty contract."""
        source = _STATE_MODULE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        recorder = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "record_engine_run"
        )
        appends = [
            node
            for node in ast.walk(recorder)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "append"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == _PRIVATE_JOURNAL
        ]
        assert len(appends) == 1

    def test_recording_still_charges_the_page(self, tmp_path) -> None:
        """The encapsulation must not have quietly dropped the round-6 contract."""
        state = _state(tmp_path)
        state.record_engine_run(_run(state), page_nums=[1])
        assert state.pages[1].page_cost_usd == 0.0002
