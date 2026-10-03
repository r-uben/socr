"""GH-976: ``resolve_ollama_host`` must keep URL userinfo and not mangle it.

``http://user:pass@host:9`` used to come back as ``http://[user:pass@host:9]``:
the credential colon pushed the host token over the "more than one colon means a
bare IPv6 literal" rule, and the whole token got bracketed. Pure string
function, so absolute outputs are safe to pin (no provider involved).
"""

from __future__ import annotations

import logging

import pytest

from socr.core.ollama_utils import safe_host_label
from socr.tables.extract import resolve_ollama_host

#: (input, expected). Every one of these was wrong before the fix.
BROKEN_BEFORE = [
    ("u:p@h:9", "http://u:p@h:9"),
    ("http://u:p@h:9", "http://u:p@h:9"),
    ("http://u:p@h:9/", "http://u:p@h:9/"),
    ("http://u:p@[::1]", "http://u:p@[::1]:11434"),
    ("http://u:p@[::1]:9", "http://u:p@[::1]:9"),
    ("u:p@[::1]", "http://u:p@[::1]:11434"),
    ("u:p@::1", "http://u:p@[::1]:11434"),
    ("http://u:p@::1", "http://u:p@[::1]:11434"),
    ("https://u:p@::1/x", "https://u:p@[::1]:11434/x"),
    ("http://u:p%40w@h:9", "http://u:p%40w@h:9"),
]

#: Userinfo forms that already worked; must stay as they were.
USERINFO_ALREADY_OK = [
    ("u:p@h", "http://u:p@h:11434"),
    ("http://u:p@h", "http://u:p@h:11434"),
    ("http://u:p@h/", "http://u:p@h:11434/"),
    ("http://u@h", "http://u@h:11434"),
    ("http://u:p@h/x", "http://u:p@h:11434/x"),
    ("u:p@h/", "http://u:p@h:11434/"),
]

#: No userinfo: byte-identical to the pre-fix output.
CONTROLS = [
    ("h", "http://h:11434"),
    ("h:9", "http://h:9"),
    ("http://h", "http://h:11434"),
    ("http://h/", "http://h:11434/"),
    ("http://h:9", "http://h:9"),
    ("http://h:9/", "http://h:9/"),
    ("https://h:9/x/", "https://h:9/x/"),
    ("[::1]", "http://[::1]:11434"),
    ("[::1]:9", "http://[::1]:9"),
    ("::1", "http://[::1]:11434"),
    ("::1:11434", "http://[::1:11434]"),
    ("http://[::1]:9/", "http://[::1]:9/"),
    ("  gpu-node  ", "http://gpu-node:11434"),
]


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)


@pytest.mark.parametrize(("raw", "expected"), BROKEN_BEFORE + USERINFO_ALREADY_OK + CONTROLS)
def test_resolution_is_pinned(raw, expected) -> None:
    assert resolve_ollama_host(raw) == expected


@pytest.mark.parametrize(("raw", "expected"), BROKEN_BEFORE)
def test_userinfo_is_never_bracketed(raw, expected) -> None:
    out = resolve_ollama_host(raw)
    assert "[u:" not in out
    assert "@" in out


@pytest.mark.parametrize(("raw", "expected"), BROKEN_BEFORE + USERINFO_ALREADY_OK)
def test_userinfo_form_resolves_like_its_plain_twin(raw, expected) -> None:
    """Difference pin: adding ``u:p@`` changes nothing but that prefix."""
    plain = raw.replace("u:p%40w@", "").replace("u:p@", "").replace("u@", "")
    out = resolve_ollama_host(raw)
    twin = resolve_ollama_host(plain)
    userinfo = raw.split("@")[0].rpartition("//")[2] + "@"
    assert out.replace(userinfo, "", 1) == twin


@pytest.mark.parametrize("raw", ["u:p@h:9", "http://u:p@h:9", "http://u:p@::1"])
def test_env_var_path_matches_explicit(raw, monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_HOST", raw)
    assert resolve_ollama_host() == resolve_ollama_host(raw)


def test_unparseable_host_warning_does_not_leak_credentials(caplog) -> None:
    raw = "http://secretuser:s3cr3t@h:notaport"
    with caplog.at_level(logging.WARNING):
        out = resolve_ollama_host(raw)
    assert out == raw  # still returned verbatim for the request
    assert caplog.records, "the unparseable-host warning should still fire"
    assert "s3cr3t" not in caplog.text
    assert "secretuser" not in caplog.text


def test_label_of_a_resolved_userinfo_host_is_credential_free() -> None:
    label = safe_host_label(resolve_ollama_host("http://u:hunter2@h:9"))
    assert label == "http://h:9"
