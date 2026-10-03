"""GH-984: the conftest guard fails a test that reaches the configured Ollama host.

A leak must fail loudly, not hang. Other loopback ports must stay usable, because
many tests run their own loopback servers.
"""

from __future__ import annotations

import socket

import pytest

from conftest import ollama_connection_guard


def _listener() -> socket.socket:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    return srv


def test_other_loopback_port_is_allowed():
    srv = _listener()
    try:
        with ollama_connection_guard() as violations:
            cli = socket.create_connection(srv.getsockname(), timeout=2)
            cli.close()
        assert violations == []
    finally:
        srv.close()


def test_default_ollama_port_is_refused_and_recorded():
    with ollama_connection_guard() as violations:
        with pytest.raises(ConnectionRefusedError, match="GH-984"):
            socket.create_connection(("127.0.0.1", 11434), timeout=2)
    assert violations and violations[0].startswith("127.0.0.1:11434")


def test_connect_ex_is_refused_and_recorded():
    with ollama_connection_guard() as violations:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            assert s.connect_ex(("127.0.0.1", 11434)) != 0
        finally:
            s.close()
    assert violations


def test_configured_ollama_host_is_refused(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:23456")
    with ollama_connection_guard() as violations:
        with pytest.raises(ConnectionRefusedError):
            socket.create_connection(("127.0.0.1", 23456), timeout=2)
    assert violations


def test_guard_restores_socket_connect():
    before = socket.socket.connect
    with ollama_connection_guard():
        assert socket.socket.connect is not before
    assert socket.socket.connect is before


def test_swallowed_refusal_still_leaves_a_record():
    """A probe's own except clause hides the refusal; the record is what fails the test."""
    with ollama_connection_guard() as violations:
        try:
            socket.create_connection(("127.0.0.1", 11434), timeout=2)
        except OSError:
            pass
    assert violations
