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


# --- coverage: allowed loopback use, and both in-process HTTP clients are caught ---


@pytest.fixture
def loopback_server():
    import http.server
    import threading

    class _H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), _H)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv
    srv.shutdown()
    srv.server_close()


def test_fixture_loopback_server_works_under_the_autouse_guard(loopback_server):
    """No explicit guard here: the suite-wide autouse guard is what is active."""
    import urllib.request

    port = loopback_server.server_address[1]
    assert urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5).read() == b"ok"


def test_test_may_point_ollama_host_at_its_own_server(loopback_server, monkeypatch):
    import urllib.request

    port = loopback_server.server_address[1]
    monkeypatch.setenv("OLLAMA_HOST", f"127.0.0.1:{port}")
    # The suite-wide autouse guard captured the ambient host before this env
    # change, so it must allow this server (it would fail the test at teardown).
    assert urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5).read() == b"ok"


def test_httpx_connection_to_ambient_host_is_caught():
    import httpx

    with ollama_connection_guard() as violations:
        with pytest.raises(httpx.ConnectError):
            # Client, not httpx.get: conftest stubs the module-level httpx.get.
            with httpx.Client() as client:
                client.get("http://127.0.0.1:11434/api/tags", timeout=2)
    assert violations


def test_urllib_connection_to_ambient_host_is_caught():
    import urllib.error
    import urllib.request

    with ollama_connection_guard() as violations:
        with pytest.raises(urllib.error.URLError):
            urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=2)
    assert violations
