"""GH-976: ``resolve_ollama_host`` must keep URL userinfo and not mangle it.

``http://user:pass@host:9`` used to come back as ``http://[user:pass@host:9]``:
the credential colon pushed the host token over the "more than one colon means a
bare IPv6 literal" rule, and the whole token got bracketed. Pure string
function, so absolute outputs are safe to pin (no provider involved).
"""

from __future__ import annotations

import logging

import httpx
import pytest

from socr.core.ollama_utils import (
    raise_for_status_redacted,
    redact_credentials,
    safe_host_label,
)
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


#: Raw (unencoded) ``@`` in the password and IPv6 zone ids.
RAW_AT_AND_ZONE = [
    ("u:p@w@h:9", "http://u:p@w@h:9"),
    ("http://u:p@w@h", "http://u:p@w@h:11434"),
    ("u:p@w@[::1]:9", "http://u:p@w@[::1]:9"),
    ("fe80::1%eth0", "http://[fe80::1%eth0]:11434"),
    ("[fe80::1%25eth0]:9", "http://[fe80::1%25eth0]:9"),
    ("http://u:p@[fe80::1%25eth0]:9", "http://u:p@[fe80::1%25eth0]:9"),
    ("u:p@fe80::1%eth0", "http://u:p@[fe80::1%eth0]:11434"),
]


@pytest.mark.parametrize(("raw", "expected"), RAW_AT_AND_ZONE)
def test_raw_at_in_password_and_ipv6_zone_id(raw, expected) -> None:
    assert resolve_ollama_host(raw) == expected


def test_redact_credentials_handles_raw_at_and_several_urls() -> None:
    text = "for url 'http://u:p@w@h:9/api/tags' and https://a:b@c/x"
    assert redact_credentials(text) == "for url 'http://h:9/api/tags' and https://c/x"
    assert redact_credentials("no url here: a@b") == "no url here: a@b"


def _status_404(url: str) -> httpx.Response:
    return httpx.Response(404, request=httpx.Request("GET", url))


@pytest.mark.parametrize("password", ["hunter2", "h@nter2"])
def test_failed_request_to_a_userinfo_host_leaves_no_credentials_in_logs(
    password, monkeypatch, caplog
) -> None:
    from socr.judge import table_rung_ollama as rung

    host = f"http://sekretuser:{password}@gpu:9"
    monkeypatch.setattr(rung.httpx, "get", lambda url, **kw: _status_404(url))
    with caplog.at_level(logging.DEBUG):
        assert rung.ollama_rung_reachable("m", host) is False
    assert "unreachable" in caplog.text  # the failure really was logged
    assert password not in caplog.text
    assert "sekretuser" not in caplog.text


def test_status_error_text_carries_no_credentials() -> None:
    with pytest.raises(httpx.HTTPStatusError) as ei:
        raise_for_status_redacted(_status_404("http://sekretuser:hunter2@gpu:9/api/chat"))
    assert "hunter2" not in str(ei.value)
    assert "sekretuser" not in str(ei.value)
    assert "http://gpu:9/api/chat" in str(ei.value)
    assert ei.value.response.status_code == 404  # still a usable HTTPStatusError
    assert ei.value.__cause__ is None and ei.value.__suppress_context__


def test_exception_that_embeds_the_url_is_redacted_in_the_log(monkeypatch, caplog) -> None:
    """The log site redacts on its own, independent of ``raise_for_status``."""
    from socr.judge import table_rung_ollama as rung

    def _boom(url, **kw):
        raise httpx.ConnectError(f"cannot reach {url}")

    monkeypatch.setattr(rung.httpx, "get", _boom)
    with caplog.at_level(logging.DEBUG):
        assert rung.ollama_rung_reachable("m", "http://sekretuser:hunter2@gpu:9") is False
    assert "cannot reach http://gpu:9" in caplog.text
    assert "hunter2" not in caplog.text
    assert "sekretuser" not in caplog.text


# ---------------------------------------------------------------------------
# Round 3: structural redaction, and credentials kept out of the request URL.
# ---------------------------------------------------------------------------

PASSWORD_CHARS = ["'", '"', "%", "!", "$", ";", " ", "@", ":", "&", "(", "*", "~", "#"]


@pytest.mark.parametrize("ch", PASSWORD_CHARS)
def test_redaction_is_structural_not_a_character_class(ch) -> None:
    text = f"Client error for url 'http://us{ch}er:p{ch}ss{ch}@gpu:9/api/chat'"
    assert redact_credentials(text) == "Client error for url 'http://gpu:9/api/chat'"


def test_redaction_covers_every_url_in_a_message() -> None:
    text = "a http://u:p'1@h1:1/x then https://v:q\"2@h2/y end"
    assert redact_credentials(text) == "a http://h1:1/x then https://h2/y end"


def test_redaction_stops_at_the_path_boundary() -> None:
    text = "for url 'http://u:p@h:9/api/tags' (contact ops@example.com)"
    assert redact_credentials(text) == "for url 'http://h:9/api/tags' (contact ops@example.com)"


def test_redaction_without_a_path_errs_towards_removing_not_leaking() -> None:
    assert "secret" not in redact_credentials("failed: http://u:secret@gpu:9")
    assert redact_credentials("plain text a@b, no url") == "plain text a@b, no url"
    assert redact_credentials("http://gpu:9/x") == "http://gpu:9/x"


def test_split_userinfo_and_endpoint() -> None:
    import base64

    from socr.core.ollama_utils import ollama_endpoint, split_userinfo, urllib_auth_headers

    assert split_userinfo("http://h:9") == ("http://h:9", None)
    assert split_userinfo("http://u:p%41@h:9/x") == ("http://h:9/x", ("u", "pA"))
    assert split_userinfo("http://u:p@w@h:9") == ("http://h:9", ("u", "p@w"))
    url, extra = ollama_endpoint("http://u:p@h:9/", "/api/tags")
    assert url == "http://h:9/api/tags" and isinstance(extra["auth"], httpx.BasicAuth)
    assert ollama_endpoint("http://h:9", "/api/tags") == ("http://h:9/api/tags", {})
    assert urllib_auth_headers("http://u:p@h") == {
        "Authorization": "Basic " + base64.b64encode(b"u:p").decode()
    }
    assert urllib_auth_headers("http://h") == {}


@pytest.fixture
def loopback():
    """A real HTTP server on 127.0.0.1 recording each request's Authorization."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    seen: list[tuple[str, str, str | None]] = []
    status = {"code": 200}

    class H(BaseHTTPRequestHandler):
        def _reply(self):
            seen.append((self.command, self.path, self.headers.get("Authorization")))
            n = int(self.headers.get("Content-Length") or 0)
            if n:
                self.rfile.read(n)
            body = json.dumps({"models": [{"name": "m:latest"}], "response": "x"}).encode()
            self.send_response(status["code"])
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        do_GET = do_POST = _reply

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1], seen, status
    srv.shutdown()
    srv.server_close()


USER = "sekretuser"
PW = "p'a\"ss!$;w@rd"  # keeps ' " ! $ ; and a raw @


def _basic() -> str:
    import base64

    return "Basic " + base64.b64encode(f"{USER}:{PW}".encode()).decode()


def _assert_clean(caplog) -> None:
    for secret in (USER, "p'a", 'a"ss', "ss!$;w", "w@rd", _basic().split()[1]):
        assert secret not in caplog.text, secret


def test_real_request_keeps_credentials_out_of_url_and_logs(loopback, caplog, monkeypatch) -> None:
    from socr.core.ollama_utils import _get_tags
    from socr.judge import table_rung_ollama as rung
    from socr.judge.ollama_judge import _post_generate

    # conftest stubs httpx.get/_post_chat for hermeticity; this test needs the real thing.
    monkeypatch.setattr(httpx, "get", httpx._api.get)
    port, seen, status = loopback
    host = f"http://{USER}:{PW}@127.0.0.1:{port}"
    with caplog.at_level(logging.DEBUG):
        assert rung.ollama_rung_reachable("m", host) is True
        assert _get_tags(host, 5.0).status_code == 200
        assert _post_generate(host, "m", "p", "aW1n", 5.0) == "x"
        status["code"] = 404
        assert rung.ollama_rung_reachable("m", host) is False
    assert len(seen) == 4
    assert all(auth == _basic() for _, _, auth in seen)  # the server got the credentials
    assert all(path.startswith("/api/") for _, path, _ in seen)  # none rode in the path
    assert any("HTTP Request" in r.getMessage() for r in caplog.records), "httpx logged"
    _assert_clean(caplog)


def test_urllib_equation_paths_send_basic_auth_not_url_userinfo(loopback, caplog, tmp_path) -> None:
    from socr.math import equation_latex, recover

    port, seen, _ = loopback
    host = f"http://{USER}:{PW}@127.0.0.1:{port}"
    with caplog.at_level(logging.DEBUG):
        assert recover.latex_for_image(b"png", host=host, timeout=5.0) == "x"
        crop = tmp_path / "c.png"
        crop.write_bytes(b"png")
        equation_latex.latex_for_crop(crop, host=host, timeout=5.0)
    assert seen and all(auth == _basic() for _, _, auth in seen)
    _assert_clean(caplog)
