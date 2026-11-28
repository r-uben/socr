"""Hermetic unit tests for OCRiaEngine and table-crop two-reader verification.

Covers:
  * ``OCRiaEngine`` Ollama HTTP ``/api/chat`` (honouring ``OLLAMA_HOST``,
    ``think=False``, ``temperature=0``, chart-forbidding system prompt) and
    OpenAI-compatible vLLM mode (``OCRIA_BASE_URL``).
  * ``filter_chart_overlapping_boxes`` dropping mis-cropped charts while keeping
    genuine tables.
  * ``normalize_html_table_for_comparison`` stripping page furniture and
    deduplicating repeated colspan year headers ONLY in leading pure header rows
    without ever weakening data rows, minus signs, decimals, or cell counts.
  * ``verify_table_crop_two_readers`` requiring bidirectional agreement and
    failing closed on single-digit swaps, sign flips, truncation, unclosed HTML,
    or reader errors.

100% synthetic and hermetic — no live Ollama, no network, no external paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest

import socr
from socr.engines.ocria import (
    DEFAULT_OCRIA_MODEL,
    OCRIA_API_KEY_ENV,
    OCRIA_BASE_URL_ENV,
    OCRIA_MODEL_ENV,
    OCRIA_TABLE_NO_CHART_PROMPT,
    OCRiaConfig,
    OCRiaEngine,
    _encode_image_bytes,
)
from socr.tables.crop_ocr import (
    CropAgreementResult,
    filter_chart_overlapping_boxes,
    normalize_html_table_for_comparison,
    verify_table_crop_two_readers,
)
from socr.tables.second_reader import row_numbers

# Minimal valid 1x1 PNG bytes for hermetic image tests.
_TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0"
    b"\x00\x00\x03\x01\x01\x00\x18\xdd\x8d\xb0\x00\x00\x00\x00IEND\xaeB`\x82"
)


@pytest.fixture(autouse=True)
def _clear_ocria_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests hermetic even when OCRIA_* vars are set in the caller shell."""
    for env_var in (OCRIA_MODEL_ENV, OCRIA_BASE_URL_ENV, OCRIA_API_KEY_ENV):
        monkeypatch.delenv(env_var, raising=False)


def test_loaded_socr_is_this_checkout() -> None:
    assert Path(socr.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1] / "src")


# ---------------------------------------------------------------------------
# 1. OCRiaEngine — Ollama + vLLM transport, OLLAMA_HOST, think=False
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            req = httpx.Request("POST", "http://test/api")
            resp = httpx.Response(self.status_code, request=req)
            raise httpx.HTTPStatusError("err", request=req, response=resp)

    def json(self) -> dict:
        return self._payload


def test_ocria_ollama_honors_ollama_host_and_sends_think_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(OCRIA_BASE_URL_ENV, raising=False)
    monkeypatch.setenv("OLLAMA_HOST", "gpu-box.internal:11435")

    crop = tmp_path / "crop.png"
    crop.write_bytes(_TINY_PNG)

    captured: dict = {}

    def _fake_post(url: str, *, json: dict, timeout: float, **kwargs) -> _FakeResponse:
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return _FakeResponse(
            {
                "message": {"content": "<table><tr><th>Item</th><th>2000</th></tr></table>"},
                "done_reason": "stop",
            }
        )

    monkeypatch.setattr(httpx, "post", _fake_post)

    engine = OCRiaEngine()
    assert engine.backend == "ollama"
    assert engine.model == DEFAULT_OCRIA_MODEL
    assert engine.resolved_ollama_host() == "http://gpu-box.internal:11435"

    text, finish = engine.read_image(crop)
    assert finish == "stop"
    assert "<table>" in text
    assert captured["url"] == "http://gpu-box.internal:11435/api/chat"
    assert "localhost" not in captured["url"]
    assert captured["json"]["model"] == DEFAULT_OCRIA_MODEL
    assert captured["json"]["stream"] is False
    assert captured["json"]["think"] is False
    assert captured["json"]["options"]["temperature"] == 0.0
    sys_prompt = captured["json"]["messages"][0]["content"]
    assert "NEVER convert a chart" in sys_prompt
    assert "<table>" in sys_prompt
    assert len(captured["json"]["messages"][1]["images"]) == 1


def test_ocria_vllm_mode_uses_ocria_base_url_and_openai_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(OCRIA_BASE_URL_ENV, "http://h200-node:8000/v1/")
    monkeypatch.setenv("OCRIA_API_KEY", "test-cluster-key")

    crop = tmp_path / "crop.png"
    crop.write_bytes(_TINY_PNG)

    captured: dict = {}

    def _fake_post(
        url: str, *, headers: dict, json: dict, timeout: float, **kwargs
    ) -> _FakeResponse:
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return _FakeResponse(
            {
                "choices": [
                    {
                        "message": {"content": "<table><tr><td>GDP</td><td>2.4</td></tr></table>"},
                        "finish_reason": "length",
                    }
                ]
            }
        )

    monkeypatch.setattr(httpx, "post", _fake_post)

    engine = OCRiaEngine()
    assert engine.backend == "vllm"
    assert engine.endpoint_url() == "http://h200-node:8000/v1/chat/completions"

    text, finish = engine.read_image(crop)
    assert finish == "length"
    assert "2.4" in text
    assert captured["url"] == "http://h200-node:8000/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer test-cluster-key"
    assert captured["json"]["temperature"] == 0.0
    assert captured["json"]["messages"][0]["content"] == OCRIA_TABLE_NO_CHART_PROMPT
    user_parts = captured["json"]["messages"][1]["content"]
    assert user_parts[0]["type"] == "image_url"
    assert user_parts[0]["image_url"]["url"].startswith("data:image/png;base64,")


def test_ocria_is_available_checks_ollama_tags_and_vllm_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(OCRIA_BASE_URL_ENV, raising=False)
    monkeypatch.setenv("OLLAMA_HOST", "http://10.0.0.9:11434")
    monkeypatch.setattr("socr.engines.ocria.host_reachable", lambda host: True)
    monkeypatch.setattr(
        httpx,
        "get",
        lambda url, **kw: _FakeResponse({"models": [{"name": DEFAULT_OCRIA_MODEL}]}),
    )
    assert OCRiaEngine().is_available() is True

    monkeypatch.setattr(
        httpx,
        "get",
        lambda url, **kw: _FakeResponse({"models": [{"name": "other-model:latest"}]}),
    )
    assert OCRiaEngine().is_available() is False

    vllm_engine = OCRiaEngine(OCRiaConfig(base_url="http://h200-node:8000/v1"))
    monkeypatch.setattr(
        httpx,
        "get",
        lambda url, **kw: _FakeResponse({"data": [{"id": DEFAULT_OCRIA_MODEL}]}, 200),
    )
    assert vllm_engine.is_available() is True
    monkeypatch.setattr(httpx, "get", lambda url, **kw: _FakeResponse({"data": []}, 200))
    assert vllm_engine.is_available() is False
    monkeypatch.setattr(
        httpx,
        "get",
        lambda url, **kw: _FakeResponse({"data": [{"id": "different-model"}]}, 200),
    )
    assert vllm_engine.is_available() is False
    monkeypatch.setattr(httpx, "get", lambda url, **kw: _FakeResponse({}, 503))
    assert vllm_engine.is_available() is False


def test_ocria_encode_image_bytes_detects_magic_types_and_rejects_unknown() -> None:
    assert _encode_image_bytes(_TINY_PNG)[1] == "image/png"
    assert _encode_image_bytes(b"\xff\xd8\xff\xe0dummy")[1] == "image/jpeg"
    assert _encode_image_bytes(b"RIFF\x00\x00\x00\x00WEBPdummy")[1] == "image/webp"
    assert _encode_image_bytes(b"GIF89adummy")[1] == "image/gif"
    with pytest.raises(ValueError, match="Unsupported raw image format"):
        _encode_image_bytes(b"not-an-image")


def test_ocria_posts_enforce_total_wall_clock_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both Ollama and vLLM POSTs must run through ``call_with_total_deadline``."""
    from socr.engines import ocria as ocria_mod

    crop = tmp_path / "crop.png"
    crop.write_bytes(_TINY_PNG)
    deadlines: list[tuple[float, str]] = []

    def _spy_deadline(fn, timeout: float, *, label: str = ""):
        deadlines.append((timeout, label))
        return fn()

    monkeypatch.setattr(ocria_mod, "call_with_total_deadline", _spy_deadline)
    monkeypatch.setattr(
        httpx,
        "post",
        lambda url, **kw: _FakeResponse(
            {
                "message": {"content": "<table><tr><td>1.0</td></tr></table>"},
                "done_reason": "stop",
                "choices": [
                    {
                        "message": {"content": "<table><tr><td>1.0</td></tr></table>"},
                        "finish_reason": "stop",
                    }
                ],
            }
        ),
    )
    OCRiaEngine(timeout=42.0).read_image(crop)
    OCRiaEngine(base_url="http://h200-node:8000/v1", timeout=55.0).read_image(crop)
    assert len(deadlines) == 2
    assert deadlines[0][0] == 42.0 and "ollama" in deadlines[0][1]
    assert deadlines[1][0] == 55.0 and "vllm" in deadlines[1][1]


# ---------------------------------------------------------------------------
# 2. filter_chart_overlapping_boxes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _BoxWrapper:
    label: str
    bbox: tuple[float, float, float, float]


def test_filter_chart_overlapping_boxes_drops_chart_miscrops_and_keeps_tables() -> None:
    chart_box = (100.0, 500.0, 900.0, 1200.0)
    genuine_table = (100.0, 100.0, 900.0, 450.0)
    slight_border_touch = (100.0, 480.0, 900.0, 505.0)  # tiny overlap with chart top
    chart_as_table = (120.0, 520.0, 880.0, 1180.0)  # full chart mis-cropped as table
    subplot_inside_chart = (150.0, 550.0, 400.0, 750.0)  # low IoU vs huge chart, 100% covered!
    zero_area = (200.0, 200.0, 200.0, 400.0)

    kept = filter_chart_overlapping_boxes(
        [genuine_table, slight_border_touch, chart_as_table, subplot_inside_chart, zero_area],
        [chart_box],
    )
    assert kept == [genuine_table, slight_border_touch]


def test_filter_chart_overlapping_boxes_preserves_wrapper_objects_and_thresholds() -> None:
    t_real = _BoxWrapper("t_real", (0.0, 0.0, 200.0, 100.0))
    t_chart = _BoxWrapper("t_chart", (0.0, 200.0, 200.0, 300.0))
    chart = _BoxWrapper("chart", (0.0, 220.0, 200.0, 320.0))  # 80% cover of t_chart

    kept_default = filter_chart_overlapping_boxes([t_real, t_chart], [chart])
    assert kept_default == [t_real]

    # Raising thresholds above the overlap keeps both (proves thresholds are live).
    kept_relaxed = filter_chart_overlapping_boxes(
        [t_real, t_chart], [chart], max_iou=0.95, max_cover=0.95
    )
    assert kept_relaxed == [t_real, t_chart]


# ---------------------------------------------------------------------------
# 3. normalize_html_table_for_comparison
# ---------------------------------------------------------------------------


def test_normalize_strips_page_furniture_and_deduplicates_colspan_year_headers() -> None:
    raw_spanned = """\
```markdown
<page_header>II-14</page_header>
-5-
Some running title and chart axis ticks 1990 1995 2000

<table>
  <tr>
    <th>Measure</th>
    <th colspan="3">2000</th>
    <th colspan="2">2001</th>
  </tr>
  <tr>
    <th></th>
    <th>Q1</th><th>Q2</th><th>Q3</th>
    <th>Q1</th><th>Q2</th>
  </tr>
  <tr>
    <td>Real GDP</td>
    <td>1.6</td><td>-0.1</td><td>2.4</td>
    <td>3.1</td><td>0.0</td>
  </tr>
</table>
<page_footer>Confidential -5-</page_footer>
```"""
    raw_expanded = """\
<table>
  <tr>
    <th>Measure</th>
    <th>2000</th><th>2000</th><th>2000</th>
    <th>2001</th><th>2001</th>
  </tr>
  <tr>
    <th></th>
    <th>Q1</th><th>Q2</th><th>Q3</th>
    <th>Q1</th><th>Q2</th>
  </tr>
  <tr>
    <td>Real GDP</td>
    <td>1.6</td><td>-0.1</td><td>2.4</td>
    <td>3.1</td><td>0.0</td>
  </tr>
</table>"""

    norm_spanned = normalize_html_table_for_comparison(raw_spanned)
    norm_expanded = normalize_html_table_for_comparison(raw_expanded)

    assert "-5-" not in norm_spanned
    assert "II-14" not in norm_spanned
    assert "chart axis ticks" not in norm_spanned
    assert norm_spanned == norm_expanded
    # Exact column count preserved across every row (6 columns -> 7 pipes).
    for line in norm_expanded.splitlines():
        assert line.count("|") == 7


def test_normalize_deduplicates_year_month_header_and_preserves_data_rows() -> None:
    html_text = """\
<table>
  <tr>
    <th>Series</th>
    <th>1999 July</th><th>1999 Aug.</th><th>1999 Sept.</th>
    <th>2000 Jan.</th><th>2000 Feb.</th>
  </tr>
  <tr>
    <td>Index</td>
    <td>106.0</td><td>104.5</td><td>-0.2</td>
    <td>2000</td><td>2000</td>
  </tr>
  <tr>
    <td>Counts</td>
    <td>2000</td><td>2000</td><td>1999</td>
    <td>1999</td><td>1999</td>
  </tr>
</table>"""
    norm = normalize_html_table_for_comparison(html_text)
    lines = norm.splitlines()
    # Header row deduplicated consecutive year repeats while keeping months & cell counts.
    assert lines[0] == "| Series | 1999 July | Aug. | Sept. | 2000 Jan. | Feb. |"
    # Data rows are NEVER weakened — repeated 2000 / 1999 and -0.2 stay intact!
    assert lines[2] == "| Index | 106.0 | 104.5 | -0.2 | 2000 | 2000 |"
    assert lines[3] == "| Counts | 2000 | 2000 | 1999 | 1999 | 1999 |"


def test_normalize_preserves_non_consecutive_year_groups_in_headers() -> None:
    html_text = """\
<table>
  <tr>
    <th></th><th>1999</th><th>2000</th><th>Percent change</th>
    <th>1999</th><th>2000</th><th>Percent change</th>
  </tr>
  <tr>
    <td>Outlays</td><td>463.9</td><td>463.6</td><td>-0.1</td>
    <td>1699.7</td><td>1787.9</td><td>5.2</td>
  </tr>
</table>"""
    norm = normalize_html_table_for_comparison(html_text)
    header = norm.splitlines()[0]
    assert header.count("1999") == 2
    assert header.count("2000") == 2


def test_normalize_does_not_deduplicate_data_rows_or_continuation_tables_with_year_like_integers() -> (
    None
):
    """P0 silent-data-loss guard: ``<td>`` data rows — including first-row data
    in a headerless continuation table and empty-stub continuation rows — must
    NEVER have repeated 1800-2099 integers deduplicated."""
    # 1. Named data row directly under a <th> header.
    named_data_html = """\
<table>
  <tr><th>Category</th><th>A</th><th>B</th><th>C</th><th>D</th></tr>
  <tr><td>Total loans</td><td>1950</td><td>1950</td><td>2005</td><td>2005</td></tr>
  <tr><td>Real GDP</td><td>1.6</td><td>2.1</td><td>3.2</td><td>4.0</td></tr>
</table>"""
    norm = normalize_html_table_for_comparison(named_data_html)
    lines = norm.splitlines()
    assert lines[2] == "| Total loans | 1950 | 1950 | 2005 | 2005 |"
    assert lines[3] == "| Real GDP | 1.6 | 2.1 | 3.2 | 4.0 |"

    # 2. Headerless continuation table whose first row is <td> data (even 1-row).
    first_row_data_html = """\
<table>
  <tr><td>Total loans</td><td>1950</td><td>1950</td><td>2005</td><td>2005</td></tr>
</table>"""
    norm_first = normalize_html_table_for_comparison(first_row_data_html)
    assert norm_first.splitlines()[0] == "| Total loans | 1950 | 1950 | 2005 | 2005 |"

    # 3. Empty-stub <td> continuation row under a <thead> header.
    empty_stub_html = """\
<table>
  <thead><tr><th>Item</th><th>Banks</th><th>Thrifts</th><th>CUs</th></tr></thead>
  <tbody>
    <tr><td></td><td>1950</td><td>1950</td><td>2000</td></tr>
    <tr><td>Total</td><td>2000</td><td>2000</td><td>2000</td></tr>
  </tbody>
</table>"""
    norm_stub = normalize_html_table_for_comparison(empty_stub_html)
    stub_lines = norm_stub.splitlines()
    assert stub_lines[2] == "|  | 1950 | 1950 | 2000 |"
    assert stub_lines[3] == "| Total | 2000 | 2000 | 2000 |"


def test_normalize_leaves_unclosed_table_intact_to_fail_closed() -> None:
    truncated_html = "<table><tr><th>Item</th><th>2000</th></tr><tr><td>GDP</td><td>1.6"
    norm = normalize_html_table_for_comparison(truncated_html)
    assert "<table" in norm
    assert "</table>" not in norm


# ---------------------------------------------------------------------------
# 4. verify_table_crop_two_readers
# ---------------------------------------------------------------------------


class _StubReader:
    def __init__(
        self, model: str, output: str | Exception, finish_reason: str | None = "stop"
    ) -> None:
        self.model = model
        self._output = output
        self._finish = finish_reason

    def read_image(self, image: Path | str | bytes) -> tuple[str, str | None]:
        if isinstance(self._output, Exception):
            raise self._output
        return self._output, self._finish


_TABLE_A_HTML = """\
Table 1. Staff Forecast (percent change, annual rate)
<table>
  <tr><th>Category</th><th colspan="2">2000</th><th colspan="2">2001</th></tr>
  <tr><th></th><th>Q3</th><th>Q4</th><th>Q1</th><th>Q2</th></tr>
  <tr><td>PCE inflation</td><td>1.6</td><td>-0.1</td><td>2.3</td><td>0.4</td></tr>
  <tr><td>Core PCE</td><td>1.2</td><td>-0.3</td><td>1.9</td><td>0.2</td></tr>
</table>
Note: Seasonally adjusted.
"""

_TABLE_B_EXPANDED_HEADER_HTML = """\
<page_header>-11-</page_header>
<table>
  <tr><th>Category</th><th>2000</th><th>2000</th><th>2001</th><th>2001</th></tr>
  <tr><th></th><th>Q3</th><th>Q4</th><th>Q1</th><th>Q2</th></tr>
  <tr><td>PCE inflation</td><td>1.6</td><td>-0.1</td><td>2.3</td><td>0.4</td></tr>
  <tr><td>Core PCE</td><td>1.2</td><td>-0.3</td><td>1.9</td><td>0.2</td></tr>
</table>
"""


def test_two_readers_agree_when_only_furniture_and_spanned_headers_differ(tmp_path: Path) -> None:
    crop = tmp_path / "table.png"
    crop.write_bytes(_TINY_PNG)

    res = verify_table_crop_two_readers(
        crop,
        reader_a=_StubReader("llamaindex/ocria-1:bf16", _TABLE_A_HTML),
        reader_b=_StubReader("lighton-ocr-2", _TABLE_B_EXPANDED_HEADER_HTML),
    )
    assert isinstance(res, CropAgreementResult)
    assert res.agreed is True
    assert res.differences == ()
    assert res.reader_a_model == "llamaindex/ocria-1:bf16"
    assert res.reader_b_model == "lighton-ocr-2"
    assert "| PCE inflation | 1.6 | -0.1 | 2.3 | 0.4 |" in res.markdown
    # Shipped markdown preserves title/units/notes and full token multiset from reader A.
    assert "Table 1. Staff Forecast (percent change, annual rate)" in res.markdown
    assert "Note: Seasonally adjusted." in res.markdown
    # Comparison markdown strips furniture from reader B.
    assert "-11-" not in res.reader_b_markdown

    # When reader A has explicit repeated <th>2000</th> headers, verified markdown
    # returns the UN-DEDUPED converted reading preserving reader A's full numeric multiset.
    res_rev = verify_table_crop_two_readers(
        crop,
        reader_a=_StubReader("lighton-ocr-2", _TABLE_B_EXPANDED_HEADER_HTML),
        reader_b=_StubReader("llamaindex/ocria-1:bf16", _TABLE_A_HTML),
    )
    assert res_rev.agreed is True
    assert sorted(row_numbers(res_rev.markdown)) == sorted(
        row_numbers(
            "| Category | 2000 | 2000 | 2001 | 2001 |\n"
            "|  | Q3 | Q4 | Q1 | Q2 |\n"
            "| PCE inflation | 1.6 | -0.1 | 2.3 | 0.4 |\n"
            "| Core PCE | 1.2 | -0.3 | 1.9 | 0.2 |\n"
            "<page_header>-11-</page_header>"
        )
    )


@pytest.mark.parametrize(
    "mutated_b,label",
    [
        (_TABLE_B_EXPANDED_HEADER_HTML.replace("1.6", "1.8"), "digit_swap_1.6_vs_1.8"),
        (_TABLE_B_EXPANDED_HEADER_HTML.replace("-0.1", "0.1"), "sign_flip_-0.1_vs_0.1"),
        (_TABLE_B_EXPANDED_HEADER_HTML.replace("-0.1", "~0.1"), "tilde_minus_defect"),
        (
            _TABLE_B_EXPANDED_HEADER_HTML.replace(
                "<tr><td>Core PCE</td><td>1.2</td><td>-0.3</td><td>1.9</td><td>0.2</td></tr>",
                "",
            ),
            "dropped_data_row",
        ),
    ],
)
def test_two_readers_fail_closed_on_any_data_row_discrepancy(
    tmp_path: Path, mutated_b: str, label: str
) -> None:
    crop = tmp_path / f"{label}.png"
    crop.write_bytes(_TINY_PNG)

    res = verify_table_crop_two_readers(
        crop,
        reader_a=_StubReader("ocria-1", _TABLE_A_HTML),
        reader_b=_StubReader("glm-ocr", mutated_b),
    )
    assert res.agreed is False, f"{label} must fail closed"
    assert res.markdown == "", "disagreed crop must not ship verified markdown"
    assert res.differences, f"{label} must record concrete differences"


def test_two_readers_fail_closed_on_truncation_unclosed_empty_or_error(tmp_path: Path) -> None:
    crop = tmp_path / "t.png"
    crop.write_bytes(_TINY_PNG)

    # 1. Truncated reader (finish_reason="length")
    res_trunc = verify_table_crop_two_readers(
        crop,
        reader_a=_StubReader("ocria-1", _TABLE_A_HTML, finish_reason="stop"),
        reader_b=_StubReader("glm-ocr", _TABLE_B_EXPANDED_HEADER_HTML, finish_reason="length"),
    )
    assert res_trunc.agreed is False
    assert res_trunc.markdown == ""
    assert "truncated" in res_trunc.unknown_reason

    # 2. Unclosed HTML table (single reader AND both readers identically truncated)
    unclosed = "<table><tr><td>PCE</td><td>1.6</td></tr>"
    res_unclosed = verify_table_crop_two_readers(
        crop,
        reader_a=_StubReader("ocria-1", _TABLE_A_HTML),
        reader_b=_StubReader("glm-ocr", unclosed),
    )
    assert res_unclosed.agreed is False
    assert res_unclosed.markdown == ""
    assert "unclosed_markup" in res_unclosed.unknown_reason

    res_both_unclosed = verify_table_crop_two_readers(
        crop,
        reader_a=_StubReader("ocria-1", unclosed),
        reader_b=_StubReader("glm-ocr", unclosed),
    )
    assert res_both_unclosed.agreed is False
    assert res_both_unclosed.markdown == ""
    assert "both_unclosed_markup" in res_both_unclosed.unknown_reason

    # 2b. Unclosed page-furniture / watermark tag must fail closed even when table is closed.
    unclosed_furniture = "<page_header>II-14\n" + _TABLE_A_HTML
    res_unclosed_furn = verify_table_crop_two_readers(
        crop,
        reader_a=_StubReader("ocria-1", unclosed_furniture),
        reader_b=_StubReader("glm-ocr", _TABLE_B_EXPANDED_HEADER_HTML),
    )
    assert res_unclosed_furn.agreed is False
    assert res_unclosed_furn.markdown == ""
    assert "reader_a_unclosed_markup" in res_unclosed_furn.unknown_reason

    # 3. Empty output
    res_empty = verify_table_crop_two_readers(
        crop,
        reader_a=_StubReader("ocria-1", _TABLE_A_HTML),
        reader_b=_StubReader("glm-ocr", "   "),
    )
    assert res_empty.agreed is False
    assert res_empty.markdown == ""

    # 4. Reader raises exception
    res_err = verify_table_crop_two_readers(
        crop,
        reader_a=_StubReader("ocria-1", _TABLE_A_HTML),
        reader_b=_StubReader("glm-ocr", RuntimeError("connection reset")),
    )
    assert res_err.agreed is False
    assert res_err.markdown == ""
    assert "reader_b_error" in res_err.unknown_reason
