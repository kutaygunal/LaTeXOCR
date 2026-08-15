"""Unit and integration tests for src/ai_recognizer.py.

Covers: clean_output normalization (code fences, $ delimiters, labels,
whitespace, empty), image encoding, error handling (missing file, unreachable
Ollama), retry logic. Unit tests mock the HTTP layer so they never require a
live Ollama call; the integration test skips gracefully if Ollama is down.
"""

import json
import os
import sys
import urllib.error
import urllib.request

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ai_recognizer import (  # noqa: E402
    AIRecognizer,
    DEFAULT_RETRIES,
    RecognizerError,
    _build_prompt,
    _encode_image,
    _strip_code_fences,
    _strip_dollar_delimiters,
    _strip_prefix,
    clean_output,
    recognize,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")


def _sample_image(tier: str) -> str:
    for fname in sorted(os.listdir(IMAGES_DIR)):
        if f"_{tier}_" in fname:
            return os.path.join(IMAGES_DIR, fname)
    raise FileNotFoundError(f"no image for tier {tier!r}")


# ---------------------------------------------------------------------------
# clean_output normalization
# ---------------------------------------------------------------------------


def test_clean_output_plain():
    assert clean_output("x^2") == "x^2"


def test_clean_output_code_fence():
    assert clean_output("```latex\nx^2\n```") == "x^2"


def test_clean_output_code_fence_no_lang():
    assert clean_output("```\nx^2\n```") == "x^2"


def test_clean_output_dollar_single():
    assert clean_output("$x^2$") == "x^2"


def test_clean_output_dollar_double():
    assert clean_output("$$x^2$$") == "x^2"


def test_clean_output_label_colon():
    assert clean_output("latex: x^2") == "x^2"


def test_clean_output_label_equals():
    assert clean_output("output = x^2") == "x^2"


def test_clean_output_label_case_insensitive():
    assert clean_output("LaTeX: x^2") == "x^2"


def test_clean_output_whitespace_collapsed():
    assert clean_output("  x  ^  2  ") == "x ^ 2"


def test_clean_output_empty():
    assert clean_output("") == ""


def test_clean_output_whitespace_only():
    assert clean_output("   \n\t  ") == ""


def test_clean_output_combined():
    raw = "```latex\n$x^2 + y^2 = z^2$\n```"
    assert clean_output(raw) == "x^2 + y^2 = z^2"


def test_clean_output_label_then_dollar_keeps_dollars():
    """Documented order strips dollars before the prefix, so a label before a
    dollar-delimited expression leaves the dollars intact (minor limitation)."""
    assert clean_output("answer: $\\frac{a}{b}$") == "$\\frac{a}{b}$"


def test_strip_code_fences_no_fence_unchanged():
    assert _strip_code_fences("x^2") == "x^2"


def test_strip_dollar_no_delim_unchanged():
    assert _strip_dollar_delimiters("x^2") == "x^2"


def test_strip_prefix_no_label_unchanged():
    assert _strip_prefix("x^2") == "x^2"


# ---------------------------------------------------------------------------
# _encode_image
# ---------------------------------------------------------------------------


def test_encode_image_returns_base64_png():
    img = np.full((32, 32), 255, dtype=np.uint8)
    b64 = _encode_image(img)
    assert isinstance(b64, str)
    assert b64
    # Decode and check PNG magic bytes.
    import base64

    raw = base64.b64decode(b64)
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"


def test_build_prompt_mentions_latex():
    p = _build_prompt()
    assert "LaTeX" in p
    assert "ONLY" in p or "only" in p


# ---------------------------------------------------------------------------
# AIRecognizer — error handling & retry (mocked HTTP)
# ---------------------------------------------------------------------------


def _make_recognizer(**kwargs):
    r = AIRecognizer(retries=3, timeout=5, **kwargs)
    # Mock the image loader so no real file is needed.
    r._load_image = lambda path: np.full((32, 32), 255, dtype=np.uint8)
    return r


def test_recognize_missing_file_raises():
    r = AIRecognizer()
    with pytest.raises(RecognizerError):
        r.recognize(os.path.join(IMAGES_DIR, "nope.png"))


def test_recognize_unreachable_ollama_raises_after_retries():
    r = _make_recognizer()
    calls = {"n": 0}

    def fake_request(b64):
        calls["n"] += 1
        raise urllib.error.URLError("connection refused")

    r._request = fake_request
    with pytest.raises(RecognizerError):
        r.recognize("fake.png")
    assert calls["n"] == r.retries


def test_recognize_retries_then_succeeds():
    r = _make_recognizer()
    calls = {"n": 0}

    def fake_request(b64):
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.URLError("transient")
        return "x^2"

    r._request = fake_request
    out = r.recognize("fake.png")
    assert out == "x^2"
    assert calls["n"] == 3


def test_recognize_empty_output_retries_then_raises():
    r = _make_recognizer()
    calls = {"n": 0}

    def fake_request(b64):
        calls["n"] += 1
        return "   \n  "

    r._request = fake_request
    with pytest.raises(RecognizerError):
        r.recognize("fake.png")
    assert calls["n"] == r.retries


def test_recognize_cleans_model_output():
    r = _make_recognizer()
    r._request = lambda b64: "```latex\n$\\frac{a}{b}$\n```"
    assert r.recognize("fake.png") == "\\frac{a}{b}"


def test_recognize_http_error_retries():
    r = _make_recognizer()
    calls = {"n": 0}

    def fake_request(b64):
        calls["n"] += 1
        raise urllib.error.HTTPError("url", 500, "err", {}, None)

    r._request = fake_request
    with pytest.raises(RecognizerError):
        r.recognize("fake.png")
    assert calls["n"] == r.retries


def test_retries_min_one():
    r = AIRecognizer(retries=0)
    assert r.retries == 1


def test_use_preprocess_loads_via_preprocess(tmp_path):
    """When use_preprocess=True, _load_image runs the shared pipeline."""
    import cv2

    r = AIRecognizer(use_preprocess=True)
    p = tmp_path / "img.png"
    cv2.imwrite(str(p), np.full((100, 100), 255, dtype=np.uint8))
    img = r._load_image(str(p))
    assert img.ndim == 2
    assert img.dtype == np.uint8


# ---------------------------------------------------------------------------
# HTTP request construction (_request / _chat_endpoint) — mocked urlopen
# ---------------------------------------------------------------------------


class _FakeResponse:
    """A context-manager response whose read() returns a JSON body."""

    def __init__(self, body: dict):
        self._body = json.dumps(body).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self._body


def _capture_urlopen(monkeypatch, body: dict):
    """Replace urllib.request.urlopen and return a dict capturing the request."""
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["req"] = req
        captured["timeout"] = timeout
        return _FakeResponse(body)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return captured


def test_chat_endpoint_trailing_slash():
    assert AIRecognizer(base_url="http://x:11434")._chat_endpoint() == \
        "http://x:11434/api/chat"
    assert AIRecognizer(base_url="http://x:11434/")._chat_endpoint() == \
        "http://x:11434/api/chat"
    assert AIRecognizer(base_url="http://x:11434///")._chat_endpoint() == \
        "http://x:11434/api/chat"


def test_request_posts_correct_payload(monkeypatch):
    captured = _capture_urlopen(monkeypatch, {"message": {"content": "x^2"}})
    r = AIRecognizer(
        base_url="http://localhost:11434/",
        model="qwen3-vl:8b",
        temperature=0.5,
        timeout=30,
    )
    out = r._request("BASE64DATA")
    assert out == "x^2"

    req = captured["req"]
    # Correct URL and method.
    assert req.full_url == "http://localhost:11434/api/chat"
    assert req.method == "POST"
    # Content-Type header (urllib normalizes the key to 'Content-type').
    assert req.headers.get("Content-type") == "application/json"
    # Timeout forwarded.
    assert captured["timeout"] == 30
    # Decoded JSON body structure.
    body = json.loads(req.data.decode("utf-8"))
    assert body["model"] == "qwen3-vl:8b"
    assert body["stream"] is False
    assert body["options"]["temperature"] == 0.5
    msg = body["messages"][0]
    assert msg["role"] == "user"
    assert msg["images"] == ["BASE64DATA"]
    assert "LaTeX" in msg["content"]


def test_request_uses_default_temperature(monkeypatch):
    captured = _capture_urlopen(monkeypatch, {"message": {"content": "y"}})
    r = AIRecognizer()  # defaults
    r._request("B64")
    body = json.loads(captured["req"].data.decode("utf-8"))
    assert body["options"]["temperature"] == 0.0


def test_request_empty_content_raises(monkeypatch):
    _capture_urlopen(monkeypatch, {"message": {"content": "   "}})
    r = AIRecognizer()
    with pytest.raises(RecognizerError):
        r._request("B64")


def test_request_missing_message_raises(monkeypatch):
    _capture_urlopen(monkeypatch, {"foo": "bar"})
    r = AIRecognizer()
    with pytest.raises(RecognizerError):
        r._request("B64")


def test_request_malformed_json_raises(monkeypatch):
    class BadResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"not json"

    def fake_urlopen(req, timeout=None):
        return BadResp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    r = AIRecognizer()
    with pytest.raises(Exception):
        r._request("B64")


# ---------------------------------------------------------------------------
# Integration (real Ollama) — skipped if unreachable
# ---------------------------------------------------------------------------


def _ollama_reachable() -> bool:
    try:
        import urllib.request

        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _ollama_reachable(), reason="Ollama not reachable")
def test_recognize_real_image_integration():
    """End-to-end against a real clean image (requires live Ollama)."""
    r = AIRecognizer(timeout=60)
    out = r.recognize(_sample_image("clean"))
    assert isinstance(out, str)
    assert out.strip() != ""
