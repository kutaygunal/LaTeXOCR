"""Tests for the FormulaNet (PP-FormulaNet_plus-L) recognizer.

The tests cover contract behaviour (defaults, missing-image handling, worker
lifecycle) and the graceful-failure path on a worker that cannot start. They
make PaddleOCR availability irrelevant by mocking the worker process, so they
run whether or not PaddlePaddle is installed on the test machine.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import formulanet_recognizer as fr  # noqa: E402
from formulanet_recognizer import (  # noqa: E402
    DEFAULT_MODEL,
    FormulaNetError,
    FormulaNetRecognizer,
    recognize,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")


def _sample_image(tier: str) -> str:
    for fname in sorted(os.listdir(IMAGES_DIR)):
        if f"_{tier}_" in fname:
            return os.path.join(IMAGES_DIR, fname)
    raise FileNotFoundError(f"no image for tier {tier!r}")


@pytest.fixture(autouse=True)
def _reset_worker_handle():
    """Reset the module-level worker handle so tests are independent."""
    old = fr._RECOGNIZER_HANDLE
    fr._RECOGNIZER_HANDLE = None
    yield
    fr._RECOGNIZER_HANDLE = old


# ---------------------------------------------------------------------------
# Protocol-aware fake worker
# ---------------------------------------------------------------------------


class _RespondingSource:
    """Fake worker stdout that answers each request line correctly.

    Protocol:
        in:  {"op": "ping"}          -> out: {"ok": true}
        in:  {"op": "recognize",...} -> out: {"ok": true, "latex": ...}
    Reads the single write placed in the sink and answers accordingly, so
    alignment stays correct across any number of calls.
    """

    def __init__(self, latex):
        self._latex = latex
        self._pending = None
        self.fail_ping = False

    def on_write(self, line: str):
        self._pending = line

    def readline(self):
        line = self._pending
        self._pending = None
        if line is None:
            return ""
        req = json.loads(line)
        if req.get("op") == "ping":
            if self.fail_ping:
                return '{"ok": false, "error": "boom"}'
            return '{"ok": true}'
        if req.get("op") == "recognize":
            return json.dumps({"ok": True, "latex": self._latex})
        return '{"ok": false, "error": "bad op"}'


class _Sink:
    def __init__(self, source):
        self._source = source

    def write(self, s):
        self._source.on_write(s.rstrip("\n"))
        return len(s)

    def flush(self):
        pass

    def close(self):
        pass


class _FakeProc:
    """Minimal Popen stand-in wired to a protocol-aware worker."""

    def __init__(self, latex=r"\frac{a}{b}", fail_ping=False):
        self._source = _RespondingSource(latex)
        self._source.fail_ping = fail_ping
        self.stdout = self._source
        self.stdin = _Sink(self._source)
        self.poll_calls = 0

    def poll(self):
        self.poll_calls += 1
        return None

    def terminate(self):
        pass

    def wait(self, timeout=None):
        pass

    def kill(self):
        pass


class _ClosedAfterPing:
    """Answers ping then returns EOF, simulating a worker that dies mid-call."""

    def __init__(self):
        self._asked = False
        self.stdin = self
        self.stdout = self
        self.poll_calls = 0

    def write(self, s):
        self._asked = True
        return len(s)

    def flush(self):
        pass

    def close(self):
        pass

    def readline(self):
        if not self._asked:
            self._asked = True
            return '{"ok": true}'
        return ""

    def poll(self):
        self.poll_calls += 1
        return None

    def terminate(self):
        pass

    def wait(self, timeout=None):
        pass

    def kill(self):
        pass


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------


def test_default_model_name():
    assert DEFAULT_MODEL == "PP-FormulaNet_plus-L"


def test_recognizer_defaults():
    rec = FormulaNetRecognizer()
    assert rec.model_name == DEFAULT_MODEL
    assert rec.device == "auto"
    assert rec.batch_size == 1
    assert rec.use_preprocess is False


def test_recognizer_batch_size_min_one():
    rec = FormulaNetRecognizer(batch_size=0)
    assert rec.batch_size == 1


def test_device_set_from_arg():
    assert FormulaNetRecognizer(device="gpu:0").device == "gpu:0"


def test_recognize_missing_image_raises_before_worker(monkeypatch):
    """A missing image is reported before the worker is ever spawned."""
    monkeypatch.setattr(fr.subprocess, "Popen", _raise_BrokenPipe)
    rec = FormulaNetRecognizer()
    with pytest.raises(FormulaNetError):
        rec.recognize(os.path.join(IMAGES_DIR, "does_not_exist.png"))


def _raise_BrokenPipe(*a, **k):
    raise AssertionError("Popen should not have been called")


def test_worker_failure_reported_as_formulanet_error(monkeypatch):
    """If the worker cannot start, recognize raises FormulaNetError (not a
    crash), so the benchmark harness marks the engine 'unavailable'."""
    monkeypatch.setattr(fr.subprocess, "Popen",
                        lambda *a, **k: _FakeProc(fail_ping=True))
    rec = FormulaNetRecognizer(device="cpu")
    with pytest.raises(FormulaNetError):
        rec.recognize(_sample_image("clean"))


def test_worker_crash_recovers_cleanly(monkeypatch):
    """A lost worker raises FormulaNetError (not a low-level exception)."""
    monkeypatch.setattr(fr.subprocess, "Popen",
                        lambda *a, **k: _ClosedAfterPing())
    rec = FormulaNetRecognizer(device="cpu")
    with pytest.raises(FormulaNetError):
        rec.recognize(_sample_image("clean"))


def test_convenience_wrapper_uses_worker(monkeypatch):
    """recognize() delegates to a FormulaNetRecognizer instance and reuses the
    module-level handle across calls."""
    monkeypatch.setattr(fr.subprocess, "Popen",
                        lambda *a, **k: _FakeProc(latex=r"\frac{a}{b}"))
    out = recognize(_sample_image("clean"))
    assert out == r"\frac{a}{b}"
    assert fr._RECOGNIZER_HANDLE is not None
    out2 = recognize(_sample_image("clean"))
    assert out2 == r"\frac{a}{b}"


def test_worker_output_empty_is_error(monkeypatch):
    """A recognize response reporting no formula surfaces as FormulaNetError."""
    proc = _FakeProc()
    # Override the source so an empty-latex recognize returns ok=False, which is
    # exactly what the real worker does when it cannot produce a formula.
    real = proc._source.readline

    def readline():
        line = proc._source._pending
        proc._source._pending = None
        if line is None:
            return ""
        req = json.loads(line)
        if req.get("op") == "ping":
            return '{"ok": true}'
        if req.get("op") == "recognize":
            return '{"ok": false, "error": "no formula produced"}'
        return '{"ok": false, "error": "bad op"}'

    proc._source.readline = readline
    monkeypatch.setattr(fr.subprocess, "Popen", lambda *a, **k: proc)
    rec = FormulaNetRecognizer(device="cpu")
    with pytest.raises(FormulaNetError):
        rec.recognize(_sample_image("clean"))
