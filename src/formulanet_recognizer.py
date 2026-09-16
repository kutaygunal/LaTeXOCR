"""PP-FormulaNet_plus-L recognizer via an isolated PaddleOCR worker process.

Wraps PaddlePaddle's ``PP-FormulaNet_plus-L`` formula recognition model so that
a formula image is converted into a LaTeX string. The model is pre-trained on a
large formula corpus (Chinese dissertations, textbooks, exam papers, math
journals, arXiv); no training is performed on the project dataset, so the
pipeline is safe to evaluate on the held-out ``test_set``.

Why an isolated worker?
-----------------------
Paddle and PyTorch cannot coexist in a single Windows process: both bundle a
binary named ``cudnn_cnn64_9.dll``, and whichever loads second crashes with a
Windows "procedure not found" error. ``paddleocr`` imports ``torch`` internally,
so importing it in the main process (where the AI and own-code engines also
live) is unsafe. To keep all three engines in one app, we run PaddleOCR in a
long-lived child process (``_formulanet_worker.py``) and talk to it over a
newline-delimited JSON protocol on stdin/stdout. The worker loads the ~735 MB
model once and serves every image.

Public API
----------
- ``FormulaNetRecognizer`` : stateful IPC client to the worker process.
- ``recognize(image_path) -> str`` : convenience wrapper using defaults.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

# Path to the worker script (same directory as this module).
_WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "_formulanet_worker.py")

# Default PaddleOCR model name.
DEFAULT_MODEL = "PP-FormulaNet_plus-L"

# PaddlePaddle install hint (platform-specific wheel, so not in requirements.txt).
INSTALL_HINT = (
    "PaddleOCR / PaddlePaddle are not installed. Install them first, e.g. "
    "`python -m pip install paddlepaddle==3.0.0 paddleocr` "
    "or `python -m pip install paddlepaddle-gpu==3.0.0 -i "
    "https://www.paddlepaddle.org.cn/packages/stable/cu126/`."
)


class FormulaNetError(Exception):
    """Raised when the FormulaNet recognizer cannot produce a result."""


class FormulaNetRecognizer:
    """Recognize LaTeX equations in images using PP-FormulaNet_plus-L.

    Parameters
    ----------
    model_name : str
        PaddleOCR formula model name (default ``PP-FormulaNet_plus-L``).
    device : str | None
        Worker ``FORMULANET_DEVICE`` (``"gpu:0"`` or ``"cpu"``). Defaults to CPU
        for portability; pass ``gpu:0`` for speed.
    batch_size : int
        Batch size for a single prediction call (default 1).
    use_preprocess : bool
        Unused and deprecated for this engine. The FormulaNet model expects a
        full-resolution, cropped formula image, so the project's aggressive
        64-px preprocessing would hurt it.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        batch_size: int = 1,
        use_preprocess: bool = False,
    ) -> None:
        self.model_name = model_name
        # None -> worker auto-detects (GPU if present).
        self.device = device or "auto"
        self.batch_size = max(1, int(batch_size))
        self.use_preprocess = use_preprocess
        self._proc: subprocess.Popen | None = None

    # -- worker lifecycle ---------------------------------------------------

    def _ensure_worker(self) -> subprocess.Popen:
        """Spawn the worker if not already running, then return it."""
        if self._proc is not None and self._proc.poll() is None:
            return self._proc

        env = dict(os.environ)
        if self.device != "auto":
            env["FORMULANET_DEVICE"] = self.device
        env.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        try:
            proc = subprocess.Popen(
                [sys.executable, _WORKER],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                bufsize=1,
            )
        except OSError as exc:
            raise FormulaNetError(f"Could not start FormulaNet worker: {exc}") \
                from exc

        # Check the worker booted (model load may raise a fatal error).
        try:
            self._ping(proc)
        except Exception:
            self._shutdown(proc)
            raise
        self._proc = proc
        return proc

    def _request(self, proc: subprocess.Popen, payload: dict) -> dict:
        """Send one request and return the parsed response, resyncing on error."""
        assert proc.stdin is not None and proc.stdout is not None
        try:
            proc.stdin.write(json.dumps(payload) + "\n")
            proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            # Worker died part-way; resynchronize from a fresh process.
            if self._proc is proc:
                self._proc = None
            raise FormulaNetError(f"FormulaNet worker lost: {exc}") from exc
        raw = proc.stdout.readline()
        if not raw:
            if self._proc is proc:
                self._proc = None
            raise FormulaNetError("FormulaNet worker closed its output")
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise FormulaNetError(f"Bad FormulaNet worker response: {raw!r}") \
                from exc

    def _ping(self, proc: subprocess.Popen) -> None:
        resp = self._request(proc, {"op": "ping"})
        if not resp.get("ok"):
            raise FormulaNetError(
                f"FormulaNet worker failed to start: {resp.get('error')}"
            )

    @staticmethod
    def _shutdown(proc: subprocess.Popen) -> None:
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except Exception:
            pass
        try:
            proc.terminate()
        except Exception:
            pass
        try:
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    def close(self) -> None:
        """Terminate the worker subprocess."""
        if self._proc is not None:
            self._shutdown(self._proc)
            self._proc = None

    # -- recognition ---------------------------------------------------------

    def recognize(self, image_path: str) -> str:
        """Recognize the LaTeX equation in ``image_path``.

        Returns the LaTeX string produced by PP-FormulaNet_plus-L. Raises
        :class:`FormulaNetError` if PaddleOCR is unavailable, the worker cannot
        start, or the model returns nothing useful.
        """
        if not os.path.isfile(image_path):
            raise FormulaNetError(f"Image not found: {image_path}")
        proc = self._ensure_worker()
        resp = self._request(
            proc,
            {"op": "recognize", "path": os.path.abspath(image_path),
             "batch_size": self.batch_size},
        )
        if resp.get("ok"):
            return resp.get("latex", "")
        error = resp.get("error") or "unknown worker error"
        if resp.get("fatal"):
            # Model failed to load; tear down so next call retries cleanly.
            self.close()
        raise FormulaNetError(f"FormulaNet recognition failed: {error}")


_RECOGNIZER_HANDLE: FormulaNetRecognizer | None = None


def recognize(image_path: str, **kwargs) -> str:
    """Convenience wrapper: recognize an image with default settings.

    A module-level worker is reused across calls so a benchmark over many
    images does not respawn the model per image. Extra keyword arguments are
    forwarded to :class:`FormulaNetRecognizer`.
    """
    global _RECOGNIZER_HANDLE
    if _RECOGNIZER_HANDLE is None:
        _RECOGNIZER_HANDLE = FormulaNetRecognizer(**kwargs)
    return _RECOGNIZER_HANDLE.recognize(image_path)
