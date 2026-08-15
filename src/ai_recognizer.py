"""Local AI recognizer using the Ollama vision model.

Converts a LaTeX equation image into a LaTeX string by calling the local
Ollama ``qwen3-vl:8b`` vision-language model over the HTTP API. The model is
pre-trained; no training is performed on the project dataset, so it is safe
to evaluate on the held-out ``test_set``.

Public API
----------
- ``AIRecognizer`` : configurable recognizer (base URL, model, temperature).
- ``recognize(image_path) -> str`` : convenience wrapper using defaults.
- ``clean_output(raw) -> str`` : normalize raw model output into LaTeX.
"""

from __future__ import annotations

import base64
import json
import re
import time
import urllib.error
import urllib.request

import cv2
import numpy as np

from preprocess import preprocess

# Default Ollama endpoint and model.
DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3-vl:8b"

# Number of attempts for a single recognition call.
DEFAULT_RETRIES = 3

# Low temperature keeps the model's output deterministic.
DEFAULT_TEMPERATURE = 0.0

# Timeout (seconds) for a single HTTP request to Ollama.
DEFAULT_TIMEOUT = 120


class RecognizerError(Exception):
    """Raised when the AI recognizer cannot produce a result."""


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------


def _encode_image(image: np.ndarray) -> str:
    """Encode a grayscale image array as a base64 PNG string."""
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise RecognizerError("Could not encode image to PNG")
    return base64.b64encode(buf.tobytes()).decode("ascii")


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def _build_prompt() -> str:
    """Return the tuned prompt that instructs the model to emit only LaTeX."""
    return (
        "You are an expert at converting images of mathematical equations into "
        "LaTeX code. Look at the image and output ONLY the LaTeX code that "
        "reproduces the equation. Do not wrap it in markdown code fences, do "
        "not include dollar signs, and do not add any explanation or extra "
        "text. Output a single line of valid LaTeX."
    )


# ---------------------------------------------------------------------------
# Output cleaning / normalization
# ---------------------------------------------------------------------------


def _strip_code_fences(text: str) -> str:
    """Remove a markdown code fence (`````latex ... `````) if present."""
    match = re.search(r"```[a-zA-Z]*\s*(.*?)```", text, flags=re.DOTALL)
    if match:
        return match.group(1)
    return text


def _strip_dollar_delimiters(text: str) -> str:
    """Remove a single surrounding pair of ``$...$`` or ``$$...$$``."""
    t = text.strip()
    if t.startswith("$$") and t.endswith("$$"):
        return t[2:-2].strip()
    if t.startswith("$") and t.endswith("$"):
        return t[1:-1].strip()
    return t


def _strip_prefix(text: str) -> str:
    """Remove common leading labels the model may add (e.g. ``latex:``)."""
    t = text.strip()
    match = re.match(
        r"^(?:latex|latex code|output|result|answer)\s*[:=]\s*",
        t,
        flags=re.IGNORECASE,
    )
    if match:
        return t[match.end():].strip()
    return t


def _collapse_whitespace(text: str) -> str:
    """Collapse runs of whitespace to a single space and strip."""
    return re.sub(r"\s+", " ", text).strip()


def clean_output(raw: str) -> str:
    """Normalize raw model output into a clean single-line LaTeX string.

    Applies, in order: code-fence removal, dollar-delimiter removal, leading
    label removal, and whitespace collapsing. Returns an empty string if the
    result is empty after cleaning.
    """
    text = _strip_code_fences(raw)
    text = _strip_dollar_delimiters(text)
    text = _strip_prefix(text)
    text = _collapse_whitespace(text)
    return text


# ---------------------------------------------------------------------------
# Recognizer
# ---------------------------------------------------------------------------


class AIRecognizer:
    """Recognize LaTeX equations in images using the local Ollama vision model.

    Parameters
    ----------
    base_url : str
        Ollama HTTP base URL (default ``http://localhost:11434``).
    model : str
        Ollama model name (default ``qwen3-vl:8b``).
    temperature : float
        Sampling temperature (default 0.0 for determinism).
    retries : int
        Number of attempts before giving up (default 3).
    timeout : float
        Per-request timeout in seconds (default 120).
    use_preprocess : bool
        Whether to run the shared preprocessing pipeline before sending the
        image to the model. Defaults to False: the raw image is sent because
        the vision model reads it more accurately than the 64px binarized
        preprocessed version (empirically verified).
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        retries: int = DEFAULT_RETRIES,
        timeout: float = DEFAULT_TIMEOUT,
        use_preprocess: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.retries = max(1, int(retries))
        self.timeout = timeout
        self.use_preprocess = use_preprocess

    def _chat_endpoint(self) -> str:
        return f"{self.base_url}/api/chat"

    def _load_image(self, image_path: str) -> np.ndarray:
        """Load the image, optionally running the shared preprocessing."""
        if self.use_preprocess:
            return preprocess(image_path)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RecognizerError(f"Could not load image: {image_path}")
        return img

    def _request(self, image_b64: str) -> str:
        """POST one chat request to Ollama and return the raw model content."""
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": _build_prompt(),
                    "images": [image_b64],
                }
            ],
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._chat_endpoint(),
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        content = body.get("message", {}).get("content", "")
        if not isinstance(content, str) or not content.strip():
            raise RecognizerError("Ollama returned an empty response")
        return content

    def recognize(self, image_path: str) -> str:
        """Recognize the LaTeX equation in ``image_path``.

        Returns the cleaned LaTeX string. Raises ``RecognizerError`` if the
        image cannot be loaded or Ollama is unavailable after all retries.
        """
        image = self._load_image(image_path)
        image_b64 = _encode_image(image)

        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                raw = self._request(image_b64)
                cleaned = clean_output(raw)
                if cleaned:
                    return cleaned
                last_error = RecognizerError("Model produced empty output")
            except (
                urllib.error.URLError,
                urllib.error.HTTPError,
                TimeoutError,
                OSError,
            ) as exc:
                last_error = exc
            if attempt < self.retries - 1:
                time.sleep(0.5 * (attempt + 1))

        raise RecognizerError(
            f"AI recognition failed after {self.retries} attempts: {last_error}"
        )


def recognize(image_path: str, **kwargs) -> str:
    """Convenience wrapper: recognize an image with default settings.

    Extra keyword arguments are forwarded to :class:`AIRecognizer`.
    """
    return AIRecognizer(**kwargs).recognize(image_path)
