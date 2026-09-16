"""Isolated worker process for the FormulaNet (PP-FormulaNet_plus-L) engine.

Paddle and PyTorch cannot coexist in one Windows process: both bundle a binary
named ``cudnn_cnn64_9.dll``, and whichever loads after the other fails with a
DLL "procedure not found" error. ``paddleocr`` pulls in ``torch`` internally, so
even ``import paddleocr`` is unsafe in any process that has loaded torch (which
the AI and own-code engines do).

Solution: run PaddleOCR exclusively inside this long-lived child process and let
the main process talk to it over a simple newline-delimited JSON protocol on
stdin/stdout. The worker imports the model once and serves repeated requests,
so the ~735 MB model is not reloaded per image.

Protocol (one JSON object per line on stdin, one per line on stdout):
    in:  {"op":"recognize","path": "<image_path>", "batch_size": 1}
    out: {"ok": true,  "latex": "\\frac{a}{b}"}
         {"ok": false, "error": "message"}
    in:  {"op":"ping"}
    out: {"ok": true}
"""

from __future__ import annotations

import json
import importlib.abc
import importlib.machinery
import importlib.util
import os
import site
import sys
import types


# ---------------------------------------------------------------------------
# Torch stub
# ---------------------------------------------------------------------------
# ``paddleocr`` -> ``paddlex`` -> ``modelscope`` import ``torch`` at module
# load time even though the formula model runs entirely on Paddle. Loading a
# real torch is fatal here: torch and paddle each ship a ``cudnn_cnn64_9.dll``
# and Windows loads only one, so importing real torch crashes the worker with a
# DLL "procedure not found" error. We install a stub so those imports succeed
# WITHOUT loading any CUDA/cuDNN DLL. Paddle never uses this stub during predict.

def _install_torch_stub() -> None:
    if "torch" in sys.modules:
        return
    subs = [
        "nn", "nn.functional", "nn.init", "nn.utils", "utils", "utils.data",
        "multiprocessing", "cuda", "backends", "backends.cudnn", "optim",
        "autograd", "distributed", "hub", "serialization", "amp", "onnx",
        "utils.cpp_extension", "linalg",
    ]
    for name in subs:
        mod = types.ModuleType("torch." + name)
        mod.__path__ = []
        sys.modules["torch." + name] = mod
    t = types.ModuleType("torch")
    t.__path__ = []
    t.__spec__ = importlib.machinery.ModuleSpec("torch", None)
    for s in subs:
        setattr(t, s, sys.modules["torch." + s])

    class _Module:
        pass

    class _Linear(_Module):
        pass

    t.nn.Module = _Module
    t.nn.Linear = _Linear
    t.nn.Parameter = types.SimpleNamespace
    t.Tensor = object
    t.ByteTensor = object
    t.LongTensor = object
    t.ByteStorage = object
    t.device = lambda *a, **k: None
    t.no_grad = lambda f=None, *a, **k: (f if callable(f) else (lambda g: g))
    t.compile = lambda f, *a, **k: f
    for fn in ("cat", "tensor", "zeros", "ones", "empty", "full", "arange"):
        setattr(t, fn, lambda *a, **k: None)
    t.manual_seed = lambda s: None
    t.__version__ = "99.0-stub"
    t.int = int
    t.uint = int
    t.float = float
    t.long = int
    t.bool = bool
    cuda = sys.modules["torch.cuda"]
    cuda.device_count = lambda: 1
    cuda.set_device = lambda d: None
    cuda.manual_seed_all = lambda s: None
    cuda.is_available = lambda: True
    t.cuda = cuda
    dist = sys.modules["torch.distributed"]
    dist.is_initialized = lambda: False
    dist.get_rank = lambda: 0
    dist.get_world_size = lambda: 1
    dist.is_available = lambda: False
    sys.modules["torch"] = t


class _SpecFinder(importlib.abc.MetaPathFinder):
    """Give already-stubbed modules a spec so ``find_spec`` won't raise."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname in sys.modules:
            return importlib.machinery.ModuleSpec(fullname, None)
        return None


_install_torch_stub()
sys.meta_path.insert(0, _SpecFinder())


# ---------------------------------------------------------------------------
# PATH for Paddle's bundled CUDA/CUDA libs
# ---------------------------------------------------------------------------
# Paddle's CUDA/cuDNN libraries live under site-packages/nvidia. Windows DLL
# search does not look there by default, so add them to PATH so the cudnn/cublas
# DLLs resolve.

def _prepend_nvidia_bins() -> None:
    sp_dir = site.getsitepackages()[0]
    bin_dirs: list[str] = []
    for sub in ("cuda_runtime", "cudnn", "cublas", "cufft", "curand",
                "cusolver", "cusparse", "nvjitlink"):
        for subname in ("bin", ""):
            d = os.path.join(sp_dir, "nvidia", sub, subname)
            if os.path.isdir(d):
                bin_dirs.insert(0, d)
    # Put cuda_runtime root (contains cudart64_*.dll) first.
    existing = os.environ.get("PATH", "")
    os.environ["PATH"] = os.pathsep.join([*bin_dirs, existing])


_prepend_nvidia_bins()

import paddle  # noqa: E402

_formula_model = None


def _detect_device() -> str:
    """Pick ``gpu:0`` if a CUDA GPU is visible, else ``cpu``."""
    try:
        import paddle

        if (paddle.device.is_compiled_with_cuda()
                and paddle.device.cuda.device_count() > 0):
            return "gpu:0"
    except Exception:
        pass
    return "cpu"


def _build_model(device: str):
    global _formula_model
    if _formula_model is not None:
        return _formula_model
    from paddleocr import FormulaRecognition

    _formula_model = FormulaRecognition(model_name="PP-FormulaNet_plus-L",
                                        device=device)
    return _formula_model


def _extract_latex(output) -> str:
    for item in output:
        if item is None:
            continue
        if hasattr(item, "get"):
            value = item.get("rec_formula")
            if not value:
                res = item.get("res")
                if isinstance(res, dict):
                    value = res.get("rec_formula")
            if value:
                return str(value).strip()
        else:
            value = getattr(item, "rec_formula", None)
            if value is None:
                res = getattr(item, "res", None)
                value = getattr(res, "rec_formula", None)
            if value:
                return str(value).strip()
    return ""


def _serve(device: str) -> int:
    try:
        _build_model(device)
    except Exception as exc:  # surface init errors to the client
        sys.stdout.write(json.dumps({"ok": False, "fatal": True,
                                     "error": str(exc)}) + "\n")
        sys.stdout.flush()
        return 1

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            sys.stdout.write(json.dumps({"ok": False, "error": "bad json"}) + "\n")
            sys.stdout.flush()
            continue

        op = req.get("op")
        if op == "ping":
            sys.stdout.write(json.dumps({"ok": True}) + "\n")
        elif op == "recognize":
            path = req.get("path", "")
            batch = int(req.get("batch_size") or 1)
            result: dict
            try:
                if not os.path.isfile(path):
                    result = {"ok": False, "error": f"image not found: {path}"}
                else:
                    out = _formula_model.predict(path, batch_size=batch)
                    latex = _extract_latex(out)
                    if not latex:
                        result = {"ok": False, "error": "no formula produced"}
                    else:
                        result = {"ok": True, "latex": latex}
            except Exception as exc:  # a failing image must not kill the worker
                result = {"ok": False, "error": str(exc)}
            sys.stdout.write(json.dumps(result) + "\n")
        else:
            sys.stdout.write(json.dumps({"ok": False, "error": f"bad op {op}"})
                             + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    device = os.environ.get("FORMULANET_DEVICE") or _detect_device()
    sys.exit(_serve(device))
