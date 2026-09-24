"""Backend registry, aliases and checkpoint-format-aware selection for laya-universal."""

from __future__ import annotations

import platform
import sys
from pathlib import Path

from .base import InferenceBackend, ModelHandle

__all__ = [
    "InferenceBackend",
    "ModelHandle",
    "ONNX_WEIGHT_FILES",
    "available_backends",
    "choose_backend",
    "get_backend",
    "onnx_weight_file",
    "register_backend",
    "select_backend",
]

_BACKENDS: dict[str, InferenceBackend] = {}

# Machine preference when nothing is forced: native MLX first, then ONNX providers.
_MACHINE_ORDER = ["mlx", "onnx-cuda", "onnx-directml", "onnx-coreml", "onnx-cpu"]
_ONNX_ORDER = ["onnx-cuda", "onnx-directml", "onnx-coreml", "onnx-cpu"]

# Weight files each backend family can load: ONNX backends read *.onnx, MLX reads
# model.safetensors. Used both to pick a backend and to scope Hub downloads.
ONNX_WEIGHT_FILES = ("model.onnx", "model_fp32.onnx", "laya.onnx")

_ALIASES = {
    "onnx": "onnx-best",
    "gpu": "onnx-best",
    "dml": "onnx-directml",
    "directml": "onnx-directml",
    "cuda": "onnx-cuda",
    "coreml": "onnx-coreml",
    "cpu": "onnx-cpu",
}


def register_backend(name: str, backend: InferenceBackend) -> None:
    """Register a backend instance (also used by tests and third-party backends)."""
    _BACKENDS[name] = backend


def available_backends() -> list[str]:
    """Names of registered backends that can run on this machine."""
    return [name for name, backend in _BACKENDS.items() if backend.available()]


def get_backend(name: str) -> InferenceBackend:
    if name not in _BACKENDS:
        raise ValueError(
            f"Unknown backend {name!r}. Registered: {sorted(_BACKENDS)}. "
            "Run 'laya-universal info' to see which are available here."
        )
    return _BACKENDS[name]


def _first_available(order: list[str]) -> InferenceBackend | None:
    for name in order:
        backend = _BACKENDS.get(name)
        if backend is not None and backend.available():
            return backend
    return None


def select_backend(preferred: str | None = None) -> InferenceBackend:
    """Best backend for this machine, ignoring what the checkpoint contains.

    `preferred` accepts a registered name or an alias: 'onnx' (best ONNX provider
    available here), 'dml'/'directml', 'cuda', 'coreml', 'cpu'.
    """
    if preferred:
        key = _ALIASES.get(str(preferred).strip().lower(), str(preferred).strip().lower())
        if key == "onnx-best":
            backend = _first_available(_ONNX_ORDER)
            if backend is None:
                raise RuntimeError("No ONNX backend is available; pip install onnxruntime")
            return backend
        return get_backend(key)
    backend = _first_available(_MACHINE_ORDER)
    if backend is None:
        raise RuntimeError(
            "No inference backend available. Install onnxruntime:\n"
            "  pip install onnxruntime            (CPU, all platforms)\n"
            "  pip install onnxruntime-directml  (any GPU on Windows)\n"
            "  pip install onnxruntime-gpu        (NVIDIA GPU)"
        )
    return backend


def onnx_weight_file(model_dir) -> str | None:
    """Name of the ONNX weight file present in `model_dir`, or None."""
    path = Path(model_dir)
    for name in ONNX_WEIGHT_FILES:
        if (path / name).is_file():
            return name
    return None


def choose_backend(model_dir, preferred: str | None = None) -> InferenceBackend:
    """Pick a backend that can actually load the weights in `model_dir`.

    Auto-selection prefers the machine's best backend but never picks one that
    cannot read the checkpoint: MLX cannot read .onnx files, and the ONNX
    backends cannot read model.safetensors. An explicit `preferred` bypasses
    the format check (the backend's own loader then reports any mismatch).
    """
    if preferred:
        return select_backend(preferred)
    has_onnx = onnx_weight_file(model_dir) is not None
    has_safetensors = (Path(model_dir) / "model.safetensors").is_file()
    if not has_onnx and not has_safetensors:
        raise FileNotFoundError(
            f"{model_dir} contains neither an ONNX model ({'/'.join(ONNX_WEIGHT_FILES)}) "
            "nor model.safetensors; not a runnable Laya checkpoint"
        )
    best = select_backend()
    if has_onnx:
        if best.name.startswith("onnx"):
            return best
        onnx = _first_available(_ONNX_ORDER)  # e.g. Apple: MLX cannot read .onnx
        if onnx is None:
            raise RuntimeError("This checkpoint is ONNX, but no ONNX backend is available here")
        return onnx
    if best.name == "mlx":
        return best
    mlx = _BACKENDS.get("mlx")
    if mlx is not None and mlx.available():
        return mlx
    raise RuntimeError(
        "This checkpoint ships model.safetensors, which only the MLX backend "
        "(Apple Silicon) can read, and no ONNX model was found. Export an ONNX "
        "copy once:\n"
        '  pip install "laya-universal[export]"\n'
        "  laya-universal convert --model <checkpoint> --output ./laya-onnx\n"
        "then load('./laya-onnx')."
    )


def _init_backends() -> None:
    """Register built-in backends once at import time."""
    try:
        from .onnx_backend import ONNXBackend

        register_backend("onnx-cpu", ONNXBackend("cpu"))
        register_backend("onnx-cuda", ONNXBackend("cuda"))
        register_backend("onnx-directml", ONNXBackend("directml"))
        register_backend("onnx-coreml", ONNXBackend("coreml"))
    except ImportError:
        pass
    if sys.platform == "darwin" and platform.machine() == "arm64":
        try:
            from .mlx_backend import MLXBackend

            register_backend("mlx", MLXBackend())
        except ImportError:
            pass


_init_backends()
