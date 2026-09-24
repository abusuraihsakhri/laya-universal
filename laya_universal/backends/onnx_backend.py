"""ONNX Runtime inference backend — cross-platform (Windows/Linux/macOS).

Execution providers:
- CUDAExecutionProvider: NVIDIA GPUs (onnxruntime-gpu)
- DmlExecutionProvider: DirectML, any GPU on Windows (onnxruntime-directml)
- CoreMLExecutionProvider: Apple (onnxruntime built with CoreML)
- CPUExecutionProvider: universal fallback

Loads a directory containing model.onnx (or model_fp32.onnx / laya.onnx),
rl_agent_config.json, encoder/config.json and tokenizer/. An optional
encoder.onnx (written by `laya-universal convert`) enables mean-pooled
embeddings for `predict_shortlist` on this backend.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

import numpy as np

from . import ONNX_WEIGHT_FILES
from .base import ModelHandle

log = logging.getLogger(__name__)

_PROVIDER_MAP = {
    "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "directml": ["DmlExecutionProvider", "CPUExecutionProvider"],
    "coreml": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
    "cpu": ["CPUExecutionProvider"],
}


class ONNXBackend:
    """ONNX Runtime inference backend with provider auto-selection."""

    def __init__(self, mode: str = "cpu"):
        if mode not in _PROVIDER_MAP:
            raise ValueError(f"Unknown ONNX mode {mode!r}; expected one of {list(_PROVIDER_MAP)}")
        self._mode = mode

    @property
    def name(self) -> str:
        return f"onnx-{self._mode}"

    def available(self) -> bool:
        try:
            import onnxruntime as ort
        except ImportError:
            return False
        return _PROVIDER_MAP[self._mode][0] in ort.get_available_providers()

    def _providers(self, device=None) -> list[str]:
        """Provider list for this mode; device='cpu' forces the CPU provider."""
        import onnxruntime as ort

        if device == "cpu":
            return ["CPUExecutionProvider"]
        target = _PROVIDER_MAP[self._mode]
        available = set(ort.get_available_providers())
        return [p for p in target if p in available] or ["CPUExecutionProvider"]

    def _find_onnx(self, path: Path) -> Path | None:
        for name in ONNX_WEIGHT_FILES:
            if (path / name).is_file():
                return path / name
        return None

    def load_model(
        self,
        model_path: str,
        dtype: str = "float16",
        device: str | None = None,
        compile: bool = False,
    ) -> ModelHandle:
        """Load an exported ONNX checkpoint directory.

        `dtype` and `compile` are accepted for API parity with the MLX backend:
        ONNX precision is fixed at export time, and graph optimizations are
        always enabled.
        """
        import onnxruntime as ort

        path = Path(model_path)
        onnx_file = self._find_onnx(path)
        if onnx_file is None:
            if (path / "model.safetensors").is_file():
                raise RuntimeError(
                    f"{path} contains model.safetensors, which the ONNX backend cannot "
                    "read. Export an ONNX copy once: pip install 'laya-universal[export]' "
                    "and run laya-universal convert --model <checkpoint> --output <dir>, "
                    "then load the export directory."
                )
            raise FileNotFoundError(
                f"No ONNX model in {path}; expected one of {'/'.join(ONNX_WEIGHT_FILES)}"
            )

        cfg_path = path / "rl_agent_config.json"
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Missing {cfg_path}")
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

        enc_cfg = {}
        enc_cfg_path = path / "encoder" / "config.json"
        if enc_cfg_path.is_file():
            enc_cfg = json.loads(enc_cfg_path.read_text(encoding="utf-8"))

        providers = self._providers(device)
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        try:
            session = ort.InferenceSession(str(onnx_file), sess_options=opts, providers=providers)
        except Exception:
            if providers == ["CPUExecutionProvider"]:
                raise
            log.warning("%s failed to initialize; retrying with CPUExecutionProvider", providers[0])
            session = ort.InferenceSession(
                str(onnx_file), sess_options=opts, providers=["CPUExecutionProvider"]
            )

        names = [o.name for o in session.get_outputs()]
        if "logits" in names and "act_logits" in names:
            run_names = ["logits", "act_logits"]
        else:
            run_names = names[:2]
        if len(run_names) < 2:
            raise ValueError(f"ONNX model must expose logits and act_logits; found outputs {names}")

        encoder_file = path / "encoder.onnx"
        log.info("ONNX session created with providers: %s", session.get_providers())
        return ModelHandle(
            backend_name=self.name,
            model_dir=str(path),
            metadata={
                "session": session,
                "run_names": run_names,
                "config": cfg,
                "encoder_config": enc_cfg,
                "onnx_path": str(onnx_file),
                "device": device,
                "configured_providers": providers,
                "providers": session.get_providers(),
                "encoder_path": str(encoder_file) if encoder_file.is_file() else None,
                "encoder_session": None,
                "encoder_lock": threading.Lock(),
                "supports_embed": encoder_file.is_file(),
            },
        )

    def forward(
        self,
        handle: ModelHandle,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        marker_pos: np.ndarray,
        marker_mask: np.ndarray,
        qtype: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        session = handle.metadata["session"]
        feed = {
            "input_ids": input_ids.astype(np.int64),
            "attention_mask": attention_mask.astype(np.int64),
            "marker_pos": marker_pos.astype(np.int64),
            "marker_mask": marker_mask.astype(bool),
            "qtype": qtype.astype(np.int64),
        }
        outs = session.run(handle.metadata["run_names"], feed)
        return outs[0].astype(np.float32), outs[1].astype(np.float32)

    def _encoder_session(self, handle: ModelHandle):
        meta = handle.metadata
        if meta.get("encoder_session") is None:
            lock = meta.setdefault("encoder_lock", threading.Lock())
            with lock:
                if meta.get("encoder_session") is None:
                    import onnxruntime as ort

                    opts = ort.SessionOptions()
                    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    providers = meta.get("configured_providers") or self._providers(meta.get("device"))
                    try:
                        meta["encoder_session"] = ort.InferenceSession(
                            meta["encoder_path"], sess_options=opts, providers=providers
                        )
                    except Exception:
                        if providers == ["CPUExecutionProvider"]:
                            raise
                        log.warning(
                            "Encoder %s failed to initialize; falling back to CPUExecutionProvider",
                            providers[0],
                        )
                        meta["encoder_session"] = ort.InferenceSession(
                            meta["encoder_path"], sess_options=opts, providers=["CPUExecutionProvider"]
                        )
        return meta["encoder_session"]

    def embed_mean_pool(
        self,
        handle: ModelHandle,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
    ) -> np.ndarray:
        if not handle.metadata.get("encoder_path"):
            raise NotImplementedError(
                "This ONNX checkpoint has no encoder.onnx, so the backend cannot mean-pool "
                "encoder states. Re-export with 'laya-universal convert' (it writes "
                "encoder.onnx), or pass a dedicated embed_fn to predict_shortlist."
            )
        session = self._encoder_session(handle)
        out_name = session.get_outputs()[0].name
        hidden = session.run(
            [out_name],
            {
                "input_ids": input_ids.astype(np.int64),
                "attention_mask": attention_mask.astype(np.int64),
            },
        )[0]
        mask = attention_mask.astype(np.float32)[:, :, None]
        pooled = (hidden.astype(np.float32) * mask).sum(axis=1) / np.maximum(mask.sum(axis=1), 1.0)
        return pooled.astype(np.float32)

    @property
    def has_embed(self) -> bool:
        # Per-checkpoint capability: see ModelHandle.metadata['supports_embed'].
        return False
