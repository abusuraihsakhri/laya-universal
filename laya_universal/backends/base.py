"""Abstract backend interface for laya-universal inference engines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass
class ModelHandle:
    """Opaque handle returned by a backend after loading a model."""
    backend_name: str
    model_dir: str
    metadata: dict[str, Any] | None = None


class InferenceBackend(Protocol):
    """Protocol that every inference backend must satisfy."""

    @property
    def name(self) -> str:
        """Human-readable backend name, e.g. 'onnx-cpu', 'mlx'."""
        ...

    def available(self) -> bool:
        """Return True if this backend can run on the current machine."""
        ...

    def load_model(
        self,
        model_path: str,
        dtype: str = "float16",
        device: str | None = None,
        compile: bool = False,
    ) -> ModelHandle:
        """Load a model from `model_path` and return a handle."""
        ...

    def forward(
        self,
        handle: ModelHandle,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        marker_pos: np.ndarray,
        marker_mask: np.ndarray,
        qtype: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run a forward pass. Returns (logits, act_logits) as float32 numpy arrays."""
        ...

    def embed_mean_pool(
        self,
        handle: ModelHandle,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
    ) -> np.ndarray:
        """Mean-pool the encoder hidden states. Returns (batch, hidden_dim) float32."""
        ...

    @property
    def has_embed(self) -> bool:
        """Whether this backend supports mean-pool embedding for shortlisting."""
        ...
