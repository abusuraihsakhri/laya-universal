"""Public inference runtime — backend-agnostic. Prompt and result formats follow upstream Laya.

Runs on Windows, Linux and macOS. The backend is chosen from both the machine
and the checkpoint format: the ONNX backends read exported .onnx models, and the
MLX backend (Apple Silicon) reads model.safetensors directly. `dtype` and
`compile` are honored by the MLX backend; ONNX precision is fixed at export
time and its graph optimizations are always on.
"""

import json
import math
import warnings
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download

from .backends import ONNX_WEIGHT_FILES, ModelHandle, choose_backend
from .common import (
    QTYPES,
    TEMP_MAX,
    TEMP_MIN,
    build_sequence,
    clamp_temperature,
    collate_items,
    confidence_from_probs,
    render_options,
    temp_bucket,
)
from .prepared import PrefixCache
from .tokenizer import Tokenizer

_DTYPES = ("float32", "float16", "bfloat16")
_DEVICES = (None, "cpu", "gpu", "metal", "cuda")


def _normalize_subfolder(subfolder: str | None) -> str | None:
    """Validate and normalize subfolder path to prevent directory traversal attacks."""
    if subfolder is None:
        return None
    sub_str = str(subfolder).strip()
    if not sub_str:
        return None
    # Block Windows drive letters (C:) and UNC network paths (\\server\share)
    if ":" in sub_str or sub_str.startswith(("\\\\", "//")):
        raise ValueError("subfolder must be a relative path inside the model repository")
    # Normalize path separators to forward slashes for cross-platform safety
    sub_norm = sub_str.replace("\\", "/")
    parts = [p for p in sub_norm.split("/") if p]
    if sub_norm.startswith("/") or ".." in parts or "." in parts:
        raise ValueError("subfolder must be a relative path inside the model repository")
    return "/".join(parts)


def _looks_like_local_path(value: str) -> bool:
    """Windows-aware local-path test: absolute paths and ./ ../ .\\ ..\\ ~ prefixes."""
    return Path(value).is_absolute() or value.startswith(("./", "../", ".\\", "..\\", "~")) or (len(value) >= 2 and value[1] == ":" and value[0].isalpha())


def _hub_file_list(repo_id, *, token=None, revision=None):
    """Best-effort file listing of a Hub repo, so we only download weights the
    active backend can actually read. Returns None when the probe fails (offline,
    private repo without token, ...); the caller then downloads a superset."""
    try:
        from huggingface_hub import HfApi

        return HfApi().list_repo_files(repo_id, token=token, revision=revision)
    except Exception:
        return None


def _download_patterns(files, subfolder):
    clean_sub = _normalize_subfolder(subfolder)
    prefix = clean_sub + "/" if clean_sub else ""
    patterns = [
        prefix + "rl_agent_config.json",
        prefix + "encoder/config.json",
        prefix + "tokenizer/*",
    ]
    if files is None:
        patterns += [prefix + "model.safetensors"] + [prefix + n for n in ONNX_WEIGHT_FILES]
        return patterns
    for name in ("model.safetensors", *ONNX_WEIGHT_FILES):
        if prefix + name in files:
            patterns.append(prefix + name)
    if len(patterns) == 3:  # no recognizable weights; let snapshot_download explain
        patterns.append(prefix + "model.safetensors")
    return patterns


def resolve_model(model_id_or_path, *, token=None, subfolder=None, revision=None):
    """Locate (downloading if needed) a directory holding a Laya checkpoint."""
    clean_subfolder = _normalize_subfolder(subfolder)
    path = Path(model_id_or_path).expanduser()
    if not path.exists():
        value = str(model_id_or_path)
        if isinstance(model_id_or_path, Path) or _looks_like_local_path(value):
            raise FileNotFoundError(f"Local model directory does not exist: {value}")
        files = _hub_file_list(value, token=token, revision=revision)
        path = Path(
            snapshot_download(
                value,
                token=token,
                revision=revision,
                allow_patterns=_download_patterns(files, clean_subfolder),
            )
        )
    if clean_subfolder:
        base_resolved = path.resolve()
        target_path = (path / clean_subfolder).resolve()
        try:
            target_path.relative_to(base_resolved)
        except ValueError:
            raise ValueError(f"Path traversal detected in subfolder: {subfolder}")
        path = target_path
    for name in ("rl_agent_config.json", "encoder/config.json", "tokenizer/tokenizer.json"):
        if not (path / name).is_file():
            raise FileNotFoundError(f"Not a complete Laya checkpoint: {path / name} is missing")
    return path


class Agent:
    def __init__(
        self,
        model_id_or_path="convaiinnovations/laya",
        device=None,
        token=None,
        subfolder=None,
        *,
        dtype="float16",
        revision=None,
        batch_size=16,
        compile=False,
        pad_to_multiple=None,
        cache_prompts=False,
        backend=None,
    ):
        if dtype not in _DTYPES:
            raise ValueError(f"dtype must be one of {list(_DTYPES)}")
        if device not in _DEVICES:
            raise ValueError("device must be 'cpu', 'gpu', 'metal', 'cuda', or None")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if pad_to_multiple is not None and (
            not isinstance(pad_to_multiple, int)
            or isinstance(pad_to_multiple, bool)
            or pad_to_multiple < 1
        ):
            raise ValueError("pad_to_multiple must be a positive integer or None")
        self.dtype = dtype
        self.batch_size = batch_size
        self.pad_to_multiple = pad_to_multiple
        self._prefix_cache = PrefixCache() if cache_prompts else None
        self.model_id = str(model_id_or_path)
        self.revision = revision
        self.model_dir = resolve_model(
            model_id_or_path, token=token, subfolder=subfolder, revision=revision
        )

        self.cfg = json.loads((self.model_dir / "rl_agent_config.json").read_text(encoding="utf-8"))
        self.encoder_cfg = json.loads((self.model_dir / "encoder" / "config.json").read_text(encoding="utf-8"))
        if "encoder" not in self.cfg or "head_layers" not in self.cfg:
            raise ValueError("Laya config must specify encoder and head_layers")
        max_len = self.cfg.get("max_len", 512)
        head_max_len = self.cfg.get("head_max_len", 192)
        enc_max_pos = self.encoder_cfg.get("max_position_embeddings", 8192)
        if not 4 < head_max_len < max_len <= enc_max_pos:
            raise ValueError("Expected 4 < head_max_len < max_len <= max_position_embeddings")

        # Calibration temperatures: keep what the checkpoint shipped for
        # inspection, but only ever apply clamped values (see clamp_temperature).
        self.temperature_raw = self.cfg.get("temperature", [1.0, 1.0, 1.0])
        self.temperature_by_options_raw = self.cfg.get("temperature_by_options", {})
        if len(self.temperature_raw) != 3 or any(
            not math.isfinite(float(t)) or float(t) <= 0
            for t in [*self.temperature_raw, *self.temperature_by_options_raw.values()]
        ):
            raise ValueError("Calibration temperatures must be finite and positive")
        self.temperature = [clamp_temperature(t) for t in self.temperature_raw]
        self.temperature_by_options = {
            k: clamp_temperature(v) for k, v in self.temperature_by_options_raw.items()
        }
        rejected = [
            "%s=%.4g" % (k, float(v))
            for k, v in self.temperature_by_options_raw.items()
            if clamp_temperature(v) != float(v)
        ]
        rejected += [
            "temperature[%d]=%.4g" % (i, float(t))
            for i, t in enumerate(self.temperature_raw)
            if clamp_temperature(t) != float(t)
        ]
        if rejected:
            warnings.warn(
                "laya-universal: this checkpoint ships temperatures outside [%g, %g] which "
                "would distort confidence; clamping %s. Treat confidence from the affected "
                "buckets as uncalibrated." % (TEMP_MIN, TEMP_MAX, ", ".join(rejected)),
                RuntimeWarning,
                stacklevel=2,
            )

        self.tok = Tokenizer(self.model_dir / "tokenizer")

        # The backend must match the weights on disk; `backend=` forces a choice
        # (accepts aliases: onnx, dml, cuda, coreml, cpu, or a registered name).
        self._backend = choose_backend(str(self.model_dir), backend)
        self._handle: ModelHandle = self._backend.load_model(
            str(self.model_dir), dtype=dtype, device=device, compile=compile
        )

    @property
    def backend_name(self) -> str:
        return self._backend.name

    @staticmethod
    def _to_internal(qdef):
        if not isinstance(qdef, dict):
            raise ValueError("Each question must be a dictionary")
        kind = qdef.get("type")
        if kind not in QTYPES:
            raise ValueError(f"Unknown question type {kind!r}; expected choice, score, or noul")
        if "instructions" not in qdef:
            raise ValueError("Question is missing instructions")
        criteria = qdef.get("criteria")
        if kind == "choice":
            if isinstance(criteria, list):
                if not all(isinstance(c, str) for c in criteria):
                    raise ValueError("Choice labels must be strings")
                if len(set(criteria)) != len(criteria):
                    raise ValueError("Choice labels must be unique")
                criteria = dict.fromkeys(criteria)
            if not isinstance(criteria, dict) or not criteria:
                raise ValueError("Choice criteria must be a nonempty dictionary or list")
            if not all(isinstance(k, str) for k in criteria):
                raise ValueError("Choice labels must be strings")
        elif kind == "score":
            if not isinstance(criteria, list) or not criteria:
                raise ValueError("Score criteria must be a nonempty list")
        elif criteria is not None and not isinstance(criteria, dict):
            raise ValueError("Noul criteria must be a dictionary with false/true descriptions")
        instructions = qdef["instructions"]
        if not isinstance(instructions, str):
            instructions = json.dumps(instructions)
        return {"t": kind, "ins": instructions, "crit": criteria}

    def prepare(self, state, questions):
        """Construct upstream-compatible CPU inputs, useful for parity and profiling."""
        if self._prefix_cache is not None:
            return self._prefix_cache.prepare(self, state, questions)
        if not isinstance(questions, dict):
            raise ValueError("questions must be a dictionary keyed by question id")
        items, internal = [], []
        for qid, definition in questions.items():
            q = self._to_internal(definition)
            ids, markers = build_sequence(
                self.tok, state, q, self.cfg.get("max_len", 512), self.cfg.get("head_max_len", 192)
            )
            if len(markers) != len(render_options(q)):
                raise ValueError(f"Question {qid!r} has too many options for the token budget")
            items.append({"ids": ids, "markers": markers, "qtype": QTYPES[q["t"]]})
            internal.append(q)
        return items, internal

    def forward(self, batch):
        """Run one prepared batch through the active backend."""
        return self._backend.forward(
            self._handle,
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            marker_pos=batch["marker_pos"],
            marker_mask=batch["marker_mask"],
            qtype=batch["qtype"],
        )

    def system_one(self, state, questions):
        items, internal = self.prepare(state, questions)
        answers = {}
        question_ids = list(questions)
        for start in range(0, len(items), self.batch_size):
            chunk = items[start : start + self.batch_size]
            batch = collate_items(
                chunk,
                self.tok.pad_token_id,
                pad_to_multiple=self.pad_to_multiple,
                max_length=self.cfg.get("max_len", 512),
            )
            logits, act = self.forward(batch)
            if not np.isfinite(logits).all() or not np.isfinite(act).all():
                raise FloatingPointError("Non-finite model outputs; retry with dtype='float32'")
            act = np.exp(act - act.max(axis=-1, keepdims=True))
            act /= act.sum(axis=-1, keepdims=True)
            for row, item in enumerate(chunk):
                qid, q = question_ids[start + row], internal[start + row]
                k, qt = len(item["markers"]), item["qtype"]
                scale = self.temperature_by_options.get(temp_bucket(qt, k), self.temperature[qt])
                z = logits[row, :k] / scale
                p = np.exp(z - z.max())
                p /= p.sum()
                answer = {
                    "type": q["t"],
                    "confidence": round(confidence_from_probs(p, k), 4),
                    "action": {"act_probability": round(float(act[row, 0]), 4)},
                }
                if q["t"] == "choice":
                    labels = list(q["crit"])
                    answer.update(
                        choice=labels[int(p.argmax())],
                        probabilities={label: round(float(v), 4) for label, v in zip(labels, p)},
                    )
                elif q["t"] == "score":
                    answer.update(
                        score=round(float((np.arange(k) * p).sum()), 4),
                        legend={str(i): value for i, value in enumerate(q["crit"])},
                        probabilities={str(i): round(float(v), 4) for i, v in enumerate(p)},
                    )
                else:
                    answer.update(
                        noul=round(float(p[1]), 4),
                        confidence=round(max(float(p[1]), 1.0 - float(p[1])), 4),
                    )
                answers[qid] = answer
        return {
            "model": "laya-rl-agent",
            "answers": answers,
            "usage": {"input_tokens": sum(len(item["ids"]) for item in items), "output_tokens": 0},
        }

    predict = system_one


RLAgent = Agent


def load(
    model_id_or_path="convaiinnovations/laya", device=None, token=None, subfolder=None, **kwargs
):
    return Agent(model_id_or_path, device=device, token=token, subfolder=subfolder, **kwargs)
