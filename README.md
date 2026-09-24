# Laya Universal

**Cross-platform typed decision inference for Laya checkpoints — Windows, Linux, macOS.**

ONNX Runtime everywhere; MLX on Apple Silicon. At inference time the runtime needs
~50 MB of dependencies (numpy, tokenizers, onnxruntime) instead of the ~2 GB a
PyTorch stack costs.

## What runs where

| Checkpoint on disk | Apple Silicon | Windows / Linux / Intel macOS |
|---|---|---|
| `model.safetensors` (e.g. `aac6fef/laya-mlx`, `convaiinnovations/*`) | **MLX** (native Metal) | needs a one-time ONNX export (below) |
| Exported ONNX dir (`model.onnx`) | ONNX (CoreML/CPU) | **ONNX** (DirectML / CUDA / CPU) |

Backend selection is format-aware: it never picks a backend that cannot read the
weights on disk, and the error tells you what to do when no backend can.

## Quick start (Apple Silicon)

```bash
pip install "laya-universal[mlx]"
```

```python
import laya_universal as laya

agent = laya.load("aac6fef/laya-mlx")
result = agent.predict(
    "I was billed twice. Please refund the duplicate.",
    {"department": {"type": "choice", "instructions": "Who should handle this?",
                     "criteria": ["billing", "technical", "sales"]}},
)
print(result["answers"]["department"]["choice"])
```

## Quick start (Windows / Linux)

The published Laya checkpoints ship `model.safetensors`, which ONNX backends
cannot read, so export an ONNX copy once (torch CPU is fine for the export; the
runtime never needs it again):

```bash
pip install "laya-universal[export]"        # brings upstream laya + torch, once
laya-universal convert --model convaiinnovations/laya --output ./laya-onnx
laya-universal predict --model ./laya-onnx --state "I was billed twice." --questions questions.json
```

```python
import laya_universal as laya
agent = laya.load("./laya-onnx")
result = agent.predict("I was billed twice.", questions)
```

## Backends and GPU

| Platform | Backend | Install |
|---|---|---|
| Windows, any GPU (NVIDIA/AMD/Intel) | `onnx-directml` | `pip install onnxruntime-directml` |
| Windows/Linux, NVIDIA | `onnx-cuda` | `pip install onnxruntime-gpu` |
| macOS | `onnx-coreml` | included in onnxruntime |
| Any | `onnx-cpu` | `pip install onnxruntime` (default dep) |
| Apple Silicon | `mlx` | `pip install "laya-universal[mlx]"` |

The GPU builds ship their own `onnxruntime` module and replace the CPU build's
files, so prefer `pip uninstall onnxruntime` before switching between them.

Auto-selection prefers MLX on Apple Silicon, then CUDA, DirectML, CoreML, CPU.
Force one with an alias or registered name:

```python
agent = laya.load("./laya-onnx", backend="dml")      # or "onnx", "cuda", "coreml", "cpu", "mlx"
agent = laya.load("./laya-onnx", device="cpu")       # ONNX: forces the CPU provider
```

Check what this machine can run: `laya-universal info`.

## Shortlisting large choice sets

`predict_shortlist` needs encoder embeddings. The MLX backend mean-pools its own
encoder; ONNX backends use the `encoder.onnx` that `laya-universal convert`
writes alongside `model.onnx`. Without it, pass any dedicated `embed_fn`:

```python
from laya_universal import predict_shortlist, embed_fn_from_agent

embed = embed_fn_from_agent(agent)                    # MLX, or ONNX + encoder.onnx
result = predict_shortlist(agent, state, questions, embed, k=20)
```

## Router

Language/script routing works like laya-mlx (sub-millisecond detection, no model
load for routing alone):

```python
from laya_universal import Router

router = Router()
result = router.predict({"message": "Rechnung wurde doppelt belastet"}, questions)
print(result["routing"]["model"])   # multilingual
```

## Inherited model limitations

These come from the checkpoints themselves and apply to every runtime:

- The **base checkpoints are near chance zero-shot** on the typed-decisions
  benchmark (0.36 vs 0.32 random). The strong 0.766 figure comes from the
  fine-tuned `laya-typed-decisions` checkpoint. Treat Laya as a fast base to
  specialize, not a zero-shot decision engine.
- `noul` can follow its `false:`/`true:` option labels instead of the state,
  most strongly on the English checkpoint. Validate on your data.
- `action.act_probability` carries no usable signal (reads ~1.0 for almost
  everything). Gate on `confidence` instead.
- `laya-multilingual` has a position bias on `score` questions (rarely picks
  the first-listed level).
- Avoid boolean-word labels (`true`/`false`, `yes`/`no`) in `choice` questions.

## Status

- Backend detection verified on Windows (DirectML + CPU providers) and by unit
  tests everywhere.
- Unit tests cover prompt construction, collation, calibration clamping,
  language routing, backend selection and end-to-end `predict` against a fake
  backend — all offline, no checkpoint downloads.
- Real-checkpoint inference on Windows/Linux requires an exported ONNX model
  (see Quick start). Numerical parity between the ONNX export and the
  MLX/PyTorch originals has not yet been measured; validate on your own data
  before production use.

## License and attribution

Apache-2.0 — see LICENSE and NOTICE. Laya and its pretrained weights are by
Convai Innovations and upstream contributors; this is an independent
cross-platform port, not an official Convai Innovations release. Derived from
[NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) and
[mizorewww/laya-mlx](https://github.com/mizorewww/laya-mlx).
