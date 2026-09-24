# Laya Universal

### [Open the Documentation Site →](https://abusuraihsakhri.github.io/laya-universal/)

[![CI](https://github.com/abusuraihsakhri/laya-universal/actions/workflows/ci.yml/badge.svg)](https://github.com/abusuraihsakhri/laya-universal/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

Cross-platform inference utilities for Laya typed-decision checkpoints. The package provides a common Python API for ONNX Runtime backends on Windows, Linux, and macOS, plus an MLX backend for Apple Silicon.

## What it provides

- Typed decisions using `choice`, `score`, and `noul` question schemas.
- Format-aware backend selection for ONNX and MLX checkpoints.
- ONNX Runtime provider support for CPU, CUDA, DirectML, and CoreML when those providers are available in the installed runtime.
- Native MLX loading of `model.safetensors` on supported Apple Silicon systems.
- Hugging Face Hub checkpoint resolution with optional revisions and subfolders.
- Batched inference, bounded prompt-prefix caching, model routing, language/script routing, email preprocessing, and high-cardinality choice shortlisting.
- A CLI for backend inspection, prediction, and one-time ONNX export.

The source code is authoritative for supported behavior. The documentation site is static documentation; it does not execute the Python inference runtime in the browser.

## Installation

Clone the repository, then install exactly one inference runtime for the target environment:

```bash
git clone https://github.com/abusuraihsakhri/laya-universal.git
cd laya-universal
```

### CPU / standard ONNX Runtime

```bash
python -m pip install ".[cpu]"
```

### NVIDIA CUDA

```bash
python -m pip install ".[gpu-cuda]"
```

### Windows DirectML

```bash
python -m pip install ".[gpu-directml]"
```

### Apple Silicon MLX

```bash
python -m pip install ".[mlx]"
```

ONNX Runtime publishes separate CPU, CUDA, and DirectML distributions that should not be installed together in one environment. The project therefore keeps them as mutually exclusive optional dependencies instead of installing the CPU runtime unconditionally.

## Quick start

For an ONNX checkpoint directory:

```python
import laya_universal as laya

questions = {
    "department": {
        "type": "choice",
        "instructions": "Which team should handle this request?",
        "criteria": {
            "billing": "invoices, charges, and refunds",
            "technical": "bugs, outages, and integrations",
            "sales": "pricing, plans, and purchases",
        },
    },
    "urgent": {
        "type": "noul",
        "instructions": "Does the request communicate an urgent deadline?",
    },
}

agent = laya.load("./laya-onnx")
result = agent.predict("I was charged twice and need a refund.", questions)

print(result["answers"]["department"])
print(result["answers"]["urgent"])
```

A checkpoint directory must contain the Laya configuration and tokenizer files plus a compatible weight file. ONNX backends look for `model.onnx`, `model_fp32.onnx`, or `laya.onnx`; MLX loads `model.safetensors`.

Remote Hugging Face model IDs are supported. A safetensors-only checkpoint requires the MLX backend, or it must first be exported to ONNX on a machine with the upstream `laya` package and PyTorch installed.

## CLI

Inspect the backends visible in the current environment:

```bash
laya-universal info
```

Run a prediction from JSON questions:

```bash
laya-universal predict \
  --model ./laya-onnx \
  --state "I was charged twice and need a refund." \
  --questions questions.json
```

Export an upstream checkpoint to ONNX:

```bash
python -m pip install ".[export]"
laya-universal convert --model convaiinnovations/laya --output ./laya-onnx
```

The exporter writes the decision graph and, when the upstream model exposes its encoder, `encoder.onnx` for embedding-based shortlisting.

## Routing and presets

`Router` can select among English, multilingual, and typed-decision checkpoints using explicit model/task/language settings or the built-in script/language heuristic.

```python
router = laya.Router(auto_task_detection=True)
decision = router.route("Bitte prüfen Sie diese Rechnung.", laya.triage_questions())
print(decision)
```

Built-in question dictionaries are available through:

- `triage_questions()`
- `email_questions()`
- `guard_questions()`
- `moderation_questions()`
- `router_questions()`

These are question templates, not guarantees of model accuracy. Validate checkpoint behavior on data representative of the intended task before relying on predictions.

## Choice shortlisting

For large categorical option sets, `predict_shortlist` can reduce the choices before the final decision pass. ONNX checkpoints can use `embed_fn_from_agent` when an `encoder.onnx` graph is present; callers may also provide a separate embedding function.

```python
from laya_universal import embed_fn_from_agent, predict_shortlist

embed = embed_fn_from_agent(agent)
result = predict_shortlist(
    agent,
    "Database connection timed out",
    questions,
    embed_fn=embed,
    k=10,
)
```

Remote ONNX checkpoint downloads include `encoder.onnx` when that file exists in the repository.

## Data handling

Inference runs in the local Python process. Loading a remote Hugging Face model ID can contact the Hugging Face Hub to list and download checkpoint files; loading an existing local checkpoint does not require that model download step. Application state passed to `predict` is processed locally by the selected backend.

No server-side component is provided by this repository. GitHub Pages hosts static documentation only.

## Development

```bash
python -m pip install -e ".[cpu,dev]"
python -m pip check
python -m pip_audit
python -m ruff check laya_universal tests
python -m pytest -v
python -m build
```

CI runs linting, dependency consistency checks, dependency auditing, and the test suite on Windows and Ubuntu across Python 3.10, 3.11, and 3.12. It also verifies that the package builds successfully.

## Browser compatibility

The documentation site works as a static GitHub Pages site in modern browsers. The Python inference package itself is not currently a browser application. It depends on native Python inference runtimes and checkpoint assets, so this repository does not attempt to run the Python API through Pyodide or PyScript.

For a browser-native inference implementation, ONNX Runtime Web would require a separate JavaScript/WebAssembly integration and model compatibility testing; that is outside the current package.

## Security reporting

See [SECURITY.md](SECURITY.md) for responsible vulnerability reporting. Do not include credentials, tokens, private model artifacts, or sensitive production data in public issues.

## License and attribution

Licensed under the [Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for upstream attribution.

The package derives portions of its prompt/runtime behavior from [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya) and includes an MLX backend informed by [mizorewww/laya-mlx](https://github.com/mizorewww/laya-mlx).
