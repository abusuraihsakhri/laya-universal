# Laya Universal

[![CI](https://github.com/abusuraihsakhri/laya-universal/actions/workflows/ci.yml/badge.svg)](https://github.com/abusuraihsakhri/laya-universal/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Platforms](https://img.shields.io/badge/platforms-Windows%20%7C%20Linux%20%7C%20macOS-informational.svg)](README.md)
[![Hardware](https://img.shields.io/badge/acceleration-DirectML%20%7C%20CUDA%20%7C%20Metal%20%7C%20CPU-orange.svg)](README.md)
[![Security Audited](https://img.shields.io/badge/security-OWASP%20Top%2010%20(2025)%20Audited-success.svg)](AUDIT.md)

**Cross-platform, low-latency typed decision inference for Laya checkpoints.**

Run fast bidirectional decision models on **Windows** (DirectML GPU), **Linux** (CUDA GPU), and **macOS** (MLX Metal & CoreML) with a lightweight **~50 MB** runtime footprint — **zero PyTorch dependency at inference time**.

---

## 📑 Table of Contents

- [The Ideation & Problem Space](#-the-ideation--problem-space)
- [Architecture & Data Flow](#-architecture--data-flow)
- [Performance & Runtime Matrix](#-performance--runtime-matrix)
- [Installation Guide](#-installation-guide)
- [Quick Start](#-quick-start)
  - [Windows & Linux (DirectML / CUDA / CPU)](#windows--linux-directml--cuda--cpu)
  - [Apple Silicon (Native MLX Metal)](#apple-silicon-native-mlx-metal)
  - [Command-Line Interface (CLI)](#command-line-interface-cli)
- [Comprehensive API Guide](#-comprehensive-api-guide)
  - [1. Typed Decisions (`choice`, `score`, `noul`)](#1-typed-decisions-choice-score-noul)
  - [2. Sub-Millisecond Language & Task Routing](#2-sub-millisecond-language--task-routing)
  - [3. High-Cardinality Choice Shortlisting](#3-high-cardinality-choice-shortlisting)
  - [4. Tokenized Prefix Caching](#4-tokenized-prefix-caching)
  - [5. Production Workflow Presets](#5-production-workflow-presets)
  - [6. Email Structuring & Sanitization](#6-email-structuring--sanitization)
- [Inherited Checkpoint Limitations](#-inherited-checkpoint-limitations)
- [Security Architecture & Audit](#-security-architecture--audit)
- [Project Documentation Site](#-project-documentation-site)
- [Contributing & Attribution](#-contributing--attribution)

---

## 💡 The Ideation & Problem Space

### Why Typed Decision Models?
Modern agentic workflows and automated pipelines often rely on generative Large Language Models (LLMs) to perform simple routing, safety gating, intent triage, and validation. This is fundamentally inefficient:
* **Latency:** Generative autoregressive decoding takes **300ms to 2,500ms** per decision.
* **Cost & Bloat:** Calling cloud LLM endpoints introduces network latency, privacy concerns, API bills, and rate limits.
* **Non-Determinism:** LLMs require extensive regex or grammar parsers to extract structured choices reliably.

**Laya typed decision models** replace generative token loops with single-pass bidirectional classification (7–15ms) across formal decision types:
* `choice`: Categorical routing with normalized probability distribution.
* `score`: Ordinal multi-level ranking with calibrated expectation values.
* `noul`: Calibrated epistemological confidence on binary propositions (grounded in evidence, not sycophancy).

### The Platform Gap
1. **Upstream Laya ([`NandhaKishorM/laya`](https://github.com/NandhaKishorM/laya)):** Fully featured and portable, but requires a heavy **PyTorch stack (~2 GB)**, resulting in slow container cold-starts and excessive memory overhead.
2. **Apple MLX Port ([`mizorewww/laya-mlx`](https://github.com/mizorewww/laya-mlx)):** Blazing fast (7–14ms) on Apple Silicon, but **strictly locked to macOS ARM64**. Windows and Linux developers are unable to run it.

### The Universal Solution: `laya-universal`
`laya-universal` bridges the ecosystem gap:
* **Universal Acceleration:** Auto-detects the hardware and runs via **ONNX Runtime** on Windows/Linux, with native **DirectML** support (accelerating **any** GPU: AMD, Intel, NVIDIA) and **CUDA** on Linux, plus native **MLX** on Apple Silicon.
* **Ultralight Footprint:** Requires only NumPy, Rust Tokenizers, and ONNX Runtime (~50 MB vs. 2,000 MB).
* **Format-Aware Backend Selection:** Inspects checkpoints on disk (`.onnx` vs. `.safetensors`) and chooses the optimal runnable engine automatically.
* **Drop-in Compatibility:** Mirrors the exact public API, prompt formatting, temperature calibration, and routing logic of upstream Laya.

---

## 🏗️ Architecture & Data Flow

```
                      ┌──────────────────────────────────────────────┐
                      │    Input State (String / JSON) + Questions   │
                      └──────────────────────┬───────────────────────┘
                                             │
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │         HuggingFace Rust Tokenizer           │
                      │  ([CLS] ins [SEP] [MASK] opt0 ... [SEP] st)  │
                      └──────────────────────┬───────────────────────┘
                                             │
                         ┌───────────────────┴───────────────────┐
                         │                                       │
                         ▼                                       ▼
             ┌───────────────────────┐               ┌───────────────────────┐
             │ PrefixCache (Optional)│               │  Collate & Pad Array  │
             │   Bounded LRU Cache   │               │   (Dynamically Batched│
             └───────────┬───────────┘               └───────────┬───────────┘
                         │                                       │
                         └───────────────────┬───────────────────┘
                                             │
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │    Format-Aware Backend Selector (Registry)  │
                      └──────┬───────────────────────┬───────────────┘
                             │                       │
              Windows / Linux / macOS         Apple Silicon Only
                             │                       │
                             ▼                       ▼
                  ┌──────────────────────┐┌──────────────────────┐
                  │ ONNX Runtime Engine  ││  Native MLX Engine   │
                  │ - DirectML (Windows) ││  - Metal Framework   │
                  │ - CUDA (Linux)       ││  - model.safetensors │
                  │ - CoreML / CPU       ││  - MX Array Streams  │
                  └──────────┬───────────┘└──────────┬───────────┘
                             │                       │
                             └───────────┬───────────┘
                                         │
                                         ▼
                      ┌──────────────────────────────────────────────┐
                      │  Logits & Entropy Confidence Post-Processing │
                      │  - Temperature Clamping ([0.5, 5.0])         │
                      │  - Normalized Shannon Entropy Confidence     │
                      └──────────────────────┬───────────────────────┘
                                             │
                                             ▼
                      ┌──────────────────────────────────────────────┐
                      │   Structured Typed Output: {answers, usage}  │
                      └──────────────────────────────────────────────┘
```

---

## ⚡ Performance & Runtime Matrix

| Platform | Recommended Backend | Hardware Acceleration | Typical Latency | Dependencies Required |
|---|---|---|---|---|
| **Windows 10/11** | `onnx-directml` | Any GPU (NVIDIA, AMD, Intel ARC/Iris) | **8–14 ms** | `onnxruntime-directml` |
| **Linux (x86_64)** | `onnx-cuda` | NVIDIA GPU (Tensor Cores) | **6–11 ms** | `onnxruntime-gpu` |
| **macOS (Apple Silicon)** | `mlx` | Apple Metal GPU (Unified Memory) | **7–13 ms** | `mlx` |
| **macOS (Intel / ARM)** | `onnx-coreml` / `onnx-cpu` | Apple Neural Engine / CPU | **12–25 ms** | `onnxruntime` |
| **Any Platform** | `onnx-cpu` | Vectorized CPU (AVX2 / AVX-512) | **18–35 ms** | `onnxruntime` |
| *Upstream PyTorch* | *Torch / Cuda* | *NVIDIA only* | *15–40 ms* | *torch (~2,000 MB)* |

---

## 📦 Installation Guide

### Standard Installation (CPU Fallback, Cross-Platform)
```bash
pip install laya-universal
```

### Windows GPU Acceleration (DirectML - AMD, NVIDIA, Intel)
```bash
pip install onnxruntime-directml
pip install laya-universal
```

### Linux GPU Acceleration (NVIDIA CUDA)
```bash
pip install onnxruntime-gpu
pip install laya-universal
```

### Apple Silicon Acceleration (Metal MLX)
```bash
pip install "laya-universal[mlx]"
```

### One-Time ONNX Exporter Dependencies
To convert upstream PyTorch checkpoints (`model.safetensors`) into `.onnx` directories:
```bash
pip install "laya-universal[export]"
```

---

## 🚀 Quick Start

### Windows & Linux (DirectML / CUDA / CPU)

The published Hugging Face checkpoints (`convaiinnovations/*`) ship `model.safetensors`. To run them with sub-15ms speeds on Windows or Linux, export an ONNX copy once:

```bash
# 1. Export the model once (CPU PyTorch is fine for export)
laya-universal convert --model convaiinnovations/laya --output ./laya-onnx

# 2. Run inference anywhere with just ONNX Runtime!
```

```python
import laya_universal as laya

# Load the exported model (automatically selects DirectML on Windows or CUDA on Linux)
agent = laya.load("./laya-onnx")
print(f"Active Backend: {agent.backend_name}")

state = "Customer order #84920 was charged twice on invoice. Requesting urgent refund."

questions = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this ticket?",
        "criteria": {
            "billing": "Invoices, double charges, refunds, payment errors",
            "technical": "Software bugs, crashes, API errors",
            "sales": "Upgrades, new licenses, contracts"
        }
    },
    "urgency": {
        "type": "score",
        "instructions": "Rate customer urgency.",
        "criteria": ["low priority inquiry", "normal support issue", "critical financial/escalation block"]
    },
    "needs_refund": {
        "type": "noul",
        "instructions": "The customer is explicitly requesting a return of funds.",
        "criteria": {
            "false": "No refund requested",
            "true": "Refund or billing adjustment requested"
        }
    }
}

result = agent.predict(state, questions)

# Access typed answers:
print("Department:", result["answers"]["department"]["choice"])
print("Confidence:", result["answers"]["department"]["confidence"])
print("Urgency Score (0-2):", result["answers"]["urgency"]["score"])
print("Needs Refund (probability):", result["answers"]["needs_refund"]["noul"])
```

### Apple Silicon (Native MLX Metal)

On Apple Silicon with macOS 13.5+ and Python 3.11+, checkpoints with `model.safetensors` run directly on Metal without conversion:

```python
import laya_universal as laya

agent = laya.load("aac6fef/laya-mlx")
result = agent.predict("I was billed twice.", questions)
print(result["answers"]["department"]["choice"])
```

### Command-Line Interface (CLI)

```bash
# Inspect available backends and hardware providers on this machine:
laya-universal info

# Export a PyTorch model to ONNX:
laya-universal convert --model convaiinnovations/laya --output ./laya-onnx

# Run prediction directly from terminal:
laya-universal predict --model ./laya-onnx --state "Card charged twice" --questions questions.json
```

---

## 🛠️ Comprehensive API Guide

### 1. Typed Decisions (`choice`, `score`, `noul`)

Laya enforces three distinct question schemas:

```python
questions = {
    # 1. CHOICE: Categorical single-label selection
    "intent": {
        "type": "choice",
        "instructions": "Identify the primary user intent.",
        "criteria": ["dispute", "inquiry", "cancellation"] # Or dict with descriptions
    },
    # 2. SCORE: Calibrated expectation value across discrete levels (0 to K-1)
    "sentiment": {
        "type": "score",
        "instructions": "Rate customer satisfaction level.",
        "criteria": ["angry", "neutral", "delighted"]
    },
    # 3. NOUL: Epistemically calibrated binary probability (P(true))
    "escalate": {
        "type": "noul",
        "instructions": "This ticket requires immediate human supervisor escalation."
    }
}
```

### 2. Sub-Millisecond Language & Task Routing

The `Router` determines which checkpoint is appropriate based on language, character script, or task signature in **<0.5ms without loading any model weights**:

```python
from laya_universal import Router

router = Router()

# Route English text:
decision = router.route("I need a billing refund")
print(decision.model)  # -> 'english'

# Route German or Chinese text:
decision = router.route("Rechnung wurde doppelt belastet")
print(decision.model)  # -> 'multilingual' (detected German Latin)

decision = router.route("发票被重复扣款，请处理")
print(decision.model)  # -> 'multilingual' (detected Han script)

# Execute prediction with automatic routing:
res = router.predict("发票被重复扣款", questions)
print("Routed to:", res["routing"]["model"])
```

### 3. High-Cardinality Choice Shortlisting

When dealing with large choice sets (e.g. 50–500 categories), Laya's maximum sequence length cannot fit all options. `predict_shortlist` performs an initial semantic pre-filter using the encoder's mean-pooled representation:

```python
from laya_universal import predict_shortlist, embed_fn_from_agent

agent = laya.load("./laya-onnx")
embed_fn = embed_fn_from_agent(agent)

huge_criteria = {f"cat_{i}": f"Description for category {i}" for i in range(100)}

questions = {
    "category": {
        "type": "choice",
        "instructions": "Select the correct category.",
        "criteria": huge_criteria
    }
}

# Reduces 100 choices down to the top-k most semantically relevant before inference:
result = predict_shortlist(agent, "My database connection timed out", questions, embed_fn=embed_fn, k=10)
print(result["answers"]["category"]["choice"])
print("Shortlist candidate count:", len(result["shortlist"]["category"]["labels"]))
```

### 4. Tokenized Prefix Caching

For high-throughput servers executing the same question structures repeatedly over varying inputs, enable `cache_prompts=True`:

```python
agent = laya.load("./laya-onnx", cache_prompts=True)

# The question prefix tokens are cached in an LRU buffer; only state tokens are encoded per request.
for user_message in incoming_stream:
    result = agent.predict(user_message, questions)
```

### 5. Production Workflow Presets

Laya ships built-in question dictionaries tuned for standard industry workflows:

```python
import laya_universal as laya

# Pre-packaged decision workflows:
triage = laya.triage_questions()        # intent, urgency, sentiment, escalation
guards = laya.guard_questions()         # prompt injection, jailbreak, harmful content
moderation = laya.moderation_questions()# toxic, hate, harassment, self-harm
router = laya.router_questions()        # agent routing targets
email = laya.email_questions()          # action required, sentiment, category
```

### 6. Email Structuring & Sanitization

Clean noisy email bodies, stripping corporate disclaimer boilerplate, forwarded headers, and signatures before tokenization:

```python
from laya_universal import email_state

state = email_state(
    subject="Urgent: Invoice 4029 Incorrect",
    body="""Hello team,
    
We were charged twice. Please review attached invoice.

Best regards,
Alice Smith
Senior VP Operations

CONFIDENTIALITY NOTICE: This transmission may contain confidential information...
""",
    sender="alice@example.com",
    clean=True
)

result = agent.predict(state, laya.email_questions())
```

---

## ⚠️ Inherited Checkpoint Limitations

These traits stem directly from the underlying pretrained checkpoints and apply to all Laya runtimes:

1. **Zero-Shot Base Performance:** Base Laya checkpoints (`convaiinnovations/laya`) achieve ~0.36 accuracy zero-shot on generalized tasks (near random chance ~0.32). The benchmark figure of **0.766** requires task-specific prompt structure or the fine-tuned `laya-typed-decisions` checkpoint. Treat Laya as a foundation to calibrate and align.
2. **Boolean Option Bias:** `noul` questions can exhibit anchor bias toward option literals (`true:` / `false:`). Avoid defining boolean questions where labels invert meaning.
3. **Action Probability:** `action.act_probability` reads ~1.0 on most checkpoints and carries minimal discriminating signal. **Always gate critical decisions on `confidence`**, which uses normalized Shannon entropy:
   $$\text{Confidence} = 1 - \frac{H(p)}{\ln(k)}$$
4. **Multilingual Score Position Bias:** The multilingual checkpoint rarely selects the first-listed level in ordinal `score` questions.

---

## 🔒 Security Architecture & Audit

`laya-universal` has been rigorously evaluated and hardened against the **OWASP Top 10: 2025** threat model:

* **Path Traversal Protection (CWE-22):** The model resolver rejects Windows backslash directory traversal (`..\..`), UNC network paths (`\\server\share`), drive letters, and escapes, guaranteeing that model loading stays bounded within approved directories.
* **Safe Encoding (CWE-754):** All JSON, tokenizer, and checkpoint files are read with explicit UTF-8 decoding, preventing Windows ANSI (`cp1252`) denial-of-service crashes on international text.
* **Thread-Safe Lazy Sessions (CWE-362):** Concurrent execution on multi-threaded web servers is protected by mutual exclusion locks during lazy ONNX session initialization.
* **Isolated Provider Constraints (CWE-665):** Explicit `device="cpu"` flags cascade down to sub-graphs and encoder sessions, preventing unintended GPU allocations.

Full details and verification proofs are documented in [**AUDIT.md**](AUDIT.md).

---

## 🌐 Project Documentation Site

An interactive documentation portal with architecture diagrams, real-time decision simulators, and platform selection tools is available in the [`docs/`](docs/) directory and hosted on GitHub Pages:

👉 **[https://abusuraihsakhri.github.io/laya-universal/](https://abusuraihsakhri.github.io/laya-universal/)**

---

## 📜 Contributing & Attribution

Distributed under the **Apache-2.0 License**. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for complete licensing terms.

* Core Laya architecture and pretrained weights by **Convai Innovations** and upstream contributors ([`NandhaKishorM/laya`](https://github.com/NandhaKishorM/laya)).
* MLX Metal port based on work by **[`mizorewww/laya-mlx`](https://github.com/mizorewww/laya-mlx)**.
* `laya-universal` is an independent cross-platform engineering port maintained by **[abusuraihsakhri](https://github.com/abusuraihsakhri)**.
