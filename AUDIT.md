# Application Security Audit Report: `laya-universal`

**Assessment Standard:** OWASP Top 10: 2025 Threat Model  
**Audit Target:** `laya-universal` (v0.1.1)  
**Lead Auditor:** Application Security Architect  
**Attribution:** Sole Author `abusuraihsakhri <abusuraihsakhri@gmail.com>`  
**Status:** All Identified Vulnerabilities Fully Remediated & Verified  

---

## 🛡️ Executive Summary

`laya-universal` is a high-speed inference engine designed for cross-platform execution of Laya typed decision models across Windows, Linux, and macOS. As an AI runtime embedded directly into agentic pipelines, local processing services, and automated classifiers, it processes untrusted user inputs, executes multi-threaded inference sessions, and interfaces directly with hardware accelerators (DirectML, CUDA, Metal).

A comprehensive Static Application Security Testing (SAST) and threat modeling audit was conducted against the **OWASP Top 10: 2025** framework. 5 distinct security vulnerabilities were identified, evaluated for exploitability, and remediated with automated regression unit tests.

### Vulnerability Summary Table

| ID | Vulnerability | OWASP Top 10: 2025 | Severity | Status |
|---|---|---|---|---|
| **VULN-01** | Windows Backslash Path Traversal in `subfolder` | A01:2025 - Broken Access Control | **High** | Remediated |
| **VULN-02** | Locale-Dependent Denial of Service on Windows | A10:2025 - Mishandling of Exceptional Conditions | **Medium** | Remediated |
| **VULN-03** | Concurrency Race & Device Policy Bypass in Encoder Session | A06:2025 - Insecure Design / A02:2025 | **Medium** | Remediated |
| **VULN-04** | Information Disclosure & Uncaught Stack Traces in CLI | A10:2025 - Mishandling of Exceptional Conditions | **Low** | Remediated |
| **VULN-05** | Serialization Crash on Non-Primitive State Payloads | A10:2025 - Mishandling of Exceptional Conditions | **Low** | Remediated |

---

## 📋 Detailed Vulnerability Findings & Remediations

### 🚨 Path Traversal via Windows Directory Separator in `subfolder` (Severity: High)

* **Location:** `laya_universal/agent.py -> resolve_model / Lines 74-95`
* **OWASP Category:** `A01:2025 - Broken Access Control` (CWE-22: Improper Limitation of a Pathname to a Restricted Directory)
* **Description:**  
  The subfolder validation logic previously used `PurePosixPath(subfolder)` to check whether `".." in part.parts`. Under Windows, `PurePosixPath` does not parse backslashes (`\`) as path separators. Supplying `subfolder=r"..\..\evil"` yielded `part.parts = ('..\\..\\evil',)`, which evaluated `".." in part.parts` as `False`. When subsequently appended via `path /= subfolder`, the resulting `WindowsPath` traversed outside the safe cache directory into the host file system.
* **Impact:**  
  Arbitrary directory traversal on the host file system. An attacker able to supply model subfolders via external configuration or API requests could read arbitrary files or force the application to probe unauthorized system directories.
* **Remediation Code:**

  ```python
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


  def resolve_model(model_id_or_path, *, token=None, subfolder=None, revision=None):
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
  ```

---

### 🚨 Locale-Dependent Denial of Service on Windows via Missing UTF-8 Encoding (Severity: Medium)

* **Location:** `laya_universal/agent.py -> Agent.__init__ / Line 131`, `laya_universal/tokenizer.py -> Tokenizer.__init__ / Line 18`, `laya_universal/backends/onnx_backend.py -> ONNXBackend.load_model / Line 103`, `laya_universal/backends/mlx_backend.py -> MLXBackend.load_model / Line 45`
* **OWASP Category:** `A10:2025 - Mishandling of Exceptional Conditions` (CWE-754: Improper Check for Unusual or Exceptional Conditions)
* **Description:**  
  `Path.read_text()` was called without explicitly specifying `encoding="utf-8"`. On Windows platforms, Python defaults to `locale.getpreferredencoding(False)` (frequently `cp1252`). When checkpoints, tokenizer configuration files, or model cards include non-ASCII characters (e.g. author names with diacritics, multilingual tokens, or non-Latin vocabulary metadata), `Path.read_text()` throws an unhandled `UnicodeDecodeError`.
* **Impact:**  
  Denial of Service (DoS) and application crash when loading multilingual or non-ASCII checkpoints on Windows environments.
* **Remediation Code:**

  ```python
  # Explicitly specify utf-8 encoding across all text file reads:
  self.cfg = json.loads((self.model_dir / "rl_agent_config.json").read_text(encoding="utf-8"))
  self.encoder_cfg = json.loads((self.model_dir / "encoder" / "config.json").read_text(encoding="utf-8"))
  config = json.loads((path / "tokenizer_config.json").read_text(encoding="utf-8"))
  ```

---

### 🚨 Concurrency Race Condition & Execution Provider Isolation Bypass in `ONNXBackend._encoder_session` (Severity: Medium)

* **Location:** `laya_universal/backends/onnx_backend.py -> _encoder_session / Lines 165-188`
* **OWASP Category:** `A06:2025 - Insecure Design` / `A02:2025 - Security Misconfiguration` (CWE-362: Concurrent Execution using Shared Resource with Improper Synchronization)
* **Description:**  
  In multi-threaded servers or async worker pools, multiple requests invoking `predict_shortlist` concurrently on the same `Agent` instance accessed `handle.metadata["encoder_session"]` without synchronization. Furthermore, `_encoder_session` instantiated its session using default `self._providers()` rather than propagating the user-configured `device` constraint (e.g. `device="cpu"`), causing encoder embedding sessions to bypass hardware isolation constraints and attempt GPU allocation.
* **Impact:**  
  Potential race conditions during ONNX session initialization, thread-safety violations, and unintended GPU memory allocation contrary to explicit resource confinement.
* **Remediation Code:**

  ```python
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
  ```

---

### 🚨 Information Disclosure & Uncaught Stack Traces in CLI (Severity: Low)

* **Location:** `laya_universal/cli.py -> _run_predict / Lines 81-105`
* **OWASP Category:** `A10:2025 - Mishandling of Exceptional Conditions` (CWE-209: Generation of Error Message Containing Sensitive Information)
* **Description:**  
  The CLI previously opened user-supplied `--state-file` and `--questions` paths directly without catching filesystem exceptions (`FileNotFoundError`, `PermissionError`, `json.JSONDecodeError`). Any malformed input printed raw Python tracebacks containing host path structures and user directory layouts to `stderr`.
* **Impact:**  
  Host information disclosure in production pipelines and automated logging systems.
* **Remediation Code:**

  ```python
  def _run_predict(args):
      import laya_universal as laya

      if args.state:
          state = args.state
      elif args.state_file:
          try:
              with open(args.state_file, encoding="utf-8") as f:
                  state = json.load(f)
          except (FileNotFoundError, json.JSONDecodeError, PermissionError) as e:
              print(f"ERROR: Could not read state file {args.state_file!r}: {e}", file=sys.stderr)
              sys.exit(1)
      else:
          print("ERROR: Provide --state or --state-file", file=sys.stderr)
          sys.exit(1)

      try:
          with open(args.questions, encoding="utf-8") as f:
              questions = json.load(f)
      except (FileNotFoundError, json.JSONDecodeError, PermissionError) as e:
          print(f"ERROR: Could not read questions file {args.questions!r}: {e}", file=sys.stderr)
          sys.exit(1)

      try:
          agent = laya.load(args.model, dtype=args.dtype, device=args.device, backend=args.backend)
      except Exception as e:
          print(f"ERROR: Failed to load model: {e}", file=sys.stderr)
          sys.exit(1)
  ```

---

### 🚨 Unhandled Serialization Failures on Non-Primitive State Inputs (Severity: Low)

* **Location:** `laya_universal/common.py -> serialize_state / Lines 16-20`
* **OWASP Category:** `A10:2025 - Mishandling of Exceptional Conditions`
* **Description:**  
  `serialize_state` invoked `json.dumps(state, ensure_ascii=False)`. If an application passed structured objects containing standard Python data types (e.g. `datetime`, `UUID`, `Decimal`), `json.dumps` threw a fatal `TypeError`, halting inference.
* **Impact:**  
  Pipeline disruption and unhandled exceptions on standard structured application states.
* **Remediation Code:**

  ```python
  def serialize_state(state: Union[str, dict, list]) -> str:
      if isinstance(state, str):
          return state
      return json.dumps(state, ensure_ascii=False, default=str)
  ```

---

## 🔍 Verification & Test Proofs

All remediations have been validated against our offline test suite:

```bash
$ python -m pytest
============================= 53 passed in 0.66s ==============================

$ python -m ruff check laya_universal tests
All checks passed!
```

Dedicated regression test cases in `tests/test_agent.py` verify that:
1. Windows-style parent directory traversal (`..\escape`, `..\..\secret`) is strictly rejected.
2. Windows drive letter prefixes (`C:\Windows`) and UNC network paths (`\\evil-server\share`) are strictly rejected.
3. Subfolder traversal from local valid checkpoint directories is strictly blocked.
4. Non-primitive states serialize gracefully without exceptions.
