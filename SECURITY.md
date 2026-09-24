# Time-T Security Policy and Specifications (Phase O)

## 1. Trust Boundaries

Time-T defines explicit boundaries to defend against malicious input, corrupted weights, and arbitrary execution:

1. **Source Code Boundary**:
   - Time-T parses source through its own deterministic lexer and recursive-descent parser.
   - Stack depth is guarded (`E0509`), preventing host stack exhaustion.
   - Syntax and typing violations produce location-tagged diagnostics (`E0100`–`E0301`) without unhandled crashes.

2. **Model Checkpoint & Serialization Boundary**:
   - Checkpoint files (`.ttck`, `.safetensors`, `.npz`, `.json`) are strictly inspected before deserialization.
   - Python `pickle` is explicitly forbidden (`allow_pickle=False` in `np.load`).
   - Binary checkpoints check `manifest.json` against archive contents (`E0648`), rejecting untracked or unexpected entries.
   - Truncated files and corrupt headers raise clean diagnostics (`E0645`, `E0646`) rather than undefined behavior.

3. **Native Compilation & JIT Execution**:
   - All C source generation in `native.py` and `jit_kernels.py` uses sanitized alphanumeric identifiers and escapes string literals.
   - Invocation of system compilers (`gcc`, `clang`) uses direct `execve`-style argument lists without shell interpretation (`shell=False`).
   - Generated shared libraries are loaded from secure temporary directory paths.

## 2. Vulnerability Reporting

To report a vulnerability or unexpected crash, open an issue or security advisory referencing the exact diagnostic code and reproducing source.
