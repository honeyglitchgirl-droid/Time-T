# Time-T Threat Model & Security Posture (Phase O)

## Threat Vectors & Defenses

| Threat Vector | Severity | Mitigation in Time-T | Status |
| :--- | :--- | :--- | :--- |
| **Arbitrary Code Execution via Weights** | Critical | Avoids `pickle`; uses strictly parsed `.safetensors`, `.npz`, and custom `.ttck` zip containers with raw buffer loading. | MITIGATED |
| **Path Traversal via Modules / Checkpoints** | High | Module paths are resolved relative to the importing file and sanitized. Archive extraction checks member filenames. | MITIGATED |
| **Denial of Service via Unbounded Recursion** | Medium | Recursion depth counter in `Interpreter` and `IRExecutor` (default 1000 frames) raises `E0509` cleanly. | MITIGATED |
| **Denial of Service via Zero Division** | Low | Explicit zero checks in arithmetic evaluation (`E0507`, `E0508`) prevent uncaught process termination. | MITIGATED |
| **Shell Injection in Native Codegen** | Critical | Host compiler subprocesses are invoked directly with argument lists; no shell expansion is used. | MITIGATED |
| **Zip Bomb / Malformed Archive Deserialization** | Medium | Checkpoint loader verifies header lengths, zip entry counts, and rejects mismatched tensor shapes. | MITIGATED |
