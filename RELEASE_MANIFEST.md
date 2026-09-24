# Time-T v2.1.0 Release Archive Manifest

- **Release Archive**: `time-t-2.1.0.tar.gz`
- **Version**: `2.1.0`
- **Archive Size**: `233200` bytes
- **SHA-256 Checksum**: `e0016f570e2f5ba55673f222028c8c3d00ca77134f512dcf497201ed8f812d8d`
- **Build Timestamp**: 2026-09-24T13:45:00Z
- **Target Git Branch**: `arena/01a0d0a9-time-t`

## Archive Verification Command

```bash
sha256sum time-t-2.1.0.tar.gz
```

Expected output:
```text
e0016f570e2f5ba55673f222028c8c3d00ca77134f512dcf497201ed8f812d8d  time-t-2.1.0.tar.gz
```

## Archive Contents
- Core compiler & runtime (`timet/`)
- CLI launcher executable with 0755 permissions (`bin/time-t`)
- Standard build system (`setup.py`, `pyproject.toml`)
- 5M Causal Transformer Language Model (`models/transformer_lm_5m.py`)
- 16 runnable examples (`examples/`)
- 39 test suites with 503 unit & differential tests (`tests/`)
- Documentation specifications & binding design decisions (`docs/`)
- Hardware validation plans & physical ARM64 verification logs (`hardware/`)
- Stability stress suite (`tools/stress_test.py`)
- Security & threat model assessments (`SECURITY.md`, `THREAT_MODEL.md`)
- Production readiness candidate declaration (`PRODUCTION_READINESS.md`)
- Clean install verification report (`CLEAN_INSTALL_REPORT.md`)
- Total independent audit resolution ledger (`Time-T-v2.1.0-TOTAL-INDEPENDENT-AUDIT.md`)
