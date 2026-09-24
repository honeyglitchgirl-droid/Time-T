# Time-T v2.1.0 Release Archive Manifest

- **Release Archive**: `time-t-2.1.0.tar.gz`
- **Version**: `2.1.0`
- **Archive Size**: `229050` bytes
- **SHA-256 Checksum**: `dfe0b75045d1f8ec4790ad11682af6a81868c336807420c4ac2166c4ebd11d59`
- **Build Timestamp**: 2026-09-24T13:20:00Z
- **Target Git Branch**: `arena/01a0d0a9-time-t`

## Archive Verification Command

```bash
sha256sum time-t-2.1.0.tar.gz
```

Expected output:
```text
dfe0b75045d1f8ec4790ad11682af6a81868c336807420c4ac2166c4ebd11d59  time-t-2.1.0.tar.gz
```

## Archive Contents
- Core compiler & runtime (`timet/`)
- CLI launcher executable with 0755 permissions (`bin/time-t`)
- Standard build system (`setup.py`, `pyproject.toml`)
- 16 runnable examples (`examples/`)
- 38 test suites with 500 unit & differential tests (`tests/`)
- Documentation specifications & binding design decisions (`docs/`)
- Hardware validation plans & physical ARM64 verification logs (`hardware/`)
- Stability stress suite (`tools/stress_test.py`)
- Security & threat model assessments (`SECURITY.md`, `THREAT_MODEL.md`)
- Production readiness candidate declaration (`PRODUCTION_READINESS.md`)
- Clean install verification report (`CLEAN_INSTALL_REPORT.md`)
- Total independent audit resolution ledger (`Time-T-v2.1.0-TOTAL-INDEPENDENT-AUDIT.md`)
