# Time-T v2.1.0 Release Archive Manifest

- **Release Archive**: `time-t-2.1.0.tar.gz`
- **Version**: `2.1.0`
- **Archive Size**: `218979` bytes
- **SHA-256 Checksum**: `0cc4a28a2df049e4002ad10d974409cc3911f11864f91390f7122aa6a88c0d06`
- **Build Timestamp**: 2026-09-24T12:15:00Z
- **Target Git Branch**: `arena/01a0d0a9-time-t`

## Archive Verification Command

```bash
sha256sum time-t-2.1.0.tar.gz
```

Expected output:
```text
0cc4a28a2df049e4002ad10d974409cc3911f11864f91390f7122aa6a88c0d06  time-t-2.1.0.tar.gz
```

## Archive Contents
- Core compiler & runtime (`timet/`)
- CLI launcher executable with 0755 permissions (`bin/time-t`)
- 16 runnable examples (`examples/`)
- 37 test suites with 497 unit & differential tests (`tests/`)
- Documentation specifications & binding design decisions (`docs/`)
- Hardware validation plans (`hardware/`)
- Stability stress suite (`tools/stress_test.py`)
- Security & threat model assessments (`SECURITY.md`, `THREAT_MODEL.md`)
- Production readiness candidate declaration (`PRODUCTION_READINESS.md`)
- Clean install verification report (`CLEAN_INSTALL_REPORT.md`)
