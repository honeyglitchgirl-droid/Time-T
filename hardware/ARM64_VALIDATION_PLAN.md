# ARM64 / Android / Termux Validation Plan

## Target Platforms
- 64-bit ARMv8/ARMv9 architectures (AArch64).
- Android mobile environments running Termux / Linux userspace.

## Validation Commands for Termux / AArch64 Devices

```bash
pkg update && pkg install -y python clang git make
git clone https://github.com/honeyglitchgirl-droid/Time-T.git
cd Time-T
pip install -r requirements.txt
python3 -m timet doctor
python3 -m pytest tests/test_mobile.py -v
python3 -m timet run examples/15_mobile_inference.tt
python3 -m timet verify examples/15_mobile_inference.tt
```

## Status
- **Closed Sandbox**: PASS-STATIC (Architecture guards, uint64 offsets, endianness).
- **Physical Device**: UNVERIFIED-HARDWARE (Requires physical execution on an ARM64 phone or SBC).
