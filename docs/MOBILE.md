# Time-T — Mobile / ARM64 Strategy

**Current status: not started. No mobile or ARM64-specific code exists.**

Per master prompt §15: "Do not claim mobile performance until it is measured
on real hardware or clearly labeled simulation/emulation." This document
exists to record the *plan*, not to claim progress that hasn't happened.

## Why this is deferred (and how it's sequenced)

`docs/ROADMAP.md` places ARM64/mobile at Milestone 10, after:
- Milestone 6 (IR optimization) exists, so there's something to lower to a
  constrained target efficiently.
- Milestone 9 (native codegen) exists, so mobile isn't the first native
  backend attempted — a desktop-class native backend is a smaller, more
  debuggable first step toward "not interpreted in Python."

Building mobile support before those foundations would mean either (a)
duplicating optimization/codegen work twice, or (b) shipping an
interpreter-on-a-phone story that can't honestly be called "mobile-first
performance engineering."

## What the eventual plan targets (from the master prompt, unimplemented)

- Memory reuse, operator fusion, quantization, mixed precision, activation
  recomputation, memory mapping, streaming, a lightweight runtime.
- Explicit handling of limited RAM, battery limits, thermal throttling,
  heterogeneous CPU cores, SIMD, optional GPU/NPU, offline operation, on
  Android/Linux environments.

## What exists today that is *relevant* but not mobile-specific

- `BackendCapabilities.device_name` already reports the real host
  `platform.processor()`/`platform.machine()` — on an ARM64 host, this would
  correctly report the ARM64 identifier, since it isn't hard-coded to x86.
  This has not been tested on real ARM64 hardware in this environment, so no
  claim is made about correctness there either — it is simply not special-
  cased against ARM64.

## Honest summary

Zero benchmarks, zero binaries, zero measurements exist for mobile/ARM64.
Any claim otherwise would violate master prompt §15's explicit instruction.
