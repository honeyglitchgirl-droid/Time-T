# Hardware Validation Architecture & Master Execution Guide

This document coordinates physical hardware validation across target environments:
- x86-64 Multicore Workstations & Servers
- ARM64 Systems, Apple Silicon, & Mobile (Android/Termux)
- Dedicated GPU Accelerators (CUDA, ROCm, Metal)

In accordance with Section 2 of the audit specification, execution status in the closed sandbox is designated:
- **VERIFIED IN CLOSED SANDBOX**: x86-64 single-socket execution, GCC OpenMP, AVX2 SIMD emulation.
- **UNVERIFIED-HARDWARE**: Physical ARM64, bare-metal GPU clusters, Android/Termux devices.

See the dedicated test plans in `hardware/` for executing validation scripts on physical target machines.
