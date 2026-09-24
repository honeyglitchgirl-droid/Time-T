# CPU Validation Plan (x86_64 & Multicore Verification)

## Target Architecture
- x86_64 with AVX2, AVX-512, and OpenMP thread concurrency.

## Verification Steps

1. **Verify Compiler Toolchain**:
   ```bash
   python3 -m timet doctor --json
   ```

2. **Verify Multi-Threaded SIMD OpenMP Backend**:
   ```bash
   python3 -m pytest tests/test_simd_backend.py -v
   ```

3. **Verify JIT Native Kernel Acceleration**:
   ```bash
   python3 -m pytest tests/test_jit_kernels.py -v
   ```

4. **Verify Differential Execution**:
   ```bash
   python3 -m pytest tests/test_ir_exec_diff.py -v
   ```

5. **Measure Reproducible Performance**:
   ```bash
   python3 benchmarks/run_benchmarks.py
   ```
