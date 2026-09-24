# GPU Validation Plan (Hardware Discovery & Acceleration)

## Target Hardware
- NVIDIA GPUs with CUDA Runtime / Driver APIs
- AMD GPUs with ROCm / HIP
- Apple Silicon GPUs with Metal Performance Shaders (MPS)

## Verification Procedure on Bare-Metal GPU Hardware

1. **Query Physical Accelerators**:
   ```bash
   python3 -c "import timet.backend as b; print(b.get_default_backend().capabilities().to_json())"
   ```

2. **Verify Graceful Fallback on Systems Without GPU**:
   - Systems lacking dedicated GPU runtimes automatically bind `cpu-simd-openmp` or `cpu-numpy` with zero crashes.

3. **Status**:
   - `UNVERIFIED-HARDWARE`: Requires physical GPU device allocation.
