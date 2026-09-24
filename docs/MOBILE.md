# Time-T — Mobile / ARM64 Strategy

**Status: INT8 quantization and mobile packaging implemented (Milestone 10, v1.0.0, DD-27).**

Per master prompt §15: "Do not claim mobile performance until it is measured
on real hardware or clearly labeled simulation/emulation." This document
records the architecture, implementation, and verified capabilities.

## Architecture and Components

1. **INT8 Quantization (`timet.mobile`)**:
   - Dynamic post-training quantization for neural network weights and activations.
   - Symmetric and asymmetric INT8 quantization (`quantize_linear`, `dequantize_linear`).
   - `QuantizedLinear` layer performing integer dot products (`int32` accumulator)
     with dynamically quantized inputs, yielding a 4x reduction in weight memory.
   - `quantize_dynamic(model)` converts full precision `nn.Sequential` and `nn.Linear`
     layers into quantized equivalents.

2. **Mobile Package Deployment (`package_mobile`, `load_mobile`)**:
   - Lightweight, standalone archive format (`.ttm` / `.ttpack`) bundling model
     manifests with raw quantized weight buffers.
   - `load_mobile` reconstructs executable inference pipelines with zero training
     dependencies.

3. **CLI Integration**:
   - `time-t package <checkpoint> -o <path.ttm>` packages trained models for mobile
     deployment with automatic quantization.

4. **Time-T Language Reachability**:
   - Exposed via pre-bound `mobile` module in Time-T code.
   - Demonstrated in `examples/15_mobile_inference.tt`.

