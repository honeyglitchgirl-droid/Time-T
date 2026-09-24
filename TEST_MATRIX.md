# Time-T v2.1.0 — Test Matrix (Phase B Specification)

### Execution Summary
- **Execution Environment**: Closed Linux Sandbox (`x86_64`, Intel Xeon @ 2.60GHz, Python 3.11.2, GCC 12.2.0, NumPy 2.4.6, Pytest 9.1.1).
- **Execution Date**: 2026-09-24
- **Command**: `python3 -m pytest -v`
- **Total Collected**: 488 tests
- **Total Passed**: 488 tests (100% pass rate)
- **Total Failed**: 0
- **Total Skipped**: 0
- **Duration**: 12.46s

---

### Detailed Test Subsystem Inventory

| Subsystem / Test Suite | File | Tests | Status | Evidence Level |
| :--- | :--- | :---: | :---: | :---: |
| **AdamW Optimizer** | `tests/test_adamw.py` | 5 | PASS | PASS |
| **Autodiff Dynamic Tape** | `tests/test_autodiff.py` | 18 | PASS | PASS |
| **Backend Abstraction** | `tests/test_backend.py` | 7 | PASS | PASS |
| **JSON Checkpoints** | `tests/test_checkpoint.py` | 8 | PASS | PASS |
| **Binary Checkpoints (.ttck)** | `tests/test_checkpoint_bin.py` | 11 | PASS | PASS |
| **CLI Commands (12 subcommands)** | `tests/test_cli.py` | 14 | PASS | PASS |
| **Conv1D Cross-Correlation** | `tests/test_conv1d.py` | 12 | PASS | PASS |
| **Conv2D Cross-Correlation** | `tests/test_conv2d.py` | 13 | PASS | PASS |
| **DataLoader & Batching** | `tests/test_dataloader.py` | 10 | PASS | PASS |
| **Diagnostics & Error Spans** | `tests/test_diagnostics.py` | 3 | PASS | PASS |
| **Embedding Layer** | `tests/test_embedding.py` | 7 | PASS | PASS |
| **Runnable Examples E2E** | `tests/test_examples.py` | 17 | PASS | PASS |
| **Fuzz Testing (Lexer/Parser)** | `tests/test_fuzz.py` | 51 | PASS | PASS |
| **Interoperability (Safetensors/NPZ)**| `tests/test_interop.py` | 4 | PASS | PASS |
| **AST Interpreter** | `tests/test_interpreter.py` | 11 | PASS | PASS |
| **Typed IR Lowering** | `tests/test_ir.py` | 5 | PASS | PASS |
| **IR Execution Engine** | `tests/test_ir_exec.py` | 11 | PASS | PASS |
| **3-Engine Differential Verification**| `tests/test_ir_exec_diff.py` | 68 | PASS | PASS |
| **Native C JIT Kernels** | `tests/test_jit_kernels.py` | 5 | PASS | PASS |
| **Layer Normalization** | `tests/test_layernorm.py` | 9 | PASS | PASS |
| **Lexer Tokenization** | `tests/test_lexer.py` | 11 | PASS | PASS |
| **Learning Rate Schedulers** | `tests/test_lr_schedule.py` | 6 | PASS | PASS |
| **Mobile & INT8 Quantization** | `tests/test_mobile.py` | 7 | PASS | PASS |
| **Module Import System** | `tests/test_modules.py` | 12 | PASS | PASS |
| **AOT Native C11 Emitter** | `tests/test_native.py` | 17 | PASS | PASS |
| **Neural Network Modules** | `tests/test_nn.py` | 16 | PASS | PASS |
| **IR Optimizer Passes (-O1)** | `tests/test_optimize.py` | 20 | PASS | PASS |
| **Parser Syntax & Precedence** | `tests/test_parser.py` | 18 | PASS | PASS |
| **Runtime Error Diagnostics** | `tests/test_runtime_errors.py`| 6 | PASS | PASS |
| **SIMD + OpenMP Backend** | `tests/test_simd_backend.py` | 3 | PASS | PASS |
| **User-Defined Structs** | `tests/test_structs.py` | 4 | PASS | PASS |
| **Tensor Numerical Operations** | `tests/test_tensor.py` | 25 | PASS | PASS |
| **Training Loops & Callbacks** | `tests/test_train.py` | 7 | PASS | PASS |
| **Training State Resumption** | `tests/test_training_state.py`| 12 | PASS | PASS |
| **Transformer & Attention** | `tests/test_transformer.py` | 12 | PASS | PASS |
| **Static Typechecker** | `tests/test_typechecker.py` | 17 | PASS | PASS |
| **Validation Splits & Modes** | `tests/test_validation.py` | 6 | PASS | PASS |
