# Time-T — Backends

See `docs/ARCHITECTURE.md` §8 and `docs/DESIGN_DECISIONS.md` DD-5.

## Interface (`timet/backend.py`)

```python
class Backend(ABC):
    name: str
    def capabilities(self) -> BackendCapabilities: ...
    def matmul(self, a, b): ...
    def elementwise(self, op, *args): ...
```

`BackendCapabilities` is a queryable, JSON-serializable descriptor:
`name, supports_f32, supports_f64, gpu, simd, max_tensor_rank, device_name`.

## Existing implementation: `NumpyCpuBackend`

- `name = "cpu-numpy"`
- `gpu = False`
- `simd` reports the actual linked BLAS library name from
  `numpy.show_config()` when available, else `"numpy-generic"` — never a
  guessed or hard-coded value.
- `device_name` reports `platform.processor()` (falls back to
  `platform.machine()`), i.e. the real host CPU identifier of the machine
  running the process, not a placeholder string.

Query it with: `time-t inspect <file> --backend --json`.

## Extensibility, proven with a test double

`tests/test_backend.py::FakeBackend` is a second, independent `Backend`
implementation that exists purely to prove the interface is genuinely
swappable (it is exercised by the test suite, not by production code) —
this is intentionally the honest way to demonstrate "the abstraction works"
without fabricating a second real hardware backend that doesn't exist.

## What does NOT exist

GPU (CUDA/Vulkan/Metal/OpenCL), ARM64-specific SIMD kernels, mobile NPU
support. No performance claim is made about any of these because none of
them have been implemented or measured (master prompt §15, §19). See
`docs/MOBILE.md`.
