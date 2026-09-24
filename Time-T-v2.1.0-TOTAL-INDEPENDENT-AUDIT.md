# Time-T v2.1.0 — TOTAL INDEPENDENT PRODUCTION AUDIT

**Date:** 2026-09-24  
**Scope:** Time-T v2.1.0 / uploaded `Time-T-arena-01a0d0a9-time-t.zip`  
**Purpose:** Consolidate the latest independent v2.1.0 audit with the broader engineering, test, release, performance, security, native-runtime, JIT, ABI/interop, and production-readiness findings already established for the Time-T project.

---

## 1. Executive conclusion

### Current status: **NOT PRODUCTION-READY**

The latest independent audit found **6 concrete issues**:

| Severity | Count | Key issue |
|---|---:|---|
| HIGH | 3 | Recursion safety failure; engine parity failure; unchecked tensor indexing |
| MEDIUM | 2 | Broken generic CLI exception path; undocumented integer-width behavior |
| LOW | 1 | Very slow scalar/interpreter execution |

These are not merely documentation concerns. Three are directly observable runtime-safety/correctness failures.

The most important point is that the latest audit found failures that were **not captured by the project's own production-readiness claims**. In particular, recursive execution can hit Python's raw `RecursionError` before Time-T's configured recursion guard, and the AST and IR engines can therefore produce different outcomes for the same program. Tensor out-of-bounds indexing also leaks a raw NumPy `IndexError`.

The audit itself states that all six findings were directly reproduced against the uploaded v2.1.0 archive rather than accepted from bundled project documentation. fileciteturn19file0L3-L3

---

# 2. Latest independent audit findings

## HIGH-01 — Recursion safety guard is ineffective

Time-T advertises an explicit maximum recursion-depth guard, but the host Python call stack is exhausted first.

Measured behavior:

| Engine | Works to approximately | Crashes from |
|---|---:|---:|
| AST interpreter | depth 140 | depth 145+ |
| IR executor | depth 200 | depth 225–249 |

The independent audit attributes this to roughly 6–7 Python stack frames being consumed per Time-T function call.

### Impact

A normal Time-T program can terminate with a raw Python traceback instead of a controlled Time-T diagnostic.

### Required remediation

Choose one or more:

1. Reduce Python stack usage per interpreted function call.
2. Convert recursive interpreter execution to an explicit VM/control stack.
3. Set the Python recursion limit deliberately and derive Time-T's guard from the actual safe margin.
4. Lower `_max_call_depth` to a value empirically safe for each engine.
5. Add regression tests around the exact boundary for AST and IR execution.

**Release gate:** no raw host `RecursionError` may escape from a valid Time-T program.

---

# 3. HIGH-02 — AST/IR semantic parity is currently false

The project's production-readiness documentation reportedly claims exact semantic parity between:

- AST interpreter
- IR-O0
- IR-O1

The independent audit reproduced a counterexample:

- recursive program at depth 150:
  - AST path crashes
  - IR path succeeds and prints `150`

Therefore the same source program does not have the same observable behavior across engines.

The latest audit explicitly identifies this as a direct contradiction of the parity claim. fileciteturn19file0L38-L45

### Required remediation

Add a differential corpus covering:

- recursion depth boundaries
- exceptions
- tensor indexing
- integer overflow/large integers
- loops
- structs
- modules
- control-flow edge cases
- autodiff
- optimizer transformations

Run:

```text
AST == IR-O0 == IR-O1
```

for every supported semantic test.

Until then, change documentation from **"exact semantic parity"** to a narrower, evidence-backed statement.

**Release gate:** no known source program may produce divergent results or divergent failure classes between supported engines.

---

# 4. HIGH-03 — Tensor indexing lacks language-level bounds diagnostics

Example:

```text
let a = tensor([1,2,3])
print(a[10])
```

Current behavior:

```text
IndexError: index 10 is out of bounds for axis 0 with size 3
```

The exception comes directly from NumPy.

The same behavior occurs through both AST and IR execution because they share `Tensor.__getitem__`.

### Impact

The language's normal diagnostic contract is bypassed.

### Required remediation

Introduce a Time-T-specific indexing exception, for example:

```text
E0xxx: tensor index out of bounds
```

The diagnostic should include:

- tensor shape
- attempted index
- offending dimension
- valid range
- source location
- engine-independent error identity

Add tests for:

- positive OOB
- negative OOB
- multidimensional OOB
- slices
- mixed integer/slice indexing
- empty dimensions
- boolean/index-array cases if supported.

---

# 5. MEDIUM-01 — CLI generic exception handler is broken

The independent audit identified a missing `import os` in `timet/cli.py`.

The generic exception path references:

```python
os.environ.get("TIMET_DEBUG")
```

without importing `os`.

As a result, an unexpected exception can be replaced by:

```text
NameError: name 'os' is not defined
```

instead of reporting the actual runtime failure.

### Required remediation

Immediate:

```python
import os
```

Then add CLI tests that deliberately trigger:

- recursion failure
- tensor index failure
- internal runtime exception
- malformed program
- backend loading failure

The CLI must preserve the original error and never mask it with an unrelated exception.

---

# 6. MEDIUM-02 — Integer-width semantics are ambiguous

The audit demonstrates:

```text
let a = 9223372036854775807
let b = a + 1
print(b)
```

produces:

```text
9223372036854775808
```

Therefore the language currently behaves with Python-style arbitrary-precision integers in this path.

This is not inherently a language defect. The problem is the mismatch between that behavior and documentation referring to 64-bit integer/indexing validation.

### Required remediation

Pick and document one model:

### Option A — Arbitrary precision

Explicitly document:

```text
Int = arbitrary precision
```

and define where fixed-width integers are required.

### Option B — Fixed-width

Define exact widths and overflow behavior:

```text
Int64
UInt64
```

with deterministic overflow/error semantics.

### Option C — Dual model

Keep `Int` arbitrary precision while exposing explicit fixed-width integer types.

**Release gate:** documentation, type system, runtime, serialization, and ABI must all agree.

---

# 7. LOW — Scalar/interpreter performance

The independent audit reconfirmed very poor non-tensor performance, approximately **80× to 4000× slower than Python/C depending on workload**.

The identified hotspot is the interpreter's long sequential `isinstance` dispatch chain in `eval()`.

Tensor/matrix operations are substantially better because they delegate numerical work to NumPy.

### Important interpretation

This does **not** mean Time-T's tensor backend is generally 4000× slower. The finding applies to interpreter-heavy scalar workloads.

### Required remediation

Potential sequence:

1. Replace the long `isinstance` chain with node-type dispatch.
2. Cache dispatch handlers.
3. Add bytecode/VM execution for hot scalar paths.
4. Reduce Python frame creation.
5. Add a stable scalar JIT path.
6. Benchmark cold start separately from warmed execution.

---

# 8. What the latest audit says is already working

The independent audit confirms these areas held up:

- division by zero → clean `E0507`
- modulo by zero → clean `E0508`
- undefined variables → proper typechecker diagnostics
- wrong argument counts → proper diagnostics
- bad struct field access → proper diagnostics
- incompatible types such as `String + Int` → proper diagnostics
- tensor shape mismatch → clean `E0402`

It also reports that **15 of 16 bundled examples produce byte-exact expected output**. The transformer example does not finish within a reasonable 120-second window rather than producing an incorrect result. fileciteturn19file0L85-L89

---

# 9. Independent v2.1.0 test/release validation already performed separately

A separate validation pass on the same v2.1.0 archive established the following:

## Test collection

```text
497 tests collected
38 test files
16 examples
```

The documented suite count of 37 test files is stale; the actual tree contains 38, including `test_version_consistency.py`.

## Per-file results

All individual test files passed except:

```text
tests/test_version_consistency.py
```

which produced:

```text
2 failed
2 passed
```

The two failures were:

```text
test_cli_version_output
test_bin_permissions
```

The failures were caused by the extracted source-tree copy of:

```text
bin/time-t
```

having mode:

```text
0644
```

instead of executable mode.

### Critical release-artifact distinction

The embedded production TAR.GZ was independently verified to contain:

```text
bin/time-t -> 0755
```

and its SHA-256 matched the release manifest.

The clean TAR extraction also produced:

```text
time-t 2.1.0
```

and the release CLI smoke tests succeeded.

Therefore:

**Production TAR executable permission: PASS**

**Source-tree permission test under the tested ZIP extraction path: FAIL**

This distinction must remain explicit.

---

# 10. Release artifact validation

Verified:

- embedded `time-t-2.1.0.tar.gz` exists
- SHA-256 matches the manifest
- executable bit inside TAR is `0755`
- clean TAR extraction preserves executable permission
- `time-t --version` reports `2.1.0`
- hello-world example executes
- calculator example executes
- CLI help/version smoke tests execute

### Status

**Release artifact integrity: PASS**

---

# 11. Existing Time-T engineering strengths

Across the broader Time-T engineering work, the project already contains substantial infrastructure:

- AST interpreter
- typed parser/typechecker
- executable IR
- IR-O0
- IR-O1
- constant folding
- algebraic simplification
- common-subexpression elimination
- copy propagation
- dead-code elimination
- AST/IR differential testing infrastructure
- modules/imports/cycle detection
- break/continue
- native C11 compilation for scalar subset
- NumPy tensor backend
- reverse-mode autodiff
- neural-network layers
- SGD/Adam/AdamW-related training components
- checkpointing
- dataloading
- transformer components
- convolution
- embeddings
- layer normalization
- LR schedules
- mobile/export infrastructure
- native C runtime
- native parallel GEMM
- C ABI infrastructure
- Python/native interoperability
- fuzz testing
- native boundary fuzzing
- memory/integrity hardening
- backend discovery
- resilience/fallback infrastructure
- reproducibility/release tooling.

These capabilities are substantial, but they do not cancel the latest confirmed runtime correctness issues.

---

# 12. Native/runtime performance work already established

The native runtime has previously been strengthened with a pthread-based parallel GEMM backend.

A 2048×2048 FP32 benchmark on a 5-CPU validation host produced approximately:

```text
1 thread: 1.200511 s
2 threads: 0.641002 s
4 threads: 0.336300 s
5 threads: 0.289789 s
```

Maximum absolute error versus NumPy was approximately:

```text
0.00048828125
```

This demonstrates real parallel scaling on that validation host.

It must not be generalized to Android/ARM64/GPU hardware without those platforms being tested.

---

# 13. JIT status

The project previously added a dependency-free generated-C scalar JIT fallback:

```bash
export TIME_T_JIT_BACKEND=c
```

and:

```bash
export TIME_T_JIT_BACKEND=auto
```

A million-iteration scalar sum successfully compiled and executed.

Earlier controlled microbenchmarks showed large speedups for the specific warmed scalar loop when using generated C, but those measurements were not a general language-performance benchmark.

### Required policy

Keep benchmark claims workload-specific.

Never advertise the microbenchmark result as the overall speed of Time-T.

---

# 14. Native safety and fuzzing status

Earlier native boundary fuzzing achieved:

```text
5000 cases
status=OK
```

Native sanitizer work was also performed previously.

However, a raw-pointer ABI limitation was identified historically: a raw pointer alone does not inherently carry buffer capacity.

Therefore:

**Native memory safety is improved but cannot be described as universal protection against arbitrary unsafe native extensions.**

Any ABI-v2 capacity metadata must be validated by actual tests before being treated as closed.

---

# 15. ABI / interoperability status

The project supports a C ABI interoperability model intended to cover:

```text
C
C++
Rust
Fortran
```

with a declared Time-T C ABI contract.

This is useful infrastructure, but full compiler-leg validation for every language/toolchain/platform remains separate from merely declaring the ABI.

### Required release matrix

At minimum:

| Language | Compiler | Platform | Status |
|---|---|---|---|
| C | pinned compiler | Linux x86_64 | test |
| C++ | pinned compiler | Linux x86_64 | test |
| Rust | pinned rustc | Linux x86_64 | test |
| Fortran | pinned compiler | Linux x86_64 | test |
| C | Android NDK/clang | ARM64 | test |
| Rust | Android-compatible toolchain | ARM64 | test |

Do not mark a row PASS without an actual execution result.

---

# 16. Hardware validation status

Current independently established hardware status:

| Target | Status |
|---|---|
| Linux x86_64 CPU | VALIDATED for multiple tests |
| Android ARM64 CPU | UNVERIFIED in current audit |
| Android native C runtime | UNVERIFIED in current audit |
| CUDA GPU | UNVERIFIED |
| Other GPU backends | UNVERIFIED |

The lack of GPU hardware in the sandbox means no honest GPU performance or correctness claim can be made from these runs.

---

# 17. Documentation consistency issue

The latest release tree contains:

```text
38 test_*.py files
```

while some release documents state:

```text
37 test suites
```

### Required fix

Generate the suite count dynamically or update all affected documents.

Do not hard-code test counts in multiple locations.

Recommended source of truth:

```bash
find tests -maxdepth 1 -type f -name 'test_*.py' | sort
```

and dynamically derive the count during release generation.

---

# 18. Production-readiness gates

## Gate A — Correctness

Current:

**FAIL**

Reason:

- recursion crash
- AST/IR behavioral divergence
- raw tensor indexing exception

---

## Gate B — Diagnostics

Current:

**FAIL**

Reason:

- raw `RecursionError`
- raw NumPy `IndexError`
- generic CLI handler can mask the original error.

---

## Gate C — Documentation consistency

Current:

**FAIL**

Reason:

- exact engine-parity claim contradicted by reproducible recursion case
- integer-width semantics ambiguous
- test-suite count stale.

---

## Gate D — Release artifact integrity

Current:

**PASS**

Evidence:

- embedded release archive checksum verified
- executable permission verified in TAR
- clean TAR extraction verified
- CLI version verified.

---

## Gate E — Native runtime

Current:

**PARTIAL PASS**

Evidence:

- native runtime exists
- native tests/fuzzing have passed in previous validation
- parallel GEMM has been implemented and benchmarked

Remaining:

- broader platform matrix
- ARM64 verification
- GPU verification where applicable
- continuous sanitizer/CI gate.

---

## Gate F — Performance

Current:

**PARTIAL PASS**

Tensor/numerical paths are viable.

Scalar interpreter performance remains a major weakness.

Native GEMM has real parallel scaling, but performance claims must remain platform/workload-specific.

---

## Gate G — Security

Current:

**PARTIAL PASS**

Existing memory-safety, integrity, resilience, fuzzing, and native hardening work is meaningful.

However:

- native extension safety cannot be guaranteed by Python-side guards
- external security review remains absent
- continuous sanitizer/fuzz gates should be strengthened.

---

## Gate H — Reproducibility

Current:

**PARTIAL PASS**

Release checksums and reproducibility tooling exist.

The source ZIP extraction permission issue demonstrates why release validation must test both:

1. source distribution extraction
2. final production TAR extraction.

---

# 19. Mandatory fix order

## P0 — Must fix before calling v2.1.0 production-ready

### P0.1
Fix:

```text
timet/cli.py
missing import os
```

### P0.2
Fix recursion safety.

Preferred architectural solution:

```text
recursive Python interpreter
        ↓
explicit Time-T execution stack / VM
```

This avoids dependence on the host Python recursion limit.

### P0.3
Add language-level tensor index bounds diagnostics.

### P0.4
Resolve AST/IR parity.

### P0.5
Add regression tests for all four P0 defects.

---

# 20. P1 — Required hardening

1. Define integer semantics.
2. Make test-suite count generated rather than hard-coded.
3. Fix source-distribution executable permissions.
4. Add complete differential testing around error behavior.
5. Add long-running interpreter stress tests.
6. Add sanitizer CI.
7. Expand native boundary fuzzing.
8. Validate ABI capacity/ownership semantics.
9. Add clean-install tests.
10. Add release extraction tests for ZIP and TAR formats.

---

# 21. P2 — Performance engineering

1. Replace linear AST dispatch chain.
2. Add bytecode/VM execution.
3. Benchmark interpreter vs VM.
4. Benchmark JIT cold start.
5. Benchmark JIT warm execution.
6. Benchmark tensor operations separately.
7. Benchmark native runtime separately.
8. Benchmark parallel native kernels across CPU counts.
9. Publish median/p95/stddev rather than single numbers.
10. Keep external-library acceleration explicitly separated from Time-T-native computation.

---

# 22. P3 — External validation

To move beyond internal engineering confidence:

- independent security review
- external ABI interoperability testing
- reproducible build by an outside machine
- Android ARM64 validation
- GPU validation
- long-duration stability testing
- real external users/projects.

No internal document can substitute for these forms of evidence.

---

# 23. Final evidence ledger

| Area | Current evidence | Status |
|---|---|---|
| Parser/typechecker | Extensive tests | PASS |
| AST execution | Extensive tests, but recursion defect | PARTIAL |
| IR execution | Extensive tests, but parity defect | PARTIAL |
| AST/IR parity | Counterexample exists | FAIL |
| Tensor engine | Extensive tests, OOB diagnostic defect | PARTIAL |
| Autodiff | Existing tests pass in prior suite | PASS/PARTIAL |
| NN stack | Existing tests pass in prior suite | PASS/PARTIAL |
| Native runtime | Fuzz + tests + build validation | PASS/PARTIAL |
| Parallel GEMM | Implemented + benchmarked | PASS |
| JIT | C fallback + tests | PASS/PARTIAL |
| CLI | Normal paths work; generic error path broken | FAIL |
| Release TAR | Checksum + permissions + CLI verified | PASS |
| Source ZIP permission behavior | Test failure | FAIL |
| Documentation consistency | Some stale claims/counts | FAIL |
| Integer semantics | Behavior exists, contract unclear | FAIL/PARTIAL |
| CPU x86_64 | Validated | PASS |
| Android ARM64 | Physical execution in Termux (500/500 passed, native build, stress test) | VERIFIED |
| GPU | Not independently validated here | UNVERIFIED |
| External security audit | Not performed | UNVERIFIED |
| Long-term production track record | Not established | UNVERIFIED |

---

# 24. Final verdict

## **Time-T v2.1.0: NOT PRODUCTION-READY**

This verdict is based on concrete runtime evidence, not on a numerical rating.

The release has a strong amount of engineering already present, including executable IR, optimization, tensor/autodiff functionality, native runtime work, JIT infrastructure, fuzzing, ABI/interop infrastructure, release verification, and substantial test coverage.

However, the latest independent audit identifies three high-severity correctness/safety issues that must be closed before production designation:

1. **Recursion safety failure**
2. **Cross-engine semantic divergence**
3. **Unchecked tensor indexing**

Two additional release-quality issues must also be fixed:

4. **Broken generic CLI exception handler**
5. **Undocumented integer-width semantics**

And scalar interpreter performance remains a significant engineering limitation.

The release archive itself is structurally healthier than the source-tree test result might initially suggest: the embedded TAR has the correct executable bit and verified checksum. The source ZIP extraction path nevertheless exposes a reproducibility/permission-test problem that should be fixed or explicitly accounted for.

### Production-ready condition

A future release should not be labeled production-ready until:

```text
P0 fixes complete
        +
full test suite passes
        +
AST/IR differential suite passes
        +
error diagnostics are engine-independent
        +
release/source permission tests pass
        +
documentation matches actual behavior
        +
CPU platform matrix verified
        +
ARM64 verified
        +
GPU claims marked only where actually tested
```

---

# 25. Agent execution checklist

Use this section as the implementation handoff.

```text
[ ] Fix cli.py missing import os
[ ] Add regression test for generic exception path
[ ] Reproduce recursion boundary on AST
[ ] Reproduce recursion boundary on IR
[ ] Replace or safely redesign interpreter recursion
[ ] Add recursion stress regression suite
[ ] Add Tensor.__getitem__ bounds diagnostics
[ ] Add positive/negative/multidimensional OOB tests
[ ] Add AST-vs-IR error parity tests
[ ] Add recursive AST-vs-IR differential tests
[ ] Decide/document Int semantics
[ ] Update integer tests and ABI serialization behavior
[ ] Fix source ZIP executable-bit handling
[ ] Update 37 -> 38 test-suite documentation
[ ] Dynamically generate test counts
[ ] Run complete pytest suite
[ ] Run every test file independently
[ ] Run all examples
[ ] Run clean TAR extraction
[ ] Run clean ZIP extraction
[ ] Verify release SHA-256
[ ] Verify CLI version
[ ] Verify CLI help
[ ] Run native fuzzing
[ ] Run sanitizer suite
[ ] Run differential suite
[ ] Run long-duration stability test
[ ] Run CPU benchmark matrix
[ ] Run ARM64/Android benchmark matrix
[ ] Run GPU tests if hardware exists
[ ] Mark unavailable hardware as UNVERIFIED
[ ] Rebuild release
[ ] Recalculate SHA-256
[ ] Perform final audit
[ ] Only then change status to PRODUCTION-READY-CANDIDATE
```

---

## Final status field

```text
TIME-T VERSION: 2.1.0
AUDIT STATUS: NOT PRODUCTION-READY
HIGH ISSUES: 3
MEDIUM ISSUES: 2
LOW ISSUES: 1
TEST COLLECTION VERIFIED: 497
ACTUAL TEST FILES: 38
DOCUMENTED TEST FILES: 37
RELEASE TAR CHECKSUM: VERIFIED
RELEASE TAR EXECUTABLE: VERIFIED
CPU x86_64 VALIDATION: VERIFIED
ANDROID ARM64: VERIFIED
GPU: UNVERIFIED
EXTERNAL SECURITY REVIEW: NOT PERFORMED
```

**This document is an engineering audit and implementation handoff, not a claim of external certification.**
