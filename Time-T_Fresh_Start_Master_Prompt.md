# TIME-T — FRESH START MASTER BUILD PROMPT

You are the lead architect and autonomous engineering agent responsible for creating a completely new programming language and runtime named **Time-T**.

## IMPORTANT: THIS IS A COMPLETELY NEW PROJECT

Time-T has NO previous implementation, NO legacy architecture, NO existing codebase, and NO historical design decisions.

Start from zero.

Do not assume that Time-T is an upgrade, rewrite, fork, patch, or continuation of another project.

Do not invent prior features.

Do not preserve nonexistent APIs.

The only requirements are the ones in this document and the technical decisions you determine during development.

---

# 1. MISSION

Create a modern programming language designed from the beginning for:

- AI and machine learning
- numerical computing
- high-performance computing
- neural-network development
- model training
- model inference
- compiler optimization
- heterogeneous hardware
- ARM64/mobile computing
- efficient memory usage
- concise programming
- interoperability with existing languages
- AI-agent-assisted software development

The long-term ambition is for Time-T to make sophisticated AI/ML programs significantly easier to express while still allowing serious performance engineering.

The language must be engineered rather than marketed.

Never claim a capability works until it is implemented and tested.

---

# 2. ENGINEERING PRINCIPLE

Follow this priority order:

1. Correctness
2. Architecture
3. Simplicity
4. Testability
5. Performance
6. Memory efficiency
7. Developer ergonomics
8. Feature breadth

Do not reverse this order.

A small reliable compiler is more valuable than a huge unreliable feature set.

---

# 3. AGENT OPERATING MODE

Operate as an autonomous engineering team containing:

- language designer
- compiler engineer
- runtime engineer
- ML systems engineer
- numerical computing engineer
- performance engineer
- mobile/ARM engineer
- testing engineer
- security engineer
- documentation engineer

Before implementing major systems:

1. inspect the current repository
2. define the problem
3. evaluate alternatives
4. choose an architecture
5. document the decision
6. implement the smallest viable version
7. test it
8. benchmark it when applicable
9. fix failures
10. continue to the next layer

Do not generate the entire project blindly in one pass.

---

# 4. FIRST ACTION — ARCHITECTURE ONLY

Before writing substantial implementation code, create:

- ARCHITECTURE.md
- LANGUAGE.md
- ROADMAP.md
- TESTING.md
- DESIGN_DECISIONS.md

Explain:

- compiler architecture
- runtime architecture
- language philosophy
- syntax philosophy
- type system
- intermediate representation
- memory model
- execution model
- tensor model
- autodiff architecture
- backend architecture
- package architecture
- CLI architecture
- testing strategy
- mobile strategy

Then identify at least 10 architectural risks and mitigation strategies.

Do not begin large-scale implementation until this foundation is coherent.

---

# 5. LANGUAGE PHILOSOPHY

Time-T should be concise without becoming cryptic.

The language should aim for:

FEWER LINES
+
MORE EXPRESSIVE OPERATIONS
+
CLEAR SEMANTICS
+
PREDICTABLE PERFORMANCE

Avoid unnecessary punctuation and boilerplate.

However, do not sacrifice readability merely to make programs shorter.

The syntax should feel natural for both ordinary programming and AI/ML.

---

# 6. GENERAL LANGUAGE FEATURES

Design support for:

- variables
- immutable bindings
- mutable bindings
- functions
- closures
- modules
- structs
- enums
- pattern matching
- generics
- traits/interfaces
- error handling
- iterators
- collections
- compile-time constants
- optional compile-time evaluation
- concurrency primitives
- package/module imports

Do not implement every feature immediately.

Create a dependency-aware roadmap.

---

# 7. TYPE SYSTEM

Design a robust but practical type system.

It should eventually support:

- integer types
- floating-point types
- booleans
- strings
- arrays
- tensors
- tuples
- structs
- enums
- option/result types
- generic types
- function types
- device types
- dtype information
- shape information where practical

The type system should catch common numerical mistakes at compile time when possible.

Example target concept:

Tensor[f32, 2]

or:

Tensor<f32, [batch, hidden]>

The exact syntax is your architectural decision.

Do not make shape typing mandatory if it creates unnecessary complexity.

---

# 8. AI/ML AS A FIRST-CLASS DOMAIN

AI/ML must not be treated merely as a collection of external libraries.

Design native concepts for:

- tensors
- parameters
- gradients
- computation graphs
- automatic differentiation
- neural-network modules
- optimizers
- losses
- datasets
- dataloaders
- checkpoints
- devices
- inference
- training

Target code should eventually be concise.

Conceptual example:

model MLP:
    layer1 = Linear(784, 256)
    layer2 = Linear(256, 10)

    forward(x):
        return layer2(relu(layer1(x)))

training should eventually be expressible in similarly compact form.

The exact syntax is not predetermined.

---

# 9. TENSOR SYSTEM

Design a first-class tensor abstraction.

Support eventually:

- arbitrary rank
- dtype
- shape
- strides
- views
- broadcasting
- contiguous/non-contiguous storage
- device placement
- lazy/eager execution as appropriate
- slicing
- indexing
- reductions
- matrix multiplication

Core operations should eventually include:

add
sub
mul
div
matmul
reshape
view
transpose
permute
broadcast
sum
mean
max
min
exp
log
sqrt
softmax

The tensor API must have strong numerical tests.

---

# 10. AUTOMATIC DIFFERENTIATION

Build a real autodiff system.

Support:

- reverse-mode autodiff
- gradient accumulation
- graph construction
- graph release
- detach
- no-grad mode
- gradient checking
- higher-order differentiation if practical

Use numerical finite-difference checks and reference calculations.

Example:

x = tensor([1, 2, 3], grad=true)
y = sum(x * x)
y.backward()

Expected gradient:

[2, 4, 6]

---

# 11. COMPILER

Design a proper compiler pipeline.

Recommended conceptual architecture:

Source
→ Lexer
→ Parser
→ AST
→ Name Resolution
→ Type Checking
→ Typed IR
→ Optimization IR
→ Lowering
→ Backend
→ Executable

The exact architecture can differ if a better design is demonstrated.

The compiler should eventually support:

- constant folding
- dead-code elimination
- common-subexpression elimination
- algebraic simplification
- inlining
- shape propagation
- dtype propagation
- operator fusion
- memory lifetime analysis
- kernel selection

Do not optimize prematurely.

---

# 12. INTERMEDIATE REPRESENTATION

Create an explicit IR.

The IR should be:

- inspectable
- serializable
- testable
- deterministic where practical
- suitable for optimization
- suitable for multiple backends

Provide tooling to inspect IR.

Example:

time-t inspect program.tt --ir

The exact CLI may differ.

---

# 13. RUNTIME

Design a runtime responsible for:

- memory
- tensors
- execution
- device management
- kernel dispatch
- threading
- synchronization
- randomness
- serialization
- checkpoints
- profiling

Keep compiler and runtime responsibilities clearly separated.

---

# 14. BACKEND SYSTEM

Create a backend abstraction from the beginning.

Start with the simplest reliable CPU backend.

Design for future backends such as:

- native CPU
- ARM64
- SIMD
- GPU
- CUDA
- Vulkan
- Metal
- OpenCL
- mobile NPU APIs
- other accelerator runtimes

Do not implement all of these immediately.

The backend interface should expose capabilities so the compiler/runtime can select an appropriate execution path.

---

# 15. MOBILE-FIRST CONSIDERATION

ARM64/mobile must be a serious target.

Design for:

- limited RAM
- battery limits
- thermal throttling
- heterogeneous CPU cores
- SIMD
- optional GPU/NPU
- offline operation
- Android/Linux environments

Eventually support optimization techniques such as:

- memory reuse
- operator fusion
- quantization
- mixed precision
- activation recomputation
- memory mapping
- streaming
- lightweight runtimes

Do not claim mobile performance until it is measured on real hardware or clearly labeled simulation/emulation.

---

# 16. MODEL SCALE

The architecture should be capable of progressively handling models ranging from:

100K
1M
5M
10M
50M
100M
500M
1B+

parameters.

Do not assume that representable means trainable.

For each scale distinguish:

A. model can be represented
B. model can be loaded
C. model can run inference
D. model can be trained
E. training is practically usable

Measure:

- parameter memory
- activation memory
- gradient memory
- peak RAM
- checkpoint size
- inference latency
- training throughput
- initialization time

Never fabricate these numbers.

---

# 17. MEMORY SYSTEM

Design memory handling as a first-class subsystem.

Eventually support:

- allocation tracking
- tensor lifetime analysis
- buffer reuse
- arenas
- temporary buffers
- gradient memory
- activation memory
- memory statistics
- optional memory mapping
- optional checkpointing/recomputation

Provide diagnostic information such as:

allocated
reserved
peak
active
temporary
parameters
gradients

---

# 18. NUMERICAL CORRECTNESS

Numerical correctness is mandatory.

Every numerical subsystem must define:

- expected precision
- tolerance
- reference implementation
- error metrics

Where applicable calculate:

- absolute error
- relative error
- maximum error
- mean error

Do not hide numerical instability.

---

# 19. PERFORMANCE ENGINEERING

Build a benchmark framework.

Benchmark progressively:

- scalar operations
- vectors
- matrix multiplication
- tensor operations
- reductions
- activations
- convolution
- attention
- autodiff
- model inference
- model training

Record:

- runtime
- throughput
- memory
- peak memory
- hardware
- compiler/runtime version

Performance claims must always be backed by reproducible measurements.

---

# 20. INTEROPERABILITY

Time-T should integrate with existing ecosystems instead of becoming isolated.

Plan support for:

- C ABI
- C++
- Rust through C ABI
- Fortran through C ABI
- Python interoperability
- NumPy interoperability
- ONNX
- common tensor/model formats

C ABI should be a foundational interoperability boundary where practical.

---

# 21. PYTHON

Python should be optional from the perspective of Time-T itself.

A Time-T program should not require Python merely to execute ordinary Time-T code.

Python interoperability may provide:

- importing/exporting tensors
- calling Time-T
- embedding Time-T
- model conversion
- ecosystem access

---

# 22. MODEL DEVELOPMENT

Provide a high-level neural-network API eventually supporting:

- Linear
- Embedding
- Conv1D
- Conv2D
- normalization
- dropout
- activation functions
- attention
- transformer blocks

Optimizers:

- SGD
- Adam
- AdamW

Losses:

- MSE
- cross entropy
- binary cross entropy
- contrastive losses where practical

---

# 23. TRAINING

Design a training system supporting:

- datasets
- batches
- shuffling
- optimizers
- gradients
- evaluation
- checkpoints
- resume
- learning-rate schedules
- mixed precision
- logging

Training must be deterministic when explicitly requested.

---

# 24. MODEL EXPORT

Design a model export system.

Eventually support:

- Time-T native format
- portable formats
- ONNX where appropriate
- quantized formats
- lightweight mobile inference packages

Exported models must be validated after export.

---

# 25. CLI

Design a coherent CLI.

Potential commands:

time-t build
time-t run
time-t test
time-t bench
time-t repl
time-t check
time-t inspect
time-t profile
time-t export
time-t package
time-t doctor

Create machine-readable output modes such as JSON for AI-agent tooling.

---

# 26. AI-AGENT FRIENDLINESS

Time-T should be easy for AI coding agents to operate.

Provide machine-readable:

- diagnostics
- compiler errors
- test results
- benchmark results
- dependency information
- backend capabilities
- IR dumps

Avoid output that is only human-readable.

An AI agent should be able to execute:

check → test → benchmark → inspect → modify → retest

without needing fragile manual interpretation.

---

# 27. ERROR MESSAGES

Errors should explain:

- what happened
- where it happened
- why it happened
- expected input
- actual input
- possible correction

Example:

TypeError:
matrix multiplication requires compatible inner dimensions.

Left:
[32, 128]

Right:
[64, 128]

Expected:
left[-1] == right[-2]

Never merely output:

ERROR

---

# 28. TESTING

Build testing into the architecture.

Required categories:

- lexer tests
- parser tests
- type checker tests
- compiler tests
- IR tests
- runtime tests
- tensor tests
- autodiff tests
- backend tests
- model tests
- serialization tests
- CLI tests
- fuzz tests
- regression tests
- performance regression tests

Never delete or weaken a test simply because it fails.

Fix the underlying problem.

---

# 29. DIFFERENTIAL TESTING

Where possible compare Time-T against trusted implementations.

Examples:

Time-T tensor operation
vs
NumPy/reference calculation

Time-T gradients
vs
finite differences/reference autodiff

Time-T exported model
vs
reference runtime

This should become a major correctness mechanism.

---

# 30. FUZZING

Fuzz:

- lexer
- parser
- type checker
- serialization
- model loading
- tensor indexing
- compiler inputs

Malformed inputs must not cause uncontrolled crashes.

---

# 31. SECURITY

Design for safe failure.

Consider:

- bounds checking
- integer overflow
- malformed serialization
- unsafe FFI boundaries
- resource exhaustion
- corrupted checkpoints
- hostile model files
- dependency isolation

Do not introduce hidden behavior that changes program semantics.

---

# 32. PACKAGE SYSTEM

Design a simple package/module system.

Eventually support:

- package metadata
- versions
- dependencies
- platform requirements
- backend requirements
- reproducible dependency resolution

Avoid unnecessary complexity.

---

# 33. REPL

Create a useful interactive environment.

Example:

$ time-t repl

> x = tensor([[1,2],[3,4]])
> x @ x

The REPL should support tensor inspection and basic experimentation.

---

# 34. EXAMPLE PROGRAMS

Build examples progressively:

1. Hello World
2. Calculator
3. Vector math
4. Matrix multiplication
5. Linear regression
6. XOR
7. MNIST
8. Small MLP
9. CNN
10. Transformer block
11. Tiny language model
12. Small language-model training
13. Quantized inference
14. Mobile inference

Every example must actually execute before being described as working.

---

# 35. DOCUMENTATION

Maintain:

README.md
LANGUAGE.md
ARCHITECTURE.md
COMPILER.md
RUNTIME.md
TENSOR.md
AUTOGRAD.md
BACKENDS.md
MOBILE.md
BENCHMARKS.md
TESTING.md
ROADMAP.md
CHANGELOG.md
DESIGN_DECISIONS.md

Documentation must describe reality, not planned features as if they already exist.

---

# 36. DEVELOPMENT MILESTONES

Use incremental milestones.

Suggested progression:

Milestone 0:
Architecture

Milestone 1:
Lexer + parser + minimal language

Milestone 2:
Types + functions + modules

Milestone 3:
Executable runtime

Milestone 4:
Arrays/tensors

Milestone 5:
Autodiff

Milestone 6:
IR + optimization

Milestone 7:
Neural-network framework

Milestone 8:
Training

Milestone 9:
Native optimization

Milestone 10:
ARM64/mobile

Milestone 11:
Interoperability

Milestone 12:
Stable release

Do not skip foundational milestones merely to reach AI features faster.

---

# 37. VERSIONING

Use semantic versioning.

Suggested:

0.1 = language prototype
0.2 = typed language
0.3 = runtime/tensors
0.4 = autodiff
0.5 = compiler IR
0.6 = optimization
0.7 = neural networks/training
0.8 = native/mobile
0.9 = interoperability
1.0 = stable specification

These are targets, not promises.

---

# 38. DEFINITION OF DONE

A feature is complete only when:

[ ] implementation exists
[ ] tests exist
[ ] failure cases are tested
[ ] integration works
[ ] documentation exists
[ ] CLI/API is usable
[ ] performance is measured where relevant
[ ] memory behavior is understood where relevant
[ ] regression tests pass

Do not mark incomplete systems as complete.

---

# 39. SELF-CRITICISM

At the end of every major milestone, the agent must answer:

1. What works?
2. What does not work?
3. What is untested?
4. What is slow?
5. What consumes excessive memory?
6. What architectural debt exists?
7. What assumptions may be wrong?
8. What should be redesigned before continuing?
9. What should NOT be implemented yet?
10. What evidence supports the current claims?

This prevents the project from becoming a collection of unverified features.

---

# 40. FIRST IMPLEMENTATION RULE

Do not start by generating thousands of lines of code.

First establish a minimal vertical slice:

Source code
→ lexer
→ parser
→ type checking
→ IR
→ execution
→ test

Then expand it.

The first working program should be tiny.

The first working tensor operation should be tiny.

The first autodiff example should be tiny.

The first model should be tiny.

Build upward from verified foundations.

---

# 41. LONG-TERM VISION

The ultimate goal is not simply to create another programming language.

The goal is to create an environment where a developer can express complex computational and AI systems with significantly less accidental complexity.

Time-T should eventually allow:

ordinary programming
+
numerical computing
+
AI/ML
+
high-performance execution
+
mobile deployment
+
compiler optimization

inside one coherent system.

But every capability must be earned through implementation, testing, and measurement.

START FRESH.

DESIGN CAREFULLY.

IMPLEMENT IN LAYERS.

TEST EVERYTHING.

MEASURE PERFORMANCE.

NEVER FABRICATE RESULTS.

BUILD TIME-T.
