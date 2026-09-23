# Time-T — Automatic Differentiation

See `docs/DESIGN_DECISIONS.md` DD-4 for the architecture choice (dynamic
tape / Wengert list, eager mode, like PyTorch/autograd/micrograd).

## How it works

1. Every differentiable `Tensor` op creates a new `Tensor` and, if any input
   `requires_grad`, attaches an `autodiff.Node` recording the parent tensors
   and a `backward_fn` closure mapping an output-gradient to a tuple of
   input-gradients.
2. `Tensor.backward(grad=None)`:
   - Requires a scalar (`size == 1`) output if `grad` is omitted (raises a
     clear `TensorError E0401` otherwise).
   - Computes a reverse topological order over the tape reachable from the
     output tensor (`autodiff.topological_order`).
   - Walks that order in reverse, accumulating gradients into a `grads` dict
     keyed by tensor identity, and into `.grad` on leaf tensors
     (`requires_grad=True` and no `_node`, i.e. not itself the output of a
     tracked op).
   - Broadcasted operations correctly sum-reduce incoming gradients back down
     to the original (pre-broadcast) input shape (`_unbroadcast`).
3. `no_grad()` is a context manager that disables tape creation for its
   duration; ops performed inside it never attach a `Node` regardless of
   `requires_grad` on their inputs.
4. `Tensor.detach()` returns a new tensor with the same values but
   `requires_grad=False` and no tape node — gradients never flow through it.

## Correctness methodology (master prompt §18, §29)

Every backward rule has:
- A **differential test** against the literal master-prompt worked example
  (`x=[1,2,3]`, `y=sum(x*x)`, `y.backward()` → `x.grad == [2,4,6]`), kept as
  a standalone regression test that can never silently regress.
- A **finite-difference gradient check** (`tests/numerics.py::
  finite_difference_grad`, central difference, `h=1e-4`) for every
  operation, comparing against the analytic gradient with the tolerances in
  `docs/DESIGN_DECISIONS.md` DD-7 (`rtol=1e-3, atol=1e-4`), and reporting max
  absolute/relative error on failure — never a bare `assert`.

## Known correctness caveats: kink points

ReLU's gradient at exactly `x = 0` is mathematically undefined (subgradient
∈ [0, 1]); this implementation returns `0` there (via `x > 0`), matching the
common convention. Finite-difference checks deliberately avoid `x == 0`
exactly in test inputs, since a symmetric finite difference straddling the
kink does not have a single "correct" value to compare against either
subgradient choice — this is documented here rather than papered over with a
looser tolerance.

The same applies (v0.2.0) to `clip` at its clamp boundaries and
`max`/`min` at ties: gradient checks avoid exact kink inputs, and the
implementation returns the PyTorch-convention subgradient (strictly-inside
mask for `clip`, even split across ties for `max`/`min`).

Non-differentiable-by-design ops: `argmax` and `one_hot` carry no gradient
at all (they produce indices/constants), matching standard frameworks.

## What's not implemented yet

- Higher-order differentiation (gradient of a gradient) — not implemented,
  not claimed.
- Gradient checkpointing / activation recomputation for memory savings.
- Graph-mode (static/IR-level) autodiff for whole-program optimization —
  see `docs/DESIGN_DECISIONS.md` DD-4's revisit trigger.
