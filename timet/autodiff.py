"""Reverse-mode automatic differentiation (dynamic tape / Wengert list).

See docs/DESIGN_DECISIONS.md DD-4. Each differentiable Tensor op appends a
Node to the tape recording its inputs and a backward closure; Tensor.backward
walks the tape in reverse, accumulating gradients into leaf tensors.
"""
from __future__ import annotations

import contextlib
from typing import Callable, List, Optional


_grad_enabled = True


def is_grad_enabled() -> bool:
    return _grad_enabled


@contextlib.contextmanager
def no_grad():
    global _grad_enabled
    prev = _grad_enabled
    _grad_enabled = False
    try:
        yield
    finally:
        _grad_enabled = prev


class Node:
    """One entry in the dynamic autodiff tape.

    `inputs` are the parent Tensors, `backward_fn` maps the incoming gradient
    (matching `output`'s shape) to a tuple of gradients, one per input.
    """

    __slots__ = ("inputs", "backward_fn", "output_ref", "name")

    def __init__(self, inputs, backward_fn: Callable, output_ref, name: str):
        self.inputs = inputs
        self.backward_fn = backward_fn
        self.output_ref = output_ref
        self.name = name


def topological_order(root) -> List:
    """Return tensors in an order such that all consumers precede producers
    are visited before their inputs (reverse topo order for backprop)."""
    visited = set()
    order = []

    def visit(t):
        if id(t) in visited:
            return
        visited.add(id(t))
        node = t._node
        if node is not None:
            for inp in node.inputs:
                visit(inp)
        order.append(t)

    visit(root)
    return order
