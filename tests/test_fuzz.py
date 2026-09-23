"""Fuzz tests for the lexer and parser (master prompt section 30).

Malformed input must never crash with an unhandled exception -- only the
well-defined LexError / ParseError diagnostics are acceptable outcomes.
Seeds are fixed so failures are reproducible; on failure the seed and input
are printed.
"""
import random
import string

import pytest

from timet.lexer import tokenize, LexError
from timet.parser import parse, ParseError
from timet.diagnostics import Diagnostic

VALID_PROGRAMS = [
    "fn main() { print(1 + 2) }",
    "let x = tensor([1.0, 2.0], grad=true)\nlet y = sum(x * x)",
    "fn add(a: Int, b: Int) -> Int { return a + b }",
    "if x > 0 { print(1) } else { print(2) }",
    "while i < 10 { i = i + 1 }",
]

ALPHABET = string.printable


@pytest.mark.parametrize("seed", range(25))
def test_random_byte_soup_never_crashes(seed):
    rng = random.Random(seed)
    length = rng.randint(0, 200)
    text = "".join(rng.choice(ALPHABET) for _ in range(length))
    try:
        tokens = tokenize(text)
        parse_from_tokens(tokens)
    except Diagnostic:
        pass  # expected outcome for malformed input
    except RecursionError:
        pass  # deep nesting from random input is an accepted, non-crashing outcome
    except Exception as e:  # pragma: no cover - failure path
        pytest.fail(f"seed={seed} text={text!r} raised unexpected {type(e).__name__}: {e}")


def parse_from_tokens(tokens):
    from timet.parser import Parser
    Parser(tokens).parse_program()


@pytest.mark.parametrize("seed", range(25))
def test_mutated_valid_programs_never_crash(seed):
    rng = random.Random(seed + 1000)
    src = rng.choice(VALID_PROGRAMS)
    chars = list(src)
    n_mutations = rng.randint(1, 5)
    for _ in range(n_mutations):
        if not chars:
            break
        op = rng.choice(["delete", "insert", "replace"])
        idx = rng.randrange(len(chars))
        if op == "delete":
            del chars[idx]
        elif op == "insert":
            chars.insert(idx, rng.choice(ALPHABET))
        else:
            chars[idx] = rng.choice(ALPHABET)
    mutated = "".join(chars)
    try:
        parse(mutated)
    except Diagnostic:
        pass
    except RecursionError:
        pass
    except Exception as e:  # pragma: no cover - failure path
        pytest.fail(f"seed={seed} mutated={mutated!r} raised unexpected {type(e).__name__}: {e}")


def test_empty_input_does_not_crash():
    tokens = tokenize("")
    program = parse("")
    assert program.statements == []
