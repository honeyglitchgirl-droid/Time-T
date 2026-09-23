import io
from timet.interpreter import run_source


def run_and_capture(src):
    out = []
    run_source(src, stdout_write=out.append)
    return out


def test_arithmetic():
    out = run_and_capture("fn main() { print(1 + 2 * 3) }")
    assert out == ["7"]


def test_let_and_var():
    src = """
    fn main() {
        let x = 5
        var y = 0
        y = y + x
        print(y)
    }
    """
    assert run_and_capture(src) == ["5"]


def test_if_else():
    src = """
    fn main() {
        let x = -3
        if x > 0 {
            print("positive")
        } else if x < 0 {
            print("negative")
        } else {
            print("zero")
        }
    }
    """
    assert run_and_capture(src) == ["negative"]


def test_while_loop():
    src = """
    fn main() {
        var i = 0
        var total = 0
        while i < 5 {
            total = total + i
            i = i + 1
        }
        print(total)
    }
    """
    assert run_and_capture(src) == ["10"]


def test_function_call_and_return():
    src = """
    fn add(a: Int, b: Int) -> Int {
        return a + b
    }
    fn main() {
        print(add(2, 3))
    }
    """
    assert run_and_capture(src) == ["5"]


def test_closures():
    src = """
    fn make_adder(n: Int) -> (Int) -> Int {
        fn adder(x: Int) -> Int {
            return x + n
        }
        return adder
    }
    fn main() {
        let add5 = make_adder(5)
        print(add5(10))
    }
    """
    assert run_and_capture(src) == ["15"]


def test_if_expression():
    src = """
    fn main() {
        let x = -5
        let sign = if x >= 0 { 1 } else { -1 }
        print(sign)
    }
    """
    assert run_and_capture(src) == ["-1"]


def test_bool_printing():
    src = 'fn main() { print(true) print(false) }'
    assert run_and_capture(src) == ["true", "false"]


def test_string_concat():
    src = 'fn main() { print("a" + "b") }'
    assert run_and_capture(src) == ["ab"]


def test_assert_builtin_passes():
    src = "fn main() { assert(1 == 1, \"should be equal\") print(\"ok\") }"
    assert run_and_capture(src) == ["ok"]


def test_assert_builtin_fails():
    import pytest
    from timet.interpreter import RuntimeErr
    src = 'fn main() { assert(1 == 2, "nope") }'
    with pytest.raises(RuntimeErr):
        run_and_capture(src)
