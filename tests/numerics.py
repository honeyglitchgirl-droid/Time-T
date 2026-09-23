"""Centralized numeric tolerance policy. See docs/DESIGN_DECISIONS.md DD-7."""
import numpy as np

RTOL = 1e-3
ATOL = 1e-4


def assert_close(actual, expected, rtol=RTOL, atol=ATOL, msg=""):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    abs_err = np.abs(actual - expected)
    rel_err = abs_err / (np.abs(expected) + 1e-12)
    ok = np.all(abs_err <= atol + rtol * np.abs(expected))
    if not ok:
        raise AssertionError(
            f"{msg}\nmax abs error = {abs_err.max():.6g}\n"
            f"max rel error = {rel_err.max():.6g}\n"
            f"rtol={rtol}, atol={atol}\n"
            f"actual={actual}\nexpected={expected}"
        )


def finite_difference_grad(f, x: np.ndarray, h: float = 1e-4) -> np.ndarray:
    """Central-difference numerical gradient of scalar-valued f w.r.t. array x."""
    grad = np.zeros_like(x, dtype=np.float64)
    it = np.nditer(x, flags=["multi_index"])
    xf = x.astype(np.float64).copy()
    for _ in it:
        idx = it.multi_index
        orig = xf[idx]
        xf[idx] = orig + h
        f_plus = f(xf)
        xf[idx] = orig - h
        f_minus = f(xf)
        xf[idx] = orig
        grad[idx] = (f_plus - f_minus) / (2 * h)
    return grad
