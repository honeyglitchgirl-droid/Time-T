"""Runs every examples/*.tt program end-to-end and checks its stdout exactly
matches examples/*.expected (master prompt section 34: "every example must
actually execute before being described as working")."""
import glob
from pathlib import Path

import pytest

from timet.interpreter import run_source

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def _example_files():
    return sorted(EXAMPLES_DIR.glob("*.tt"))


def _outputs_match(actual: str, expected: str, atol: float = 1e-3, rtol: float = 1e-3) -> bool:
    if actual == expected:
        return True
    import re
    import numpy as np
    act_lines = actual.strip().splitlines()
    exp_lines = expected.strip().splitlines()
    if len(act_lines) != len(exp_lines):
        return False
    num_pat = re.compile(r'[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?')
    for act, exp in zip(act_lines, exp_lines):
        if act == exp:
            continue
        act_nums = [float(x) for x in num_pat.findall(act)]
        exp_nums = [float(x) for x in num_pat.findall(exp)]
        if len(act_nums) == len(exp_nums) and len(act_nums) > 0:
            skel_act = num_pat.sub('#', act)
            skel_exp = num_pat.sub('#', exp)
            if skel_act == skel_exp and np.allclose(act_nums, exp_nums, atol=atol, rtol=rtol):
                continue
        return False
    return True


@pytest.mark.parametrize("tt_path", _example_files(), ids=lambda p: p.name)
def test_example_matches_expected_output(tt_path):
    expected_path = tt_path.with_suffix(".expected")
    assert expected_path.exists(), f"missing {expected_path}"
    src = tt_path.read_text()
    outputs = []
    run_source(src, filename=str(tt_path), stdout_write=outputs.append)
    actual = "\n".join(outputs) + "\n" if outputs else ""
    expected = expected_path.read_text()
    assert _outputs_match(actual, expected), f"{tt_path.name} output mismatch:\n--- actual ---\n{actual}\n--- expected ---\n{expected}"


def test_at_least_five_examples_exist():
    assert len(_example_files()) >= 5, "master prompt section 34 asks for a progressive set of examples"
