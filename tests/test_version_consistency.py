import subprocess
import sys
from pathlib import Path

import pytest
import timet


def test_package_version():
    """Verify package canonical version is 2.1.0."""
    assert timet.__version__ == "2.1.0"


def test_cli_version_output():
    """Verify bin/time-t and python3 -m timet output 2.1.0."""
    res = subprocess.run(
        [sys.executable, "-m", "timet", "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "2.1.0" in res.stdout or "2.1.0" in res.stderr

    # Also test bin/time-t
    bin_path = Path(__file__).parent.parent / "bin" / "time-t"
    assert bin_path.exists()
    assert (bin_path.stat().st_mode & 0o111) != 0, "bin/time-t must be executable"

    res_bin = subprocess.run(
        [str(bin_path), "--version"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "2.1.0" in res_bin.stdout or "2.1.0" in res_bin.stderr


def test_readme_version_consistency():
    """Ensure README matches canonical version and does not contain obsolete version headers."""
    readme_path = Path(__file__).parent.parent / "README.md"
    assert readme_path.exists()
    content = readme_path.read_text()

    assert "v2.1.0" in content, "README must reference current version v2.1.0"
    assert "## What exists today (v2.1.0)" in content or "v2.1.0" in content
    # Ensure obsolete versions are not in the primary header
    assert "## What exists today (v1.1.0)" not in content
    assert "## What exists today (v1.2.0)" not in content
    assert "## What exists today (v1.3.0)" not in content
    assert "## What exists today (v2.0.0)" not in content


def test_bin_permissions():
    """Ensure bin/time-t retains 0755 permissions."""
    bin_path = Path(__file__).parent.parent / "bin" / "time-t"
    mode = oct(bin_path.stat().st_mode)[-3:]
    assert mode in ("755", "775"), f"bin/time-t permissions {mode} must be 755 executable"
