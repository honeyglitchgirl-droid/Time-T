import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "timet", *args],
        cwd=str(REPO_ROOT), capture_output=True, text=True,
    )


def test_check_valid_file():
    proc = run_cli("check", "examples/01_hello_world.tt")
    assert proc.returncode == 0


def test_check_json_output():
    proc = run_cli("check", "examples/01_hello_world.tt", "--json")
    payload = json.loads(proc.stdout)
    assert payload["status"] == "ok"


def test_check_invalid_file_reports_error(tmp_path):
    bad = tmp_path / "bad.tt"
    bad.write_text("let x: Int = \"oops\"")
    proc = run_cli("check", str(bad), "--json")
    payload = json.loads(proc.stdout)
    assert payload["status"] == "error"
    assert proc.returncode == 1


def test_run_hello_world():
    proc = run_cli("run", "examples/01_hello_world.tt")
    assert proc.returncode == 0
    assert "Hello, Time-T!" in proc.stdout


def test_run_json_mode():
    proc = run_cli("run", "examples/01_hello_world.tt", "--json")
    payload = json.loads(proc.stdout)
    assert payload["status"] == "ok"
    assert payload["stdout"] == ["Hello, Time-T!"]


def test_inspect_ir():
    proc = run_cli("inspect", "examples/02_calculator.tt", "--ir")
    assert proc.returncode == 0
    assert "fn add" in proc.stdout


def test_inspect_backend_json():
    proc = run_cli("inspect", "examples/01_hello_world.tt", "--backend", "--json")
    payload = json.loads(proc.stdout)
    assert payload["backend"]["name"] == "cpu-numpy"


def test_not_implemented_commands_report_honestly():
    for cmd in ("build", "profile", "export", "package", "doctor"):
        proc = run_cli(cmd, "--json")
        payload = json.loads(proc.stdout)
        assert payload["status"] == "not_implemented"
        assert proc.returncode == 2


def test_version_flag():
    proc = run_cli("--version")
    assert proc.returncode == 0
    assert "time-t" in proc.stdout
