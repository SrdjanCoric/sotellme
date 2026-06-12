import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path

from sotellme.cli import CLOSING_MESSAGE

STUB_PARSER_DRIVER = """\
import sys

import sotellme.cli as cli
from sotellme.profile import CandidateProfile, Role


def stub_parser(cv_text: str) -> CandidateProfile:
    return CandidateProfile(
        roles=[Role(title="Senior Engineer", organization="Acme")],
        projects=[],
        quantified_claims=[],
        technologies=[],
    )


cli._build_profile_parser = lambda config: stub_parser
sys.exit(cli.main(sys.argv[1:]))
"""


def _write_driver(tmp_path: Path) -> Path:
    driver = tmp_path / "driver.py"
    driver.write_text(STUB_PARSER_DRIVER)
    return driver


def _session_env(data_dir: Path) -> dict[str, str]:
    return os.environ | {
        "SOTELLME_DATA_DIR": str(data_dir),
        "SOTELLME_PROVIDER": "anthropic",
        "ANTHROPIC_API_KEY": "dummy-key-no-calls-made",
        "PYTHONUNBUFFERED": "1",
    }


def _read_until(proc: subprocess.Popen[str], marker: str, timeout: float = 30.0) -> str:
    assert proc.stdout is not None
    deadline = time.monotonic() + timeout
    seen = ""
    while time.monotonic() < deadline:
        ready, _, _ = select.select([proc.stdout], [], [], 0.2)
        if ready:
            seen += proc.stdout.readline()
            if marker in seen:
                return seen
    raise AssertionError(f"Timed out waiting for {marker!r}; saw: {seen!r}")


def test_killed_session_resumes_from_checkpoint(tmp_path: Path) -> None:
    cv = tmp_path / "cv.md"
    cv.write_text("# Jane Doe\nSenior Engineer at Acme")
    env = _session_env(tmp_path / "data")
    driver = _write_driver(tmp_path)

    proc = subprocess.Popen(
        [sys.executable, str(driver), "interview", "--cv", str(cv)],
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        _read_until(proc, "blank line or /done")
    finally:
        proc.kill()
        proc.wait()
    assert proc.returncode == -signal.SIGKILL

    resumed = subprocess.run(
        [sys.executable, str(driver), "resume"],
        env=env,
        input="I led the Acme migration.\n\n",
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert "recent project" in resumed.stdout
    assert CLOSING_MESSAGE in resumed.stdout
