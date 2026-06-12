import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path

from sotellme.cli import CLOSING_MESSAGE

STUB_AGENTS_DRIVER = """\
import sys

import sotellme.cli as cli
from sotellme.coverage import StarFlags
from sotellme.profile import CandidateProfile, Role
from sotellme.role import default_role_context


def stub_parser(cv_text):
    return CandidateProfile(
        roles=[Role(title="Engineer", organization="Acme")],
        projects=[],
        quantified_claims=[],
        technologies=[],
    )


def keyword_flagger(answer):
    return StarFlags(
        situation="situation" in answer,
        task="task" in answer,
        action="action" in answer,
        result="result" in answer,
        quantified_result="quantified" in answer,
    )


class StubInterviewer:
    def competency_question(self, profile, transcript, competency):
        return "Tell me about the Acme migration you led."

    def probe_question(self, profile, transcript, gaps):
        return f"Follow-up: what about the {gaps[0]}?"

    def motivation_question(self, context, posting_text, transcript, topic):
        return f"Why this {topic}?"

    def closing_turn(self, transcript):
        return "That covers the migration, thanks for walking me through it."


_real_engine = cli.InterviewEngine


def _one_competency_engine(**kwargs):
    kwargs["max_competencies"] = 1
    return _real_engine(**kwargs)


cli.InterviewEngine = _one_competency_engine
cli._build_profile_parser = lambda config: stub_parser
cli._build_star_flagger = lambda config: keyword_flagger
cli._build_interviewer = lambda config: StubInterviewer()
cli._build_role_builder = lambda config: (lambda posting_text: default_role_context())
sys.exit(cli.main(sys.argv[1:]))
"""


def _write_driver(tmp_path: Path) -> Path:
    driver = tmp_path / "driver.py"
    driver.write_text(STUB_AGENTS_DRIVER)
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
        _read_until(proc, "What level is this interview for?")
        assert proc.stdin is not None
        proc.stdin.write("mid\n")
        proc.stdin.flush()
        _read_until(proc, "blank line or /done")
    finally:
        proc.kill()
        proc.wait()
    assert proc.returncode == -signal.SIGKILL

    weak_then_complete = (
        "I handled the situation and the task.\n\nsituation task action result quantified\n\n"
    )
    resumed = subprocess.run(
        [sys.executable, str(driver), "resume"],
        env=env,
        input=weak_then_complete,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert "Acme migration" in resumed.stdout
    assert "what about the action" in resumed.stdout
    assert "thanks for walking me through it" in resumed.stdout
    assert CLOSING_MESSAGE in resumed.stdout
