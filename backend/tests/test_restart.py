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
from sotellme.assessor import AnswerAssessment, StarFlags
from sotellme.director import DirectorDecision
from sotellme.grader import AnswerScore, SessionGrade
from sotellme.profile import CandidateProfile, Role
from sotellme.role import default_role_context


def stub_parser(cv_text):
    return CandidateProfile(
        roles=[Role(title="Engineer", organization="Acme")],
        projects=[],
        quantified_claims=[],
        technologies=[],
    )


def keyword_assessor(topic, transcript):
    answer = transcript[-1].answer
    return AnswerAssessment(
        star=StarFlags(
            situation="situation" in answer,
            task="task" in answer,
            action="action" in answer,
            result="result" in answer,
            quantified_result="quantified" in answer,
        ),
        sufficient_signal="quantified" in answer,
        claims_worth_chasing=[],
    )


class StubDirector:
    def decide(self, situation):
        if not situation.transcript:
            return DirectorDecision(
                action="new_topic", subject="the Acme migration", reason="opener"
            )
        if situation.assessments[-1].assessment.sufficient_signal:
            return DirectorDecision(action="wrap_up", reason="enough signal")
        return DirectorDecision(
            action="follow_up", subject="the action they took", reason="story incomplete"
        )


class StubInterviewer:
    def question_for(self, decision, profile, context, brief, transcript):
        if decision.action == "new_topic":
            return "Tell me about the Acme migration you led."
        return f"Follow-up: what about {decision.subject}?"

    def closing_turn(self, transcript):
        return "That covers the migration, thanks for walking me through it."


def stub_grader(transcript, target_level):
    return SessionGrade(
        scores=[
            AnswerScore(
                question=turn.question,
                star=StarFlags(
                    situation=True, task=True, action=True, result=True, quantified_result=True
                ),
                specificity="high",
                ownership="clear",
                weak_or_missing=[],
                gap="",
                rationale="Complete STAR with a measured outcome at the target level.",
                score=4,
            )
            for turn in transcript
        ]
    )


cli._build_profile_parser = lambda config: stub_parser
cli._build_assessor = lambda config: keyword_assessor
cli._build_director = lambda config: StubDirector()
cli._build_interviewer = lambda config: StubInterviewer()
cli._build_role_builder = lambda config: (lambda posting_text: default_role_context())
cli._build_researcher = lambda config: (lambda posting_text, context: "")
cli._build_grader = lambda config: stub_grader
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
    assert "what about the action they took" in resumed.stdout
    assert "thanks for walking me through it" in resumed.stdout
    assert "Scorecard" in resumed.stdout
    assert CLOSING_MESSAGE in resumed.stdout
