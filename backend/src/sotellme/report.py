from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from sotellme.coach import CoachReport
from sotellme.interviewer import Turn

SHARE_INVITATION = (
    "Did this session feel off? Open an issue and share it so the coaching can get better: "
    "https://github.com/SrdjanCoric/sotellme/issues"
)

_REPORT_PREFIX = "sotellme-report-"
_REPORT_GLOB = f"{_REPORT_PREFIX}*.md"


def report_filename(when: datetime) -> str:
    return f"{_REPORT_PREFIX}{when:%Y%m%d-%H%M%S}.md"


def render_report(report: CoachReport, transcript: Sequence[Turn]) -> str:
    sections: list[str] = ["# Interview coaching report"]
    if report.summary.strip():
        sections.append("## How it went\n\n" + report.summary.strip())
    if report.answer_advice:
        blocks = ["## What to work on, answer by answer"]
        for advice in report.answer_advice:
            blocks.append(f"### {advice.question}\n\n{advice.diagnosis}\n\n**Fix:** {advice.fix}")
        sections.append("\n\n".join(blocks))
    if report.drills:
        blocks = ["## Drills"]
        for drill in report.drills:
            blocks.append(f"### {drill.focus}\n\n{drill.exercise}")
        sections.append("\n\n".join(blocks))
    if report.study_plan.strip():
        sections.append("## Study plan\n\n" + report.study_plan.strip())
    if transcript:
        blocks = ["## Transcript"]
        for index, turn in enumerate(transcript, start=1):
            blocks.append(f"### Q{index}. {turn.question}\n\n{turn.answer}")
        sections.append("\n\n".join(blocks))
    sections.append("---\n\n" + SHARE_INVITATION)
    return "\n\n".join(sections) + "\n"


def _free_report_path(directory: Path, filename: str) -> Path:
    candidate = directory / filename
    stem, suffix = candidate.stem, candidate.suffix
    counter = 2
    while candidate.exists():
        candidate = directory / f"{stem}-{counter}{suffix}"
        counter += 1
    return candidate


def write_report(
    report: CoachReport, transcript: Sequence[Turn], directory: Path, when: datetime
) -> Path:
    path = _free_report_path(directory, report_filename(when))
    path.write_text(render_report(report, transcript))
    return path


def list_reports(directory: Path) -> list[Path]:
    return sorted(directory.glob(_REPORT_GLOB), reverse=True)
