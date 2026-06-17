from datetime import datetime
from pathlib import Path

from sotellme.coach import AnswerAdvice, CoachReport, Drill
from sotellme.interviewer import Turn
from sotellme.report import (
    SHARE_INVITATION,
    list_reports,
    render_report,
    report_filename,
    write_report,
)


def a_coach_report() -> CoachReport:
    return CoachReport(
        summary="Clear stories, but you keep stopping before the result.",
        answer_advice=[
            AnswerAdvice(
                question="Tell me about the migration.",
                diagnosis="You named the cutover but never said how it landed.",
                fix="End the migration story with the latency you measured after it shipped.",
            )
        ],
        drills=[
            Drill(
                focus="Stating the result",
                exercise="Retell a project in four sentences, the last one a number.",
            )
        ],
        study_plan="Turn each project into a STAR story that ends on a metric.",
    )


def a_transcript() -> list[Turn]:
    return [
        Turn(question="Tell me about the migration.", answer="I led the cutover over a weekend."),
        Turn(question="Why this company?", answer="I want to work on the payments product."),
    ]


def test_render_report_includes_the_summary() -> None:
    rendered = render_report(a_coach_report(), a_transcript())

    assert "Clear stories, but you keep stopping before the result." in rendered


def test_render_report_includes_per_answer_advice_with_the_concrete_gap_and_fix() -> None:
    rendered = render_report(a_coach_report(), a_transcript())

    assert "Tell me about the migration." in rendered
    assert "You named the cutover but never said how it landed." in rendered
    assert "End the migration story with the latency you measured after it shipped." in rendered


def test_render_report_includes_drills() -> None:
    rendered = render_report(a_coach_report(), a_transcript())

    assert "Stating the result" in rendered
    assert "Retell a project in four sentences, the last one a number." in rendered


def test_render_report_includes_the_study_plan() -> None:
    rendered = render_report(a_coach_report(), a_transcript())

    assert "Turn each project into a STAR story that ends on a metric." in rendered


def test_render_report_includes_the_full_transcript() -> None:
    rendered = render_report(a_coach_report(), a_transcript())

    assert "I led the cutover over a weekend." in rendered
    assert "Why this company?" in rendered
    assert "I want to work on the payments product." in rendered


def test_render_report_always_ends_with_the_share_invitation() -> None:
    rendered = render_report(a_coach_report(), a_transcript())

    assert rendered.rstrip().endswith(SHARE_INVITATION)
    assert "https://github.com/SrdjanCoric/sotellme/issues" in SHARE_INVITATION


def test_render_report_keeps_the_footer_and_transcript_when_there_is_nothing_to_coach() -> None:
    empty = CoachReport(summary="", answer_advice=[], drills=[], study_plan="")

    rendered = render_report(empty, a_transcript())

    assert SHARE_INVITATION in rendered
    assert "## What to work on" not in rendered
    assert "## Drills" not in rendered
    assert "I led the cutover over a weekend." in rendered


def test_report_filename_is_timestamped() -> None:
    assert report_filename(datetime(2026, 6, 14, 9, 5, 3)) == "sotellme-report-20260614-090503.md"


def test_write_report_writes_the_rendered_markdown_and_returns_the_path(tmp_path: Path) -> None:
    when = datetime(2026, 6, 14, 9, 5, 3)

    path = write_report(a_coach_report(), a_transcript(), tmp_path, when)

    assert path == tmp_path / "sotellme-report-20260614-090503.md"
    assert path.read_text() == render_report(a_coach_report(), a_transcript())


def test_write_report_suffixes_on_a_same_second_collision(tmp_path: Path) -> None:
    when = datetime(2026, 6, 14, 9, 5, 3)

    first = write_report(a_coach_report(), a_transcript(), tmp_path, when)
    second = write_report(a_coach_report(), a_transcript(), tmp_path, when)
    third = write_report(a_coach_report(), a_transcript(), tmp_path, when)

    assert first.name == "sotellme-report-20260614-090503.md"
    assert second.name == "sotellme-report-20260614-090503-2.md"
    assert third.name == "sotellme-report-20260614-090503-3.md"
    assert first.exists() and second.exists() and third.exists()


def test_list_reports_returns_report_files_newest_first(tmp_path: Path) -> None:
    write_report(a_coach_report(), a_transcript(), tmp_path, datetime(2026, 6, 10, 8, 0, 0))
    write_report(a_coach_report(), a_transcript(), tmp_path, datetime(2026, 6, 14, 8, 0, 0))
    (tmp_path / "notes.md").write_text("not a report")

    reports = list_reports(tmp_path)

    assert [path.name for path in reports] == [
        "sotellme-report-20260614-080000.md",
        "sotellme-report-20260610-080000.md",
    ]
