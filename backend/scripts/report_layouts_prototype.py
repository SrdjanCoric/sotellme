"""Live prototype of candidate web-report layouts for task 0038 (mockup-first).

Dev-time only, throwaway. Renders the SAME mock report under four distinct,
Streamlit-native layouts so the user can flip between them in-app and pick one:

    uv run streamlit run scripts/report_layouts_prototype.py

Nothing here ships. Once a layout is chosen it is implemented for real in
`render_report_view` (`web.py`); this script can then be deleted.
"""

from __future__ import annotations

import streamlit as st
from report_mocks import LEVEL, mock_coach, mock_grade, mock_transcript

from sotellme.coach import CoachReport
from sotellme.grader import SessionGrade
from sotellme.interviewer import Turn

STAR_LABELS = {
    "situation": "situation",
    "task": "task",
    "action": "action",
    "result": "result",
    "quantified_result": "quantified result",
}


# --- Shared, layout-independent rendering pieces -----------------------------


def average_score(grade: SessionGrade) -> float:
    return sum(s.score for s in grade.scores) / len(grade.scores)


def render_headline_metrics(grade: SessionGrade) -> None:
    """Compact at-a-glance numbers: level, questions scored, average, range."""
    scores = [s.score for s in grade.scores]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Level", LEVEL.capitalize())
    c2.metric("Answers scored", len(scores))
    c3.metric("Average", f"{average_score(grade):.1f} / 5")
    c4.metric("Range", f"{min(scores)}–{max(scores)}")


def render_scorecard(grade: SessionGrade) -> None:
    """Per-answer scores as a native table plus expandable detail."""
    rows = [
        {
            "Turn": s.turn_index,
            "Score": f"{s.score} / 5",
            "Specificity": s.specificity,
            "Ownership": "—" if s.ownership == "not_applicable" else s.ownership,
            "Question": s.question,
        }
        for s in grade.scores
    ]
    st.dataframe(rows, hide_index=True, width="stretch")
    for s in grade.scores:
        if s.weak_or_missing or s.gap:
            with st.expander(f"Turn {s.turn_index} · {s.score}/5 — what to tighten"):
                if s.weak_or_missing:
                    named = ", ".join(STAR_LABELS[e] for e in s.weak_or_missing)
                    st.markdown(f"**Weak or missing:** {named}")
                if s.gap:
                    st.markdown(s.gap)
    if grade.skipped:
        with st.expander("Turns not scored"):
            for turn in grade.skipped:
                st.markdown(f"**Turn {turn.turn_index}.** {turn.question} — _{turn.reason}_")


def render_summary(coach: CoachReport) -> None:
    st.markdown(coach.summary)


def render_answer_advice(coach: CoachReport) -> None:
    for advice in coach.answer_advice:
        st.markdown(f"**{advice.question}**")
        st.markdown(advice.diagnosis)
        st.markdown(f"**Fix:** {advice.fix}")
        st.markdown("")


def render_drills(coach: CoachReport) -> None:
    for drill in coach.drills:
        st.markdown(f"**{drill.focus}**")
        st.markdown(drill.exercise)


def render_study_plan(coach: CoachReport) -> None:
    st.markdown(coach.study_plan)


def render_coaching_body(coach: CoachReport) -> None:
    """The full coaching content, minus the summary (which layouts place themselves)."""
    if coach.answer_advice:
        st.markdown("#### What to work on, answer by answer")
        render_answer_advice(coach)
    if coach.drills:
        st.markdown("#### Drills")
        render_drills(coach)
    if coach.study_plan.strip():
        st.markdown("#### Study plan")
        render_study_plan(coach)


def render_transcript(transcript: list[Turn]) -> None:
    for turn in transcript:
        st.chat_message("assistant").write(turn.question)
        st.chat_message("user").write(turn.answer)


# --- The four candidate layouts ----------------------------------------------


def layout_tabs(grade: SessionGrade, coach: CoachReport, transcript: list[Turn]) -> None:
    st.caption(f"Interviewing at the {LEVEL} level.")
    tab_coaching, tab_scorecard, tab_transcript = st.tabs(["Coaching", "Scorecard", "Transcript"])
    with tab_coaching:
        render_summary(coach)
        st.divider()
        render_coaching_body(coach)
    with tab_scorecard:
        render_headline_metrics(grade)
        st.divider()
        render_scorecard(grade)
    with tab_transcript:
        render_transcript(transcript)


def layout_tabs_no_scorecard(
    grade: SessionGrade, coach: CoachReport, transcript: list[Turn]
) -> None:
    """Two tabs only — coaching and transcript — no scorecard at all."""
    st.caption(f"Interviewing at the {LEVEL} level.")
    tab_coaching, tab_transcript = st.tabs(["Coaching", "Transcript"])
    with tab_coaching:
        render_summary(coach)
        st.divider()
        render_coaching_body(coach)
    with tab_transcript:
        render_transcript(transcript)


def layout_tabs_thin_scores(
    grade: SessionGrade, coach: CoachReport, transcript: list[Turn]
) -> None:
    """Two tabs — coaching keeps only a thin at-a-glance score strip up top, no table."""
    st.caption(f"Interviewing at the {LEVEL} level.")
    tab_coaching, tab_transcript = st.tabs(["Coaching", "Transcript"])
    with tab_coaching:
        render_headline_metrics(grade)
        st.divider()
        render_summary(coach)
        st.divider()
        render_coaching_body(coach)
    with tab_transcript:
        render_transcript(transcript)


def layout_sectioned(grade: SessionGrade, coach: CoachReport, transcript: list[Turn]) -> None:
    st.caption(f"Interviewing at the {LEVEL} level.")
    st.header("Summary")
    render_headline_metrics(grade)
    st.markdown("")
    render_summary(coach)
    st.divider()
    st.header("Scorecard")
    render_scorecard(grade)
    st.divider()
    st.header("Coaching")
    render_coaching_body(coach)
    st.divider()
    with st.expander("Full transcript"):
        render_transcript(transcript)


def layout_summary_card(grade: SessionGrade, coach: CoachReport, transcript: list[Turn]) -> None:
    with st.container(border=True):
        st.caption(f"Interviewing at the {LEVEL} level.")
        render_headline_metrics(grade)
        st.markdown("")
        render_summary(coach)
    st.markdown("")
    render_coaching_body(coach)
    st.divider()
    with st.expander("Scorecard detail"):
        render_scorecard(grade)
    with st.expander("Full transcript"):
        render_transcript(transcript)


def layout_two_column(grade: SessionGrade, coach: CoachReport, transcript: list[Turn]) -> None:
    st.caption(f"Interviewing at the {LEVEL} level.")
    render_headline_metrics(grade)
    st.divider()
    left, right = st.columns([2, 3])
    with left:
        st.subheader("Scorecard")
        render_scorecard(grade)
    with right:
        st.subheader("Coaching")
        render_summary(coach)
        st.divider()
        render_coaching_body(coach)
    st.divider()
    with st.expander("Full transcript"):
        render_transcript(transcript)


LAYOUTS = {
    "1 · Tabs (coaching first)": layout_tabs,
    "1a · Tabs — Coaching + Transcript only (NO scorecard)": layout_tabs_no_scorecard,
    "1b · Tabs — Coaching + Transcript, thin score strip": layout_tabs_thin_scores,
    "2 · Single scroll, sectioned": layout_sectioned,
    "3 · Summary card on top": layout_summary_card,
    "4 · Two column": layout_two_column,
}


def main() -> None:
    st.set_page_config(page_title="Report layout prototype", layout="wide")
    grade, coach, transcript = mock_grade(), mock_coach(), mock_transcript()

    with st.sidebar:
        st.title("Report layouts")
        st.caption("Task 0038 · same mock data, four layouts. Pick one.")
        choice = st.radio("Layout", list(LAYOUTS.keys()))

    st.title("Your interview report")
    LAYOUTS[choice](grade, coach, transcript)


if __name__ == "__main__":
    main()
