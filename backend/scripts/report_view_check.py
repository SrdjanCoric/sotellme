"""Visual check of the real `render_report_view` across report states (task 0038 [verify]).

Dev-time only, throwaway. Drives the actual `render_report_view` from `web.py` — not a mockup —
over mock sessions, so the chosen Coaching/Transcript tab layout and its degraded states can be
eyeballed without running a full keyed interview:

    uv run streamlit run scripts/report_view_check.py

Flip the state in the sidebar: full coaching, early-terminated (partial), early-terminated
(empty), and graded-but-no-coaching.
"""

from __future__ import annotations

import streamlit as st
from report_mocks import mock_coach, mock_grade, mock_profile, mock_transcript

from sotellme.coach import CoachReport
from sotellme.engine import SessionSnapshot
from sotellme.grader import SessionGrade
from sotellme.web import WebState, render_report_view, state_from_snapshot


def _state(
    *,
    grade: SessionGrade,
    coach: CoachReport | None,
    ended_early: bool = False,
) -> WebState:
    return state_from_snapshot(
        SessionSnapshot(
            thread_id="check",
            question=None,
            needs_level=False,
            level="senior",
            profile=mock_profile(),
            transcript=mock_transcript(),
            finished=True,
            ended_early=ended_early,
            closing="Thanks for walking me through it.",
            grade=grade,
            coach=coach,
        )
    )


def states() -> dict[str, WebState]:
    return {
        "Full coaching": _state(grade=mock_grade(), coach=mock_coach()),
        "Early-terminated · partial": _state(
            grade=mock_grade(), coach=mock_coach(), ended_early=True
        ),
        "Early-terminated · empty": _state(
            grade=SessionGrade(scores=[]), coach=None, ended_early=True
        ),
        "Graded but no coaching": _state(grade=SessionGrade(scores=[]), coach=None),
    }


def main() -> None:
    st.set_page_config(page_title="Report view check", layout="centered")
    with st.sidebar:
        st.title("Report state")
        st.caption("Task 0038 · the real render_report_view, not a mockup.")
        all_states = states()
        choice = st.radio("State", list(all_states))
    st.title("Your interview report")
    render_report_view(all_states[choice])


if __name__ == "__main__":
    main()
