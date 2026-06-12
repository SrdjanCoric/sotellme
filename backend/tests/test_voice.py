from voice import voice_tells


def test_the_scanner_catches_slop_tells() -> None:
    gushy = "Great answer — that's impressive! Tell me more."

    tells = voice_tells(gushy)

    assert "em dash" in tells
    assert "exclamation mark" in tells
    assert "great answer" in tells
    assert "impressive" in tells


def test_a_plain_grounded_question_is_clean() -> None:
    question = "You said you cut latency by 38%. What was going on before that?"

    assert voice_tells(question) == []
