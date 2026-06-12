from voice import voice_tells

from sotellme.voice import sanitize


def test_the_scanner_catches_slop_tells() -> None:
    gushy = "Great answer — that's impressive! Tell me more."

    tells = voice_tells(gushy)

    assert "em dash" in tells
    assert "exclamation mark" in tells
    assert "great answer" in tells
    assert "impressive" in tells


def test_the_scanner_catches_every_dash_variant() -> None:
    assert "en dash" in voice_tells("From 2017 – 2021 things changed.")
    assert "double hyphen" in voice_tells("It worked -- mostly.")


def test_a_plain_grounded_question_is_clean() -> None:
    question = "You said you cut latency by 38%. What was going on before that?"

    assert voice_tells(question) == []


def test_sanitize_turns_every_dash_variant_into_a_plain_one() -> None:
    assert sanitize("It shipped—eventually—late.") == "It shipped - eventually - late."
    assert sanitize("It worked – mostly -- fine.") == "It worked - mostly - fine."


def test_sanitize_leaves_clean_text_alone() -> None:
    question = "You said you cut latency by 38%. What was going on before that?"

    assert sanitize(question) == question


def test_sanitized_output_carries_no_dash_tells() -> None:
    assert voice_tells(sanitize("Latency dropped—a lot.")) == []
