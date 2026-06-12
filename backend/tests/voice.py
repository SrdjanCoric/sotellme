from sotellme.voice import DASH_TELLS

BANNED_PHRASES = (
    "great answer",
    "great question",
    "impressive",
    "fantastic",
    "excellent",
    "amazing",
    "awesome",
    "wonderful",
    "love that",
    "perfect",
    "i appreciate",
)


def voice_tells(text: str) -> list[str]:
    lowered = text.lower()
    tells: list[str] = []
    tells.extend(tell for dash, tell in DASH_TELLS.items() if dash in text)
    if "!" in text:
        tells.append("exclamation mark")
    tells.extend(phrase for phrase in BANNED_PHRASES if phrase in lowered)
    return tells
