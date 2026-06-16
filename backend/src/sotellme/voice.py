import re

DASH_TELLS = {"—": "em dash", "–": "en dash", "--": "double hyphen"}

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

_AI_DASH_RUN = re.compile(r"\s*(?:—|–|--+)\s*")


def sanitize(text: str) -> str:
    return _AI_DASH_RUN.sub(" - ", text).strip()


def voice_tells(text: str) -> list[str]:
    lowered = text.lower()
    tells: list[str] = []
    tells.extend(tell for dash, tell in DASH_TELLS.items() if dash in text)
    if "!" in text:
        tells.append("exclamation mark")
    tells.extend(phrase for phrase in BANNED_PHRASES if phrase in lowered)
    return tells
