import re

DASH_TELLS = {"—": "em dash", "–": "en dash", "--": "double hyphen"}

_AI_DASH_RUN = re.compile(r"\s*(?:—|–|--+)\s*")


def sanitize(text: str) -> str:
    return _AI_DASH_RUN.sub(" - ", text).strip()
