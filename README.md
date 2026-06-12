# sotellme

A mock behavioral interviewer that runs in your terminal. You hand it your CV and a job
posting; it runs an adaptive interview grounded in your actual experience, grades every answer
against a STAR rubric, and writes a coaching report with fixes and drills for the
competencies where you were weak.

A vague or unquantified story gets a follow-up probe, a
complete one advances to the next competency, and follow-ups are capped so a single story
can't eat the session. The routing decision is made by a pure, LLM-free function
(`next_action(state) -> Probe | NextCompetency | Stop`), so the adaptive behavior is
unit-tested rather than prompted.

## Status

Nothing is installable yet; the engine is being built phase by phase. When it ships, the
install will be `uvx sotellme` from PyPI.

## Bring your own key

There's no account and no server. You choose an LLM provider, set your API key in an
environment variable, and the tool calls that provider directly. Transcripts, scores, and
reports stay on your machine; data leaves it only as API calls to the provider you picked. Model selection has two internal slots, a fast one for interviewing and a smart
one for grading and coaching, filled with recommended defaults per provider and individually
overridable.

## Development

Requires Python 3.12+, managed with `uv`:

```sh
cd backend
uv sync
uv run ruff check . && uv run mypy --strict src tests && uv run pytest
```

Secrets are read only from the environment and never enter prompts; the test asserting this
lands with the engine skeleton. Langfuse tracing is a dev-time option that activates only
when its env vars are set.
