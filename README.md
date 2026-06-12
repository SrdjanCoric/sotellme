# sotellme

A mock behavioral interviewer that runs in your terminal. You hand it your CV and a job
posting; it runs an adaptive interview grounded in what your CV actually claims and in the
competencies the posting cares about. Grading and a coaching report are the next phases on
the plan; today the tool conducts the interview itself.

The interview adapts the way a careful human interviewer does. A story that's missing its
setting, your own actions, or a measurable outcome gets a follow-up probe; a complete story
advances the session to the next competency, and follow-ups are capped so one story can't eat
the session. The routing decision is made by a pure, LLM-free function
(`next_action(state) -> Probe | NextCompetency | Stop`), so the adaptive behavior is
unit-tested rather than prompted.

## What works today

A full interview session: the tool parses your CV (PDF, markdown, or plain text), derives a
competency plan from the job posting, asks one grounded question per competency with
follow-up probes where a story has gaps, closes with a short motivation segment ("why this
company", "why this role"), and signs off. If the posting names a published values framework
(Amazon's Leadership Principles, for example), the round maps onto those principles instead
of the default competency set. The target level is deduced from the posting when it states
one and asked at the start when it doesn't; it is never silently defaulted. A killed session
resumes where it left off with `sotellme resume`.

## Running it

Not on PyPI yet; run it from source with `uv`:

```sh
cd backend
uv sync
uv run sotellme interview --cv path/to/cv.pdf --job https://jobs.example.com/senior-backend
```

`--job` takes a link, a file (PDF, markdown, or text), or the pasted posting text. For a
link, the tool prefers the page's embedded `JobPosting` structured data (the block job boards
publish for search engines; LinkedIn job pages carry it) and falls back to the page's visible
text; Workable postings are read through Workable's public API. Pages that only build their
content with JavaScript in the browser can't be read; the error says so, and pasting the
posting text always works. The flag is also optional; without a posting the interview covers
a default competency set and skips the motivation segment.

Answers are multi-line: end one with a blank line or `/done`. `uv run sotellme resume` picks
up the latest interrupted session.

## Bring your own key

There's no account and no server. Pick a provider with `SOTELLME_PROVIDER` (or `--provider`)
and set its key:

| Provider       | Key variable        | Default models (fast / smart)             |
| -------------- | ------------------- | ----------------------------------------- |
| `google_genai` | `GOOGLE_API_KEY`    | gemini-3.1-pro-preview for both slots     |
| `anthropic`    | `ANTHROPIC_API_KEY` | claude-sonnet-4-6 / claude-opus-4-8       |
| `openai`       | `OPENAI_API_KEY`    | gpt-5.4-mini / gpt-5.5                    |

The fast slot runs the interviewer and the answer analysis; the smart slot is reserved for
the grading and coaching phases. Both are overridable with `SOTELLME_FAST_MODEL` and
`SOTELLME_SMART_MODEL` or the matching flags. The eval suites run against `google_genai`
with an `anthropic` judge; that combo is the recommended one.

Transcripts and session state stay on your machine. Data leaves it only as API calls to the
provider you picked, plus one plain HTTP GET when you pass `--job` a link. That fetch happens
once, in ordinary code, before any model runs; the model itself has no tools and no network
access, so a hostile page can't make it fetch anything, read your environment, or send data
anywhere. Secrets are read only from the environment and never enter prompts. Tests pin
these properties (`tests/test_fetch.py`, `tests/test_secret_isolation.py`).

## Development

Requires Python 3.12+, managed with `uv`:

```sh
cd backend
uv sync
uv run ruff check . && uv run mypy && uv run pytest
```

The deterministic suite runs without API keys; the eval tests are key-gated and skip when no
provider key is set. Langfuse tracing is a dev-time option that activates only when its env
vars are set.
