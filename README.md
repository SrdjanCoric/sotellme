# sotellme

A mock behavioral interviewer that runs in your terminal. You hand it your CV and a job
posting; it researches the company, then runs an interview grounded in what your CV actually
claims and in what the company actually makes. Grading and a coaching report are the next
phases on the plan; today the tool conducts the interview itself.

An LLM interview director runs the session the way a trained interviewer would: it opens by
asking who you are, digs into your most significant work, picks a few targeted stories to
match the role, asks why this company, and stops when it has enough signal. A typical session
runs 8 to 14 questions and a strong candidate earns a shorter one. Follow-ups chase whatever
is most interesting in your last answer (an impact number left unexplained, a decision that
needs a why), not a checklist of story elements. Pure logic guarantees the boundaries the
director can't cross: a hard question cap, a guaranteed closing turn, and a ceiling on web
fetches, all unit-tested.

## What works today

A full interview session: the tool parses your CV (PDF, markdown, or plain text), reads the
posting into a weighted competency picture, fetches a handful of public pages about the
company to build a brief, and interviews you against all of it, so a question about
motivation can name the actual product you'd be working on. If the
posting names a published values framework (Amazon's Leadership Principles, for example),
the round leans on those principles. The target level is deduced from the posting when it
states one and asked at the start when it doesn't; it is never silently defaulted, it shapes
which competencies get emphasis, and the interviewer itself never sees it. A killed session
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
posting text always works. The flag is also optional; without a posting the interview runs on
a default competency set, with no company research to ground it.

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

The fast slot runs the whole interview side: the CV parser, the company researcher, the
director, the answer assessor, and the interviewer. The smart slot is reserved for the
grading and coaching phases. Both are overridable with `SOTELLME_FAST_MODEL` and
`SOTELLME_SMART_MODEL` or the matching flags. The eval suites run against `google_genai`
with an `anthropic` judge; that combo is the recommended one.

Transcripts and session state stay on your machine. Data leaves it only as API calls to the
provider you picked, plus plain HTTP GETs to public web pages: one for `--job` when you pass
a link, and up to six more chosen by the research agent while it builds the company brief.
Those fetches are capped per session, truncated per page, refused outright for localhost and
private addresses, and carry no credentials, because the fetching code never reads your
environment. Your API key is read only by the infrastructure code that calls the provider
and never enters any prompt, so no hostile page or posting can talk the model into revealing
it; the model has nothing to reveal. Tests pin these properties (`tests/test_fetch.py`,
`tests/test_secret_isolation.py`, `tests/test_injection.py`).

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
