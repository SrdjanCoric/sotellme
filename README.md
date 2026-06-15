# sotellme

A mock behavioral interviewer that runs in your terminal, built from your CV and the
job you're actually chasing.

## Why I built it

I built this because behavioral interviews are where good candidates trip up. The
questions sound easy, so most people wing them, and the usual prep ("tell me about a
time you failed") is too generic to help with the specific job in front of you.

sotellme makes the practice specific. You give it your CV and a job posting, it reads
up on the company, and then it interviews you against all three at once, so when it
asks why you want the role it can name the product you'd actually be building. At the
end it grades every answer and walks you through the weak ones: what went wrong, and
what to say instead.

## What it does

It starts by parsing your CV (PDF, markdown, or plain text) and reading the posting
into a weighted picture of the competencies the role cares about. Then it pulls a
handful of public pages to put together a short brief on the company, and the interview
runs against all of that. If the posting points to a published values framework, say
Amazon's Leadership Principles, the questions lean on those. It works out the target
level from the posting when the posting states one, and asks you at the start when it
doesn't. That level decides which competencies get weight, and the interviewer itself
never sees it.

A language model runs the session the way a real interviewer would. It opens by asking
who you are, digs into your biggest piece of work, picks a few stories that fit the
role, asks why this company, and wraps up once it has enough to go on. Most sessions
run 8 to 14 questions, and if you answer well you'll get a shorter one. Follow-ups go
after whatever's most interesting in your last answer, like an impact number you
mentioned but never explained, instead of marching through a checklist. The limits it
can't cross (a cap on questions, a guaranteed closing question, a ceiling on web
fetches, a token budget that ends a long session early) are plain code, and they're
unit-tested.

It also screens what you type before it reaches the interview. Go off-topic, or try to
talk it into doing something else, and it nudges you back to the question; the second
off-topic reply in a row wraps the session up. Anything genuinely abusive ends it on one
calm line. Either way, the real answers you gave still get graded.

Once you're done, the smart model reads the whole transcript and scores every answer on
STAR structure, specificity, and ownership against your target level, then prints a
scorecard that names what's weak in each one. The coach takes those scores and turns
them into something you can act on: a fix for each weak answer, drills for the habits it
keeps seeing across the session, and a short study plan. All of it, plus the transcript,
gets written to a Markdown report, and it prints you the path.

It tells you what the run costs. Before the interview starts it shows a rough estimate
for the model you picked, and when it ends it prints the tokens it used and what they
cost, broken out by model and by input versus output. When the provider caches the
repeated system prompt it also notes what that saved. The prices ship with the tool as a
static list, so they're estimates; check your provider's current rates when the number
matters.

## Running it

Run it from source with [`uv`](https://docs.astral.sh/uv/):

```sh
cd backend
uv sync
uv run sotellme interview --cv path/to/cv.pdf --job https://jobs.example.com/senior-backend
```

`--job` takes a link, a file (PDF, markdown, or text), or pasted posting text. For a
link, the tool prefers the page's embedded `JobPosting` data and falls back to the
visible text; Workable postings are read through Workable's public API. Pages that only
render in the browser with JavaScript can't be read, and pasting the text always works.
`--job` is optional; without it the interview runs on a default competency set, with no
company research to ground it.

There's also a local web app if you'd rather not type paths behind flags. Install the
web extra and launch it:

```sh
cd backend
uv sync --extra web
uv run sotellme web
```

It uploads your CV, takes the posting as a link or a paste, runs the interview as a chat,
and lays the report out on the page, with a button to save it to a Markdown file when you
want one. It runs locally just like the CLI, on your own key, with nothing leaving the
machine but the calls to your provider. The terminal version is still there, and it's the
path the tests cover.

A few more things you can do:

- Answers are multi-line with real line editing (Home, End, arrow keys, word jumps).
  Enter starts a new line; Esc then Enter sends, or put `/done` on its own line.
  `Ctrl-X Ctrl-E` opens the answer in your `$EDITOR`.
- `uv run sotellme resume` picks up the latest interrupted session.
- `uv run sotellme reports` lists the reports in the current directory, newest first.
- `uv run sotellme grade session.json --level senior` grades a transcript you already
  have (a JSON list of `{question, answer}` pairs) without running a live interview, so
  you can replay a past session against a changed rubric or model.

## Configuration

There's no account and no server. Pick a provider with `SOTELLME_PROVIDER` (or
`--provider`) and set its key:

| Provider       | Key variable        | Default models (fast / smart)         |
| -------------- | ------------------- | ------------------------------------- |
| `google_genai` | `GOOGLE_API_KEY`    | gemini-3.5-flash / gemini-3.1-pro-preview |
| `anthropic`    | `ANTHROPIC_API_KEY` | claude-sonnet-4-6 / claude-opus-4-8   |
| `openai`       | `OPENAI_API_KEY`    | gpt-5.4-mini / gpt-5.5                |

The fast slot runs the interview side (CV parser, company researcher, director, answer
assessor, interviewer); the smart slot runs the end-of-session grader and coach. You can
override both with `SOTELLME_FAST_MODEL` and `SOTELLME_SMART_MODEL` or the matching
flags. The eval suites run against `google_genai` with an `anthropic` judge, which is
the combo I'd reach for.

The models themselves come from a catalog. Out of the box it ships the per-provider
defaults in the table above. To change what's on offer, write a `~/.sotellme/models.toml`
that lists the models you want and the default for each provider, and that's what the
picker shows. In the web app's Advanced section you can go finer and set a model for each step on its
own, a cheap one for the company research and a stronger one for the questions and the
grading, mixing providers once you've set more than one key. The file holds model names
only; your API keys stay in the environment. It also carries the per-model prices behind
the cost estimates, including the reduced rate for cached input, so you can correct a rate
that's drifted.

The session has a token budget, 400,000 by default, that ends the interview early if a
run goes long and keeps a reserved share back to grade and coach what you gave. Change it
with `SOTELLME_TOKEN_BUDGET`.

Your transcripts and session state stay on your machine. The only things that leave it
are API calls to whichever provider you picked, plus plain HTTP GETs to public pages:
one for a `--job` link, and up to six more for the company brief. Those fetches are
capped per session, truncated per page, and refused for localhost and private
addresses. Your API key is read only by the code that calls the provider and never goes
into a prompt, so no hostile page or posting can talk the model into leaking it. Tests
pin all of this down (`tests/test_fetch.py`, `tests/test_secret_isolation.py`,
`tests/test_injection.py`).

## Development

Requires Python 3.12+, managed with `uv`:

```sh
cd backend
uv sync
uv run ruff check . && uv run mypy && uv run pytest
```

The deterministic suite runs without API keys; the eval tests skip when no provider key
is set. Langfuse tracing only kicks in when its env vars are set.
