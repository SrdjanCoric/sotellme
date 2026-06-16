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

You give it your CV and the job you're chasing, and it interviews you against both, plus
a short brief it builds on the company from a handful of public pages. It runs the
session the way a real interviewer would: it opens on who you are, digs into your biggest
piece of work, picks the stories that fit the role, and chases the interesting thread in
your last answer rather than marching through a checklist. Most sessions run 8 to 14
questions. When you're done, the smart model reads the whole transcript and scores every
answer on STAR structure, specificity, and ownership against your target level, then
writes you a Markdown report: a scorecard that names what's weak, a fix for each soft
answer, and a short study plan. It also tells you what the run cost.

## Quickstart

Set one provider key first (see [Configuration](#configuration)):

```sh
export ANTHROPIC_API_KEY=...   # or GOOGLE_API_KEY, or OPENAI_API_KEY
```

The easiest way in is the local web app. Pull in the web extra and launch it:

```sh
uvx --from "sotellme[web]" sotellme web
```

It opens in your browser: upload your CV, paste a posting or drop in a link, run the
interview as a chat, and read the report on the page, with a button to save it as
Markdown. Everything runs locally on your own key.

If you'd rather stay in the terminal, run the interview straight from
[`uvx`](https://docs.astral.sh/uv/), no clone needed:

```sh
uvx sotellme interview --cv path/to/cv.pdf --job https://jobs.example.com/senior-backend
```

`--job` takes a link, a file (PDF, markdown, or text), or pasted posting text, and it's
optional; without it the interview runs on a default competency set with no company
research to ground it. For a link the tool prefers the page's embedded `JobPosting` data
and falls back to the visible text, and Workable postings are read through their public
API. Pages that only render with JavaScript can't be read, and pasting the text always
works.

Answers are multi-line with real line editing (Home, End, arrow keys, word jumps). Enter
starts a new line; Esc then Enter sends, or put `/done` on its own line.

### Commands

| Command | What it does |
| --- | --- |
| `sotellme interview --cv <path> [--job <link\|file\|text>]` | Start a new interview session. |
| `sotellme resume` | Pick up the latest interrupted session. |
| `sotellme reports` | List the coaching reports in this directory, newest first. |
| `sotellme grade <transcript.json> --level <junior\|mid\|senior\|staff>` | Grade a transcript you already have (a JSON list of `{question, answer}` pairs) without running a live interview. |
| `sotellme web` | Launch the local web UI in your browser (needs the `web` extra). |

`interview`, `resume`, and `grade` also take `--provider`, `--fast-model`, and
`--smart-model` to override the model picks.

## Privacy and limits

Your transcripts and session state stay on your machine. The only things that leave it
are API calls to whichever provider you picked, plus plain HTTP GETs to public pages: one
for a `--job` link, and up to six more for the company brief. Those fetches are capped per
session, truncated per page, and refused for localhost and private addresses. Your API key
is read only by the code that calls the provider and never goes into a prompt, so no
hostile page or posting can talk the model into leaking it (`tests/test_fetch.py`,
`tests/test_secret_isolation.py`, `tests/test_injection.py`).

A cap on questions, a guaranteed closing question, a ceiling on web fetches, and a token
budget that ends a long session early are all plain code, and they're unit-tested. The
tool also screens what you type before it reaches the interview, so going off-topic nudges
you back and a second off-topic reply in a row wraps the session up. Either way the real
answers you gave still get graded.

## Configuration

There's no account and no server. Pick a provider with `SOTELLME_PROVIDER` (or
`--provider`, or the dropdown in the web app) and set its key:

| Provider       | Key variable        | Default models (fast / smart)             |
| -------------- | ------------------- | ----------------------------------------- |
| `google_genai` | `GOOGLE_API_KEY`    | gemini-3.5-flash / gemini-3.1-pro-preview |
| `anthropic`    | `ANTHROPIC_API_KEY` | claude-sonnet-4-6 / claude-opus-4-8       |
| `openai`       | `OPENAI_API_KEY`    | gpt-5.4-mini / gpt-5.5                    |

The fast slot runs the interview side (CV parser, company researcher, answer assessor,
interviewer); the smart slot runs the director that makes every probe-or-move-on call,
plus the end-of-session grader and coach. In the CLI you set those two slots with
`SOTELLME_FAST_MODEL` / `SOTELLME_SMART_MODEL` or the matching flags. The web app goes
finer: its Advanced section pins a model to each step on its own, so you can put a cheap
one on the company research and a stronger one on the questions and the grading, and mix
providers once you've set more than one key. The eval suites run against `google_genai`
with an `anthropic` judge, which is the combo I'd reach for.

Both draw their choices from the same catalog, which ships the per-provider defaults in
the table above. To change what's on offer, write a `~/.sotellme/models.toml` listing the
models you want and the default for each provider, and that's what the web app's dropdowns
show. The file holds model names plus the per-model prices behind the cost
estimates (including the reduced rate for cached input), so you can correct a rate that's
drifted; your API keys stay in the environment.

The session has a token budget, 400,000 by default, that ends the interview early if a run
goes long and keeps a reserved share back to grade and coach what you gave. Change it with
`SOTELLME_TOKEN_BUDGET`.

## Development

Requires Python 3.12+, managed with `uv`. The package takes its long description from the
repo's `README.md`, so stage that and the license into `backend/` once before the first
sync:

```sh
cd backend
python3 scripts/prepare_package.py
uv sync
uv run ruff check . && uv run mypy && uv run pytest
```

The deterministic suite runs without any API keys, and it's the whole CI gate.

The judgment agents (grader, coach, assessor, role builder, profile parser) are tuned
separately in Langfuse. Stand up a local instance, export `LANGFUSE_PUBLIC_KEY`,
`LANGFUSE_SECRET_KEY`, and `LANGFUSE_HOST`, then sync the committed cases and run one
agent over its dataset:

```sh
uv sync --extra tracing
uv run python scripts/evals.py upload
uv run python scripts/evals.py run grader --limit 2   # small calibration run first
uv run python scripts/evals.py run grader
```

Each run lands in Langfuse with a deterministic score per case, so you can read the
outputs, edit a prompt, run it again, and compare the two runs side by side. It also
prints the run's token count and estimated cost per model, priced from `models.toml`, so
you can size a full run from a `--limit` sample before committing to it. Only the
synthetic `evals/*.json` cases ever go in, and Langfuse stays off unless its env vars are
set, for evals and for live-session tracing alike.

The questions the system asks get their own eval. `scripts/simulate.py` runs a full
interview against a synthetic candidate: the real interviewer and director loop ask, while
a candidate-simulator answers in character from a persona under `evals/personas/`. The
personas span every level from junior to staff and a mix of answering styles, complete
STAR stories, thin answers, blurred ownership, off-topic drift, confident bluffing, and
injection attempts, so a run also exercises the guardrail and how the loop recovers. An
LLM judge on the smart slot scores each question on relevance, whether it probes the
flagged gap, level-appropriateness, whether it leads the candidate, and follow-up
discipline, plus a coverage verdict for the session.

```sh
uv run python scripts/simulate.py upload
uv run python scripts/simulate.py run --persona senior-strong --persona junior-thin
uv run python scripts/simulate.py run
```

Before a run it estimates the cost across the chosen personas and the judge passes and
asks first for anything over $3.50; pass `--yes` to skip the prompt in a script. Each
persona is a Langfuse dataset item tagged with its level and answer mix, so the
question-quality scores compare run to run and slice by both, and the session transcripts
land under `evals/sessions/`. The personas are synthetic, the same PII rule as everything
else.
