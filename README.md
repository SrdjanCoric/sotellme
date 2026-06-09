# sotellme

A voice-based behavioral interview simulator and coach. You upload a CV and point it at a job
posting; it researches the company, runs an adaptive mock interview grounded in your actual
experience, then grades each answer against a STAR rubric and produces specific coaching feedback
with drills for weak competencies.

The interview adapts per answer: vague or unquantified answers get follow-up probes, complete
stories advance to the next competency, and follow-ups are capped per story. Question routing is
decided by a pure, LLM-free function (`nextAction(state) -> Probe | NextCompetency | Stop`), so
the adaptive behavior is unit-tested rather than prompted. The core is text-in/text-out; the
voice layer (streaming STT/TTS with barge-in) is a swappable adapter on top.

## Architecture

- `backend/` — Python. The engine is a callable library orchestrated with LangGraph (checkpointed
  state, `interrupt()` at human turns), served through FastAPI. Pydantic models at every boundary.
- `frontend/` — React (Vite, TypeScript, Tailwind, TanStack Query).
- Observability and evals run on self-hosted Langfuse. Deterministic modules (coverage logic,
  guardrails, validation) are tested with pytest; LLM-judgment modules are evaluated with
  LLM-as-judge datasets, including a comparison against single-prompt and fixed-question-list
  baselines. Eval results are published in `evals/RESULTS.md` once the harness phase lands.

## Knowledge base

The agents are grounded in two local folders that are not part of the repo:

- `how-to-interview/` — read by the interviewer and orchestrator
- `how-to-answer/` — read by the grader and coach

The folder split is a firewall: the interviewer has no read path to the grading rubric. Tests run
against small fixture knowledge bases committed under `tests/fixtures/`, so the suite and CI work
without the real content. Running real sessions requires supplying your own two folders. CVs are
handled the same way: tests use a synthetic fixture CV, and no real personal data is committed.

## Development

Requires Python 3.12+ (managed with `uv`) and Node 20+.

```sh
# backend
cd backend
uv sync
uv run pytest

# frontend
cd frontend
npm install
npm run dev
```

Copy `.env.example` to `.env` for API keys and Langfuse credentials; secrets are read only from
the environment and never enter prompts.

## Status

Early development.
