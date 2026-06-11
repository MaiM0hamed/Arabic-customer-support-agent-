# Arabic Customer Support Agent

A raw tool-calling triage agent for Arabic-language e-commerce customer
support, built with FastAPI, SQLAlchemy/PostgreSQL, and the OpenRouter API
(Qwen / DeepSeek / Llama). No agent frameworks (LangGraph, LlamaIndex,
CrewAI, AutoGen, SmolAgents) are used.

## Architecture

```
agent/
  orchestrator.py   # sequential tool-calling pipeline
  models.py         # pydantic request/response models
  nlp/              # sanitizer, dialect detector, sentiment, entities, urgency
  tools/            # classify_intent, lookup_order, search_kb,
                     # draft_response, route_team, escalate
api/                # FastAPI app, routes, schemas
database/           # SQLAlchemy models, postgres session, schema.sql
llm/                # OpenRouter client + prompt templates
observability/      # JSON logging, JSONL trajectories, DB persistence
evaluation/         # metrics, routing accuracy, LLM judge, error analysis
scripts/            # seed_db, sample_dataset, run_demo
data/               # orders, knowledge base, taxonomies, test sets
tests/              # pytest unit/integration tests
```

## Setup (WSL Ubuntu + Conda)

```bash
conda create -n arabic-agent python=3.11 -y
conda activate arabic-agent
cd /mnt/c/Users/FreeComp/Arabic-customer-support-agent-
pip install -r requirements.txt
cp .env.example .env   # then fill in OPENROUTER_API_KEY and DATABASE_URL
```

## Database

Requires a running PostgreSQL instance matching `DATABASE_URL`.

```bash
python -m scripts.seed_db
```

This creates the `orders` and `triage_runs` tables (see
`database/schema.sql`) and seeds sample orders from `data/orders.json`.

## Running the API

```bash
uvicorn api.main:app --reload
```

Endpoints:

- `POST /triage` — run the full triage pipeline on a customer message
- `GET /health` — health check
- `GET /triage/{run_id}` — fetch a persisted triage run
- `GET /runs` — list recent triage runs

## Pipeline

```
message -> sanitizer -> dialect detection -> entity extraction -> sentiment
        -> classify intent -> lookup order -> search kb -> assess urgency
        -> route team -> draft response -> optional escalation -> triage record
```

Every run produces a JSONL trajectory under `logs/trajectories/{run_id}.jsonl`
and a row in `triage_runs`.

## Evaluation

```bash
python -m scripts.sample_dataset   # builds gold_test.csv from labeled_test_set.json
python -m evaluation.run_eval      # computes precision/recall/F1 + routing accuracy
```

Reports are written to `logs/evaluations/`.

## Tests

```bash
pytest -q
```
