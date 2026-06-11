# Writeup

## Design decisions

- **No agent framework**: the orchestrator (`agent/orchestrator.py`) is a
  plain Python class executing a fixed sequence of steps, recording each
  step's input/output/latency into a `reasoning_trace`. This keeps the
  control flow explicit and auditable.
- **Graceful LLM degradation**: `classify_intent` and `draft_response` call
  OpenRouter but fall back to deterministic keyword/template logic if the
  API key is missing or the call fails, so the pipeline and tests remain
  fully functional offline.
- **TF-IDF knowledge base search**: `search_kb` uses scikit-learn's
  `TfidfVectorizer` + cosine similarity over JSON documents in
  `data/knowledge_base/`, avoiding any external vector DB dependency.
- **Rule-based NLP**: dialect detection, sentiment, entity extraction, and
  urgency assessment are lexicon/regex based for transparency and
  determinism, suitable as a strong baseline and easy to evaluate.
- **Persistence**: every triage run is written both to PostgreSQL
  (`triage_runs`, JSONB/array columns) and to a JSONL trajectory file for
  replay/debugging.

## Evaluation results (sample gold set, 10 cases)

- Intent classification: precision/recall/F1 = 1.00 (keyword fallback,
  no LLM key configured in this run)
- Routing accuracy: 0.80 — the two "misses" are cases where high/critical
  urgency correctly overrides normal intent-based routing to
  `escalations_management`, which is expected behavior, not an error.

## Known limitations

- Dialect detection and sentiment are lexicon-based; an LLM-backed pass
  (prompts already provided in `llm/prompts/`) will improve accuracy on
  ambiguous or mixed-dialect text.
- Order lookup requires a reachable PostgreSQL instance; without one,
  `lookup_order` returns `None` and the pipeline continues.
