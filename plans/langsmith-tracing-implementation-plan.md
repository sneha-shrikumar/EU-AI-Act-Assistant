# Add LangSmith Tracing to the EU AI Act RAG Pipeline

> Status: **Done 2026-07-12.**

## Context

The RAG pipeline in `eu-ai-act-rag/` had **no observability** — only `print()`
statements. The goal is to see, per question, exactly what the pipeline did:
which chunks were retrieved, what prompt was sent to the model, and the latency,
token counts, and cost of each step. The two stages (retrieval, generation) appear
as **separate, nested spans** under one trace per question.

Key constraint: this is **plain-Python RAG, not LangChain** (raw `openai` SDK →
OpenRouter + `chromadb` + `sentence-transformers`). So there is no LangChain callback
to piggyback on — we use the standalone `langsmith` SDK (`@traceable` decorators +
`wrap_openai`). All spans nest automatically via the call stack (contextvars).

Locked decisions:
- **Cost** → OpenRouter's real charged cost (the model id `anthropic/claude-haiku-4.5`
  isn't in LangSmith's price registry, so its auto-cost would be $0). Requested via
  `extra_body={"usage": {"include": True}}` and surfaced into the generation run.
- **Data region** → US default endpoint (`https://api.smith.langchain.com`). User is in
  Singapore with no EU data-residency requirement, and LangSmith has no APAC region.

Everything is a **no-op when `LANGSMITH_TRACING` is unset**, so the pipeline still runs
without a LangSmith key.

## Target trace shape (per question)

```
answer_question            (chain)   inputs: {question, k}
├── retrieve               (retriever) output renders as documents (chunk text + similarity)
└── generate_answer        (chain)
    └── chat.completions    (llm)     inputs: system+user messages, model; usage: tokens; cost
```
An out-of-corpus question (refusal gate 1) shows a `retrieve` child and **no** generation
child — a useful, correct signal.

## Changes

1. **`requirements.txt`** — add `langsmith>=0.1.100`.
2. **`.env` / `.env.example`** — add `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`,
   `LANGSMITH_PROJECT`, `LANGSMITH_ENDPOINT`. `config.py` already calls `load_dotenv()`.
3. **`llm.py`** — a dedicated `@traceable(run_type="llm", name="ChatOpenAI")`
   wrapper (`_chat_completion`) makes the single LLM call with
   `extra_body={"usage":{"include":True}}`, then sets `usage_metadata` on that llm run
   (tokens **and** `total_cost` = OpenRouter's real `usage.cost`) via
   `get_current_run_tree().set(...)`. `generate_answer` stays a `chain` run that calls it.
   (We do **not** use `wrap_openai`: it hardcodes a token-only usage extractor with no
   hook for cost, and cost must land on the llm run — putting it on the parent would
   double-count tokens.)
4. **`query.py`** — decorate `_retrieve` as a `retriever` run with a `process_outputs`
   adapter (renders chunks as documents without changing the return value); decorate
   `answer_question` as the parent `chain` run.

## Verification

1. No-op path: `LANGSMITH_TRACING` unset → identical behavior, no LangSmith calls.
2. Tracing on: set the four vars, ask one in-corpus and one out-of-corpus question.
3. LangSmith UI (US region): confirm the nested tree, retrieved chunks + similarity,
   full system+user prompt, model id, token counts, latency, and native `total_cost` on
   the `ChatOpenAI` llm run.
4. Out-of-corpus → retriever child, no generation child.

See `langsmith-tracing-technical-design.md` for the function-by-function design and gotchas.
