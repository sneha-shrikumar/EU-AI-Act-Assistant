# LangSmith Tracing — Detailed Technical Design

> Status: **Done 2026-07-12. Companion to the implementation plan.**

## 0. Primitives chosen

Plain-Python RAG (no LangChain), so use the low-level `langsmith` SDK:

- `@traceable` from `langsmith`:
  - `run_type="chain"` on `answer_question` → the parent trace per question.
  - `run_type="retriever"` on the retrieval function → retrieved chunks render as "documents".
  - `run_type="chain"` on `generate_answer` → clean parent boundary for the generation step.
- `wrap_openai` from `langsmith.wrappers` around the `OpenAI` client → auto-captures request
  messages (system + user prompt), model id, `response.usage` token counts, and latency for
  the LLM call, nested under the active `@traceable` parent.
- Cost: OpenRouter's `anthropic/claude-haiku-4.5` isn't in LangSmith's price registry, so
  auto-cost is $0. Request real cost via `extra_body={"usage": {"include": True}}` and surface
  it into the run.
- No-op when `LANGSMITH_TRACING` unset — `@traceable`, `wrap_openai`, and
  `get_current_run_tree()` are all inert.

Nesting is automatic via contextvars: `answer_question` calls the retriever and
`generate_answer` (which invokes the wrapped client → an llm run). No manual run-tree plumbing.

## 1. `requirements.txt`
Add `langsmith>=0.1.100` (floor pin, matching existing style). Install into the venv.

## 2. Environment / `.env` + `.env.example`
LangSmith reads these directly at trace time:
- `LANGSMITH_TRACING=true` — master switch; when absent, everything is a no-op.
- `LANGSMITH_API_KEY=ls__...`
- `LANGSMITH_PROJECT=eu-ai-act-rag`
- `LANGSMITH_ENDPOINT=https://api.smith.langchain.com` (US; EU would be
  `https://eu.api.smith.langchain.com`).

`config.py` already calls `load_dotenv()` at import, before any traced call executes — no
import reordering needed. `config.py` mirroring of these vars is cosmetic/optional.

## 3. `llm.py`
- Imports: `from langsmith import traceable, get_current_run_tree`.
- `_client()`: plain `OpenAI(base_url=..., api_key=...)` with the missing-key
  `RuntimeError` guard. (No `wrap_openai` — see the cost note below for why.)
- `_chat_completion(messages)` — a dedicated **`@traceable(run_type="llm", name="ChatOpenAI",
  metadata={"ls_provider": "openrouter", "ls_model_name": config.OPENROUTER_MODEL})`** wrapper
  that:
  1. Calls `_client().chat.completions.create(..., extra_body={"usage": {"include": True}})`.
  2. Sets `usage_metadata` (tokens + real cost) on **this** llm run:
     ```python
     rt = get_current_run_tree()
     if rt is not None and (usage := getattr(response, "usage", None)) is not None:
         um = {"input_tokens": usage.prompt_tokens, "output_tokens": usage.completion_tokens,
               "total_tokens": usage.total_tokens}
         cost = getattr(usage, "cost", None)   # OpenRouter-specific
         if cost is not None:
             um["total_cost"] = float(cost)
         rt.set(usage_metadata=um)
     ```
     `get_current_run_tree()` → `None` when tracing off (safe no-op).
- `generate_answer()`: `@traceable(run_type="chain", name="generate_answer")`; builds the
  messages and calls `_chat_completion(messages)` inside the existing try/except.

### Cost approach — why a custom llm run instead of `wrap_openai`
- `wrap_openai` renders the LLM call nicely but hardcodes a **token-only** usage extractor
  (`_create_usage_metadata`) with no hook to inject cost, and it closes the llm run before our
  code regains control — so we can't attach cost to it.
- Cost must live on the **llm run** (which already holds the tokens). Putting `usage_metadata`
  on the parent `generate_answer` run would **double-count tokens** (parent token totals are
  rolled up from the child).
- So we own the llm run via `@traceable(run_type="llm")` and set `usage_metadata` with
  `total_cost` = OpenRouter's authoritative `usage.cost`. This populates LangSmith's **native**
  token *and* Cost dashboards. Verified live: new llm runs carry `total_cost` (e.g. 0.00365409);
  runs created before the fix stay `None`.
- Alternative (not used): a LangSmith model price map for `anthropic/claude-haiku-4.5` — static
  rates to maintain, and wrong if OpenRouter reroutes to a different-priced provider.

## 4. `query.py`
- Import: `from langsmith import traceable`.
- `_retrieve()` — retriever run with output adapter (keeps the internal
  `{text, citation, similarity, metadata}` return shape that callers depend on):
  ```python
  def _as_documents(outputs):
      # langsmith 0.10.x passes process_outputs the RAW return value (a list here);
      # some versions wrap a non-dict return as {"output": [...]}. Accept both.
      chunks = outputs.get("output", []) if isinstance(outputs, dict) else outputs
      return {"documents": [
          {"page_content": c["text"], "type": "Document",
           "metadata": {**c["metadata"], "similarity": c["similarity"]}}
          for c in chunks
      ]}

  @traceable(run_type="retriever", name="retrieve", process_outputs=_as_documents)
  def _retrieve(question, k):
      ...  # unchanged
  ```
  `process_outputs` only affects logged output. Retriever input `{question, k}` captured
  automatically. Optional nice-to-have: decorate `embed_query` as `run_type="embedding"` to
  break out embedding latency (out of scope).
- `answer_question()` — `@traceable(run_type="chain", name="answer_question")`. Parent trace;
  children form automatically. No REPL/`__main__` changes.

## 5. Gotchas
- **Broad `try/except` in `generate_answer`** swallows everything into `{"error": str(exc)}`.
  Keep cost/trace code strictly defensive (`getattr`/`is not None`) so it can never turn a good
  response into a fake error. Failed LLM calls show as "success with an error payload in output",
  not errored runs — acceptable; do NOT narrow the except (it would break the "never crash the
  eval loop" contract).
- **Retriever rendering** needs `{page_content, metadata}` — handled by `process_outputs`,
  leaving `_retrieve`'s contract intact.
- **`get_current_run_tree()`** only returns non-None inside a traced function with tracing on —
  hence the LLM call lives in the decorated `_chat_completion` so cost lands on the llm run.
- **OpenRouter `usage.cost`** is non-standard, only present with the `extra_body` flag — guard
  with `getattr`.
- **Model not in price registry** → LangSmith's *computed* cost is null; we override it by
  writing `total_cost` into the llm run's `usage_metadata`, which the native Cost dashboard reads.
- **`SYSTEM_PROMPT`** module-level f-string is unaffected by decoration; the `messages` passed to
  `_chat_completion` become the llm run's inputs, so the full prompt appears automatically — no
  manual logging.
- **Wrong `LANGSMITH_ENDPOINT`** sends traces to the other region and they "won't appear" — we
  use US default deliberately.
- **Refusal gates:** gate-1 (pre-LLM) traces show a retriever child and no generation child;
  gate-2 (model refusal) produces a full trace. Both render naturally.

## 6. Verification
1. No-op: `LANGSMITH_TRACING` unset → identical behavior, no network calls.
2. On: set the four vars; ask one in-corpus + one out-of-corpus question.
3. LangSmith UI (US): parent `answer_question`; `retrieve` child rendering chunk text +
   `similarity`; `generate_answer` → `ChatOpenAI` llm run with system+user messages, model id,
   prompt/completion/total tokens, latency, and native `total_cost`.
4. Out-of-corpus → retriever child, no generation child.
5. Error path: bad key → generation run still records, pipeline returns error dict, no crash.
6. Confirm `response.usage.cost` is actually populated (inspect one response); if omitted, the
   guard leaves cost unset rather than erroring.

## Critical files
- `eu-ai-act-rag/llm.py`
- `eu-ai-act-rag/query.py`
- `eu-ai-act-rag/config.py` (optional)
- `eu-ai-act-rag/requirements.txt`
- `eu-ai-act-rag/.env` and `eu-ai-act-rag/.env.example`
