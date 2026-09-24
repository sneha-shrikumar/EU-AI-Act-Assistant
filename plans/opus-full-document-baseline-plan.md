# Plan: RAG vs. full-document Claude Opus 5.5 (with no retrieval)

> Status: **Done 2026-09-24. Re-run on the updated golden set: `docs/analysis/fulldoc_rerun_f0d6b8ee.md`.**

## Context
The user wants to know whether the EU AI Act RAG pipeline (bge-small retrieval, top-5 chunks, Haiku 4.5 as the generator) beats a "vanilla" setup, meaning a frontier model given the **entire Act** in its context and no retrieval. The baseline must run through **Claude Opus 5.5 on OpenRouter** as fresh, stateless API calls, not through this Claude Code session. Both systems answer the same 41-question golden dataset and are graded by the same three Sonnet 5 judges. The results go on a shareable comparison page.

## Approach

### 1. Save the plan (first build step, per CLAUDE.md)
Copy this plan to `eu-ai-act-rag/plans/opus-full-document-baseline-plan.md` before writing any code.

### 2. `config.py`: baseline knobs
- `BASELINE_MODEL = "anthropic/claude-opus-5.5"`. At the start of the build, check this ID against OpenRouter's `GET /api/v1/models`. If Opus 5.5 isn't listed there, stop and ask the user before switching to an equivalent model (e.g. `anthropic/claude-opus-5`).
- `BASELINE_EFFORT = "medium"`. Set it explicitly because Opus 5.5 always thinks, and effort is the only control over how much.

### 3. New `baseline_full_doc.py` (sibling of `query.py`)
- `load_full_text()`: reuses `ingest.extract_pdf_text(config.PDF_PATH)`, the same header/footer-stripped text the RAG pipeline indexes, and caches it with `lru_cache`. Log the token count once, from the first response's `usage.prompt_tokens`.
- `answer_question_full_doc(question) -> dict` returns the **same shape** as `query.answer_question` (`answer`, `refused`, `citations`, `retrieved_chunks`, `error`), so the eval harness and judges work unchanged:
  - It **reuses `llm.SYSTEM_PROMPT`** (cite-or-refuse), so the only differences between the two systems are retrieval vs. full document and the generator model.
  - Messages: the system prompt, then a user message whose content blocks are `[full Act text + cache_control ephemeral]` followed by `Question: ...`. The question comes after the cache breakpoint, so all 41 calls share one cached prefix.
  - Uses `llm._client()` and `llm._set_llm_usage()` so OpenRouter's real cost lands in LangSmith. It is `@traceable(run_type="llm", ls_model_name=BASELINE_MODEL)`.
  - **No `temperature`**, because Opus 5.5 rejects sampling parameters. The call sends `extra_body={"usage": {"include": True}, "reasoning": {"effort": BASELINE_EFFORT}}` and `max_tokens=16000`.
  - `refused` uses the same tolerant match as `query.py`. `citations` holds the provisions the answer cites inline, pulled out with a regex (e.g. `Article 6(2)`, `Recital 26`, `Annex III`).
  - `retrieved_chunks = [{"citation": "EU AI Act (full text)", "text": full_text}]`, so groundedness is judged against the whole document.

### 4. `evals/run_langsmith_eval.py`: add a `--system {rag,full_doc}` flag
- Selects `answer_question` or `answer_question_full_doc` as the target. The default stays `rag`, so current behaviour is unchanged.
- `experiment_prefix`: `eu-ai-act-rag` or `eu-ai-act-opus55-fulldoc`. Metadata records `system` and the generator model.
- `full_doc` forces `max_concurrency=1` and warms the cache with the first question before the rest, so later calls read the cache instead of all writing it in parallel.
- After `evaluate(...)`, it writes a per-example JSON to `evals/results/<experiment>.json`: id, category, question, answer, refused, citations, the three scores plus judge comments, latency, and cost. The report is built from these files.

### 5. `evals/judges.py`: make full-document groundedness affordable
Without caching, groundedness on a roughly 190k-token context costs about $16 per run on Sonnet 5. In `groundedness`, split the user prompt into two content blocks: the excerpts, then the answer. Put `cache_control` on the excerpts block **only when it is large** (e.g. over 20k characters). RAG runs are unaffected (their excerpts are about 2.5k tokens, so they're never cached), and the full-document runs reuse one cached prefix. The prompt text doesn't change, so scores stay comparable with earlier experiments.

### 6. Run both arms fresh, on the same judges and the same day
```
python evals/run_langsmith_eval.py --system full_doc --limit 3   # smoke test: cache hits, cost, parseable judges
python evals/run_langsmith_eval.py --system rag
python evals/run_langsmith_eval.py --system full_doc
```
Estimated cost of the full-document arm, assuming about 190k input tokens (Opus 5.5 is $4 in / $20 out per million tokens; cache writes about $5, cache reads $0.20):
- One cache write: about $1.
- 40 cache reads: about $1.50.
- Output and thinking: about $1–2.

That's **about $3–5 for Opus**, plus about $2 for cached judging. The RAG arm costs under $1. If caching fails, the Opus arm rises to about $31. The smoke test checks `cache_read_input_tokens` before the full run starts.

### 7. Showcase: new `evals/compare_report.py` and a published artifact
- Reads the two results JSONs and writes one self-contained HTML page, published with the Artifact tool. It loads `artifact-design` and `dataviz` first. The page contains:
  - **Headline tiles:** groundedness, relevance and correctness (x/41) for each system, plus total cost, cost per question, and median latency.
  - **Correctness by category:** a grouped bar chart covering single, multi-hop, exemption, and the other categories.
  - **Refusal behaviour:** correct refusals on the `should_answer=no` rows, and false refusals on answerable rows.
  - **Per-question table:** the question, both answers (collapsible), the scores side by side, and a filter for "questions where the two systems disagree".
  - **Caveats box:**
    - The generators differ (Haiku 4.5 vs. Opus 5.5), so the gap mixes the effect of retrieval with the effect of model size.
    - Opus 5.5 may know the Act from its training data, although the prompt restricts it to the text provided.
    - The judge is Sonnet 5, a different model from both generators.

## Critical files
- New: `baseline_full_doc.py`, `evals/compare_report.py`, `plans/opus-full-document-baseline-plan.md`
- Modified: `config.py`, `evals/run_langsmith_eval.py`, `evals/judges.py` (groundedness caching only)
- Reused unchanged: `ingest.extract_pdf_text`, `llm.SYSTEM_PROMPT`, `llm._client`, `llm._set_llm_usage`, `judges.ALL_JUDGES`, the LangSmith dataset "EU AI Acts evaluation"

## Verification
1. Run `python -c "import baseline_full_doc as b; print(b.answer_question_full_doc('what is an AI system'))"`. Expect an answer that cites Article 3.
2. Run the `--limit 3` smoke test. In LangSmith, check for:
   - cache reads greater than 0 on calls 2 and 3;
   - real cost populated;
   - all 9 judge scores non-null.
3. Run both full experiments and confirm there are 41 rows each with no `error` values. Any `score=None` rows are counted and shown on the report, not silently dropped.
4. Open the published artifact and check it in light and dark mode and at phone width.
