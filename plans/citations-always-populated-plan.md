# Always populate `citations` from retrieved chunks, regardless of refusal

> Status: **Done 2026-08-13.**

## Context

This is a bug in the RAG tool's own output, not an eval-scoring issue.
`answer_question()` in `query.py` is the tool's public entry point — every
caller (the LangSmith runner, a future API, direct use) gets back
`{answer, refused, citations, retrieved_chunks, error}`. The user noticed
that whenever `refused=True`, `citations` comes back `[]` even when chunks
were clearly retrieved and are referenced in the `answer` prose itself, e.g.:

```
{"answer":"I cannot find an answer to this in the EU AI Act excerpts provided.\n\n
The excerpts detail specific prohibited AI practices in Article 5(1) ...",
 "citations":[], "error":null, "refused":true}
```

That's a real defect: the tool retrieved and used specific excerpts (Article
5(1) here) to write its explanation, but the structured `citations` field —
the machine-readable record of which sources were consulted — silently
drops them whenever the answer is a refusal. Anyone consuming `citations`
programmatically (rather than parsing the prose) sees "no sources" on every
refusal, which is wrong: sources were retrieved and used regardless of
whether the model decided to answer or refuse.

Root cause (confirmed by reading `query.py:72-122`): `citations` is not
parsed from the model's answer text — it's a straight derivation from
`retrieved_chunks[i]["citation"]` (itself copied from Chroma metadata at
retrieval time, `query.py:65`). It is deliberately zeroed out in three of the
function's four return branches:

- **Gate 1** (`query.py:87-95`, similarity-threshold refusal): hardcoded
  `"citations": []` literal — never reads `retrieved_chunks` at all, even
  though `retrieved_chunks` itself is populated and returned.
- **Error branch** (`query.py:99-106`, LLM call raised): hardcoded
  `"citations": []`.
- **Gate 2** (`query.py:108-114`, model self-refused): `citations = [] if
  refused else [c["citation"] for c in retrieved_chunks]` — the correct
  list-comprehension is right there, just short-circuited by the ternary.

Fix: compute the citation list once, from `retrieved_chunks`, unconditionally
— and use it in all four return branches instead of hardcoding/gating `[]`.
`retrieved_chunks` is already documented as "always populated, even on
refusal" (`query.py:81`); `citations` should mirror that.

## Change

**File:** `eu-ai-act-rag/query.py`, inside `answer_question()` (lines
72-122).

1. Right after `retrieved_chunks = _retrieve(question, k)` (line 85), add:
   ```python
   citations = [c["citation"] for c in retrieved_chunks]
   ```
2. Replace the hardcoded `"citations": []` in the gate-1 return (line 92)
   with `"citations": citations,`.
3. Replace the hardcoded `"citations": []` in the error-branch return
   (line 103) with `"citations": citations,`.
4. In gate 2, delete the conditional line `citations = [] if refused else
   [c["citation"] for c in retrieved_chunks]` (line 114) — `citations` is
   already computed and correct from step 1; just use it directly in the
   final return dict.
5. Update the docstring at `query.py:80` (`"citations": list[str],`) to note
   it's always populated from retrieved chunks, same as `retrieved_chunks`,
   independent of `refused`/`error`.

No other files change. This is purely a `query.py` fix to the tool's return
value. (Downstream consumers of the top-level `citations` field —
`evals/run_langsmith_eval.py`, which passes it through untouched, and
`eval_retrieval.py`, which scores against it — will incidentally see more
accurate data after this fix, but that's a side effect, not the goal.
`evals/run_evals.py` derives its own per-chunk citation directly from
`retrieved_chunks` metadata and never reads the top-level `citations` field
at all, so it's unaffected.)

## Verification

1. Run `python -c "from query import answer_question; import json;
   print(json.dumps(answer_question('...'), indent=2))"` (from
   `eu-ai-act-rag/`, venv active) on the entertainment/children question from
   the user's example (or any question known to trigger a gate-2 refusal) and
   confirm `citations` is now non-empty and lists the articles the answer
   text actually discusses (e.g. `["Article 5(1)"]`).
2. Also test a gate-1 refusal (a question with no topically-close chunks,
   similarity below `config.SIMILARITY_THRESHOLD`) and confirm `citations`
   reflects whatever low-similarity chunks were still retrieved, since
   `retrieved_chunks` is populated even there.
3. Run `evals/run_evals.py --limit 5` as a smoke test to confirm nothing else
   broke (it doesn't consume top-level `citations`, so this is a regression
   check only).
