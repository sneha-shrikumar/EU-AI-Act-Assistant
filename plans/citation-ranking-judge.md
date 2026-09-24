# Citation Ranking (Top-3) Judge for EU AI Act RAG

> Status: **Done 2026-08-13. Implemented as `eval_citation_ranking.py`.**

## Context

`eu-ai-act-rag/eval_retrieval.py` already judges retrieval **recall** (does the
full set of expected article/annex/recital refs appear anywhere among the
tool's citations). The user now wants a second, separate code-as-judge that
checks retrieval **ranking**: for `should_answer=yes` questions, does at least
one of the expected refs show up within the **top 3** citations the pipeline
returned (citations are already an ordered list, see `evals/run_langsmith_eval.py`
`run_pipeline` → `result["citations"]`). This is a quality signal beyond plain
recall — even if the right chunk was retrieved, it should be judged worse if it
wasn't ranked near the top.

**User-confirmed scoring rule:**
- Applies **only** to rows where `should_answer == "yes"`. Rows with
  `should_answer == "no"` are left blank/N/A — this judge does not score
  refusal behavior (that's `eval_retrieval.py`'s job).
- Score = 1 if **at least one** parsed expected ref is present among the refs
  parsed from `citations[:3]`; else 0.
- Rows with no parseable `expected_source` are N/A (consistent with
  `eval_retrieval.py`'s existing convention).

## Design

New file: **`eu-ai-act-rag/eval_citation_ranking.py`**, sitting alongside
`eval_retrieval.py` and reusing its already-refactored pieces instead of
duplicating LangSmith/parsing logic:

```python
from eval_retrieval import parse_refs, fetch_experiment, DATASET_NAME
```

- `parse_refs` — already handles both the strict citation format and the loose
  golden `expected_source` format (kept as-is, no changes needed).
- `fetch_experiment(client, dataset_name, experiment_name)` — already sources
  purely from the LangSmith experiment's own runs + their reference outputs
  (no CSV, no full-dataset scan). Reused unchanged.

### New scoring function

```python
def score_ranking_row(expected_refs, citations, should_answer, expected_source):
    """Returns (score, note); score is 1, 0, or None for N/A."""
    if should_answer != "yes":
        return None, "N/A -- ranking judge only scores should_answer=yes"
    if not expected_refs:
        note = "N/A -- no article/annex/recital in expected source" if expected_source.strip() \
            else "N/A -- no expected source recorded"
        return None, note

    top3 = citations[:3]
    top3_refs = set()
    for c in top3:
        top3_refs |= parse_refs(c)

    if expected_refs & top3_refs:
        return 1, ""
    return 0, "no expected ref in top 3"
```

This mirrors `eval_retrieval.py`'s `score_row` shape/conventions but is a
distinct function (different rule — "any expected ref in top-3 window" vs.
"all expected refs anywhere") so the two judges stay independently readable.

### `evaluate()` loop

Same shape as `eval_retrieval.py::evaluate`:
```python
client = Client()
experiment, rows = fetch_experiment(client, dataset_name, experiment_name)

results = []
for run, reference in rows:
    question = (run.inputs or {}).get("question", "")
    expected_source = reference.get("expected_source") or ""
    should_answer = (reference.get("should_answer") or "yes").strip().lower()
    citations = (run.outputs or {}).get("citations") or []

    expected_refs = parse_refs(expected_source)
    score, note = score_ranking_row(expected_refs, citations, should_answer, expected_source)
    results.append({
        "question": question, "expected_source": expected_source,
        "top3_citations": citations[:3], "all_citations": citations,
        "should_answer": should_answer, "score": score, "note": note,
    })
```

### Excel output

Dedicated `write_excel` in the new file (columns differ enough from
`eval_retrieval.py`'s to not force a shared abstraction):

| question (input) | expected_source | top-3 citations (output) | should_answer | score | note |

Same visual conventions as the existing evaluator (bold header, wrap text,
red fill on score=0), plus a summary block: examples, scored (should_answer=yes
count), hits, hit rate. Sheet title: `"citation ranking"`. Default output path:
`eval_ranking_<experiment-name>.xlsx`.

### CLI

Mirrors `eval_retrieval.py`'s argparse setup exactly:
```
python eval_citation_ranking.py                    # latest experiment
python eval_citation_ranking.py --experiment NAME
python eval_citation_ranking.py --dataset "..." --out results.xlsx
```

No `--selftest` needed — `parse_refs` is already covered by
`eval_retrieval.py --selftest`, and `score_ranking_row` is a thin wrapper with
no new parsing logic to unit-test.

## Files touched

- **New:** `eu-ai-act-rag/eval_citation_ranking.py`
- **New (per CLAUDE.md):** save this approved plan into `plans/` (e.g.
  `plans/citation-ranking-judge.md`) as the first implementation step, before
  any code is written.

## Verification

1. `python -m py_compile eval_citation_ranking.py` — syntax/import sanity
   check (confirms the import from `eval_retrieval` resolves).
2. Run `python eval_citation_ranking.py` against the real experiment; confirm
   the xlsx opens, `should_answer=no` rows are blank/N/A, and spot-check a
   handful of `should_answer=yes` rows by hand (does the expected ref actually
   sit in positions 1–3 of the citations list for score=1 rows, and truly
   absent from the top 3 for score=0 rows).
