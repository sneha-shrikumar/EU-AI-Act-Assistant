# Retrieval Recall Evaluator for EU AI Act RAG

> Status: **Done 2026-08-23. Implemented as `eval_retrieval.py`.**

## Context

The RAG pipeline (`eu-ai-act-rag/`) answers EU AI Act questions and traces every run to LangSmith. A first experiment was run against the LangSmith dataset **"EU AI Acts evaluation"**, and manual error analysis showed retrieval recall failure is the dominant failure mode (~47% of flagged issues). The user wants a repeatable, automated evaluator that checks — for every question in an experiment — whether the retrieved chunks' citations cover at least one of the article/annex/recital numbers listed in the golden `expected_source`, and writes the results to an Excel sheet.

**Data source (user-confirmed):** fetch outputs from the LangSmith experiment via the API (not re-running the pipeline, not parsing a CSV export).

## Verified LangSmith structures (inspected live)

- Dataset `EU AI Acts evaluation` — examples:
  - `inputs`: `{"question": ...}`
  - `outputs` (reference): `{"should_answer": "yes"/"no", "expected_answer": ..., "expected_source": "Recital 24, 25, article 2"}`
- Experiment = a project whose `reference_dataset_id` is the dataset (current one: `eu-ai-act-rag-9ed05d77`). Root runs have:
  - `run.inputs`: `{"question": ...}`
  - `run.outputs`: `{"answer", "citations": ["Article 3", "Article 2(1)", "Recital 1", ...], "refused": bool, "error"}`
  - `run.reference_example_id` → links run to its dataset example.

## Citation formats to reconcile

- Chunk citations (tool output): `Article 6(2)`, `Recital 26`, `Annex III(2)` — **annexes in Roman numerals**, optional `(paragraph)` suffix.
- `expected_source` (golden, free-form): `"article 10(2), 10(5)"`, `"Article 6, annex 3"`, `"Recital 24, 25, article 2"`, `"Article 9-15, 16"`, `"Section 4, article 28"`, typo `"artcile 97"` — **annexes in Arabic numerals**, bare continuation numbers, ranges, typos.

## New file: `eu-ai-act-rag/eval_retrieval.py`

Single self-contained script (mirrors the style of `query.py`), run as:

```
python eval_retrieval.py                # scores the latest experiment on the dataset
python eval_retrieval.py --experiment eu-ai-act-rag-9ed05d77
python eval_retrieval.py --dataset "EU AI Acts evaluation" --out results.xlsx
```

### 1. Reference parsing (`parse_refs(text) -> set[tuple[str, int]]`)

Normalize both `expected_source` and each citation string into a set of `(kind, number)` pairs, where kind ∈ {`article`, `annex`, `recital`}:

- Tolerant keyword regex: `art\w*` → article (covers "artcile" typo), `annex\w*`, `recital\w*`. Ignore "Section N" (not a chunk citation type; the paired article number still matches).
- After a keyword, consume a comma-separated run of numbers so bare continuations inherit the kind: `"Recital 24, 25, article 2"` → {(recital,24), (recital,25), (article,2)}.
- Ranges: `9-15` → 9..15 inclusive.
- Strip paragraph/point suffixes: `10(2)` → 10; `13(a)` → 13 (article-level match per user requirement).
- Roman numerals → Arabic for annexes (`Annex III(2)` → (annex, 3)).

### 2. Scoring per example

- `should_answer == "yes"` and `expected_source` parses to ≥1 ref: **score = 1 if the parsed expected refs ∩ parsed citation refs is non-empty, else 0** (at least one expected article/annex/recital appears among retrieved-chunk citations, paragraph ignored).
- `should_answer == "no"` (out-of-scope): **score = 1 if `refused` is True, else 0** (user-confirmed), with a note "scored on refusal".
- `should_answer == "yes"` but `expected_source` empty/unparseable: score = blank, note "N/A — no expected source".

### 3. LangSmith fetch

Using `langsmith.Client` (env loaded via `dotenv`, same as the pipeline):
1. `read_dataset(dataset_name)` → `list_projects(reference_dataset_id=ds.id)`; pick `--experiment` by name or the most recent by start time.
2. `list_runs(project_id=..., is_root=True)`; join each run to its example via `reference_example_id`.
3. Order rows to match the dataset example order.

### 4. Excel output (openpyxl)

`eval_results_<experiment-name>.xlsx` in `eu-ai-act-rag/`, one row per example:

| question (input) | expected_source | citations (output) | answer | refused | score (1/0) | note |

Plus a summary block: examples evaluated, recall hits/total, recall %. Basic formatting only (bold header, red fill on score=0 rows).

### 5. Dependency

Add `openpyxl` to `requirements.txt` and `pip install` it into the venv (not currently installed; pandas is unnecessary).

## Files touched

- **New:** `eu-ai-act-rag/eval_retrieval.py`
- **Edit:** `eu-ai-act-rag/requirements.txt` (add `openpyxl`)
- **New (per CLAUDE.md):** save this plan into the `plans` folder as the first implementation step.

## Verification

1. Unit-check the parser inline (a `--selftest` flag or simple asserts) against the tricky golden strings: `"article 10(2), 10(5)"`, `"Recital 24, 25, article 2"`, `"Article 9-15, 16"`, `"Annex III(2)"` vs `"annex 3"`, `"artcile 97"`, `"13(a)"` vs `"article 13"`.
2. Run `python eval_retrieval.py` against the real experiment `eu-ai-act-rag-9ed05d77`; confirm the xlsx opens, has one row per dataset example, and spot-check ~5 rows against the manual axial-code analysis (e.g. the known "Annex 11 vs XI" recall failures should score 0, well-retrieved single-hop questions should score 1).
3. Confirm the 3 out-of-scope questions are scored on refusal.
