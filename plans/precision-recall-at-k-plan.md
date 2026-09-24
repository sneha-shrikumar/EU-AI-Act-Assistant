# Add fractional precision@k and recall@k to the retrieval evaluator

> Status: **Done 2026-09-23.**

## Context

The eval harness currently reports two recall-flavoured numbers and no precision at all:

- `eval_retrieval.py` — **strict, all-or-nothing recall@5**: scores 1 only if *every*
  expected reference appears among the citations (`expected_refs <= citation_refs`).
  A question needing Annex III + Article 9 that retrieves only Article 9 scores 0,
  not 0.5. Latest run: 23/40 = 58%.
- `eval_citation_ranking.py` — **hit rate@3**: 1 if *any* expected ref is in the top 3.
  Despite the name, this is not precision. Latest run: 23/36 = 64%.

Nothing measures precision, so nothing penalises retrieving junk. That matters
concretely: the obvious next experiment is raising `TOP_K` from 5, and under the
current metrics that can only look like an improvement — recall is monotonic in k.
Without precision there is no way to see the cost (more irrelevant chunks in the
prompt, more tokens, more material for the LLM to miscite).

**Goal:** report fractional precision@k and recall@k at k = 1, 3, 5 per question,
plus macro-averages, so the tradeoff is visible.

This reads the already-finished LangSmith experiment. **No pipeline re-run, no LLM
calls, no cost.**

## Approved decisions

| Decision | Choice |
|---|---|
| Location | Extend `eval_retrieval.py` and its existing Excel output. No third script. |
| Metrics | P@k and R@k at k = 1, 3, 5, plus macro-averages. No MRR/F1. |
| Redundancy | **Count each chunk.** Three chunks of an expected Article 9 = 3 relevant items, P@5 = 0.60. Textbook precision@k, comparable to published numbers. |

## Definitions to implement

Relevance is judged at provision level `(kind, number)` — the granularity the golden
`expected_source` uses. Reuse the existing `parse_refs` (`eval_retrieval.py:81`),
which already expands ranges and both dash characters.

A retrieved chunk is **relevant** iff `parse_refs(citation) & expected_refs` is non-empty.

```
Recall@k    = | expected_refs  ∩  union(parse_refs(c) for c in citations[:k]) |
              ----------------------------------------------------------------
                                  | expected_refs |

Precision@k =  count of citations[:k] that are relevant
               ------------------------------------------
                        min(k, len(citations))
```

Three subtleties the code must get right:

1. **Precision counts chunks, recall counts provisions.** A packed citation like
   `"Recitals 1-2"` expands to two refs — that legitimately satisfies two expected
   refs for recall, but it is still **one** retrieved item for precision.
2. **Rank order matters.** `citations` is ordered by similarity: `query.py:87` builds
   it from `_retrieve`'s return, which is Chroma's ranked result. `citations[0]` is
   rank 1. The `[:k]` slice depends on this; note it in a comment so nobody reorders it.
3. **Denominator is `min(k, len(citations))`**, not bare `k`, so a run that returned
   fewer than k chunks isn't unfairly penalised. With `TOP_K = 5` these are equal today.

Score the same rows the existing scorer scores: skip `should_answer != "yes"` and
rows with no parseable `expected_refs`, leaving those cells blank exactly as the
current `score` column does for N/A.

Summary figures are **macro-averages** (mean of per-question values), not micro
(pooled counts). Label them as such in the sheet — the two differ whenever questions
have different numbers of expected refs.

## Changes to `eval_retrieval.py`

- New pure function computing the six values for one row, given
  `expected_refs` and the ordered `citations`. Pure and dependency-free so
  `--selftest` can cover it.
- Call it in the `evaluate()` loop (around line 380) alongside the existing
  `score_row`; store the six values on the result dict.
- `write_excel`: six new columns after `score` — `R@1, R@3, R@5, P@1, P@3, P@5` —
  formatted to 2 decimals, blank for N/A rows. Extend the `widths` map.
- Summary block: append macro-averaged `R@k` / `P@k` rows beneath the existing
  `failure rate` row.
- Terminal output: one extra line printing the macro-averages.
- Module docstring: document the new metrics and the caveat below.

### Caveat to record in the docstring and the sheet

The golden `expected_source` lists only the *primary* sources for each question. A
retrieved chunk that is genuinely useful but unlisted is counted as irrelevant, so
**reported precision is a lower bound**, not a true precision. Recall is unaffected.
This matters most for broad scenario questions whose answer legitimately draws on
several provisions the golden row does not enumerate.

## Verification

1. Extend the existing `--selftest` (`eval_retrieval.py:408`) with synthetic cases
   covering: no hits (all zero); a hit at rank 1 only (`R@1` > 0); a hit at rank 4
   (`R@1 = R@3 = 0`, `R@5` > 0); a packed citation satisfying two expected refs
   (recall 1.0 from a single chunk, precision counts it once); three chunks of one
   expected provision (`P@5 = 0.60`, confirming the redundancy decision).
2. Re-run `python eval_retrieval.py --experiment eu-ai-act-rag-afc59a56` — free,
   reads the stored experiment.
3. **Cross-check against the existing strict score:** every row with `score == 1`
   must have `R@5 == 1.0` (both require full coverage). The converse holds except
   for the two paragraph-completeness failures (Annex 11, Article 112), which will
   show `R@5 == 1.0` with `score == 0`. Any other disagreement means a bug.
4. Confirm `R@1 <= R@3 <= R@5` for every row — recall is monotonic in k.
5. Eyeball the 17 known failures: they should show `R@5 < 1.0`, except those same
   two paragraph-completeness rows.

## Out of scope

- MRR, F1, micro-averaged variants.
- Merging `eval_citation_ranking.py` into this script.
- Any `TOP_K` change or retrieval-strategy work — this only adds measurement.
