# Results comparison: `afc59a56` (baseline) vs `342c66bc` (latest)

Date: 2026-09-24

| | Baseline `afc59a56` | Latest `342c66bc` |
|---|---|---|
| Source file | `eval_combined_eu-ai-act-rag-afc59a56.xlsx` | `eval_all_eu-ai-act-rag-342c66bc.xlsx` |
| TOP_K | 5 | 8 |
| Retrieval | similarity only | metadata-first for named provisions, then similarity |
| Prompt | citation label only | citation label + `Location:` (chapter/section) line |
| Generation scoring | by hand | LLM judge (Sonnet 5) + 2 human overrides |

Changes between the two runs:
1. **Section/chapter line.** Each excerpt shown to the LLM now carries its
   chapter and section (`plans/section-metadata-and-top-k-8-plan.md`).
2. **TOP_K 5 -> 8** (same plan).
3. **Metadata-first retrieval.** If the question names an article, annex,
   recital, section, chapter or the preamble, those chunks are looked up by
   metadata and put first (`plans/metadata-first-retrieval-plan.md`).

> **Caveat:** the baseline's generation scores were hand-scored; the latest
> run's come from the LLM judge plus two human overrides. Some differences are
> the scorer, not the pipeline. The retrieval scores are computed by code and
> are not affected.

## Aggregate scores

| Metric | Baseline | Latest | Change |
|---|---|---|---|
| Retrieval (strict) | 0.575 | 0.700 | +0.125 |
| Recall@1 | 0.377 | 0.433 | +0.056 |
| Recall@3 | 0.567 | 0.623 | +0.056 |
| Recall@5 | 0.646 | 0.674 | +0.028 |
| Recall@8 | — | 0.757 | new |
| Precision@1 (lower bound) | 0.500 | 0.556 | +0.056 |
| Precision@5 (lower bound) | 0.239 | 0.267 | +0.028 |
| Groundedness | 1.000 | 0.976 | -0.024 (baseline hand-scored; refusals score 1) |
| Answer relevance | 0.634 | 0.732 | +0.098 |
| Answer correctness | 0.634 | 0.683 | +0.049 |

## Per-question changes (9 of 41)

Scores are shown as retrieval / groundedness / relevance / correctness.

### Improved by metadata-first retrieval (the question names a provision)

| Question | Before | After | What changed |
|---|---|---|---|
| what does article 97 talk about? | 0/1/0/0, refused | 1/1/1/1 | Article 97 was not retrieved; now it's pinned first and answered. |
| what does Annex 11 talk about? | 0/1/0/0 | 1/1/1/1 | Annex XI is now pinned first. The baseline 0 for the answer was likely a hand-scoring slip: the answer nearly matched the reference word for word. |
| summarize article 112 | 0/1/1/1 | 1/1/1/1 | All 3 of Article 112's chunks are now retrieved, so the strict completeness check passes. |

### Improved by the section/chapter line

| Question | Before | After | What changed |
|---|---|---|---|
| which section talks about notifying everyone along with the procedure? | 1/1/0/0 | 1/1/1/1 | The answer now names Chapter III, Section 4 and Article 30 instead of only Article 30. |

### Improved by K=8 (the expected article only appears in slots 7–8)

| Question | Before | After | What changed |
|---|---|---|---|
| what article talks about classifying ai models as ones with systemic risk? | 0/1/0/0, refused | 1/1/1/1 | Article 51 is now at rank 7, and the question is no longer refused. |
| how must high-risk systems be tested? | 0/1/1/1 | 1/1/1/1 | Article 60 is now at ranks 7–8, so the strict retrieval check passes. |

### Got worse

| Question | Before | After | Cause |
|---|---|---|---|
| is sex trafficking a prohibited AI practice? | 1/1/1/1 | 1/1/0/0, refused | **Real regression.** Retrieval is fine (Article 5 at rank 1); the LLM chose to refuse. Needs looking into. |
| when is the AI system considered high risk? | 0/1/1/1 | 0/1/1/0 | **Scorer difference.** The judge is stricter than the hand score: the answer covers only 2 of the 8 Annex III areas. Already flagged in `plans/llm-as-judge-evaluators-plan.md`. |
| We are a company selling AI software for Children in Italy. What applies to us? | 0/1/0/0 | 0/0/1/0 | Mixed: relevance up, groundedness down. Retrieval still misses Annex III. |

## Open issues

- The sex-trafficking refusal: retrieval is correct, so the LLM's refusal behaviour
  is the thing to examine.
- 10 answerable questions are still refused in `342c66bc`; this is now the
  main bottleneck.
- "what is this document dated as?" still fails: it doesn't use the word
  "preamble", so metadata-first retrieval never fires.
- The Italy question still misses Annex III.
