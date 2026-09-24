# Decision log

Every change was written as a plan before any code: the problem, the hypothesis, how it
would be checked, and, where it applies, a decision rule fixed in advance. Each plan's
Results section records what happened. Dates are when the plan was completed.

| Date | Plan | Problem or hypothesis | Outcome | Decision |
|---|---|---|---|---|
| 2026-07-12 | [LangSmith tracing](../plans/langsmith-tracing-implementation-plan.md) ([design](../plans/langsmith-tracing-technical-design.md)) | Can't improve what you can't see | Retrieval and generation traced, with real OpenRouter cost for each call | Foundation |
| 2026-07-19 | [Chapter/section context](../plans/chapter-section-context-plan.md) | Chunks lost their place in the Act's hierarchy | Chapter and section titles added to chunk text and metadata | Kept |
| 2026-08-13 | [Citations always populated](../plans/citations-always-populated-plan.md) | Refusals returned no `citations`, so evals couldn't see what had been retrieved | Citations always come from the retrieved chunks | Bug fix |
| 2026-08-13 | [Split parenthetical paragraphs](../plans/chunking-parenthetical-paragraphs.md) | Article 3 (68 definitions) was stored as one 2,574-word chunk | Split into individual definitions | Bug fix |
| 2026-08-13 | [Citation-ranking judge](../plans/citation-ranking-judge.md) | Recall alone ignores whether the right source ranks near the top | Top-3 citation hit metric | Kept |
| 2026-08-13 | [Retrieval completeness check](../plans/retrieval-completeness-check.md) | "Summarize Article 112" passed with only 3 of its 12 chunks retrieved | Strict whole-provision check for questions that name a provision | Kept |
| 2026-08-23 | [Retrieval recall evaluator](../plans/retrieval-recall-evaluator.md) | Error analysis: about 47% of issues were retrieval misses | Code-based retrieval scoring against the golden `expected_source` | Kept |
| 2026-09-20 | [Token-window chunking](../plans/token-window-chunking-plan.md) | Word-count chunks don't match what the embedder sees | 500-token windows inside provisions, measured with bge's own tokenizer | Kept |
| 2026-09-20 | [Fix packing fallout](../plans/fix-packing-titles-and-ranges-plan.md) | Packing dropped provision titles and created ranges ("Recitals 1–2") that couldn't be parsed | Titles restored, ranges made parseable | Bug fix |
| 2026-09-23 | [Precision/recall@k](../plans/precision-recall-at-k-plan.md) | A strict pass/fail score always rewards a larger k | Fractional P@k and R@k, with precision reported as a lower bound | Kept |
| 2026-09-23 | [LLM-as-judge](../plans/llm-as-judge-evaluators-plan.md) | Hand scoring can't be re-run | 3 Sonnet 5 judges; 95% (relevance) and 92% (correctness) agreement with hand labels | Kept |
| 2026-09-24 | [Section line + top-k 8](../plans/section-metadata-and-top-k-8-plan.md) | "Which section…" answers omitted the section; sources sat at ranks 6–8 | Strict retrieval 0.575 → 0.650; answers flat; 11 answerable questions refused | Kept. Refusal became the priority |
| 2026-09-24 | [Metadata-first retrieval](../plans/metadata-first-retrieval-plan.md) | bge-small can't match "Article 97" | Strict retrieval 0.650 → 0.700; Article 97 answered | Kept |
| 2026-09-24 | [Retrieval options analysis](analysis/retrieval-options-analysis.md) | Would BM25, hybrid, a reranker or a bigger embedder fix retrieval? | Reranker +3/61 but slow; hybrid −1; others within noise | Rejected. Relaxing refusals ranked #1 |
| 2026-09-24 | [Human-label calibration](../plans/human-label-calibration-plan.md) | The judge was wrong on 2 rows | Corrections go into `human_labels.json` and from there into the judge prompts; 98.4% agreement | Kept |
| 2026-09-24 | [Relax the refusal rule](../plans/relax-refusal-rule-plan.md) | Refusals come from the prompt, not missing evidence | Answerable questions refused 10 → 4; out-of-scope still 4/4; groundedness −2 | Kept |
| 2026-09-24 | [Chunking strategies](../plans/structural-chunking-plan.md) | Chunk shape is the remaining retrieval lever (rule: must win by ≥ 3 of 61) | Best +2, under the margin | Low-risk variant B kept. Retrieval tuning exhausted |
| 2026-09-24 | [Opus 5.5 whole-Act baseline](../plans/opus-full-document-baseline-plan.md) | How much does retrieval leave on the table? | Correctness 40/41 at $0.086 per question ([re-run](analysis/fulldoc_rerun_f0d6b8ee.md)) | Used as the ceiling reference |
| 2026-09-24 | [Agentic RAG](../plans/agentic-rag-plan.md) | Wording gaps and multi-step questions need query rewriting and a coverage check | Ladder: reflect +8/61 kept, select not kept. Correctness 0.76 → 0.85. **Failed the refusal guardrail** | Not shipped on the first attempt |
| 2026-09-24 | [Ship the agent](../plans/ship-agent-default-plan.md) | Pinned provisions were lost in fusion; golden set relabelled | Recall@8 0.94; all guardrails pass | **Shipped as default** |
| 2026-09-24 | [GitHub release](../plans/github-portfolio-release-plan.md) | Package the project as a portfolio piece | This repo | – |

Other comparisons:
- [afc59a56 vs 342c66bc](analysis/results-comparison-afc59a56-vs-342c66bc.md)
- [342c66bc vs e6e5c965](analysis/compare_342c66bc_vs_e6e5c965.md)
