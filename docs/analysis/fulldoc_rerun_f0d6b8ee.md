# Full-document Opus 5.5 baseline: re-run on the updated golden set (2026-09-24)

Experiment `eu-ai-act-opus55-fulldoc-f0d6b8ee`. The whole Act goes to
`anthropic/claude-opus-5.5` via OpenRouter, with no retrieval. The judge is
`anthropic/claude-sonnet-5`. There are 41 questions and no generation errors.

Re-run because the golden set in LangSmith changed:
- Questions 25 and 26 were relabelled as answerable.
- The expected sources for CV screening and importers/distributors changed.

## Results vs the other systems

The cost is OpenRouter's actual charge, read from LangSmith, and covers
generation only (no judge).

| Metric | full-doc new `f0d6b8ee` | full-doc old `b69ae716` | rag `ecfbeb79` | agentic `317b0fea` |
|---|---|---|---|---|
| answer correctness | **40/41 (0.976)** | 38/39 (0.974) | 30/41 (0.732) | 37/41 (0.902) |
| answer relevance | **41/41 (1.000)** | 41/41 (1.000) | 34/41 (0.829) | 41/41 (1.000) |
| groundedness | 37/40 (0.925)\* | 41/41 (1.000) | 37/41 (0.902) | 40/41 (0.976) |
| answerable questions refused | 0 | 0 | 5 | 0 |
| should-refuse questions refused (33, 34) | 2/2 | 2/2 | 2/2 | 2/2 |
| mean / median latency | 16.7 s / 15.7 s | 14.7 s / 12.6 s | 5.0 s / 4.8 s | 14.9 s / 15.2 s |
| cost per question | $0.086 | $0.084 | $0.005 | $0.022 |
| cost for 41 questions | $3.53 | $3.44 | $0.20 | $0.88 |

\* One groundedness score is missing: question 4 (placing on the market vs
making available). The judge call failed with OpenRouter HTTP 402,
`in_flight_budget_exhausted`, meaning credits ran out mid-run. This is a
billing problem, not a model problem.

The old run `b69ae716` was scored on the previous labels, so it isn't strictly
comparable.

## Failures

- **Correctness, Q26** ("fine for an AI system plotting terrorism in
  Singapore"). The answer explains territorial scope under Article 2(1)(c).
  The judge marked it wrong because the reference answer still says the Act
  doesn't specify extraterritorial reach. The question was relabelled as
  answerable, but its reference text looks stale. Check it in LangSmith.
- **Groundedness failures:**
  - Q14: importers vs distributors vs deployers comparison table.
  - Q36: US company providing AI services.
  - Q5: AI literacy, with the claim that "Chapter I has no sections".

  The judge calls some of these citations unsupported, for example
  Article 26 for deployer obligations. That article is in the full text, so
  at least some of these may be judge misses over the ~190k-token context.
  They have not been checked by hand.

## Takeaways

- Full-document Opus is still the accuracy ceiling. It gets 40/41 correct,
  against 37/41 for agentic and 30/41 for RAG.
- Agentic comes within 3 questions of that ceiling at about 1/4 of the cost
  per question, with similar latency.
- RAG is about 17× cheaper than full-document Opus. It still loses most of
  its correctness gap through refusing answerable questions (5 of them).
