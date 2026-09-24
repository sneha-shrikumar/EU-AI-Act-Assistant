# Eval comparison: eu-ai-act-rag-342c66bc vs. eu-ai-act-rag-e6e5c965

Source files: `eval_all_eu-ai-act-rag-342c66bc.xlsx` (baseline) and
`eval_all_eu-ai-act-rag-e6e5c965.xlsx` (new). Judge reasons come from the
LangSmith feedback on `eu-ai-act-rag-e6e5c965`.

Both runs use the same setup: Haiku 4.5 as the generator, top_k=8 retrieval,
Sonnet 5 as the judge, and the same 41 questions.

## Summary

e6e5c965 is the better run. It refuses 6 fewer questions, which improves
relevance and correctness. The cost is three answers that used to be grounded
and now cite the wrong provision.

The retrieval metrics are identical in both runs (R@8 0.757, P@8 0.212), so all
of the difference comes from the answer-generation step.

## Scores (judge-only, out of 41)

The 342c66bc headline numbers include human overrides
(`evals/human_labels.json`). e6e5c965 has no overrides, so the judge-only
figures are the fair comparison.

| Metric        | 342c66bc | e6e5c965 | Change |
|---------------|---------:|---------:|-------:|
| Groundedness  | 40       | 38       | −2     |
| Relevance     | 29       | 35       | **+6** |
| Correctness   | 27       | 32       | **+5** |
| Refusals      | 14       | 8        | −6     |

In the table below, scores are shown as groundedness / relevance / correctness,
before → after.

## Improved (6), all from fewer false refusals

| # | Question | G/R/C | Note |
|---|---|---|---|
| 10 | Is sex trafficking a prohibited AI practice? | 100 → 111 | Now answers correctly |
| 23 | Company selling AI recruitment software across Europe | 100 → 111 | Now answers correctly |
| 24 | Is AI entertainment for children a prohibited practice? | 100 → 111 | Now answers correctly |
| 26 | How must examination of biases be carried out? | 100 → 111 | Now answers correctly |
| 7 | Deploying AI for CV screening: what obligations apply? | 100 → 110 | Answers now, but hedges on whether CV screening is high-risk and omits the testing and monitoring obligations. Correctness is still 0. |
| 31 | AI software for children in Italy | 010 → 110 | Now grounded, but hedges on the high-risk classification. Correctness is still 0. |

## Got worse (3): grounded answers that now cite the wrong provision

These answers are still relevant and correct. Groundedness failed because a
claim is attached to the wrong provision.

| # | Question | G/R/C | Judge's reason |
|---|---|---|---|
| 14 | What AI systems are excluded from the scope of the Act? | 111 → 011 | It cites Art. 2(4) for the military, defence and national-security exclusion. Art. 2(4) actually covers third-country public authorities and law-enforcement cooperation. |
| 21 | Information to be submitted for a high-risk system when registering | 111 → 011 | It mixes up the Annex IX registration data with the Art. 60(4) real-world testing plan submission. |
| 25 | AI tool that evaluates students' exam answers | 111 → 011 | It cites Annex III(3)(d) but supports it with Recital 56 wording on evaluating learning outcomes, which belongs to (3)(b). |

## Unchanged (32)

### Pass all three judges in both runs (25)

These are #2, 3, 4, 5, 6, 8, 11, 12, 13, 17, 18, 19, 27, 28, 29, 30, 32, 33,
35, 37 and 40. The list also includes #20, 22, 34 and 41, the questions the Act
does not answer, which both runs correctly refuse.

### Still failing (7)

| # | Question | G/R/C | Likely cause |
|---|---|---|---|
| 1 | What data must be listed in the EU database for high-risk systems? | 100 | Retrieval does not return Annex VIII |
| 9 | Does R&D fall under a high-risk AI system? | 100 | Refused; probably a retrieval gap |
| 15 | What is this document dated as? | 100 | Refused; probably a retrieval gap |
| 16 | Who is a deployer? How do we ensure transparency for deployers? | 100 | No longer refuses, but says it can't find the definition. Art. 3(4) is not retrieved. |
| 36 | How many annexes do we have? | 100 | Refused; probably a retrieval gap |
| 38 | When is an AI system considered high risk? | 110 | Correctness 0 |
| 39 | Penalties for deployers of AI systems targeting minorities | 100 | Refused |

## Takeaways

1. The change made the model more willing to answer from partial excerpts.
   That fixed 4 answers outright, but it also caused the citation slips on
   #14, #21 and #25 and the hedging on #7 and #31.
2. **Next prompt fix:** add a rule to cite only the label of the excerpt a claim
   actually comes from. This targets the misattributions.
3. **Next retrieval fix:** #1, #9, #15, #16 and #36 need the right passages
   retrieved. Better prompting cannot recover them.
