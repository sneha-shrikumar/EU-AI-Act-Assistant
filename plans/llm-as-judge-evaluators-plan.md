# LLM-as-judge evaluators for Groundedness, Answer Relevance, Answer Correctness

> Status: **Done 2026-09-23. Results below.**

## Context

`evals/run_langsmith_eval.py` currently runs all 41 golden questions through
`answer_question()` and records answer/latency/cost, but scores **nothing**.
The three score columns in the exported experiment CSVs
(`eu-ai-act-rag-afc59a56 experiment 23sep.csv` and friends) were filled in by
hand in the LangSmith UI. Aggregating that file gives:

| Metric | Score |
|---|---|
| Groundedness | 100.0% (41/41) |
| Answer Relevance | 63.4% (26/41) |
| Answer Correctness | 65.0% (26/40, 1 row left blank) |

Hand-scoring is why one row is blank and why re-running the experiment
produces no scores at all. This plan replaces the manual pass with three
automated LLM-as-judge evaluators attached to `evaluate()`.

### What the baseline numbers actually mean

All 15 relevance failures are in the first 17 rows; rows 18-41 are perfect.
Every failing row has `should_answer: "yes"`, and 11 of the 15 are outright
refusals emitting `config.REFUSAL_MESSAGE`. So the 100% groundedness is
partly an artifact: **a refusal is trivially grounded**. The judges below
reproduce that convention deliberately (see Decisions) so new experiments stay
comparable to this baseline, rather than silently redefining the metric.

## Decisions (confirmed with the user)

1. **Refusals score 1 for groundedness**, exactly as the hand-scored baseline
   did. Not skipped, not zeroed. Keeps new runs comparable with the existing
   CSVs. The refusal artifact is documented here instead of being corrected
   away.
2. **Binary 0/1 scores**, matching the values already in the CSVs. No graded
   0.0-1.0 — a 0.5 would not be comparable to anything in the baseline.
3. **Correctness honours `reference_outputs.should_answer`.** When
   `should_answer == "no"`, a refusal is the *right* behaviour and scores 1;
   answering anyway scores 0. This measures the refusal gate as a feature
   rather than counting it as a failure. Four rows in the golden dataset (21,
   24, 35, 40) are `should_answer: "no"`, so this is load-bearing today, not
   just future-proofing.
4. **Answer Relevance also honours `should_answer`** (revised during
   implementation -- see Verification). The original intent was "any refusal
   scores 0", but replaying the judges against the hand-scored baseline showed
   that rule disagreeing with the manual labels on exactly the four
   `should_answer: "no"` rows (21, 24, 35, 40), every one hand-scored
   relevance=1. The baseline's own convention is that declining an
   unanswerable question *is* the responsive answer, so relevance now matches
   correctness: refusal + `should_answer: "no"` scores 1, refusal to an
   answerable question scores 0. Both are short-circuited without an LLM call.

## Judge model

Judging with the same model that wrote the answer invites self-preference
bias, so the judge uses a *different, stronger* model than the generator:

- generator: `config.OPENROUTER_MODEL` = `anthropic/claude-haiku-4.5`
- judge: new `config.JUDGE_MODEL` = `anthropic/claude-sonnet-5`
  (confirmed available on the OpenRouter `/models` endpoint)

Both go through the existing OpenRouter client, temperature 0.

## Changes

### 1. `config.py`

Add a Judging section:

```python
JUDGE_MODEL = "anthropic/claude-sonnet-5"
JUDGE_TEMPERATURE = 0
```

### 2. `evals/judges.py` (new)

One shared `_call_judge(system_prompt, user_prompt)` helper plus three
evaluator functions with LangSmith's
`(inputs, outputs, reference_outputs)` signature, each returning
`{"key": ..., "score": 0|1|None, "comment": <judge reasoning>}`.

Robustness rules, following `llm.py`'s existing "return errors, never raise"
style — a judge must never turn a good pipeline run into a failed experiment
row:

- The judge is asked for a bare JSON object `{"reasoning": str, "score": 0|1}`.
  `response_format` is *not* relied on (OpenRouter's json-mode support varies
  by upstream provider); instead the response is parsed tolerantly — strip
  ```` ```json ```` fences, then fall back to extracting the first `{...}`.
- On an API error or an unparseable response, return `score=None` with the
  raw text in `comment`. A missing score is honest; a fabricated `0` is not.
- `@traceable(run_type="llm")` on the judge call so judge tokens/cost render
  in LangSmith, reusing `llm.py::_set_llm_usage` for real OpenRouter cost.

The three judges:

| key | sees | 1 means |
|---|---|---|
| `groundedness` | retrieved excerpts + answer | every factual claim traceable to an excerpt; citation labels point at the excerpt the claim came from. Refusal short-circuits to 1. |
| `answer_relevance` | question + answer | the answer addresses what was actually asked. Refusal short-circuits per `should_answer` (see decision 4). |
| `answer_correctness` | question + reference answer + answer | same substantive legal content as the reference; paraphrase, extra correct detail and differing citation format are all fine. `should_answer` short-circuits per decision 3. |

Groundedness deliberately never sees the question and correctness never sees
the excerpts, so each judge scores one axis and cannot launder a failure on
one into a pass on another.

### 3. `evals/run_langsmith_eval.py`

- `run_pipeline` must additionally return the retrieved excerpts — the
  groundedness judge cannot work without them, and today they are dropped:

  ```python
  "contexts": [
      {"citation": c["citation"], "text": c["text"]}
      for c in result["retrieved_chunks"]
  ],
  ```

- Pass `evaluators=[groundedness, answer_relevance, answer_correctness]` to
  `evaluate()`.
- Add `--no-judges` to run the pipeline without paying for judging, and
  record `judge_model` in the experiment `metadata` so a run scored by a
  different judge is never silently compared against one that wasn't.

## Verification

1. `python evals/run_langsmith_eval.py --limit 3` — confirm three score
   columns appear in the LangSmith experiment view and that judge reasoning
   shows in each feedback comment.
2. Sanity-check judge agreement against the hand-scored 23sep baseline: the
   refusal rows should land groundedness=1 / relevance=0 / correctness=0,
   reproducing the manual labels. Material disagreement means the judge
   prompt needs tightening, not the pipeline.
3. Full run, then compare aggregate to the 100% / 63.4% / 65.0% baseline.

### Verification results

Step 2 was run for real by replaying all 41 rows of the 23sep CSV through the
judges (groundedness excluded -- that export predates `contexts` in the run
output, so there are no excerpts to grade against):

| judge | agreement with hand labels |
|---|---|
| `answer_relevance` | 39/41 = **95.1%** |
| `answer_correctness` | 36/39 = **92.3%** |

The replay changed two design decisions rather than being a rubber stamp:

- It caught the relevance/`should_answer` bug in decision 4 above. Before the
  fix, relevance agreement was 85.4%; all six disagreements were explainable,
  four of them by that one rule.
- It caught the correctness judge being too lenient with terse multi-element
  references: row 6 ("which section talks about notifying everyone") has
  reference "Section 4, article 30", and the answer names only Article 30.
  The judge passed it; the prompt now states explicitly that every element of
  a terse reference counts.

Three residual disagreements were examined individually and left alone,
because the judge looks at least as defensible as the hand label:

- **row 17** (`what does Annex 11 talk about?`) -- judge 1, hand 0. The answer
  reproduces the reference ("technical documentation for providers of
  general-purpose AI models") almost verbatim. The hand label looks like a
  scoring slip.
- **row 38** (`when is the AI system considered high risk?`) -- judge 0, hand
  1. The reference lists the eight Annex III domains; the full 1521-character
  answer mentions only two of the eight and instead gives the Article 6
  structural rule. Confirmed by substring check.
- **row 37** (biometric categorisation in a workplace) -- judge 0, hand 1.
  Reference is a flat "Yes, high risk"; the answer hedges across a high-risk
  branch and a prohibited branch. Genuinely borderline.

Chasing these three would be overfitting the prompt to one experiment.

### Dataset gap found while verifying

Rows 10 (`what data must be listed in the EU database for high-risk systems?`)
and 27 (`how must high-risk systems be tested?`) have an **empty
`expected_answer`** in the golden dataset -- only `expected_source` is filled
in. The correctness judge returns `score=None` with an explanatory comment for
these rather than inventing a verdict, which is also why the 23sep CSV has a
blank correctness cell. Filling in those two reference answers would take
correctness from 39 gradeable rows to 41.
