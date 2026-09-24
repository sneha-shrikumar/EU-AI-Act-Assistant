# Case study: taking an EU AI Act assistant from basic RAG to a gated agentic pipeline

*An AI product management case study. Every number here comes from a named experiment
in [`evals/results/`](../evals/results/) or [`results/eval-sheets/`](../results/eval-sheets/),
or from the plan file that recorded it at the time.*

---

## 1. What I was trying to achieve

**Users:** product managers, engineers and compliance leads at companies that build or
deploy AI in the EU.

**What they need done:** *"Tell me what the AI Act says about my situation, and show me
where it says it."*

Typical questions:
- "If we deploy AI for CV screening, what obligations apply?"
- "Is an AI system that provides entertainment to children a prohibited practice?"
- "What's the difference between placing on the market and making available?"

**Why a general chatbot isn't enough.** The Act is 113 articles, 13 annexes and 180
recitals. The answer to a scenario question is usually spread across several of them.
For example, Annex III says CV screening is high-risk, Article 6 says what that means,
and Article 26 lists the deployer's duties. A fluent answer that cites the wrong article
does real harm in a compliance setting. So the product principles were:

1. **Grounded or silent.** Every claim comes from the Act's text and carries a citation.
   No outside legal knowledge.
2. **Answer the question the user actually asked.** A refusal is not a safe default. An
   unnecessary refusal fails the user just as surely as a wrong answer does.
3. **Measured, not vibes.** Each change is judged against a fixed dataset, with a
   decision rule written down before the results come in.

## 2. Defining quality: four KPIs with thresholds

| KPI | Target | Reasoning |
|---|---|---|
| **Correctness**: the share of 41 golden questions whose answer matches the reference in substance | **≥ 0.90** (at most 4 wrong) | The core promise. At 9 in 10, users trust the assistant for first-pass research and check the citations on the answers that matter. The target can be reached (the whole-Act Opus ceiling scores 0.98) but isn't a given (basic RAG scores 0.73). |
| **Groundedness**: every claim traces to a retrieved excerpt | **≥ 0.95** | "Never invent." An invented obligation is more dangerous than a missing one. It isn't 1.00 because judges vary by 1–2 questions between runs, so every flagged answer is reviewed by hand. |
| **Correct refusals**: questions outside the Act are declined | **100%** | Zero tolerance for bluffing. This is a hard gate, not an average. Its counterweight, wrongly refusing answerable questions, is tracked alongside it. |
| **Cost per question** | **≤ $0.03** | Keeps the product viable (1,000 questions a month ≈ $30), and stops quality being "bought" with the largest model on every question. |

Supporting metrics, used for diagnosis, not as ship criteria:

| Metric | Why |
|---|---|
| Answerable questions wrongly refused | The counterweight to correct refusals. |
| Unsupported citations (≤ 5% of answers) | Catches a provision that's cited but was never retrieved. |
| Answer relevance | Separates "wrong" from "didn't answer". |
| Retrieval: strict pass, recall@1/3/5/8, precision@k, top-3 citation hit | Shows *where* a failure happens. Computed by code, so free of judge noise. |
| Latency, LLM calls per question | The other price of quality. |

**Honesty note:** during the project, ship decisions used *relative* rules fixed in
advance, for example "must add ≥ 3 sources" and "groundedness no more than 1 question
below baseline". The absolute thresholds above were set when the project was written
up, and the shipped version was checked against them afterwards.

Definitions and formulas are in [metrics.md](metrics.md).

## 3. The golden dataset

- **41 questions** written by hand in LangSmith. Each has a reference answer, the
  expected source provisions, a `should_answer` flag and a category.
- **Categories:**
  - 18 single-provision questions
  - 16 multi-hop questions
  - 6 out-of-scope questions
  - 1 exemption question
- The dataset is in [`evals/golden_dataset.csv`](../evals/golden_dataset.csv). The live
  copy in LangSmith is the source of truth.

**It was relabelled once, deliberately.** Two questions started as "should refuse":
- "What countries does the EU act cover?"
- "How much fine will an AI system plotting terrorism in Singapore get in the EU?"

Both are answerable from Article 2 (territorial scope), so I moved them to answerable.
That left 2 should-refuse questions. I also corrected the expected sources for 2
questions.

**The risk:** this relabelling happened *after* the agent failed the refusal guardrail
on exactly those two questions (section 6). That is a moving-the-goalposts risk, and
I'm naming it openly. The mitigations:
- The relabel is justified by the Act's text, not by the result.
- Both systems were re-run fresh on the new labels.
- The old results are kept.

**Known dataset debt:**
- Q26's reference answer still says the Act doesn't specify extraterritorial reach,
  which contradicts its new label.
- 2 questions had no reference answer when the judges were first built.

## 4. The evaluation system

### 4.1 Start by hand, then automate
The first experiment (`afc59a56`) was scored by hand in LangSmith:
- groundedness 100%
- relevance 63.4%
- correctness 63.4%

The error analysis on it showed two things that shaped everything afterwards:

- **Retrieval failures were the biggest category**, about 47% of flagged issues.
  → I built a code-based retrieval evaluator
  ([`eval_retrieval.py`](../eval_retrieval.py)). It checks whether the expected
  provisions were retrieved, and later added a strict whole-provision completeness check
  and recall/precision@k.
- **The 100% groundedness was partly fake.** 11 of the 15 relevance failures were
  outright refusals, and *a refusal is trivially grounded*.
  → From then on, refusals were tracked as their own metric.

### 4.2 LLM-as-judge
Hand scoring doesn't scale and can't be re-run, so I built three binary 0/1 judges in
[`evals/judges.py`](../evals/judges.py):

| Judge | Sees | Scores 1 when |
|---|---|---|
| Groundedness | excerpts + answer (not the question) | every claim traces to an excerpt, cited correctly |
| Answer relevance | question + answer | the answer addresses what was asked |
| Answer correctness | question + reference + answer (not the excerpts) | the legal substance matches the reference |

Design choices a PM should be able to defend:
- **A different model family judges.** Claude Sonnet 5 judges Claude Haiku 4.5's
  answers, so a model never grades its own output.
- **Each judge sees only what its axis needs.** A failure on one axis can't be hidden
  by a pass on another.
- **Binary scores, not 1–10.** They're comparable with the hand-scored baseline and
  easier to calibrate.
- **A judge error returns "no score", never a made-up 0.** A missing score is honest; a
  fabricated one is not.

**Validation before trusting it.** I replayed the 41 hand-scored answers through the
judges:
- relevance agreed 95.1% of the time;
- correctness agreed 92.3%.

The replay also caught a real judge bug and a leniency problem, and both were fixed. I
reviewed the remaining 3 disagreements and left them alone. In 2 of them the judge was
at least as defensible as the hand label, and tuning the prompt to one experiment would
be overfitting.

### 4.3 Human-in-the-loop calibration
When I disagree with a judge, I edit the score in the eval spreadsheet and run
`run_all_evals.py --import-labels`. That does two things:
- it records the correction, with a one-line *lesson*, in
  [`evals/human_labels.json`](../evals/human_labels.json);
- the lessons are added to each judge's prompt automatically on the next run.

The first two corrections taught general rules:
- *"Answering and then naming a limit is not evasion."*
- *"A caveat that refines the conclusion isn't a contradiction."*

Judge–human agreement on that run was **98.4% (121 of 123 judgments)**.

### 4.4 One command
[`run_all_evals.py`](../run_all_evals.py) runs these steps in order:
1. Offline parser self-tests. They're free, and the run stops if one fails.
2. The golden set through the chosen system (`--system rag|agentic`).
3. The three judges.
4. The retrieval metrics.
5. A per-question spreadsheet with a summary sheet.

It also re-scores an old run at no cost (`--experiment`) and imports human labels.
Every run is traced in LangSmith with real OpenRouter cost for each call.

## 5. The iteration path: basic RAG to agentic

Each change had a written plan with a hypothesis and a check. The full list is in
[decision-log.md](decision-log.md). The main path:

| # | Change | Hypothesis | Result | Decision |
|---|---|---|---|---|
| 0 | Basic RAG (July): word-count chunks, bge-small, top 5, Haiku 4.5, LangSmith tracing | – | Not yet measured | – |
| 1 | Structure-aware chunking: 500-token windows inside provisions, chapter/section breadcrumbs, and a fix for Article 3's 68 definitions stored as one chunk | Legal units embed better than arbitrary cuts | First measured baseline `afc59a56`: correctness 0.63 (hand-scored), strict retrieval 0.58 | Baseline |
| 2 | Show Chapter/Section in the prompt, and top-k 5 → 8 | "Which section…?" questions fail at generation, not retrieval | Strict retrieval 0.58 → 0.65, recall@8 0.73; answers unchanged | Kept. **11 answerable questions still refused**: that's now the bottleneck |
| 3 | Metadata-first retrieval: "Article 97" is looked up by metadata, not embedding | bge-small can't match bare numbers | Strict 0.65 → 0.70; "article 97" goes from refused to correct | Kept |
| 4 | Test other retrieval methods (BM25, hybrid, cross-encoder reranker, bigger embedders) | Better ranking finds missing sources | Reranker +3/61, but too slow; hybrid −1; bge-base/large −2/−1 | **Rejected.** Ranking isn't the problem |
| 5 | Relax the refusal rule: apply provisions to the user's scenario, answer partially, and name the gaps | Refusals come from the prompt, not missing evidence | Answerable questions refused: **10 → 4**; correctness 0.66 → 0.78; out-of-scope questions still refused 4/4; groundedness −2 questions | Kept. One groundedness loss is a real "stretching a provision" risk |
| 6 | Four chunking strategies compared across the whole corpus, with the rule fixed in advance: must beat today by ≥3 of 61 sources | Chunk shape is the last lever on the retrieval side | Best variant +2, **under the margin**. Low-risk overlap fix (+1) kept | Chunking and ranking are exhausted |
| 7 | **Agentic RAG** (analyse → retrieve → reflect), built as a ladder with one rung at a time | The remaining failures are wording gaps ("CV screening" vs "recruitment or selection of natural persons") and multi-step questions | Sources found **39 → 47 / 61**; correctness 0.76 → 0.85 | Better on the KPI, **but failed the refusal guardrail** (section 6) |
| 8 | Annex 11 fix (keep named provisions on top after fusion), then ship | RRF fusion was pushing pinned provisions out of the top 8 | Recall@8 0.91 → 0.94; all guardrails pass on the updated labels | **Shipped as default** |

### Metrics over time

| Run | What changed | Correctness | Relevance | Groundedness | Retrieval (strict) | Recall@8 | Latency |
|---|---|---|---|---|---|---|---|
| `afc59a56` | Baseline (hand-scored, K=5) | 0.634 | 0.634 | 1.000\* | 0.575 | – (R@5 0.646) | – |
| `f9bbf5eb` | + K=8, section line, metadata-first | 0.692 | 0.707 | 0.951 | 0.700 | 0.757 | 4.5 s |
| `342c66bc` | + judge calibration (human-reviewed) | 0.683 | 0.732 | 1.000 | 0.700 | 0.757 | 4.8 s |
| `e6e5c965` | + relaxed refusal rule | 0.780 | 0.854 | 0.927 | 0.700 | 0.757 | 5.0 s |
| `275ca613` | + chunking variant B | 0.756 | 0.854 | 0.902 | 0.750 | 0.778 | 5.2 s |
| `e7c27ad6` | Agentic (reflect), old labels | 0.854 | 1.000 | 0.951 | 0.800 | 0.883 | 14.9 s |
| `317b0fea` | Agentic, updated labels | 0.902 | 1.000 | 0.976 | 0.895 | 0.912 | 14.9 s |
| `35fe63c2` | + Annex 11 fix (**shipped**) | 0.878 → **0.927**† | 1.000 | 0.927 → **0.976**† | 0.895 | 0.940 | 15.6 s |

\* In the hand-scored baseline, refusals scored 1 for groundedness.
† After human review of the judge's failures. 4 judge scores were overturned (2 correctness, 2 groundedness), and each is logged with its lesson in `evals/human_labels.json`.

**How to read this table:**
- The scorer changed over the project: hand scoring, then the judge, then the
  calibrated judge. So judge metrics from different eras aren't strictly comparable.
- Retrieval metrics are computed by code and *are* comparable.
- The golden set changed at `317b0fea`.
- A single judged run moves by 1–4 questions from noise alone (section 8), so a change
  under about 0.05 is not a signal.

### What the agent actually is

It is a **fixed pipeline, not a free-running agent loop**. Haiku makes the decisions
*inside* each stage. The code fixes the order and the budget:
- **① Analyse:** restate the question, state assumptions (e.g. "you deploy rather than
  build"), and split it into sub-questions, each rewritten into the Act's vocabulary.
- **② Retrieve:** search the original question *and* every rewrite, then fuse the
  results with reciprocal rank fusion (RRF). The evidence can never be worse than the
  basic RAG's.
- **③ Reflect:** for each sub-question, check whether the evidence covers it. If not,
  search again. Follow cross-references only when they appear *in retrieved text*,
  never from the model's memory.
- **Answer:** the same grounded generator, plus a post-check for unsupported citations.

**Built as a ladder, with a pre-set rule:** a stage is kept only if it adds at least 3
expected sources over the last kept stage, averaged over 2 runs.

| Rung | Stages on | Sources found (of 61) | Kept? |
|---|---|---|---|
| 0 | Basic RAG | 39 | – |
| 1 | + analyse and rewrite | 41 | No (+2) |
| 2 | + reflect | 47 | **Yes (+8)** |
| 3 | + LLM select/rerank | 47 | No (tie) |

Rung 3 improved ranking a lot (sources at rank 1 went from 16 to 30), but the rule
measured found-in-top-8, so it wasn't kept. That's the rule working as written. It's
also a known blind spot, logged as a follow-up.

## 6. The release gate

The gate was written down before the agent's first full eval. **All** of these must
hold:
1. Every should-refuse question is still refused.
2. Groundedness doesn't fall more than 1 question below the baseline.
3. Unsupported citations appear in ≤ 5% of answers.
4. Correctness improves.

**First attempt (`9c0d12f7`, `e7c27ad6`):** correctness rose 0.756 → 0.854, and
groundedness and citations passed. But **the agent answered 2 of the 4 should-refuse
questions**. With richer evidence, it answered from Article 2 where basic RAG had
refused. **Decision: not shipped.** Basic RAG stayed the default, and the agent sat
behind a flag.

**Then:**
- On review, both "should-refuse" questions were answerable from the Act, so they were
  relabelled (see the caveat in section 3).
- A separate bug was fixed: named provisions were being pushed out of the top 8 by RRF
  fusion (Annex 11: recall@8 went from 1 to 0).
- Both systems were re-run fresh.

**All guardrails passed:**
- 2 of 2 should-refuse questions refused.
- Groundedness above the RAG baseline.
- Unsupported citations in 2 of 41 answers (4.9%), one of them a known false positive.

**Checked against the four KPIs** after I reviewed the judge's failures by hand
(`35fe63c2`):
- correctness 0.927 ✅
- groundedness 0.976 ✅
- correct refusals 2/2 ✅
- $0.022 per question ✅

The review overturned 4 judge scores. On the judge alone, correctness (0.878) and
groundedness (0.927) sat just under target, which is why every flagged answer gets a
human look.

**Decision: shipped as the default.** `python query.py --baseline` keeps the basic RAG
available.

## 7. Ceiling check: skip retrieval and give Opus 5.5 the whole Act

To see how much quality retrieval leaves on the table, I sent the **entire Act**
(about 190k tokens) to Claude Opus 5.5 for every question, with the same "answer only
from the text" rules. I used prompt caching so this cost about $3.50 per run instead of
about $30.

| | Basic RAG | Agentic (shipped) | Whole Act, Opus 5.5 |
|---|---|---|---|
| Answer correctness | 30/41 (0.73) | 38/41 (0.93)† | **40/41 (0.98)** |
| Answer relevance | 0.83 | 1.00 | 1.00 |
| Groundedness | 0.90 | 40/41 (0.98)† | 0.93 (37/40) |
| Answerable questions refused | 5 | 0 | 0 |
| Cost per question | **$0.005** | $0.022 | $0.086 |
| Cost for all 41 questions | $0.20 | $0.88 | $3.53 |
| Mean latency | **5.0 s** | 15 s | 17 s |

Experiment IDs:
- Basic RAG: `ecfbeb79`
- Agentic: `317b0fea` and `35fe63c2`
- Whole Act: `f0d6b8ee`, re-run on the updated golden set

**What it tells me:**
- **The agent closes most of the gap.** The ceiling is about 2 questions above it,
  at about 4× the cost.
- † The agent's score is human-reviewed. RAG and Opus are judge-only, so the gap may be
  slightly understated.
- **The comparison mixes two variables.** Opus 5.5 is a much stronger model than
  Haiku 4.5, so the gap isn't purely a retrieval effect. It may also know the Act from
  its training data. The clean next experiment is Opus with retrieval, or Haiku with
  the whole Act.
- **The whole-Act approach isn't free of failures either.** 3 groundedness fails and
  1 correctness fail. The correctness fail is the stale Q26 reference.

Details: [analysis/fulldoc_rerun_f0d6b8ee.md](analysis/fulldoc_rerun_f0d6b8ee.md).

## 8. Failures I hit, and what they taught me

| Failure | What happened | Lesson |
|---|---|---|
| **Grounded refusals** | The hand-scored baseline showed 100% groundedness, because refusals count as grounded | Track refusals as a first-class metric. A perfect guardrail number can hide a bad product. |
| **Over-refusal** | 10 of 37 answerable questions were refused, and 2 of them had the answer at rank 1 | A cautious-sounding prompt rule ("when in doubt, refuse") was the single biggest quality problem. Read the failing answers before tuning retrieval. |
| **Relaxed rule → stretched answers** | After relaxing refusals, one answer stretched a provision to fit the scenario | Every quality lever has a counter-metric. Groundedness caught this. |
| **Ranking and chunking dead end** | Reranker, hybrid search, bigger embedders and 4 chunking strategies were all within noise | Pre-set noise margins stopped me shipping +2/61 "wins" that were tuned to the test set. The real gap was vocabulary, which needed query rewriting. |
| **Judge noise** | The same code scored correctness 0.732 and 0.707 on two runs; 1–4 questions flip between runs | Don't read a single judged run as a trend. Use code-computed retrieval metrics for attribution. |
| **RRF fusion demoted named provisions** | "What does Annex 11 talk about?" lost Annex XI from the top 8 after fusion | Merging several searches needs an explicit rule to always keep the provision the user named. |
| **The goalposts question** | The guardrail failure led to a golden-set relabel | Justify every relabel against the source, re-run everything, and disclose it. |
| **Stale reference answer (Q26)** | The label changed, but the reference text didn't | A golden set needs the same review discipline as code. |
| **Judge misses on long context** | On the ~190k-token whole-Act runs, the judge called real citations (e.g. Article 26) "unsupported" | LLM judges degrade with context length. Hand-check a sample before trusting them. |
| **Infrastructure** | OpenRouter out-of-credit errors (HTTP 402) left one judge score missing; a LangSmith sync race scored one question from empty outputs; concurrent Chroma clients raced during start-up | A missing score is reported, never filled with 0. Keep the per-question JSON locally as the source of truth. |
| **Some multi-provision questions still miss a source** | Even after shipping, the agent misses an expected source on 4 questions: requirements for high-risk systems, R&D, deployer + transparency, and recruitment penalties | These need chained reasoning (classify first, then look up obligations) that one reflect pass doesn't always do. |

## 9. Trade-offs

| If you need… | Pick | Because |
|---|---|---|
| Lowest cost and latency, and simple lookups ("what does Article 5 say?") | **Basic RAG** | $0.005 and 5 s. Fine when the question names the provision. |
| Scenario questions from real users (the default) | **Agentic** | +8 correct answers over basic RAG, and 0 answerable questions refused, at about 4× cost and about 3× latency |
| The highest accuracy on a small volume of high-stakes questions | **Whole Act to Opus 5.5** | +2 more correct answers, at about 4× the agent's cost. The per-question cost doesn't depend on how hard the question is. It also scales badly if the corpus grows beyond one Act. |

**The 15-second latency is the agent's main product cost.** It's acceptable for a
research-style assistant. It's not acceptable for inline help.

## 10. Next steps

1. **Reduce judge noise:** run each eval 3 times and report the mean and range, so that
   changes of ≤ 0.05 can be judged.
2. **Pay down dataset debt:** fix Q26's reference answer, hand-check the 3
   whole-Act groundedness failures, and grow the golden set past 41 questions,
   especially scenario and should-refuse questions.
3. **Clean ceiling experiment:** Opus 5.5 *with* retrieval, and Haiku with the whole
   Act, to separate the model effect from the retrieval effect.
4. **Cut agent latency:** run stages in parallel, use a smaller or faster model for
   reflect, and skip analyse for questions that name a provision.
5. **Revisit LLM selection** (rung 3), judged on answer quality instead of top-8 recall.
6. **Put the gate in CI:** run the eval on every change to prompts or retrieval, and
   add a cost/latency budget to the release gate alongside quality.
7. **Multi-step classification** for "is my system high-risk, and what follows?"
   questions, which account for most of the sources the agent still misses.
