# Case study: taking an EU AI Act assistant from basic RAG to a gated agentic pipeline

*An AI product management case study. Every number comes from a named experiment in
[`evals/results/`](../evals/results/) or [`results/eval-sheets/`](../results/eval-sheets/).*

---

## 1. The problem

**Users:** product managers, engineers and compliance leads building or deploying AI in
the EU. **Job to be done:** *"Tell me what the AI Act says about my situation, and show
me where it says it."* For example: "If we deploy AI for CV screening, what obligations
apply?"

The Act is 113 articles, 13 annexes and 180 recitals, and a scenario answer is usually
spread across several of them (Annex III → Article 6 → Article 26). A fluent answer that
cites the wrong article does real harm. Product principles:

1. **Grounded or silent.** Every claim comes from the Act and carries a citation.
2. **Answer what was asked.** An unnecessary refusal fails the user as surely as a wrong answer.
3. **Measured, not vibes.** Each change is judged on a fixed dataset, with the decision rule written down first.

## 2. KPIs

| KPI | Target | Reasoning |
|---|---|---|
| **Correctness** (41 golden questions) | **≥ 0.90** | Good enough for first-pass research. Reachable (whole-Act Opus scores 0.98) but not a given (basic RAG scores 0.73). |
| **Groundedness** | **≥ 0.95** | An invented obligation is worse than a missing one. Not 1.00 because judges vary by 1–2 questions per run. |
| **Correct refusals** | **100%** | Hard gate. Tracked alongside its counterweight: answerable questions wrongly refused. |
| **Cost per question** | **≤ $0.03** | Keeps the product viable and stops quality being bought with the biggest model. |

Diagnostic metrics (not ship criteria): wrongly refused questions, unsupported citations,
answer relevance, code-computed retrieval metrics (recall@k, precision@k), latency.
Definitions are in [metrics.md](metrics.md).

**Honesty note:** during the project, ship decisions used *relative* rules fixed in
advance (e.g. "must add ≥ 3 sources"). The absolute thresholds above were set at
write-up, and the shipped version was checked against them afterwards.

## 3. Golden dataset

41 hand-written questions ([`evals/golden_dataset.csv`](../evals/golden_dataset.csv)),
each with a reference answer, expected sources and a `should_answer` flag: 18
single-provision, 16 multi-hop, 6 out-of-scope, 1 exemption.

**One deliberate relabel.** Two "should refuse" questions ("What countries does the EU
act cover?" and a terrorism-in-Singapore fine question) are answerable from Article 2,
so I moved them to answerable. This happened *after* the agent failed the refusal gate
on exactly those questions (section 6) — a moving-the-goalposts risk. Mitigations: the
relabel is justified by the Act's text, both systems were re-run fresh, and old results
are kept. Known debt: Q26's reference answer still contradicts its new label.

## 4. Evaluation system

**Start by hand.** The first experiment (`afc59a56`) was hand-scored: groundedness 100%,
relevance and correctness 63.4%. Error analysis showed:
- **Retrieval was the biggest failure category** (~47%) → built a code-based retrieval
  evaluator ([`eval_retrieval.py`](../eval_retrieval.py)).
- **The 100% groundedness was partly fake.** 11 of 15 relevance failures were refusals,
  and a refusal is trivially grounded → refusals became their own metric.

**LLM-as-judge.** Three binary judges ([`evals/judges.py`](../evals/judges.py)) —
groundedness, relevance, correctness. Design choices:
- Sonnet 5 judges Haiku 4.5, so no model grades itself.
- Each judge sees only what its axis needs (the correctness judge never sees excerpts).
- Binary, not 1–10: comparable with hand scores and easier to calibrate.
- A judge error returns "no score", never a made-up 0.

**Validation against my hand labels** (41 answers,
[`judge_agreement.py`](../evals/judge_agreement.py)):

| Judge | Agreement | TPR | TNR |
|---|---|---|---|
| Relevance | **95.1%** (39/41) | 1.00 (26/26) | 0.87 (13/15) |
| Correctness | **94.9%** (37/39)\* | 0.96 (24/25) | 0.93 (13/14) |

\* 2 questions have no reference answer, so the judge correctly returns no score.

TNR matters most: a judge that passes bad answers lets them ship silently, while a low
TPR only costs review time. The disagreements were 1 judge error, 1 defensible strict
call and 1 labelling slip of mine. Groundedness couldn't be replayed (excerpts weren't
stored), so on the shipped run I reviewed every groundedness fail: 2 of 3 overturned.

**Human-in-the-loop calibration.** When I disagree with a judge, I correct the score in
the eval sheet with a one-line lesson ([`human_labels.json`](../evals/human_labels.json)),
and lessons are injected into the judge prompt on the next run (e.g. *"answering and then
naming a limit is not evasion"*). Agreement after calibration: **98.4% (121/123)**.

Everything runs from one command, [`run_all_evals.py`](../run_all_evals.py): self-tests,
golden set, judges, retrieval metrics, per-question spreadsheet, traced in LangSmith with
real cost per call.

## 5. Iteration path

Each change had a written hypothesis and check ([decision-log.md](decision-log.md)).

| # | Change | Result | Decision |
|---|---|---|---|
| 1 | Structure-aware chunking (legal units, breadcrumbs) | Baseline: correctness 0.63, strict retrieval 0.58 | Baseline |
| 2 | Section in prompt, top-k 5 → 8 | Strict retrieval 0.65 | Kept. **11 answerable questions still refused** |
| 3 | Metadata lookup for "Article 97"-style queries | Strict retrieval 0.70 | Kept |
| 4 | BM25, hybrid, reranker, bigger embedders | All within noise (reranker +3/61 but too slow) | **Rejected.** Ranking isn't the problem |
| 5 | Relax refusal rule: apply provisions, answer partially, name gaps | Wrongly refused **10 → 4**; correctness 0.66 → 0.78; groundedness −2 | Kept, with a "stretching a provision" risk |
| 6 | Four chunking strategies, pre-set rule: beat today by ≥ 3/61 sources | Best +2, under the margin | Chunking and ranking exhausted |
| 7 | **Agentic RAG** (analyse → retrieve → reflect) | Sources **39 → 47/61**; correctness 0.76 → 0.85 | **Failed the refusal gate** (section 6) |
| 8 | Keep named provisions on top after fusion | Recall@8 0.91 → 0.94; all gates pass | **Shipped** |

### Metrics over time

| Run | What changed | Correctness | Groundedness | Retrieval (strict) | Recall@8 | Latency |
|---|---|---|---|---|---|---|
| `afc59a56` | Baseline (hand-scored, K=5) | 0.634 | 1.000\* | 0.575 | – | – |
| `f9bbf5eb` | + K=8, section line, metadata-first | 0.692 | 0.951 | 0.700 | 0.757 | 4.5 s |
| `e6e5c965` | + relaxed refusal rule | 0.780 | 0.927 | 0.700 | 0.757 | 5.0 s |
| `e7c27ad6` | Agentic, old labels | 0.854 | 0.951 | 0.800 | 0.883 | 14.9 s |
| `35fe63c2` | + Annex 11 fix, updated labels (**shipped**) | **0.927**† | **0.976**† | 0.895 | 0.940 | 15.6 s |

\* Refusals scored as grounded. † After human review; judge-only scores were 0.878 and 0.927.

The scorer changed over time (hand → judge → calibrated judge) and a judged run moves
1–4 questions from noise alone, so only retrieval metrics are strictly comparable.

### What the agent is

A **fixed pipeline, not a free-running loop**: Haiku decides inside each stage, the code
fixes order and budget.
1. **Analyse:** state assumptions and split into sub-questions rewritten into the Act's vocabulary ("CV screening" → "recruitment or selection of natural persons").
2. **Retrieve:** search the original and every rewrite, fuse with RRF — never worse than basic RAG.
3. **Reflect:** per sub-question, search again if evidence is missing; follow only cross-references found in retrieved text.
4. **Answer:** same grounded generator, plus an unsupported-citation check.

Built as a ladder: a stage is kept only if it adds ≥ 3 sources over the last kept stage.
Analyse alone added +2 (not kept alone), reflect added **+8** (kept), LLM reranking tied
(not kept, though it doubled rank-1 hits — a known blind spot of the rule).

## 6. Release gate

Written before the agent's first full eval. All must hold: every should-refuse question
refused; groundedness no more than 1 question below baseline; unsupported citations ≤ 5%;
correctness improves.

**First attempt:** correctness 0.756 → 0.854, but **the agent answered 2 of 4
should-refuse questions**. Not shipped; basic RAG stayed default.

**After** the relabel (section 3) and a fix for RRF pushing named provisions out of the
top 8, both systems were re-run. All gates passed, and the shipped run met all four KPIs:
correctness 0.927, groundedness 0.976, refusals 2/2, $0.022 per question. On the judge
alone, correctness and groundedness sat just under target — which is why every flagged
answer gets a human look. `query.py --baseline` keeps basic RAG available.

## 7. Ceiling check: the whole Act to Opus 5.5

I sent the entire Act (~190k tokens, prompt-cached) to Opus 5.5 for every question.

| | Basic RAG | Agentic (shipped) | Whole Act, Opus 5.5 |
|---|---|---|---|
| Correctness | 0.73 | 0.93† | **0.98** |
| Groundedness | 0.90 | 0.98† | 0.93 |
| Answerable questions refused | 5 | 0 | 0 |
| Cost per question | **$0.005** | $0.022 | $0.086 |
| Latency | **5 s** | 15 s | 17 s |

† Human-reviewed; the others are judge-only.

The agent closes most of the gap; the ceiling is ~2 questions higher at ~4× the cost.
The comparison mixes model and retrieval effects (Opus is stronger and may know the Act
from training), so the clean follow-up is Opus with retrieval or Haiku with the whole
Act. Details: [analysis/fulldoc_rerun_f0d6b8ee.md](analysis/fulldoc_rerun_f0d6b8ee.md).

**When to use which:** basic RAG for cheap lookups that name a provision; the agent for
real scenario questions (the default); whole-Act Opus for a small volume of high-stakes
questions. The agent's 15 s latency is fine for research, not for inline help.

## 8. Lessons

| Failure | Lesson |
|---|---|
| **Over-refusal:** 10 of 37 answerable questions refused, 2 with the answer at rank 1 | A cautious prompt rule was the biggest quality problem. Read failing answers before tuning retrieval. |
| **Relaxing refusals stretched one answer** | Every quality lever needs a counter-metric; groundedness caught it. |
| **Ranking and chunking dead end** | Pre-set noise margins stopped me shipping +2/61 "wins" tuned to the test set. The real gap was vocabulary. |
| **Judge noise:** 1–4 questions flip between identical runs | Don't read one judged run as a trend; use code-computed metrics for attribution. |
| **RRF fusion demoted the provision the user named** | Merging searches needs an explicit rule to keep named provisions. |
| **Judge misses on ~190k-token context** | LLM judges degrade with context length; hand-check before trusting. |
| **4 multi-provision questions still miss a source** | They need chained reasoning (classify, then look up obligations) that one reflect pass doesn't always do. |

## 9. Next steps

1. **Reduce judge noise:** 3 runs per eval with mean and range; hand-check a sample of judge *passes* each release to track TNR.
2. **Pay down dataset debt:** fix Q26 and grow the set past 41, especially scenario and should-refuse questions.
3. **Clean ceiling experiment:** separate model effect from retrieval effect.
4. **Cut agent latency:** parallel stages, a faster reflect model, skip analyse when a provision is named.
5. **Gate in CI** with a cost/latency budget, and multi-step classification for "is my system high-risk, and what follows?"
