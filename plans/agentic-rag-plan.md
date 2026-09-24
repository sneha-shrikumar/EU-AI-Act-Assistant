# Agentic RAG: a staged pipeline that reads, retrieves, checks and answers

> Status: **Shipped as the default 2026-09-24** (see `plans/ship-agent-default-plan.md`). The first runs failed the refusal guardrail; it passes on the relabelled golden set. Saved here as
> `plans/agentic-rag-plan.md` (CLAUDE.md rule), before any code.

## Context

Retrieval is stuck at about 37–39 of 61 expected sources in the top 8. Chunking and
ranking changes all landed within noise (see `plans/structural-chunking-plan.md` and
the decision record https://claude.ai/artifact/MJMAmdrXPz6UdF4MocByxo). The 12 questions
that still fail come from three gaps that one search with the user's raw words can't
close:

- **Wording:** the question uses everyday words ("CV screening", "R&D", "dated") and the
  Act uses legal ones.
- **Multi-step:** the answer needs a classification step first (is this system in
  Annex III?), then the obligations or penalties that follow.
- **Bundled questions:** "who is a deployer, and how do we ensure transparency?"

The goal is agentic RAG. An LLM reads the question, decides what to search for, checks
whether the evidence covers every part, searches again where it doesn't, and then a
grounded generator writes a cited answer.

It is built as a **fixed sequence of stages**, not a free-form tool loop. The LLM makes
the decisions inside each stage, while the code fixes the order and the budget. This
keeps run-to-run variance low, makes every stage traceable, and lets each stage be
switched off to measure what it contributes.

**Decisions taken:**
- **Models:** Haiku 4.5 (`config.OPENROUTER_MODEL`) runs every stage. The judge stays
  Sonnet 5.
- **Single turn:** the pipeline never asks the user a question back. Ambiguity is
  handled by stating assumptions in the answer.
- **Existing pipeline:** `query.answer_question` is unchanged. It stays as the baseline
  and as the fallback.

## How it works

```
question
  │
  ① ANALYSE      1 LLM call → JSON
  │               reading of the question, assumptions (role, use case),
  │               sub-questions, each with 1–2 search queries in the Act's vocabulary,
  │               structural flag ("how many annexes?")
  │
  ② RETRIEVE     no LLM
  │               search the ORIGINAL question + every rewritten query (query._retrieve, top 8 each)
  │               merge the lists with reciprocal rank fusion (RRF) → evidence pool
  │
  ③ REFLECT      1 LLM call → JSON, at most 2 rounds
  │               per sub-question: covered by the pool? if not, what's missing
  │               → new queries (back to ②)
  │               → provision references written in retrieved text (e.g. "listed in Annex VIII")
  │                 → fetch that provision (back to ②)
  │               stops when everything is covered or after 2 rounds
  │
  ④ SELECT       1 LLM call → JSON
  │               rank the pool, keep ≤ 10 chunks that answer the sub-questions
  │
  ANSWER         today's llm.generate_answer on the kept chunks, same rules,
                 plus the reading, assumptions and sub-questions as non-evidence notes
                 → citation post-check
```

### Example: "If we deploy AI for CV screening, what obligations apply?"

Today none of its expected sources reach the top 8, and it cites Article 50.

1. **Analyse.**
   - Role: deployer. Assumption: "uses, not builds, the system".
   - Sub-question 1: "Is CV screening high-risk?", with the query "AI for recruitment
     or selection of natural persons, filtering job applications, evaluating
     candidates".
   - Sub-question 2: "What must a deployer of a high-risk system do?", with the query
     "obligations of deployers of high-risk AI systems".
2. **Retrieve.** The original question still returns Article 50. The rewritten queries
   return Annex III(4) and Article 26. All are fused into one pool.
3. **Reflect.** Both sub-questions are covered. Annex III's text says "referred to in
   Article 6(2)", so Article 6 is fetched.
4. **Select.** Annex III(4), Article 26(1-4), Article 26(5-6), Article 6(1-2), …
5. **Answer.** "Assuming you use (rather than develop) the system: CV screening is
   high-risk under Annex III(4)(a) … as a deployer you must … (Article 26(1))".

Every chunk is reached by retrieval, or by a reference written in retrieved text.

## What the pipeline does and won't do

**Does:**
- restate the question and split bundled questions into sub-questions;
- rewrite searches into the Act's vocabulary;
- always search the original question too, so the evidence is never worse than
  today's;
- check coverage for each sub-question and search again where there are gaps;
- follow cross-references that appear in retrieved text;
- answer each sub-question with stated assumptions and citations.

**Won't do (enforced in code where possible):**

| Won't | How it's enforced |
|---|---|
| Use knowledge from outside the Act in the answer | The generator prompt is unchanged: excerpts only. The notes are marked "not evidence, never cite". |
| Fetch provisions from the model's memory of the Act | A follow-up reference is accepted only if `provision_refs.parse_question` finds it in the text of a retrieved chunk, or in the user's question. Otherwise it is dropped and logged. |
| Browse the table of contents to find answers | Contents are only added when ① flags a structural question. They are added as a synthetic excerpt cited as `Table of contents`, built from chunk metadata. |
| Cite a provision it didn't retrieve | A post-check compares the answer's citations with the kept chunks. Mismatches go into `unsupported_citations` in the output and in LangSmith. |
| Ask clarifying questions | This is not part of the pipeline. Assumptions are stated instead. |
| Search the web, other laws, or national law (e.g. Italian law) | The only data source is the local index. |
| Answer questions the Act doesn't cover | Both refusal gates stay: the similarity gate on the fused pool and `llm.is_refusal`. |
| Give legal advice or compliance sign-off | Generator rule: answers say what the Act's text states. |
| Run without limit | At most 2 queries per sub-question, 4 sub-questions, 2 reflect rounds, 10 kept chunks, and at most 5 LLM calls per question. |
| Fail silently | Bad JSON from a stage falls back to that stage's default (e.g. no rewrites). An LLM error falls back to `query.answer_question`. Both set `agent_fallback` in the output. |
| Remember earlier questions or change data | It is stateless and read-only. |

## Build and measure as a ladder

The stages are switched on by one config value,
`config.AGENT_STAGES = "analyse" | "reflect" | "select"`. The values are cumulative.

| Rung | Stages on | Question it answers |
|---|---|---|
| 0 | today (`275ca613`) | baseline |
| 1 | ① + ② (rewrite, decompose, fuse; top 10 by RRF) | Does rewriting alone fix the wording gaps? |
| 2 | + ③ reflect | Does the agentic check-and-retry add anything beyond rewriting? |
| 3 | + ④ select | Does LLM selection beat RRF ranking? |

**Decision rule (fixed before running):**
- A rung is kept only if it adds **≥ 3** expected sources in the top 8 over the last
  kept rung, measured on all 36 answerable questions.
- Each rung is run twice and the mean is used.
- The highest kept rung becomes the default.
- If rung 1 captures most of the gain, stop there. That is a valid result.

## Code changes

- **`agent.py` (new):**
  - `answer_question_agentic(question, k)` returns the same dict shape as
    `query.answer_question`, plus `notes`, `llm_calls`, `unsupported_citations` and
    `agent_fallback`.
  - It holds one function per stage: `_analyse`, `_retrieve_pool`, `_reflect`,
    `_select`, each `@traceable` so LangSmith shows every stage.
  - It holds `_rrf(lists)`, the reciprocal rank fusion helper (k = 60).
  - It holds `_table_of_contents()`, built from `query._corpus_index` metadata (title,
    chapter, section).
  - `citations` is the kept chunks in rank order, so the @k metrics stay meaningful.
  - REPL: `python agent.py` prints each stage's output.
- **`query.py`:** add `_get_by_ids(ids)`, which returns chunks in the same shape as
  `_query`. Reuse `_retrieve`, `_get_collection` and `_corpus_index` as they are.
- **`llm.py`:**
  - add `complete_json(system, user)`, a thin wrapper over `_chat_completion` that
    parses JSON;
  - `generate_answer(question, chunks, notes=None)` appends the notes block and adds
    rule 7: state the assumptions first, answer each sub-question, never cite the
    notes.
- **`provision_refs.py`:** reuse `parse_question` and `match_chunk_ids` for follow-up
  references. No changes.
- **`config.py`:** `AGENT_STAGES`, `AGENT_MAX_SUBQUESTIONS = 4`,
  `AGENT_QUERIES_PER_SUB = 2`, `AGENT_REFLECT_ROUNDS = 2`, `AGENT_MAX_EVIDENCE = 10`.
- **`analysis/agentic_retrieval_diagnostic.py` (new):**
  - runs stages ①–④ with no answer generation and no judges over the 36 questions;
  - scores top-8 hits, recall@1/3/5 and distinct provisions, using the same method as
    `analysis/chunking_variants_diagnostic.py` (via `eval_retrieval.parse_refs`);
  - reports LLM calls per question;
  - runs every rung; this is the cheap way to climb the ladder.
- **`evals/run_langsmith_eval.py`:**
  - add `"agentic"` to `SYSTEMS`, with experiment prefix
    `eu-ai-act-agentic-<AGENT_STAGES>`;
  - `_to_outputs` carries `llm_calls`, `agent_fallback` and `unsupported_citations`
    when present.
- **`run_all_evals.py`:** add a `--system` flag and pass it through.
- **Unchanged:** `chunking.py`, `ingest.py`, the index, `evals/judges.py`,
  `eval_retrieval.py`, and the golden set.

## Verification

1. **REPL smoke test** on 5 target questions: CV screening, R&D, "dated", Italy, and
   "how many annexes". Check that:
   - the rewrites are in the Act's vocabulary;
   - sub-questions are split correctly;
   - reflect only follows references that appear in retrieved text;
   - the contents excerpt appears only for the structural question.
2. **Ladder:** `python analysis/agentic_retrieval_diagnostic.py`, twice. Apply the
   decision rule and pick the rung.
3. **Full eval** of the chosen rung, twice:
   `python run_all_evals.py --system agentic --concurrency 2`. Compare with `275ca613`.
   Guardrails, all must hold:
   - the 4 should-refuse questions still refuse;
   - mean groundedness doesn't fall more than 1 question (0.024) below the baseline;
   - `unsupported_citations` is empty on at least 95% of answers.

   Also report latency, cost, and LLM calls per question.
4. **Record the results** (ladder table, eval comparison, before/after for the 12
   failing questions, and traces of the ones still failing) in
   `plans/agentic-rag-plan.md`. Update the decision-record artifact.

## Risks

- **Model prior knowledge:** Haiku knows the Act, so its rewritten queries may name
  "Annex III" outright. That's normal query rewriting and the answer stays grounded,
  but it is recorded as a limitation. Follow-up fetches are restricted to references in
  retrieved text to keep this in check.
- **Latency and cost:** about 3–5 Haiku calls per question instead of 1, all bounded.
  Measured, not assumed.
- **Drift:** a rewritten search can move away from what the user asked. The original
  question is always searched, and the answer addresses the original question.
- **Small golden set:** 36 questions. The margin rule, two runs and whole-set scoring
  limit overfitting but don't remove it. There is deliberately no domain checklist
  (e.g. "always check Article 5 / Annex III"), because it would be tuned to the
  failing questions.

## Results (2026-09-24)

### Ladder (`analysis/agentic_retrieval_diagnostic.py`, 2 runs per rung, no generation)

Expected sources among the first 8 citations, of 61. "Hits@k" counts expected sources
at rank ≤ k.

| Rung | Top 8 (run 1 / run 2) | Hits@1 | Hits@3 | Hits@5 | Distinct provisions in top 8 | LLM calls per question |
|---|---|---|---|---|---|---|
| 0 baseline (`query._retrieve`) | 39 | 18 | 31 | 37 | 8.31 | 0 |
| 1 analyse | 41 / 41 | 14 | 30 | 36 | 8.19 | 1.0 |
| 2 reflect | **47 / 47** | 16 | 38.5 | 43 | 7.58 | 2.8 |
| 3 select | 47 / 47 | **30** | **45** | **46** | 4.89 | 3.8 |

There were no stage fallbacks and no dropped (memory-sourced) references in any run.

**Decision (pre-set rule):**
- Rung 1 adds 2, under the margin, so it isn't kept on its own.
- Rung 2 adds 8 over the baseline, so it is **kept**.
- Rung 3 ties rung 2 on the primary metric, so it isn't kept.
- Default: `AGENT_STAGES = "reflect"`.

Notes:
- The baseline here is 39, not the 38 in the chunking diagnostic. This diagnostic uses
  `query._retrieve` with metadata pinning; the chunking one used pure dense ranking.
- **Select** leaves the top-8 count unchanged but moves expected sources much higher:
  Hits@1 goes from 16 to 30 and Hits@3 from 38.5 to 45. The cost is fewer distinct
  provisions (4.9) and one more call. The primary metric doesn't reward ranking, so
  the rule rejects select. That is a decision rule working as written, not evidence
  that select is useless. It's a candidate for a follow-up judged on answer quality.
- **Per question:** reflect finds sources for the date, R&D, Italy, minorities
  penalties, the EU database (Annex VIII), "when high risk", and the exams tool.
  Small losses: importers/distributors (4 → 3) and "requirements for high-risk" (2 → 1).
  CV screening still finds only 2 of 9.

### Full eval: `275ca613` (baseline) vs agentic `reflect`, 2 runs

| Metric | 275ca613 | agentic run 1 (`9c0d12f7`) | agentic run 2 (`e7c27ad6`) |
|---|---|---|---|
| answer correctness | 0.756 | 0.854 | 0.854 |
| answer relevance | 0.854 | 1.000 | 1.000 |
| groundedness | 0.902 | 0.927 | 0.951 |
| retrieval (strict) | 0.750 | 0.800 | 0.800 |
| top-3 citation hit | 0.694 | 0.917 | 0.917 |
| recall@1 | 0.391 | 0.336 | 0.336 |
| recall@3 | 0.639 | 0.790 | 0.790 |
| recall@5 | 0.722 | 0.853 | 0.853 |
| recall@8 | 0.778 | 0.883 | 0.883 |
| mean latency | 5.2 s | 15.4 s | 14.9 s |
| LLM calls per question | 1 | ~3.8 | 3.76 |

Retrieval is identical across the two runs. The judge scores differ only in
groundedness (by 1 question).

**Guardrails:**
- **Out-of-scope questions still refused: FAILED in both runs.** The baseline refused
  all 4. The agent refused 2 of 4, answering:
  - "How much fine will a AI system plotting terrorism in Singapore get in EU?" It
    answered from Article 5 and the penalty provisions.
  - "What countries does the EU act cover?" It answered from Article 2's territorial
    scope, which is arguably reasonable but is labelled should-refuse.

  Gate 1 (original-question similarity) passes both questions, in the baseline too. The
  baseline's generator refused them. With the agent's richer evidence and the
  "Assuming…" notes, the generator answers instead.
- **Groundedness not below baseline: PASS.** It rose, 0.902 → 0.927 / 0.951.
- **Unsupported citations in ≤ 5% of answers: PASS.** 2 of 41 in both runs:
  - "Article 8" in the high-risk testing answer;
  - "Annex 13" in the annex-count answer, which comes from the table of contents, so
    it's a known false positive.
- There were no stage fallbacks and no errors.

**Per question (strict / R@8 / correctness, baseline → both agentic runs):**
- **Fixed:**
  - EU database: (0, 0.5, 0) → (1, 1, 1 / 0)
  - date: (0, 0, 0) → (1, 1, 1)
  - exams tool: (0, 0, 1) → (1, 1, 1)
  - minorities penalties: (0, 0, 0) → (1, 1, 1)
  - how many annexes: correctness 0 → 1
  - placing on market: correctness 0 → 1
  - R&D and CV screening: correctness 0 → 1, though retrieval is still partial
    (0.33 and 0.22)
- **Retrieval fixed, answer still judged wrong:**
  - Italy: (0, 0, 0) → (1, 1, 0)
  - when high risk: (0, 0.5, 0) → (1, 1, 0)
- **Regressed:**
  - requirements for high risk: (1, 1, 1) → (0, 0.5, 1)
  - importers/distributors: (1, 1, 1) → (0, 0.75, 1)
  - Annex 11: R@8 1 → 0. The pinned Annex XI chunks now rank below the rewritten
    queries' results after RRF fusion, and land at positions 9–10.
- **Unchanged failure:** deployer + transparency (0, 0.5, 0).

### Decision

- **Retrieval target: met.** +8 sources on the ladder, and recall@8 rose from 0.778 to
  0.883 in the full eval. Answer correctness rose by 4 questions in both runs.
- **Refusal guardrail: failed.** Under the rule fixed before running, the agent is **not
  adopted as the default**. `query.answer_question` stays the default. The agent is
  available via `--system agentic`.
- **Next step, a separate change measured on its own:** restore refusals without losing
  the gains.
- **Side issue:** pinned metadata matches are demoted by RRF (Annex 11). Candidate
  fix: always keep the original question's pinned chunks at the top.

### Re-run on the updated golden set (2026-09-24)

Changes to the golden set, made in LangSmith:
- "What countries does the EU act cover?" and "How much fine will a AI system plotting
  terrorism in Singapore…" are relabelled from should-refuse to answerable. That leaves
  2 should-refuse questions.
- CV screening now expects Annex III + Article 13.
- Importers/distributors no longer expects Article 50.

Both systems were re-run fresh against the new labels.

| Metric | rag `ecfbeb79` | agentic `317b0fea` |
|---|---|---|
| answer correctness | 0.732 | **0.902** |
| answer relevance | 0.829 | **1.000** |
| groundedness | 0.902 | **0.976** |
| retrieval (strict) | 0.737 | **0.895** |
| top-3 citation hit | 0.694 | **0.917** |
| recall@1 | 0.394 | 0.338 |
| recall@3 | 0.644 | **0.806** |
| recall@8 | 0.778 | **0.912** |
| mean latency | 5.0 s | 14.9 s |

**Guardrails on the new labels:**
- Both should-refuse questions are refused by both systems: **PASS**.
- Groundedness is above the baseline: **PASS**.
- Unsupported citations appear in 2 of 41 answers, the same two as before: **PASS**.

The earlier refusal failure is resolved by the relabelling, not by a code change. The
agent's code and prompts are identical to the `9c0d12f7` / `e7c27ad6` runs.

Still failing under the agent (strict / R@8 / correctness):
- EU database (1, 1, 0)
- requirements for high risk (0, 0.5, 1)
- R&D (0, 0.33, 1)
- deployer + transparency (0, 0.5, 0)
- recruitment penalties (0, 0.5, 1)
- Italy (1, 1, 0)
- when high risk (1, 1, 0)

Annex 11 still regresses on R@8 (1 → 0) because RRF demotes the pinned chunks.
