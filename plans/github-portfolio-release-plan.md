# Plan: Publish the EU AI Act RAG project to GitHub as an AI PM portfolio piece

> Status: **Approved 2026-09-24**, implementing now.

## Context
`eu-ai-act-rag/` is a working RAG → agentic RAG system over the EU AI Act. It has a
41-question golden set in LangSmith, three LLM judges calibrated with human labels,
retrieval metrics, a release gate, and a baseline that sends the whole Act to
Opus 5.5.

The story is scattered across 16+ plan files, loose `.xlsx` and `.md` result files,
and LangSmith. There is no README, `requirements.txt` is incomplete, and the folder
isn't a git repo.

**Goal:** a clean repo that a hiring manager for AI PM roles can read top to bottom.
It should cover:
- what problem the project solves and why;
- how quality was defined and measured;
- what was tried and what moved the numbers;
- how the ship decision was made;
- what failed;
- the trade-offs and next steps.

**Decisions from the user:**
- Tidy the folder but keep the code flat.
- Write-up is a README plus `docs/case-study.md`.
- I run `git init` and make a local commit; the user creates the GitHub repo and pushes.

**Project rule (CLAUDE.md):** on approval, the first action is to save this plan as
`eu-ai-act-rag/plans/github-portfolio-release-plan.md`, before any other change.

## 1. Cleanup (repo root = `eu-ai-act-rag/`)

**Delete (temporary or generated):**
- `~$*.xlsx` Excel lock files
- `__pycache__/`

**Exclude via `.gitignore` (kept on disk, not published):**
- Already listed: `.env`, `venv/`, `chroma_db/`, `data/*.pdf`
- Add:
  - `.claude/`
  - `chunks_export.txt` (650 KB, rebuilt by `inspect_chunks.py`)
  - `*.pyc`
  - `~$*`

**Move (no deletions of results):**
- `eval_all_*.xlsx` and `eval_combined_*.xlsx` → `results/eval-sheets/`
- In `run_all_evals.py`, change its output path to that folder (one-line change).
- `eu-ai-act-rag-afc59a56 experiment 23sep.csv` → `results/baseline-handscored-afc59a56.csv`
- `golden-dataset for EU AI Act.csv` → `evals/golden_dataset.csv`
  - Update any code that reads the old path (grep first).
- Loose write-ups → `docs/analysis/`:
  - `results-comparison-afc59a56-vs-342c66bc.md`
  - `retrieval-options-analysis.md`
  - `evals/results/compare_*.md`
  - `evals/results/fulldoc_rerun_f0d6b8ee.md`
- The 4 project plans in the parent `Portfolio1 - RAG/plans/` → `eu-ai-act-rag/plans/`, so
  the decision history is complete. `multimodal-RAG/` is a separate project and is
  left out.

**Dependencies:**
- Complete `requirements.txt`. Add the packages the code imports that are missing:
  `openai`, `langsmith`, `openpyxl`, `numpy`.
- Pin every package to the version installed in `venv` (`pip freeze`, filtered to
  these names).

**Source PDF:**
- Keep it gitignored.
- The README links the official EUR-Lex source for Regulation (EU) 2024/1689 and says
  where to save it (`data/eu_ai_act.pdf`).

**Secret check before commit:**
- Grep the tracked files for API-key patterns (`sk-`, `lsv2_`, `OPENROUTER_API_KEY=` with
  a value).
- Confirm `.env` is ignored and `.env.example` holds only placeholders.

## 2. Write-ups

### `README.md` (for a reader who stops after one screen)
- A one-line pitch, and the problem: compliance and product teams need correct,
  cited answers about the AI Act.
- Headline results table (numbers taken from the result JSONs):

  | System | Answer correctness | Cost per question | Latency |
  |---|---|---|---|
  | RAG | 0.73 | $0.005 | 5 s |
  | Agentic | 0.88–0.90 | $0.022 | 15 s |
  | Whole Act to Opus 5.5 | 0.98 | $0.086 | 17 s |

- An architecture diagram in Mermaid, which GitHub renders: ingest → structural
  chunks → Chroma → metadata-first retrieval → agent stages → Haiku answer with
  citations.
- Quickstart: set up the venv, fill in `.env`, run `ingest.py`, run `query.py`
  (with `--baseline` for plain RAG), and run `run_all_evals.py`.
- Repo map, with links to the case study and the decision log.

### `docs/case-study.md` (the full story, PM voice)
1. **Goal and users.** The job to be done, and why correct citations matter more than
   fluent answers.
2. **Defining quality.**
   - The KPI is answer correctness.
   - Supporting metrics: groundedness and relevance; retrieval strict pass, recall@k,
     precision@k, and top-3 citation hit.
   - Guardrails: out-of-scope questions are refused, groundedness doesn't drop below
     the baseline, and unsupported citations stay at or under 5%.
3. **Golden dataset.**
   - 41 questions across single, multi-hop, exemption and out-of-scope categories,
     each with expected answers and sources.
   - Why it was relabelled (Q25 and Q26 became answerable), and the risk of moving the
     goalposts that relabelling creates.
4. **Evaluation system.**
   - First, hand scoring in LangSmith. It exposed that a refusal counts as trivially
     "grounded".
   - Then the switch to three Sonnet 5 LLM judges, using a different model from the
     one that generates answers.
   - Human-label calibration: `evals/human_labels.json` holds the corrections, and they
     are fed back into the judge prompts.
   - The one-command harness, `run_all_evals.py`.
5. **Iteration log.** For each step: hypothesis, change, result, and decision. The
   steps, in date order from the plan files:
   - LangSmith tracing
   - Citations always populated
   - Token-window chunking
   - Precision/recall@k
   - LLM judges
   - Chapter/section metadata and top-k 8
   - Metadata-first retrieval
   - Human calibration
   - Relaxing the refusal rule
   - Structural chunking (variant B)
   - Agentic RAG
   - The Annex 11 fix and shipping the agent as the default

   Include a metrics-over-time table. Note where a judge change makes runs not
   strictly comparable.
6. **Release gate.** The rule was set before the runs. The agentic pipeline failed it
   the first time (it answered 2 of the 4 should-refuse questions), so it was not
   shipped. It was shipped only after the relabelling and the Annex 11 fix, with the
   reasoning written down.
7. **Ceiling check: the whole Act sent to Opus 5.5 with no retrieval.**
   - Results: 40/41 correct, $3.53 per 41-question run.
   - What it shows about the size of the gap left for retrieval to close, and the
     caveat that generator size is mixed into that gap.
8. **Failures and what they taught.**
   - The grounded-refusal artifact.
   - Judge noise of 1–4 questions between single runs.
   - RRF fusion demoting pinned provisions (Annex 11).
   - The stale reference answer for Q26.
   - Groundedness judge misses on the ~190k-token context.
   - The LangSmith sync scoring glitch.
   - Running out of OpenRouter credits (402 error).
   - CV screening recall is still low.
9. **Trade-offs.** Quality vs cost vs latency across the three systems, and when you
   would pick each one.
10. **Next steps.**
    - Average several judged runs to reduce noise.
    - Fix the Q26 reference answer, and hand-check the full-document groundedness
      failures.
    - Grow the golden set.
    - Reduce agent latency (parallel stages, a smaller reflect model).
    - Run the eval gate in CI.
    - Add a cost/latency budget to the release gate.

### `docs/decision-log.md`
A table of date, plan file, decision, and outcome, linking each file in `plans/`.
Some plans don't state their outcome; for those, add a one-line "Status" to the plan.

### `docs/metrics.md` (optional, short)
- What each metric means, and how it's computed, with file pointers:
  - `evals/judges.py`
  - `eval_retrieval.py`
  - `eval_citation_ranking.py`
- The judge prompts, and the calibration rule for human labels.

**Where the numbers come from:**
- Every number is computed from `evals/results/*.json` and the LangSmith cost query,
  using the same one-off script as the aggregation in this session.
- Nothing is copied from memory. Earlier runs that have no JSON are quoted from their
  plan files and labelled as such.

## 3. Git (local only)
1. `git init`.
2. Check `git status`: no `.env`, `venv`, `chroma_db`, PDF, or lock files, and a total
   size of about 5 MB.
3. Make one commit ("Initial public release: EU AI Act RAG → agentic, with eval
   harness"), ending with the Co-Authored-By line.
4. Give the user the `gh repo create` and `git push` commands. They run them.

## Verification
- **Tracked files:** `git ls-files` contains only the intended files, and the
  secret-pattern grep over the tracked files comes back empty.
- **Code still works after the moves:** run the free offline self-tests (step 1 of
  `run_all_evals.py`, which checks the `provision_refs` and `eval_retrieval` parsers),
  and `python query.py --help`.
- **Links:** a small script checks that every relative link in `README.md` and `docs/`
  resolves to a file that exists.
- **Numbers:** every figure in the README and case study traces back to a named
  experiment ID.
- **After the user pushes:** check that the Mermaid diagram and tables render on
  GitHub.
