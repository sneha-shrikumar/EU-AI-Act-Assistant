# EU AI Act Assistant: from basic RAG to a measured, agentic pipeline

A question-answering assistant for the EU AI Act, (Regulation (EU) 2024/1689).
It answers only from the Act's text and cites the article, annex or recital behind
every claim. The project covers how it was built, how quality was defined and
measured, and why the agentic version was shipped.

> **Portfolio note:** this is an AI product management project. The code is real and
> runs, but the main output is the decision process: a KPI, a golden dataset, LLM
> judges calibrated against human labels, a release gate fixed before each
> experiment, and a record of what worked and what didn't. Read
> **[the case study](docs/case-study.md)** for the full story.

## The problem

Product, legal and compliance teams keep asking the same kinds of questions: *Is our
CV-screening tool high-risk? What must a deployer do? What's the fine?* The Act is
144 pages across 113 articles, 13 annexes and 180 recitals. A general chatbot answers
fluently, but it can cite the wrong article or invent obligations. In compliance work
**an answer that isn't grounded is worse than no answer at all.**

**The KPI:** answer correctness on a 41-question golden set.

**Guardrails that must not get worse:**
- Groundedness.
- The rate of refusing out-of-scope questions.
- Unsupported citations stay at or under 5%.

## Results

All three systems were scored on the same golden set and judges. Cost is what
OpenRouter actually charged, excluding the judges.

| System | Answer correctness | Groundedness | Answerable questions refused | Cost per question | Mean latency |
|---|---|---|---|---|---|
| Basic RAG (Haiku 4.5, one search) | 0.73 | 0.90 | 5 | **$0.005** | **5.0 s** |
| **Agentic RAG (Haiku 4.5, shipped)** | **0.88–0.90** | 0.93–0.98 | 0 | $0.022 | 15 s |
| Whole Act sent to Opus 5.5 (no retrieval, ceiling check) | 0.98 | 0.93 | 0 | $0.086 | 17 s |

Correctness is a share of the 41 golden questions.

Over the project, answer correctness went from **0.63 to 0.90**, and recall@8 went from
**0.73 to 0.94**. The agentic pipeline gets within about 3 questions of the Opus 5.5
ceiling at **about a quarter of the cost**.

*Experiment IDs:*
- RAG: `ecfbeb79`
- Agentic: `317b0fea`, and `35fe63c2` after the fix
- Opus 5.5: `f0d6b8ee`

The per-question results are in [`evals/results/`](evals/results/).

## How it works

```mermaid
flowchart LR
    PDF[EU AI Act PDF] --> P[Structural parser<br/>articles, paragraphs, annexes,<br/>recitals, chapter/section metadata]
    P --> C[Chunks ≤500 tokens<br/>bge-small embeddings]
    C --> DB[(Chroma)]

    Q[Question] --> A1["① Analyse<br/>sub-questions, assumptions,<br/>queries in the Act's vocabulary"]
    A1 --> R["② Retrieve<br/>original + rewritten queries<br/>metadata pinning for named provisions<br/>RRF fusion"]
    DB --> R
    R --> A3["③ Reflect (≤2 rounds)<br/>is each sub-question covered?<br/>follow cross-references found in the text"]
    A3 -- gaps --> R
    A3 --> G[Answer<br/>Haiku 4.5, excerpts only,<br/>every claim cited]
    G --> CK[Citation post-check<br/>+ refusal gates]
```

Every stage is traced in LangSmith. The stages run in a fixed order with fixed limits,
not in an open-ended loop:
- at most 5 LLM calls per question;
- follow-up searches only for provisions named in retrieved text, never from the
  model's memory.

## Quickstart

```bash
python -m venv venv && venv\Scripts\activate    # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                              # add OPENROUTER_API_KEY (and LangSmith keys for evals)
```

1. Download the Act from EUR-Lex:
   [Regulation (EU) 2024/1689](https://eur-lex.europa.eu/eli/reg/2024/1689/oj).
   Save the PDF as `data/eu_ai_act.pdf`.
2. Build the index, then ask questions:

```bash
python ingest.py --reset          # parse, chunk, embed, write chroma_db/
python query.py                   # interactive: agentic pipeline (default)
python query.py --baseline        # interactive: basic single-search RAG
```

To run the evaluation, you need LangSmith access to the golden dataset. It costs money:
about $1 per run for agentic, plus the judges.

```bash
python run_all_evals.py --system agentic --concurrency 2
python run_all_evals.py --limit 3                        # cheap smoke test
python evals/run_langsmith_eval.py --system full_doc     # Opus 5.5 ceiling check (~$3.50)
```

## Repo map

| Path | What's there |
|---|---|
| [`docs/case-study.md`](docs/case-study.md) | **Start here:** the full product story |
| [`docs/decision-log.md`](docs/decision-log.md) | Every change as hypothesis, result and decision, linked to its plan |
| [`docs/metrics.md`](docs/metrics.md) | What each metric means and how it's computed |
| [`docs/analysis/`](docs/analysis/) | Deep dives: retrieval options, run comparisons, the Opus 5.5 re-run |
| [`plans/`](plans/) | The original plan written before each change, with its results |
| `ingest.py`, `chunking.py`, `embeddings.py` | Parse the PDF, chunk it by structure, and embed the chunks |
| `query.py`, `provision_refs.py`, `llm.py` | Basic RAG: metadata-first retrieval, refusal gates, answer generation |
| `agent.py` | The agentic pipeline (analyse → retrieve → reflect → answer) |
| `baseline_full_doc.py` | The whole-Act Opus 5.5 ceiling check |
| `run_all_evals.py` | One command: run the golden set, judge, score retrieval, write a spreadsheet |
| `evals/` | LLM judges, human labels, LangSmith runner, golden dataset, per-run results |
| `eval_retrieval.py`, `eval_citation_ranking.py` | Retrieval metrics: strict pass, recall/precision@k, top-3 citation hit |
| `analysis/` | Offline diagnostics used to compare chunking, embeddings and agent stages |
| `results/` | Spreadsheets for every eval run, plus the hand-scored first baseline |

## Stack

- **Models:** Claude Haiku 4.5 generates the answers, and Claude Sonnet 5 is the judge.
  Both are called through OpenRouter.
- **Retrieval:** bge-small-en-v1.5 embeddings in Chroma.
- **Tracing and evaluation:** LangSmith.
- Python 3.
