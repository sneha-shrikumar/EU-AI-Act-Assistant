# Surface Chapter/Section metadata to the LLM, then raise TOP_K to 8

> Status: **Done 2026-09-24. Results below.**

## 1. Metadata fix ("Section 4" failure)

Row 6 of experiment `eu-ai-act-rag-afc59a56`: *"which section talks about
notifying everyone along with the procedure?"*, reference `Section 4, article 30`.

- Retrieval was perfect: `Article 30(1-5)` at rank 1 (R@1 = 1).
- The answer named only **Article 30** -> relevance 0, correctness 0.

Root cause is generation, not retrieval or chunking. The chunk metadata is
correct in Chroma (`section: "4 - Notifying authorities and notified bodies"`,
`chapter: "III - HIGH-RISK AI SYSTEMS"`), and the breadcrumb is even in the
chunk text. But `llm.build_context_block` labels each excerpt only with
`[citation]`, and the system prompt tells the model to cite "in the form the
excerpt is labeled with". The hierarchy therefore reads as body noise and never
makes it into an answer.

### Changes

- `llm.py::build_context_block` -- add a `Location:` line under the citation
  label, built from the chunk's `chapter`/`section` **metadata** (not parsed
  from text), e.g.
  `Location: Chapter III (HIGH-RISK AI SYSTEMS) > Section 4 (Notifying authorities and notified bodies)`.
  Omitted for recitals/annexes/preamble, which have no chapter/section.
- `llm.py::SYSTEM_PROMPT` -- new rule: the Location line is part of the
  excerpt; when a question asks which chapter/section/article covers something,
  name the full location (chapter, section and article), citing the excerpt.

Groundedness is unaffected: the groundedness judge sees citation + chunk text,
and the chunk text's breadcrumb already states the same Chapter/Section, so a
"Section 4" claim stays traceable.

## 2. TOP_K 5 -> 8

- `config.TOP_K = 8`.
- `eval_retrieval.K_VALUES = (1, 3, 5, 8)` so @8 is reported alongside the
  existing @5 baseline (precision@8 will drop by construction -- that is the
  cost the precision/recall plan was built to expose).
- Excel column widths extended for the two new columns.

## Verification

1. `python eval_retrieval.py --selftest` still passes.
2. Run the row-6 question through `answer_question` and confirm the answer
   names Section 4 and Article 30.
3. Full experiment (`python evals/run_langsmith_eval.py`) at K=8, then
   `eval_retrieval.py` against it, compared with the `afc59a56` baseline
   (R@5 0.646, retrieval score 0.575, relevance/correctness 63.4%).

## Results (2026-09-24)

Section 4 question: the answer now reads "Chapter III (HIGH-RISK AI SYSTEMS),
Section 4 (Notifying authorities and notified bodies), Article 30". Relevance
and correctness both went 0 -> 1.

Bug found along the way: the first K=8 run (`eu-ai-act-rag-d080f125`) had 4
errored rows. `query._get_collection` opened a new Chroma `PersistentClient`
per question, and under `--concurrency 4` those clients raced during Chroma's
init. The collection is now opened once, behind a lock. Discard that run.

Clean run `eu-ai-act-rag-aa5b9822` (K=8, 0 errored rows) vs `afc59a56` (K=5):

| metric | K=5 baseline | K=8 |
|---|---|---|
| strict retrieval score | 0.575 (17/40 fail) | 0.650 (14/40 fail) |
| R@1 / R@3 / R@5 | 0.38 / 0.57 / 0.65 | 0.38 / 0.57 / 0.65 (identical, ranking unchanged) |
| R@8 | n/a | 0.73 |
| P@5 / P@8 | 0.24 / n/a | 0.24 / 0.20 |
| answer relevance | 63.4% | 63.4% (26/41) |
| answer correctness | 63.4% | 64.1% (25/39) |
| groundedness | 100% (hand-scored) | 95.1% (39/41, first judge-scored run) |

Raising K improved retrieval coverage but not the answers. 11
`should_answer: yes` questions are still refused, and that is now the main
bottleneck.
