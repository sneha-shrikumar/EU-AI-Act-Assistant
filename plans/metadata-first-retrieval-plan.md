# Metadata-first retrieval for questions that name a provision

> Status: **Done 2026-09-24. Results below.** Supersedes the "metadata fix" part of
> `section-metadata-and-top-k-8-plan.md`. That plan only added a Location line
> to the LLM prompt and never touched retrieval.

## Problem

Retrieval is pure vector similarity. When a question names a provision
outright, e.g. "what does article 97 talk about?", "summarize article 112",
"what does Annex 11 talk about?" or "what is this document dated as" (the
preamble), the embedding often doesn't rank that provision's chunks highly.
bge-small is weak at matching bare numbers, even though every chunk already
carries exact `type` / `number` / `chapter` / `section` metadata. In the K=8 run,
"what does article 97 talk about?" retrieved no Article 97 chunk at all and
refused.

## Design

1. **Parse the question for named provisions** (`provision_refs.py`, new):
   `article(s) N`, `art. N`, `annex N` (arabic or roman), `recital(s) N`,
   `section N`, `chapter N` (arabic or roman), and `preamble`. Lists and ranges
   are supported ("articles 9 to 15", "recitals 24, 25"), and paragraph
   suffixes like "10(2)" are ignored. Roman numerals must be valid, so ordinary
   words aren't read as numbers. A bare "annexes" with no number matches nothing.
2. **Match against chunk metadata** from an in-memory copy of all chunk
   metadatas, loaded once per process:
   - article: `type == Article`, `number == N`
   - recital: `type == Recital`, N inside the packed range ("12-15")
   - annex: `type == Annex`, `number == roman(N)`
   - chapter: chapter number prefix
   - section: section number prefix; if a chapter is also named, only
     sections within it (section numbers repeat across chapters)
   - preamble: `type == Preamble`
3. **Pinned slots first.** Chroma `query(ids=matched_ids)` ranks the matched
   chunks by similarity. Up to `k - config.MIN_SEMANTIC_SLOTS` of them
   (6 at K=8) go first, and the remaining slots are filled by the normal
   unfiltered similarity search, with duplicates removed. This keeps whole-provision
   questions complete (Article 112 = 3 chunks) and leaves room for related
   context. A big match set (all of Section 4 = 16 chunks) is narrowed by
   similarity.
4. **Refusal gate 1 is skipped when anything was pinned.** The user named a
   real provision, so "nothing topically close" doesn't apply. A nonexistent
   provision ("summarize article 114") matches nothing and falls through to
   normal retrieval and the gate, as before.
5. Each retrieved chunk records `retrieved_by: "metadata" | "similarity"`,
   so the LangSmith retriever trace shows which path found it.

## Verification

- `python provision_refs.py` runs the parser selftest.
- Spot-check article 97, article 112, Annex 11, preamble and Section 4.
- Full experiment at K=8 plus `eval_retrieval.py`, compared with
  `eu-ai-act-rag-aa5b9822`.

## Results (2026-09-24)

`eu-ai-act-rag-f9bbf5eb` (metadata-first, K=8) vs `eu-ai-act-rag-aa5b9822`
(K=8, similarity only). Both runs had 0 errored rows.

| metric | similarity only | metadata-first |
|---|---|---|
| strict retrieval score | 0.650 (14 fail) | 0.700 (12 fail) |
| R@1 / R@3 / R@5 / R@8 | 0.38 / 0.57 / 0.65 / 0.73 | 0.43 / 0.62 / 0.67 / 0.76 |
| P@1 | 0.50 | 0.56 |
| answer relevance | 63.4% (26/41) | 70.7% (29/41) |
| answer correctness | 64.1% (25/39) | 69.2% (27/39) |
| groundedness | 95.1% | 95.1% |

Only 5 golden questions name a provision. Of the answer-score changes, only
"what does article 97 talk about?" (refused before, now answered and correct)
is directly caused by this change. The other three rows that changed name no
provision, so the pinning path never ran for them; treat those changes as
run-to-run LLM variance. "what is this document dated as?" still fails,
because it doesn't use the word "preamble".
