# Next-fix analysis: reranking vs hybrid search vs similarity threshold

Date: 2026-09-24
Experiment analysed: `eu-ai-act-rag-342c66bc`. Pipeline: TOP_K = 8, metadata-first
retrieval, chapter/section Location line in the prompt.
Related: `results-comparison-afc59a56-vs-342c66bc.md`.

## 1. The question

After the metadata and K=8 fixes, answer relevance is 0.732 and correctness 0.683,
and **10 answerable questions are still refused**. Three candidate fixes were on the
table:

1. **Revisit the similarity threshold.** This is refusal gate 1, `SIMILARITY_THRESHOLD = 0.45`.
2. **Hybrid search.** Combine dense (embedding) similarity with keyword (BM25) search.
3. **Reranking.** Re-score a larger candidate pool with a cross-encoder and keep the best 8.

The goal was to pick the next fix from evidence rather than habit. Each option
rests on an assumption about *why* questions fail, so the first step was to find out
why they fail.

## 2. What was measured, and why

### 2.1 Which refusal gate fires

The pipeline can refuse at two points:
- **Gate 1:** the top similarity is below 0.45, and the LLM is never called.
- **Gate 2:** the LLM itself replies with the refusal sentence.

The threshold is only worth tuning if gate 1 is what's firing. So every question was
re-run through retrieval (local and free) to record its top similarity.

### 2.2 Where the missing sources rank

Reranking and hybrid search only help if the right chunk exists somewhere a better
ranker could find it. For every expected source in the golden set (61 sources across
36 answerable questions), `analysis/rank_diagnostic.py` recorded its rank out of all
373 chunks under five methods:

| Method | What it is |
|---|---|
| Dense | the current retrieval: bge-small cosine similarity |
| BM25 | keyword search (standard BM25, k1 = 1.5, b = 0.75) |
| Hybrid | dense and BM25 fused with equal-weight Reciprocal Rank Fusion (k = 60) |
| Rerank (dense pool) | `BAAI/bge-reranker-base` cross-encoder over the dense top 50 |
| Rerank (hybrid pool) | same reranker over the hybrid top 50 |

The headline number is how many of the 61 expected sources land in the **top 8**,
the slots the LLM actually sees.

**Note on the baseline:** the Dense column is *pure* similarity. It leaves out
metadata-first pinning, so named-provision questions (e.g. Article 97) look worse
here than they do in the live pipeline. That keeps the comparison between methods
fair.

## 3. Findings

### 3.1 Similarity threshold: can't be fixed by tuning

**Gate 1 never fired.** Every refusal of an answerable question came from the LLM
(gate 2).

| Question group | Top similarity |
|---|---|
| Answerable questions (all 37) | 0.63 – 0.85 |
| Out-of-scope questions (all 4) | 0.67 – 0.75 |

- **Lowering the threshold** changes nothing: nothing currently falls below 0.45.
- **Raising it** can't turn it into a scope filter. The out-of-scope questions
  ("summarize article 114", "2027 amendment", "what countries…", "terrorism in
  Singapore") score *higher* than the weakest answerable ones. To block them all,
  the threshold would have to exceed 0.75, which would also block most answerable
  questions.
- The 4 out-of-scope questions are refused correctly today, but by the LLM, not the
  threshold.

**Why:** bge-small gives every question about AI regulation a similarity of 0.6 or
more against this corpus, because the whole corpus is about AI regulation. Similarity
measures topic, not whether the answer exists. So "revisit the similarity score" isn't
a fix for either problem.

### 3.2 Hybrid search: no net gain

| Method | Expected sources in top 8 (of 61) |
|---|---|
| Dense (current) | **37** |
| BM25 only | 29 |
| Hybrid (equal RRF) | 36 |

BM25 wins where the question uses the Act's own words, and loses badly where the
wording differs:

| BM25 wins | Dense → BM25 | BM25 losses | Dense → BM25 |
|---|---|---|---|
| bias examination → Article 10 | 12 → 1 | logging → Article 12 | 1 → 67 |
| penalties (minorities) → Article 99 | 50 → 1 | entry into force → Article 113 | 2 → 111 |
| R&D → Article 2 | 103 → 3 | general-purpose AI → Recital 97 | 1 → 15 |
| recruitment → Annex III | 166 → 32 | exclusions → Recital 25 | 1 → 31 |

Equal-weight fusion averages the wins and losses into a slight net loss (36 vs 37).

**Why not now:** there's no evidence it helps overall. A *weighted* fusion, or BM25
only for questions containing legal terms, might recover the wins without the losses.
But that means tuning a new component against a 36-question test set, which risks
overfitting. It's parked, not rejected.

### 3.3 Reranking: a real but modest gain, too slow as tested

| Method | Expected sources in top 8 (of 61) |
|---|---|
| Dense (current) | 37 |
| Rerank the dense top 50 | **40** |
| Rerank the hybrid top 50 | **40** |

| Reranker wins | Dense → rerank | Reranker losses | Dense → rerank |
|---|---|---|---|
| "dated as?" → Preamble | 41 → 4 | Annex III (Italy, children) | 41 → 14 dense pool, 349 hybrid pool |
| EU database → Annex VIII | 49 → 6 | requirements → Article 9 | 2 → 10 |
| deployer → Article 3 | 31 → 4 | general-purpose AI → Article 3 | 5 → 14 |
| penalties → Article 99 | 50 → 1 | R&D → Recital 25 | 13 → 35 |
| bias → Article 10 | 12 → 1 | sandboxes → Article 57 | 2 → 7 |
| systemic risk → Article 51 | 7 → 1 | Annex XI | 5 → 13 (covered by metadata pinning) |
| testing → Article 60 | 7 → 2 | | |

- **Ceiling:** reranking can only reorder what's in the candidate pool. Sources
  ranked beyond 50 (Annex III for recruitment and CV screening, all of CV
  screening's articles) are out of reach.
- **Cost:** bge-reranker-base took about 35 seconds per question on this CPU for 50
  candidates. The pipeline's current mean latency is about 4.8 seconds, so as tested
  it's unusable. Viable versions would be:
  - a smaller cross-encoder (MiniLM-class, roughly 20× faster, likely less accurate);
  - a pool of about 20;
  - a hosted reranking API (Cohere, Jina, Voyage).

**Why:** it's the only ranking change that improved the top-8 count. The +3 of 61 is
real but small, and it has to become about 10× cheaper before it's usable.

### 3.4 What none of the three fixes

**LLM over-refusal (answer quality problem).** For "is sex trafficking a prohibited
AI practice?" and "is an AI system that provides entertainment to children a
prohibited AI practice?", Article 5 was at **rank 1** and the LLM still refused.
Other refused questions also had usable evidence in the top 8 (recruitment: Article
99 at rank 6; deployer: Article 13 at rank 8). Prompt rule 4 ("never infer beyond
what the text states… when in doubt, refuse") stops the model applying a provision
to the user's scenario. No retrieval change helps when retrieval already worked.

**Multi-step scenario questions.** "If we deploy AI for CV screening, what
obligations apply?" needs this chain:

> CV screening → Annex III(4) (employment) → high-risk → Articles 9–16

Every expected source ranks 60–345 under **every** method. The question's words
don't resemble any single chunk. This needs the question rewritten or split into
steps before retrieval, not better ranking.

**Annex III ranks poorly across the board.** For employment and children scenarios,
Annex III ranks 27–349 under dense, BM25, hybrid and reranking alike. When all four
methods agree a chunk is irrelevant, the likely cause is the chunk itself: how Annex
III is split and headed. Ranking isn't the problem.

## 4. Recommendation and rationale

| Priority | Fix | Rationale |
|---|---|---|
| 1 | **Relax the prompt's refusal rule.** Let the model answer from the evidence it has and state the gaps, instead of refusing outright. | Targets the biggest failure (10 refusals; at least 2 had the answer at rank 1). Cheap, with no latency cost. Risk: the 4 out-of-scope questions, where refusal is correct, must still be refused, so watch them in the next eval. |
| 2 | **Investigate how Annex III is chunked.** | All four methods agree it's hard to find, which points at the chunks, not the ranking. It's cited in 6 golden questions. |
| 3 | **Reranking, in a cheaper form** (small model, smaller pool, or API). | The only ranking change that helped (+3/61). Only worth it once it's about 10× faster. |
| 4 | **Rewrite or break up scenario questions** before retrieval. | The only approach that can reach CV-screening-style questions. It's more work, so it goes after the cheaper fixes. |
| — | Hybrid search | Parked. No net gain as tested; revisit with weighted fusion if reranking can't be made cheap. |
| — | Similarity threshold | Not a fix. Similarity can't separate in-scope from out-of-scope on this corpus. Keep the gate as a cheap guard for truly off-topic input, not as a quality control. |

## 5. Corrections and limitations

- **Correction:** I earlier guessed that "what is this document dated as?" fails
  because chunking strips the date as a page header. That was wrong: the Preamble
  chunk contains "of 13 June 2024". It fails on ranking (dense rank 41), and the
  reranker lifts it to rank 4.
- **Small test set:** 36 questions and 61 expected sources, so a difference of ±1–2
  in the top-8 count is within noise. The +3 for reranking is suggestive, not
  conclusive.
- **Primary sources only:** "expected source" means the golden set's primary
  sources. A method that surfaces a useful unlisted chunk gets no credit.
- **One reranker tested:** only `bge-reranker-base` with a 50-candidate pool. A
  stronger reranker (e.g. `bge-reranker-v2-m3`) might gain more; a faster one will
  likely gain less.
- **Not a full eval:** this is a ranking diagnostic, not an end-to-end run. Answer
  scores weren't re-measured under the alternative methods.

## 6. Reproducing

```powershell
.\venv\Scripts\python.exe analysis\rank_diagnostic.py eval_all_eu-ai-act-rag-342c66bc.xlsx
```

It runs fully locally at no cost. The first run downloads the reranker (about
1.1 GB), and the full run takes about 1.5 hours on CPU.

## 7. Addendum: embedding model and Annex III chunking

Script: `analysis/embedding_chunking_diagnostic.py`. Same measure as above:
expected sources in the dense top 8, out of 61, with no metadata pinning.

| Variant | Top 8 |
|---|---|
| bge-small (current) | **37** |
| bge-base | 35 |
| bge-large | 36 |
| bge-small + Annex III split into one chunk per area | 37 |

**Bigger embedding models: no net gain.** They reshuffle rather than improve.
bge-large lifts Annex VIII (49 -> 7), Article 10 for bias (12 -> 2) and the
preamble (41 -> 22), but drops Article 60 (7 -> 19), Article 3 (3 -> 16)
and Annex XI (5 -> 14). Switching isn't justified.

**Splitting Annex III per area: consistently better ranks, but not enough.**
Annex III's rank improves on every question (CV screening 81 -> 36, Italy
41 -> 15, "when high-risk" 27 -> 13, student tool 8 -> 2), but none of the
misses crosses into the top 8. The packed chunk *was* diluting the embedding;
fixing it is right but isn't sufficient on its own.

**Conclusion:** with the question as the only query, dense retrieval tops out at
about 37/61, whatever the model, ranking method or Annex III chunking. The
remaining misses are a *query* problem: scenario wording ("CV screening",
"R&D", "children in Italy") doesn't resemble the Act's legal wording, and
multi-provision questions need more than one lookup. The next lever is on the
query side: LLM query rewriting (one extra call that restates the question in
the Act's terms) as a cheap first test, then an agent for multi-step questions.
