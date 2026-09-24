# Chunking strategy: compare whole-corpus strategies, then commit the winner

> Status: **Done 2026-09-24. Variant B committed.** Replaces the earlier draft that hand-picked
> Article 3 / Annex III / the preamble. That draft was tuned to failing test
> questions, i.e. overfitting.

## Context

Retrieval is stuck at about 37/61 expected sources in the dense top 8. Every
ranking-side change was measured as roughly neutral (reranker +3, hybrid -1,
bge-base/large -2/-1). The remaining suspect is how the text is cut into chunks.

Today's chunker (`chunking.py`) **packs** neighbouring paragraphs of the same
provision up to 500 tokens, on the untested assumption (module docstring) that short
pieces "embed poorly". Packing merges neighbours whether or not they share a topic:
`Article 3(1-9)` holds 9 unrelated definitions, and `Annex III(4-5)` holds employment,
credit, insurance and emergency services. Separately, **overlap** prepends 50 tokens of
the previous chunk everywhere, including across unrelated provisions.

This plan tests **general strategies applied to the whole corpus**, with no
provision-specific rules, and commits whichever scores best overall.

## The four strategies

| Variant | Packing | Overlap |
|---|---|---|
| **A. Today** | size: pack same-provision neighbours up to 500 tokens | document-wide |
| **B. Today + overlap fix** | size | window-only |
| **C. Structural** | none: one chunk per structural unit | window-only |
| **D. Semantic** | merge same-provision neighbours only when similar | window-only |

### Packing modes (`config.CHUNK_PACKING`)

- **`"size"`** is today's `_pack`: merge while the same `_citation_root` and under 500
  tokens.
- **`"none"`** means every unit from the parser is its own chunk: each numbered
  paragraph, definition, annex point, recital and preamble clause. Units over 500
  tokens are still windowed (`_windows`), so the size limit holds.
- **`"semantic"`** is `_pack` with one extra merge condition: the incoming unit's
  embedding must have cosine similarity >= a threshold with the previous unit's.
  The same-provision rule and the 500-token budget still apply.
  - **The threshold comes from the corpus, not the test set:** the median
    similarity of all adjacent same-provision unit pairs, computed at chunk
    time. So roughly half of adjacent pairs, the more related half, are
    merge-eligible. It's fixed at the median in advance, never tuned against
    the golden questions.
  - Unit embeddings use the same bge-small passage encoder
    (`embeddings.embed_passages`).

### Overlap scope (`config.OVERLAP_SCOPE`)

- **`"document"`** is today's behaviour: every chunk gets the previous chunk's
  last 50 tokens.
- **`"window"`** carries overlap only into part 2+ of a unit that `_windows` cut at an
  arbitrary token position. That's the only place a sentence can be split. At
  structural boundaries a chunk starts with its own text.

### Preamble: same rule, no special case

`parse_preamble` currently returns the whole pre-recital text as one unit. It becomes a
structural parser like the others. It splits at the preamble's own clause starts:
the title block (up to "(Text with EEA relevance)"), "THE EUROPEAN PARLIAMENT…",
each "Having regard to…" / "After…" / "Acting in accordance…" clause, and "Whereas:".

The preamble's `_citation_root` changes from a unique root to a shared
`("Preamble",)`, so each packing mode treats it exactly like recitals:
- `"size"` repacks it into about one chunk, as today;
- `"none"` keeps one chunk per clause;
- `"semantic"` merges similar clauses.

All parts keep citation `Preamble`, so the golden `preamble 0` still matches.

## Code changes

- **`config.py`:**
  - `CHUNK_PACKING = "size"`
  - `OVERLAP_SCOPE = "document"`
  - `SEMANTIC_PACK_PERCENTILE = 50`

  These defaults reproduce today's chunks exactly until a winner is chosen.
- **`chunking.py`:**
  - `_pack` (line 422): branch on `CHUNK_PACKING`; the `"semantic"` branch adds the
    similarity check.
  - A small helper computes unit embeddings and the percentile threshold once per
    `chunk_document` call.
  - `_add_overlap` (line 503): add the `"window"` rule (`chunk["part"] >= 2` only).
  - `parse_preamble` (line 260) and `_citation_root` (line 379): the preamble
    clause parse and shared root.
  - The module docstring describes the chosen strategy and why.
- **`analysis/chunking_variants_diagnostic.py`** (new):
  - extracts the PDF once (`ingest.extract_pdf_text`);
  - for each variant, sets the two config values, runs `chunking.chunk_document`,
    and embeds in memory (no Chroma write);
  - scores all golden questions with the same method as
    `analysis/embedding_chunking_diagnostic.py`.
- Unchanged: `query.py`, `provision_refs.py`, `eval_retrieval.py`, `ingest.py`.

## Measurement and decision rule (fixed before running)

**Reported per variant:**
- **Primary:** expected sources in the dense top 8, out of 61.
- **Crowding check:** mean number of *distinct provisions* in the top 8.
  Smaller chunks can fill the slots with one article.
- **Recall@1/3/5**, chunk count, median and max chunk tokens (max must be ≤ 500).
- **Target ranks, for information only:** deployer, "dated", and Annex III for #38,
  CV screening and Italy. These don't decide the winner.

**Decision:**
1. **Sanity check:** variant A must reproduce 37/61, or stop.
2. Pick the variant with the highest top-8 count.
3. It must beat A by **≥ 3**, the noise margin on 61 sources, or A stays. If B ties
   or beats A, B is still kept, because it's the lower-risk fix.
4. **Ties** go to fewer chunks, the simpler index.

The winner is chosen on the aggregate over all 36 questions, never on the 7 failing
ones.

## Commit and verify

1. Set the winning `CHUNK_PACKING` / `OVERLAP_SCOPE` as defaults, then run
   `python ingest.py --reset`. `--reset` is required so stale ids like
   `article-3-1-9` don't linger.
2. Run `python inspect_chunks.py` to regenerate `chunks_export.txt`, then spot-check it.
3. Run `python run_all_evals.py --concurrency 2` and compare with `e6e5c965`. This is the
   real pipeline, with metadata pinning.
4. Record the variant table and eval comparison in a Results section here. Whatever
   still fails defines the agent plan's scope.

## Known interactions and risks

- **The strict completeness check penalises finer chunks.** "Summarize article 112"
  needs every paragraph of the article. With one chunk per paragraph, that's up to 13
  chunks, but metadata pinning caps at 6 of the 8 slots. The strict score could drop
  under C/D for reasons that are about the metric, not about retrieval quality.
  Report recall@k alongside it, and don't treat a strict-score drop on these
  questions alone as a regression.
- **More chunks:** C could roughly double the index. That's cheap at this corpus size.
- **Golden-set dependence:** choosing among 4 whole-corpus strategies by aggregate
  score is far less fitted than per-question rules, but it's still one 36-question
  set. That's why there's a ≥ 3 margin.
- **Semantic packing adds an embedding pass at ingest:** unit embeddings are computed
  twice, once for packing and once for the final chunks. Acceptable at about 1,000
  units.

## Results (2026-09-24)

### Variant comparison (`analysis/chunking_variants_diagnostic.py`, dense top 8, bge-small)

| Variant | Top-8 / 61 | Hits@1 | Hits@3 | Hits@5 | Distinct provisions in top 8 | Chunks | Median tok | Max tok |
|---|---|---|---|---|---|---|---|---|
| A. Today | **37** | 18 | 28 | 32 | 8.31 | 373 | 370 | 497 |
| B. Today + overlap fix | **38** | 16 | 29 | 35 | 8.33 | 373 | 319 | 497 |
| C. Structural | **39** | 19 | 30 | 35 | 5.19 | 914 | 104 | 497 |
| D. Semantic (median threshold) | **39** | 14 | 29 | 36 | 6.92 | 644 | 137 | 497 |

Target ranks (informational only, rank of the expected source):

| Target | A | B | C | D |
|---|---|---|---|---|
| deployer (Article 3) | 31 | 49 | 1 | 8 |
| dated (Preamble) | 41 | 52 | 121 | 93 |
| #38 Annex III | 27 | 22 | 5 | 3 |
| CV screening Annex III | 81 | 118 | 118 | 70 |
| Italy Annex III | 41 | 43 | 28 | 37 |

**Decision:** the sanity check passed (A = 37). The best variants, C and D at 39, beat A
by only 2, under the ≥ 3 margin. B (38) beats A, so **B is committed**:
`CHUNK_PACKING = "size"`, `OVERLAP_SCOPE = "window"`.

Notes:
- C and D pay for their +2 with crowding. Distinct provisions in the top 8 drop from
  about 8.3 to 5.2 and 6.9, and the index grows 2.5x and 1.7x.
- C and D sharply improve single-definition and single-area lookups (deployer, #38),
  but bury the preamble (dated).

### Full pipeline eval: `e6e5c965` (A) -> `275ca613` (B)

| Metric | e6e5c965 | 275ca613 |
|---|---|---|
| groundedness | 0.927 | 0.902 |
| answer_relevance | 0.854 | 0.854 |
| answer_correctness | 0.780 | 0.756 |
| retrieval (strict) | 0.700 | 0.750 |
| top-3 citation hit | 0.694 | 0.694 |
| recall@1 | 0.433 | 0.391 |
| recall@3 | 0.623 | 0.639 |
| recall@5 | 0.674 | 0.722 |
| recall@8 | 0.757 | 0.778 |

Per-question changes:
- **Fixed:** importers/distributors/deployers (strict 0 -> 1) and bias examination
  (strict 0 -> 1, R@8 0 -> 1).
- **Regressed:** students' exams (R@8 0.5 -> 0; the answer is still correct).
- **Judge flip with unchanged retrieval:** "placing on market vs making available"
  (correctness 1 -> 0; strict and R@8 are both still 1). This is generation or
  judge variance, not chunking.

### Still failing, which defines the agent plan's scope

strict < 1, R@8 < 1, or correctness < 1:
- EU database data (Article 71 / Annex VIII): strict 0, R@8 0.5
- CV screening (Annex III + Articles 9-16): strict 0, R@8 0
- R&D / high risk (Recitals 24-25, Article 2): strict 0, R@8 0
- document date (Preamble): strict 0, R@8 0
- deployer + transparency (Articles 3, 13): strict 0, R@8 0.5
- recruitment penalties (Article 99, Annex III): strict 0, R@8 0.5, correct
- students' exams (Article 6, Annex III): strict 0, R@8 0, correct
- children in Italy (Annex III): strict 0, R@8 0
- when high risk (Article 6, Annex III): strict 0, R@8 0.5
- deployer penalties for minorities (Articles 99, 5): strict 0, R@8 0
- how many annexes (no retrieval target): correctness 0
- placing on market vs making available: correctness 0 (retrieval fine)

Annex III is a missing source in 5 of these. It stays reachable only through vocabulary-bridging,
which dense retrieval on this corpus doesn't do. That is the case for the agent or
query-rewrite step, not for more chunking changes.
