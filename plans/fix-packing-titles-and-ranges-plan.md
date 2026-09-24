# Fix the packing fallout: restore provision titles, make ranges parseable

> Status: **Done 2026-09-20.**

## Context

The chunking rewrite introduced **packing** — merging consecutive small provisions
until they approach 500 tokens — because the Act's natural units are tiny
(article paragraphs median 65 tokens, 85% under 150). Packing is what gets the
corpus from 905 chunks at a median of 78 tokens to 365 chunks at a median of 370.

Packing stays. But it invented a multi-provision metadata format (`number = "1-2"`,
`citation = "Recitals 1–2"`) that the rest of the pipeline cannot parse, and the
rewrite separately dropped provision titles from chunk text. Review surfaced three
symptoms; investigation found two real defects and one false alarm.

### What was actually wrong

| Symptom reported | Finding |
|---|---|
| Recitals 3, 8, 9, 12 missing | **False alarm.** All 180 recitals are present with their `(N)` markers. `inspect_chunks.py:39` calls `int(meta["number"])`, which throws on `"1-2"` and falls back to `0`, sorting all 53 packed chunks to the top of the export. Recital 3 sits at line 1410. The file is mis-ordered, not incomplete. |
| Article 7's heading missing | **Real.** `_provision_units` writes `title` to metadata but never into chunk text, and the breadcrumb prints only `Article 7`. `"Amendments to Annex III"` appears in 0 chunks. Affects all 113 articles; annexes get no breadcrumb at all. |
| — (not reported, found while investigating) | **Real.** `_merge` writes `paragraph = "1-5"` with an ASCII hyphen but `citation = "Article 70(1–5)"` with an en dash. `eval_retrieval.py`'s paragraph-completeness check compares these two strings, so it never matches and scores false 0s. |

Article-level eval scoring is unaffected: `parse_refs` (eval_retrieval.py:81) already
expands both dash types and ranges. `query.py` and `eval_citation_ranking.py` only
pass citations through `parse_refs`, so neither needs changes.

## Changes

### 1. `chunking.py` — restore provision titles to chunk text

The title is the most retrieval-relevant line in a provision and is currently absent.
Fold it into the breadcrumb line rather than a separate line, so it costs the token
budget once:

```
Chapter III: HIGH-RISK AI SYSTEMS
Section 1: Classification of AI systems as high-risk
Article 7: Amendments to Annex III
1. The Commission is empowered to adopt delegated acts...
```

- `_apply_context`: change the final line from `f"Article {number}"` to
  `f"Article {number}: {title}"` when a title exists.
- **Annexes currently get no breadcrumb at all** — `_apply_context` is only called
  from `parse_articles`. Give `parse_annexes` an equivalent breadcrumb
  (`Annex III: <title>`), since annex titles are just as informative.
- `_emit` already charges `breadcrumb` against the 500-token budget, so the longer
  breadcrumb is accounted for automatically — no budget change needed.

### 2. `chunking.py` — one dash character everywhere

Delete `RANGE_DASH` (the en dash) and use an ASCII hyphen in `_merge` for both
`citation` and `number`/`paragraph`. This fixes the hyphen/en-dash mismatch that
breaks paragraph completeness scoring, and stops grep/terminal tooling treating
the export as a binary file. `parse_refs` already accepts either character, so
nothing downstream regresses.

Also use a plain hyphen in the `chapter`/`section` metadata strings
(`f"{num} - {title}"`) for the same reason.

### 3. `inspect_chunks.py` — sort by the first number in a range

`sort_key` (line 36-44) must not fall back to `0` on a range. Add a small helper
that takes the leading number of `"1-2"` / `"12-15"`, and use it for both `number`
and `paragraph`:

```python
def _lead_int(value, default=0):
    """First number of a plain value or a packed range ("12-15" -> 12)."""
```

This restores document order in `chunks_export.txt`, which is what made the
recitals look missing.

### 4. `eval_retrieval.py` — make paragraph completeness range-aware

Two functions, both in the paragraph-completeness path only:

- **`CITATION_RE` (line 129)** — accept plural kind words (`Recitals`, `Articles`,
  `Annexes`) so packed citations parse at all.
- **`_citation_detail` (line 132)** — return the **set** of paragraph labels a
  citation covers, expanding `"1-5"` to `{"1","2","3","4","5"}`, instead of a single
  opaque label. Callers at line 218 union these into `covered`.
- **`build_paragraph_index` (line 151)** — expand a packed `paragraph` value the
  same way before adding to the index, so `corpus_paragraphs` holds individual
  paragraph labels. Also skip the `int(number_raw)` `continue` for recital-style
  ranges by reading the leading number.

Share one range-expansion helper between the two rather than duplicating it.

## Files

- `chunking.py` — `_apply_context`, `parse_annexes`, `_merge`, drop `RANGE_DASH`
- `inspect_chunks.py` — `sort_key` + new `_lead_int`
- `eval_retrieval.py` — `CITATION_RE`, `_citation_detail`, `build_paragraph_index`
- `plans/` — copy this plan in as the first action (per CLAUDE.md)

## Verification

1. `venv\Scripts\python.exe` over the real PDF via `chunk_document`, asserting:
   - `"Amendments to Annex III"` present; **0** article/annex chunks whose title is
     absent from their text
   - every chunk `<= 500` tokens (breadcrumb now longer — confirm nothing tips over)
   - `chunk["paragraph"] in chunk["citation"]` for every packed chunk (the dash fix)
   - 180/180 recitals, 113/113 articles, 13/13 annexes, no gaps, no duplicate ids
2. `python ingest.py --reset` then `python inspect_chunks.py`, and confirm
   `chunks_export.txt` opens in document order: `Recitals 1-2`, `Recital 3`,
   `Recitals 4-7`, `Recital 8`, ... — the check that directly answers the report.
3. `python eval_retrieval.py --selftest` (existing parser self-test) plus a direct
   check that `_citation_detail("Article 70(1-5)")` expands to five paragraph labels
   and that `build_paragraph_index` returns individual labels for a packed article.
4. Re-run the retrieval eval against a LangSmith experiment only if you want the
   score delta; not required to validate these fixes.
