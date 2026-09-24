# Fix: unsplit articles using parenthetical paragraph numbering

> Status: **Done 2026-08-13.**

## Context

While building the retrieval-completeness check for `eval_retrieval.py`, we
found Article 3 (definitions) has **zero** paragraph-level chunks in the
corpus — the whole article (2574 words, ~68 definitions) is stored as one
giant chunk. Root cause: `chunking.py::PARAGRAPH_MARKER_RE` only recognizes
the period style (`"1. text"`), but Article 3's definitions are numbered
`"(1) 'AI system' means..."` — a parenthetical style that never matches, so
`_split_provision` falls into its "no internal numbering to split on" branch
and keeps the whole 2574-word block as one chunk.

Checked every other article that exceeds `MAX_CHUNK_WORDS` (220) but wasn't
split (4 total): **Article 3** and **Article 108** (amendment list, `"(1) in
Article 17..."`) both use this same parenthetical style — genuine bug.
Articles 16 and 66 are correctly whole: they only contain lettered sub-points
(`(a)(b)(c)`, some with roman-numeral sub-sub-points), no top-level digit
numbering at all, so nothing should split there.

A single unsplit 2574-word Article 3 chunk is bad for retrieval: a question
about one specific definition (e.g. "what is an AI system") has to compete
via similarity against a chunk embedding that represents ~68 unrelated
definitions at once, diluting relevance.

## Fix

`chunking.py` — add a fallback marker regex, tried only when the primary
period-style pattern finds nothing in that article's block (so the common
case is untouched):

```python
PAREN_PARAGRAPH_MARKER_RE = re.compile(r"^[ \t]*\((\d{1,2})\)\s+", re.MULTILINE)
```

In `_split_provision`, after the existing
`first_para_match = PARAGRAPH_MARKER_RE.search(block)`:

```python
para_marker_re = PARAGRAPH_MARKER_RE
if not first_para_match:
    first_para_match = PAREN_PARAGRAPH_MARKER_RE.search(block)
    para_marker_re = PAREN_PARAGRAPH_MARKER_RE
```

...and use `para_marker_re` (not the hardcoded `PARAGRAPH_MARKER_RE`) in the
later `_split_by_header(body, para_marker_re)` call. Everything downstream
(citation format, chunk_id, paragraph field) is untouched — `para_num` just
comes from a different regex.

Safety check: the new pattern requires digits only inside the parens
(`\(\d{1,2}\)`), anchored at line start — won't match lettered points
`(a)`/`(i)`, footnote markers `(*)`, or inline parenthetical refs like
"Regulation (EU) 2018/1139" (not at line start). Verified against Articles
16/66's raw text that neither now spuriously splits.

## Rebuild

This only takes effect after re-chunking + re-embedding:
1. `python ingest.py --reset` — drops and rebuilds the `chroma_db/`
   collection (regenerable from the PDF + chunking logic, not user data).
2. `python inspect_chunks.py` (no args) — regenerates `chunks_export.txt`
   from the rebuilt collection, per its own docstring.

## Verification

1. Sanity-run `chunk_document()` against the extracted PDF text before
   touching the DB; confirm Article 3 and 108 now produce multiple chunks,
   and total chunk count elsewhere doesn't spike (would indicate a false
   match).
2. After `ingest.py --reset` + `inspect_chunks.py`, confirm via
   `eval_retrieval.py::build_paragraph_index()` that `("article", 3)` and
   `("article", 108)` now report multiple paragraph labels.
3. Spot check a definition question in the REPL (`python query.py`) — e.g.
   "what is an AI system" — and confirm the citation is a specific
   `Article 3(N)` rather than the old whole-article citation.
