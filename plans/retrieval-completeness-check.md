# Whole-Provision Completeness Check for eval_retrieval.py

> Status: **Done 2026-08-13. Implemented in `eval_retrieval.py`.**

## Context

Screenshot review (`eu-ai-act-rag/issue with retrieval 1.png`) showed two rows
scored 1 by `eval_retrieval.py` that the user considers failures:
`"summarize article 112"` (expected_source `"article 112"`) and `"what does
article 97 talk about?"` (expected_source `"article 97"`). Both currently pass
because the existing rule only checks that *some* chunk of the expected
article was cited (paragraph ignored, per the module's article-level design).
But the retrieved citations only covered a few of each article's paragraphs
(Article 112: 3 of 12 chunks; Article 97: 1 of 5 chunks), so the generated
answer was necessarily incomplete.

Investigation surfaced a real pipeline constraint: `config.TOP_K = 5` — the
pipeline always retrieves exactly 5 chunks, pure vector similarity, no
article-aware logic (`query.py::_retrieve`). So full coverage is structurally
impossible for any article/annex with more than 5 paragraph-chunks. The user
confirmed (via AskUserQuestion) that the eval should flag this as a failure
anyway — strict "all chunks must be present" — since it's a real gap worth
surfacing, not eval noise to suppress.

Checked the golden CSV: `expected_source` is almost always a bare
article/annex number (no paragraph). Exactly one row narrows to specific
paragraphs (`"article 10(2), 10(5)"`), and several rows list multiple
*different* articles (multi-topic questions, e.g.
`"article 23, article24, article 26, article 50"`). So the new check must
apply **only** to rows whose expected_source is a single bare article/annex
reference — not paragraph-specific rows, not multi-reference rows.

**Correction after first pass:** a single bare article/annex reference is
*not* sufficient on its own — many rows resolve to one (large) article
without the question being "about the whole article." E.g. `"what is an AI
system"` → `article 3` (a huge, many-paragraph definitions article); the
question only needs one definition, not full-article coverage. The
discriminator the user confirmed: the completeness check should only fire
when the **question text itself names that article/annex number**
("summarize article 112", "what does article 97 talk about") — reusing
`parse_refs` on the question is enough to detect this (`parse_refs(question)
⊇ expected_refs`). Checked against every row in the golden CSV: with this
gate added, only the two flagged rows (Article 112, Article 97) qualify among
should_answer=yes rows, plus `"what does Annex 11 talk about?"` which fits
the identical pattern — everything else is correctly excluded.

**User-confirmed design (via AskUserQuestion):**
- Ground truth for "how many paragraph-chunks does Article X have": query the
  **live Chroma DB** at eval time (same `chroma_db/` collection `query.py`
  reads), not the static `chunks_export.txt` dump.
- Threshold: **strict** — score 0 unless every paragraph-chunk of the article
  is represented among the citations.

## Design

All changes in `eu-ai-act-rag/eval_retrieval.py` (no new file):

### 1. `build_paragraph_index()` — live corpus ground truth

```python
def build_paragraph_index() -> dict[tuple[str, int], set[str]]:
    """{(kind, number): {paragraph_label, ...}} for every Article/Annex chunk
    in the live corpus (queried from Chroma, same store query.py reads).
    Whole-provision entries (paragraph == "") are omitted -- single chunk,
    nothing to check completeness against."""
    import chromadb
    import config

    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    collection = client.get_collection(config.COLLECTION_NAME)
    metadatas = collection.get(include=["metadatas"])["metadatas"]

    index: dict[tuple[str, int], set[str]] = {}
    for meta in metadatas:
        kind = (meta.get("type") or "").lower()
        paragraph = meta.get("paragraph")
        if kind not in ("article", "annex") or not paragraph:
            continue
        number_raw = meta.get("number") or ""
        if kind == "annex" and ROMAN_RE.match(number_raw):
            number = _roman_to_int(number_raw)
        else:
            try:
                number = int(number_raw)
            except ValueError:
                continue  # e.g. "47a" -- not handled anywhere else either
        index.setdefault((kind, number), set()).add(paragraph)
    return index
```

Called once per `evaluate()` run (not per row) and threaded through to
`score_row`. Reuses `ROMAN_RE`/`_roman_to_int` already defined for annex
numerals.

### 2. `_citation_paragraph(citation)` — extract the paragraph label a strict
chunk citation carries (citations are always `"Article N(P)"` / `"Article N"`
/ `"Annex III(P)"`, per `chunking.py::_split_provision`)

```python
CITATION_RE = re.compile(r"^(Article|Annex|Recital)\s+([A-Za-z0-9]+)(?:\(([^)]+)\))?$")

def _citation_detail(citation: str):
    """('article'|'annex'|'recital', number:int, paragraph:str|None) or None
    if the citation string doesn't match the strict chunk-citation format."""
    m = CITATION_RE.match(citation.strip())
    if not m:
        return None
    kind_word, number_raw, paragraph = m.groups()
    kind = kind_word.lower()
    if kind == "annex" and ROMAN_RE.match(number_raw):
        number = _roman_to_int(number_raw)
    else:
        try:
            number = int(number_raw)
        except ValueError:
            return None
    return kind, number, paragraph
```

### 3. `score_row` — add the completeness branch

New signature: `score_row(expected_refs, citation_refs, citations, should_answer, refused, expected_source, question, paragraph_index)`.

After the existing "all expected refs subset of citation refs" check passes
(i.e. would currently score 1), add:

```python
if (len(expected_refs) == 1 and not PARA_SUFFIX_RE.search(expected_source)
        and expected_refs <= parse_refs(question)):
    kind, number = next(iter(expected_refs))
    corpus_paragraphs = paragraph_index.get((kind, number))
    if corpus_paragraphs and len(corpus_paragraphs) > 1:
        covered = {
            detail[2] for c in citations
            if (detail := _citation_detail(c)) and (detail[0], detail[1]) == (kind, number) and detail[2]
        }
        missing = corpus_paragraphs - covered
        if missing:
            missing_sorted = sorted(missing, key=lambda p: (0, int(p)) if p.isdigit() else (1, p))
            return 0, (f"incomplete retrieval: missing {kind} {number} paragraph(s) "
                       f"{', '.join(missing_sorted)} ({len(covered)}/{len(corpus_paragraphs)} retrieved)")
return 1, ""
```

- `len(expected_refs) == 1` — only single-reference rows (skips multi-article
  questions like the importers/distributors/deployers example).
- `not PARA_SUFFIX_RE.search(expected_source)` — skips the one row that
  already narrows to specific paragraphs (`"article 10(2), 10(5)"`); reuses
  the existing regex that strips `(N)`/`(a)` suffixes.
- `expected_refs <= parse_refs(question)` — the question text must itself
  name that article/annex number. This is what excludes rows like `"what is
  an AI system"` → `article 3` (no number in the question) while including
  `"summarize article 112"` / `"what does article 97 talk about"` (number
  named directly).
- `len(corpus_paragraphs) > 1` — provisions kept whole (short articles, no
  paragraph split) are skipped: the existing subset check already guarantees
  full coverage for a 1-chunk provision.
- Note format mirrors the existing "missing: ..." style added for the recall
  rule, listing exactly which paragraphs were absent and a coverage fraction.

### 4. `evaluate()` — wire it through

```python
paragraph_index = build_paragraph_index()
...
score, note = score_row(
    expected_refs, citation_refs, citations, should_answer, refused,
    expected_source, paragraph_index,
)
```

`build_paragraph_index()` requires the local `chroma_db/` to exist (same
prerequisite `query.py` already has) — if missing, surface the same
`RuntimeError`-style message `query.py::_get_collection` uses rather than a
raw traceback.

## Files touched

- **Edit:** `eu-ai-act-rag/eval_retrieval.py` (only file changed)
- **New (per CLAUDE.md):** this plan, saved to `plans/` before implementation.

## Verification

1. `python -m py_compile eval_retrieval.py`.
2. `python eval_retrieval.py --selftest` — confirm untouched (paragraph logic
   isn't part of `parse_refs`).
3. Inline check: call `build_paragraph_index()` against the real
   `chroma_db/` and confirm `("article", 112)` has 12 paragraph labels and
   `("article", 97)` has 5 — matches the manual `chunks_export.txt` grep done
   during investigation.
4. Run `python eval_retrieval.py` against the real experiment; confirm the
   two flagged rows ("summarize article 112", "what does article 97 talk
   about?") now score 0 with a note listing the missing paragraphs, while a
   short single-chunk article (e.g. "article 4") and multi-article rows are
   unaffected.
