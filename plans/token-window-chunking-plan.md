# Plan: Rewrite chunking.py as 500-token windows with 50-token overlap

> Status: **Done 2026-09-20.**
**Scope:** `chunking.py` (full rewrite), `config.py` (new knobs), `requirements.txt` (explicit `transformers`)

## Goal

Replace the current word-count-threshold chunking strategy (`config.MAX_CHUNK_WORDS = 220`,
keep-whole-or-split-by-paragraph) with fixed **500-token chunks and 50-token overlap**,
while keeping every existing metadata field intact.

## Approved decisions

| Decision | Choice |
|---|---|
| Window scope | **Window within provisions, pack short ones.** Parse the legal structure first, window inside any unit over budget, and merge consecutive short sibling units up to the budget. |
| Tokenizer | **The embedding model's own tokenizer** (`BAAI/bge-small-en-v1.5` via `transformers.AutoTokenizer`). No drift between what we measure and what the embedder sees. No new dependency (`transformers` already ships with `sentence-transformers`). |

## Constraint that shapes the design

`bge-small-en-v1.5` has a **512-token max sequence length**. A 500-token chunk plus the
Chapter/Section/Article breadcrumb that `_apply_context` prepends would exceed 512 and be
**silently truncated at embed time**, throwing away the tail of the chunk.

Therefore the 500-token budget is a budget for the **finished chunk text, breadcrumb
included**. The breadcrumb is tokenized per-provision and its cost is subtracted before
windowing the body. Nothing the chunker emits can be truncated by the embedder.

The tokenizer is also **uncased**, so token ids must never be decoded back to text
(that would lowercase the whole corpus and mangle `AI`, `CE`, `EU`). We use the fast
tokenizer's `return_offsets_mapping=True` and slice the **original string** by character
offsets instead.

## Unit model

1. **Parse** (unchanged in intent, this is the metadata source, not a strategy choice):
   zones → Preamble / Recitals / Articles (with Chapter+Section state) / Annexes,
   via the existing header regexes.
2. **Base unit** = the finest structural level available: a numbered paragraph/point of an
   Article or Annex where one exists, otherwise the whole provision. Keeping paragraphs as
   the base unit preserves the fine-grained `paragraph` field and `Article 6(2)`-style
   citations that the eval harness ranks on. Size no longer decides whether we split —
   the parser always produces paragraph units, and the packer merges them back up.
3. **Pack**: walk the ordered unit list, merging consecutive units while the combined token
   count stays within budget **and** they share a *citation root*:
   - Recitals: any run of consecutive recitals → `Recitals 12–15`, `number = "12-15"`
   - Article paragraphs: same article only → `Article 6(1)–(3)`
   - Annex points: same annex only → `Annex III(1)–(4)`
   - Preamble: never packed
   Packing does **not** cross article boundaries, because `number`, `title` and `citation`
   would then have no single correct value. This trades some size uniformity for metadata
   that stays exact — the explicit priority of this rewrite.
4. **Window**: any unit still over budget is sliced into 500-token windows stepping by
   `500 - 50 = 450` tokens. Each window keeps the parent's full metadata and gains a
   `part` index; the trailing window is merged backwards if it would be a sliver
   (< overlap size), so we never emit a 12-token orphan.

## Metadata contract (unchanged — `ingest.py` writes these verbatim)

`text`, `type`, `number`, `paragraph`, `title`, `chapter`, `section`, `citation`,
`source`, `chunk_id`.

Two fields gain range forms, which are still strings and still parse the same way:
- `number`: `"12-15"` for a packed recital run
- `paragraph`: `"1-3"` for packed paragraphs

New internal-only field `part` (int, 1-based) distinguishing windows of one oversized unit;
it is folded into `chunk_id` and not written to Chroma, so `ingest.py` needs no change.

`chunk_id` stays unique via the existing `seen_ids` collision counter, with the part index
appended as `#2` before the `__N` collision suffix.

## config.py changes

```python
CHUNK_TOKENS = 500          # replaces MAX_CHUNK_WORDS
CHUNK_OVERLAP_TOKENS = 50
EMBEDDING_MAX_TOKENS = 512  # bge-small hard limit; CHUNK_TOKENS must stay <= this
```

`MAX_CHUNK_WORDS` is removed — nothing outside `chunking.py` reads it (verified by grep).

## Verification

1. `python -c "from chunking import chunk_document"` + run over the real PDF text.
2. Assert every emitted chunk tokenizes to `<= 500` tokens under the bge tokenizer.
3. Print the token-length distribution (min / median / max / count under 100) to confirm
   packing actually filled the windows.
4. Confirm chunk counts still look sane against the known shape of the Act
   (~180 recitals, ~113 articles, ~13 annexes) and that all `chunk_id`s are unique.
5. Re-run `python ingest.py --reset` is the user's call, not part of this change.

---

## Revision 2 — 2026-09-20 (post-review)

User review of the emitted chunks found three defects. All fixed.

### 1. Recital numbers absent from chunk text (chunking bug)
`_split_by_header` returns the text *after* the `(N)` marker, so the number
survived only in metadata. Packed runs had no visible boundary between one
recital and the next. **Fix:** `parse_recitals` re-attaches `"(N) "` to each
recital body, mirroring the `"N. "` prefix articles already got.

### 2. Article 71(5) truncated mid-sentence (PDF extraction bug, pre-existing)
Not a chunking fault — `ingest.py::_body_clip` lost the line before chunking ran.
Page 99's footnote separator sits at `x0=41.8`, outside the `55 <= x0 <= 80`
detection window, so no separator was found and the `0.93 * H = 783.0` fallback
clipped the body line at `y=789.4`.
**Fix:** widen the indent tolerance to `40 <= x0 <= 80` and raise the fallback
margin to `0.955 * H`.

### 3. No overlap across provision boundaries (design change)
Revision 1 scoped overlap to *within* a provision, so only 46 of 335 chunks had
any and Article 71 → 72 had none. The requirement is 50-token overlap
throughout.
**Fix:** overlap is now a single global post-pass, `_add_overlap`, applied in
document order across the whole corpus. `_windows` slices non-overlapping, and
every chunk is prefixed with `"… "` plus the previous chunk's last 50 tokens,
taken from that chunk's own body so duplication never compounds.

Chunk layout is now: `breadcrumb / "… " + carry-over / body`, and the body
budget reserves room for all three (`500 − breadcrumb − 50 − 4`).

### Revision 2 verification
365 chunks; max 497 tokens (0 over 500); 364/364 overlaps verified against the
preceding chunk; 180/180 recitals, 113/113 articles, 13/13 annexes, no gaps;
0 duplicate ids; every recital number present in chunk text; Article 71(5)
restored in full.
