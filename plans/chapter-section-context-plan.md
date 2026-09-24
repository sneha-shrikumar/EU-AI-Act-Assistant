# Plan: Preserve Chapter/Section hierarchy in Article chunks

> Status: **Done 2026-07-19.**

## Problem
The EU AI Act is hierarchical: **CHAPTER → SECTION → Article**. The current
`chunking.py` discards the chapter/section context: `_strip_chapter_section_noise`
(lines 77–84) truncates article blocks at the first CHAPTER/SECTION marker and
throws it away. As a result an Article chunk (e.g. Article 16) never records that
it belongs to *Chapter III — HIGH-RISK AI SYSTEMS → Section 3 — Obligations of
providers and deployers…*. This context is lost for both:
- **embedding/retrieval** — section titles are strong thematic signals; and
- **metadata/citation/display**.

Confirmed raw format (via probe): a CHAPTER/SECTION header line is followed by its
title on the next non-empty line. Chapters may contain sections, or articles
directly. Chapter titles are UPPERCASE; section titles are sentence-case.

## Decision (approved)
Inject chapter/section **titles into the embedded chunk text AND store them as
metadata** (option: "Titles in text + metadata"). Preserve source casing for the
titles (avoid title-casing, which would mangle "AI" → "Ai").

## Changes

### 1. `chunking.py`
- Add `CHAPTER_HEADER_RE` and `SECTION_HEADER_RE` (number-only header lines).
- Rewrite `parse_articles` as a single ordered pass over CHAPTER / SECTION /
  ARTICLE markers (sorted by position), maintaining `current_chapter` and
  `current_section` state. A new chapter resets the section. Each header's title
  is the next non-empty line after the header.
- Because article blocks now end at the next structural marker, trailing
  chapter/section noise no longer leaks in — `_strip_chapter_section_noise`
  becomes unnecessary for articles (kept as a defensive no-op or removed).
- After `_split_provision` builds each article chunk, prepend a breadcrumb header
  to `chunk["text"]`:
  ```
  Chapter III: HIGH-RISK AI SYSTEMS
  Section 3: Obligations of providers and deployers of high-risk AI systems and other parties
  Article 16
  <existing title + body>
  ```
  (The article number was previously absent from the text entirely — this also
  fixes that.)
- Add `chapter` and `section` keys to each article chunk dict
  (e.g. `"III — HIGH-RISK AI SYSTEMS"`, `"3 — Obligations of providers…"`, or
  `""` when absent). Recitals/Annexes get `""`.

### 2. `ingest.py`
- Add `chapter` and `section` to the metadata dict written to Chroma.

### 3. Re-ingest
- `python ingest.py --reset` — required; existing chunks are already embedded
  without the new context.

## Not changed
- Recitals and Annexes (they don't sit under the article chapter/section tree).
- Citation strings stay `Article N` / `Article N(p)` (legal citations are by
  article); chapter/section live in metadata + text breadcrumb.
- `query.py` / `llm.py` need no changes (metadata is passed through generically).

## Verification
- Unit-test `chunk_document` in isolation (no embeddings) to confirm every article
  carries the correct chapter/section and the breadcrumb is prepended.
- Spot-check boundary cases: Chapter IV (article directly under chapter, no
  section), Chapter V Section 1 (section resets on new chapter), first article of
  Chapter I.
