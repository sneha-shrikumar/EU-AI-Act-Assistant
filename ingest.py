"""Run this ONCE (and again after --reset if the source PDF changes) to
build the persisted Chroma vector store. Separate from query.py on purpose:
ingestion is slow (PDF parsing + embedding) and should not run on every
question.

Usage:
    python ingest.py            # incremental upsert
    python ingest.py --reset    # drop and rebuild the collection from scratch
"""
import sys
from collections import Counter

import chromadb
import fitz  # PyMuPDF

import config
from chunking import chunk_document
from embeddings import embed_passages


def _body_clip(page):
    """Return the y-band (header_cut, footnote_cut) of a page's body, so we can
    extract text WITHOUT the running header, the page footer, or the page-bottom
    footnotes. These all interrupt provisions that span a page break -- a recital
    continuing onto the next page gets the header/footnotes injected mid-text,
    truncating it -- so they must be removed before chunking.

    We locate them from the OJ page furniture's horizontal rules:
      * the header rule sits at the very top (y < 80);
      * the footnote separator is a short (~41-57pt) rule indented at x~=42-67
        in the lower half of the page. Both the width and the indent vary
        between pages, so the tolerances here are deliberately loose -- too
        tight and the rule goes undetected, the fallback margin applies, and
        real body text is clipped.
    A page with several such short rules is an amendment page with boxed tables
    (Articles 103-110), where those rules are NOT footnote separators -- there we
    fall back to a fixed bottom margin so we never clip real article text.
    """
    H, W = page.rect.height, page.rect.width
    header_ys, sep_ys = [], []
    for drawing in page.get_drawings():
        for item in drawing["items"]:
            if item[0] == "l" and abs(item[1].y - item[2].y) < 1:
                y, x0, w = item[1].y, min(item[1].x, item[2].x), abs(item[2].x - item[1].x)
            elif item[0] == "re" and item[1].height < 3:
                y, x0, w = item[1].y0, item[1].x0, item[1].width
            else:
                continue
            if w < 20:
                continue
            if y < 80:
                header_ys.append(y)
            elif y > 0.55 * H and 40 <= w <= 60 and 40 <= x0 <= 80:
                sep_ys.append(round(y))

    header_cut = max(header_ys) if header_ys else 0.065 * H

    clusters = []
    for y in sorted(set(sep_ys)):
        if not clusters or y - clusters[-1][-1] > 10:
            clusters.append([y])
        else:
            clusters[-1].append(y)
    # 0.955 (not 0.93) as the fallback: on pages where the separator isn't
    # detected, 0.93*H lands ABOVE the last body line (e.g. page 99 puts the
    # closing line of Article 71(5) at y=789 with H=842), silently truncating a
    # provision mid-sentence. The real footer furniture starts around 0.96*H.
    footnote_cut = min(clusters[0]) if len(clusters) == 1 else 0.955 * H

    return header_cut, footnote_cut, W


def extract_pdf_text(pdf_path) -> str:
    # PyMuPDF, not pypdf: the official EUR-Lex OJ PDF extracts cleanly under
    # PyMuPDF, whereas pypdf injects spurious spaces inside words ("Ar ticle",
    # "REGUL A TION"), which silently breaks every structural regex.
    #
    # We clip each page to its body rectangle (see _body_clip) so headers,
    # footers, and footnotes are dropped while PyMuPDF's own reading order --
    # which correctly places each "(N)" recital number BEFORE its paragraph --
    # is preserved. Rebuilding the text ourselves from spans re-orders the
    # hanging numbers and misaligns every recital, so we must let get_text do it.
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"No PDF found at {pdf_path}. Place the EU AI Act PDF there first."
        )
    doc = fitz.open(str(pdf_path))
    pages = []
    for page in doc:
        header_cut, footnote_cut, width = _body_clip(page)
        clip = fitz.Rect(0, header_cut, width, footnote_cut)
        pages.append(page.get_text("text", clip=clip))
    return "\n".join(pages)


def build_collection(reset: bool = False):
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)

    if reset:
        try:
            client.delete_collection(config.COLLECTION_NAME)
        except Exception:
            pass  # collection didn't exist yet -- fine

    collection = client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def main():
    reset = "--reset" in sys.argv

    print(f"Reading {config.PDF_PATH} ...")
    raw_text = extract_pdf_text(config.PDF_PATH)

    print(f"Chunking into {config.CHUNK_TOKENS}-token windows "
          f"({config.CHUNK_OVERLAP_TOKENS}-token overlap) ...")
    chunks = chunk_document(raw_text)
    if not chunks:
        print("No chunks were produced -- check that the PDF text extracted "
              "correctly and that chunking.py's regexes match this document's "
              "formatting.")
        sys.exit(1)

    counts = Counter(c["type"] for c in chunks)
    print("Chunk counts by type:")
    for provision_type, count in counts.items():
        print(f"  {provision_type}: {count}")
    print(f"  Total: {len(chunks)}")
    print("(Sanity check -- these are CHUNK counts, not provision counts:"
          " short consecutive provisions are packed into one chunk and long"
          " ones are split across several, so neither matches the Act's ~180"
          " recitals / 113 articles / 13 annexes directly. Use"
          " inspect_chunks.py to check coverage; a count that collapses"
          " toward zero means the regexes in chunking.py need adjusting for"
          " this PDF's formatting.)")

    print(f"Embedding {len(chunks)} chunks with {config.EMBEDDING_MODEL_NAME} ...")
    texts = [c["text"] for c in chunks]
    embeddings = embed_passages(texts)

    print(f"Writing to persisted Chroma store at {config.CHROMA_DIR} ...")
    collection = build_collection(reset=reset)
    collection.upsert(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[
            {
                "type": c["type"],
                "number": c["number"],
                "paragraph": c["paragraph"] or "",
                "title": c["title"] or "",
                "chapter": c["chapter"],
                "section": c["section"],
                "citation": c["citation"],
                "source": c["source"],
            }
            for c in chunks
        ],
    )

    print(f"Done. Collection '{config.COLLECTION_NAME}' now has "
          f"{collection.count()} chunks.")


if __name__ == "__main__":
    main()
