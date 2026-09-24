"""Browse the chunks stored in the vector database.

Run from the eu-ai-act-rag folder in PowerShell:

    venv\\Scripts\\python.exe inspect_chunks.py
        -> prints a summary and writes ALL chunks to chunks_export.txt
           (open that in Notepad and use Ctrl+F to search)

    venv\\Scripts\\python.exe inspect_chunks.py Article 5
        -> prints every chunk whose citation contains "Article 5"

    venv\\Scripts\\python.exe inspect_chunks.py Recital 28
    venv\\Scripts\\python.exe inspect_chunks.py Annex III
"""
import sys
from collections import Counter

import chromadb

import config

ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
TYPE_ORDER = {"Recital": 0, "Article": 1, "Annex": 2}


def roman_to_int(s: str) -> int:
    vals = [ROMAN.get(c, 0) for c in s]
    total = 0
    for i, v in enumerate(vals):
        total += -v if i + 1 < len(vals) and v < vals[i + 1] else v
    return total


def lead_int(value, default: int = 0) -> int:
    """First number of a plain value or of a packed range.

    chunking.py packs short consecutive provisions into one chunk and records
    the span in the metadata ("1-2" for Recitals 1-2, "1-5" for Article 70's
    paragraphs 1 to 5). A bare int() raises on those, and the old except-branch
    fell back to 0 -- which sorted EVERY packed chunk to the top of the export
    and made the single-provision chunks in between look like they were
    missing. Sorting on the range's first number restores document order."""
    try:
        return int(str(value).split("-")[0])
    except (ValueError, TypeError, AttributeError):
        return default


def sort_key(meta: dict):
    if meta["type"] == "Annex":
        number = roman_to_int(meta["number"])
    else:
        number = lead_int(meta["number"])
    return (TYPE_ORDER.get(meta["type"], 9), number, lead_int(meta["paragraph"]))


def load_rows():
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    collection = client.get_collection(config.COLLECTION_NAME)
    data = collection.get(include=["documents", "metadatas"])
    rows = list(zip(data["ids"], data["documents"], data["metadatas"]))
    rows.sort(key=lambda r: sort_key(r[2]))
    return rows


def print_chunk(chunk_id: str, text: str, meta: dict, stream=sys.stdout):
    stream.write("=" * 70 + "\n")
    stream.write(f"[{meta['citation']}]   (id: {chunk_id})\n")
    stream.write(text.strip() + "\n\n")


def main():
    rows = load_rows()
    query = " ".join(sys.argv[1:]).strip().lower()

    if query:
        matches = [r for r in rows if query in r[2]["citation"].lower()]
        print(f"{len(matches)} chunk(s) whose citation contains '{query}':\n")
        for chunk_id, text, meta in matches:
            print_chunk(chunk_id, text, meta)
        if not matches:
            print("(no matches -- try e.g. 'Article 5', 'Recital 28', 'Annex III')")
        return

    counts = Counter(meta["type"] for _, _, meta in rows)
    print(f"Collection '{config.COLLECTION_NAME}' holds {len(rows)} chunks:")
    for provision_type, count in sorted(counts.items()):
        print(f"  {provision_type}: {count}")

    out_path = config.BASE_DIR / "chunks_export.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        for chunk_id, text, meta in rows:
            print_chunk(chunk_id, text, meta, stream=f)

    print(f"\nWrote all {len(rows)} chunks to:\n  {out_path}")
    print("Open it in Notepad and press Ctrl+F to search.")
    print("\nOr filter right here in the terminal, for example:")
    print("  venv\\Scripts\\python.exe inspect_chunks.py Article 5")


if __name__ == "__main__":
    main()
