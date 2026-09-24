"""Query side of the pipeline: run repeatedly (unlike ingest.py, which runs
once). Exposes answer_question() as a plain function so it can be imported
directly by an eval harness, plus a terminal REPL when run as __main__.
"""
import threading

import chromadb
from langsmith import traceable

import config
import provision_refs
from embeddings import embed_query
from llm import generate_answer, is_refusal


_collection = None
_corpus_index: tuple[list[str], list[dict]] | None = None  # (ids, metadatas)
_collection_lock = threading.Lock()


def _get_collection():
    """Opened once per process and reused. Building a fresh PersistentClient
    per question races under `run_langsmith_eval.py --concurrency N`: threads
    initialising Chroma's shared system at the same moment fail with
    "Could not connect to tenant default_tenant" / KeyError, and those rows
    silently score as empty retrievals."""
    global _collection, _corpus_index
    with _collection_lock:
        if _collection is None:
            client = chromadb.PersistentClient(path=config.CHROMA_DIR)
            try:
                _collection = client.get_collection(config.COLLECTION_NAME)
            except Exception as exc:
                raise RuntimeError(
                    f"Collection '{config.COLLECTION_NAME}' not found at "
                    f"{config.CHROMA_DIR}. Run `python ingest.py` first."
                ) from exc
            # Every chunk's metadata, held in memory for metadata-first
            # matching (a few hundred small dicts). Loaded with the collection
            # so both come from the same snapshot of the index.
            everything = _collection.get(include=["metadatas"])
            _corpus_index = (everything["ids"], everything["metadatas"])
        return _collection


def _as_documents(outputs) -> dict:
    """Reshape _retrieve's internal dicts into LangSmith's document format so the
    retriever run renders each chunk as a document (with its similarity score) in
    the UI. Only affects what is logged -- _retrieve's real return value, which
    answer_question/llm.py depend on, is untouched.

    LangSmith hands process_outputs the raw return value: a list here, though some
    versions wrap a non-dict return as {"output": [...]}. Accept both shapes."""
    chunks = outputs.get("output", []) if isinstance(outputs, dict) else outputs
    return {
        "documents": [
            {
                "page_content": c["text"],
                "type": "Document",
                "metadata": {**c["metadata"], "similarity": c["similarity"],
                             "retrieved_by": c["retrieved_by"]},
            }
            for c in chunks
        ]
    }


def _query(collection, query_embedding, n: int, retrieved_by: str,
           ids: list[str] | None = None) -> list[dict]:
    """One similarity query, optionally restricted to `ids`, ranked by
    similarity."""
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n,
        ids=ids,
        include=["documents", "metadatas", "distances"],
    )
    out = []
    for chunk_id, text, metadata, distance in zip(
        results["ids"][0], results["documents"][0],
        results["metadatas"][0], results["distances"][0],
    ):
        out.append({
            "id": chunk_id,
            "text": text,
            "citation": metadata["citation"],
            # Collection is configured with hnsw:space "cosine", where Chroma
            # reports distance = 1 - cosine_similarity.
            "similarity": 1 - distance,
            "metadata": metadata,
            "retrieved_by": retrieved_by,
        })
    return out


@traceable(run_type="retriever", name="retrieve", process_outputs=_as_documents)
def _retrieve(question: str, k: int, pin: bool = True) -> list[dict]:
    """Metadata first, then similarity.

    pin=False skips the metadata step (pure similarity). agent.py uses it for
    LLM-rewritten queries, so a provision the model names from its own memory
    of the Act is never pinned in -- only provisions the USER named are.

    If the question names provisions outright ("article 97", "Annex XI",
    "Chapter III Section 4", "the preamble"), the chunks whose METADATA match
    are pinned to the top -- up to k - MIN_SEMANTIC_SLOTS of them, ranked by
    similarity among themselves. bge-small is weak at matching bare numbers,
    so pure vector search can miss the very provision the user named. The
    remaining slots are filled by ordinary similarity search, de-duplicated."""
    collection = _get_collection()
    query_embedding = embed_query(question)

    ids, metadatas = _corpus_index
    named_ids = provision_refs.match_chunk_ids(
        provision_refs.parse_question(question), ids, metadatas
    ) if pin else []
    pinned = []
    if named_ids:
        cap = max(k - config.MIN_SEMANTIC_SLOTS, 1)
        pinned = _query(collection, query_embedding, min(len(named_ids), cap),
                        "metadata", ids=named_ids)

    pinned_ids = {c["id"] for c in pinned}
    # Over-fetch by len(pinned) so de-duplication can still fill every slot.
    semantic = _query(collection, query_embedding, k + len(pinned), "similarity")
    fill = [c for c in semantic if c["id"] not in pinned_ids][: k - len(pinned)]
    return pinned + fill


@traceable(run_type="chain", name="answer_question")
def answer_question(question: str, k: int = config.TOP_K) -> dict:
    """Runs retrieval + (maybe) generation for one question.

    Returns:
        {
            "answer": str,
            "refused": bool,
            "citations": list[str],  # always populated from retrieved_chunks,
                                      # independent of refused/error
            "retrieved_chunks": list[dict],  # always populated, even on refusal
            "error": str | None,
        }
    """
    retrieved_chunks = _retrieve(question, k)
    citations = [c["citation"] for c in retrieved_chunks]

    named_provision = any(c["retrieved_by"] == "metadata" for c in retrieved_chunks)
    top_similarity = max((c["similarity"] for c in retrieved_chunks), default=0.0)
    if not retrieved_chunks or (
        not named_provision and top_similarity < config.SIMILARITY_THRESHOLD
    ):
        # Refusal gate 1: nothing in the corpus is even topically close. Skipped
        # when the question named a provision that exists -- that chunk is
        # relevant by definition, whatever its embedding similarity.
        return {
            "answer": config.REFUSAL_MESSAGE,
            "refused": True,
            "citations": citations,
            "retrieved_chunks": retrieved_chunks,
            "error": None,
        }

    result = generate_answer(question, retrieved_chunks)

    if result["error"]:
        return {
            "answer": None,
            "refused": False,
            "citations": citations,
            "retrieved_chunks": retrieved_chunks,
            "error": result["error"],
        }

    answer = result["answer"]
    # Refusal gate 2: model decided the retrieved chunks don't actually answer it.
    # See llm.is_refusal: a full refusal, not a partial answer that mentions
    # one. Citations still reflect what was retrieved -- the chunks were
    # consulted even when the model declines to give a substantive answer.
    refused = is_refusal(answer)

    return {
        "answer": answer,
        "refused": refused,
        "citations": citations,
        "retrieved_chunks": retrieved_chunks,
        "error": None,
    }


def _repl(baseline: bool = False):
    """Answers with the agentic pipeline (agent.py) by default -- it beat this
    module's single-search pipeline on the golden set (see
    plans/agentic-rag-plan.md). --baseline answers with answer_question."""
    if baseline:
        answer_fn, label = answer_question, "baseline RAG"
    else:
        from agent import answer_question_agentic  # agent imports this module
        answer_fn, label = answer_question_agentic, "agentic RAG"
    print(f"EU AI Act {label} -- ask a question (type 'exit' or 'quit' to stop).\n")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        result = answer_fn(question)

        if result["error"]:
            print(f"\n[error] {result['error']}\n")
            continue

        print(f"\n{result['answer']}")
        if result["citations"]:
            print(f"\nCited: {', '.join(result['citations'])}")
        print()


if __name__ == "__main__":
    import sys

    _repl(baseline="--baseline" in sys.argv[1:])
