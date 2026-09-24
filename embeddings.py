"""Local sentence-transformers embeddings, shared by ingest.py and query.py.

Kept in one place so both sides of the pipeline always use the exact same
model and pre/post-processing -- a mismatch here silently wrecks retrieval
quality.
"""
from functools import lru_cache

from sentence_transformers import SentenceTransformer

import config


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    # Cached so repeated calls (e.g. many queries in one CLI session, or an
    # eval harness importing this module) only load the model once.
    return SentenceTransformer(config.EMBEDDING_MODEL_NAME)


def embed_passages(texts: list[str]) -> list[list[float]]:
    """Embed document chunks for indexing. No instruction prefix -- bge's
    convention is that only queries get the retrieval instruction."""
    embeddings = _model().encode(
        texts, normalize_embeddings=True, show_progress_bar=len(texts) > 50
    )
    return embeddings.tolist()


def embed_query(text: str) -> list[float]:
    """Embed a user question for retrieval. Adds the bge query instruction
    prefix, which measurably improves retrieval quality for this model
    family."""
    prefixed = f"{config.QUERY_INSTRUCTION_PREFIX}{text}"
    embedding = _model().encode([prefixed], normalize_embeddings=True)
    return embedding[0].tolist()
