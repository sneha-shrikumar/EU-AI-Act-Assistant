"""Central configuration for the EU AI Act RAG pipeline.

Every tunable knob lives here so ingest.py and query.py always agree, and so
the eval harness (coming next) has one place to sweep values from.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths -------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PDF_PATH = DATA_DIR / "eu_ai_act.pdf"
CHROMA_DIR = str(BASE_DIR / "chroma_db")
COLLECTION_NAME = "eu_ai_act"

# --- Embedding model -----------------------------------------------------
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
# bge models are trained to use this instruction prefix on QUERIES only,
# never on the indexed passages. See embeddings.py.
QUERY_INSTRUCTION_PREFIX = "Represent this sentence for searching relevant passages: "

# --- Chunking --------------------------------------------------------------
# Fixed-size token windows. "Token" means whatever EMBEDDING_MODEL_NAME's own
# tokenizer says it means -- chunking.py loads that exact tokenizer so the
# sizes we enforce are the sizes the embedder actually sees.
CHUNK_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50
# Hard max sequence length of bge-small-en-v1.5. Anything longer is silently
# TRUNCATED at embed time, with no error, so CHUNK_TOKENS must stay under it --
# and the Chapter/Section breadcrumb counts against CHUNK_TOKENS, not on top.
EMBEDDING_MAX_TOKENS = 512
assert CHUNK_TOKENS <= EMBEDDING_MAX_TOKENS
assert 0 <= CHUNK_OVERLAP_TOKENS < CHUNK_TOKENS
# How parsed units become chunks (see chunking._pack and
# plans/structural-chunking-plan.md):
#   "size"     -- merge same-provision neighbours up to CHUNK_TOKENS
#   "none"     -- one chunk per structural unit (paragraph/point/recital/clause)
#   "semantic" -- like "size", but only merge neighbours whose embeddings are at
#                 least as similar as the corpus's SEMANTIC_PACK_PERCENTILE-th
#                 adjacent same-provision pair
CHUNK_PACKING = "size"
SEMANTIC_PACK_PERCENTILE = 50
# Where the CHUNK_OVERLAP_TOKENS carry-over is applied (see _add_overlap):
#   "document" -- every chunk gets the previous chunk's tail, across provisions
#   "window"   -- only windows 2+ of a unit that _windows split mid-text
OVERLAP_SCOPE = "window"  # chosen by analysis/chunking_variants_diagnostic.py
assert CHUNK_PACKING in ("size", "none", "semantic")
assert OVERLAP_SCOPE in ("document", "window")

# --- Retrieval -------------------------------------------------------------
TOP_K = 8
# Slots always left for plain similarity search when a question names a
# provision and query.py pins its metadata matches first (see _retrieve), so
# related context still gets in alongside the named provision.
MIN_SEMANTIC_SLOTS = 2
# Cosine similarity (1 - cosine distance) below this => refuse without ever
# calling the LLM. First-pass guess; tune with the eval harness once it exists.
SIMILARITY_THRESHOLD = 0.45

# --- Agentic retrieval (agent.py) ---------------------------------------------
# A fixed sequence of LLM-driven stages; see plans/agentic-rag-plan.md.
# Cumulative: "analyse" = rewrite/decompose + fused retrieval, "reflect" adds the
# coverage check and follow-up searches, "select" adds LLM evidence selection.
AGENT_STAGES = "reflect"  # kept by analysis/agentic_retrieval_diagnostic.py (see plan)
AGENT_MAX_SUBQUESTIONS = 4
AGENT_QUERIES_PER_SUB = 2
AGENT_REFLECT_ROUNDS = 2
# Chunks handed to the generator (and reported as citations, in rank order).
AGENT_MAX_EVIDENCE = 10
# How many fused-pool chunks the reflect/select stages get to read.
AGENT_POOL_VIEW = 20
assert AGENT_STAGES in ("analyse", "reflect", "select")

# --- Generation --------------------------------------------------------------
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "anthropic/claude-haiku-4.5"

# --- Full-document baseline (baseline_full_doc.py) ---------------------------
# A "vanilla LLM" comparison arm: no retrieval, the whole Act in context.
# Opus 5.5 always thinks and rejects sampling params (no temperature); effort
# is its only knob, and its default is medium -- set explicitly so the run is
# reproducible if that default ever moves.
BASELINE_MODEL = "anthropic/claude-opus-5.5"
BASELINE_EFFORT = "medium"
BASELINE_MAX_TOKENS = 16000

# --- Judging ---------------------------------------------------------------
# The eval judges (evals/judges.py) deliberately use a DIFFERENT, stronger
# model than OPENROUTER_MODEL above: asking a model to grade its own output
# invites self-preference bias, where it rates its own phrasing as correct.
JUDGE_MODEL = "anthropic/claude-sonnet-5"
JUDGE_TEMPERATURE = 0

REFUSAL_MESSAGE = (
    "I cannot find an answer to this in the EU AI Act excerpts provided."
)

SOURCE_LABEL = "EU AI Act (Regulation (EU) 2024/1689)"
