"""Full-document baseline: the "vanilla LLM" arm the RAG pipeline is compared
against. No retrieval at all -- every question is sent to BASELINE_MODEL with
the ENTIRE extracted Act in context.

Deliberately mirrors query.answer_question so the eval harness and judges can
treat both systems identically:
  * same SYSTEM_PROMPT (cite-or-refuse) as llm.py, so the only differences
    between arms are retrieval-vs-full-document and the generator model;
  * same text the RAG pipeline indexes (ingest.extract_pdf_text), so neither
    arm sees content the other cannot;
  * same return shape, with retrieved_chunks = [the whole document] so the
    groundedness judge grades against everything the model was shown.

Cost: the Act is ~190k tokens, so the document block carries cache_control
and sits BEFORE the question -- every call after the first reads the cached
prefix at ~5% of the input price instead of paying for it again.
"""
import re
from functools import lru_cache

from langsmith import traceable

import config
from ingest import extract_pdf_text
from llm import SYSTEM_PROMPT, _client, _set_llm_usage, is_refusal

FULL_DOC_CITATION = "EU AI Act (full text)"

# Inline citations the model writes, per SYSTEM_PROMPT rule 2: "(Article 6(2))",
# "(Recital 26)", "(Annex III)". Captured so the baseline's `citations` field
# lists the provisions it actually relied on -- the RAG arm lists what it
# retrieved, which is the closest analogue a no-retrieval system has.
_CITATION_RE = re.compile(
    # No trailing \b: after "Article 3(1)" the next char is ")" -- a \b there
    # fails and backtracking silently truncates the match to "Article 3".
    r"\b(Article\s+\d+(?:\(\d+\))*(?:\([a-z]\))?|Recital\s+\d+\b|Annex\s+[IVXL]+\b)",
    re.IGNORECASE,
)


@lru_cache(maxsize=1)
def load_full_text() -> str:
    return extract_pdf_text(config.PDF_PATH)


def _extract_citations(answer: str) -> list[str]:
    seen, out = set(), []
    for match in _CITATION_RE.findall(answer or ""):
        norm = re.sub(r"\s+", " ", match).strip()
        norm = norm[0].upper() + norm[1:]
        if norm.lower() not in seen:
            seen.add(norm.lower())
            out.append(norm)
    return out


@traceable(
    run_type="llm",
    name="ChatOpenAI",
    metadata={"ls_provider": "openrouter", "ls_model_name": config.BASELINE_MODEL},
)
def _full_doc_completion(messages: list[dict]):
    """Mirrors llm._chat_completion, minus `temperature`: Opus 5.5 rejects
    sampling params, and always thinks -- effort is the only control."""
    response = _client().chat.completions.create(
        model=config.BASELINE_MODEL,
        max_tokens=config.BASELINE_MAX_TOKENS,
        messages=messages,
        extra_body={
            "usage": {"include": True},
            "reasoning": {"effort": config.BASELINE_EFFORT},
        },
    )
    _set_llm_usage(response)
    return response


def cache_stats(response) -> dict:
    """OpenRouter's cache counters, for verifying the document prefix is
    actually being reused (0 cached tokens after call 1 = paying full price)."""
    usage = getattr(response, "usage", None)
    details = getattr(usage, "prompt_tokens_details", None)
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "cached_tokens": getattr(details, "cached_tokens", None),
        "cache_write_tokens": getattr(details, "cache_write_tokens", None),
        "cost": getattr(usage, "cost", None),
    }


@traceable(run_type="chain", name="answer_question_full_doc")
def answer_question_full_doc(question: str) -> dict:
    """Same contract as query.answer_question. Errors are returned, not raised."""
    full_text = load_full_text()
    retrieved_chunks = [{"citation": FULL_DOC_CITATION, "text": full_text}]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {
                "type": "text",
                "text": f"Excerpts from the EU AI Act:\n\n[{FULL_DOC_CITATION}]\n{full_text}",
                # Breakpoint AFTER the document, BEFORE the question: all
                # questions share this prefix byte-for-byte.
                "cache_control": {"type": "ephemeral"},
            },
            {"type": "text", "text": f"\n\nQuestion: {question}"},
        ]},
    ]

    try:
        response = _full_doc_completion(messages)
        answer = (response.choices[0].message.content or "").strip()
    except Exception as exc:  # noqa: BLE001 -- same contract as llm.generate_answer
        return {"answer": None, "refused": False, "citations": [],
                "retrieved_chunks": retrieved_chunks, "error": str(exc),
                "usage": {}}

    refused = is_refusal(answer)

    return {
        "answer": answer,
        "refused": refused,
        "citations": _extract_citations(answer),
        "retrieved_chunks": retrieved_chunks,
        "error": None,
        "usage": cache_stats(response),
    }


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "what is an AI system"
    r = answer_question_full_doc(q)
    print(r["error"] or r["answer"])
    print("\nCited:", ", ".join(r["citations"]))
    print("Usage:", r["usage"])
