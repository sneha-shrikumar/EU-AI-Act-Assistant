"""OpenRouter generation call with a "cite, answer what you can, refuse only
when nothing applies" system prompt.

This is the second refusal gate (the first is the retrieval-similarity cutoff
in query.py): the model refuses when the chunks contain nothing relevant.
It used to refuse whenever it was "in doubt", which turned partial answers
and scenario questions ("is X a prohibited practice?") into refusals even with
the governing article at rank 1 -- see plans/relax-refusal-rule-plan.md.
"""
from openai import OpenAI
from langsmith import get_current_run_tree, traceable

import config

SYSTEM_PROMPT = f"""You are a legal research assistant answering questions about \
the EU AI Act using ONLY the excerpts provided below. These excerpts are the \
complete evidence available to you -- you have no other knowledge of the Act.

Rules, no exceptions:
1. Answer using ONLY information contained in the excerpts below. Never use \
outside knowledge, even if you believe it is correct.
2. Every factual claim you make must be followed by an inline citation to the \
excerpt it came from, in the form the excerpt is labeled with (e.g. "(Article 6(2))" \
or "(Recital 26)").
3. Applying a provision to the user's situation is NOT outside knowledge -- it \
is the job. If an excerpt sets out a rule, prohibition or category that covers \
the situation described in the question, apply it: say which provision covers \
the situation and why, citing it. Do not refuse just because the excerpt does \
not mention the user's exact example word for word.
4. If the excerpts answer only part of the question, answer that part and then \
state plainly which part the excerpts do not cover (e.g. "The excerpts do not \
say ..."). Never fill that gap with plausible-sounding legal knowledge.
5. Refuse only when the excerpts contain nothing relevant to the core of the \
question, or when the question depends on something the excerpts do not \
contain (a provision, amendment, date or event that does not appear in them). \
A full refusal must be EXACTLY this sentence and nothing else: \
"{config.REFUSAL_MESSAGE}" \
Never use that sentence inside a partial answer -- use the wording from rule 4.
6. Some excerpts carry a "Location:" line giving the Chapter and Section they \
sit under. That line is part of the excerpt. When a question asks which \
chapter, section or article covers something, name the full location -- \
chapter, section AND article (e.g. "Chapter III, Section 4 (Notifying \
authorities and notified bodies), Article 30") -- citing the excerpt as usual.
"""


def is_refusal(answer: str | None) -> bool:
    """True when the answer OPENS with the refusal sentence.

    Tolerant of casing and a short lead-in ("Sorry -- I cannot find ...").
    Whatever follows is ignored: the model often explains a refusal ("the
    excerpts do not include Article 114 ..."), and that is still a refusal.
    The sentence appearing LATER in a long answer does not count -- that is a
    partial answer, which rule 4 of the prompt now allows (and tells the model
    to phrase differently). Shared by query.py, baseline_full_doc.py and
    evals/judges.py so the pipeline's `refused` flag and the judges agree."""
    if not answer:
        return False
    message = config.REFUSAL_MESSAGE.rstrip(".").lower()
    head = answer.strip().lower()[: len(message) + 40]
    return message in head


def _client() -> OpenAI:
    if not config.OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Add it to eu-ai-act-rag/.env "
            "(see .env.example)."
        )
    return OpenAI(base_url=config.OPENROUTER_BASE_URL, api_key=config.OPENROUTER_API_KEY)


def _location_line(metadata: dict) -> str:
    """"Location: Chapter III (HIGH-RISK AI SYSTEMS) > Section 4 (...)" from
    the chunk's chapter/section METADATA, or "" when it has neither (recitals,
    annexes, preamble).

    The excerpt label is otherwise just the citation, and rule 2 tells the
    model to cite in label form -- so without this line the hierarchy only
    appears as breadcrumb text inside the body, and "which section covers X?"
    gets answered with a bare article number. Metadata is stored as
    "<number> - <title>" (see chunking._apply_context)."""
    parts = []
    for label in ("Chapter", "Section"):
        value = (metadata.get(label.lower()) or "").strip()
        if not value:
            continue
        number, _, title = value.partition(" - ")
        parts.append(f"{label} {number} ({title})" if title else f"{label} {number}")
    return f"Location: {' > '.join(parts)}" if parts else ""


def build_context_block(chunks: list[dict]) -> str:
    parts = []
    for chunk in chunks:
        header = f"[{chunk['citation']}]"
        location = _location_line(chunk.get("metadata") or {})
        if location:
            header += f"\n{location}"
        parts.append(f"{header}\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


def _set_llm_usage(response) -> None:
    """Attach token counts AND OpenRouter's real per-call cost to the current
    LLM run's usage_metadata, so LangSmith's native token *and* cost dashboards
    populate (the model id isn't in LangSmith's price registry, so without this
    its computed cost is null).

    Deliberately defensive: get_current_run_tree() is None when tracing is off,
    and `usage.cost` is an OpenRouter-specific field -- nothing here may raise
    and turn a good LLM response into a fake error (see generate_answer's except).
    """
    rt = get_current_run_tree()
    if rt is None:
        return
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    um = {
        "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
    }
    cost = getattr(usage, "cost", None)
    if cost is not None:
        # `cost` is what OpenRouter actually charges for the call -- treat it as
        # authoritative for total_cost so the native Cost dashboard is exact.
        um["total_cost"] = float(cost)
    rt.set(usage_metadata=um)


@traceable(
    run_type="llm",
    name="ChatOpenAI",
    metadata={"ls_provider": "openrouter", "ls_model_name": config.OPENROUTER_MODEL},
)
def _chat_completion(messages: list[dict]):
    """The single LLM call, traced as an `llm` run so LangSmith renders the
    prompt (messages) and populates token/cost/latency dashboards. Kept separate
    from generate_answer so the usage_metadata (incl. real cost) lands on the
    llm run itself -- putting it on the parent would double-count tokens."""
    response = _client().chat.completions.create(
        model=config.OPENROUTER_MODEL,
        temperature=0,
        messages=messages,
        # Ask OpenRouter to return real usage + cost for this call.
        extra_body={"usage": {"include": True}},
    )
    _set_llm_usage(response)
    return response


# Appended to SYSTEM_PROMPT only when agent.py passes query-analysis notes.
NOTES_RULE = f"""7. After the excerpts you may get NOTES from an earlier analysis of the \
question: a reading of it, assumptions, and sub-questions. The notes are NOT \
evidence -- never cite them and never take a fact from them. Use them only to \
structure the answer: open with one sentence stating the assumptions \
("Assuming ..."), then answer each sub-question in turn from the excerpts. If \
you refuse under rule 5, output only "{config.REFUSAL_MESSAGE}", with no \
assumptions sentence before it.
"""


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return text.strip()


def complete_json(system: str, user: str) -> dict:
    """One LLM call whose reply must be a single JSON object. Raises
    ValueError on unparseable output so each agent stage can fall back to its
    own default rather than guessing."""
    import json

    response = _chat_completion(
        [{"role": "system", "content": system}, {"role": "user", "content": user}]
    )
    text = _strip_fences(response.choices[0].message.content or "")
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"no JSON object in reply: {text[:200]!r}")
    return json.loads(text[start:end + 1])


def _notes_block(notes: dict) -> str:
    lines = ["NOTES (not evidence -- never cite):"]
    if notes.get("reading"):
        lines.append(f"Reading of the question: {notes['reading']}")
    if notes.get("assumptions"):
        lines.append("Assumptions: " + "; ".join(notes["assumptions"]))
    if notes.get("sub_questions"):
        lines.append("Sub-questions:")
        lines += [f"  {i}. {q}" for i, q in enumerate(notes["sub_questions"], 1)]
    return "\n".join(lines)


@traceable(run_type="chain", name="generate_answer")
def generate_answer(question: str, chunks: list[dict], notes: dict | None = None) -> dict:
    """Calls OpenRouter for a grounded answer. Returns
    {"answer": str, "error": str | None} -- errors are returned, not raised,
    so callers (including a future eval harness looping over many questions)
    don't crash on a single bad request.

    `notes` (agent.py only) carries the query analysis -- reading,
    assumptions, sub-questions -- which shapes the answer but is never
    evidence (see NOTES_RULE)."""
    context_block = build_context_block(chunks)
    user_prompt = f"Excerpts from the EU AI Act:\n\n{context_block}\n\n"
    system = SYSTEM_PROMPT
    if notes:
        user_prompt += f"{_notes_block(notes)}\n\n"
        system += NOTES_RULE
    user_prompt += f"Question: {question}"
    try:
        response = _chat_completion(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ]
        )
        answer = response.choices[0].message.content.strip()
        return {"answer": answer, "error": None}
    except Exception as exc:  # noqa: BLE001 -- deliberately broad, see docstring
        return {"answer": None, "error": str(exc)}
