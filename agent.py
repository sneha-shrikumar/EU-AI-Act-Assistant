"""Agentic RAG: a fixed sequence of LLM-driven stages in front of the same
grounded generator query.py uses. See plans/agentic-rag-plan.md.

    1. ANALYSE   (LLM)  reading, assumptions, sub-questions, and search queries
                        rewritten into the Act's own vocabulary
    2. RETRIEVE  (code) search the ORIGINAL question plus every rewrite, fuse
                        the ranked lists with reciprocal rank fusion (RRF)
    3. REFLECT   (LLM)  is each sub-question covered? if not: new searches, and
                        provisions the retrieved TEXT refers to -> back to 2
    4. SELECT    (LLM)  rank the pool, keep <= AGENT_MAX_EVIDENCE chunks
    then llm.generate_answer on the kept chunks, plus a citation post-check.

The LLM decides what to search and when the evidence is enough; the code fixes
the order and the budget, so runs vary little and every stage can be switched
off (config.AGENT_STAGES) to measure what it contributes.

What it will NOT do, and where that is enforced:
  * pin provisions the model names from its own memory: rewritten queries run
    with pin=False, and a follow-up reference is fetched only if the user's
    question or a retrieved chunk's text names it (_allowed_refs);
  * answer from anything but retrieved excerpts: generation is llm.py's
    grounded prompt; the analysis notes are passed as "not evidence";
  * cite what it did not retrieve: _unsupported_citations flags it;
  * browse the table of contents: added only for structural questions;
  * loop without limit: every stage has a hard cap from config.

Run directly for a REPL that prints each stage:  python agent.py
"""
import json
import re

from langsmith import traceable

import config
import provision_refs
import query
from embeddings import embed_query
from eval_retrieval import parse_refs
from llm import complete_json, generate_answer, is_refusal

RRF_K = 60  # the standard RRF constant; damps the weight of rank-1 hits

ANALYSE_PROMPT = f"""You plan searches over the EU AI Act (Regulation (EU) \
2024/1689) for a retrieval system. You do NOT answer the question.

Return ONLY a JSON object:
{{"reading": "one sentence: what the user wants to know",
  "assumptions": ["facts the question implies but does not state, e.g. the user's role (provider, deployer, importer, distributor) or use case"],
  "sub_questions": [{{"question": "...", "queries": ["...", "..."]}}],
  "structural": false}}

Rules:
- Split the question into at most {config.AGENT_MAX_SUBQUESTIONS} self-contained \
sub-questions; use exactly one if it asks one thing.
- For each sub-question write 1-{config.AGENT_QUERIES_PER_SUB} search queries \
phrased the way the legal text itself would state the answer: formal legal \
vocabulary, full terms instead of abbreviations or everyday words (e.g. \
"research and development", not "R&D"). Queries are matched by semantic \
similarity, so write descriptive phrases, not keyword lists or questions.
- Never put article, annex or recital numbers in a query unless the user \
named them.
- "assumptions" is [] when nothing needs assuming.
- "structural" is true ONLY when the question asks how the Act is organised: \
how many annexes, articles or recitals it has, or which chapter or section a \
topic sits in. Questions about the Act's date, title, scope or content are \
NOT structural."""

REFLECT_PROMPT = """You check whether excerpts retrieved from the EU AI Act \
cover each sub-question well enough to answer it. You do NOT answer.

Return ONLY a JSON object:
{"coverage": [{"sub_question": "...", "covered": true, "missing": ""}],
 "new_queries": [],
 "follow_refs": []}

Rules:
- covered = the excerpts contain the text needed to answer that sub-question.
- new_queries: up to 3 search phrases for what is missing, in the Act's formal \
vocabulary, as descriptive phrases. [] if everything is covered.
- follow_refs: up to 3 provisions that an excerpt's text EXPLICITLY refers to \
and that are needed for the answer (e.g. an excerpt says "the information \
listed in Annex VIII" and the question asks what that information is). Write \
each like "Annex VIII" or "Article 6". Never name a provision that does not \
appear in the excerpts' text. [] if none."""

SELECT_PROMPT = f"""You choose which excerpts of the EU AI Act a writer \
needs to answer the question. You do NOT answer.

Return ONLY a JSON object: {{"keep": [excerpt numbers, most important first]}}

Rules:
- Keep at most {config.AGENT_MAX_EVIDENCE}.
- Put first the excerpts that directly state the rule, definition, \
obligation, list or date the question asks about.
- Also keep excerpts the answer depends on (a definition, the classification \
that triggers an obligation).
- Leave out excerpts that are off-topic for every sub-question."""

# Parenthetical citations in an answer: "(Article 6(2))", "(Recitals 24-25)",
# "(Annex III(4), Article 26(1))" -- one level of nested parentheses.
_CITATION_RE = re.compile(
    r"\(((?:Articles?|Annex(?:es)?|Recitals?|Preamble)\b[^()]*(?:\([^()]*\)[^()]*)*)\)"
)


class _Budget:
    """Counts LLM calls for one question; stages consult it before calling."""
    def __init__(self):
        self.calls = 0

    def call_json(self, system: str, user: str) -> dict:
        self.calls += 1
        return complete_json(system, user)


# --- Pool: every chunk retrieved so far, fused by RRF -------------------------

class _Pool:
    def __init__(self):
        self.chunks: dict[str, dict] = {}
        self.lists: list[list[str]] = []  # ranked id lists, one per search

    def add(self, results: list[dict]) -> None:
        ranked = []
        for c in results:
            seen = self.chunks.get(c["id"])
            if seen is None or c["similarity"] > seen["similarity"]:
                self.chunks[c["id"]] = c
            ranked.append(c["id"])
        if ranked:
            self.lists.append(ranked)

    def ranked(self) -> list[dict]:
        scores: dict[str, float] = {}
        for ranked in self.lists:
            for rank, cid in enumerate(ranked, 1):
                scores[cid] = scores.get(cid, 0.0) + 1 / (RRF_K + rank)
        order = sorted(scores, key=lambda cid: -scores[cid])
        return [self.chunks[cid] for cid in order]


def _as_excerpts(chunks: list[dict]) -> str:
    return "\n\n---\n\n".join(
        f"[{i}] {c['citation']}\n{c['text']}" for i, c in enumerate(chunks, 1)
    )


def _strings(value, cap: int) -> list[str]:
    if not isinstance(value, list):
        return []
    return [s.strip() for s in value if isinstance(s, str) and s.strip()][:cap]


# --- Stage 1: analyse ---------------------------------------------------------

@traceable(run_type="chain", name="agent.analyse")
def _analyse(question: str, budget: _Budget, fallbacks: list[str]) -> dict:
    """Default on bad output: the question itself as the only sub-question,
    with no rewrites -- i.e. exactly today's single search."""
    default = {"reading": "", "assumptions": [], "structural": False,
               "sub_questions": [{"question": question, "queries": []}]}
    try:
        raw = budget.call_json(ANALYSE_PROMPT, f"Question: {question}")
    except ValueError:
        fallbacks.append("analyse")
        return default
    subs = []
    for sub in (raw.get("sub_questions") or [])[: config.AGENT_MAX_SUBQUESTIONS]:
        if isinstance(sub, dict) and isinstance(sub.get("question"), str):
            subs.append({"question": sub["question"].strip(),
                         "queries": _strings(sub.get("queries"), config.AGENT_QUERIES_PER_SUB)})
    if not subs:
        # Valid JSON with no sub-questions (common for one-line structural
        # questions): keep the analysis, search the question as asked.
        subs = default["sub_questions"]
    return {
        "reading": raw.get("reading") if isinstance(raw.get("reading"), str) else "",
        "assumptions": _strings(raw.get("assumptions"), 4),
        "structural": raw.get("structural") is True,
        "sub_questions": subs,
    }


# --- Stage 2: retrieve --------------------------------------------------------

@traceable(run_type="chain", name="agent.retrieve")
def _search(pool: _Pool, queries: list[str], k: int) -> None:
    """Rewritten queries: similarity only (pin=False), never metadata pinning."""
    for q in queries:
        pool.add(query._retrieve(q, k, pin=False))


def _allowed_refs(question: str, chunks: list[dict]) -> dict[str, set[str]]:
    """Provisions the user named, or that retrieved text names. A follow-up
    reference outside this set came from the model's memory, and is dropped."""
    allowed: dict[str, set[str]] = {}
    for text in [question] + [c["text"] for c in chunks]:
        for kind, values in provision_refs.parse_question(text).items():
            allowed.setdefault(kind, set()).update(values)
    return allowed


@traceable(run_type="chain", name="agent.follow_refs")
def _follow(pool: _Pool, refs: list[str], question: str, trace: dict) -> None:
    ids, metadatas = query._corpus_index
    allowed = _allowed_refs(question, list(pool.chunks.values()))
    qvec = embed_query(question)
    for ref in refs:
        wanted = provision_refs.parse_question(ref)
        kept = {kind: values & allowed.get(kind, set())
                for kind, values in wanted.items()}
        kept = {kind: values for kind, values in kept.items() if values}
        if not kept:
            trace["dropped_refs"].append(ref)
            continue
        matched = provision_refs.match_chunk_ids(kept, ids, metadatas)
        if not matched:
            continue
        trace["followed_refs"].append(ref)
        # A whole provision can be many chunks (Article 3 has ~10); keep the
        # few most relevant to the question.
        pool.add(query._query(query._get_collection(), qvec, min(len(matched), 4),
                              "reference", ids=matched))


# --- Stage 3: reflect ---------------------------------------------------------

@traceable(run_type="chain", name="agent.reflect")
def _reflect(question: str, plan: dict, pool: _Pool, budget: _Budget,
             fallbacks: list[str]) -> dict:
    """Default on bad output: nothing more to fetch."""
    view = pool.ranked()[: config.AGENT_POOL_VIEW]
    subs = "\n".join(f"{i}. {s['question']}" for i, s in enumerate(plan["sub_questions"], 1))
    user = (f"Question: {question}\n\nSub-questions:\n{subs}\n\n"
            f"Excerpts:\n\n{_as_excerpts(view)}")
    try:
        raw = budget.call_json(REFLECT_PROMPT, user)
    except ValueError:
        fallbacks.append("reflect")
        return {"coverage": [], "new_queries": [], "follow_refs": []}
    return {
        "coverage": raw.get("coverage") if isinstance(raw.get("coverage"), list) else [],
        "new_queries": _strings(raw.get("new_queries"), 3),
        "follow_refs": _strings(raw.get("follow_refs"), 3),
    }


# --- Stage 4: select ----------------------------------------------------------

@traceable(run_type="chain", name="agent.select")
def _select(question: str, plan: dict, pool: _Pool, budget: _Budget,
            fallbacks: list[str]) -> list[dict]:
    """Default on bad output: the RRF order."""
    view = pool.ranked()[: config.AGENT_POOL_VIEW]
    subs = "\n".join(f"{i}. {s['question']}" for i, s in enumerate(plan["sub_questions"], 1))
    user = (f"Question: {question}\n\nSub-questions:\n{subs}\n\n"
            f"Excerpts:\n\n{_as_excerpts(view)}")
    try:
        raw = budget.call_json(SELECT_PROMPT, user)
    except ValueError:
        fallbacks.append("select")
        return view[: config.AGENT_MAX_EVIDENCE]
    keep, seen = [], set()
    for n in raw.get("keep") or []:
        if isinstance(n, int) and 1 <= n <= len(view) and n not in seen:
            seen.add(n)
            keep.append(view[n - 1])
    if not keep:
        fallbacks.append("select")
        return view[: config.AGENT_MAX_EVIDENCE]
    return keep[: config.AGENT_MAX_EVIDENCE]


# --- Table of contents (structural questions only) ----------------------------

def _first_line(title: str | None) -> str:
    # Some titles carry the provision's lead-in sentence after a newline
    # (e.g. Article 3's "For the purposes of this Regulation ..."); the
    # contents list wants the heading only.
    return (title or "").strip().split("\n", 1)[0].strip()


def _table_of_contents() -> dict:
    """A synthetic excerpt listing the Act's structure, built from chunk
    metadata. Cited as "Table of contents" -- it is not a provision."""
    _, metadatas = query._corpus_index
    articles, annexes, recitals = {}, {}, 0
    for m in metadatas:
        if m.get("type") == "Article":
            articles.setdefault(m["number"], m)
        elif m.get("type") == "Annex":
            annexes.setdefault(m["number"], _first_line(m.get("title")))
        elif m.get("type") == "Recital":
            recitals = max(recitals, int(str(m["number"]).split("-")[-1]))

    def art_key(n):
        return int(re.match(r"\d+", n).group())

    lines = [f"The Act has a preamble, {recitals} recitals, {len(articles)} "
             f"articles and {len(annexes)} annexes.", ""]
    chapter = section = None
    for num in sorted(articles, key=art_key):
        m = articles[num]
        if m.get("chapter") != chapter:
            chapter, section = m.get("chapter"), None
            lines.append(f"Chapter {chapter}")
        if m.get("section") and m.get("section") != section:
            section = m.get("section")
            lines.append(f"  Section {section}")
        lines.append(f"    Article {num}: {_first_line(m.get('title'))}")
    lines.append("")
    for num in sorted(annexes, key=provision_refs._roman_to_int):
        lines.append(f"Annex {num}: {annexes[num]}")
    return {"id": "table-of-contents", "text": "\n".join(lines),
            "citation": "Table of contents", "similarity": 1.0,
            "metadata": {"citation": "Table of contents", "type": "Contents"},
            "retrieved_by": "contents"}


# --- Orchestration ------------------------------------------------------------

@traceable(run_type="chain", name="agent.gather_evidence")
def gather_evidence(question: str, k: int = config.TOP_K,
                    stages: str | None = None) -> dict:
    """Stages 1-4 without answer generation. Returns the kept evidence (rank
    order), the analysis notes, whether refusal gate 1 passed, and a trace.
    Used by answer_question_agentic and by the retrieval diagnostic."""
    stages = stages or config.AGENT_STAGES
    query._get_collection()  # ensures _corpus_index is loaded
    budget, fallbacks = _Budget(), []
    trace = {"queries": [], "rounds": [], "followed_refs": [], "dropped_refs": []}

    plan = _analyse(question, budget, fallbacks)

    pool = _Pool()
    original = query._retrieve(question, k)  # pinning on: the USER named these
    pool.add(original)
    rewrites = [q for s in plan["sub_questions"] for q in s["queries"]]
    trace["queries"] = rewrites
    _search(pool, rewrites, k)

    if stages in ("reflect", "select"):
        for _ in range(config.AGENT_REFLECT_ROUNDS):
            verdict = _reflect(question, plan, pool, budget, fallbacks)
            trace["rounds"].append(verdict)
            if not verdict["new_queries"] and not verdict["follow_refs"]:
                break
            _search(pool, verdict["new_queries"], k)
            _follow(pool, verdict["follow_refs"], question, trace)

    if stages == "select":
        evidence = _select(question, plan, pool, budget, fallbacks)
    else:
        evidence = pool.ranked()[: config.AGENT_MAX_EVIDENCE]

    # Provisions the USER named (metadata-pinned on the original question) go
    # first, exactly as query._retrieve orders them for the baseline. RRF
    # fusion otherwise ranks them under the rewrites' results -- "what does
    # Annex 11 talk about?" lost Annex XI from the top 8 that way.
    pinned = [c for c in original if c["retrieved_by"] == "metadata"]
    pinned_ids = {c["id"] for c in pinned}
    evidence = (pinned + [c for c in evidence if c["id"] not in pinned_ids]
                )[: config.AGENT_MAX_EVIDENCE]

    if plan["structural"]:
        # Appended after the cap, never in front: the analyse stage sometimes
        # flags borderline questions ("what is it dated?") as structural, and
        # the contents list must not push a real provision out of the ranks.
        evidence = evidence + [_table_of_contents()]

    # Refusal gate 1, judged on the ORIGINAL question's search exactly as
    # query.answer_question does: rewrites can make an out-of-scope question
    # look topical, so they don't get a vote here.
    named = any(c["retrieved_by"] == "metadata" for c in original)
    top = max((c["similarity"] for c in original), default=0.0)
    return {
        "evidence": evidence,
        "notes": {"reading": plan["reading"], "assumptions": plan["assumptions"],
                  "sub_questions": [s["question"] for s in plan["sub_questions"]]},
        "gate_ok": bool(original) and (named or top >= config.SIMILARITY_THRESHOLD),
        "llm_calls": budget.calls,
        "fallbacks": fallbacks,
        "trace": trace,
    }


def _unsupported_citations(answer: str, evidence: list[dict]) -> list[str]:
    """Provisions cited in the answer that no kept chunk covers."""
    kept = set().union(*(parse_refs(c["citation"]) for c in evidence)) if evidence else set()
    cited = set()
    for inner in _CITATION_RE.findall(answer or ""):
        # Drop paragraph groups first: parse_refs would read the bare "3" in
        # "(Article 71(2) and (3))" as a second article.
        cited |= parse_refs(re.sub(r"\([^()]*\)", "", inner))
    return sorted(f"{kind} {num}" for kind, num in cited - kept)


@traceable(run_type="chain", name="answer_question_agentic")
def answer_question_agentic(question: str, k: int = config.TOP_K) -> dict:
    """Same return shape as query.answer_question, plus:
        notes                  reading / assumptions / sub-questions
        llm_calls              LLM calls spent, generation included
        unsupported_citations  cited provisions not among the kept chunks
        agent_fallback         stages that fell back to their default, or
                               ["pipeline"] if the whole thing fell back to
                               query.answer_question
    """
    try:
        gathered = gather_evidence(question, k)
    except Exception as exc:  # noqa: BLE001 -- any failure: today's pipeline
        result = query.answer_question(question, k)
        result.update(notes=None, llm_calls=None, unsupported_citations=[],
                      agent_fallback=["pipeline"], agent_error=str(exc))
        return result

    evidence = gathered["evidence"]
    base = {
        "citations": [c["citation"] for c in evidence],
        "retrieved_chunks": evidence,
        "notes": gathered["notes"],
        "agent_fallback": gathered["fallbacks"],
        "agent_trace": gathered["trace"],
        "unsupported_citations": [],
    }
    if not evidence or not gathered["gate_ok"]:
        return {**base, "answer": config.REFUSAL_MESSAGE, "refused": True,
                "error": None, "llm_calls": gathered["llm_calls"]}

    result = generate_answer(question, evidence, gathered["notes"])
    calls = gathered["llm_calls"] + 1
    if result["error"]:
        return {**base, "answer": None, "refused": False,
                "error": result["error"], "llm_calls": calls}
    answer = result["answer"]
    return {**base, "answer": answer, "refused": is_refusal(answer), "error": None,
            "llm_calls": calls,
            "unsupported_citations": _unsupported_citations(answer, evidence)}


def _repl():
    print(f"EU AI Act agentic RAG (stages: {config.AGENT_STAGES}) -- 'exit' to stop.\n")
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
        g = gather_evidence(question)
        print("\n-- notes:", json.dumps(g["notes"], indent=2, ensure_ascii=False))
        print("-- rewritten queries:", *g["trace"]["queries"], sep="\n   ")
        for i, r in enumerate(g["trace"]["rounds"], 1):
            print(f"-- reflect round {i}:", json.dumps(r, ensure_ascii=False))
        print("-- followed refs:", g["trace"]["followed_refs"],
              "| dropped (not in retrieved text):", g["trace"]["dropped_refs"])
        print("-- evidence:", ", ".join(c["citation"] for c in g["evidence"]))
        print(f"-- gate ok: {g['gate_ok']}  llm calls: {g['llm_calls']}  "
              f"fallbacks: {g['fallbacks']}")
        if not g["gate_ok"]:
            print(f"\n{config.REFUSAL_MESSAGE}\n")
            continue
        result = generate_answer(question, g["evidence"], g["notes"])
        answer = result["answer"] or f"[error] {result['error']}"
        print(f"\n{answer}")
        unsupported = _unsupported_citations(result["answer"], g["evidence"])
        if unsupported:
            print(f"\n[unsupported citations: {', '.join(unsupported)}]")
        print()


if __name__ == "__main__":
    _repl()
