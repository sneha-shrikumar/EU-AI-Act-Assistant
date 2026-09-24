"""LLM-as-judge evaluators for the LangSmith experiment: groundedness,
answer relevance, answer correctness.

These replace the hand-scoring that produced the score columns in the
exported experiment CSVs. Conventions are deliberately inherited from that
manual baseline so new runs stay comparable with it -- see
plans/llm-as-judge-evaluators-plan.md for why each one is what it is:

  * scores are binary 0/1, never graded, because the baseline is 0/1;
  * a refusal scores groundedness=1 -- a refusal asserts nothing, so it is
    trivially grounded;
  * relevance and correctness both honour reference_outputs["should_answer"]:
    when the golden dataset marks a question unanswerable from the Act,
    refusing IS the right response and scores 1 on both. Only a refusal to an
    answerable question scores 0.

Each judge sees only what its own axis needs: groundedness never sees the
question, correctness never sees the excerpts. That keeps one axis from
laundering a failure on another -- a fluent answer to the wrong question
cannot talk its way into a groundedness pass.

Robustness follows llm.py's house style: a judge RETURNS problems instead of
raising them. A judging failure must never turn a perfectly good pipeline run
into a failed experiment row, so an API error or an unparseable verdict comes
back as score=None with the detail in the comment. A missing score is honest;
a fabricated 0 would quietly poison the aggregate.
"""
import json
import re
import sys
from pathlib import Path

# Importable both as `evals.judges` and as a sibling of the pipeline modules.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langsmith import traceable  # noqa: E402

import config  # noqa: E402
from llm import _client, _set_llm_usage, is_refusal  # noqa: E402

# --- Shared judge plumbing --------------------------------------------------

_JSON_INSTRUCTION = (
    "Reply with ONLY a JSON object, no prose and no code fence, in exactly "
    "this shape:\n"
    '{"reasoning": "<one or two sentences justifying the score>", '
    '"score": 0 or 1}'
)


HUMAN_LABELS_PATH = Path(__file__).resolve().parent / "human_labels.json"


def load_human_labels() -> list[dict]:
    """Human corrections of past judge verdicts (see run_all_evals.py
    --import-labels). Missing or unreadable file -> no calibration."""
    try:
        return json.loads(HUMAN_LABELS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []


def _calibration(key: str) -> str:
    """A "Calibration from human review" block for one judge's system prompt,
    built from every human correction of that judge. Read at call time, so a
    newly imported correction takes effect on the next run with no prompt
    edit. The lesson is the part meant to generalise; the specific case
    anchors it."""
    labels = [lab for lab in load_human_labels() if lab.get("key") == key]
    if not labels:
        return ""
    lines = ["CALIBRATION FROM HUMAN REVIEW. A human reviewer overruled these "
             "past verdicts of yours. Apply the lessons to every answer you "
             "grade, not only to these questions:"]
    for lab in labels:
        lines.append(
            f"- Question: {lab['question']!r}. You scored {lab.get('judge_score')} "
            f"because: {(lab.get('judge_reasoning') or '').strip()} "
            f"Human verdict: {lab['human_score']}."
            + (f" Lesson: {lab['lesson']}" if lab.get("lesson") else "")
        )
    return "\n".join(lines)


def _extract_json(text: str) -> dict | None:
    """Parse the judge's verdict tolerantly.

    OpenRouter's json-mode support varies by upstream provider, so we do not
    rely on response_format and instead cope with what models actually emit:
    a bare object, a fenced code block, or an object with a stray sentence
    around it. Returns None when nothing parseable is found -- the caller
    turns that into score=None rather than guessing.
    """
    if not text:
        return None
    candidate = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", candidate, re.DOTALL)
    if fenced:
        candidate = fenced.group(1).strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        # Fall back to the first {...} span in the text.
        brace = re.search(r"\{.*\}", candidate, re.DOTALL)
        if not brace:
            return None
        try:
            parsed = json.loads(brace.group(0))
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


@traceable(
    run_type="llm",
    name="Judge",
    metadata={"ls_provider": "openrouter", "ls_model_name": config.JUDGE_MODEL},
)
def _judge_completion(messages: list[dict]):
    response = _client().chat.completions.create(
        model=config.JUDGE_MODEL,
        temperature=config.JUDGE_TEMPERATURE,
        messages=messages,
        extra_body={"usage": {"include": True}},
    )
    _set_llm_usage(response)
    return response


def _call_judge(key: str, system_prompt: str, user_prompt: str | list[dict]) -> dict:
    """Run one judge and normalise its verdict into LangSmith feedback.

    `user_prompt` may be a list of content blocks, so a caller can put a
    cache_control breakpoint on a large shared prefix (see groundedness)."""
    try:
        calibration = _calibration(key)
        system = f"{system_prompt}\n\n{calibration}" if calibration else system_prompt
        response = _judge_completion([
            {"role": "system", "content": f"{system}\n\n{_JSON_INSTRUCTION}"},
            {"role": "user", "content": user_prompt},
        ])
        raw = (response.choices[0].message.content or "").strip()
    except Exception as exc:  # noqa: BLE001 -- see module docstring
        return {"key": key, "score": None, "comment": f"judge error: {exc}"}

    verdict = _extract_json(raw)
    if verdict is None or "score" not in verdict:
        return {"key": key, "score": None,
                "comment": f"unparseable judge response: {raw[:500]}"}

    try:
        score = int(verdict["score"])
    except (TypeError, ValueError):
        return {"key": key, "score": None,
                "comment": f"non-numeric score in judge response: {raw[:500]}"}

    if score not in (0, 1):
        return {"key": key, "score": None,
                "comment": f"score outside 0/1: {raw[:500]}"}

    return {"key": key, "score": score,
            "comment": str(verdict.get("reasoning", "")).strip()}


def _is_refusal(answer: str | None) -> bool:
    """Same test as the pipeline's own `refused` flag (llm.is_refusal)."""
    return is_refusal(answer)


def _refused(outputs: dict) -> bool:
    """Prefer the pipeline's own flag; fall back to text matching when the
    experiment predates it (e.g. re-scoring an older run)."""
    if isinstance(outputs.get("refused"), bool):
        return outputs["refused"]
    return _is_refusal(outputs.get("answer"))


# --- Judge 1: groundedness --------------------------------------------------

_GROUNDEDNESS_SYSTEM = """You are a strict grader checking whether an answer \
is GROUNDED in a set of source excerpts from the EU AI Act.

You are NOT judging whether the answer is helpful, complete, or responsive to \
any question. Judge one thing only: is every factual claim in the answer \
actually supported by the excerpts you were given?

Score 1 only if ALL of the following hold:
- Every factual claim in the answer can be traced to a specific excerpt.
- Inline citations point at the excerpt the claim actually came from, not at \
a different one.
- The answer adds no legal knowledge from outside the excerpts, however \
correct that outside knowledge may be.

Score 0 if the answer states anything the excerpts do not support, \
misattributes a claim to the wrong excerpt, or generalises beyond what the \
text says.

Paraphrase is fine. Quoting is fine. Hedged language ("the excerpts suggest") \
is fine as long as the substance is in the text."""


# Excerpt blocks above this size get a prompt-cache breakpoint. RAG excerpts
# (a few thousand tokens, different every row) stay well under it and are never
# cached -- a cache write would only add a surcharge there.
_CACHE_SOURCE_MIN_CHARS = 20_000


def groundedness(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Is every claim in the answer supported by the retrieved excerpts?

    A refusal short-circuits to 1: it makes no factual claims, so there is
    nothing to be ungrounded about. This reproduces the hand-scored baseline
    (which was 41/41 partly for exactly this reason) rather than quietly
    redefining the metric -- see the plan for the caveat.
    """
    key = "groundedness"
    if outputs.get("error"):
        return {"key": key, "score": None,
                "comment": f"pipeline error, nothing to grade: {outputs['error']}"}
    if _refused(outputs):
        return {"key": key, "score": 1,
                "comment": "Refusal: asserts no facts, so trivially grounded."}

    contexts = outputs.get("contexts") or []
    if not contexts:
        return {"key": key, "score": None,
                "comment": "no retrieved excerpts recorded in outputs; cannot grade"}

    excerpts = "\n\n---\n\n".join(
        f"[{c.get('citation', '?')}]\n{c.get('text', '')}" for c in contexts
    )
    source = f"SOURCE EXCERPTS:\n\n{excerpts}\n\n"
    answer = f"=====\n\nANSWER TO GRADE:\n\n{outputs.get('answer', '')}"
    if len(source) < _CACHE_SOURCE_MIN_CHARS:
        return _call_judge(key, _GROUNDEDNESS_SYSTEM, source + answer)
    # Full-document baseline: the "excerpts" are the whole ~180k-token Act and
    # identical on every row, so cache them. Same text the judge would see as
    # one string -- only the billing changes, so scores stay comparable.
    return _call_judge(key, _GROUNDEDNESS_SYSTEM, [
        {"type": "text", "text": source, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": answer},
    ])


# --- Judge 2: answer relevance ----------------------------------------------

_RELEVANCE_SYSTEM = """You are a strict grader checking whether an answer is \
RELEVANT to the question asked.

You are NOT judging whether the answer is factually correct, well cited, or \
grounded in any source. Judge one thing only: does this text address what the \
user actually asked?

Score 1 if the answer engages with the specific question asked and a reader \
would come away with the thing they asked for.

Score 0 if the answer is off-topic, answers an adjacent-but-different \
question, is evasive, or discusses the general subject area without \
addressing the specific ask. A multi-part question needs every part addressed \
to score 1. If the question asks for a specific locator -- a section, article \
or annex number -- an answer that names a different kind of locator has not \
addressed the ask.

Being honest about limits is NOT evasion. If the answer gives the rules that \
apply to the user's specific scenario and then says what the source does not \
settle, it has addressed the ask and scores 1. Score 0 for evasion only when \
the answer gives nothing usable about the scenario itself.

Length is irrelevant. A short direct answer scores 1; a long essay that \
circles the question scores 0."""


def answer_relevance(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Does the answer address the question that was asked?

    Refusals are resolved against `reference_outputs["should_answer"]` rather
    than always scoring 0. Replaying the hand-scored 23sep baseline showed a
    flat refusal=0 rule disagreeing with the manual labels on exactly the four
    `should_answer: "no"` rows (21, 24, 35, 40), every one of them hand-scored
    1: when a question cannot be answered from the Act, declining IS the
    responsive thing to do, and the baseline treats it that way. Only a
    refusal to an answerable question leaves the ask unaddressed.
    """
    key = "answer_relevance"
    if outputs.get("error"):
        return {"key": key, "score": None,
                "comment": f"pipeline error, nothing to grade: {outputs['error']}"}

    should_answer = str(reference_outputs.get("should_answer", "yes")).strip().lower()
    if _refused(outputs):
        if should_answer == "no":
            return {"key": key, "score": 1,
                    "comment": "Refusal is the responsive answer: the golden "
                               "dataset marks this unanswerable from the Act."}
        return {"key": key, "score": 0,
                "comment": "Refused a question the golden dataset marks "
                           "answerable, so the ask went unaddressed."}

    user = (
        f"QUESTION:\n{inputs.get('question', '')}\n\n"
        f"=====\n\nANSWER TO GRADE:\n\n{outputs.get('answer', '')}"
    )
    return _call_judge(key, _RELEVANCE_SYSTEM, user)


# --- Judge 3: answer correctness --------------------------------------------

_CORRECTNESS_SYSTEM = """You are a strict grader comparing a candidate answer \
against a REFERENCE answer drawn from the EU AI Act.

Judge one thing only: does the candidate convey the same substantive legal \
content as the reference?

Score 1 if the candidate states the substance of the reference. All of these \
are fine and must NOT cost a point:
- paraphrasing instead of quoting;
- extra correct detail beyond the reference;
- a different citation format, or citing the same provision differently;
- different structure, ordering, or headings.

Score 0 if the candidate contradicts the reference, omits the key point the \
reference makes, or answers only part of a multi-part reference.

References are often terse and every element in one counts. If the reference \
names more than one thing -- e.g. "Section 4, article 30" -- the candidate \
must cover all of them; naming only the article when the reference also names \
the section is an omission and scores 0.

The reference is the authority. If the candidate says something the reference \
does not cover, ignore it unless it contradicts the reference.

Caveats and conditions are not contradictions. If the candidate reaches the \
reference's conclusion (e.g. "high-risk") and adds a condition under which a \
different outcome would apply, that refines the answer and still scores 1. \
Score 0 for contradiction only when the candidate's conclusion cannot be \
reconciled with the reference's."""


def answer_correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Does the answer match the golden reference answer?

    `reference_outputs["should_answer"]` is honoured first: when the golden
    dataset says a question is NOT answerable from the Act, refusing is the
    correct behaviour and scores 1, while answering anyway scores 0. Without
    this the refusal gate -- a deliberate feature of the pipeline -- would be
    scored as a defect every time it fires correctly.
    """
    key = "answer_correctness"
    if outputs.get("error"):
        return {"key": key, "score": None,
                "comment": f"pipeline error, nothing to grade: {outputs['error']}"}

    refused = _refused(outputs)
    should_answer = str(reference_outputs.get("should_answer", "yes")).strip().lower()

    if should_answer == "no":
        return {"key": key,
                "score": 1 if refused else 0,
                "comment": ("Correctly refused an unanswerable question."
                            if refused else
                            "Answered a question the golden dataset marks "
                            "unanswerable from the Act.")}

    if refused:
        return {"key": key, "score": 0,
                "comment": "Refused a question the golden dataset marks answerable."}

    reference = reference_outputs.get("expected_answer", "")
    if not reference:
        return {"key": key, "score": None,
                "comment": "no expected_answer in reference_outputs; cannot grade"}

    user = (
        f"QUESTION:\n{inputs.get('question', '')}\n\n"
        f"=====\n\nREFERENCE ANSWER:\n\n{reference}\n\n"
        f"=====\n\nCANDIDATE ANSWER TO GRADE:\n\n{outputs.get('answer', '')}"
    )
    return _call_judge(key, _CORRECTNESS_SYSTEM, user)


ALL_JUDGES = [groundedness, answer_relevance, answer_correctness]
