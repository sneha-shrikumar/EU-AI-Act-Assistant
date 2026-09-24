# Relax the LLM refusal rule

> Status: **Done 2026-09-24. Results below.** Priority 1 in `retrieval-options-analysis.md`.

## Problem

In `eu-ai-act-rag-342c66bc`, 10 answerable questions were refused, all by the
LLM (gate 2). The similarity gate never fired. At least 2 of the 10 had the answer
at rank 1: "is sex trafficking a prohibited AI practice?" and "is an AI system that
provides entertainment to children a prohibited AI practice?", both with Article 5
at rank 1. The cause is prompt rule 4 in `llm.py`:

> "Never guess, infer beyond what the text states... When in doubt, refuse per rule 3."

This rule reads *applying* a provision to the user's scenario as "inferring",
and any gap as "doubt", so partial answers become full refusals.

## Changes

1. **`llm.py` SYSTEM_PROMPT**
   - Keep: answer only from the excerpts, cite every claim, add no outside
     legal knowledge.
   - New: applying a provision in the excerpts to the user's situation is
     allowed, because that's what the question asks for. Say which provision
     and why it covers the situation.
   - New: if the excerpts answer part of the question, answer that part and state
     plainly what they don't cover.
   - Refusal only when the excerpts have nothing relevant to the core of the
     question, or the question depends on something they don't contain (a
     provision, amendment or event not in them). Nonexistent provisions or events
     should still be refused, to protect the 4 out-of-scope questions.
   - The exact refusal sentence is used **only** for a full refusal and must never
     appear inside a partial answer. The gap statement uses different wording.
2. **Refusal detection** (`query.py`, and `_is_refusal` in `evals/judges.py`):
   currently "the refusal sentence appears anywhere in the answer". A partial
   answer that happened to quote it would be flagged as refused. Change to "the
   answer opens with the refusal sentence" (a short lead-in is tolerated, and
   any explanation after it still counts as a refusal). A first version that
   required the answer to be short missed refusals where the model explains
   why; the spot check caught this.

## Verification

Full `run_all_evals.py` at `--concurrency 2` (earlier runs hit OpenRouter 402 credit
errors at 4), compared with `342c66bc`.

- **Primary metric, judge-independent:** refusals of answerable questions
  (was 10/37), and the 4 out-of-scope questions must still be refused.
- The judge prompts changed after `342c66bc` (human-label calibration), so
  relevance and correctness shifts are partly the judge. The refusal counts
  and groundedness are the cleaner signal for this change.
- Groundedness must not drop: answering more must not mean inventing more.

## Results (2026-09-24)

Run `eu-ai-act-rag-e6e5c965` compared with `eu-ai-act-rag-342c66bc` (judge-only scores). No judge
errors at `--concurrency 2`.

| metric | 342c66bc | e6e5c965 |
|---|---|---|
| **answerable questions refused** | **10 / 37** | **4 / 37** |
| out-of-scope questions refused (should be all) | 4 / 4 | 4 / 4 |
| answer relevance | 0.707 | 0.854 |
| answer correctness | 0.659 | 0.780 |
| groundedness | 0.976 | 0.927 |
| retrieval metrics | unchanged (prompt-only change) | unchanged |

- **Newly answered, now correct:** sex trafficking, children's entertainment,
  bias examination, AI recruitment penalties. Newly answered but still judged
  incorrect: CV screening (relevance 1, correctness 0) and deployer.
- **The 4 remaining refusals are all retrieval misses or structural questions:**
  R&D (R@8 = 0), "dated as?" (preamble not retrieved), "how many annexes?"
  (no single chunk answers it), and penalties for minorities (R@8 = 0). The model
  now refuses only when the evidence really isn't there.
- **Groundedness cost: 3 answers dropped 1 -> 0**
  - *Exclusions from scope:* the answer cites military/defence to "Article
    2(4)". The old answer had the **same** misattribution and was passed, so this
    one is a pre-existing error that the judge caught this time, not a regression.
  - *Student exam tool:* both old and new answers classify it under Annex
    III(3)(d) (exam monitoring), which doesn't fit grading; the old answer was
    passed. The new answer admits the mismatch ("while your primary function is
    evaluating answers...") yet still applies (3)(d), which the judge flagged. It's
    the same pre-existing stretch, made visible, and the relaxed rule didn't fix it.
  - *Real-world testing:* the new, more expansive answer adds a synthesised
    claim that goes past the excerpt (it conflates the Article 60(4) submission
    with Annex IX registration). This one **is** caused by the relaxed rule:
    "apply the provision" can become "stretch the provision".
- **Caveat:** the relevance and correctness judge prompts gained human-label
  calibration after 342c66bc, so part of those gains is the judge. The refusal
  counts are judge-independent and are the reliable signal here.

Follow-up worth considering: add a line to rule 3 telling the model to name the
closest provision's actual scope when it doesn't clearly fit, rather than
claiming it applies. That targets the stretch cases (student exam tool,
real-world testing).
