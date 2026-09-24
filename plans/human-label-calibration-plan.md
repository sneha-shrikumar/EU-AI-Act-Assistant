# Human label overrides: recalculate scores and calibrate the judges

> Status: **Done 2026-09-24.**

## Trigger

In `eval_all_eu-ai-act-rag-342c66bc.xlsx` the user overrode two judge scores
from 0 to 1:

| judge | question | judge's reason for 0 | what the judge got wrong |
|---|---|---|---|
| answer_relevance | "how does this act apply if we are a company providing Ai services to the US?" | answer "explicitly states it cannot answer the specific case" | The answer gave the applicable territorial-scope rules for the scenario, then flagged a limit. Being honest about scope is not evasion. |
| answer_correctness | "If we use AI for biometric categorisation in a workplace, is it prohibited, high-risk, or neither?" (ref: "Yes, high risk") | a prohibited-vs-high-risk caveat "contradicts" the reference | The candidate still reaches "high-risk". A caveat that refines the conclusion without reversing it is extra detail, not contradiction. |

## Changes

1. **`evals/human_labels.json`** (new) is the store of human corrections. Each
   entry holds experiment, question, judge key, judge score, human score,
   judge reasoning and a `lesson` (the general rule the correction teaches).
2. **`evals/judges.py`**
   - Prompt rules generalised from the two corrections: relevance
     (answer-then-flag-limits is relevant), correctness (caveats that refine
     but don't reverse the reference's conclusion are fine).
   - Each judge's system prompt gets a "Calibration from human review"
     section built from `human_labels.json` entries for that judge, so future
     corrections feed in automatically without prompt edits.
3. **`run_all_evals.py --import-labels <edited.xlsx>`**
   - Diffs the edited judge columns against the judge's LangSmith scores and
     appends new corrections to `human_labels.json` (the `lesson` can be
     filled in afterwards).
   - The xlsx is rebuilt with human labels applied (overridden cells marked
     blue), and the summary is recalculated. It reports both the corrected
     score and the raw judge score, plus judge-vs-human agreement. A backup of
     the edited file is kept.
   - Any later rebuild of that experiment's sheet also applies its human labels,
     so re-scoring never silently reverts them.

## Caveat

The corrected questions are in the golden set itself, so the calibration
examples partly "teach to the test" on those two rows. The generalised prompt
rules are the part that should transfer. Agreement on the corrected rows
alone is not evidence the judge improved.

## Verification

- Import the user's sheet. Relevance should become 30/41 and correctness
  28/41.
- Replay the two corrected rows through the updated judges (2 judge calls)
  and confirm both now score 1.
