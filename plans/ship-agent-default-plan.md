# Ship the agent: Annex 11 fix, then make it the default

> Status: **Done 2026-09-24.** This follows `plans/agentic-rag-plan.md`.

## 1. Keep named provisions on top (Annex 11 fix)

**Problem:** "what does Annex 11 talk about?" retrieves Annex XI through metadata pinning
on the original question. RRF fusion then ranks those chunks below the rewritten
queries' results, and they land at positions 9–10, so R@8 goes from 1 to 0.

**Fix, in `agent.gather_evidence`:** after the evidence is chosen (RRF or select), put
the original question's `retrieved_by == "metadata"` chunks first, in their existing
order, and cap the total at `AGENT_MAX_EVIDENCE`. The number of pinned chunks is
already capped at `TOP_K - MIN_SEMANTIC_SLOTS` by `query._retrieve`. This is the same
"metadata first" contract `query._retrieve` gives the baseline.

**Check:**
- The named-provision questions (Annex 11, Article 97, Article 112) have the named
  provision at rank 1.
- One `reflect` pass over the 36 answerable questions scores no lower than 47/61 in
  the top 8.

## 2. Make the agent the default

- `python query.py` answers with `agent.answer_question_agentic`.
- `python query.py --baseline` keeps the old single-search pipeline.
- The eval harness keeps `--system rag|agentic`. Its default stays `rag`, so older
  comparisons still mean the same thing.
- Document the golden-set relabelling (2 questions moved from should-refuse to
  answerable) next to the results. That is already in `plans/agentic-rag-plan.md`.

## Results

- **Named provisions:** Annex 11, Article 97 and Article 112 all come back with the
  named provision at ranks 1–3.
- **One `reflect` pass on the updated golden set:** 48 of 53 expected sources in the
  top 8, with mean recall@8 at 0.940. Before the fix, the eval `317b0fea` had 0.912. The
  gain is Annex 11 recovered; nothing else got worse.
- **Still missing a source:** requirements for high risk, R&D, deployer +
  transparency, and recruitment penalties.
- **REPL:** `python query.py` now runs the agent, and `python query.py --baseline`
  runs the old pipeline. Both were checked.
- **Full eval after the fix** (same updated golden set):

| Metric | rag `dd5e7afc` | agentic before fix `317b0fea` | agentic after fix `35fe63c2` |
|---|---|---|---|
| answer correctness | 0.707 | 0.902 | 0.878 |
| answer relevance | 0.805 | 1.000 | 1.000 |
| groundedness | 0.951 | 0.976 | 0.927 |
| retrieval (strict) | 0.737 | 0.895 | 0.895 |
| top-3 citation hit | 0.694 | 0.917 | 0.944 |
| recall@1 | 0.394 | 0.338 | 0.394 |
| recall@3 | 0.644 | 0.806 | 0.833 |
| recall@8 | 0.778 | 0.912 | 0.940 |
| mean latency | 5.0 s | 14.9 s | 15.6 s |

  - **Retrieval changes are all from the fix:** Annex 11's R@8 goes from 0 to 1, and
    recall@1 and top-3 rise.
  - **Judge changes are on questions the fix doesn't touch.** None of them names a
    provision, and their retrieval is identical between the two runs:
    - "how must high-risk systems be tested?": correctness 1 → 0;
    - EU database and deployer: groundedness 1 → 0.

    That's 1 correctness flip and 2 groundedness flips, within the 1–4-question noise
    of single judged runs.
  - The baseline itself moved between `ecfbeb79` and `dd5e7afc` with no code change:
    correctness 0.732 → 0.707, groundedness 0.902 → 0.951. That's judge noise.
  - **Guardrails:** both should-refuse questions refused (PASS); unsupported
    citations 2 of 41, the same two as before (PASS). Groundedness is below this
    run's baseline (0.927 vs 0.951) but above the earlier baseline run (0.902);
    judge noise, as above.
  - **Scoring glitch:** the first scoring of `35fe63c2` read "summarize article 114"
    back from LangSmith before the run had synced (empty outputs), so it scored as
    not refused. Re-scoring with `--experiment`, which is free, fixed it. The local
    results JSON had the correct refusal throughout.
