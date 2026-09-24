"""Run every golden-dataset question through the RAG pipeline as a LangSmith
Experiment, recording each answer together with its latency and cost.

How latency and cost get recorded (both automatic -- no work here):
  * latency  -- LangSmith times each example's target run and shows it as the
                run's Latency.
  * cost/tokens -- llm.py's _set_llm_usage already attaches usage_metadata
                (including OpenRouter's real per-call `total_cost`) to the traced
                LLM child run. Because that run nests under this experiment's
                run, the experiment's Cost/Tokens columns populate on their own.

Prerequisite: LANGSMITH_TRACING=true and LANGSMITH_API_KEY in .env (already set),
so the @traceable chain in query.py/llm.py actually emits runs.

Run from anywhere:
    python evals/run_langsmith_eval.py               # all 41 questions, judged
    python evals/run_langsmith_eval.py --limit 3     # smoke test on first N
    python evals/run_langsmith_eval.py --concurrency 4
    python evals/run_langsmith_eval.py --no-judges   # answers only, no scoring
    python evals/run_langsmith_eval.py --system full_doc   # Opus 5.5 baseline:
                                   # no retrieval, whole Act in context
Each run also writes evals/results/<experiment>.json; feed a rag and a
full_doc experiment to evals/compare_report.py for the comparison page.

Scoring: evals/judges.py attaches three LLM-as-judge evaluators
(groundedness, answer_relevance, answer_correctness), replacing the manual
scoring that produced the score columns in the exported experiment CSVs.
"""
import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

# --- Make the sibling pipeline modules importable, and load creds -----------
BASE_DIR = Path(__file__).resolve().parent.parent  # eu-ai-act-rag/
sys.path.insert(0, str(BASE_DIR))
load_dotenv(BASE_DIR / ".env")

from langsmith import Client, evaluate   # noqa: E402
import config                            # noqa: E402
from query import answer_question        # noqa: E402
from agent import answer_question_agentic  # noqa: E402
from baseline_full_doc import answer_question_full_doc  # noqa: E402
from judges import ALL_JUDGES            # noqa: E402

DATASET_NAME = "EU AI Acts evaluation"


# --system choice -> (answer function, experiment prefix, generator model).
# Both answer functions return the same shape, so everything downstream --
# outputs, judges, LangSmith columns -- is identical between arms.
SYSTEMS = {
    "rag": (answer_question, "eu-ai-act-rag", config.OPENROUTER_MODEL),
    "full_doc": (answer_question_full_doc, "eu-ai-act-opus55-fulldoc",
                 config.BASELINE_MODEL),
    # Prefix names the stage set, so ladder rungs never share an experiment
    # name (see plans/agentic-rag-plan.md).
    "agentic": (answer_question_agentic, f"eu-ai-act-agentic-{config.AGENT_STAGES}",
                config.OPENROUTER_MODEL),
}

# Extra fields the agentic system returns; carried into the run outputs when
# present so they show up in LangSmith and evals/results/*.json.
AGENT_FIELDS = ("notes", "llm_calls", "agent_fallback", "unsupported_citations",
                "agent_trace")


def make_target(answer_fn):
    def run_pipeline(inputs: dict) -> dict:
        """Target function: one dataset example's inputs -> pipeline outputs.

        The @traceable chain inside answer_fn nests under the experiment run
        LangSmith creates for this call, so the answer is stored as the run
        output and latency/cost are captured around it."""
        return _to_outputs(answer_fn(inputs["question"]))
    return run_pipeline


def _to_outputs(result: dict) -> dict:
    return {
        "answer": result["answer"],
        "refused": result["refused"],
        "citations": result["citations"],
        # The groundedness judge grades the answer against the excerpts the
        # model was actually shown, so they have to survive into the run
        # output -- citations alone (labels, no text) are not enough to check
        # whether a claim is supported.
        "contexts": [
            {"citation": c["citation"], "text": c["text"]}
            for c in result["retrieved_chunks"]
        ],
        "error": result["error"],
        **{f: result[f] for f in AGENT_FIELDS if f in result},
    }


def dump_results(results, experiment_name: str, system: str, gen_model: str) -> Path:
    """Write one JSON row per example to evals/results/<experiment>.json, for
    evals/compare_report.py. Cost is NOT here -- LangSmith aggregates it onto
    the root run asynchronously, so compare_report fetches it afterwards.
    `contexts` is dropped: for full_doc it is the whole Act on every row."""
    rows = []
    for row in results:
        run, example = row["run"], row["example"]
        outputs = run.outputs or {}
        latency = ((run.end_time - run.start_time).total_seconds()
                   if run.end_time and run.start_time else None)
        rows.append({
            "run_id": str(run.id),
            "id": (example.metadata or {}).get("id"),
            "category": (example.metadata or {}).get("category"),
            "question": example.inputs.get("question"),
            "should_answer": (example.outputs or {}).get("should_answer"),
            "expected_answer": (example.outputs or {}).get("expected_answer"),
            "answer": outputs.get("answer"),
            "refused": outputs.get("refused"),
            "citations": outputs.get("citations"),
            "error": outputs.get("error"),
            **{f: outputs[f] for f in AGENT_FIELDS if f in outputs},
            "latency_s": latency,
            "scores": {
                r.key: {"score": r.score, "comment": r.comment}
                for r in row["evaluation_results"]["results"]
            },
        })
    out_dir = BASE_DIR / "evals" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{experiment_name}.json"
    out_path.write_text(json.dumps({
        "experiment": experiment_name, "system": system, "model": gen_model,
        "judge_model": config.JUDGE_MODEL, "rows": rows,
    }, indent=2, default=str), encoding="utf-8")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the golden dataset through the "
                                             "RAG pipeline as a LangSmith experiment.")
    ap.add_argument("--system", choices=sorted(SYSTEMS), default="rag",
                    help="rag = the retrieval pipeline (default); agentic = "
                         "agent.py's staged pipeline (config.AGENT_STAGES); "
                         "full_doc = no retrieval, whole Act in BASELINE_MODEL's "
                         "context.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only run the first N examples (cheap smoke test).")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="Parallel examples. Keep low to respect OpenRouter limits.")
    ap.add_argument("--no-judges", action="store_true",
                    help="Run the pipeline without LLM-as-judge scoring (no "
                         "judge tokens billed). The experiment still records "
                         "answers, latency and cost.")
    args = ap.parse_args()

    client = Client()

    # data can be a dataset name (all examples) or an explicit list (for --limit).
    if args.limit is not None:
        data = list(client.list_examples(dataset_name=DATASET_NAME, limit=args.limit))
        print(f"Running first {len(data)} example(s) from {DATASET_NAME!r}.")
    else:
        data = DATASET_NAME
        print(f"Running ALL examples from {DATASET_NAME!r}.")

    evaluators = [] if args.no_judges else ALL_JUDGES
    print("Judges: " + (", ".join(j.__name__ for j in evaluators) if evaluators
                        else "disabled (--no-judges)"))

    answer_fn, prefix, gen_model = SYSTEMS[args.system]
    concurrency = args.concurrency
    if args.system == "full_doc" and concurrency != 1:
        # Parallel first calls would each WRITE the ~180k-token cache instead
        # of one writing and the rest reading it.
        print("full_doc: forcing --concurrency 1 so the document cache is reused.")
        concurrency = 1
    print(f"System: {args.system} (generator {gen_model})")

    results = evaluate(
        make_target(answer_fn),
        data=data,
        evaluators=evaluators,
        client=client,
        experiment_prefix=prefix,
        max_concurrency=concurrency,
        metadata={
            "system": args.system,
            "model": gen_model,
            **({"top_k": config.TOP_K,
                "similarity_threshold": config.SIMILARITY_THRESHOLD}
               if args.system in ("rag", "agentic") else
               {"effort": config.BASELINE_EFFORT}),
            **({"agent_stages": config.AGENT_STAGES,
                "agent_max_evidence": config.AGENT_MAX_EVIDENCE}
               if args.system == "agentic" else {}),
            # Recorded so a run scored by one judge is never silently compared
            # against a run scored by another -- or by hand.
            "judge_model": None if args.no_judges else config.JUDGE_MODEL,
        },
    )

    name = getattr(results, "experiment_name", None)
    out_path = dump_results(results, name, args.system, gen_model)
    print(f"\nDone. Experiment: {name}")
    print(f"Per-question results written to {out_path}")
    print("Open it in the LangSmith UI (Datasets & Testing -> "
          f"{DATASET_NAME} -> Experiments) to see per-question answer, "
          "latency, and cost, plus the aggregate totals.")


if __name__ == "__main__":
    main()
