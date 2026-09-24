"""Build the RAG-vs-full-document comparison page from two experiment dumps.

Reads the per-question JSONs that run_langsmith_eval.py writes to
evals/results/, adds each run's real cost (LangSmith aggregates child-LLM
cost onto the root run asynchronously, which is why the dump can't carry it),
and fills evals/report_template.html with the combined data.

    python evals/compare_report.py <rag_experiment> <full_doc_experiment>
    # -> evals/results/rag_vs_opus_report.html
"""
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
load_dotenv(BASE_DIR / ".env")

from langsmith import Client  # noqa: E402

import config  # noqa: E402

RESULTS_DIR = BASE_DIR / "evals" / "results"
TEMPLATE = BASE_DIR / "evals" / "report_template.html"
OUT = RESULTS_DIR / "rag_vs_opus_report.html"


def _load(experiment: str, client: Client) -> dict:
    data = json.loads((RESULTS_DIR / f"{experiment}.json").read_text(encoding="utf-8"))
    costs = {
        str(run.id): float(run.total_cost) if run.total_cost is not None else None
        for run in client.list_runs(project_name=experiment, is_root=True)
    }
    for row in data["rows"]:
        row["cost"] = costs.get(row["run_id"])
    missing = sum(r["cost"] is None for r in data["rows"])
    if missing:
        print(f"warning: {experiment}: {missing} row(s) have no cost yet in LangSmith")
    return data


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    client = Client()
    rag, full = (_load(name, client) for name in sys.argv[1:])

    payload = {
        "systems": {
            "rag": {"label": "RAG pipeline", "model": rag["model"],
                    "experiment": rag["experiment"],
                    "detail": f"bge-small retrieval, top-{config.TOP_K} chunks"},
            "full": {"label": "Opus 5.5, full document", "model": full["model"],
                     "experiment": full["experiment"],
                     "detail": f"whole Act in context, effort {config.BASELINE_EFFORT}"},
        },
        "judge_model": rag["judge_model"],
        "rows": {"rag": rag["rows"], "full": full["rows"]},
    }
    blob = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/null", blob)
    OUT.write_text(html, encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
