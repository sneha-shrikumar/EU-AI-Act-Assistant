"""Upload the hand-labelled golden test set to LangSmith as a Dataset.

Why this exists instead of `Client.upload_csv`: the source CSV is exported from
Excel as **Windows-1252**, not UTF-8 (curly quotes/em-dashes are single high
bytes like 0x91). `upload_csv` reads UTF-8 and dies with
"'utf-8' codec can't decode byte 0x91". We also have to fix a trailing space in
the "Output " header and split the messy cells into clean input/output fields.

Column mapping (per row of the golden CSV):
  inputs   = {"question"}
  outputs  = {"expected_source", "should_answer", "expected_answer"}
  metadata = {"id", "category"}   # so evals can slice by category

Idempotent: if the target dataset already holds examples, it refuses unless you
pass --replace (which deletes existing examples first).

Run from anywhere:
    python evals/upload_dataset.py
    python evals/upload_dataset.py --replace
    python evals/upload_dataset.py --name "Some other dataset name"
"""
import argparse
import csv
import sys
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client

BASE_DIR = Path(__file__).resolve().parent.parent  # eu-ai-act-rag/
CSV_PATH = BASE_DIR / "golden dataset for EU AI Act.csv"
CSV_ENCODING = "cp1252"  # Excel-on-Windows export; NOT utf-8 (see module docstring)
DEFAULT_DATASET = "EU AI Acts evaluation"


def load_rows(csv_path: Path) -> list[dict]:
    """Read the golden CSV (cp1252), trimming whitespace from every header and
    value so the trailing-space 'Output ' header and stray cell padding are gone."""
    with open(csv_path, encoding=CSV_ENCODING, newline="") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [(h or "").strip() for h in reader.fieldnames]
        rows = []
        for raw in reader:
            rows.append({(k or "").strip(): (v or "").strip() for k, v in raw.items()})
    return rows


def to_example(row: dict) -> dict:
    """One golden row -> one LangSmith example (inputs / outputs / metadata)."""
    return {
        "inputs": {"question": row.get("question", "")},
        "outputs": {
            "expected_source": row.get("expected_source", ""),
            "should_answer": row.get("should_answer", ""),
            "expected_answer": row.get("Output", ""),
        },
        "metadata": {"id": row.get("id", ""), "category": row.get("category", "")},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload the golden test set to LangSmith.")
    parser.add_argument("--name", default=DEFAULT_DATASET, help="Target dataset name.")
    parser.add_argument("--replace", action="store_true",
                        help="If the dataset already has examples, delete them first.")
    args = parser.parse_args()

    load_dotenv(BASE_DIR / ".env")

    if not CSV_PATH.exists():
        raise SystemExit(f"CSV not found: {CSV_PATH}")

    rows = load_rows(CSV_PATH)
    examples = [to_example(r) for r in rows if r.get("question")]
    print(f"Parsed {len(examples)} examples from {CSV_PATH.name} ({CSV_ENCODING}).")

    client = Client()

    # get-or-create the dataset by name
    existing = next((d for d in client.list_datasets(dataset_name=args.name)), None)
    if existing is None:
        dataset = client.create_dataset(
            dataset_name=args.name,
            description="Hand-labelled golden test set for the EU AI Act RAG pipeline.",
        )
        print(f"Created dataset {args.name!r} (id={dataset.id}).")
    else:
        dataset = existing
        print(f"Found existing dataset {args.name!r} (id={dataset.id}, "
              f"examples={dataset.example_count}).")
        if dataset.example_count and not args.replace:
            raise SystemExit(
                f"Dataset already has {dataset.example_count} examples. "
                "Re-run with --replace to overwrite them."
            )
        if dataset.example_count and args.replace:
            ids = [ex.id for ex in client.list_examples(dataset_id=dataset.id)]
            client.delete_examples(example_ids=ids)
            print(f"Deleted {len(ids)} existing examples (--replace).")

    client.create_examples(dataset_id=dataset.id, examples=examples)

    final = client.read_dataset(dataset_id=dataset.id)
    print(f"Done. {args.name!r} now has {final.example_count} examples.")
    host = str(client.api_url).replace("api.", "").rstrip("/")
    print("View it in the LangSmith UI under Datasets & Testing.")


if __name__ == "__main__":
    main()
