"""Phase 5 — build a stratified 100-query sample for cross-evaluator validation.

Draws 20 queries from each of the five 300-query FinDER splits with a fixed
seed so the sample is fully reproducible. Output schema:
  query_id, query, group

A second file ``cross_eval_sample_100_ids.csv`` carries just the query_ids so
downstream filtering scripts can join cheaply.
"""
import csv
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = REPO_ROOT / "random_queries_csv"
OUTPUT_DIR = REPO_ROOT / "output"
PER_GROUP = 20
SEED = 42

GROUPS = [
    "Linq-AI-Research_FinDER-300-01",
    "Linq-AI-Research_FinDER-300-02",
    "Linq-AI-Research_FinDER-300-03",
    "Linq-AI-Research_FinDER-300-04",
    "Linq-AI-Research_FinDER-300-05",
]


def main():
    rng = random.Random(SEED)
    sampled_rows = []
    for group in GROUPS:
        path = SPLITS_DIR / f"{group}.csv"
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if len(rows) < PER_GROUP:
            raise RuntimeError(f"{path} has only {len(rows)} rows, need {PER_GROUP}")
        # Shallow copy + random sample without replacement
        for row in rng.sample(rows, PER_GROUP):
            sampled_rows.append(
                {"query_id": row["query_id"], "query": row["query"], "group": group}
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_full = OUTPUT_DIR / "cross_eval_sample_100.csv"
    with out_full.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "query", "group"], quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(sampled_rows)

    out_ids = OUTPUT_DIR / "cross_eval_sample_100_ids.csv"
    with out_ids.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "group"])
        writer.writeheader()
        for row in sampled_rows:
            writer.writerow({"query_id": row["query_id"], "group": row["group"]})

    print(f"Wrote {len(sampled_rows)} rows ({PER_GROUP} per group, seed={SEED})")
    print(f"  Full sample: {out_full}")
    print(f"  ID-only:     {out_ids}")


if __name__ == "__main__":
    main()
